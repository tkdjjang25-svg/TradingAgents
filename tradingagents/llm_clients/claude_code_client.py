"""LLM client that runs Claude through the local Claude Code CLI in headless mode.

Instead of an API key, each call shells out to ``claude -p`` and is billed to
the claude.ai subscription the CLI is logged into (``claude auth login``).

The CLI is used as a plain completion engine: built-in tools, settings files
and session transcripts are disabled. When tools are bound, they are served to
the CLI by a small MCP server (``claude_code_mcp``) so the model calls them
natively and the CLI runs the whole tool loop; the call then returns the final
answer with no pending tool calls. ``with_structured_output`` maps to
``--json-schema``.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from .base_client import BaseLLMClient
from .validators import validate_model


class ClaudeCodeError(RuntimeError):
    """Raised when the claude CLI is missing or returns an error result."""


_DEFAULT_SYSTEM = "You are a helpful assistant."
_MCP_SERVER = "ta"

_AS_OF_NOTE = """

## Analysis date

The analysis date is {as_of}; treat it as today. Your context may also show a
"currentDate" taken from the machine clock: ignore it, and never use or mention
information from after {as_of}."""

_TOOLS_NOTE = """

## Data

Retrieve every price, indicator value, news item and financial figure you cite
with the tools available to you. Never state a number that did not come from a
tool result in this conversation; if a tool fails, say the data is unavailable."""

_NO_TOOLS_RETRY = (
    "\n\n(Your previous attempt answered without calling any tool. Call the data tools "
    "first, then write the answer using only their results.)"
)


def _text(content: Any) -> str:
    """Flatten LangChain message content (str or list of blocks) to text."""
    if isinstance(content, str):
        return content
    parts = []
    for item in content or []:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(p for p in parts if p)


def render_messages(messages: List[BaseMessage]) -> Tuple[str, str]:
    """Split a message list into (system prompt, user prompt) for ``claude -p``.

    ``claude -p`` takes a single user turn, so a multi-turn history is
    serialized into a tagged transcript.
    """
    system_parts = [_text(m.content) for m in messages if isinstance(m, SystemMessage)]
    convo = [m for m in messages if not isinstance(m, SystemMessage)]
    system = "\n\n".join(p for p in system_parts if p)

    if len(convo) == 1 and isinstance(convo[0], HumanMessage):
        return system, _text(convo[0].content)

    sections = []
    for m in convo:
        if isinstance(m, HumanMessage):
            sections.append(f"[USER]\n{_text(m.content)}")
        elif isinstance(m, AIMessage):
            sections.append(f"[ASSISTANT]\n{_text(m.content)}")
        elif isinstance(m, ToolMessage):
            sections.append(f"[TOOL RESULT: {m.name or m.tool_call_id}]\n{_text(m.content)}")
        else:
            sections.append(f"[{m.type.upper()}]\n{_text(m.content)}")

    prompt = (
        "Conversation so far:\n\n"
        + "\n\n".join(sections)
        + "\n\nWrite the next [ASSISTANT] message. Do not include the [ASSISTANT] tag."
    )
    return system, prompt


def _parse_json_text(text: str) -> Any:
    """Parse JSON from a model reply, tolerating a surrounding code fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(stripped)


def tool_ref(tool: Any) -> Dict[str, str]:
    """Describe a LangChain tool by import path so the MCP server process can load it."""
    func = getattr(tool, "func", None)
    if func is None:
        raise ValueError(f"ChatClaudeCode can only bind @tool functions, got {tool!r}")
    return {"module": func.__module__, "name": tool.name}


def resolve_claude_bin(explicit: Optional[str] = None) -> str:
    """Locate the claude executable (explicit arg > CLAUDE_CODE_BIN > PATH > ~/.local/bin)."""
    if explicit:
        return explicit
    if os.environ.get("CLAUDE_CODE_BIN"):
        return os.environ["CLAUDE_CODE_BIN"]

    found = shutil.which("claude")
    if found and os.name == "nt" and not found.lower().endswith(".exe"):
        # npm installs a .cmd shim, which routes arguments through cmd.exe and
        # mangles the quoting of --json-schema. Prefer the native binary behind it.
        native = Path(found).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if native.exists():
            return str(native)
    if found:
        return found

    local = Path.home() / ".local" / "bin" / ("claude.exe" if os.name == "nt" else "claude")
    if local.exists():
        return str(local)
    raise ClaudeCodeError(
        "Claude Code CLI not found. Install it, or set CLAUDE_CODE_BIN to the claude executable."
    )


def _subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    # An API key in the environment takes priority over the subscription login,
    # which would defeat the purpose of this provider.
    env.pop("ANTHROPIC_API_KEY", None)
    if env.pop("CLAUDECODE", None):
        # Launched from inside a Claude Code session: drop the host's session
        # wiring (proxy URL, session ids) so the child authenticates on its own.
        env.pop("ANTHROPIC_BASE_URL", None)
        for key in list(env):
            if (key.startswith("CLAUDE_CODE_") and key != "CLAUDE_CODE_OAUTH_TOKEN") or key.startswith(
                "CLAUDE_AGENT_SDK_"
            ):
                del env[key]
    # Without this, every call also spends a Haiku request on generating a
    # session title and writes it under ~/.claude/projects, despite
    # --no-session-persistence.
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


def resolve_model_id(model: str, claude_bin: Optional[str] = None, timeout: float = 30) -> Optional[str]:
    """Return the full model id the CLI uses for ``model`` (an alias like "opus"), or None.

    Reads the CLI's startup event and stops it there, before the model answers.
    The CLI echoes ids it does not know, so this does not prove the model is usable.
    """
    cmd = [
        resolve_claude_bin(claude_bin), "-p", "--model", model, "--output-format", "stream-json", "--verbose",
        "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", env=_subprocess_env(), cwd=tempfile.gettempdir(),
        )
    except OSError:
        return None
    timer = threading.Timer(timeout, proc.kill)
    timer.start()
    try:
        proc.stdin.write(".")
        proc.stdin.close()
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "system" and event.get("subtype") == "init":
                return event.get("model")
        return None
    finally:
        timer.cancel()
        proc.kill()
        proc.wait()


def _mcp_config(tools: List[Dict[str, str]], as_of: Optional[str], log_path: Path) -> Dict[str, Any]:
    from tradingagents.dataflows.config import get_config

    return {
        "mcpServers": {
            _MCP_SERVER: {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "tradingagents.llm_clients.claude_code_mcp"],
                "env": {
                    "TA_MCP_TOOLS": json.dumps([[t["module"], t["name"]] for t in tools]),
                    "TA_MCP_CONFIG": json.dumps(get_config(), default=str),
                    "TA_MCP_AS_OF": as_of or "",
                    "TA_MCP_LOG": str(log_path),
                    "PYTHONIOENCODING": "utf-8",
                },
            }
        }
    }


class ChatClaudeCode(BaseChatModel):
    """LangChain chat model backed by ``claude -p`` (Claude Code headless mode)."""

    model: str = "sonnet"
    claude_bin: Optional[str] = None
    timeout: float = 900
    # Set by TradingAgentsGraph.propagate(); caps tool date arguments and tells
    # the model to ignore the real date the CLI injects into its context.
    as_of_date: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model}

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[tool_ref(t) for t in tools], **kwargs)

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        if include_raw or not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise NotImplementedError("ChatClaudeCode supports plain Pydantic schemas only")
        bound = self.bind(response_schema=schema.model_json_schema())
        return bound | RunnableLambda(lambda msg: schema.model_validate_json(msg.content))

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        tools: Optional[List[Dict[str, str]]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, prompt = render_messages(messages)
        system = system or _DEFAULT_SYSTEM
        if self.as_of_date:
            system += _AS_OF_NOTE.format(as_of=self.as_of_date)
        if tools:
            system += _TOOLS_NOTE

        data, tool_log = self._run(system, prompt, response_schema, tools)
        if tools and not tool_log:
            # Without data the model writes plausible but invented figures.
            data, tool_log = self._run(system, prompt + _NO_TOOLS_RETRY, response_schema, tools)
            if not tool_log:
                raise ClaudeCodeError(
                    "claude -p answered twice without calling any data tool; refusing an analysis with no data"
                )
        text = data.get("result") or ""
        metadata = {
            "model": self.model,
            # Full model id(s) the alias resolved to, e.g. ["claude-opus-4-8"].
            "model_ids": list(data.get("modelUsage") or {}),
            "usage": data.get("usage"),
            "duration_ms": data.get("duration_ms"),
            "num_turns": data.get("num_turns"),
            "tools_executed": tool_log,
        }

        if response_schema is not None:
            structured = data.get("structured_output")
            if structured is None:
                structured = _parse_json_text(text)
            content = json.dumps(structured, ensure_ascii=False)
        else:
            content = text
        # Tools already ran inside the CLI, so there are never pending tool calls.
        message = AIMessage(content=content, response_metadata=metadata)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _run(
        self,
        system: str,
        prompt: str,
        response_schema: Optional[Dict[str, Any]],
        tools: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        with tempfile.TemporaryDirectory(prefix="ta-claude-") as workdir:
            # The system prompt goes through a file: analyst prompts can exceed
            # the Windows command-line limit (~32K chars).
            system_file = Path(workdir) / "system.txt"
            system_file.write_text(system, encoding="utf-8")
            cmd = [
                resolve_claude_bin(self.claude_bin),
                "-p",
                "--model", self.model,
                "--output-format", "json",
                "--system-prompt-file", str(system_file),
                "--tools", "",
                "--strict-mcp-config",
                "--setting-sources", "",
                "--no-session-persistence",
            ]
            tool_log_path = Path(workdir) / "tool_calls.jsonl"
            env = _subprocess_env()
            if tools:
                # By default the CLI starts the first turn while MCP servers are
                # still connecting, so the model sees no data tools and answers
                # from memory. Block until the server is up instead.
                env["MCP_CONNECTION_NONBLOCKING"] = "0"
                env.setdefault("MCP_TIMEOUT", "60000")
                mcp_file = Path(workdir) / "mcp.json"
                mcp_file.write_text(json.dumps(_mcp_config(tools, self.as_of_date, tool_log_path)), encoding="utf-8")
                allowed = ",".join(f"mcp__{_MCP_SERVER}__{t['name']}" for t in tools)
                cmd += ["--mcp-config", str(mcp_file), "--allowedTools", allowed]
            if response_schema is not None:
                cmd += ["--json-schema", json.dumps(response_schema, ensure_ascii=False)]
            try:
                # cwd is an empty temp dir so no project CLAUDE.md gets picked up.
                proc = subprocess.run(
                    cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=self.timeout, cwd=workdir, env=env,
                )
            except subprocess.TimeoutExpired as exc:
                raise ClaudeCodeError(f"claude -p timed out after {self.timeout:.0f}s") from exc

            tool_log = []
            if tool_log_path.exists():
                tool_log = [json.loads(line) for line in tool_log_path.read_text(encoding="utf-8").splitlines() if line]

        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            detail = (proc.stderr or proc.stdout).strip()[:500]
            raise ClaudeCodeError(f"claude -p exited with code {proc.returncode}: {detail}")

        if data.get("is_error"):
            detail = data.get("result") or data.get("subtype") or "unknown error"
            if data.get("api_error_status") == 401:
                detail += " (run `claude auth login` in a terminal to log in again)"
            raise ClaudeCodeError(f"claude -p failed: {detail}")
        return data, tool_log


class ClaudeCodeClient(BaseLLMClient):
    """Client for Claude via the local Claude Code CLI (subscription login, no API key)."""

    provider = "claude-code"

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}
        for key in ("timeout", "callbacks", "claude_bin"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return ChatClaudeCode(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("claude-code", self.model)
