import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents.llm_clients.claude_code_client import (
    ChatClaudeCode,
    ClaudeCodeClient,
    ClaudeCodeError,
    _subprocess_env,
    render_messages,
    resolve_model_id,
)
from tradingagents.llm_clients.factory import create_llm_client


@tool
def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Fetch OHLCV data for a ticker."""
    return "csv"


class Decision(BaseModel):
    action: str
    reason: str


def _completed(payload: dict) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(payload), stderr="")


class FakeCLI:
    """subprocess.run stand-in that records each call and can emulate MCP tool calls."""

    def __init__(self, result="ok", tool_calls_per_run=(1,), **extra):
        self.result, self.tool_calls_per_run, self.extra = result, list(tool_calls_per_run), extra
        self.cmds, self.systems, self.prompts, self.mcp_configs = [], [], [], []

    def __call__(self, cmd, input, **kwargs):
        self.cmds.append(cmd)
        self.envs = getattr(self, "envs", []) + [kwargs["env"]]
        self.prompts.append(input)
        self.systems.append(Path(cmd[cmd.index("--system-prompt-file") + 1]).read_text(encoding="utf-8"))
        if "--mcp-config" in cmd:
            config = json.loads(Path(cmd[cmd.index("--mcp-config") + 1]).read_text(encoding="utf-8"))
            self.mcp_configs.append(config)
            log = Path(config["mcpServers"]["ta"]["env"]["TA_MCP_LOG"])
            for _ in range(self.tool_calls_per_run.pop(0) if self.tool_calls_per_run else 0):
                with open(log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"name": "get_stock_data", "args": {}, "chars": 3, "error": False}) + "\n")
        return _completed({"is_error": False, "result": self.result, **self.extra})


@pytest.mark.unit
class TestRenderMessages(unittest.TestCase):
    def test_single_human_message_is_sent_verbatim(self):
        system, prompt = render_messages([SystemMessage("be brief"), HumanMessage("hello")])
        self.assertEqual(system, "be brief")
        self.assertEqual(prompt, "hello")

    def test_multi_turn_history_is_serialized_as_transcript(self):
        msgs = [
            HumanMessage("NVDA"),
            AIMessage("looking"),
            ToolMessage("Date,Close\n2026-01-02,190", tool_call_id="c1", name="get_stock_data"),
        ]
        _, prompt = render_messages(msgs)
        self.assertIn("[USER]\nNVDA", prompt)
        self.assertIn("[ASSISTANT]\nlooking", prompt)
        self.assertIn("[TOOL RESULT: get_stock_data]\nDate,Close", prompt)


@pytest.mark.unit
@patch("tradingagents.llm_clients.claude_code_client.resolve_claude_bin", return_value="claude")
class TestChatClaudeCode(unittest.TestCase):
    def test_plain_invoke_uses_headless_flags(self, _bin):
        cli = FakeCLI(result="pong")
        with patch("tradingagents.llm_clients.claude_code_client.subprocess.run", cli):
            result = ChatClaudeCode(model="haiku").invoke("ping")

        self.assertEqual(result.content, "pong")
        cmd = cli.cmds[0]
        for flag in ("-p", "--no-session-persistence", "--strict-mcp-config", "--system-prompt-file"):
            self.assertIn(flag, cmd)
        self.assertNotIn("--mcp-config", cmd)
        self.assertEqual(cmd[cmd.index("--tools") + 1], "")
        self.assertEqual(cmd[cmd.index("--model") + 1], "haiku")
        self.assertEqual(cli.prompts[0], "ping")

    def test_bound_tools_are_served_over_mcp_and_run_inside_the_cli(self, _bin):
        cli = FakeCLI(result="report based on data", num_turns=3, modelUsage={"claude-sonnet-4-6": {}})
        with patch("tradingagents.llm_clients.claude_code_client.subprocess.run", cli):
            result = ChatClaudeCode().bind_tools([get_stock_data]).invoke("analyze NVDA")

        self.assertEqual(result.content, "report based on data")
        self.assertEqual(result.tool_calls, [])  # LangGraph must not try to run them again
        self.assertEqual(len(result.response_metadata["tools_executed"]), 1)
        self.assertEqual(result.response_metadata["model_ids"], ["claude-sonnet-4-6"])
        cmd = cli.cmds[0]
        self.assertEqual(cmd[cmd.index("--allowedTools") + 1], "mcp__ta__get_stock_data")
        self.assertEqual(cli.envs[0]["MCP_CONNECTION_NONBLOCKING"], "0")  # wait for tools before answering
        self.assertIn("Never state a number that did not come from a", cli.systems[0])

    def test_mcp_config_points_at_the_tool_by_import_path(self, _bin):
        cli = FakeCLI()
        with patch("tradingagents.llm_clients.claude_code_client.subprocess.run", cli):
            llm = ChatClaudeCode(as_of_date="2026-11-11")
            llm.bind_tools([get_stock_data]).invoke("x")
        server = cli.mcp_configs[0]["mcpServers"]["ta"]
        self.assertEqual(server["args"], ["-m", "tradingagents.llm_clients.claude_code_mcp"])
        self.assertEqual(json.loads(server["env"]["TA_MCP_TOOLS"]), [[get_stock_data.func.__module__, "get_stock_data"]])
        self.assertEqual(server["env"]["TA_MCP_AS_OF"], "2026-11-11")
        self.assertIn("data_vendors", json.loads(server["env"]["TA_MCP_CONFIG"]))

    def test_answer_without_tool_use_is_retried_then_rejected(self, _bin):
        cli = FakeCLI(tool_calls_per_run=(0, 1))
        with patch("tradingagents.llm_clients.claude_code_client.subprocess.run", cli):
            ChatClaudeCode().bind_tools([get_stock_data]).invoke("analyze")
        self.assertEqual(len(cli.cmds), 2)
        self.assertIn("answered without calling any tool", cli.prompts[1])

        cli = FakeCLI(tool_calls_per_run=(0, 0))
        with patch("tradingagents.llm_clients.claude_code_client.subprocess.run", cli):
            with self.assertRaises(ClaudeCodeError):
                ChatClaudeCode().bind_tools([get_stock_data]).invoke("analyze")

    def test_as_of_date_is_stated_in_the_system_prompt(self, _bin):
        cli = FakeCLI()
        with patch("tradingagents.llm_clients.claude_code_client.subprocess.run", cli):
            ChatClaudeCode(as_of_date="2026-11-11").invoke("hi")
        self.assertIn("The analysis date is 2026-11-11", cli.systems[0])
        self.assertIn("ignore it", cli.systems[0])

    def test_structured_output_passes_schema_and_parses(self, _bin):
        cli = FakeCLI(result="", structured_output={"action": "BUY", "reason": "momentum"})
        with patch("tradingagents.llm_clients.claude_code_client.subprocess.run", cli):
            result = ChatClaudeCode().with_structured_output(Decision).invoke("decide")

        self.assertEqual(result, Decision(action="BUY", reason="momentum"))
        cmd = cli.cmds[0]
        schema = json.loads(cmd[cmd.index("--json-schema") + 1])
        self.assertEqual(set(schema["required"]), {"action", "reason"})

    def test_structured_output_falls_back_to_json_in_text(self, _bin):
        cli = FakeCLI(result='```json\n{"action": "SELL", "reason": "overbought"}\n```')
        with patch("tradingagents.llm_clients.claude_code_client.subprocess.run", cli):
            result = ChatClaudeCode().with_structured_output(Decision).invoke("decide")
        self.assertEqual(result.action, "SELL")

    def test_auth_error_mentions_login(self, _bin):
        payload = {"is_error": True, "api_error_status": 401, "result": "OAuth access token has expired."}
        with patch("tradingagents.llm_clients.claude_code_client.subprocess.run", return_value=_completed(payload)):
            with self.assertRaises(ClaudeCodeError) as ctx:
                ChatClaudeCode().invoke("hi")
        self.assertIn("claude auth login", str(ctx.exception))


class _FakePopen:
    def __init__(self, lines):
        self.stdin = type("W", (), {"write": lambda *a: None, "close": lambda *a: None})()
        self.stdout = iter(lines)
        self.killed = False

    def kill(self):
        self.killed = True

    def wait(self):
        return 0


@pytest.mark.unit
@patch("tradingagents.llm_clients.claude_code_client.resolve_claude_bin", return_value="claude")
class TestResolveModelId(unittest.TestCase):
    def test_reads_model_from_init_event_and_stops_the_cli(self, _bin):
        proc = _FakePopen(["not json", json.dumps({"type": "system", "subtype": "init", "model": "claude-opus-4-8"})])
        with patch("tradingagents.llm_clients.claude_code_client.subprocess.Popen", return_value=proc) as popen:
            self.assertEqual(resolve_model_id("opus"), "claude-opus-4-8")
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--model") + 1], "opus")
        self.assertTrue(proc.killed)

    def test_returns_none_without_init_event_or_binary(self, _bin):
        with patch("tradingagents.llm_clients.claude_code_client.subprocess.Popen", return_value=_FakePopen([])):
            self.assertIsNone(resolve_model_id("opus"))
        with patch("tradingagents.llm_clients.claude_code_client.subprocess.Popen", side_effect=OSError):
            self.assertIsNone(resolve_model_id("opus"))


@pytest.mark.unit
class TestSubprocessEnv(unittest.TestCase):
    def test_api_key_is_always_removed(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "x", "ANTHROPIC_BASE_URL": "https://proxy"}, clear=True):
            env = _subprocess_env()
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "https://proxy")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1")

    def test_host_claude_code_session_vars_are_removed(self):
        host = {
            "CLAUDECODE": "1",
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:1234",
            "CLAUDE_CODE_SESSION_ID": "s",
            "CLAUDE_AGENT_SDK_VERSION": "v",
            "CLAUDE_CODE_OAUTH_TOKEN": "keep",
            "PATH": "p",
        }
        with patch.dict("os.environ", host, clear=True):
            env = _subprocess_env()
        self.assertEqual(
            env,
            {"CLAUDE_CODE_OAUTH_TOKEN": "keep", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "PATH": "p"},
        )


@pytest.mark.unit
class TestFactory(unittest.TestCase):
    def test_factory_returns_claude_code_client(self):
        client = create_llm_client("claude-code", "sonnet")
        self.assertIsInstance(client, ClaudeCodeClient)
        self.assertIsInstance(client.get_llm(), ChatClaudeCode)
        self.assertTrue(client.validate_model())


if __name__ == "__main__":
    unittest.main()
