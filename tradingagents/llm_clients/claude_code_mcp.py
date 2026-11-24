"""Minimal stdio MCP server that exposes TradingAgents data tools to ``claude -p``.

ChatClaudeCode launches this as a subprocess of the claude CLI (via
``--mcp-config``) so the model calls the tools natively instead of through a
text protocol. Configuration comes from environment variables:

    TA_MCP_TOOLS   JSON list of [module, attribute] pairs naming LangChain tools
    TA_MCP_SCHEMAS optional JSON list of MCP tool descriptors computed by the parent.
                   When present the server answers initialize/tools/list at once and
                   imports the tools (yfinance, pandas: ~1.5 s) in a background thread,
                   so the CLI's first model turn is not held up by the imports.
    TA_MCP_CONFIG  JSON dataflow config (tradingagents.dataflows.config)
    TA_MCP_AS_OF   analysis date (YYYY-MM-DD); date arguments are clamped to it
    TA_MCP_LOG     optional path; one JSON line is appended per tool call
"""

import importlib
import json
import os
import sys
import threading
from typing import Any, Dict, List, Optional

_DATE_ARGS = ("start_date", "end_date", "curr_date")


def _parse_day(value: Any) -> Optional[str]:
    """Strict YYYY-MM-DD parse (after trimming); anything else is None."""
    from datetime import datetime

    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def clamp_dates(args: Dict[str, Any], as_of: Optional[str], accepted: Optional[set] = None) -> Dict[str, Any]:
    """Cap date arguments at the analysis date so no data after it can be fetched.

    The claude CLI injects the machine's real date into the model's context,
    which would otherwise let a backdated analysis request future prices.
    Dates are parsed strictly (so " 2099-12-31" or "2099/12/31" cannot slip past a
    text comparison); unparseable values become the analysis date. ``accepted`` is the
    set of argument names the tool takes: an omitted ``curr_date``/``end_date`` is
    filled with the analysis date, because some tools skip their date filter when it
    is missing.
    """
    if not as_of:
        return args
    out = dict(args)
    for key, value in args.items():
        if key in _DATE_ARGS and value is not None:
            day = _parse_day(value)
            out[key] = as_of if day is None or day > as_of else day
    for key in ("curr_date", "end_date"):
        if accepted and key in accepted and out.get(key) is None:
            out[key] = as_of
    return out


def load_tools(spec: str) -> Dict[str, Any]:
    tools = {}
    for module, attr in json.loads(spec):
        tool = getattr(importlib.import_module(module), attr)
        tools[tool.name] = tool
    return tools


def mcp_tool_schema(tool: Any) -> Dict[str, Any]:
    """MCP tools/list entry for a LangChain tool."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    fn = convert_to_openai_tool(tool)["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "inputSchema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


def handle(request: Dict[str, Any], tools: Dict[str, Any], as_of: Optional[str], log_path: Optional[str],
           schemas: Optional[List[Dict[str, Any]]] = None):
    """Return the JSON-RPC response for one request, or None for notifications."""
    method, req_id = request.get("method"), request.get("id")
    if req_id is None:
        return None

    if method == "initialize":
        result = {
            "protocolVersion": request.get("params", {}).get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "tradingagents", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": schemas if schemas is not None else [mcp_tool_schema(t) for t in tools.values()]}
    elif method == "tools/call":
        params = request.get("params", {})
        name = params.get("name")
        if name not in tools:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Unknown tool: {name}"}}
        accepted = set(getattr(tools[name], "args", {}) or {})
        args = clamp_dates(params.get("arguments") or {}, as_of, accepted)
        try:
            text, is_error = str(tools[name].invoke(args)), False
        except Exception as exc:  # report to the model instead of crashing the server
            text, is_error = f"Tool error: {type(exc).__name__}: {exc}", True
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"name": name, "args": args, "chars": len(text), "error": is_error}) + "\n")
        result = {"content": [{"type": "text", "text": text}], "isError": is_error}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def main() -> None:
    # Tools (yfinance, pandas) may print; keep stdout reserved for JSON-RPC.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    as_of = os.environ.get("TA_MCP_AS_OF") or None
    log_path = os.environ.get("TA_MCP_LOG") or None
    schemas = json.loads(os.environ["TA_MCP_SCHEMAS"]) if os.environ.get("TA_MCP_SCHEMAS") else None
    loaded: Dict[str, Any] = {"tools": {}}

    def load() -> None:
        try:
            from tradingagents.dataflows.config import set_config

            if os.environ.get("TA_MCP_CONFIG"):
                set_config(json.loads(os.environ["TA_MCP_CONFIG"]))
            loaded["tools"] = load_tools(os.environ.get("TA_MCP_TOOLS", "[]"))
        except Exception as exc:  # surfaced on the first tools/call
            loaded["error"] = f"{type(exc).__name__}: {exc}"

    loader = threading.Thread(target=load, name="tool-loader", daemon=True)
    loader.start()
    if schemas is None:
        loader.join()  # no precomputed schemas: tools/list needs the real tools

    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("method") == "tools/call":
            loader.join()
            if "error" in loaded and request.get("id") is not None:
                text = f"Tool server failed to load the data tools: {loaded['error']}"
                response = {"jsonrpc": "2.0", "id": request["id"],
                            "result": {"content": [{"type": "text", "text": text}], "isError": True}}
                protocol_out.write(json.dumps(response, ensure_ascii=False) + "\n")
                protocol_out.flush()
                continue
        response = handle(request, loaded["tools"], as_of, log_path, schemas)
        if response is not None:
            protocol_out.write(json.dumps(response, ensure_ascii=False) + "\n")
            protocol_out.flush()


if __name__ == "__main__":
    main()
