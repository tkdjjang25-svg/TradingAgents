"""Minimal stdio MCP server that exposes TradingAgents data tools to ``claude -p``.

ChatClaudeCode launches this as a subprocess of the claude CLI (via
``--mcp-config``) so the model calls the tools natively instead of through a
text protocol. Configuration comes from environment variables:

    TA_MCP_TOOLS   JSON list of [module, attribute] pairs naming LangChain tools
    TA_MCP_CONFIG  JSON dataflow config (tradingagents.dataflows.config)
    TA_MCP_AS_OF   analysis date (YYYY-MM-DD); date arguments are clamped to it
    TA_MCP_LOG     optional path; one JSON line is appended per tool call
"""

import importlib
import json
import os
import sys
from typing import Any, Dict, Optional

from langchain_core.utils.function_calling import convert_to_openai_tool

_DATE_ARGS = ("start_date", "end_date", "curr_date")


def clamp_dates(args: Dict[str, Any], as_of: Optional[str]) -> Dict[str, Any]:
    """Cap date arguments at the analysis date so no data after it can be fetched.

    The claude CLI injects the machine's real date into the model's context,
    which would otherwise let a backdated analysis request future prices.
    """
    if not as_of:
        return args
    return {
        k: as_of if k in _DATE_ARGS and isinstance(v, str) and v > as_of else v
        for k, v in args.items()
    }


def load_tools(spec: str) -> Dict[str, Any]:
    tools = {}
    for module, attr in json.loads(spec):
        tool = getattr(importlib.import_module(module), attr)
        tools[tool.name] = tool
    return tools


def handle(request: Dict[str, Any], tools: Dict[str, Any], as_of: Optional[str], log_path: Optional[str]):
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
        result = {"tools": []}
        for tool in tools.values():
            fn = convert_to_openai_tool(tool)["function"]
            result["tools"].append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "inputSchema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
    elif method == "tools/call":
        params = request.get("params", {})
        name, args = params.get("name"), clamp_dates(params.get("arguments") or {}, as_of)
        if name not in tools:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Unknown tool: {name}"}}
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

    from tradingagents.dataflows.config import set_config

    if os.environ.get("TA_MCP_CONFIG"):
        set_config(json.loads(os.environ["TA_MCP_CONFIG"]))
    tools = load_tools(os.environ.get("TA_MCP_TOOLS", "[]"))
    as_of = os.environ.get("TA_MCP_AS_OF") or None
    log_path = os.environ.get("TA_MCP_LOG") or None

    for line in sys.stdin:
        if not line.strip():
            continue
        response = handle(json.loads(line), tools, as_of, log_path)
        if response is not None:
            protocol_out.write(json.dumps(response, ensure_ascii=False) + "\n")
            protocol_out.flush()


if __name__ == "__main__":
    main()
