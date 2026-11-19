import json
import tempfile
import unittest
from pathlib import Path

import pytest
from langchain_core.tools import tool

from tradingagents.llm_clients.claude_code_mcp import clamp_dates, handle


@tool
def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Fetch OHLCV data for a ticker."""
    return f"{symbol} {start_date}..{end_date}"


@tool
def broken(symbol: str) -> str:
    """Always fails."""
    raise RuntimeError("vendor down")


TOOLS = {"get_stock_data": get_stock_data, "broken": broken}


def _call(method, params=None, req_id=1, as_of=None, log=None):
    return handle({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}, TOOLS, as_of, log)


@pytest.mark.unit
class TestClampDates(unittest.TestCase):
    def test_dates_after_the_analysis_date_are_capped(self):
        args = {"symbol": "KORU", "start_date": "2026-05-01", "end_date": "2026-11-19", "curr_date": "2026-11-18"}
        self.assertEqual(
            clamp_dates(args, "2026-11-11"),
            {"symbol": "KORU", "start_date": "2026-05-01", "end_date": "2026-11-11", "curr_date": "2026-11-11"},
        )

    def test_no_analysis_date_leaves_args_alone(self):
        args = {"end_date": "2030-01-01"}
        self.assertEqual(clamp_dates(args, None), args)


@pytest.mark.unit
class TestHandle(unittest.TestCase):
    def test_initialize_echoes_protocol_version(self):
        resp = _call("initialize", {"protocolVersion": "2025-06-18"})
        self.assertEqual(resp["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("tools", resp["result"]["capabilities"])

    def test_notifications_get_no_response(self):
        self.assertIsNone(handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, TOOLS, None, None))

    def test_tools_list_exposes_json_schema(self):
        tools = {t["name"]: t for t in _call("tools/list")["result"]["tools"]}
        self.assertEqual(set(tools["get_stock_data"]["inputSchema"]["required"]), {"symbol", "start_date", "end_date"})

    def test_tools_call_clamps_dates_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "calls.jsonl"
            resp = _call(
                "tools/call",
                {"name": "get_stock_data", "arguments": {"symbol": "KORU", "start_date": "2026-05-01", "end_date": "2026-11-19"}},
                as_of="2026-11-11", log=str(log),
            )
            entry = json.loads(log.read_text(encoding="utf-8"))
        self.assertEqual(resp["result"]["content"][0]["text"], "KORU 2026-05-01..2026-11-11")
        self.assertFalse(resp["result"]["isError"])
        self.assertEqual(entry["args"]["end_date"], "2026-11-11")

    def test_tool_exception_is_reported_not_raised(self):
        resp = _call("tools/call", {"name": "broken", "arguments": {"symbol": "X"}})
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("vendor down", resp["result"]["content"][0]["text"])

    def test_unknown_tool_and_method_are_errors(self):
        self.assertIn("error", _call("tools/call", {"name": "nope", "arguments": {}}))
        self.assertEqual(_call("resources/list")["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
