import json
import tempfile
import unittest
from pathlib import Path

import pytest
from langchain_core.tools import tool

from tradingagents.llm_clients.claude_code_mcp import clamp_dates, handle, mcp_tool_schema


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

    def test_padded_or_malformed_dates_cannot_bypass_the_cap(self):
        out = clamp_dates({"curr_date": " 2099-12-31", "end_date": "2099/12/31", "start_date": " 2026-01-05 "},
                          "2026-11-11")
        self.assertEqual(out, {"curr_date": "2026-11-11", "end_date": "2026-11-11", "start_date": "2026-01-05"})

    def test_missing_optional_date_is_filled_when_the_tool_accepts_it(self):
        out = clamp_dates({"ticker": "NVDA"}, "2026-11-11", accepted={"ticker", "freq", "curr_date"})
        self.assertEqual(out["curr_date"], "2026-11-11")
        self.assertNotIn("end_date", out)  # not a parameter of this tool
        self.assertEqual(clamp_dates({"ticker": "NVDA"}, "2026-11-11", accepted={"ticker"}), {"ticker": "NVDA"})


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

    def test_precomputed_schemas_answer_tools_list_without_loaded_tools(self):
        schemas = [mcp_tool_schema(get_stock_data)]
        listing = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, {}, None, None, schemas)
        self.assertEqual(listing["result"]["tools"], schemas)
        self.assertEqual(schemas[0]["name"], "get_stock_data")
        self.assertIn("symbol", schemas[0]["inputSchema"]["properties"])


@pytest.mark.unit
class TestServerProcess(unittest.TestCase):
    """Runs the real server as the CLI would: stdio JSON-RPC in a subprocess."""

    def _run(self, env_extra, messages):
        import os
        import subprocess
        import sys

        env = {**os.environ, "PYTHONIOENCODING": "utf-8", **env_extra}
        proc = subprocess.run(
            [sys.executable, "-m", "tradingagents.llm_clients.claude_code_mcp"],
            input="".join(json.dumps(m) + "\n" for m in messages), capture_output=True, text=True,
            env=env, timeout=120, cwd=str(Path(__file__).resolve().parents[1]),
        )
        return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]

    def test_lazy_loading_serves_list_then_runs_the_real_tool(self):
        from tests.mcp_fixture_tools import echo_range

        tools = [["tests.mcp_fixture_tools", "echo_range"]]
        out = self._run(
            {"TA_MCP_TOOLS": json.dumps(tools), "TA_MCP_SCHEMAS": json.dumps([mcp_tool_schema(echo_range)])},
            [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "echo_range", "arguments": {"symbol": "NVDA", "start_date": "a", "end_date": "b"}}}],
        )
        self.assertEqual(out[0]["result"]["tools"][0]["name"], "echo_range")
        self.assertEqual(out[1]["result"]["content"][0]["text"], "NVDA a..b")

    def test_tool_import_failure_is_reported_on_call(self):
        out = self._run(
            {"TA_MCP_TOOLS": json.dumps([["no_such_module_xyz", "t"]]), "TA_MCP_SCHEMAS": json.dumps([])},
            [{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "t", "arguments": {}}}],
        )
        self.assertTrue(out[0]["result"]["isError"])
        self.assertIn("failed to load", out[0]["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
