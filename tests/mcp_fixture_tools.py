"""Tiny tool module the MCP server subprocess can import in tests (no pytest import here)."""
from langchain_core.tools import tool


@tool
def echo_range(symbol: str, start_date: str, end_date: str) -> str:
    """Echo the arguments back."""
    return f"{symbol} {start_date}..{end_date}"
