"""Shared helpers for the DART, Naver News and SEC EDGAR clients.

Everything here is thread-safe: the analysts and the watchlist briefings may call
the data tools from several threads at once.
"""

import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"


def get_env(name: str) -> str:
    """Setting from the environment, falling back to the repo's .env file.

    The fallback matters for the MCP server process the claude CLI starts: it does
    not always inherit the parent's environment. Secrets are never hard-coded.
    """
    value = os.getenv(name)
    if value is None and _ENV_FILE.exists():
        value = dotenv_values(_ENV_FILE).get(name)
    return (value or "").strip()


def cache_dir(name: str) -> Path:
    """data/cache/<name>, created on demand (honours TRADINGAGENTS_HOME / CACHE_DIR)."""
    from tradingagents.default_config import DEFAULT_CONFIG

    path = Path(DEFAULT_CONFIG["data_cache_dir"]) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


class RateLimiter:
    """Allows at most one request every ``min_interval`` seconds, across threads."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.min_interval
        if delay > 0:
            time.sleep(delay)


_KR_SUFFIX = re.compile(r"^(\d{6})\.(KS|KQ)$", re.IGNORECASE)
_KR_BARE = re.compile(r"^\d{6}$")
_US_TICKER = re.compile(r"^[A-Z][A-Z0-9]{0,6}([.-][A-Z]{1,2})?$")


def korean_code(ticker: str) -> Optional[str]:
    """6-digit KRX code for 005930.KS / 005930.KQ / 005930, else None."""
    t = (ticker or "").strip().upper()
    m = _KR_SUFFIX.match(t)
    if m:
        return m.group(1)
    return t if _KR_BARE.match(t) else None


def market_of(ticker: str) -> str:
    """'KR' for Korean listings, 'US' for plain US-style symbols, else 'OTHER'.

    Crypto pairs (BTC-USD), indices (^GSPC) and other exchanges (7203.T) are OTHER.
    """
    if korean_code(ticker):
        return "KR"
    t = (ticker or "").strip().upper()
    if t.endswith("-USD") or t.startswith("^") or "=" in t:
        return "OTHER"
    if "." in t and not re.search(r"\.[ABC]$", t):  # BRK.B is US share class; 7203.T, VOD.L are not
        return "OTHER"
    return "US" if _US_TICKER.match(t) else "OTHER"
