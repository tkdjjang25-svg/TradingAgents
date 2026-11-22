import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.dataflows import sec_edgar as sec  # noqa: E402
from tradingagents.dataflows.external_common import RateLimiter  # noqa: E402

MODULE_LIMITER = sec._LIMITER  # captured before the tests swap in a no-wait limiter
UA = "TestApp tester@example.com"

TICKERS = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
    "2": {"cik_str": 1067983, "ticker": "BRK-A", "title": "BERKSHIRE HATHAWAY INC"},
}

_FILINGS = [
    # form, filingDate, reportDate, accession, primaryDocument, description, items
    ("8-K", "2026-11-25", "2026-11-25", "0001045810-26-000200", "late.htm", "8-K", "8.01"),
    ("10-Q", "2026-11-19", "2026-10-26", "0001045810-26-000190", "nvda-10q.htm", "10-Q", ""),
    ("8-K", "2026-11-19", "2026-11-19", "0001045810-26-000189", "nvda-8k.htm", "8-K", "2.02,9.01"),
    ("4", "2026-11-18", "", "0001045810-26-000188", "xslF345X05/form4.xml", "", ""),
    ("4", "2026-11-15", "", "0001045810-26-000187", "xslF345X05/form4.xml", "", ""),
    ("4", "2026-11-10", "", "0001045810-26-000186", "xslF345X05/form4.xml", "", ""),
    ("8-K", "2026-11-12", "2026-11-10", "0001045810-26-000185", "officer-8k.htm", "8-K", "5.02"),
    ("144", "2026-11-05", "", "0001045810-26-000184", "primary_doc.xml", "", ""),
    ("8-K", "2026-10-01", "2026-10-01", "0001045810-26-000150", "old.htm", "8-K", "8.01"),
]
_KEYS = ("form", "filingDate", "reportDate", "accessionNumber", "primaryDocument",
         "primaryDocDescription", "items")


def _block(rows):
    return {k: [r[i] for r in rows] for i, k in enumerate(_KEYS)}


SUBMISSIONS = {"cik": "1045810", "name": "NVIDIA CORP", "filings": {"recent": _block(_FILINGS), "files": []}}

EIGHT_K_HTML = """<html><head><title>8-K</title><style>.x{color:red}</style>
<script>var tracking = 1;</script></head><body>
<div style="display:none"><ix:header><ix:hidden>dei:EntityCentralIndexKey 0001045810</ix:hidden></ix:header></div>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p><p>FORM 8-K</p>
<p>Check the appropriate box below if the Form 8-K filing is intended to simultaneously satisfy</p>
<table><tr><td>Emerging growth company</td><td>&#9744;</td></tr></table>
<p><b>Item&nbsp;2.02 Results of Operations and Financial Condition.</b></p>
<p>On November 19, 2026, NVIDIA Corporation issued a press release announcing results
for its third quarter &amp; record data center revenue.</p>
<p>Item 9.01 Financial Statements and Exhibits.</p><p>Exhibit 99.1 Press release</p>
<p>SIGNATURES</p><p>Pursuant to the requirements of the Securities Exchange Act of 1934</p>
</body></html>"""


def _resp(status=200, json_data=None, text="", headers=None):
    r = MagicMock()
    r.status_code = status
    r.text = text if json_data is None else json.dumps(json_data)
    r.headers = headers or {"Content-Type": "text/html" if json_data is None else "application/json"}
    if json_data is None:
        r.json.side_effect = ValueError("not json")
    else:
        r.json.return_value = json_data
    return r


def _router(calls, overrides=None):
    """Fake requests.get: serves the fixtures by URL and records (url, headers)."""
    overrides = overrides or {}

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers or {}, timeout))
        if url in overrides:
            o = overrides[url]
            return o.pop(0) if isinstance(o, list) else o
        if url == sec.TICKERS_URL:
            return _resp(json_data=TICKERS)
        if url.startswith("https://data.sec.gov/submissions/CIK0001045810"):
            return _resp(json_data=SUBMISSIONS)
        if url.endswith(".htm"):
            return _resp(text=EIGHT_K_HTML)
        return _resp(status=404)

    return fake_get


class _SecTestBase(unittest.TestCase):
    """Temp cache dir, fixed User-Agent, no real sleeps, empty module caches."""

    env = {"SEC_EDGAR_USER_AGENT": UA}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)
        self.calls = []
        self.sleep = MagicMock()
        patches = [
            patch.object(sec, "cache_dir", side_effect=lambda name: self.cache),
            patch.object(sec, "get_env", side_effect=lambda name: self.env.get(name, "")),
            patch.object(sec, "_LIMITER", RateLimiter(0)),
            patch.object(sec, "_sleep", self.sleep),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)
        self._reset_caches()
        self.addCleanup(self._reset_caches)

    @staticmethod
    def _reset_caches():
        sec._tickers = None
        sec._tickers_time = 0.0
        sec._sub_cache.clear()
        sec._sub_key_locks.clear()

    def route(self, overrides=None):
        p = patch.object(sec.requests, "get", side_effect=_router(self.calls, overrides))
        p.start()
        self.addCleanup(p.stop)

    def urls(self):
        return [c[0] for c in self.calls]


@pytest.mark.unit
class TestTickerToCik(_SecTestBase):
    def test_lookup_including_share_class(self):
        self.route()
        self.assertEqual(sec.get_cik("NVDA"), {"cik": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"})
        self.assertEqual(sec.get_cik("brk.b")["cik"], 1067983)
        self.assertEqual(sec.get_cik("BRK-B")["ticker"], "BRK-B")
        self.assertIsNone(sec.get_cik("KORU"))
        self.assertEqual(self.urls().count(sec.TICKERS_URL), 1)  # kept in memory

    def test_disk_cache_reused_then_refreshed_after_seven_days(self):
        self.route()
        sec.get_cik("NVDA")
        path = self.cache / "company_tickers.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), TICKERS)
        self.assertEqual(list(self.cache.glob("*.tmp")), [])  # atomic write left no temp file

        self._reset_caches()  # new process: fresh file on disk, no request
        self.assertEqual(sec.get_cik("NVDA")["cik"], 1045810)
        self.assertEqual(self.urls().count(sec.TICKERS_URL), 1)

        old = time.time() - 8 * 24 * 3600
        os.utime(path, (old, old))
        self._reset_caches()
        sec.get_cik("NVDA")
        self.assertEqual(self.urls().count(sec.TICKERS_URL), 2)
        self.assertGreater(path.stat().st_mtime, old + 24 * 3600)

    def test_stale_file_used_when_refresh_fails(self):
        path = self.cache / "company_tickers.json"
        path.write_text(json.dumps(TICKERS), encoding="utf-8")
        old = time.time() - 30 * 24 * 3600
        os.utime(path, (old, old))
        self.route({sec.TICKERS_URL: _resp(status=403)})
        self.assertEqual(sec.get_cik("NVDA")["cik"], 1045810)


@pytest.mark.unit
class TestUserAgent(_SecTestBase):
    def test_user_agent_and_gzip_headers_sent(self):
        self.route()
        sec.get_filings("NVDA", "2026-11-01", "2026-11-22", excerpts=0)
        self.assertTrue(self.calls)
        for _, headers, timeout in self.calls:
            self.assertEqual(headers["User-Agent"], UA)
            self.assertEqual(headers["Accept-Encoding"], "gzip, deflate")
            self.assertTrue(0 < timeout <= 30)


@pytest.mark.unit
class TestNotConfigured(_SecTestBase):
    env = {}

    def test_no_request_without_user_agent(self):
        self.route()
        out = sec.get_filings("NVDA", "2026-11-01", "2026-11-22")
        self.assertIn("SEC_EDGAR_USER_AGENT", out)
        self.assertIn("not configured", out)
        self.assertIsNone(sec.get_cik("NVDA"))
        self.assertIn("not configured", sec.get_filing_text("https://www.sec.gov/Archives/x.htm"))
        self.assertEqual(self.calls, [])


@pytest.mark.unit
class TestFilings(_SecTestBase):
    def test_window_lookahead_items_and_counts(self):
        self.route()
        out = sec.get_filings("NVDA", "2026-10-15", "2026-11-22", excerpts=0)
        self.assertTrue(out.startswith("## SEC EDGAR filings: NVDA (NVIDIA CORP, 2026-10-15 ~ 2026-11-22)"))
        self.assertNotIn("late.htm", out)  # filed after the analysis date
        self.assertNotIn("old.htm", out)  # before the window
        self.assertIn("10-Q (period 2026-10-26)", out)
        self.assertIn("Items: 2.02 Results of Operations and Financial Condition |", out)  # 9.01 dropped
        self.assertIn("5.02 Departure/Appointment of Directors or Officers", out)
        self.assertIn("https://www.sec.gov/Archives/edgar/data/1045810/000104581026000189/nvda-8k.htm", out)
        self.assertIn("Form 3/4/5 insider filings: 3 in range, latest 2026-11-18.", out)
        self.assertNotIn("form4.xml", out)  # Form 4s are counted, not listed
        self.assertIn("Other filings: 144 x1 (latest 2026-11-05).", out)
        listed = [l for l in out.splitlines() if l.startswith("- 2026")]
        self.assertEqual([l[2:12] for l in listed], ["2026-11-19", "2026-11-19", "2026-11-12"])

    def test_end_date_cuts_same_window(self):
        self.route()
        out = sec.get_filings("NVDA", "2026-10-15", "2026-11-18", excerpts=0)
        self.assertNotIn("10-Q", out)
        self.assertIn("officer-8k.htm", out)
        self.assertIn("latest 2026-11-18", out)

    def test_excerpt_is_clean_item_text(self):
        self.route()
        out = sec.get_filings("NVDA", "2026-10-15", "2026-11-22", excerpts=1)
        excerpt_lines = [l for l in out.splitlines() if l.startswith("  > ")]
        self.assertEqual(len(excerpt_lines), 1)  # only the newest 8-K
        ex = excerpt_lines[0]
        self.assertTrue(ex.startswith("  > Item 2.02 Results of Operations"))
        self.assertIn("third quarter & record data center revenue", ex)
        for junk in ("<", "tracking", "color:red", "Check the appropriate box", "EntityCentralIndexKey",
                     "SIGNATURES", "Pursuant to the requirements", "Exhibit 99.1"):
            self.assertNotIn(junk, ex)
        self.assertEqual(sum(u.endswith(".htm") for u in self.urls()), 1)

    def test_submissions_cached_between_calls(self):
        self.route()
        sec.get_filings("NVDA", "2026-10-15", "2026-11-22", excerpts=0)
        sec.get_filings("NVDA", "2026-11-01", "2026-11-22", excerpts=0)
        self.assertEqual(sum("submissions/CIK" in u for u in self.urls()), 1)

    def test_limit(self):
        self.route()
        out = sec.get_filings("NVDA", "2026-10-15", "2026-11-22", limit=1, excerpts=0)
        self.assertEqual(sum(l.startswith("- 2026") for l in out.splitlines()), 1)
        self.assertIn("2 more material filings", out)

    def test_older_history_pages_fetched_when_window_predates_recent(self):
        page_url = sec.SUBMISSIONS_PAGE_URL.format(name="CIK0001045810-submissions-001.json")
        old_rows = [("10-K", "2025-02-26", "2025-01-26", "0001045810-25-000023", "nvda-10k.htm", "10-K", "")]
        subs = json.loads(json.dumps(SUBMISSIONS))
        subs["filings"]["files"] = [{"name": "CIK0001045810-submissions-001.json",
                                     "filingFrom": "2020-01-01", "filingTo": "2025-12-31"}]
        self.route({sec.SUBMISSIONS_URL.format(cik=1045810): _resp(json_data=subs),
                    page_url: _resp(json_data=_block(old_rows))})
        out = sec.get_filings("NVDA", "2025-01-01", "2025-03-31")
        self.assertIn("10-K (period 2025-01-26) | Annual report", out)

    def test_non_us_and_unknown_tickers(self):
        self.route()
        self.assertIn("US-listed companies only", sec.get_filings("005930.KS", "2026-11-01", "2026-11-22"))
        self.assertIn("US-listed companies only", sec.get_filings("BTC-USD", "2026-11-01", "2026-11-22"))
        self.assertEqual(self.calls, [])
        out = sec.get_filings("KORU", "2026-11-01", "2026-11-22")
        self.assertIn("not in the SEC company ticker list", out)
        self.assertIn("SEC EDGAR", sec.get_filings("NVDA", "bad", "2026-11-22"))


@pytest.mark.unit
class TestHttpStatus(_SecTestBase):
    def test_403_explains_user_agent(self):
        self.route({sec.TICKERS_URL: _resp(status=403)})
        out = sec.get_filings("NVDA", "2026-11-01", "2026-11-22")
        self.assertIn("403", out)
        self.assertIn("User-Agent", out)
        self.assertEqual(len(self.calls), 1)  # no retry on 403

    def test_429_backs_off_and_retries_once(self):
        self.route({sec.TICKERS_URL: [_resp(status=429), _resp(json_data=TICKERS)]})
        self.assertEqual(sec.get_cik("NVDA")["cik"], 1045810)
        self.sleep.assert_called_once_with(sec.BACKOFF_429)

        self._reset_caches()
        (self.cache / "company_tickers.json").unlink()
        self.calls.clear()
        self.route({sec.TICKERS_URL: [_resp(status=429), _resp(status=429)]})
        out = sec.get_filings("NVDA", "2026-11-01", "2026-11-22")
        self.assertIn("429", out)
        self.assertEqual(len(self.calls), 2)

    def test_timeout_and_5xx_retry_once_then_short_error(self):
        sub_url = sec.SUBMISSIONS_URL.format(cik=1045810)
        self.route({sub_url: [_resp(status=503), _resp(status=502)]})
        out = sec.get_filings("NVDA", "2026-11-01", "2026-11-22")
        self.assertIn("server error", out)
        self.assertEqual(self.urls().count(sub_url), 2)

        state = {"n": 0}
        real = _router(self.calls)

        def flaky(url, headers=None, timeout=None):
            state["n"] += 1
            if state["n"] == 1:
                raise requests.Timeout("slow")
            return real(url, headers=headers, timeout=timeout)

        self._reset_caches()
        with patch.object(sec.requests, "get", side_effect=flaky):
            self.assertIn("## SEC EDGAR filings: NVDA", sec.get_filings("NVDA", "2026-11-01", "2026-11-22"))
        with patch.object(sec.requests, "get", side_effect=requests.Timeout("slow")):
            self.assertIn("timed out", sec.get_filing_text("https://www.sec.gov/Archives/a.htm"))


@pytest.mark.unit
class TestRateLimit(_SecTestBase):
    def test_requests_are_spaced(self):
        self.assertIsInstance(MODULE_LIMITER, RateLimiter)
        self.assertGreaterEqual(MODULE_LIMITER.min_interval, 0.12)

        clock = {"t": 1000.0}
        fake_time = MagicMock()
        fake_time.monotonic.side_effect = lambda: clock["t"]
        fake_time.sleep.side_effect = lambda s: clock.__setitem__("t", clock["t"] + s)
        stamps = []

        def get(url, headers=None, timeout=None):
            stamps.append(clock["t"])
            return _resp(text="<p>doc</p>")

        with patch("tradingagents.dataflows.external_common.time", fake_time), \
                patch.object(sec, "_LIMITER", RateLimiter(sec.MIN_INTERVAL)), \
                patch.object(sec.requests, "get", side_effect=get):
            for _ in range(4):
                sec.get_filing_text("https://www.sec.gov/Archives/a.htm")
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        self.assertEqual(len(gaps), 3)
        for gap in gaps:
            self.assertGreaterEqual(gap, sec.MIN_INTERVAL - 1e-9)


@pytest.mark.unit
class TestFilingText(_SecTestBase):
    def test_url_allow_list(self):
        self.route()
        for bad in ("http://www.sec.gov/Archives/a.htm", "https://evil.example.com/a.htm",
                    "https://www.sec.gov.evil.com/a.htm", "file:///etc/passwd", "", None):
            self.assertIn("Only https://www.sec.gov/", sec.get_filing_text(bad))
        self.assertEqual(self.calls, [])

    def test_clean_text_and_truncation(self):
        self.route()
        out = sec.get_filing_text("https://www.sec.gov/Archives/edgar/data/1045810/x/nvda-8k.htm")
        self.assertTrue(out.startswith("### SEC document: https://www.sec.gov/"))
        self.assertIn("Item 2.02 Results of Operations", out)
        self.assertNotIn("<", out.split("\n", 1)[1])
        self.assertNotIn("Check the appropriate box", out)
        short = sec.get_filing_text("https://www.sec.gov/Archives/edgar/data/1045810/x/nvda-8k.htm",
                                    max_chars=200)
        self.assertLessEqual(len(short.split("\n", 1)[1]), 205)

    def test_binary_documents_rejected(self):
        self.route({"https://www.sec.gov/a.pdf": _resp(text="%PDF", headers={"Content-Type": "application/pdf"})})
        self.assertIn("Unsupported", sec.get_filing_text("https://www.sec.gov/a.pdf"))


if __name__ == "__main__":
    unittest.main()
