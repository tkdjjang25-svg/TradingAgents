import io
import json
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.dataflows import dart  # noqa: E402

KEY = "SECRETKEY1234567890abcdef"

_CORP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<result>
  <list><corp_code>00126380</corp_code><corp_name>삼성전자</corp_name><stock_code>005930</stock_code><modify_date>20240101</modify_date></list>
  <list><corp_code>00999999</corp_code><corp_name>비상장회사</corp_name><stock_code> </stock_code><modify_date>20240101</modify_date></list>
  <list><corp_code>00258801</corp_code><corp_name>카카오</corp_name><stock_code>035720</stock_code><modify_date>20240101</modify_date></list>
  <list><corp_code>01111111</corp_code><corp_name>에코프로비엠</corp_name><stock_code>247540</stock_code><modify_date>20240101</modify_date></list>
</result>"""


def _corp_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("CORPCODE.xml", _CORP_XML.encode("utf-8"))
    return buf.getvalue()


def _resp(json_data=None, content=b"", status_code=200):
    r = MagicMock()
    r.status_code = status_code
    r.content = content
    if json_data is None:
        r.json.side_effect = ValueError("no json")
    else:
        r.json.return_value = json_data
    return r


class _DartTestBase(unittest.TestCase):
    """Temp cache dir, fresh in-memory cache, no rate-limit/retry sleeps, a fake key."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)
        patches = [
            patch.object(dart, "cache_dir", lambda name: self.cache),
            patch.object(dart, "_CORP", dart._CorpCache()),
            patch.object(dart._LIMITER, "wait", lambda: None),
            patch.object(dart, "_RETRY_DELAY", 0),
            patch.object(dart, "get_env", lambda name: KEY if name == "DART_API_KEY" else ""),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)

    def route(self, handlers):
        """Patch requests.get; handlers maps endpoint -> callable(params) -> response."""
        calls = []

        def fake_get(url, params=None, timeout=None):
            endpoint = url.rsplit("/", 1)[-1]
            calls.append((endpoint, dict(params or {})))
            self.assertEqual(params.get("crtfc_key"), KEY)
            self.assertIsNotNone(timeout)
            return handlers[endpoint](params)

        p = patch.object(dart.requests, "get", side_effect=fake_get)
        p.start()
        self.addCleanup(p.stop)
        return calls


def _corp_handler(params):
    return _resp(content=_corp_zip())


@pytest.mark.unit
class TestCorpCodes(_DartTestBase):
    def test_parses_zip_and_skips_unlisted(self):
        self.route({"corpCode.xml": _corp_handler})
        info = dart.get_corp_info("005930.KS")
        self.assertEqual(info, {"corp_code": "00126380", "corp_name": "삼성전자", "stock_code": "005930"})
        payload = json.loads((self.cache / "corp_codes.json").read_text(encoding="utf-8"))
        self.assertEqual(set(payload["corps"]), {"005930", "035720", "247540"})
        self.assertIn("fetched_at", payload)
        self.assertEqual(list(self.cache.glob("*.tmp")), [])

    def test_ticker_forms(self):
        self.route({"corpCode.xml": _corp_handler})
        self.assertEqual(dart.get_corp_info("035720.KS")["corp_name"], "카카오")
        self.assertEqual(dart.get_corp_info("247540.kq")["corp_code"], "01111111")
        self.assertEqual(dart.get_corp_info("005930")["corp_code"], "00126380")
        self.assertIsNone(dart.get_corp_info("AAPL"))
        self.assertIsNone(dart.get_corp_info("7203.T"))
        self.assertIsNone(dart.get_corp_info("123456.KS"))  # not in the mapping

    def test_memory_and_file_cache_reuse(self):
        calls = self.route({"corpCode.xml": _corp_handler})
        dart.get_corp_info("005930.KS")
        dart.get_corp_info("035720.KS")
        self.assertEqual(len(calls), 1)
        # New process: memory empty, file cache fresh -> still no download.
        with patch.object(dart, "_CORP", dart._CorpCache()):
            self.assertEqual(dart.get_corp_info("005930.KS")["corp_code"], "00126380")
        self.assertEqual(len(calls), 1)

    def test_refresh_after_seven_days(self):
        calls = self.route({"corpCode.xml": _corp_handler})
        old = time.time() - 8 * 24 * 3600
        (self.cache / "corp_codes.json").write_text(json.dumps(
            {"fetched_at": old, "corps": {"005930": {"corp_code": "OLD", "corp_name": "옛이름"}}}),
            encoding="utf-8")
        self.assertEqual(dart.get_corp_info("005930.KS")["corp_code"], "00126380")
        self.assertEqual(len(calls), 1)
        payload = json.loads((self.cache / "corp_codes.json").read_text(encoding="utf-8"))
        self.assertGreater(payload["fetched_at"], old)

    def test_six_day_old_cache_is_used(self):
        calls = self.route({"corpCode.xml": _corp_handler})
        (self.cache / "corp_codes.json").write_text(json.dumps(
            {"fetched_at": time.time() - 6 * 24 * 3600,
             "corps": {"005930": {"corp_code": "CACHED", "corp_name": "삼성전자"}}}), encoding="utf-8")
        self.assertEqual(dart.get_corp_info("005930.KS")["corp_code"], "CACHED")
        self.assertEqual(calls, [])

    def test_stale_cache_used_when_refresh_fails(self):
        calls = self.route({"corpCode.xml": lambda p: _resp(status_code=503)})
        (self.cache / "corp_codes.json").write_text(json.dumps(
            {"fetched_at": time.time() - 30 * 24 * 3600,
             "corps": {"005930": {"corp_code": "STALE", "corp_name": "삼성전자"}}}), encoding="utf-8")
        self.assertEqual(dart.get_corp_info("005930.KS")["corp_code"], "STALE")
        n = len(calls)
        self.assertEqual(n, 2)  # one retry on 5xx
        dart.get_corp_info("005930.KS")
        self.assertEqual(len(calls), n)  # no re-download right after a failure

    def test_corpcode_key_error_xml(self):
        self.route({"corpCode.xml": lambda p: _resp(
            content="<result><status>010</status><message>등록되지 않은 키</message></result>".encode())})
        msg = dart.get_disclosures("005930.KS", "2026-11-01", "2026-11-22")
        self.assertIn("010", msg)
        self.assertNotIn(KEY, msg)


def _row(dt, name, no_suffix, filer="삼성전자"):
    return {"rcept_dt": dt, "report_nm": name, "flr_nm": filer, "rcept_no": f"{dt}{no_suffix}",
            "corp_name": "삼성전자"}


@pytest.mark.unit
class TestDisclosures(_DartTestBase):
    def _list(self, rows=None, status="000", total_page=1):
        def handler(params):
            if status != "000":
                return _resp({"status": status, "message": "msg"})
            return _resp({"status": "000", "message": "정상", "page_no": params["page_no"],
                          "total_count": len(rows), "total_page": total_page, "list": rows})
        return self.route({"corpCode.xml": _corp_handler, "list.json": handler})

    def test_formatting_stars_and_ordering(self):
        rows = [
            _row("20261120", "임원ㆍ주요주주특정증권등소유상황보고서", "000101", "홍길동"),
            _row("20261118", "단일판매ㆍ공급계약체결", "000102"),
            _row("20261115", "기업설명회(IR)개최(안내공시)", "000103"),
            _row("20261110", "주요사항보고서(자기주식취득결정)", "000104"),
            _row("20261105", "연결재무제표기준영업(잠정)실적(공정공시)", "000105"),
        ]
        calls = self._list(rows)
        out = dart.get_disclosures("005930.KS", "2026-11-01", "2026-11-22")
        lines = out.splitlines()
        self.assertTrue(lines[0].startswith("## DART"))
        self.assertIn("삼성전자", lines[0])
        self.assertIn("2026-11-01 ~ 2026-11-22", lines[0])
        items = [l for l in lines if l.startswith("- ")]
        self.assertEqual(len(items), 5)
        # Important filings first (newest first among them), then routine ones.
        self.assertIn("단일판매", items[0])
        self.assertIn("자기주식", items[1])
        self.assertIn("영업(잠정)실적", items[2])
        self.assertTrue(all(i.startswith("- ★") for i in items[:3]))
        self.assertFalse(any("★" in i for i in items[3:]))
        self.assertIn("2026-11-20", items[3])
        self.assertIn("홍길동", items[3])
        self.assertIn("https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20261118000102", items[0])
        params = calls[-1][1]
        self.assertEqual((params["bgn_de"], params["end_de"], params["corp_code"]),
                         ("20261101", "20261122", "00126380"))
        self.assertNotIn(KEY, out)

    def test_never_includes_filings_after_end_date(self):
        rows = [_row("20261125", "단일판매ㆍ공급계약체결", "000201"),
                _row("20261121", "분기보고서 (2026.09)", "000202")]
        self._list(rows)
        out = dart.get_disclosures("005930.KS", "2026-11-01", "2026-11-22")
        self.assertNotIn("20261125", out)
        self.assertNotIn("2026-11-25", out)
        self.assertIn("분기보고서", out)

    def test_limit_and_paging(self):
        pages = {1: [_row("20261120", f"기타공시{i}", f"{i:06d}") for i in range(99)]
                 + [_row("20261102", "유상증자결정", "888888")],
                 2: [_row("20261101", "무상증자결정", "999999")]}
        calls = self.route({"corpCode.xml": _corp_handler, "list.json": lambda p: _resp({
            "status": "000", "total_count": 101, "total_page": 2, "list": pages[p["page_no"]]})})
        out = dart.get_disclosures("005930.KS", "2026-10-01", "2026-11-22", limit=5)
        items = [l for l in out.splitlines() if l.startswith("- ")]
        self.assertEqual(len(items), 5)
        self.assertIn("유상증자", items[0])  # important filing surfaces despite routine noise
        self.assertIn("생략", out)
        self.assertEqual(len([c for c in calls if c[0] == "list.json"]), 1)  # one full page is enough

        out = dart.get_disclosures("005930.KS", "2026-10-01", "2026-11-22", limit=150)
        items = [l for l in out.splitlines() if l.startswith("- ")]
        self.assertEqual(len(items), 101)
        self.assertIn("유상증자", items[0])
        self.assertIn("무상증자", items[1])
        self.assertEqual([c[1]["page_no"] for c in calls if c[0] == "list.json"], [1, 1, 2])

    def test_status_013_no_data(self):
        self._list(status="013")
        out = dart.get_disclosures("005930.KS", "2026-11-01", "2026-11-22")
        self.assertIn("No disclosures in range", out)

    def test_status_020_limit(self):
        self._list(status="020")
        out = dart.get_disclosures("005930.KS", "2026-11-01", "2026-11-22")
        self.assertIn("limit", out.lower())
        self.assertIn("020", out)

    def test_key_errors(self):
        for status in ("010", "011", "012", "901"):
            with self.subTest(status=status):
                with patch.object(dart, "_CORP", dart._CorpCache()):
                    p = patch.object(dart.requests, "get", side_effect=lambda url, params=None, timeout=None: (
                        _resp(content=_corp_zip()) if url.endswith("corpCode.xml")
                        else _resp({"status": status, "message": "x"})))
                    with p:
                        out = dart.get_disclosures("005930.KS", "2026-11-01", "2026-11-22")
                self.assertIn(status, out)
                self.assertIn("key", out.lower())
                self.assertNotIn(KEY, out)

    def test_non_korean_and_unknown(self):
        self.route({"corpCode.xml": _corp_handler})
        self.assertIn("Korean", dart.get_disclosures("NVDA", "2026-11-01", "2026-11-22"))
        self.assertIn("123456", dart.get_disclosures("123456.KQ", "2026-11-01", "2026-11-22"))
        self.assertIn("Korean", dart.get_financial_summary("BTC-USD", "2026-11-22"))

    def test_missing_key(self):
        with patch.object(dart, "get_env", lambda name: ""):
            with patch.object(dart.requests, "get") as get:
                out = dart.get_disclosures("005930.KS", "2026-11-01", "2026-11-22")
                out2 = dart.get_financial_summary("005930.KS", "2026-11-22")
                self.assertIsNone(dart.get_corp_info("005930.KS"))
            get.assert_not_called()
        for o in (out, out2):
            self.assertIn("DART_API_KEY", o)
            self.assertIn("not configured", o)

    def test_network_error_scrubs_key(self):
        url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={KEY}"
        err = requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {url}")
        with patch.object(dart.requests, "get", side_effect=err) as get:
            out = dart.get_disclosures("005930.KS", "2026-11-01", "2026-11-22")
        self.assertEqual(get.call_count, 2)  # one retry
        self.assertIn("ConnectionError", out)
        self.assertNotIn(KEY, out)
        self.assertIn("crtfc_key=***", out)

    def test_timeout_on_list_scrubbed(self):
        def boom(params):
            raise requests.Timeout(f"Read timed out. url=...list.json?crtfc_key={KEY}&corp_code=1")
        self.route({"corpCode.xml": _corp_handler, "list.json": boom})
        out = dart.get_disclosures("005930.KS", "2026-11-01", "2026-11-22")
        self.assertIn("Timeout", out)
        self.assertNotIn(KEY, out)


def _fin_rows(rcept_no, fs_divs=("CFS", "OFS"), revenue=(79_000_000_000_000, 67_000_000_000_000)):
    rows = []
    for fs in fs_divs:
        mult = 1 if fs == "CFS" else 0.5
        for sj, name, cur, prev, add_cur, add_prev in [
            ("IS", "매출액", revenue[0], revenue[1], revenue[0] * 3, revenue[1] * 3),
            ("IS", "영업이익", 9_100_000_000_000, -2_400_000_000_000, 20_000_000_000_000, 5_000_000_000_000),
            ("IS", "당기순이익(손실)", 10_000_000_000_000, 5_800_000_000_000, None, None),
            ("BS", "자산총계", 500_000_000_000_000, 455_000_000_000_000, None, None),
            ("BS", "부채총계", 100_000_000_000_000, 92_000_000_000_000, None, None),
            ("BS", "자본총계", 400_000_000_000_000, 363_000_000_000_000, None, None),
        ]:
            fmt = lambda v: "" if v is None else f"{int(v * mult):,}"
            rows.append({"rcept_no": rcept_no, "fs_div": fs, "sj_div": sj, "account_nm": name,
                         "thstrm_amount": fmt(cur), "frmtrm_amount": fmt(prev),
                         "thstrm_add_amount": fmt(add_cur), "frmtrm_add_amount": fmt(add_prev),
                         "currency": "KRW"})
    return rows


@pytest.mark.unit
class TestFinancialSummary(_DartTestBase):
    def _fin(self, reports):
        """reports: {(year, reprt_code): rows or None (=013)}"""
        def handler(params):
            rows = reports.get((int(params["bsns_year"]), params["reprt_code"]))
            if rows is None:
                return _resp({"status": "013", "message": "조회된 데이타가 없습니다."})
            return _resp({"status": "000", "message": "정상", "list": rows})
        return self.route({"corpCode.xml": _corp_handler, "fnlttSinglAcnt.json": handler})

    def test_prefers_cfs_and_shows_yoy(self):
        self._fin({(2026, "11014"): _fin_rows("20261114000123"),
                   (2025, "11011"): _fin_rows("20260310000456")})
        out = dart.get_financial_summary("005930.KS", "2026-11-22")
        self.assertTrue(out.startswith("## DART"))
        self.assertIn("2026 3분기보고서 (공시 2026-11-14)", out)
        self.assertIn("2025 사업보고서 (공시 2026-03-10)", out)
        self.assertIn("연결재무제표", out)
        self.assertNotIn("별도재무제표", out)
        self.assertIn("| 매출액 | 79.00조 | 67.00조 | +17.9% |", out)
        self.assertIn("| 영업이익 | 9.10조 | -2.40조 | 흑자전환 |", out)
        self.assertIn("| 매출액(누적) | 237.00조 | 201.00조 | +17.9% |", out)
        self.assertIn("| 자산총계 | 500.00조 | 455.00조 | +9.9% |", out)
        self.assertIn("rcpNo=20261114000123", out)
        self.assertNotIn(KEY, out)

    def test_falls_back_to_ofs(self):
        self._fin({(2026, "11014"): _fin_rows("20261114000123", fs_divs=("OFS",))})
        out = dart.get_financial_summary("005930.KS", "2026-11-22")
        self.assertIn("별도재무제표", out)
        self.assertIn("| 매출액 | 39.50조 |", out)

    def test_drops_reports_filed_after_curr_date(self):
        calls = self._fin({(2026, "11014"): _fin_rows("20261114000123"),
                           (2026, "11012"): _fin_rows("20260814000777"),
                           (2025, "11011"): _fin_rows("20260310000456")})
        out = dart.get_financial_summary("005930.KS", "2026-11-10")
        self.assertNotIn("3분기보고서", out)
        self.assertNotIn("20261114", out)
        self.assertIn("2026 반기보고서 (공시 2026-08-14)", out)
        self.assertIn("2025 사업보고서", out)
        # Periods that had not ended yet are never requested.
        asked = [(c[1]["bsns_year"], c[1]["reprt_code"]) for c in calls if c[0] == "fnlttSinglAcnt.json"]
        self.assertNotIn(("2026", "11011"), asked)

    def test_skips_no_data_and_reports_none(self):
        self._fin({})
        out = dart.get_financial_summary("005930.KS", "2026-11-22")
        self.assertIn("No periodic financial report", out)

    def test_limit_error(self):
        self.route({"corpCode.xml": _corp_handler,
                    "fnlttSinglAcnt.json": lambda p: _resp({"status": "020", "message": "limit"})})
        out = dart.get_financial_summary("005930.KS", "2026-11-22")
        self.assertIn("020", out)

    def test_bad_date(self):
        self.route({"corpCode.xml": _corp_handler})
        self.assertIn("Invalid date", dart.get_financial_summary("005930.KS", "yesterday"))


if __name__ == "__main__":
    unittest.main()
