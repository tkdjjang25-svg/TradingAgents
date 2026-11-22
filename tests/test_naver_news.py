import hashlib
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.dataflows import naver_news as nn  # noqa: E402
from tradingagents.dataflows.external_common import RateLimiter  # noqa: E402

CID = "test-client-id-XYZ"
SECRET = "test-secret-ABC123"


def _env(cid=CID, secret=SECRET):
    values = {"NAVER_CLIENT_ID": cid, "NAVER_CLIENT_SECRET": secret}
    return patch.object(nn, "get_env", side_effect=lambda name: values.get(name, ""))


def _item(title, pub, origin="https://www.hankyung.com/article/1", link=None, desc="본문 요약"):
    return {"title": title, "originallink": origin,
            "link": link or "https://n.news.naver.com/mnews/article/015/0000000001",
            "description": desc, "pubDate": pub}


def _resp(status=200, items=None, total=None, body=None):
    r = MagicMock()
    r.status_code = status
    if body is not None:
        r.json.return_value = body
    else:
        items = items or []
        r.json.return_value = {"items": items, "total": len(items) if total is None else total}
    return r


class _Base(unittest.TestCase):
    def setUp(self):
        patches = [patch.object(nn, "_limiter", RateLimiter(0)),
                   patch.object(nn, "_RETRY_BACKOFF", 0)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


@pytest.mark.unit
class TestCleaning(unittest.TestCase):
    def test_tags_and_entities(self):
        raw = "<b>삼성전자</b>, &quot;HBM&quot; 공급 &amp; 증설 &lt;단독&gt; 투자자&#39;s   관심"
        self.assertEqual(nn._clean_text(raw), '삼성전자, "HBM" 공급 & 증설 <단독> 투자자\'s 관심')

    def test_double_escaped(self):
        self.assertEqual(nn._clean_text("&amp;quot;반도체&amp;quot;"), '"반도체"')
        self.assertEqual(nn._clean_text("&lt;b&gt;반도체&lt;/b&gt; 호황"), "반도체 호황")

    def test_publisher_mapping_and_fallback(self):
        self.assertEqual(nn._publisher("https://www.hankyung.com/a/1", ""), "한국경제")
        self.assertEqual(nn._publisher("https://m.mk.co.kr/news/1", ""), "매일경제")
        self.assertEqual(nn._publisher("https://biz.chosun.com/it/1", ""), "조선비즈")
        self.assertEqual(nn._publisher("https://www.chosun.com/economy/1", ""), "조선일보")
        self.assertEqual(nn._publisher("https://www.yna.co.kr/view/1", ""), "연합뉴스")
        self.assertEqual(nn._publisher("https://www.smallpaper.co.kr/n/1", ""), "smallpaper.co.kr")
        self.assertEqual(nn._publisher("", "https://n.news.naver.com/mnews/article/1"), "네이버뉴스")

    def test_summary_noise_removed(self):
        raw = ("(서울=연합뉴스) 홍길동 기자 = 삼성전자가 HBM4 양산을 시작했다. "
               "hong@example.com 무단전재 및 재배포 금지")
        self.assertEqual(nn._clean_summary(raw), "삼성전자가 HBM4 양산을 시작했다.")
        raw2 = "[이데일리 김철수 기자] SK하이닉스 실적 발표 <저작권자 ⓒ 이데일리>"
        self.assertEqual(nn._clean_summary(raw2), "SK하이닉스 실적 발표")

    def test_summary_trimmed_on_word_boundary(self):
        raw = " ".join(["반도체업황"] * 60)
        out = nn._clean_summary(raw)
        self.assertLessEqual(len(out), 161)
        self.assertTrue(out.endswith("…"))
        self.assertTrue(out[:-1].endswith("반도체업황"))  # not cut mid-word


@pytest.mark.unit
class TestDedupe(unittest.TestCase):
    def _arts(self, rows):
        from datetime import datetime
        out = []
        for title, ts, url, summary in rows:
            dt = datetime.fromisoformat(ts).replace(tzinfo=nn.KST)
            out.append({"title": title, "summary": summary, "publisher": "x",
                        "published": ts, "url": url, "_dt": dt})
        return sorted(out, key=lambda a: a["_dt"], reverse=True)

    def test_syndicated_variants_merged_keep_earliest(self):
        arts = self._arts([
            ("[속보] 삼성전자, 3분기 영업이익 12조원…시장 전망 상회", "2026-11-20 09:05", "u1", "a"),
            ("삼성전자 3분기 영업이익 12조원, 시장 전망 상회(종합)", "2026-11-20 11:00", "u2", "longer one"),
            ("[단독]삼성전자, 3분기 영업이익 12조원… 시장전망 상회", "2026-11-20 09:30", "u3", "b"),
            ("SK하이닉스, HBM4 엔비디아 공급 확정", "2026-11-20 10:00", "u4", "c"),
        ])
        out = nn._dedupe(arts)
        self.assertEqual(len(out), 2)
        samsung = next(a for a in out if "삼성" in a["title"])
        self.assertEqual(samsung["url"], "u1")  # earliest report kept
        self.assertEqual(samsung["similar"], 2)
        self.assertEqual(out[0]["url"], "u4")  # newest first

    def test_exact_url_dupe_and_distinct_titles_kept(self):
        arts = self._arts([
            ("삼성전자 주가 상승", "2026-11-20 10:00", "same", "a"),
            ("완전히 다른 제목의 기사", "2026-11-20 09:00", "same", "b"),
            ("삼성전자 주가 하락", "2026-11-20 08:00", "other", "c"),
        ])
        out = nn._dedupe(arts)
        self.assertEqual([a["title"] for a in out], ["삼성전자 주가 상승", "삼성전자 주가 하락"])


@pytest.mark.unit
class TestSearch(_Base):
    def test_window_filter_and_kst_conversion(self):
        items = [
            _item("미래 기사", "Mon, 23 Nov 2026 00:10:00 +0900"),  # after end_date -> dropped
            # 2026-11-22 14:59 UTC == 23:59 KST on the end date -> kept
            _item("<b>마감 직전</b> 기사", "Sun, 22 Nov 2026 14:59:00 +0000",
                  origin="https://www.mk.co.kr/news/2"),
            # 2026-11-22 15:00 UTC == 2026-11-23 00:00 KST -> look-ahead, dropped
            _item("자정 기사 UTC", "Sun, 22 Nov 2026 15:00:00 +0000"),
            _item("기간 내 기사", "Fri, 20 Nov 2026 09:30:00 +0900", origin="https://www.yna.co.kr/v/3"),
            _item("너무 오래된 기사", "Tue, 17 Nov 2026 23:59:00 +0900"),
        ]
        with _env(), patch.object(nn.requests, "get", return_value=_resp(items=items)) as get:
            out = nn.search_news("삼성전자", "2026-11-18", "2026-11-22")
        self.assertEqual([a["title"] for a in out], ["마감 직전 기사", "기간 내 기사"])
        self.assertEqual(out[0]["published"], "2026-11-22 23:59")
        self.assertEqual(out[0]["publisher"], "매일경제")
        self.assertEqual(out[1]["url"], "https://www.yna.co.kr/v/3")
        self.assertNotIn("_dt", out[0])
        _, kwargs = get.call_args
        self.assertEqual(kwargs["params"]["sort"], "date")
        self.assertEqual(kwargs["params"]["display"], 100)
        self.assertEqual(kwargs["headers"]["X-Naver-Client-Id"], CID)
        self.assertEqual(kwargs["timeout"], nn._TIMEOUT)

    def _page(self, day, n=100, prefix="기사"):
        return [_item(f"{prefix} {hashlib.md5(f'{day}-{i}'.encode()).hexdigest()[:16]}",
                      f"Fri, {day} Nov 2026 {10 + i % 10:02d}:{i % 60:02d}:00 +0900",
                      origin=f"https://e.example.com/{day}/{i}") for i in range(n)]

    def test_paging_stops_once_older_than_start(self):
        page1 = self._page(23)  # all after end_date: need the next page
        page2 = self._page(19) + []  # reaches before start_date -> stop
        page2[-1] = _item("오래된 기사", "Mon, 16 Nov 2026 10:00:00 +0900", origin="https://e.example.com/old")
        with _env(), patch.object(nn.requests, "get",
                                  side_effect=[_resp(items=page1, total=900),
                                               _resp(items=page2, total=900)]) as get:
            out = nn.search_news("삼성전자", "2026-11-18", "2026-11-22", limit=200)
        self.assertEqual(get.call_count, 2)
        self.assertEqual([c.kwargs["params"]["start"] for c in get.call_args_list], [1, 101])
        self.assertTrue(out)
        self.assertTrue(all(a["published"].startswith("2026-11-19") for a in out))

    def test_paging_capped_and_stops_on_short_page(self):
        pages = [_resp(items=self._page(22), total=1000)] * 5
        with _env(), patch.object(nn.requests, "get", side_effect=pages) as get:
            nn.search_news("x", "2026-11-01", "2026-11-22", limit=100)
        self.assertLessEqual(get.call_count, nn._MAX_PAGES)

        with _env(), patch.object(nn.requests, "get",
                                  return_value=_resp(items=self._page(22, n=30), total=30)) as get:
            nn.search_news("x", "2026-11-01", "2026-11-22", limit=100)
        self.assertEqual(get.call_count, 1)

    def test_paging_stops_when_enough_stories(self):
        with _env(), patch.object(nn.requests, "get",
                                  return_value=_resp(items=self._page(22), total=1000)) as get:
            out = nn.search_news("x", "2026-11-01", "2026-11-22", limit=5)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(len(out), 5)

    def test_missing_credentials(self):
        with _env(cid="", secret=""), patch.object(nn.requests, "get") as get:
            self.assertEqual(nn.search_news("삼성전자"), [])
            msg = nn.get_news_naver("삼성전자", "2026-11-19", "2026-11-22")
        get.assert_not_called()
        self.assertIn("not configured", msg)
        self.assertIn("NAVER_CLIENT_ID", msg)

    def test_bad_date(self):
        with _env():
            with self.assertRaises(ValueError):
                nn.search_news("x", "2026/11/01", "2026-11-22")
            self.assertIn("YYYY-MM-DD", nn.get_news_naver("x", "nope", "2026-11-22"))


@pytest.mark.unit
class TestErrors(_Base):
    def test_401_no_retry_and_no_secret_leak(self):
        with _env(), patch.object(nn.requests, "get",
                                  return_value=_resp(401, body={"errorCode": "024"})) as get:
            with self.assertRaises(nn.NaverNewsError) as ctx:
                nn.search_news("삼성전자")
            msg = nn.get_news_naver("삼성전자", "2026-11-19", "2026-11-22")
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(get.call_count, 2)  # one call each, no retry on 401
        self.assertIn("401", msg)
        self.assertIn("credentials", msg)
        self.assertNotIn(SECRET, msg)
        self.assertNotIn(CID, msg)

    def test_429_retried_once_then_error(self):
        with _env(), patch.object(nn.requests, "get", return_value=_resp(429)) as get:
            msg = nn.get_news_naver("삼성전자", "2026-11-19", "2026-11-22")
        self.assertEqual(get.call_count, 2)
        self.assertIn("429", msg)
        self.assertNotIn(SECRET, msg)

    def test_429_then_success(self):
        ok = _resp(items=[_item("기사", "Fri, 20 Nov 2026 09:30:00 +0900")])
        with _env(), patch.object(nn.requests, "get", side_effect=[_resp(429), ok]):
            out = nn.search_news("x", "2026-11-19", "2026-11-22")
        self.assertEqual(len(out), 1)

    def test_timeout_retried_then_error(self):
        with _env(), patch.object(nn.requests, "get",
                                  side_effect=requests.Timeout(f"secret {SECRET}")) as get:
            msg = nn.get_news_naver("x", "2026-11-19", "2026-11-22")
        self.assertEqual(get.call_count, 2)
        self.assertIn("Timeout", msg)
        self.assertNotIn(SECRET, msg)

    def test_500_retry_and_partial_results_kept(self):
        page1 = TestSearch._page(self, 21)
        with _env(), patch.object(nn.requests, "get",
                                  side_effect=[_resp(items=page1, total=900), _resp(500), _resp(500)]):
            out = nn.search_news("x", "2026-11-01", "2026-11-22", limit=100)
        self.assertTrue(out)  # page 1 survives a failing page 2


@pytest.mark.unit
class TestMarkdown(_Base):
    def test_format(self):
        items = [
            _item("<b>삼성전자</b> &quot;HBM4&quot; 양산", "Fri, 20 Nov 2026 09:30:00 +0900",
                  origin="https://www.hankyung.com/article/9",
                  desc="(서울=연합뉴스) 홍길동 기자 = 삼성전자가 양산을 시작했다."),
            _item("[속보] 삼성전자 \"HBM4\" 양산", "Fri, 20 Nov 2026 10:30:00 +0900",
                  origin="https://www.newsis.com/view/9"),
        ]
        with _env(), patch.object(nn.requests, "get", return_value=_resp(items=items)):
            md = nn.get_news_naver("삼성전자", "2026-11-19", "2026-11-22")
        self.assertTrue(md.startswith('## Naver News: "삼성전자" (2026-11-19 ~ 2026-11-22)'))
        self.assertIn('### 삼성전자 "HBM4" 양산 (한국경제, 2026-11-20 09:30, +1 similar)', md)
        self.assertIn("\n삼성전자가 양산을 시작했다.\n", md)
        self.assertIn("https://www.hankyung.com/article/9", md)
        self.assertNotIn("newsis", md)
        self.assertNotIn("<b>", md)
        self.assertNotIn(SECRET, md)

    def test_empty(self):
        with _env(), patch.object(nn.requests, "get", return_value=_resp(items=[])):
            md = nn.get_news_naver("없는회사", "2026-11-19", "2026-11-22")
        self.assertIn("No Naver News articles", md)


if __name__ == "__main__":
    unittest.main()
