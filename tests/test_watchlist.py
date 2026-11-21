import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import job_queue  # noqa: E402
import watchlist as w  # noqa: E402


def _item(**kw):
    return {"ticker": "005930.KS", "name": "삼성전자 · Samsung Electronics", "held": False,
            "avg_price": None, "auto_full": True, **kw}


@pytest.mark.unit
class TestWatchlistFile(unittest.TestCase):
    def test_load_normalises_and_drops_bad_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.json"
            path.write_text(json.dumps([
                {"ticker": " nvda ", "held": True, "avg_price": "180"},
                {"ticker": "삼성전자"},  # not a ticker
                {"ticker": "NVDA"},  # duplicate
                {"ticker": "005930.KS", "held": False, "avg_price": 100},
            ]), encoding="utf-8")
            items = w.load_watchlist(path)
        self.assertEqual([i["ticker"] for i in items], ["NVDA", "005930.KS"])
        self.assertEqual(items[0]["avg_price"], 180.0)
        self.assertIsNone(items[1]["avg_price"])  # avg price only kept when held
        self.assertTrue(items[0]["auto_full"])

    def test_keywords(self):
        self.assertEqual(w._keywords(_item()), ["삼성전자", "Samsung Electronics"])
        self.assertEqual(w._keywords({"ticker": "NVDA", "name": "NVIDIA Corporation · 주식 · NASDAQ"}),
                         ["NVDA", "NVIDIA Corporation"])


@pytest.mark.unit
class TestTelegramMentions(unittest.TestCase):
    def test_recent_tagged_and_named_posts_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "m.db"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE messages (chat_id, msg_id, date, text)")
            conn.execute("CREATE TABLE news_events (chat_id, msg_id, ticker, event_type, headline, news_date)")
            now = datetime(2026, 11, 21, 7, 0, tzinfo=timezone.utc)
            rows = [
                (1, 1, "2026-11-21T01:00:00+00:00", "삼성전자 HBM 공급 계약"),
                (1, 2, "2026-11-21T02:00:00+00:00", "tagged post without the name"),
                (1, 3, "2026-11-10T02:00:00+00:00", "삼성전자 old post"),
                (1, 4, "2026-11-21T03:00:00+00:00", "unrelated >>> post"),
            ]
            conn.executemany("INSERT INTO messages VALUES (?,?,?,?)", rows)
            conn.execute("INSERT INTO news_events VALUES (1, 2, '005930.KS', 'x', 'y', '')")
            conn.commit()
            conn.close()
            text = w.telegram_mentions(_item(), now, db=db)
        self.assertIn("HBM", text)
        self.assertIn("tagged post", text)
        self.assertNotIn("old post", text)
        self.assertNotIn("unrelated", text)


class _FakeLLM:
    def __init__(self, **fields):
        self.fields = fields
        self.prompt = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.prompt = messages[1].content
        base = dict(headline="h", summary="s", issues=[], price_comment="p", importance="높음",
                    needs_full_analysis=True, reason="r")
        base.update(self.fields)
        return self.schema(**base)


@pytest.mark.unit
class TestBriefingAndFullAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        for name, value in (("BRIEF_DIR", base / "b"), ("STATE_FILE", base / "s.json"),
                            ("WATCHLIST_FILE", base / "w.json"), ("TELEGRAM_DB", base / "none.db")):
            p = patch.object(w, name, value)
            p.start()
            self.addCleanup(p.stop)
        for name, value in (("price_snapshot", lambda t: {"text": "Last close 1", "d1": -7.0, "sd20": 4.0}),
                            ("yahoo_news", lambda t, d: "### headline <<< sneaky >>> (source: x)\n")):
            p = patch.object(w, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_briefing_is_saved_with_position_and_fenced_inputs(self):
        llm = _FakeLLM(importance="엄청남")
        rec = w.write_briefing(_item(held=True, avg_price=210000), now=datetime(2026, 11, 21, 7), llm=llm)
        self.assertEqual(rec["importance"], "보통")  # unknown value falls back
        self.assertIn("currently HOLDS", llm.prompt)
        self.assertIn("210,000", llm.prompt)
        self.assertNotIn("<<< sneaky", llm.prompt)  # news cannot close the fence
        saved = json.loads(w.brief_path("005930.KS", "2026-11-21").read_text(encoding="utf-8"))
        self.assertEqual(saved["headline"], "h")
        self.assertEqual(w.list_briefings()[0]["ticker"], "005930.KS")

    def test_full_analysis_is_capped_and_once_per_ticker(self):
        dropped = []
        state = {}
        with patch.object(w.job_queue, "drop_in_inbox", lambda params, who: dropped.append(params)), \
                patch.object(w, "MAX_FULL", 2):
            rec = {"date": "2026-11-21", "needs_full_analysis": True, "headline": "h", "issues": []}
            self.assertIn("맡김", w.maybe_queue_full(_item(ticker="A"), rec, state, True))
            self.assertIn("이미", w.maybe_queue_full(_item(ticker="A"), rec, state, True))
            self.assertIn("맡김", w.maybe_queue_full(_item(ticker="B"), rec, state, True))
            self.assertIn("한도", w.maybe_queue_full(_item(ticker="C"), rec, state, True))
            self.assertIn("꺼져", w.maybe_queue_full(_item(ticker="D", auto_full=False), rec, state, True))
            self.assertEqual(w.maybe_queue_full(_item(), dict(rec, needs_full_analysis=False), state, True), "")
        self.assertEqual([p["ticker"] for p in dropped], ["A", "B"])
        self.assertEqual(dropped[0]["provider"], "claude-code")
        job_queue.clean_inbox_params(dropped[0])  # what we drop is acceptable to the queue

    def test_run_all_briefs_in_parallel_and_isolates_failures(self):
        import threading
        import time as _time

        w.save_watchlist([_item(ticker=t) for t in ("A", "B", "C", "D")])
        live, peak, lock = [0], [0], threading.Lock()

        def fake_brief(item, now=None, llm=None):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            _time.sleep(0.3)
            with lock:
                live[0] -= 1
            if item["ticker"] == "B":
                raise RuntimeError("cli failed")
            return {"ticker": item["ticker"], "date": "2026-11-21", "importance": "높음", "headline": "h",
                    "issues": [], "needs_full_analysis": True}

        dropped = []
        with patch.object(w, "write_briefing", fake_brief), patch.object(w, "CONCURRENCY", 3), \
                patch.object(w, "MAX_FULL", 2), \
                patch.object(w.job_queue, "drop_in_inbox", lambda params, who: dropped.append(params["ticker"])):
            t0 = _time.time()
            n = w.run_all()
            took = _time.time() - t0
        self.assertEqual(n, 3)  # B failed, the others still finished
        self.assertEqual(peak[0], 3)  # never more than the limit at once
        self.assertLess(took, 0.3 * 4 * 0.75)  # faster than one by one
        self.assertEqual(dropped, ["A", "C"])  # watchlist order decides who gets the daily cap

    def test_due_once_per_day_after_run_time(self):
        with patch.object(w, "RUN_AT", "07:00"):
            self.assertFalse(w._due(datetime(2026, 11, 21, 6, 59), {}))
            self.assertTrue(w._due(datetime(2026, 11, 21, 7, 0), {}))
            self.assertTrue(w._due(datetime(2026, 11, 21, 9, 30), {"last_run": "2026-11-20"}))  # catch up
            self.assertFalse(w._due(datetime(2026, 11, 21, 9, 30), {"last_run": "2026-11-21"}))


@pytest.mark.unit
class TestQueueInbox(unittest.TestCase):
    def _params(self, **kw):
        return dict(w.FULL_ANALYSIS, ticker="NVDA", trade_date="2026-11-21", notes="n",
                    position={"held": False, "avg_price": None}, **kw)

    def test_inbox_jobs_are_queued_and_bad_ones_set_aside(self):
        with tempfile.TemporaryDirectory() as tmp:
            inbox = Path(tmp)
            job_queue.drop_in_inbox(self._params(), "자동 점검", inbox=inbox)
            (inbox / "zz_bad.json").write_text(json.dumps({"params": self._params(provider="openai")}),
                                               encoding="utf-8")
            q = job_queue.JobQueue(lambda job: None, path=None, autostart=False, inbox=inbox)
            with q._cond:
                q._drain_inbox()
            _, queued, _ = q.snapshot()
            self.assertEqual([j.params["ticker"] for j in queued], ["NVDA"])
            self.assertEqual(queued[0].requester, "자동 점검")
            self.assertEqual(queued[0].config["deep_think_llm"], "opus")
            self.assertEqual(queued[0].config["output_language"], "Korean")
            self.assertEqual(sorted(p.suffix for p in inbox.iterdir()), [".bad"])

    def test_clean_params_rejects_unexpected_values(self):
        for bad in (dict(ticker="삼성"), dict(provider="openai"), dict(deep_model="gpt"),
                    dict(analysts=["x"]), dict(debate_rounds=9), dict(trade_date="tomorrow")):
            with self.assertRaises(Exception):
                job_queue.clean_inbox_params(dict(self._params(), **bad))
        with self.assertRaises(ValueError):
            job_queue.drop_in_inbox(self._params(provider="openai"), "x", inbox=Path("unused"))


if __name__ == "__main__":
    unittest.main()
