import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import news_event_agent as agent  # noqa: E402
import telegram_collector as tc  # noqa: E402

CHAT = -1001234567890


def _msg(msg_id, when, text="코스트코 특별배당 발표 " * 5):
    return SimpleNamespace(id=msg_id, date=when, message=text, sender=SimpleNamespace(title="채널"))


class FakeLLM:
    def __init__(self, batches=None, answer="요약: 테스트 요약\n\n본문"):
        self.batches, self.answer, self.prompts = list(batches or []), answer, []

    def with_structured_output(self, schema):
        llm = self

        class _Runner:
            def invoke(self, messages):
                llm.prompts.append(messages[-1].content)
                return schema.model_validate(llm.batches.pop(0))

        return _Runner()

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        return SimpleNamespace(content=self.answer)


def _reaction(sessions_after=1):
    return {"ticker": "COST", "d0": "2026-11-11", "sessions_after": sessions_after, "d0_complete": True,
            "table": "| t |", "summary": {"D0 종가 등락": "+2.01%"}, "last_date": "2026-11-12"}


@pytest.mark.unit
class TestLocateD0(unittest.TestCase):
    days = [date(2026, 11, 9), date(2026, 11, 10), date(2026, 11, 11)]

    def test_before_close_is_same_day_after_close_is_next_day(self):
        tz = timezone(timedelta(hours=-5))
        self.assertEqual(agent.locate_d0(self.days, datetime(2026, 11, 10, 9, 0, tzinfo=tz)), 1)
        self.assertEqual(agent.locate_d0(self.days, datetime(2026, 11, 10, 17, 21, tzinfo=tz)), 2)
        self.assertIsNone(agent.locate_d0(self.days, datetime(2026, 11, 11, 18, 0, tzinfo=tz)))

    def test_weekend_news_maps_to_next_session(self):
        self.assertEqual(agent.locate_d0(self.days, datetime(2026, 11, 8, 12, 0)), 0)

    def test_benchmarks(self):
        self.assertEqual(agent.benchmark_for("005930.KS"), "^KS11")
        self.assertEqual(agent.benchmark_for("NVDA"), "^GSPC")


@pytest.mark.unit
class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.db = Path(self.tmp.name) / "m.db"
        conn = tc.open_db(self.db)
        t0 = datetime(2026, 11, 10, 22, 18, tzinfo=timezone.utc)
        tc.upsert(conn, [tc.message_row(_msg(1, t0), CHAT), tc.message_row(_msg(2, t0 + timedelta(minutes=3)), CHAT),
                         tc.message_row(_msg(3, t0, text="ㅋㅋ"), CHAT)])
        conn.close()
        self.conn = agent.open_db(self.db)
        patcher = patch.object(agent, "REPORT_DIR", Path(self.tmp.name) / "reports")
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_classify_then_one_report_for_same_ticker_same_day(self):
        batch = {"items": [
            {"msg_id": 1, "events": [{"ticker": "cost", "event_type": "자사주/배당", "headline": "특별배당"}]},
            {"msg_id": 2, "events": [{"ticker": "COST", "event_type": "실적 발표", "headline": "사상 최대 실적"},
                                     {"ticker": "WMT", "event_type": "실적 발표", "headline": "extra"}]},
        ]}
        llm = FakeLLM([batch])
        with patch.object(agent, "_llm", return_value=llm):
            self.assertEqual(agent.classify_pending(self.conn, "2026-11-01"), 3)
        events = self.conn.execute("SELECT msg_id, ticker FROM news_events ORDER BY msg_id").fetchall()
        self.assertEqual([tuple(e) for e in events], [(1, "COST"), (2, "COST")])  # one ticker per post, upper-cased
        self.assertNotIn("ㅋㅋ", llm.prompts[0])  # chatter is not sent to the model

        with patch.object(agent, "_llm", return_value=llm), \
                patch.object(agent, "price_reaction", return_value=_reaction()), \
                patch.object(agent, "_company_line", return_value="Costco"), \
                patch.object(agent, "_benchmark_line", return_value="^GSPC: -"):
            self.assertEqual(agent.process_reports(self.conn, "2026-11-01"), 1)
            self.assertEqual(agent.process_reports(self.conn, "2026-11-01"), 0)  # D0 done, D20 not due yet
        report_prompt = llm.prompts[-1]
        self.assertIn("특별배당", report_prompt)
        self.assertIn("사상 최대 실적", report_prompt)
        rows = agent.read_reports(path=self.db)
        self.assertEqual({r["msg_id"] for r in rows}, {1, 2})
        self.assertEqual(len({r["path"] for r in rows}), 1)
        self.assertEqual(rows[0]["summary"], "테스트 요약")
        self.assertTrue(Path(rows[0]["path"]).read_text(encoding="utf-8").startswith("# COST"))

    def test_daily_limit_stops_automatic_reports(self):
        self.conn.execute("INSERT INTO news_events VALUES (?, 1, 'COST', 'x', 'y', '2026-11-10T22:18:00+00:00')", (CHAT,))
        self.conn.commit()
        with patch.object(agent, "DAILY_LIMIT", 0), patch.object(agent, "price_reaction") as pr:
            self.assertEqual(agent.process_reports(self.conn, "2026-11-01"), 0)
        pr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
