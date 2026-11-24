"""When an event's D0 / D+20 documents are due, and the pipeline status shown in the app."""

import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import news_event_agent as n  # noqa: E402


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


@pytest.mark.unit
class TestReportTiming(unittest.TestCase):
    def test_us_news_after_friday_close_waits_for_monday(self):
        # Fri 2026-11-20 21:53 UTC = 16:53 New York, after the close -> Monday 11-23 16:00 NY
        self.assertEqual(n.first_close_after("NVDA", _utc(2026, 11, 20, 21, 53)), _utc(2026, 11, 23, 21, 0))
        # Sunday news -> Monday close
        self.assertEqual(n.first_close_after("AMZN", _utc(2026, 11, 22, 13, 3)), _utc(2026, 11, 23, 21, 0))

    def test_us_news_during_session_counts_that_day(self):
        self.assertEqual(n.first_close_after("SPY", _utc(2026, 11, 10, 16, 37)), _utc(2026, 11, 10, 21, 0))

    def test_korean_close_is_1530_kst(self):
        # 14:00 KST on a Monday -> that day's 15:30 KST close (06:30 UTC)
        self.assertEqual(n.first_close_after("005930.KS", _utc(2026, 11, 23, 5, 0)), _utc(2026, 11, 23, 6, 30))
        # 16:00 KST -> next day
        self.assertEqual(n.first_close_after("005930.KS", _utc(2026, 11, 23, 7, 0)), _utc(2026, 11, 24, 6, 30))

    def test_crypto_trades_on_weekends(self):
        self.assertEqual(n.first_close_after("BTC-USD", _utc(2026, 11, 21, 10, 0)), _utc(2026, 11, 21, 23, 59))

    def test_twenty_sessions_skip_weekends(self):
        d0 = _utc(2026, 11, 23, 21, 0)  # Monday close
        self.assertEqual(n.add_sessions(d0, "NVDA", 20), _utc(2026, 12, 21, 21, 0))  # 4 weeks later, Monday
        self.assertEqual(n.add_sessions(d0, "NVDA", 5), _utc(2026, 11, 30, 21, 0))


@pytest.mark.unit
class TestPipelineStatus(unittest.TestCase):
    def test_states_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "m.db"
            conn = sqlite3.connect(db)
            conn.executescript("""
                CREATE TABLE messages (chat_id, msg_id, date, text);
                CREATE TABLE runs (started_at, finished_at, chat_id, new, updated, error);
                CREATE TABLE agent_meta (key, value);
                CREATE TABLE news_classified (chat_id, msg_id, classified_at, n_events);
                CREATE TABLE news_events (chat_id, msg_id, ticker, event_type, headline, news_date);
                CREATE TABLE news_reports (chat_id, msg_id, ticker, stage, path, summary, model, created_at, error);
            """)
            conn.executemany("INSERT INTO messages VALUES (1, ?, ?, 't')", [
                (1, "2026-11-20T21:53:00+00:00"), (2, "2026-11-22T13:03:00+00:00"), (3, "2026-11-10T22:00:00+00:00"),
                (4, "2026-11-21T05:00:00+00:00"), (5, "2026-11-22T13:20:00+00:00"),
            ])
            conn.execute("INSERT INTO runs VALUES ('2026-11-22T13:30:00+00:00', '2026-11-22T13:30:02+00:00', 1, 0, 0, NULL)")
            conn.execute("INSERT INTO agent_meta VALUES ('since', '2026-11-20T02:00:00+00:00')")
            conn.executemany("INSERT INTO news_classified VALUES (1, ?, '2026-11-22T13:10:00+00:00', 1)",
                             [(1,), (2,), (4,)])
            conn.executemany("INSERT INTO news_events VALUES (1, ?, ?, 'x', ?, ?)", [
                (1, "NVDA", "SEC 조사", "2026-11-20T21:53:00+00:00"),
                (2, "AMZN", "인수", "2026-11-22T13:03:00+00:00"),
                (3, "COST", "실적", "2026-11-10T22:00:00+00:00"),
                (4, "ORCL", "계약", "2026-11-21T05:00:00+00:00"),
            ])
            conn.execute("INSERT INTO news_reports VALUES (1, 4, 'ORCL', 'D0', 'p', 's', 'opus', '2026-11-24', NULL)")
            conn.commit()
            conn.close()

            s = n.pipeline_status(db, now=_utc(2026, 11, 23, 21, 40))  # Monday, 40 min after the NY close
            before_close = n.pipeline_status(db, now=_utc(2026, 11, 23, 15, 0))
        self.assertEqual(s["unclassified"], 1)  # message 5 (message 3 is before 'since')
        self.assertEqual(s["last_message"], "2026-11-22T13:20:00+00:00")
        states = {e["ticker"]: e for e in s["events"]}
        self.assertIn("1차 작성 차례", states["NVDA"]["state"])
        self.assertIn("1차 작성 차례", states["AMZN"]["state"])
        self.assertIn("자동 대상 아님", states["COST"]["state"])
        self.assertFalse(states["COST"]["auto"])
        self.assertIn("1차 완료", states["ORCL"]["state"])
        self.assertIsNone(states["ORCL"]["d0_eta"])
        self.assertIn("장 마감 대기", {e["ticker"]: e for e in before_close["events"]}["NVDA"]["state"])


if __name__ == "__main__":
    unittest.main()
