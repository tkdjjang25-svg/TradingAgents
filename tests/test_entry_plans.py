"""Entry/exit price plans: extraction, daily-close checks, watchlist trigger and the chart."""

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import entry_plans as ep  # noqa: E402
import price_chart  # noqa: E402
import watchlist as w  # noqa: E402


class _FakeLLM:
    """with_structured_output(schema).invoke(messages) -> schema(**fields)"""

    def __init__(self, fields):
        self.fields, self.prompt, self.calls = fields, "", 0

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.calls += 1
        self.prompt = messages[-1].content
        return self.schema(**self.fields)


NVDA_PLAN = {
    "summary": "지금은 매수 보류 — 178.9 종가 돌파 또는 158.7~162.6 지지 확인 시 소량 진입",
    "triggers": [
        {"role": "entry", "kind": "close_above", "price": 178.9, "label": "B안: 178.9 종가 돌파 시 진입"},
        {"role": "entry", "kind": "close_between", "low": 162.6, "high": 158.7, "days": 2,
         "label": "A안: 158.7~162.6 이틀 지지", "other_conditions": "거래량 감소"},
        {"role": "stop", "kind": "close_below", "price": 171.5, "label": "B안 손절 171.5"},
        {"role": "cancel", "kind": "close_below", "price": 124.5, "label": "124.5 이탈 시 계획 폐기"},
        {"role": "entry", "kind": "close_above", "label": "가격 없는 조건"},
        {"role": "moon", "kind": "close_above", "price": 1, "label": "알 수 없는 역할"},
    ],
    "checks": ["11/12 8-K 원문 확인"],
}


def _record(run_id="r1", ticker="NVDA", saved_at="2026-11-24T07:30:00", held=False):
    return {
        "id": run_id, "ticker": ticker, "trade_date": "2026-11-24", "saved_at": saved_at, "decision": "Hold",
        "position": {"held": held, "avg_price": None},
        "state": {"final_trade_decision": "**Rating**: Hold\n165에서는 사지 않는다. 178.9 돌파 시 진입.",
                  "trader_investment_plan": "wait"},
    }


@pytest.mark.unit
class TestExtract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name) / "plans"
        self.hist = Path(self.tmp.name) / "history"
        self.hist.mkdir()

    def _save_run(self, rec):
        (self.hist / f"{rec['id']}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")

    def test_extract_keeps_only_watchable_triggers(self):
        llm = _FakeLLM(NVDA_PLAN)
        plan = ep.extract(_record(), llm=llm, base=self.base)
        self.assertIn("178.9 돌파", llm.prompt)
        self.assertIn("미보유", llm.prompt)
        self.assertEqual(len(plan["triggers"]), 4)  # no-price and unknown-role triggers dropped
        zone = plan["triggers"][1]
        self.assertEqual((zone["low"], zone["high"], zone["days"]), (158.7, 162.6, 2))  # swapped into order
        self.assertEqual(ep.load_plan("r1", self.base)["summary"], NVDA_PLAN["summary"])

    def test_backfill_newest_first_and_failures_are_not_retried_at_once(self):
        self._save_run(_record("old", saved_at="2026-10-01T09:00:00"))  # older than 30 days
        self._save_run(_record("a", saved_at="2026-11-23T09:00:00"))
        self._save_run(_record("b", saved_at="2026-11-24T09:00:00"))
        self._save_run(dict(_record("warn", saved_at="2026-11-24T10:00:00"), warning="bad"))
        llm = _FakeLLM(NVDA_PLAN)
        now = datetime(2026, 11, 24, 12)
        self.assertEqual(ep.backfill(limit=1, llm=llm, base=self.base, history_dir=self.hist, now=now), 1)
        self.assertIsNotNone(ep.load_plan("b", self.base))
        self.assertIsNone(ep.load_plan("a", self.base))
        ep.backfill(limit=5, llm=llm, base=self.base, history_dir=self.hist, now=now)
        self.assertIsNotNone(ep.load_plan("a", self.base))
        self.assertIsNone(ep.load_plan("old", self.base))
        self.assertIsNone(ep.load_plan("warn", self.base))

        class Broken:
            def with_structured_output(self, schema):
                raise RuntimeError("cli down")

        self._save_run(_record("c", saved_at="2026-11-24T11:00:00"))
        self.assertIsNone(ep.ensure_plan("c", llm=Broken(), base=self.base, history_dir=self.hist))
        llm.calls = 0
        self.assertIsNone(ep.ensure_plan("c", llm=llm, base=self.base, history_dir=self.hist))
        self.assertEqual(llm.calls, 0)  # waits RETRY_AFTER before trying again


@pytest.mark.unit
class TestEvaluate(unittest.TestCase):
    plan = {"run_id": "r1", "trade_date": "2026-11-24", "decision": "Hold", "summary": "s",
            "triggers": ep.sanitize(NVDA_PLAN["triggers"])}

    def test_cutoff_is_first_session_the_analysis_did_not_see(self):
        # saved 07:30 KST 11-24 = 17:30 New York 11-23 -> first unseen US session is 11-24
        self.assertEqual(ep.first_unseen_session("NVDA", "2026-11-24T07:30:00+09:00"), "2026-11-24")
        # Korean stock analysed after the 15:30 close -> next day
        self.assertEqual(ep.first_unseen_session("005930.KS", "2026-11-24T20:00:00+09:00"), "2026-11-25")
        self.assertEqual(ep.first_unseen_session("005930.KS", "2026-11-24T10:00:00+09:00"), "2026-11-24")

    def test_new_breakout_fires_entry(self):
        closes = [("2026-11-20", 125.5), ("2026-11-23", 165.05), ("2026-11-24", 180.2)]
        r = ep.evaluate(self.plan, closes, held=False, cutoff="2026-11-24")
        self.assertEqual(r["status"], "entry")
        self.assertEqual(r["fired"], ["B안: 178.9 종가 돌파 시 진입"])
        self.assertTrue(ep.is_triggered(r))

    def test_no_new_close_or_already_met_does_not_fire(self):
        closes = [("2026-11-20", 125.5), ("2026-11-23", 180.0)]
        r = ep.evaluate(self.plan, closes, held=False, cutoff="2026-11-24")  # nothing after the analysis
        self.assertEqual(r["status"], "waiting")
        # already above 178.9 at the close the analysis saw -> not a new signal
        r = ep.evaluate(self.plan, [("2026-11-20", 180.0), ("2026-11-24", 181.0)], held=False, cutoff="2026-11-24")
        self.assertEqual(r["status"], "waiting")
        self.assertTrue(r["triggers"][0]["met"])
        self.assertFalse(r["triggers"][0]["fresh"])

    def test_zone_needs_consecutive_closes_and_nearest_is_reported(self):
        one = [("2026-11-23", 165.0), ("2026-11-24", 160.0)]
        r = ep.evaluate(self.plan, one, held=False, cutoff="2026-11-24")
        self.assertEqual(r["status"], "waiting")
        self.assertIn("A안", r["nearest"])  # inside the zone (0%) but needs a second close
        two = one + [("2026-11-25", 161.0)]
        self.assertEqual(ep.evaluate(self.plan, two, held=False, cutoff="2026-11-24")["status"], "entry")

    def test_stop_only_matters_when_held_and_cancel_wins(self):
        closes = [("2026-11-23", 175.0), ("2026-11-24", 170.0)]
        self.assertEqual(ep.evaluate(self.plan, closes, held=False, cutoff="2026-11-24")["status"], "waiting")
        self.assertEqual(ep.evaluate(self.plan, closes, held=True, cutoff="2026-11-24")["status"], "stop")
        crash = [("2026-11-23", 165.0), ("2026-11-24", 120.0)]
        self.assertEqual(ep.evaluate(self.plan, crash, held=True, cutoff="2026-11-24")["status"], "cancel")

    def test_prompt_section(self):
        r = ep.evaluate(self.plan, [("2026-11-24", 170.0)], held=False, cutoff="2026-11-24")
        text = ep.prompt_section(r)
        self.assertIn("price plan", text)
        self.assertIn("B안", text)
        self.assertEqual(ep.prompt_section(None), "")


@pytest.mark.unit
class TestWatchlistTrigger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_met_condition_queues_full_analysis_with_note(self):
        dropped = []
        entry = {"status": "entry", "status_ko": ep.STATUS_KO["entry"], "fired": ["B안: 178.9 종가 돌파 시 진입"],
                 "trade_date": "2026-11-24", "decision": "Hold", "last_close": 180.2, "last_date": "2026-11-24"}
        rec = {"date": "2026-11-25", "needs_full_analysis": False, "headline": "h", "issues": [], "entry_watch": entry}
        item = {"ticker": "NVDA", "held": False, "avg_price": None, "auto_full": True}
        with patch.object(w.job_queue, "drop_in_inbox", lambda params, who: dropped.append(params)), \
                patch.object(w, "MAX_FULL", 2):
            note = w.maybe_queue_full(item, rec, {}, True)
            self.assertIn("진입 조건 충족", note)
            waiting = dict(rec, entry_watch=dict(entry, status="waiting", fired=[]))
            self.assertEqual(w.maybe_queue_full(dict(item, ticker="AMD"), waiting, {}, True), "")
        self.assertIn("178.9", dropped[0]["notes"])

    def test_watch_uses_latest_run_and_saves_last_check(self):
        hist, base = Path(self.tmp.name) / "h", Path(self.tmp.name) / "p"
        hist.mkdir()
        for rec in (_record("old", saved_at="2026-11-20T09:00:00"), _record("new", saved_at="2026-11-24T07:30:00")):
            (hist / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        llm = _FakeLLM(NVDA_PLAN)
        r = ep.watch({"ticker": "NVDA", "held": False}, [("2026-11-23", 165.0), ("2026-11-24", 180.0)],
                     llm=llm, base=base, history_dir=hist)
        self.assertEqual((r["run_id"], r["status"]), ("new", "entry"))
        self.assertEqual(ep.load_plan("new", base)["last_check"]["status"], "entry")
        self.assertIsNone(ep.watch({"ticker": "AMD"}, [], llm=llm, base=base, history_dir=hist))


@pytest.mark.unit
class TestChart(unittest.TestCase):
    def _history(self, n=260):
        idx = pd.bdate_range("2026-05-01", periods=n)
        close = pd.Series(range(100, 100 + n), index=idx, dtype=float)
        return pd.DataFrame({"Open": close - 1, "High": close + 2, "Low": close - 3, "Close": close,
                             "Volume": 1000.0}, index=idx)

    def test_prepare_and_build(self):
        df = price_chart.prepare(self._history(), 126)
        self.assertEqual(len(df), 126)
        self.assertTrue(df["MA120"].notna().all())  # averages use the longer history
        self.assertTrue(df["up"].all())
        plan = {"triggers": ep.sanitize(NVDA_PLAN["triggers"])}
        levels = price_chart.levels_from_plan(plan, avg_price=230.0)
        self.assertEqual(len(levels), 5)
        spec = price_chart.build_chart(df, levels).to_dict()
        self.assertEqual(len(spec["vconcat"]), 2)
        far = price_chart.hidden_levels(levels, df)  # last close 359: 124.5 and the zone are > 35% away
        self.assertIn("124.5 이탈 시 계획 폐기", [lv["label"] for lv in far])
        self.assertTrue(price_chart.prepare(pd.DataFrame(), 10).empty)

    def test_short_labels_merge_and_spread(self):
        plan = {"triggers": [
            {"role": "entry", "kind": "close_above", "price": 282.0, "label": "A안: 272.66 회복 후 282 위 종가 마감 시 1/3 진입"},
            {"role": "entry", "kind": "close_above", "price": 280.5, "label": "경로 B: 지지 확인 후 진입",
             "other_conditions": "RSI 30 위"},
            {"role": "cancel", "kind": "close_below", "price": 258.0, "label": "258 아래 종가 + 거래량 2배면 진입 보류"},
            {"role": "stop", "kind": "close_below", "price": 258.0, "label": "진입 후 종가 258 이탈 시 손절"},
        ]}
        levels = price_chart.levels_from_plan(plan, avg_price=1234.5)
        self.assertEqual([lv["short"] for lv in levels],
                         ["A안 진입 282", "B안 진입 280.5", "계획 폐기 258", "손절 258", "내 평단 1,234"])
        rows = price_chart._label_rows(levels[:4], 250.0, 300.0, 340)
        self.assertEqual(list(rows["text"]), ["계획 폐기 · 손절 258", "B안 진입 280.5", "A안 진입 282"])
        gap = 50.0 * price_chart.LABEL_PX / 340
        self.assertTrue(all(b - a >= gap - 1e-9 for a, b in zip(rows["ly"], rows["ly"][1:])))  # no overlap
        self.assertLessEqual(rows["ly"].max(), 300.0 - gap + 1e-9)  # top label not clipped
        legend = price_chart.legend_rows(levels)
        self.assertEqual(legend[0]["차트 이름"], "내 평단 1,234")
        self.assertEqual(legend[2]["함께 볼 것"], "RSI 30 위")
