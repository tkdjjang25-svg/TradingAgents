import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import pytest

# history_store는 app.py 옆(프로젝트 루트)에 있는 앱 전용 모듈이다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import history_store  # noqa: E402

FINAL_STATE = {
    "company_of_interest": "NVDA",
    "trade_date": "2026-11-13",
    "market_report": "market",
    "sentiment_report": "",
    "news_report": "news",
    "fundamentals_report": "",
    "investment_debate_state": {"bull_history": "bull", "bear_history": "bear", "judge_decision": "judge"},
    "investment_plan": "plan",
    "trader_investment_plan": "trader",
    "risk_debate_state": {
        "aggressive_history": "aggr",
        "conservative_history": "cons",
        "neutral_history": "neut",
        "judge_decision": "pm",
    },
    "final_trade_decision": "**Rating**: Hold\n\nreasons",
}


def _engine_log(results_dir: Path, ticker: str, trade_date: str, age_sec: float = 3600) -> Path:
    """엔진의 _log_state()와 같은 형식의 로그를 만든다 (trader 키 이름이 다름)."""
    log = results_dir / ticker / "TradingAgentsStrategy_logs" / f"full_states_log_{trade_date}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    data = dict(FINAL_STATE, company_of_interest=ticker, trade_date=trade_date)
    data["trader_investment_decision"] = data.pop("trader_investment_plan")
    log.write_text(json.dumps(data), encoding="utf-8")
    old = time.time() - age_sec
    os.utime(log, (old, old))
    return log


@pytest.mark.unit
class TestHistoryStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.history = root / "history"
        self.results = root / "logs"

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_load_roundtrip(self):
        meta = {"provider": "claude-code", "deep_model": "opus", "quick_model": "sonnet",
                "analysts": ["market", "news"], "elapsed_sec": 672, "llm_calls": 10}
        path = history_store.save_run(FINAL_STATE, "Hold", meta, history_dir=self.history)

        runs = history_store.list_runs(self.history)
        self.assertEqual(len(runs), 1)
        self.assertNotIn("state", runs[0])
        self.assertEqual(runs[0]["source"], "app")
        self.assertEqual(runs[0]["provider"], "claude-code")

        record = history_store.load_run(path.stem, self.history)
        self.assertEqual(record["decision"], "Hold")
        self.assertEqual(record["state"]["risk_debate_state"]["aggressive_history"], "aggr")

    def test_same_second_saves_do_not_overwrite(self):
        a = history_store.save_run(FINAL_STATE, "Hold", {}, history_dir=self.history)
        b = history_store.save_run(FINAL_STATE, "Hold", {}, history_dir=self.history)
        self.assertNotEqual(a, b)
        self.assertEqual(len(history_store.list_runs(self.history)), 2)

    def test_import_engine_logs_normalizes_and_parses_rating(self):
        _engine_log(self.results, "SCO", "2026-05-06")
        self.assertEqual(history_store.import_engine_logs(self.results, self.history), 1)

        run = history_store.list_runs(self.history)[0]
        self.assertEqual((run["ticker"], run["trade_date"], run["source"]), ("SCO", "2026-05-06", "engine-log"))
        self.assertEqual(run["decision"], "Hold")
        self.assertEqual(run["analysts"], ["market", "news"])
        record = history_store.load_run(run["id"], self.history)
        self.assertEqual(record["state"]["trader_investment_plan"], "trader")

    def test_import_is_idempotent_but_picks_up_overwritten_log(self):
        log = _engine_log(self.results, "NVDA", "2026-11-11")
        history_store.import_engine_logs(self.results, self.history)
        self.assertEqual(history_store.import_engine_logs(self.results, self.history), 0)

        # 같은 종목·기준일 재분석으로 엔진 로그가 덮어써지면 새 건으로 보관한다.
        newer = time.time() - 60
        os.utime(log, (newer, newer))
        self.assertEqual(history_store.import_engine_logs(self.results, self.history), 1)
        self.assertEqual(len(history_store.list_runs(self.history)), 2)

    def test_app_saved_run_is_not_imported_again(self):
        log = _engine_log(self.results, "NVDA", "2026-11-13")
        history_store.save_run(FINAL_STATE, "Hold", {}, engine_log=log, history_dir=self.history)
        self.assertEqual(history_store.import_engine_logs(self.results, self.history), 0)

    def test_fresh_engine_log_waits_for_app_to_claim_it(self):
        _engine_log(self.results, "NVDA", "2026-11-13", age_sec=1)
        self.assertEqual(history_store.import_engine_logs(self.results, self.history), 0)

    def test_markdown_export_contains_sections(self):
        path = history_store.save_run(FINAL_STATE, "Hold", {"analysts": ["market"]}, history_dir=self.history)
        md = history_store.to_markdown(history_store.load_run(path.stem, self.history))
        self.assertTrue(md.startswith("# NVDA · 기준일 2026-11-13"))
        for heading in ("## 최종 거래 의사결정", "## 시장 분석", "## 리스크 토론 — 공격적"):
            self.assertIn(heading, md)
        self.assertNotIn("## 소셜 분석", md)  # 빈 리포트는 생략


if __name__ == "__main__":
    unittest.main()


@pytest.mark.unit
class TestDecisionKo(unittest.TestCase):
    def test_five_tier_ratings_are_translated(self):
        from tradingagents.agents.utils.rating import RATINGS_5_TIER

        self.assertEqual(set(history_store.DECISION_KO), set(RATINGS_5_TIER))
        self.assertEqual(history_store.decision_ko("Underweight"), "비중 축소 (Underweight)")
        self.assertEqual(history_store.decision_ko("buy"), "매수 (buy)")
        self.assertEqual(history_store.decision_ko("Strong Buy"), "Strong Buy")
        self.assertEqual(history_store.decision_ko(""), "")
