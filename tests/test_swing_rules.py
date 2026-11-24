"""Swing rules: sizing, signals, regime, 10-point checklist, journal/slump protocol, backtest, daily screen."""

import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import swing_backtest as bt  # noqa: E402
import swing_rules as sr  # noqa: E402
import watchlist as w  # noqa: E402


def _frame(closes, vols=None, spread=1.0):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2025-01-01", periods=len(closes))
    vols = np.full(len(closes), 1_000_000.0) if vols is None else np.asarray(vols, dtype=float)
    return pd.DataFrame({"Open": closes - 0.5, "High": closes + spread, "Low": closes - spread,
                         "Close": closes, "Volume": vols}, index=idx)


def _breakout(after=()):
    """280일 상승 → 20일 박스 → 거래량 3배 돌파 → after."""
    closes = list(np.linspace(100, 200, 280)) + [200.0] * 20 + [206.0] + list(after)
    vols = [100_000_000.0] * 300 + [300_000_000.0] + [100_000_000.0] * len(after)
    return _frame(closes, vols)


ACCOUNT = {"open_risk": 0.0, "open_trades": 0, "blocked": False, "stage": 0, "mult": 1.0, "reasons": []}
UP = {"market": "KS", "name": "코스피", "regime": "up", **sr.REGIME["up"], "close": 1, "ma_slow": 1, "slow": 60}


@pytest.mark.unit
class TestPlan(unittest.TestCase):
    def test_one_percent_rule_and_caps(self):
        # 매뉴얼 예시: 3,000만 원, 10,000원 매수, 9,300원 손절 → 428주, 비중 14.3%
        p = sr.plan_trade(10_000, 9_300, equity=30_000_000)
        self.assertEqual(p["qty"], 428)
        self.assertAlmostEqual(p["weight"], 14.27, places=2)
        self.assertEqual((p["t1"], p["t2"]), (11_400, 12_800))
        # 손절이 가까우면 수량이 커지지만 1종목 20% 상한에 걸린다
        p = sr.plan_trade(10_000, 9_700, equity=30_000_000)
        self.assertTrue(p["capped"])
        self.assertEqual(p["qty"], 600)
        # 차트상 자리가 −8% 보다 멀면 −7% 로 두고 '패스' 표시
        p = sr.plan_trade(10_000, 9_000)
        self.assertEqual((round(p["stop"]), p["too_far"], p["qty"]), (9_300, True, None))


@pytest.mark.unit
class TestSignals(unittest.TestCase):
    def test_breakout_needs_volume_and_no_chase(self):
        df = sr.indicators(_breakout())
        row = df.iloc[-1]
        self.assertTrue(row["TREND"])
        self.assertEqual(sr.setup_of(row), "A")
        self.assertAlmostEqual(sr.structural_stop(row, "A"), 199 * 0.99)
        quiet = _breakout()
        quiet.iloc[-1, quiet.columns.get_loc("Volume")] = 1_000_000
        self.assertIsNone(sr.setup_of(sr.indicators(quiet).iloc[-1]))
        chased = _breakout()
        chased.iloc[-1, chased.columns.get_loc("Close")] = 230.0  # 기준가 +14%
        self.assertNotEqual(sr.setup_of(sr.indicators(chased).iloc[-1]), "A")

    def test_regime(self):
        up = sr.regime_series(_frame(np.linspace(100, 200, 150)), "KS")
        down = sr.regime_series(_frame(np.linspace(200, 100, 150)), "KS")
        self.assertEqual((up.iloc[-1], down.iloc[-1]), ("up", "down"))
        r = sr.market_regime("KS", hist=_frame(np.linspace(200, 100, 150)))
        self.assertEqual((r["regime"], r["risk_factor"]), ("down", 0.0))


@pytest.mark.unit
class TestDataParsing(unittest.TestCase):
    def test_flows_and_filings(self):
        rows = [{"foreignerPureBuyQuant": "+1,000", "organPureBuyQuant": "-200", "bizdate": "20261123"}] * 3 + \
               [{"foreignerPureBuyQuant": "-5,000", "organPureBuyQuant": "+100"}] * 17
        f = sr.parse_flows(rows)
        self.assertEqual((f["buy_days5"], f["sum20"]), (3, 3 * 800 - 17 * 4900))
        c = sr.classify_filings(["주요사항보고서(전환사채권발행결정)", "[기재정정]주요사항보고서(유상증자결정)",
                                 "관리종목지정", "관리종목지정해제", "주요사항보고서(전환사채권발행결정)(철회)"])
        self.assertEqual(c["dilution"], 2)
        self.assertEqual(c["danger"], ["관리종목지정"])


@pytest.mark.unit
class TestChecklist(unittest.TestCase):
    def _items(self, **kw):
        df = sr.indicators(_breakout())
        plan = sr.plan_trade(206.0, sr.structural_stop(df.iloc[-1], "A"), equity=10_000_000)
        args = dict(regime=UP, fund={"etf": False, "mcap": 1e13, "op4": 1e10, "op_last2": True,
                                     "debt_ratio": 50, "eps_yoy": 30},
                    flows={"sum20": 1e6, "buy_days5": 4}, filings={"dilution": 0, "danger": []},
                    account=ACCOUNT, held=None)
        args.update(kw)
        return {i["no"]: i for i in sr.checklist("123456.KS", df, plan=plan, setup="A", **args)}

    def test_clean_breakout_passes(self):
        items = self._items()
        self.assertEqual([n for n, i in items.items() if i["status"] != "pass"], [])
        self.assertEqual(sr.verdict(list(items.values())), "✅ 매수 후보")

    def test_failures(self):
        items = self._items(regime={**UP, "regime": "down", **sr.REGIME["down"]},
                            held={"held": True, "avg_price": 300.0},
                            filings={"dilution": 3, "danger": []}, flows={"sum20": -1, "buy_days5": 1})
        self.assertEqual(items[1]["status"], "fail")
        self.assertIn("CB·BW·유증 3회", items[3]["detail"])
        self.assertEqual(items[4]["status"], "fail")
        self.assertIn("물타기", items[10]["detail"])
        etf = self._items(fund={"etf": True})
        self.assertEqual(etf[3]["status"], "na")
        full = self._items(account={**ACCOUNT, "open_risk": 4.5})
        self.assertEqual(full[10]["status"], "fail")


@pytest.mark.unit
class TestJournalAndSlump(unittest.TestCase):
    def test_slump_stages(self):
        t = sr.SlumpTracker()
        for d in range(1, 4):
            t.add(-1.0, date(2026, 10, d))
        self.assertEqual((t.status(date(2026, 10, 5))["stage"], t.status(date(2026, 10, 5))["mult"]), (1, 0.5))
        for d in range(4, 6):
            t.add(-1.0, date(2026, 10, d))
        st = t.status(date(2026, 10, 6))
        self.assertEqual(st["stage"], 2)
        self.assertTrue(st["blocked"])  # 1주 중단
        self.assertFalse(t.status(date(2026, 10, 13))["blocked"])
        t.add(1.0, date(2026, 11, 1))
        t.add(1.0, date(2026, 11, 2))
        self.assertEqual(t.status(date(2026, 11, 3))["stage"], 1)  # 2연속 수익 → 한 단계 회복
        m = sr.SlumpTracker()
        m.add(-6.5, date(2026, 11, 2))
        self.assertTrue(m.status(date(2026, 11, 20))["blocked"])  # 월 −6% → 그달 종료
        self.assertFalse(m.status(date(2026, 12, 1))["blocked"] and m.stage < 2)

    def test_journal_open_risk_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.json"
            t = sr.add_trade("123456.KS", 10_000, 9_500, 600, 30_000_000, path=path)
            st = sr.journal_status(sr.load_journal(path))
            self.assertAlmostEqual(st["open_risk"], 600 * 500 / 30_000_000 * 100)
            sr.add_exit(t["id"], 11_000, 200, "2026-11-20", path=path)
            sr.update_stop(t["id"], 10_050, path=path)
            self.assertEqual(sr.journal_status(sr.load_journal(path))["open_risk"], 0)  # 본전 위로 올림
            done = sr.add_exit(t["id"], 10_050, 1000, "2026-11-24", path=path)  # 남은 400주만 처리
            self.assertTrue(done["closed"])
            res = sr.trade_result(done)
            # (200주×1,000 + 400주×50 − 왕복 비용 0.25%×10,000×600) ÷ (500×600)
            self.assertAlmostEqual(res["r"], (200_000 + 20_000 - 15_000) / 300_000)


@pytest.mark.unit
class TestBacktest(unittest.TestCase):
    def test_breakout_scales_out_and_trails(self):
        rally = list(np.linspace(207, 260, 30)) + list(np.linspace(258, 220, 10))
        df = sr.indicators(_breakout(rally))
        regime = pd.Series("up", index=df.index)
        trades = bt.simulate_ticker("X.KS", df, regime, setups=("A",))
        first = trades[0]
        self.assertEqual(first.setup, "A")
        reasons = [e[3] for e in first.exits]
        self.assertEqual(reasons[:2], ["2R 익절", "4R 익절"])
        self.assertAlmostEqual(sum(e[2] for e in first.exits), 1.0)
        self.assertGreater(first.result(0.0025)["r_mult"], 2)

    def test_stop_and_down_regime(self):
        crash = [205.0, 190.0, 185.0, 180.0] + [180.0] * 5
        df = sr.indicators(_breakout(crash))
        t = bt.simulate_ticker("X.KS", df, pd.Series("up", index=df.index), setups=("A",))[0]
        self.assertEqual(t.exits[0][3], "손절")
        self.assertLess(t.result(0)["r_mult"], -0.9)
        self.assertEqual(bt.simulate_ticker("X.KS", df, pd.Series("down", index=df.index), setups=("A",)), [])

    def test_portfolio_limits_and_run(self):
        rally = list(np.linspace(207, 260, 30)) + list(np.linspace(258, 220, 10))
        frames = {f"T{i}.KS": _breakout(rally) for i in range(8)}
        frames["^KS11"] = _frame(np.linspace(100, 200, 341))

        def loader(t, start=None):
            return frames[t]

        res = bt.run(list(frames)[:-1], years=1, setups=("A",), loader=loader)
        self.assertGreater(res["signals"]["n"], 0)
        self.assertLessEqual(res["portfolio"]["n"], res["signals"]["n"])
        self.assertGreater(res["portfolio"]["skipped"]["동시 보유 한도"] + res["portfolio"]["skipped"]["오픈 리스크 한도"], 0)


@pytest.mark.unit
class TestDailyScreen(unittest.TestCase):
    def test_due_on_weekdays_after_close(self):
        with patch.object(w, "SCREEN_AT", "16:10"):
            self.assertFalse(w._screen_due(datetime(2026, 11, 24, 16, 0), {}))
            self.assertTrue(w._screen_due(datetime(2026, 11, 24, 16, 10), {}))
            self.assertFalse(w._screen_due(datetime(2026, 11, 24, 17, 0), {"last_screen": "2026-11-24"}))
            self.assertFalse(w._screen_due(datetime(2026, 11, 28, 17, 0), {}))  # 토요일

    def test_screen_runs_two_stages(self):
        frames = {"A.KS": _breakout(), "B.KS": _frame(np.linspace(200, 100, 300))}
        calls = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(sr, "SCREEN_FILE", Path(tmp) / "s.json"), \
                patch.object(sr, "JOURNAL_FILE", Path(tmp) / "j.json"), \
                patch.object(sr, "load_history", lambda t, **kw: frames.get(t, _frame(np.linspace(100, 200, 300)))), \
                patch.object(sr, "fundamentals", lambda t: calls.append(t) or {"etf": False}), \
                patch.object(sr, "investor_flows", lambda t: None), \
                patch.object(sr, "risk_filings", lambda t: None):
            out = sr.run_screen(["A.KS", "B.KS"], {"risk_pct": 1.0})
            self.assertEqual(calls, ["A.KS"])  # 신호가 난 종목만 심화 점검
            rows = {r["ticker"]: r for r in out["rows"]}
            self.assertEqual(rows["A.KS"]["setup"], "A")
            self.assertIsNone(rows["B.KS"]["setup"])
            self.assertEqual(sr.load_screen()["count"], 2)


if __name__ == "__main__":
    unittest.main()
