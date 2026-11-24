"""swing_rules 의 규칙(가격 조건만)을 과거 일봉에 돌려 보는 백테스트.

- 신호는 그날 종가로 판정하고, 다음 날 시가에 산다 (미래 데이터를 쓰지 않음).
- 돌파(A)는 다음 날 시가가 기준가 +5% 를 넘으면 사지 않는다(추격 금지).
- 손절가에 닿으면 손절가(갭 하락이면 시가)에 판다. 같은 날 고가·저가 순서를 모르므로 손절을 먼저 본다.
- 2R 에서 1/3, 4R 에서 1/3 익절. 2R 뒤 손절선은 본전, 4R 뒤에는 2R 가격.
- 1차 익절 전: 종가 20일선 이탈 → 다음 날 시가 정리, 10거래일 안에 +5% 못 가면 종가 정리.
- 1차 익절 후 남은 물량: 종가 10일선 이탈 또는 최고 종가 대비 −10% → 다음 날 시가 정리.
- 지수가 하락 국면이면 신규 진입하지 않는다. 역추세(C)는 상승 국면에서만.
- 포트폴리오: 1% 룰(국면·슬럼프 반영), 1종목 20%, 동시 6종목, 오픈 리스크 5%.

재무·수급·공시 필터는 과거 시점 값을 구할 수 없어 빠져 있다. 현재 상장 종목만 쓰므로 생존 편향이 있다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import swing_rules as sr

R = sr.RULES


@dataclass
class Trade:
    ticker: str
    setup: str
    signal_date: date
    entry_date: date
    entry: float
    stop0: float
    regime: str
    exits: List[tuple] = field(default_factory=list)  # (date, price, fraction, reason)

    @property
    def r(self) -> float:
        return self.entry - self.stop0

    @property
    def exit_date(self) -> date:
        return self.exits[-1][0]

    def result(self, cost: float) -> Dict[str, Any]:
        gross = sum(f * (p - self.entry) for _, p, f, _ in self.exits)
        net = gross - cost * self.entry
        return {"ret_pct": float(net / self.entry * 100), "r_mult": float(net / self.r) if self.r > 0 else 0.0,
                "days": (self.exit_date - self.entry_date).days}


def simulate_ticker(ticker: str, df: pd.DataFrame, regime: pd.Series, setups: Iterable[str] = ("A", "B"),
                    start: Optional[date] = None) -> List[Trade]:
    """한 종목에서 한 번에 한 포지션씩."""
    setups = set(setups)
    reg = regime.reindex(df.index, method="ffill").fillna("neutral")
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    ma_exit = df[f"MA{R['exit_ma']}"].to_numpy(float)
    ma_trail = df[f"MA{R['trail_ma']}"].to_numpy(float)
    pivot = df["PIVOT"].to_numpy(float)
    sig = {s: df[f"SIG_{s}"].fillna(False).to_numpy(bool) for s in ("A", "B", "C")}
    dates = [pd.Timestamp(d).date() for d in df.index]
    trades: List[Trade] = []
    i, n = 0, len(df)
    while i < n - 1:
        if (start and dates[i] < start) or reg.iloc[i] == "down":
            i += 1
            continue
        setup = next((s for s in ("A", "B") if s in setups and sig[s][i]), None)
        if setup is None and "C" in setups and sig["C"][i] and reg.iloc[i] == "up":
            setup = "C"
        if setup is None:
            i += 1
            continue
        j = i + 1
        entry = o[j]
        if setup == "A" and entry > pivot[i] * (1 + R["chase_pct"] / 100):
            i += 1
            continue
        plan = sr.plan_trade(entry, sr.structural_stop(df.iloc[i], setup))
        if plan["too_far"] or entry <= plan["stop"]:
            i += 1
            continue
        t = Trade(ticker, setup, dates[i], dates[j], float(entry), float(plan["stop"]), reg.iloc[i])
        stop, t1, t2 = plan["stop"], plan["t1"], plan["t2"]
        left, t1_done, t2_done, peak = 1.0, False, False, entry
        exit_next = None
        k = j
        while k < n and left > 1e-9:
            if exit_next and k > j:
                t.exits.append((dates[k], float(o[k]), left, exit_next))
                left = 0
                break
            if l[k] <= stop:
                t.exits.append((dates[k], float(min(o[k], stop)) if k > j else stop, left,
                                "손절" if not t1_done else "본전·이익 보전"))
                left = 0
                break
            if not t1_done and h[k] >= t1:
                t.exits.append((dates[k], float(max(o[k], t1)) if k > j else t1, 1 / 3, "2R 익절"))
                left -= 1 / 3
                t1_done, stop = True, max(stop, plan["breakeven"])
            if t1_done and not t2_done and h[k] >= t2:
                t.exits.append((dates[k], float(max(o[k], t2)) if k > j else t2, 1 / 3, "4R 익절"))
                left -= 1 / 3
                t2_done, stop = True, max(stop, t1)
            peak = max(peak, c[k])
            if not t1_done:
                if c[k] < ma_exit[k]:
                    exit_next = "20일선 이탈"
                elif k - j + 1 >= R["time_stop_days"] and peak / entry - 1 < R["time_stop_gain"] / 100:
                    t.exits.append((dates[k], float(c[k]), left, "시간 손절"))
                    left = 0
                    break
            elif c[k] < ma_trail[k] or c[k] < peak * (1 - R["trail_pct"] / 100):
                exit_next = "트레일링"
            k += 1
        if left > 1e-9:  # 데이터 끝: 마지막 종가로 평가
            t.exits.append((dates[-1], float(c[-1]), left, "보유 중(기간 끝)"))
        trades.append(t)
        i = max(k, j) + 1
    return trades


def portfolio(trades: List[Trade], start_equity: float = 100.0, risk_pct: float = R["risk_pct"],
              use_slump: bool = True) -> Dict[str, Any]:
    """진입일 순으로 자금 규칙을 적용. 손익은 청산일에 반영(실현 기준)."""
    events = sorted(trades, key=lambda t: (t.entry_date, t.ticker))
    equity = start_equity
    open_pos: List[Dict[str, Any]] = []
    taken, skipped = [], {"동시 보유 한도": 0, "오픈 리스크 한도": 0, "슬럼프 중단": 0, "같은 종목 보유 중": 0}
    curve = [(events[0].entry_date if events else date.today(), equity)]
    tracker = sr.SlumpTracker()

    def close_until(d: date):
        nonlocal equity
        for p in sorted([p for p in open_pos if p["trade"].exit_date <= d], key=lambda p: p["trade"].exit_date):
            res = p["trade"].result(sr.COST[sr.market_of(p["trade"].ticker)])
            pnl = p["amount"] * res["ret_pct"] / 100
            equity += pnl
            impact = pnl / p["equity_at_entry"] * 100
            tracker.add(impact, p["trade"].exit_date)
            curve.append((p["trade"].exit_date, equity))
            taken.append({**res, "trade": p["trade"], "weight": p["weight"], "impact": impact})
            open_pos.remove(p)

    for t in events:
        close_until(t.entry_date)
        status = tracker.status(t.entry_date) if use_slump else {"blocked": False, "mult": 1.0}
        if any(p["trade"].ticker == t.ticker for p in open_pos):
            skipped["같은 종목 보유 중"] += 1
            continue
        if len(open_pos) >= R["max_positions"]:
            skipped["동시 보유 한도"] += 1
            continue
        if status["blocked"]:
            skipped["슬럼프 중단"] += 1
            continue
        risk = risk_pct * sr.REGIME[t.regime]["risk_factor"] * status["mult"] * (0.5 if t.setup == "C" else 1)
        stop_pct = t.r / t.entry * 100
        weight = min(risk / stop_pct * 100, R["max_weight"]) if stop_pct > 0 else 0
        this_risk = weight * stop_pct / 100
        if sum(p["risk"] for p in open_pos) + this_risk > R["max_open_risk"] + 1e-9:
            skipped["오픈 리스크 한도"] += 1
            continue
        if weight <= 0:
            continue
        open_pos.append({"trade": t, "weight": weight, "amount": equity * weight / 100, "risk": this_risk,
                         "equity_at_entry": equity})
    close_until(date.max)
    return {"taken": taken, "skipped": skipped, "curve": curve, "final": equity, "start": start_equity}


def stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"n": 0}
    r = np.array([x["r_mult"] for x in results])
    wins, losses = r[r > 0], r[r <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    return {"n": len(r), "win_rate": len(wins) / len(r) * 100,
            "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
            "expectancy_r": float(r.mean()), "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else None,
            "avg_days": float(np.mean([x["days"] for x in results]))}


def max_drawdown(curve: List[tuple]) -> float:
    peak, mdd = -1e18, 0.0
    for _, v in curve:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak * 100 if peak > 0 else 0)
    return mdd


def run(tickers: List[str], years: int = 5, setups: Iterable[str] = ("A", "B"), use_slump: bool = True,
        progress: Optional[Callable[[float, str], None]] = None,
        loader: Callable[..., pd.DataFrame] = None) -> Dict[str, Any]:
    loader = loader or sr.load_history
    today = date.today()
    start = date(today.year - years, today.month, min(today.day, 28))
    warm = f"{start.year - 2}-{start.month:02d}-01"
    regimes: Dict[str, pd.Series] = {}
    all_trades: List[Trade] = []
    errors = []
    for n, t in enumerate(tickers, 1):
        if progress:
            progress(n / len(tickers), f"{t} ({n}/{len(tickers)})")
        m = sr.market_of(t)
        if m not in regimes:
            idx = loader(sr.INDEX[m][0], start=warm)
            regimes[m] = sr.regime_series(idx, m) if not idx.empty else pd.Series(dtype=object)
        h = loader(t, start=warm)
        if h.empty or len(h) < 300:
            errors.append(t)
            continue
        all_trades += simulate_ticker(t, sr.indicators(h), regimes[m], setups, start=start)
    by_trade = [{**t.result(sr.COST[sr.market_of(t.ticker)]), "trade": t} for t in all_trades]
    pf = portfolio(all_trades, use_slump=use_slump)
    days = max(1, (pf["curve"][-1][0] - pf["curve"][0][0]).days) if pf["curve"] else 1
    total = (pf["final"] / pf["start"] - 1) * 100
    cagr = ((pf["final"] / pf["start"]) ** (365 / days) - 1) * 100 if pf["final"] > 0 else -100
    by_setup = {s: stats([x for x in by_trade if x["trade"].setup == s]) for s in sorted({t.setup for t in all_trades})}
    by_year: Dict[int, List] = {}
    for x in pf["taken"]:
        by_year.setdefault(x["trade"].exit_date.year, []).append(x)
    return {
        "start": str(start), "end": str(today), "tickers": len(tickers), "errors": errors,
        "signals": stats(by_trade), "by_setup": by_setup,
        "portfolio": {**stats(pf["taken"]), "total_pct": float(total), "cagr_pct": float(cagr),
                      "mdd_pct": float(max_drawdown(pf["curve"])),
                      "skipped": pf["skipped"]},
        "by_year": {y: {**stats(v), "pnl_pct": float(sum(x["impact"] for x in v))} for y, v in sorted(by_year.items())},
        "curve": pf["curve"],
        "trades": [{"종목": x["trade"].ticker, "셋업": sr.SETUP_KO[x["trade"].setup], "신호일": str(x["trade"].signal_date),
                    "매수일": str(x["trade"].entry_date), "매수가": round(x["trade"].entry, 4),
                    "손절가": round(x["trade"].stop0, 4), "청산일": str(x["trade"].exit_date),
                    "청산 사유": " → ".join(e[3] for e in x["trade"].exits), "R": round(x["r_mult"], 2),
                    "수익률%": round(x["ret_pct"], 2), "보유일": x["days"]} for x in by_trade],
    }
