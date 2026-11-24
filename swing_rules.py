"""스윙 매매 규칙: 시장 국면, 진입 셋업, 1% 룰 수량, 10계명 점검, 스크리너, 매매일지·슬럼프 단계.

미너비니·다바스·오닐·BNF·남석관·성현우의 원칙을 숫자로 옮긴 출발값이다. 수치는 RULES 에 모여 있다.

데이터
- 가격·재무: yfinance (일봉, 분기 손익·재무상태표)
- 기관·외국인 순매수(국내): 네이버 증권 모바일 JSON (최근 20거래일)
- 위험 공시(국내): OpenDART 공시 목록 (CB·BW·유상증자 횟수, 관리종목·거래정지·불성실공시 등)
투자경고·위험 같은 시장경보는 DART 에 없어서 직접 확인하도록 남긴다.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from tradingagents.default_config import TRADINGAGENTS_HOME

log = logging.getLogger("swing")

SWING_DIR = Path(TRADINGAGENTS_HOME) / "swing"
SETTINGS_FILE = SWING_DIR / "settings.json"
JOURNAL_FILE = SWING_DIR / "journal.json"
SCREEN_FILE = SWING_DIR / "screen_latest.json"
_LOCK = threading.Lock()

RULES = {
    "risk_pct": 1.0,          # 1회 매매 최대 손실 (총자산 대비 %)
    "max_weight": 20.0,       # 1종목 최대 비중 %
    "max_positions": 6,       # 동시 보유 종목 수
    "max_open_risk": 5.0,     # 보유 종목이 전부 손절될 때 합계 손실 %
    "stop_pct": 7.0,          # 기본 손절 −7%
    "stop_limit": 8.0,        # 차트상 손절 자리가 이보다 멀면 매수하지 않는다
    "chase_pct": 5.0,         # 돌파 기준가 +5% 넘게 오른 뒤에는 사지 않는다
    "breakout_vol": 2.0,      # 돌파일 거래량 ≥ 20일 평균 × 2
    "pullback_vol": 0.7,      # 눌림 구간 거래량 ≤ 20일 평균 × 0.7
    "hot_5d": 30.0,           # 5일 상승률 이 이상이면 과열
    "dev20_max": 115.0,       # 20일선 이격도(%) 이 이상이면 과열
    "reversal_dev25": 85.0,   # BNF: 25일선 대비 −15% 이하
    "t1_r": 2.0, "t2_r": 4.0,  # 분할 익절 지점 (R 배수)
    "trail_ma": 10, "trail_pct": 10.0,  # 남은 물량: 종가 10일선 이탈 또는 최고 종가 대비 −10%
    "exit_ma": 20,            # 1차 익절 전: 종가 20일선 이탈 시 정리
    "time_stop_days": 10, "time_stop_gain": 5.0,  # 10거래일 안에 +5% 못 가면 정리
    "max_debt_ratio": 200.0,  # 부채비율 %
    "max_dilution": 1,        # 최근 1년 CB·BW·유상증자 결정이 이 횟수를 넘으면 탈락
    "chase_gap_pct": 3.0,     # 아침 갭이 감시가보다 이만큼 높으면 주문 취소
}
MIN_TURNOVER = {"KS": 100e8, "KQ": 50e8, "US": 20e6}   # 20일 평균 거래대금
MIN_MCAP = {"KS": 3000e8, "KQ": 1500e8, "US": 2e9}
COST = {"KS": 0.0025, "KQ": 0.0025, "US": 0.001}       # 왕복 수수료·세금
INDEX = {"KS": ("^KS11", "코스피"), "KQ": ("^KQ11", "코스닥"), "US": ("^GSPC", "S&P500")}
REGIME_MA = {"KS": (20, 60), "KQ": (20, 60), "US": (50, 200)}
REGIME = {
    "up": {"ko": "🟢 상승", "exposure": 100, "risk_factor": 1.0},
    "neutral": {"ko": "🟡 중립", "exposure": 50, "risk_factor": 0.5},
    "down": {"ko": "🔴 하락", "exposure": 20, "risk_factor": 0.0},
}
SETUP_KO = {"A": "돌파", "B": "20일선 눌림", "C": "역추세(25일선 이격)", "W": "돌파 대기"}

UNIVERSES = {
    "코스피 대형주": [
        "005930.KS", "000660.KS", "373220.KS", "207940.KS", "005380.KS", "000270.KS", "068270.KS",
        "035420.KS", "105560.KS", "055550.KS", "005490.KS", "051910.KS", "006400.KS", "012330.KS",
        "035720.KS", "028260.KS", "066570.KS", "003670.KS", "096770.KS", "034730.KS", "015760.KS",
        "032830.KS", "086790.KS", "010130.KS", "009150.KS", "011200.KS", "017670.KS", "018260.KS",
        "033780.KS", "003550.KS", "042660.KS", "012450.KS", "329180.KS", "010140.KS", "047810.KS",
    ],
    "코스닥 주요주": [
        "247540.KQ", "086520.KQ", "196170.KQ", "028300.KQ", "263750.KQ", "141080.KQ", "058470.KQ",
        "145020.KQ", "357780.KQ", "039030.KQ", "240810.KQ", "277810.KQ", "214150.KQ", "095340.KQ",
    ],
    "미국 대형주": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "NFLX", "COST",
        "JPM", "LLY", "V", "MA", "ORCL", "CRM", "ADBE", "QCOM", "MU", "PLTR", "UBER", "ANET", "NOW",
    ],
}


# --------------------------------------------------------------------------- 공통


def market_of(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".KS"):
        return "KS"
    if t.endswith(".KQ"):
        return "KQ"
    return "US"


def currency_of(ticker: str) -> str:
    return "USD" if market_of(ticker) == "US" else "KRW"


def fmt_price(value: Optional[float], ticker: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if currency_of(ticker) == "KRW":
        return f"{value:,.0f}원"
    return f"${value:,.2f}"


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def load_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    s = {"equity_krw": None, "equity_usd": None, "risk_pct": RULES["risk_pct"],
         "universes": ["코스피 대형주", "코스닥 주요주", "미국 대형주"], "extra_tickers": [],
         "auto_screen": True}
    s.update(_read(path or SETTINGS_FILE, {}))
    return s


def save_settings(settings: Dict[str, Any], path: Optional[Path] = None) -> None:
    _write(path or SETTINGS_FILE, settings)


def equity_for(ticker: str, settings: Dict[str, Any]) -> Optional[float]:
    v = settings.get("equity_usd" if currency_of(ticker) == "USD" else "equity_krw")
    return float(v) if v else None


# --------------------------------------------------------------------------- 가격·지표


def load_history(ticker: str, period: str = "2y", start: Optional[str] = None) -> pd.DataFrame:
    import tradingagents.dataflows  # noqa: F401  (yfinance 캐시를 data/cache/yfinance 로)
    import yfinance as yf

    t = yf.Ticker(ticker)
    h = t.history(start=start, auto_adjust=False) if start else t.history(period=period, auto_adjust=False)
    if h is None or h.empty:
        return pd.DataFrame()
    h = h[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
    h = h[(h["Close"] > 0) & (h["High"] > 0)]
    h.index = pd.to_datetime([pd.Timestamp(d).date() for d in h.index])
    return h


def indicators(h: pd.DataFrame) -> pd.DataFrame:
    """일봉 → 규칙 판정에 쓰는 지표. 오늘 값에 오늘 이후 데이터는 쓰지 않는다."""
    df = h.copy()
    c, hi, lo, v = df["Close"], df["High"], df["Low"], df["Volume"]
    for n in (5, 10, 20, 25, 50, 60, 120, 200):
        df[f"MA{n}"] = c.rolling(n).mean()
    df["VOL20"] = v.shift(1).rolling(20).mean()                 # 오늘을 뺀 20일 평균 거래량
    df["TURN20"] = (c * v).rolling(20).mean()                   # 20일 평균 거래대금
    df["HI52"] = hi.rolling(252, min_periods=120).max()
    df["LO52"] = lo.rolling(252, min_periods=120).min()
    df["RET5"] = (c / c.shift(5) - 1) * 100
    df["DEV20"] = c / df["MA20"] * 100
    df["DEV25"] = c / df["MA25"] * 100
    df["PIVOT"] = hi.shift(1).rolling(20).max()                 # 직전 20일 박스 상단
    df["BOXLOW"] = lo.shift(1).rolling(10).min()                # 마지막 수축 구간 저점
    df["LOW5"] = lo.rolling(5).min()
    df["LOW3"] = lo.rolling(3).min()
    df["VOL3"] = v.shift(1).rolling(3).mean()

    def rng(n: int, shift: int) -> pd.Series:
        return (hi.shift(shift).rolling(n).max() / lo.shift(shift).rolling(n).min() - 1) * 100

    df["RNG1"], df["RNG2"], df["RNG3"] = rng(20, 21), rng(10, 11), rng(10, 1)
    df["CONTRACT"] = (df["RNG1"] > df["RNG2"]) & (df["RNG2"] > df["RNG3"])
    df["DRYUP"] = v.shift(1).rolling(5).mean() < df["VOL20"] * 0.8
    df["TREND"] = ((df["MA20"] > df["MA60"]) & (df["MA60"] > df["MA120"]) & (c > df["MA60"])
                   & (c >= df["HI52"] * 0.75) & (c >= df["LO52"] * 1.3))
    df["SIG_A"] = (df["TREND"] & (c > df["PIVOT"]) & (v >= RULES["breakout_vol"] * df["VOL20"])
                   & (c <= df["PIVOT"] * (1 + RULES["chase_pct"] / 100)))
    df["SIG_B"] = (df["TREND"] & (lo <= df["MA20"] * 1.01) & (c > df["MA20"]) & (c > df["Open"])
                   & (df["VOL3"] <= RULES["pullback_vol"] * df["VOL20"]))
    df["SIG_C"] = (df["DEV25"] <= RULES["reversal_dev25"]) & (c > df["Open"])
    df["SIG_W"] = df["TREND"] & (c >= df["PIVOT"] * 0.97) & (c <= df["PIVOT"])
    return df


def structural_stop(row: pd.Series, setup: str) -> Optional[float]:
    """셋업별 차트상 손절 자리 (−1% 여유)."""
    key = {"A": "BOXLOW", "W": "BOXLOW", "B": "LOW5", "C": "LOW3"}.get(setup)
    v = row.get(key) if key else None
    return float(v) * 0.99 if v is not None and not pd.isna(v) else None


def setup_of(row: pd.Series, allow_reversal: bool = True) -> Optional[str]:
    for s in ("A", "B"):
        if bool(row.get(f"SIG_{s}")):
            return s
    if allow_reversal and bool(row.get("SIG_C")):
        return "C"
    if bool(row.get("SIG_W")):
        return "W"
    return None


def plan_trade(entry: float, struct_stop: Optional[float], equity: Optional[float] = None,
               risk_pct: float = RULES["risk_pct"], max_weight: float = RULES["max_weight"],
               cost: float = 0.0025) -> Dict[str, Any]:
    """손절가(−7% 와 차트상 자리 중 높은 쪽), 2R·4R 목표, 1% 룰 수량."""
    floor = entry * (1 - RULES["stop_pct"] / 100)
    stop = max(floor, struct_stop) if struct_stop and struct_stop < entry else floor
    too_far = bool(struct_stop and struct_stop < entry * (1 - RULES["stop_limit"] / 100))
    r = entry - stop
    out = {
        "entry": entry, "stop": stop, "stop_pct": (1 - stop / entry) * 100, "struct_stop": struct_stop,
        "too_far": too_far, "r": r, "t1": entry + RULES["t1_r"] * r, "t2": entry + RULES["t2_r"] * r,
        "breakeven": entry * (1 + cost), "risk_pct": risk_pct, "qty": None, "amount": None,
        "weight": None, "loss": None, "capped": False,
    }
    if equity and r > 0 and risk_pct > 0:
        qty = math.floor(equity * risk_pct / 100 / r)
        cap = math.floor(equity * max_weight / 100 / entry)
        out["capped"] = qty > cap
        qty = min(qty, cap)
        out.update(qty=qty, amount=qty * entry, weight=qty * entry / equity * 100, loss=qty * r)
    return out


def chart_levels(plan: Dict[str, Any], ticker: str = "") -> List[Dict[str, Any]]:
    """price_chart 가격선 형식."""
    from price_chart import ROLE_COLOR

    def lv(role, label, y):
        return {"role": role, "label": label, "y": float(y), "y2": None, "color": ROLE_COLOR[role]}

    return [
        lv("entry", f"스윙 진입 {fmt_price(plan['entry'], ticker)}", plan["entry"]),
        lv("stop", f"손절 {fmt_price(plan['stop'], ticker)} (−{plan['stop_pct']:.1f}%)", plan["stop"]),
        lv("take_profit", f"2R 1/3 익절 {fmt_price(plan['t1'], ticker)}", plan["t1"]),
        lv("take_profit", f"4R 1/3 익절 {fmt_price(plan['t2'], ticker)}", plan["t2"]),
    ]


# --------------------------------------------------------------------------- 시장 국면


def regime_series(index_hist: pd.DataFrame, market: str) -> pd.Series:
    fast, slow = REGIME_MA[market]
    c = index_hist["Close"]
    mf, ms = c.rolling(fast).mean(), c.rolling(slow).mean()
    rising = ms > ms.shift(5)
    up = (c > mf) & (mf > ms) & rising
    down = (c < ms) & (mf < ms)
    out = pd.Series("neutral", index=c.index)
    out[up] = "up"
    out[down] = "down"
    out[ms.isna()] = "neutral"
    return out


def market_regime(market: str, hist: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    symbol, name = INDEX[market]
    if hist is None:
        hist = load_history(symbol, period="2y")
    if hist.empty:
        return {"market": market, "name": name, "symbol": symbol, "regime": "neutral", **REGIME["neutral"],
                "error": "지수 데이터 없음"}
    fast, slow = REGIME_MA[market]
    c = hist["Close"]
    key = regime_series(hist, market).iloc[-1]
    return {"market": market, "name": name, "symbol": symbol, "regime": key, **REGIME[key],
            "close": float(c.iloc[-1]), "date": str(c.index[-1].date()), "fast": fast, "slow": slow,
            "ma_fast": float(c.rolling(fast).mean().iloc[-1]), "ma_slow": float(c.rolling(slow).mean().iloc[-1]),
            "above_slow": bool(c.iloc[-1] > c.rolling(slow).mean().iloc[-1])}


# --------------------------------------------------------------------------- 펀더멘털·수급·공시

_NAVER_LIMIT = threading.Lock()
_NAVER_NEXT = [0.0]


def _num(text) -> float:
    try:
        return float(str(text).replace(",", "").replace("+", "").strip() or 0)
    except ValueError:
        return 0.0


def parse_flows(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """네이버 trend 응답(최신순) → 기관+외국인 순매수 요약."""
    if not rows:
        return None
    net = [(_num(r.get("foreignerPureBuyQuant")) + _num(r.get("organPureBuyQuant"))) for r in rows[:20]]
    return {"days": len(net), "sum20": sum(net), "buy_days5": sum(1 for x in net[:5] if x > 0),
            "foreign20": sum(_num(r.get("foreignerPureBuyQuant")) for r in rows[:20]),
            "organ20": sum(_num(r.get("organPureBuyQuant")) for r in rows[:20]),
            "date": rows[0].get("bizdate")}


def investor_flows(ticker: str) -> Optional[Dict[str, Any]]:
    """국내 종목의 최근 20거래일 기관·외국인 순매수(주). 미국 종목이거나 실패하면 None."""
    if market_of(ticker) == "US":
        return None
    import requests

    code = ticker.split(".")[0]
    with _NAVER_LIMIT:
        wait = _NAVER_NEXT[0] - time.monotonic()
        _NAVER_NEXT[0] = max(time.monotonic(), _NAVER_NEXT[0]) + 0.3
    if wait > 0:
        time.sleep(wait)
    try:
        r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/trend", params={"pageSize": 20},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        return parse_flows(r.json())
    except Exception as exc:  # noqa: BLE001
        log.info("수급 조회 실패 %s: %s", ticker, type(exc).__name__)
        return None


def fundamentals(ticker: str) -> Dict[str, Any]:
    """시가총액, 최근 4분기 영업이익, 부채비율, 분기 EPS 증가율. 못 구한 값은 None."""
    import tradingagents.dataflows  # noqa: F401
    import yfinance as yf

    t = yf.Ticker(ticker)
    out: Dict[str, Any] = {"etf": False, "mcap": None, "op4": None, "op_last2": None, "debt_ratio": None,
                           "eps_yoy": None}
    try:
        out["mcap"] = float(t.fast_info["marketCap"] or 0) or None
    except Exception:  # noqa: BLE001
        pass
    try:
        out["etf"] = (t.get_info() or {}).get("quoteType") in ("ETF", "MUTUALFUND")
    except Exception:  # noqa: BLE001
        pass
    if out["etf"]:
        return out
    try:
        q = t.quarterly_income_stmt
        if q is not None and "Operating Income" in q.index:
            op = pd.to_numeric(q.loc["Operating Income"], errors="coerce").dropna()
            if len(op) >= 4:
                out["op4"] = float(op.iloc[:4].sum())
            if len(op) >= 2:
                out["op_last2"] = bool((op.iloc[:2] > 0).all())
        for key in ("Diluted EPS", "Basic EPS"):
            if q is not None and key in q.index:
                eps = pd.to_numeric(q.loc[key], errors="coerce")
                if len(eps) >= 5 and pd.notna(eps.iloc[0]) and pd.notna(eps.iloc[4]) and eps.iloc[4] > 0:
                    out["eps_yoy"] = float((eps.iloc[0] / eps.iloc[4] - 1) * 100)
                break
    except Exception:  # noqa: BLE001
        pass
    try:
        b = t.quarterly_balance_sheet
        if b is not None and "Total Liabilities Net Minority Interest" in b.index and "Stockholders Equity" in b.index:
            liab = float(pd.to_numeric(b.loc["Total Liabilities Net Minority Interest"], errors="coerce").dropna().iloc[0])
            eq = float(pd.to_numeric(b.loc["Stockholders Equity"], errors="coerce").dropna().iloc[0])
            out["debt_ratio"] = liab / eq * 100 if eq > 0 else 9999.0
    except Exception:  # noqa: BLE001
        pass
    return out


_DILUTION = ("전환사채권발행결정", "신주인수권부사채권발행결정", "유상증자결정")
_DANGER = ("관리종목", "투자주의환기", "상장폐지", "불성실공시", "매매거래정지", "감사의견거절", "부적정",
           "회생절차", "부도", "횡령", "배임", "자본잠식")


def classify_filings(names: Iterable[str]) -> Dict[str, Any]:
    dilution, danger = [], []
    for n in names:
        compact = "".join(str(n).split())
        if any(k in compact for k in _DILUTION) and "철회" not in compact:
            dilution.append(n)
        if any(k in compact for k in _DANGER) and "해제" not in compact:
            danger.append(n)
    return {"dilution": len(dilution), "dilution_names": dilution[:5], "danger": danger[:5]}


def risk_filings(ticker: str, days: int = 365) -> Optional[Dict[str, Any]]:
    """최근 1년 DART 공시에서 희석(CB·BW·유증)과 위험 공시. 국내가 아니거나 DART 미설정이면 None."""
    if market_of(ticker) == "US":
        return None
    from tradingagents.dataflows import dart

    info, err = dart._lookup(ticker)
    if err:
        return {"error": err}
    key = dart.get_env("DART_API_KEY")
    end = date.today()
    names, page = [], 1
    try:
        while True:
            data = dart._get_json("list.json", {
                "corp_code": info["corp_code"], "bgn_de": (end - timedelta(days=days)).strftime("%Y%m%d"),
                "end_de": end.strftime("%Y%m%d"), "page_no": page, "page_count": 100,
            }, key)
            names.extend(r.get("report_nm", "") for r in data.get("list") or [])
            if page >= int(data.get("total_page") or 1) or page >= 5:
                break
            page += 1
    except dart._NoData:
        pass
    except dart._DartError as exc:
        return {"error": str(exc)}
    return classify_filings(names)


# --------------------------------------------------------------------------- 매매일지·슬럼프


class SlumpTracker:
    """연속 손실·계좌 낙폭으로 단계를 정한다. 백테스트와 매매일지가 같이 쓴다.

    단계 0 정상 · 1 위험 절반(3연속 손절) · 2 1주 중단 후 절반(5연속 또는 낙폭 6%) · 3 2주 중단 후 1/4(낙폭 10%).
    2연속 수익마다 한 단계 회복.
    """
    MULT = {0: 1.0, 1: 0.5, 2: 0.5, 3: 0.25}
    PAUSE = {2: 7, 3: 14}

    def __init__(self):
        self.stage, self.losses, self.wins = 0, 0, 0
        self.cum, self.peak = 0.0, 0.0
        self.pause_until: Optional[date] = None
        self.month: Dict[str, float] = {}

    def add(self, impact_pct: float, when: date) -> None:
        self.cum += impact_pct
        self.peak = max(self.peak, self.cum)
        key = when.strftime("%Y-%m")
        self.month[key] = self.month.get(key, 0.0) + impact_pct
        if impact_pct < 0:
            self.losses += 1
            self.wins = 0
            new = 1 if self.losses >= 3 else 0
            if self.losses >= 5 or self.drawdown >= 6:
                new = 2
            if self.drawdown >= 10:
                new = 3
            if new > self.stage:
                self.stage = new
                if new in self.PAUSE:
                    self.pause_until = when + timedelta(days=self.PAUSE[new])
        else:
            self.losses = 0
            self.wins += 1
            if self.wins >= 2 and self.stage > 0:
                self.stage -= 1
                self.wins = 0

    @property
    def drawdown(self) -> float:
        return self.peak - self.cum

    def status(self, today: date) -> Dict[str, Any]:
        month_loss = self.month.get(today.strftime("%Y-%m"), 0.0)
        paused = bool(self.pause_until and today < self.pause_until)
        reasons = []
        if self.stage == 1:
            reasons.append(f"{self.losses}연속 손절 → 1회 위험 절반")
        if self.stage >= 2:
            reasons.append(f"슬럼프 {self.stage}단계 (연속 손절 {self.losses}회, 낙폭 {self.drawdown:.1f}%)")
        if paused:
            reasons.append(f"{self.pause_until} 까지 매매 중단")
        if month_loss <= -6:
            reasons.append(f"이번 달 손실 {month_loss:.1f}% → 이달 신규 매수 종료")
        return {"stage": self.stage, "mult": self.MULT[self.stage], "losses": self.losses,
                "drawdown": self.drawdown, "month_pnl": month_loss, "pause_until": self.pause_until,
                "blocked": paused or month_loss <= -6, "reasons": reasons}


def load_journal(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    return _read(path or JOURNAL_FILE, [])


def save_journal(trades: List[Dict[str, Any]], path: Optional[Path] = None) -> None:
    with _LOCK:
        _write(path or JOURNAL_FILE, trades)


def remaining_qty(t: Dict[str, Any]) -> float:
    return float(t["qty"]) - sum(float(x["qty"]) for x in t.get("exits", []))


def add_trade(ticker: str, entry: float, stop: float, qty: float, equity: float, setup: str = "",
              note: str = "", opened: Optional[str] = None, path: Optional[Path] = None) -> Dict[str, Any]:
    trades = load_journal(path)
    t = {"id": f"{datetime.now():%Y%m%d%H%M%S}_{ticker}", "ticker": ticker, "currency": currency_of(ticker),
         "opened": opened or str(date.today()), "entry": float(entry), "stop0": float(stop), "stop": float(stop),
         "qty": float(qty), "equity": float(equity), "setup": setup, "note": note, "exits": [], "closed": False}
    trades.append(t)
    save_journal(trades, path)
    return t


def add_exit(trade_id: str, price: float, qty: float, when: Optional[str] = None,
             path: Optional[Path] = None) -> Dict[str, Any]:
    trades = load_journal(path)
    t = next(x for x in trades if x["id"] == trade_id)
    qty = min(float(qty), remaining_qty(t))
    if qty <= 0:
        raise ValueError("남은 수량이 없습니다")
    t["exits"].append({"date": when or str(date.today()), "price": float(price), "qty": qty})
    if remaining_qty(t) <= 1e-9:
        t["closed"], t["closed_at"] = True, t["exits"][-1]["date"]
    save_journal(trades, path)
    return t


def update_stop(trade_id: str, stop: float, path: Optional[Path] = None) -> None:
    trades = load_journal(path)
    for t in trades:
        if t["id"] == trade_id:
            t["stop"] = float(stop)
    save_journal(trades, path)


def delete_trade(trade_id: str, path: Optional[Path] = None) -> None:
    save_journal([t for t in load_journal(path) if t["id"] != trade_id], path)


def trade_result(t: Dict[str, Any]) -> Dict[str, Any]:
    cost = COST[market_of(t["ticker"])]
    sold = sum(float(x["qty"]) for x in t.get("exits", []))
    pnl = sum(float(x["qty"]) * (float(x["price"]) - t["entry"]) for x in t.get("exits", []))
    pnl -= cost * t["entry"] * sold
    risk = (t["entry"] - t["stop0"]) * t["qty"]
    return {"pnl": pnl, "r": pnl / risk if risk > 0 else None,
            "impact_pct": pnl / t["equity"] * 100 if t.get("equity") else 0.0}


def journal_status(trades: List[Dict[str, Any]], today: Optional[date] = None) -> Dict[str, Any]:
    today = today or date.today()
    tracker = SlumpTracker()
    for t in sorted((x for x in trades if x.get("closed")), key=lambda x: x.get("closed_at", "")):
        tracker.add(trade_result(t)["impact_pct"], date.fromisoformat(t["closed_at"]))
    open_trades = [t for t in trades if not t.get("closed")]
    open_risk = sum(max(0.0, t["entry"] - t["stop"]) * remaining_qty(t) / t["equity"] * 100
                    for t in open_trades if t.get("equity"))
    st = tracker.status(today)
    st.update(open_trades=len(open_trades), open_risk=open_risk,
              open_tickers=[t["ticker"] for t in open_trades])
    return st


# --------------------------------------------------------------------------- 10계명


def _item(no, name, rule, status, detail):
    return {"no": no, "name": name, "rule": rule, "status": status, "detail": detail}


ICON = {"pass": "✅", "fail": "❌", "warn": "⚠️", "na": "➖"}


def checklist(ticker: str, df: pd.DataFrame, regime: Dict[str, Any], plan: Dict[str, Any], setup: Optional[str],
              fund: Optional[Dict[str, Any]], flows: Optional[Dict[str, Any]], filings: Optional[Dict[str, Any]],
              account: Optional[Dict[str, Any]], held: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    row = df.iloc[-1]
    m = market_of(ticker)
    c = float(row["Close"])
    items = []

    # 1 시장 국면
    if regime.get("error"):
        items.append(_item(1, "시장 국면", "지수가 장기 이평선 위", "warn", regime["error"]))
    else:
        items.append(_item(1, "시장 국면", f"{regime['name']} > {regime['slow']}일선, 하락 국면이면 신규 매수 금지",
                           "fail" if regime["regime"] == "down" else "pass",
                           f"{regime['ko']} · {regime['name']} {regime['close']:,.0f} / {regime['slow']}일선 "
                           f"{regime['ma_slow']:,.0f} · 권장 비중 {regime['exposure']}%"))

    # 2 종목 추세
    parts = [f"20>60>120 정배열 {'O' if row['MA20'] > row['MA60'] > row['MA120'] else 'X'}",
             f"52주 고가 대비 {(c / row['HI52'] - 1) * 100:+.1f}%", f"52주 저가 대비 {(c / row['LO52'] - 1) * 100:+.0f}%"]
    items.append(_item(2, "종목 추세", "정배열, 60일선 위, 52주 고가 −25% 이내, 저가 +30% 이상",
                       "pass" if bool(row["TREND"]) else "fail", " · ".join(parts)))

    # 3 재무 안전
    if fund is None:
        items.append(_item(3, "재무 안전", "흑자·부채비율 200% 이하·위험 공시 없음", "warn", "재무 데이터 미조회"))
    elif fund.get("etf"):
        items.append(_item(3, "재무 안전", "흑자·부채비율 200% 이하·위험 공시 없음", "na", "ETF — 재무 점검 대상 아님 "
                           "(레버리지 ETF 는 배수만큼 수량을 줄이세요)"))
    else:
        bad, unknown, notes = [], [], []
        if fund.get("mcap") is None:
            unknown.append("시가총액")
        elif fund["mcap"] < MIN_MCAP[m]:
            bad.append(f"시총 {fund['mcap'] / (1e8 if m != 'US' else 1e9):,.0f}{'억' if m != 'US' else 'B'} < 기준")
        if fund.get("op4") is None:
            unknown.append("영업이익")
        elif fund["op4"] <= 0 and not fund.get("op_last2"):
            bad.append("최근 4분기 영업적자")
        else:
            notes.append("영업이익 흑자" if fund["op4"] > 0 else "최근 2분기 흑자 전환")
        if fund.get("debt_ratio") is None:
            unknown.append("부채비율")
        elif fund["debt_ratio"] > RULES["max_debt_ratio"]:
            bad.append(f"부채비율 {fund['debt_ratio']:.0f}%")
        else:
            notes.append(f"부채비율 {fund['debt_ratio']:.0f}%")
        if fund.get("eps_yoy") is not None:
            notes.append(f"분기 EPS {fund['eps_yoy']:+.0f}% (가점 기준 +25%)")
        if filings:
            if filings.get("error"):
                unknown.append("공시(DART)")
            else:
                if filings["dilution"] > RULES["max_dilution"]:
                    bad.append(f"1년 CB·BW·유증 {filings['dilution']}회")
                if filings["danger"]:
                    bad.append("위험 공시: " + ", ".join(filings["danger"][:2]))
                if not filings["dilution"] and not filings["danger"]:
                    notes.append("1년 희석·위험 공시 없음")
        if m != "US":
            notes.append("투자경고·위험 지정은 증권사 앱에서 확인")
        status = "fail" if bad else ("warn" if unknown else "pass")
        items.append(_item(3, "재무 안전", "흑자·부채비율 200% 이하·관리/경고·CB 남발 아님", status,
                           " · ".join(bad + notes + ([f"확인 못함: {', '.join(unknown)}"] if unknown else []))))

    # 4 유동성·수급
    turn = row.get("TURN20")
    turn_ok = bool(turn and not pd.isna(turn) and turn >= MIN_TURNOVER[m])
    unit = (1e8, "억") if m != "US" else (1e6, "M$")
    detail = [f"20일 평균 거래대금 {turn / unit[0]:,.0f}{unit[1]} (기준 {MIN_TURNOVER[m] / unit[0]:,.0f}{unit[1]})"
              if turn and not pd.isna(turn) else "거래대금 없음"]
    if m == "US":
        status = "pass" if turn_ok else "fail"
        detail.append("미국 종목은 기관·외국인 수급 데이터 없음")
    elif flows is None:
        status = "warn" if turn_ok else "fail"
        detail.append("수급 조회 실패")
    else:
        flow_ok = flows["sum20"] > 0 and flows["buy_days5"] >= 3
        status = "pass" if turn_ok and flow_ok else "fail"
        detail.append(f"기관+외국인 20일 {flows['sum20']:+,.0f}주, 최근 5일 중 순매수 {flows['buy_days5']}일")
    items.append(_item(4, "유동성·수급", "거래대금 기준 이상, 기관+외국인 20일 순매수·5일 중 3일 이상", status,
                       " · ".join(detail)))

    # 5 타점
    if setup in ("A", "B", "C"):
        extra = ""
        if setup == "A":
            extra = f" · 거래량 {row['Volume'] / row['VOL20']:.1f}배" + (" · VCP 수축" if bool(row["CONTRACT"]) else "")
        items.append(_item(5, "타점", "박스 돌파+거래량 2배, 또는 20일선 눌림+거래량 감소", "pass" if setup != "C" else "warn",
                           f"{SETUP_KO[setup]} 신호{extra}" + (" — 역추세는 보조 전략, 비중 절반" if setup == "C" else "")))
    elif setup == "W":
        items.append(_item(5, "타점", "박스 돌파+거래량 2배, 또는 20일선 눌림+거래량 감소", "warn",
                           f"돌파 대기 — 박스 상단 {fmt_price(row['PIVOT'], ticker)} 위 +0.5% 에 감시 매수 가능"))
    else:
        items.append(_item(5, "타점", "박스 돌파+거래량 2배, 또는 20일선 눌림+거래량 감소", "fail", "오늘 매수 신호 없음"))

    # 6 추격 금지
    chase = (c / row["PIVOT"] - 1) * 100 if row.get("PIVOT") and not pd.isna(row["PIVOT"]) else None
    hot = []
    if setup == "A" and chase is not None and chase > RULES["chase_pct"]:
        hot.append(f"기준가 +{chase:.1f}%")
    if row["RET5"] >= RULES["hot_5d"]:
        hot.append(f"5일 +{row['RET5']:.0f}%")
    if row["DEV20"] >= RULES["dev20_max"]:
        hot.append(f"20일선 이격 {row['DEV20']:.0f}%")
    items.append(_item(6, "추격 금지", "기준가 +5% 이내, 5일 상승 30% 미만, 20일선 이격 115% 미만",
                       "fail" if hot else "pass",
                       ", ".join(hot) if hot else f"5일 {row['RET5']:+.1f}% · 20일선 이격 {row['DEV20']:.0f}%"))

    # 7 손절가
    items.append(_item(7, "손절가", "−7% 이내 차트상 자리, −8% 보다 멀면 패스",
                       "fail" if plan["too_far"] else "pass",
                       f"손절 {fmt_price(plan['stop'], ticker)} (−{plan['stop_pct']:.1f}%)"
                       + (f" · 차트상 자리 {fmt_price(plan['struct_stop'], ticker)} 가 −8% 보다 멂" if plan["too_far"] else
                          " · 매수 즉시 감시 주문 등록")))

    # 8 수량
    if plan.get("qty") is None:
        items.append(_item(8, "수량", "(총자산×1%)÷(매수가−손절가), 1종목 20% 이하", "warn",
                           "설정에서 총자산을 입력하면 수량을 계산합니다"))
    elif plan["qty"] <= 0:
        items.append(_item(8, "수량", "(총자산×1%)÷(매수가−손절가), 1종목 20% 이하", "fail",
                           "허용 위험이 0 (하락 국면 또는 슬럼프 중단)"))
    else:
        items.append(_item(8, "수량", "(총자산×1%)÷(매수가−손절가), 1종목 20% 이하", "pass",
                           f"{plan['qty']:,}주 · {fmt_price(plan['amount'], ticker)} · 비중 {plan['weight']:.1f}% · "
                           f"손절 시 −{fmt_price(plan['loss'], ticker)} (위험 {plan['risk_pct']:.2f}%)"
                           + (" · 20% 상한으로 줄임" if plan["capped"] else "")))

    # 9 손익비
    hi52 = float(row["HI52"])
    room = (hi52 - plan["entry"]) / plan["r"] if plan["r"] > 0 else 0
    at_high = plan["entry"] >= hi52 * 0.98
    items.append(_item(9, "손익비", "1차 목표 2R, 위쪽 저항(52주 고가)까지 3R 이상",
                       "pass" if at_high or room >= 3 else "fail",
                       "신고가권 — 위쪽 저항 없음" if at_high else f"52주 고가 {fmt_price(hi52, ticker)} 까지 {room:.1f}R"))

    # 10 계좌 상태
    fails, warns, notes = [], [], []
    if account:
        this = plan["risk_pct"] if plan.get("qty") else 0
        total = account["open_risk"] + this
        if total > RULES["max_open_risk"]:
            fails.append(f"오픈 리스크 {account['open_risk']:.1f}% + 이번 {this:.1f}% 가 5% 초과")
        else:
            notes.append(f"오픈 리스크 {account['open_risk']:.1f}% → {total:.1f}%")
        if account["open_trades"] >= RULES["max_positions"]:
            fails.append(f"이미 {account['open_trades']}종목 보유 (최대 {RULES['max_positions']})")
        if account["blocked"]:
            fails += account["reasons"]
        elif account["stage"] >= 1:
            warns += account["reasons"]
    if held and held.get("held") and held.get("avg_price") and c < float(held["avg_price"]):
        fails.append(f"평단 {fmt_price(float(held['avg_price']), ticker)} 아래로 보유 중 — 지금 사면 물타기")
    items.append(_item(10, "계좌 상태", "오픈 리스크 5% 이하, 슬럼프 단계 확인, 물타기 아님",
                       "fail" if fails else ("warn" if warns else "pass"),
                       " · ".join(fails + warns + notes) or "매매일지 기록 없음 — 첫 매매"))
    return items


def verdict(items: List[Dict[str, Any]]) -> str:
    st = [i["status"] for i in items]
    if "fail" in st:
        return "❌ 보류"
    if "warn" in st:
        return "⚠️ 확인 필요"
    return "✅ 매수 후보"


def effective_risk(settings: Dict[str, Any], regime: Dict[str, Any], account: Optional[Dict[str, Any]]) -> float:
    risk = float(settings.get("risk_pct") or RULES["risk_pct"]) * regime.get("risk_factor", 1.0)
    if account:
        risk = 0.0 if account["blocked"] else risk * account["mult"]
    return risk


def analyze(ticker: str, settings: Optional[Dict[str, Any]] = None, hist: Optional[pd.DataFrame] = None,
            regime: Optional[Dict[str, Any]] = None, account: Optional[Dict[str, Any]] = None,
            held: Optional[Dict[str, Any]] = None, deep: bool = True,
            entry: Optional[float] = None, stop: Optional[float] = None) -> Dict[str, Any]:
    """한 종목: 지표 → 셋업 → 계획 → 10계명. deep=False 면 재무·수급·공시를 건너뛴다(스크리너 1단계)."""
    settings = settings or load_settings()
    m = market_of(ticker)
    hist = load_history(ticker) if hist is None else hist
    if hist.empty or len(hist) < 130:
        return {"ticker": ticker, "error": "가격 데이터가 부족합니다 (상장 6개월 미만이거나 티커 오류)"}
    df = indicators(hist)
    row = df.iloc[-1]
    regime = regime or market_regime(m)
    setup = setup_of(row, allow_reversal=regime.get("regime") == "up")
    if account is None:
        account = journal_status(load_journal())
    risk = effective_risk(settings, regime, account)
    if setup == "C":
        risk /= 2
    entry = float(entry) if entry else (float(row["PIVOT"]) * 1.005 if setup == "W" else float(row["Close"]))
    struct = float(stop) if stop else structural_stop(row, setup or "A")
    plan = plan_trade(entry, struct, equity_for(ticker, settings), risk, cost=COST[m])
    fund = flows = filings = None
    if deep:
        fund = fundamentals(ticker)
        flows = investor_flows(ticker)
        filings = risk_filings(ticker)
    items = checklist(ticker, df, regime, plan, setup, fund, flows, filings, account, held)
    return {"ticker": ticker, "date": str(df.index[-1].date()), "close": float(row["Close"]), "setup": setup,
            "setup_ko": SETUP_KO.get(setup, "-"), "plan": plan, "items": items, "verdict": verdict(items),
            "passed": sum(i["status"] in ("pass", "na") for i in items), "regime": regime, "deep": deep,
            "pivot": float(row["PIVOT"]) if not pd.isna(row["PIVOT"]) else None, "df": df}


# --------------------------------------------------------------------------- 스크리너


def universe(settings: Dict[str, Any], watch: Iterable[str] = ()) -> List[str]:
    out: List[str] = []
    for name in settings.get("universes") or []:
        out += UNIVERSES.get(name, [])
    out += list(settings.get("extra_tickers") or []) + list(watch)
    seen, uniq = set(), []
    for t in out:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def run_screen(tickers: List[str], settings: Optional[Dict[str, Any]] = None,
               progress: Optional[Callable[[float, str], None]] = None, held: Optional[Dict[str, Dict]] = None,
               save: bool = True) -> Dict[str, Any]:
    """1단계: 가격 지표만으로 전 종목 판정 → 2단계: 신호가 있는 종목만 재무·수급·공시까지."""
    settings = settings or load_settings()
    held = held or {}
    regimes = {m: market_regime(m) for m in {market_of(t) for t in tickers}}
    account = journal_status(load_journal())
    done = [0]

    def light(t):
        try:
            r = analyze(t, settings, regime=regimes[market_of(t)], account=account, held=held.get(t), deep=False)
        except Exception as exc:  # noqa: BLE001
            r = {"ticker": t, "error": f"{type(exc).__name__}: {exc}"}
        done[0] += 1
        if progress:
            progress(done[0] / len(tickers) * 0.6, f"가격 점검 {done[0]}/{len(tickers)} · {t}")
        return r

    with ThreadPoolExecutor(max_workers=4) as pool:
        first = list(pool.map(light, tickers))
    picks = [r for r in first if not r.get("error") and r.get("setup")]
    results = {r["ticker"]: r for r in first}
    for i, r in enumerate(picks, 1):
        if progress:
            progress(0.6 + 0.4 * i / max(1, len(picks)), f"재무·수급·공시 {i}/{len(picks)} · {r['ticker']}")
        try:
            results[r["ticker"]] = analyze(r["ticker"], settings, regime=regimes[market_of(r["ticker"])],
                                           account=account, held=held.get(r["ticker"]), deep=True)
        except Exception as exc:  # noqa: BLE001
            log.info("심화 점검 실패 %s: %s", r["ticker"], exc)
    rows = []
    for t in tickers:
        r = results[t]
        if r.get("error"):
            rows.append({"ticker": t, "error": r["error"]})
            continue
        rows.append({
            "ticker": t, "date": r["date"], "close": r["close"], "setup": r["setup"], "setup_ko": r["setup_ko"],
            "verdict": r["verdict"] if r["deep"] else ("—" if not r["setup"] else r["verdict"]),
            "passed": r["passed"], "deep": r["deep"], "pivot": r["pivot"],
            "plan": {k: v for k, v in r["plan"].items()},
            "items": r["items"],
        })
    out = {"created_at": datetime.now().isoformat(timespec="seconds"), "count": len(tickers),
           "regimes": {m: {k: v for k, v in g.items()} for m, g in regimes.items()}, "rows": rows}
    if save:
        _write(SCREEN_FILE, out)
    return out


def load_screen() -> Optional[Dict[str, Any]]:
    return _read(SCREEN_FILE, None)
