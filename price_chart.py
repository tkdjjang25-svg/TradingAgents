"""네이버·토스 증권 앱 같은 일봉 캔들 차트 (Altair) 위에 분석의 가격 조건을 겹쳐 그린다.

- 한국식 색: 상승 빨강, 하락 파랑
- 이동평균 5·20·60·120일, 아래에 거래량
- 진입·추가 매수(초록), 손절(보라), 목표가(주황), 계획 폐기·보류(회색) 가격선, 구간 조건은 띠로
- 주말·휴장일은 빈칸 없이 붙여 그린다
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

import altair as alt
import pandas as pd

UP, DOWN = "#e5383b", "#2f6fdf"
MA_COLORS = {"MA5": "#10b981", "MA20": "#ef4444", "MA60": "#f59e0b", "MA120": "#8b5cf6"}
ROLE_COLOR = {
    "entry": "#16a34a", "add": "#16a34a", "stop": "#7c3aed",
    "take_profit": "#ea580c", "cancel": "#6b7280", "wait": "#9ca3af", "avg": "#0f766e",
}
ROLE_DASH = {"stop": [6, 4], "cancel": [2, 3], "wait": [2, 3], "avg": [8, 3]}
PERIOD_DAYS = {"3개월": 63, "6개월": 126, "1년": 252}
# 마지막 종가에서 이보다 먼 가격선은 차트를 납작하게 만들어서 표에만 둔다
MAX_LEVEL_GAP = 0.35


def load_ohlc(ticker: str, days: int = 126) -> pd.DataFrame:
    """일봉 OHLCV + 이동평균. 데이터가 없으면 빈 DataFrame."""
    import tradingagents.dataflows  # noqa: F401  (yfinance 캐시를 data/cache/yfinance 로)
    import yfinance as yf

    h = yf.Ticker(ticker).history(period="2y", auto_adjust=False)
    return prepare(h, days)


def prepare(h: pd.DataFrame, days: int) -> pd.DataFrame:
    if h is None or h.empty:
        return pd.DataFrame()
    df = h[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    for n in (5, 20, 60, 120):
        df[f"MA{n}"] = df["Close"].rolling(n).mean()
    df = df.tail(days)
    df.index = [pd.Timestamp(d).date() for d in df.index]
    df = df.rename_axis("date").reset_index()
    df["d"] = [str(d) for d in df["date"]]
    df["up"] = df["Close"] >= df["Open"]
    df["chg"] = df["Close"].pct_change() * 100
    return df


ROLE_SHORT = {
    "entry": "진입", "add": "추가 매수", "stop": "손절", "take_profit": "목표가",
    "cancel": "계획 폐기", "wait": "진입 보류", "avg": "내 평단",
}
# 같은 가격(0.3% 이내)의 선은 이름을 한 줄로 합친다
SAME_PRICE = 0.003
LABEL_PX = 14  # 가격선 이름 한 줄의 최소 간격(px)


def fmt_level(y: float) -> str:
    return f"{y:,.0f}" if abs(y) >= 1000 else f"{y:,.2f}".rstrip("0").rstrip(".")


def _route_tag(label: str) -> str:
    """'A안: …', '경로 B …' 같은 갈래 이름을 앞에 붙인다."""
    m = re.match(r"\s*(?:경로\s*)?([A-Z가-힣]{1,2}안|[A-Z])\s*[:：)]", label or "")
    if not m:
        return ""
    tag = m.group(1)
    return tag if tag.endswith("안") else f"{tag}안"


def levels_from_plan(plan: Optional[Dict[str, Any]], avg_price: Optional[float] = None) -> List[Dict[str, Any]]:
    """entry_plans 의 조건 → 차트 가격선/구간.

    label 은 분석의 원래 문장, short 는 차트에 그릴 짧은 이름(예: 'A안 진입 282').
    """
    out = []
    for t in (plan or {}).get("triggers") or []:
        label = t.get("label") or t["role"]
        tag = _route_tag(label)
        name = (f"{tag} " if tag else "") + ROLE_SHORT.get(t["role"], t["role"])
        base = {"role": t["role"], "label": label, "detail": t.get("other_conditions") or "",
                "color": ROLE_COLOR.get(t["role"], "#6b7280")}
        if t["kind"] == "close_between":
            out.append({**base, "y": t["low"], "y2": t["high"],
                        "short": f"{name} {fmt_level(t['low'])}~{fmt_level(t['high'])}"})
        else:
            out.append({**base, "y": t["price"], "y2": None, "short": f"{name} {fmt_level(t['price'])}"})
    if avg_price:
        out.append({"role": "avg", "label": "내 평균단가", "y": float(avg_price), "y2": None,
                    "short": f"내 평단 {fmt_level(float(avg_price))}", "color": ROLE_COLOR["avg"]})
    return out


def _label_rows(levels: List[Dict[str, Any]], lo: float, hi: float, height: int) -> pd.DataFrame:
    """가격선 이름: 같은 가격은 한 줄로 합치고, 가까운 줄은 위아래로 벌려 겹치지 않게."""
    rows: List[Dict[str, Any]] = []
    for lv in sorted(levels, key=lambda v: v.get("y2") or v["y"]):
        y = lv.get("y2") or lv["y"]
        text = lv.get("short") or lv["label"]
        if rows and abs(y / rows[-1]["y"] - 1) <= SAME_PRICE:
            prev = rows[-1]["text"]
            if text != prev:
                # '손절 258' + '계획 폐기 258' → '손절 · 계획 폐기 258'
                p_name, _, p_price = prev.rpartition(" ")
                name, _, price = text.rpartition(" ")
                rows[-1]["text"] = (f"{p_name} · {name} {price}" if p_name and name and p_price == price
                                    else f"{prev} · {text}")
            continue
        rows.append({"y": y, "text": text, "color": lv["color"]})
    gap = (hi - lo) * LABEL_PX / max(height, 1)
    ly = [r["y"] for r in rows]
    for i in range(1, len(ly)):  # 아래에서 위로 밀어 올림
        ly[i] = max(ly[i], ly[i - 1] + gap)
    if ly and ly[-1] > hi - gap:  # 위로 넘치면(글자가 잘림) 위에서 아래로 다시 내림
        ly[-1] = hi - gap
        for i in range(len(ly) - 2, -1, -1):
            ly[i] = min(ly[i], ly[i + 1] - gap)
    for r, y in zip(rows, ly):
        r["ly"] = y
    return pd.DataFrame(rows)


def legend_rows(levels: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    """차트 아래 '선 설명' 표: 짧은 이름 · 원래 조건 문장 · 함께 볼 것."""
    out = []
    for lv in sorted(levels, key=lambda v: -(v.get("y2") or v["y"])):
        out.append({"차트 이름": lv.get("short") or lv["label"],
                    "뜻": lv["label"], "함께 볼 것": lv.get("detail") or "-"})
    return out


def _visible(levels: Iterable[Dict[str, Any]], last: float) -> List[Dict[str, Any]]:
    keep = []
    for lv in levels:
        ys = [lv["y"]] + ([lv["y2"]] if lv.get("y2") else [])
        if all(abs(y / last - 1) <= MAX_LEVEL_GAP for y in ys):
            keep.append(lv)
    return keep


def build_chart(df: pd.DataFrame, levels: Iterable[Dict[str, Any]] = (), height: int = 340) -> alt.VConcatChart:
    """캔들 + 이동평균 + 가격 조건 / 거래량 두 칸짜리 차트."""
    last = float(df["Close"].iloc[-1])
    levels = _visible(levels, last)
    step = max(1, len(df) // 8)
    ticks = list(df["d"].iloc[::step])
    x = alt.X("d:O", axis=alt.Axis(title=None, values=ticks, labelAngle=0,
                                   labelExpr="slice(datum.value, 5)"),
              scale=alt.Scale(paddingInner=0.25))
    color = alt.condition("datum.up", alt.value(UP), alt.value(DOWN))
    tooltip = [
        alt.Tooltip("d:N", title="날짜"), alt.Tooltip("Open:Q", title="시가", format=",.4~f"),
        alt.Tooltip("High:Q", title="고가", format=",.4~f"), alt.Tooltip("Low:Q", title="저가", format=",.4~f"),
        alt.Tooltip("Close:Q", title="종가", format=",.4~f"), alt.Tooltip("chg:Q", title="등락(%)", format="+.2f"),
        alt.Tooltip("Volume:Q", title="거래량", format=",.0f"),
    ]
    lo = min([float(df["Low"].min())] + [lv["y"] for lv in levels])
    hi = max([float(df["High"].max())] + [lv.get("y2") or lv["y"] for lv in levels])
    pad = (hi - lo) * 0.04 or hi * 0.02
    y_scale = alt.Scale(domain=[lo - pad, hi + pad], zero=False, nice=False)

    base = alt.Chart(df).encode(x=x)
    wick = base.mark_rule(strokeWidth=1).encode(
        y=alt.Y("Low:Q", scale=y_scale, title=None, axis=alt.Axis(orient="right", format=",.4~f")),
        y2="High:Q", color=color, tooltip=tooltip,
    )
    body = base.mark_bar().encode(y="Open:Q", y2="Close:Q", color=color, tooltip=tooltip)
    ma = base.transform_fold(list(MA_COLORS), as_=["이평", "값"]).mark_line(strokeWidth=1.2, opacity=0.85).encode(
        y="값:Q",
        color=alt.Color("이평:N", scale=alt.Scale(domain=list(MA_COLORS), range=list(MA_COLORS.values())),
                        legend=alt.Legend(orient="top-left", title=None, direction="horizontal")),
    )
    layers = [ma, wick, body]

    if levels:
        lv = pd.DataFrame(levels)
        bands = lv[lv["y2"].notna()]
        lines = lv[lv["y2"].isna()]
        if not bands.empty:
            layers.insert(0, alt.Chart(bands).mark_rect(opacity=0.13).encode(
                y="y:Q", y2="y2:Q", color=alt.Color("color:N", scale=None),
                tooltip=[alt.Tooltip("label:N", title="조건")]))
        for role, group in lines.groupby("role"):
            dash = ROLE_DASH.get(role, [1, 0])
            layers.append(alt.Chart(group).mark_rule(strokeWidth=1.6, strokeDash=dash).encode(
                y="y:Q", color=alt.Color("color:N", scale=None),
                tooltip=[alt.Tooltip("label:N", title="조건"), alt.Tooltip("y:Q", title="가격", format=",.4~f")]))
        # 이름은 짧게, 겹치면 위아래로 벌림. 밀려난 이름은 왼쪽 짧은 가로선으로 원래 선과 이음
        names = _label_rows(levels, lo - pad, hi + pad, height)
        moved = names[(names["ly"] - names["y"]).abs() > 1e-9]
        if not moved.empty:
            layers.append(alt.Chart(moved).mark_rule(strokeWidth=1, opacity=0.6).encode(
                x=alt.value(0), x2=alt.value(4), y="y:Q", y2="ly:Q", color=alt.Color("color:N", scale=None)))
        layers.append(alt.Chart(names).mark_text(
            align="left", baseline="bottom", dx=6, dy=-1, fontSize=11, fontWeight="bold",
            stroke="white", strokeWidth=3, strokeOpacity=0.85,  # 캔들·선 위에서도 읽히게 흰 테두리
        ).encode(x=alt.value(0), y="ly:Q", text="text:N"))
        layers.append(alt.Chart(names).mark_text(
            align="left", baseline="bottom", dx=6, dy=-1, fontSize=11, fontWeight="bold",
        ).encode(x=alt.value(0), y="ly:Q", text="text:N", color=alt.Color("color:N", scale=None)))

    price = alt.layer(*layers).properties(height=height).resolve_scale(color="independent")
    volume = base.mark_bar().encode(
        y=alt.Y("Volume:Q", title=None, axis=alt.Axis(orient="right", format="~s", tickCount=3)),
        color=color, tooltip=tooltip,
    ).properties(height=80)
    return alt.vconcat(price, volume, spacing=4).resolve_scale(x="shared")


def hidden_levels(levels: Iterable[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
    """차트에서 뺀(너무 먼) 가격 조건 — 화면에 따로 적어 준다."""
    last = float(df["Close"].iloc[-1])
    shown = _visible(levels, last)
    return [lv for lv in levels if lv not in shown]
