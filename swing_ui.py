"""앱의 '📐 스윙 매매' 탭: 시장 국면, 10계명 점검, 스크리너, 매매일지, 백테스트, 설정."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

import price_chart
import swing_backtest
import swing_rules as sr
import watchlist

STATUS_KO = {"pass": "✅ 통과", "fail": "❌ 탈락", "warn": "⚠️ 확인", "na": "➖ 해당 없음"}


@st.cache_data(ttl=600, show_spinner=False)
def hist(ticker: str) -> pd.DataFrame:
    return sr.load_history(ticker)


@st.cache_data(ttl=1800, show_spinner=False)
def regime(market: str) -> Dict[str, Any]:
    return sr.market_regime(market)


@st.cache_data(ttl=3600, show_spinner=False)
def deep_data(ticker: str) -> Dict[str, Any]:
    return {"fund": sr.fundamentals(ticker), "flows": sr.investor_flows(ticker), "filings": sr.risk_filings(ticker)}


def _held() -> Dict[str, Dict[str, Any]]:
    return {i["ticker"]: i for i in watchlist.load_watchlist()}


def analyze(ticker: str, settings: Dict[str, Any], deep: bool = True, entry: Optional[float] = None,
            stop: Optional[float] = None) -> Dict[str, Any]:
    """앱용: 가격·재무를 캐시해 두고 10계명을 매번 다시 판정한다(설정·매매일지가 바로 반영되도록)."""
    h = hist(ticker)
    m = sr.market_of(ticker)
    res = sr.analyze(ticker, settings, hist=h, regime=regime(m), account=sr.journal_status(sr.load_journal()),
                     held=_held().get(ticker), deep=False, entry=entry, stop=stop)
    if res.get("error") or not deep:
        return res
    d = deep_data(ticker)
    items = sr.checklist(ticker, res["df"], res["regime"], res["plan"], res["setup"], d["fund"], d["flows"],
                         d["filings"], sr.journal_status(sr.load_journal()), _held().get(ticker))
    res.update(items=items, verdict=sr.verdict(items), deep=True,
               passed=sum(i["status"] in ("pass", "na") for i in items))
    return res


def swing_levels(ticker: str, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """관심 종목 차트에 겹칠 손절·2R·4R 선 (재무 조회 없이 가격만)."""
    try:
        res = analyze(ticker, settings, deep=False)
    except Exception:  # noqa: BLE001
        return []
    return [] if res.get("error") else sr.chart_levels(res["plan"], ticker)


def items_table(items: List[Dict[str, Any]]) -> None:
    st.dataframe(pd.DataFrame({
        "#": [i["no"] for i in items], "항목": [i["name"] for i in items],
        "결과": [STATUS_KO[i["status"]] for i in items], "내용": [i["detail"] for i in items],
        "기준": [i["rule"] for i in items],
    }), hide_index=True, width="stretch",
        column_config={"#": st.column_config.NumberColumn(width="small"),
                       "내용": st.column_config.TextColumn(width="large")})


def plan_block(ticker: str, plan: Dict[str, Any], setup: Optional[str], pivot: Optional[float]) -> None:
    p = lambda v: sr.fmt_price(v, ticker)  # noqa: E731
    cols = st.columns(5)
    cols[0].metric("매수가", p(plan["entry"]))
    cols[1].metric("손절가", p(plan["stop"]), f"−{plan['stop_pct']:.1f}%", delta_color="off")
    cols[2].metric("2R · 1/3 익절", p(plan["t1"]), f"+{(plan['t1'] / plan['entry'] - 1) * 100:.1f}%", delta_color="off")
    cols[3].metric("4R · 1/3 익절", p(plan["t2"]), f"+{(plan['t2'] / plan['entry'] - 1) * 100:.1f}%", delta_color="off")
    if plan.get("qty"):
        cols[4].metric("수량 (1% 룰)", f"{plan['qty']:,}주", f"비중 {plan['weight']:.1f}%", delta_color="off")
        st.caption(f"투입 {p(plan['amount'])} · 손절 시 손실 {p(plan['loss'])} (총자산의 {plan['risk_pct']:.2f}%)"
                   + (" · 1종목 20% 상한으로 줄임" if plan["capped"] else ""))
    elif plan.get("qty") == 0:
        cols[4].metric("수량", "0주", "허용 위험 0", delta_color="off")
    else:
        cols[4].metric("수량", "-", "총자산 미입력", delta_color="off")

    watch = f"현재가 ≥ {p(pivot * 1.005)}" if setup == "W" and pivot else None
    lines = []
    if watch:
        lines.append(f"- **감시 매수:** {watch} 이면 지정가 {p(pivot * 1.03)} 이내로 매수 (기준가 +5% 넘게는 추격 금지)")
    lines += [
        f"- **손절 감시:** 현재가 ≤ {p(plan['stop'])} → **시장가** 전량 매도 (체결 즉시 등록)",
        f"- **1차 익절:** {p(plan['t1'])} 지정가로 1/3 매도 → 체결되면 손절 감시를 본전 {p(plan['breakeven'])} 로 수정",
        f"- **2차 익절:** {p(plan['t2'])} 지정가로 1/3 매도 → 손절 감시를 {p(plan['t1'])} 로 수정",
        f"- **나머지 1/3:** 트레일링 스탑 '최고가 대비 −{sr.RULES['trail_pct']:.0f}%' 또는 종가 {sr.RULES['trail_ma']}일선 이탈 시 매도",
        f"- **1차 익절 전 정리:** 종가 {sr.RULES['exit_ma']}일선 이탈, 또는 {sr.RULES['time_stop_days']}거래일 안에 "
        f"+{sr.RULES['time_stop_gain']:.0f}% 못 가면",
        f"- **아침 확인:** 시초가가 감시가보다 +{sr.RULES['chase_gap_pct']:.0f}% 넘게 갭이면 매수 주문 취소",
    ]
    with st.expander("🧾 증권사 예약 주문 세팅 (자동감시·조건주문)"):
        st.markdown("\n".join(lines))
        st.caption("증권사마다 기능 이름이 다릅니다 (키움 '자동감시주문', 토스증권 '조건주문' 등). "
                   "갭 하락으로 시작하면 손절가보다 낮게 체결될 수 있습니다.")


def chart(ticker: str, plan: Dict[str, Any], key: str) -> None:
    period = st.segmented_control("기간", list(price_chart.PERIOD_DAYS), default="6개월", key=f"{key}_p",
                                  label_visibility="collapsed") or "6개월"
    df = price_chart.prepare(hist(ticker), price_chart.PERIOD_DAYS[period])
    if df.empty:
        return
    levels = sr.chart_levels(plan, ticker)
    held = _held().get(ticker) or {}
    if held.get("held") and held.get("avg_price"):
        levels.append({"role": "avg", "label": "내 평균단가", "y": float(held["avg_price"]), "y2": None,
                       "color": price_chart.ROLE_COLOR["avg"]})
    st.altair_chart(price_chart.build_chart(df, levels), width="stretch")
    far = price_chart.hidden_levels(levels, df)
    if far:
        st.caption("차트 범위 밖: " + " · ".join(lv["label"] for lv in far))


# --------------------------------------------------------------------------- 하위 화면


def _regime_panel(settings: Dict[str, Any]) -> None:
    acct = sr.journal_status(sr.load_journal())
    cols = st.columns(3)
    for col, m in zip(cols, ("KS", "KQ", "US")):
        r = regime(m)
        with col.container(border=True):
            st.markdown(f"**{r['name']}** {r['ko']}")
            if r.get("error"):
                st.caption(r["error"])
                continue
            st.caption(f"{r['date']} 종가 {r['close']:,.0f} · {r['fast']}일선 {r['ma_fast']:,.0f} · "
                       f"{r['slow']}일선 {r['ma_slow']:,.0f}")
            risk = sr.effective_risk(settings, r, acct)
            st.markdown(f"주식 비중 최대 **{r['exposure']}%** · 1회 위험 **{risk:.2f}%**"
                        + (" · 🚫 신규 매수 중단" if risk <= 0 else ""))
    st.markdown("##### 📒 계좌 상태 (매매일지 기준)")
    c = st.columns(4)
    c[0].metric("슬럼프 단계", f"{acct['stage']}단계", f"위험 ×{acct['mult']}", delta_color="off")
    c[1].metric("보유 종목", f"{acct['open_trades']} / {sr.RULES['max_positions']}")
    c[2].metric("오픈 리스크", f"{acct['open_risk']:.1f}%", f"한도 {sr.RULES['max_open_risk']:.0f}%", delta_color="off")
    c[3].metric("이번 달 손익", f"{acct['month_pnl']:+.1f}%", f"낙폭 {acct['drawdown']:.1f}%", delta_color="off")
    for reason in acct["reasons"]:
        st.warning(reason)
    with st.expander("📏 적용 중인 규칙 수치"):
        st.markdown(
            f"- 1회 위험 {settings.get('risk_pct', 1.0)}% × 국면(상승 1 · 중립 0.5 · 하락 0) × 슬럼프 배수\n"
            f"- 1종목 최대 {sr.RULES['max_weight']:.0f}% · 동시 {sr.RULES['max_positions']}종목 · "
            f"오픈 리스크 {sr.RULES['max_open_risk']:.0f}%\n"
            f"- 손절 −{sr.RULES['stop_pct']:.0f}% (차트상 자리가 −{sr.RULES['stop_limit']:.0f}% 보다 멀면 패스) · "
            f"익절 2R·4R 에서 1/3씩 · 나머지 트레일링\n"
            f"- 돌파: 거래량 {sr.RULES['breakout_vol']:.0f}배, 기준가 +{sr.RULES['chase_pct']:.0f}% 이내 · "
            f"눌림: 20일선, 거래량 {sr.RULES['pullback_vol']:.0%} 이하 · 역추세: 25일선 이격 "
            f"{sr.RULES['reversal_dev25']:.0f}% 이하(상승 국면만, 비중 절반)\n"
            "- 슬럼프: 3연속 손절 → 절반 · 5연속 또는 낙폭 6% → 1주 중단 · 낙폭 10% → 2주 중단 후 1/4 · "
            "월 −6% → 그달 종료 · 2연속 수익마다 한 단계 회복\n"
            "- 시장 국면: 국내 지수 20·60일선, 미국 S&P500 50·200일선")


def _check_panel(settings: Dict[str, Any]) -> None:
    items = watchlist.load_watchlist()
    c1, c2, c3 = st.columns([2, 1, 1])
    options = [i["ticker"] for i in items]
    pick = c1.selectbox("관심 종목에서 고르기", options, index=0 if options else None, key="swing_pick",
                        placeholder="관심 종목 없음")
    typed = c1.text_input("또는 티커 입력", key="swing_typed", placeholder="예: 000660.KS, NVDA").strip().upper()
    ticker = typed or pick
    entry = c2.number_input("매수가 (0=자동)", min_value=0.0, value=0.0, key="swing_entry", format="%.4g")
    stop = c3.number_input("손절가 (0=자동)", min_value=0.0, value=0.0, key="swing_stop", format="%.4g")
    if st.button("🔍 점검", type="primary", disabled=not ticker, key="swing_go"):
        st.session_state["swing_target"] = ticker
    if not ticker or st.session_state.get("swing_target") != ticker:
        st.caption("종목을 고르고 '점검'을 누르세요. 매수가·손절가를 바꾸면 바로 다시 계산합니다.")
        return
    with st.spinner(f"{ticker} 점검 중 (재무·수급·공시 조회)…"):
        try:
            res = analyze(ticker, settings, entry=entry or None, stop=stop or None)
        except Exception as exc:  # noqa: BLE001
            st.error(f"점검 실패: {type(exc).__name__}: {exc}")
            return
    if res.get("error"):
        st.warning(res["error"])
        return
    st.markdown(f"### {res['verdict']} — {ticker} · 통과 {res['passed']}/10")
    st.caption(f"{res['date']} 종가 {sr.fmt_price(res['close'], ticker)} · 오늘 신호: {res['setup_ko']}"
               + ("" if entry else " · 매수가는 " + ("박스 상단 +0.5% (돌파 대기)" if res["setup"] == "W" else "오늘 종가") + " 기준"))
    items_table(res["items"])
    plan_block(ticker, res["plan"], res["setup"], res.get("pivot"))
    chart(ticker, res["plan"], key=f"chk_{ticker}")
    with st.expander("📒 이 계획으로 매수했다면 매매일지에 기록"):
        with st.form(f"add_{ticker}"):
            f1, f2, f3, f4 = st.columns(4)
            px = f1.number_input("실제 매수가", value=float(res["plan"]["entry"]), format="%.4g")
            sp = f2.number_input("손절가", value=float(res["plan"]["stop"]), format="%.4g")
            qty = f3.number_input("수량", min_value=0.0, value=float(res["plan"].get("qty") or 0), step=1.0)
            day = f4.date_input("매수일", value=date.today())
            if st.form_submit_button("기록", type="primary"):
                eq = sr.equity_for(ticker, settings)
                if not eq:
                    st.error("설정에서 총자산을 먼저 입력하세요 (손익을 계좌 대비 %로 계산하는 데 필요).")
                elif qty <= 0 or sp >= px:
                    st.error("수량은 0보다 크고, 손절가는 매수가보다 낮아야 합니다.")
                else:
                    sr.add_trade(ticker, px, sp, qty, eq, setup=res["setup"] or "", opened=str(day))
                    st.toast(f"{ticker} 매매일지에 기록", icon="📒")
                    st.rerun()


def _screen_panel(settings: Dict[str, Any]) -> None:
    held = _held()
    tickers = sr.universe(settings, held.keys())
    st.caption(f"대상 {len(tickers)}종목 (설정의 종목군 + 추가 티커 + 관심 종목). 1단계에서 가격 규칙으로 전부 거르고, "
               "신호가 난 종목만 재무·수급·공시까지 봅니다. 평일 16:10 에 자동으로도 돌립니다.")
    if st.button("🔍 지금 스크리닝", type="primary"):
        bar = st.progress(0.0, "시작")
        sr.run_screen(tickers, settings, progress=lambda f, t: bar.progress(min(1.0, f), t), held=held)
        bar.empty()
    data = sr.load_screen()
    if not data:
        st.info("아직 스크리닝 결과가 없습니다.")
        return
    st.markdown(f"**{data['created_at'].replace('T', ' ')}** 결과 · {data['count']}종목 · "
                + " · ".join(f"{g['name']} {g['ko']}" for g in data["regimes"].values()))
    rows = [r for r in data["rows"] if not r.get("error")]
    show_all = st.toggle("신호 없는 종목도 보기", key="screen_all")
    shown = rows if show_all else [r for r in rows if r.get("setup")]
    order = {"✅ 매수 후보": 0, "⚠️ 확인 필요": 1, "❌ 보류": 2, "—": 3}
    shown.sort(key=lambda r: (order.get(r["verdict"], 3), -r["passed"]))
    if not shown:
        st.info("오늘은 규칙에 맞는 신호가 난 종목이 없습니다.")
    else:
        st.dataframe(pd.DataFrame({
            "판정": [r["verdict"] for r in shown], "종목": [r["ticker"] for r in shown],
            "이름": [(held.get(r["ticker"]) or {}).get("name", "") for r in shown],
            "신호": [r["setup_ko"] for r in shown], "통과": [f"{r['passed']}/10" for r in shown],
            "종가": [sr.fmt_price(r["close"], r["ticker"]) for r in shown],
            "매수가": [sr.fmt_price(r["plan"]["entry"], r["ticker"]) for r in shown],
            "손절가": [f"{sr.fmt_price(r['plan']['stop'], r['ticker'])} (−{r['plan']['stop_pct']:.1f}%)" for r in shown],
            "2R": [sr.fmt_price(r["plan"]["t1"], r["ticker"]) for r in shown],
            "탈락 항목": [", ".join(i["name"] for i in r["items"] if i["status"] == "fail") or "-" for r in shown],
        }), hide_index=True, width="stretch")
        for r in shown[:15]:
            if r.get("setup"):
                with st.expander(f"{r['verdict']} {r['ticker']} — {r['setup_ko']} · 통과 {r['passed']}/10"):
                    items_table(r["items"])
    errors = [r for r in data["rows"] if r.get("error")]
    if errors:
        st.caption("데이터를 못 가져온 종목: " + ", ".join(r["ticker"] for r in errors))


def _journal_panel(settings: Dict[str, Any]) -> None:
    trades = sr.load_journal()
    acct = sr.journal_status(trades)
    st.caption(f"슬럼프 {acct['stage']}단계 · 오픈 리스크 {acct['open_risk']:.1f}% · 낙폭 {acct['drawdown']:.1f}% · "
               f"이번 달 {acct['month_pnl']:+.1f}% — 이 값들이 10계명 10번과 1회 위험 계산에 쓰입니다.")
    open_t = [t for t in trades if not t.get("closed")]
    if open_t:
        st.markdown("##### 보유 중")
        rows = []
        for t in open_t:
            try:
                last = float(hist(t["ticker"])["Close"].iloc[-1])
            except Exception:  # noqa: BLE001
                last = None
            r = t["entry"] - t["stop0"]
            rows.append({"종목": t["ticker"], "매수일": t["opened"], "매수가": sr.fmt_price(t["entry"], t["ticker"]),
                         "남은 수량": sr.remaining_qty(t), "손절가": sr.fmt_price(t["stop"], t["ticker"]),
                         "현재가": sr.fmt_price(last, t["ticker"]),
                         "현재 R": round((last - t["entry"]) / r, 2) if last and r > 0 else None,
                         "2R": sr.fmt_price(t["entry"] + 2 * r, t["ticker"]), "4R": sr.fmt_price(t["entry"] + 4 * r, t["ticker"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with st.form("journal_exit"):
            st.markdown("**매도 기록 / 손절가 수정**")
            f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
            pick = f1.selectbox("거래", open_t, format_func=lambda t: f"{t['ticker']} · {t['opened']} · {t['entry']:g}")
            px = f2.number_input("매도가 또는 새 손절가", min_value=0.0, format="%.4g")
            qty = f3.number_input("매도 수량", min_value=0.0, step=1.0)
            day = f4.date_input("날짜", value=date.today())
            b1, b2, b3 = st.columns(3)
            sell = b1.form_submit_button("매도 기록", type="primary")
            move = b2.form_submit_button("손절가 수정")
            drop = b3.form_submit_button("거래 삭제")
            if sell and px > 0 and qty > 0:
                sr.add_exit(pick["id"], px, qty, str(day))
                st.rerun()
            if move and px > 0:
                sr.update_stop(pick["id"], px)
                st.rerun()
            if drop:
                sr.delete_trade(pick["id"])
                st.rerun()
    closed = [t for t in trades if t.get("closed")]
    if closed:
        st.markdown("##### 끝난 거래")
        st.dataframe(pd.DataFrame([{
            "종목": t["ticker"], "매수일": t["opened"], "청산일": t.get("closed_at"),
            "매수가": sr.fmt_price(t["entry"], t["ticker"]), "R": round(sr.trade_result(t)["r"] or 0, 2),
            "손익": sr.fmt_price(sr.trade_result(t)["pnl"], t["ticker"]),
            "계좌 대비": f"{sr.trade_result(t)['impact_pct']:+.2f}%",
        } for t in sorted(closed, key=lambda x: x.get("closed_at", ""), reverse=True)]), hide_index=True, width="stretch")
    if not trades:
        st.info("기록된 거래가 없습니다. '🔍 종목 점검 10계명'에서 매수 후 기록하면 오픈 리스크와 슬럼프 단계가 계산됩니다.")


def _backtest_panel(settings: Dict[str, Any]) -> None:
    st.caption("가격 규칙(국면·추세·돌파/눌림·손절·분할 익절·트레일링·시간 손절)과 자금 규칙(1% 룰·20%·6종목·오픈 리스크 5%·슬럼프)"
               "을 과거 일봉에 적용합니다. 재무·수급·공시 필터는 과거 값을 구할 수 없어 빠져 있고, 지금 상장된 종목만 쓰므로 "
               "결과가 실제보다 좋게 나올 수 있습니다(생존 편향). 수수료·세금은 국내 0.25%, 미국 0.1% 왕복 반영.")
    c1, c2, c3 = st.columns([2, 1, 1])
    groups = c1.multiselect("종목군", list(sr.UNIVERSES) + ["관심 종목"], default=["코스피 대형주"], key="bt_groups")
    extra = c1.text_input("추가 티커 (쉼표)", key="bt_extra")
    years = c2.slider("기간(년)", 1, 10, 5, key="bt_years")
    setups = c3.multiselect("셋업", ["A", "B", "C"], default=["A", "B"], format_func=lambda s: sr.SETUP_KO[s],
                            key="bt_setups")
    slump = c3.toggle("슬럼프 규칙 적용", value=True, key="bt_slump")
    tickers = []
    for g in groups:
        tickers += [i["ticker"] for i in watchlist.load_watchlist()] if g == "관심 종목" else sr.UNIVERSES[g]
    tickers += [t.strip().upper() for t in extra.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))
    if st.button(f"🧪 백테스트 실행 ({len(tickers)}종목)", type="primary", disabled=not tickers or not setups):
        bar = st.progress(0.0)
        st.session_state["bt_result"] = swing_backtest.run(
            tickers, years=years, setups=setups, use_slump=slump, progress=lambda f, t: bar.progress(min(1.0, f), t))
        bar.empty()
    res = st.session_state.get("bt_result")
    if not res:
        return
    p, s = res["portfolio"], res["signals"]
    st.markdown(f"**{res['start']} ~ {res['end']}** · {res['tickers']}종목 · 신호 {s.get('n', 0)}건 중 실제 매매 {p.get('n', 0)}건")
    if not p.get("n"):
        st.info("기간 안에 규칙을 통과한 매매가 없습니다.")
        return
    m = st.columns(6)
    m[0].metric("총수익률", f"{p['total_pct']:+.1f}%")
    m[1].metric("연환산", f"{p['cagr_pct']:+.1f}%")
    m[2].metric("최대 낙폭", f"−{p['mdd_pct']:.1f}%")
    m[3].metric("승률", f"{p['win_rate']:.0f}%")
    m[4].metric("기대값", f"{p['expectancy_r']:+.2f}R")
    m[5].metric("손익비(PF)", f"{p['profit_factor']:.2f}" if p.get("profit_factor") else "-")
    st.caption(f"평균 수익 {p['avg_win_r']:+.2f}R · 평균 손실 {p['avg_loss_r']:+.2f}R · 평균 보유 {p['avg_days']:.0f}일 · "
               "건너뛴 신호: " + ", ".join(f"{k} {v}건" for k, v in p["skipped"].items() if v) or "없음")
    curve = pd.DataFrame(res["curve"], columns=["날짜", "자산"])
    curve["날짜"] = pd.to_datetime(curve["날짜"])
    st.altair_chart(alt.Chart(curve).mark_line(interpolate="step-after").encode(
        x=alt.X("날짜:T", title=None), y=alt.Y("자산:Q", title="자산 (시작 100)", scale=alt.Scale(zero=False)),
        tooltip=["날짜:T", alt.Tooltip("자산:Q", format=".1f")]).properties(height=240), width="stretch")
    c1, c2 = st.columns(2)
    c1.markdown("**셋업별 (자금 규칙 적용 전 모든 신호)**")
    c1.dataframe(pd.DataFrame([{"셋업": sr.SETUP_KO[k], "건수": v["n"], "승률%": round(v.get("win_rate", 0)),
                                "기대값R": round(v.get("expectancy_r", 0), 2), "PF": round(v["profit_factor"], 2)
                                if v.get("profit_factor") else None} for k, v in res["by_setup"].items()]),
                 hide_index=True, width="stretch")
    c2.markdown("**연도별 (실제 매매)**")
    c2.dataframe(pd.DataFrame([{"연도": y, "건수": v["n"], "승률%": round(v.get("win_rate", 0)),
                                "계좌 손익%": round(v["pnl_pct"], 1)} for y, v in res["by_year"].items()]),
                 hide_index=True, width="stretch")
    with st.expander(f"모든 신호 {len(res['trades'])}건"):
        st.dataframe(pd.DataFrame(res["trades"]), hide_index=True, width="stretch")
    if res["errors"]:
        st.caption("데이터 부족으로 뺀 종목: " + ", ".join(res["errors"]))


def _settings_panel(settings: Dict[str, Any]) -> None:
    with st.form("swing_settings"):
        c1, c2, c3 = st.columns(3)
        krw = c1.number_input("국내 계좌 총자산 (원)", min_value=0.0, value=float(settings.get("equity_krw") or 0),
                              step=1_000_000.0, format="%.0f")
        usd = c2.number_input("미국 계좌 총자산 ($)", min_value=0.0, value=float(settings.get("equity_usd") or 0),
                              step=1000.0, format="%.0f")
        risk = c3.select_slider("1회 위험 (총자산 대비 %)", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                                value=float(settings.get("risk_pct") or 1.0))
        groups = st.multiselect("스크리닝 종목군", list(sr.UNIVERSES), default=settings.get("universes") or [])
        extra = st.text_input("추가 티커 (쉼표)", value=", ".join(settings.get("extra_tickers") or []))
        auto = st.toggle("평일 16:10 자동 스크리닝", value=bool(settings.get("auto_screen", True)))
        if st.form_submit_button("저장", type="primary"):
            settings.update(equity_krw=krw or None, equity_usd=usd or None, risk_pct=risk, universes=groups,
                            extra_tickers=[t.strip().upper() for t in extra.split(",") if t.strip()], auto_screen=auto)
            sr.save_settings(settings)
            st.toast("스윙 설정 저장", icon="✅")
            st.rerun()
    st.caption("총자산은 이 PC 의 data/swing/settings.json 에만 저장됩니다. 수량 = 총자산 × 1회 위험 ÷ (매수가 − 손절가).")


def render() -> None:
    settings = sr.load_settings()
    st.markdown("거장들의 원칙(미너비니·다바스·오닐·BNF·남석관·성현우)을 숫자 규칙으로 옮긴 스윙 매매 도구입니다. "
                "**판단 보조용**이며 실제 매매 결정은 직접 하셔야 합니다.")
    if not settings.get("equity_krw") and not settings.get("equity_usd"):
        st.info("⚙️ 설정에서 총자산을 입력하면 1% 룰 수량이 계산됩니다.")
    # 열린 하위 탭만 실행 (나머지까지 매번 그리면 화면이 느려짐)
    tabs = st.tabs(["🧭 시장 국면", "🔍 종목 점검 10계명", "📋 스크리너", "📒 매매일지", "🧪 백테스트", "⚙️ 설정"],
                   key="swing_tab", on_change="rerun")
    panels = (_regime_panel, _check_panel, _screen_panel, _journal_panel, _backtest_panel, _settings_panel)
    for tab, panel in zip(tabs, panels):
        if not tab.open:
            continue
        with tab:
            try:
                panel(settings)
            except Exception as exc:  # noqa: BLE001
                st.error(f"화면을 그리지 못했습니다: {type(exc).__name__}: {exc}")
