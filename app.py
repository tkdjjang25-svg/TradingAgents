"""TradingAgents Streamlit UI (한글).

실행 (로그인 게이트웨이를 통해서만 접속 가능):
    .venv\\Scripts\\python.exe auth_gateway.py serve

분석은 서버의 백그라운드 스레드에서 돌기 때문에, 실행 중에 화면을 조작하거나
브라우저를 닫아도 중단되지 않는다. 끝난 분석은 history_store가 자동 저장한다.
"""
import hmac
import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# .env에서 API 키 로드 (tradingagents import 전에 수행 — 일부 클라이언트는
# import 시점에 환경변수를 읽음). override=True로 .env 변경이 항상 반영되도록 함.
load_dotenv(override=True)

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

import entry_plans
import history_store
import news_event_agent
import price_chart
import swing_rules
import swing_ui
import telegram_collector
import job_queue
import ticker_catalog
import watchlist
from tradingagents.agents.utils.agent_utils import normalize_position

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TradingAgents",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# 접근 제어: 로그인 게이트웨이(auth_gateway.py)를 거친 요청만 받는다.
# 게이트웨이는 실행할 때마다 새 비밀값을 TA_GATEWAY_SECRET 으로 넘기고, 모든 요청에
# X-TA-Gateway 헤더로 붙인다. 이 값이 없거나 다르면 (예: 같은 PC 에서 Streamlit 포트로
# 직접 접속) 화면을 보여주지 않는다.
# ---------------------------------------------------------------------------
def _gateway_session() -> str:
    expected = os.environ.get("TA_GATEWAY_SECRET", "")
    headers = st.context.headers
    sent = headers.get("X-TA-Gateway", "") or ""
    if not expected or not hmac.compare_digest(sent.encode(), expected.encode()):
        st.error("🔒 로그인 게이트웨이를 통해서만 접속할 수 있습니다. 관리자에게 접속 주소를 문의하세요.")
        st.stop()
    return headers.get("X-TA-Session", "") or ""


SESSION_ID = _gateway_session()
log = logging.getLogger("app")

st.title("📈 TradingAgents")
# 로그아웃은 게이트웨이의 /logout (세션 폐기 + 쿠키 삭제) 으로
st.sidebar.link_button("🔓 로그아웃", "/logout", width="stretch")
st.caption("LangGraph + LLM 기반 멀티에이전트 트레이딩 분석")


# ---------------------------------------------------------------------------
# 분석 대기열 (job_queue) — 서버 전체가 공유, 한 번에 한 건씩 순서대로 실행
# ---------------------------------------------------------------------------
_mmss = job_queue.mmss


@st.cache_resource
def job_slot() -> dict:
    """서버 프로세스 전체가 공유하는 실행 슬롯 (한 번에 분석 1건)."""
    return {"job": None}


# 대기열 도입 전 방식으로 시작된 분석. 끝날 때까지 대기열은 다음 작업을 시작하지 않는다.
legacy_slot = job_slot()


def _legacy_running() -> bool:
    job = legacy_slot["job"]
    return job is not None and getattr(job, "status", None) == "running"


queue = job_queue.shared_queue(
    lambda: job_queue.JobQueue(
        runner=lambda job: job_queue.run_job(job, TradingAgentsGraph), busy=_legacy_running,
        inbox=job_queue.INBOX_DIR,
    )
)


# ---------------------------------------------------------------------------
# 사이드바 — 설정 폼
# ---------------------------------------------------------------------------
# Claude Code 헤드리스(claude -p) — API 키 없이 claude.ai 구독 로그인으로 호출.
# 별칭은 설치된 CLI가 정한 최신 모델로 연결된다 (CLI를 업데이트하면 따라 바뀜).
CLAUDE_CODE_DEEP = ["opus", "sonnet"]
CLAUDE_CODE_QUICK = ["sonnet", "haiku", "opus"]
_ALIASES = {"opus", "sonnet", "haiku"}


def _cli_stamp() -> float:
    """설치된 CLI 파일의 수정 시각. CLI를 업데이트하면 바뀌어 아래 캐시가 새로 확인된다."""
    from tradingagents.llm_clients.claude_code_client import resolve_claude_bin

    try:
        return Path(resolve_claude_bin()).stat().st_mtime
    except Exception:
        return 0.0


@st.cache_data(ttl=6 * 3600, show_spinner="CLI 모델 확인 중…")
def _resolve_alias(alias: str, cli_stamp: float):
    from tradingagents.llm_clients.claude_code_client import resolve_model_id

    return resolve_model_id(alias)


def claude_model_id(alias: str):
    """별칭(opus 등)이 지금 설치된 CLI에서 가리키는 실제 모델 ID."""
    return _resolve_alias(alias, _cli_stamp())


def pretty_model_id(model_id: str) -> str:
    """claude-opus-4-8 → Opus 4.8, claude-haiku-4-5-20251001 → Haiku 4.5"""
    parts = model_id.removeprefix("claude-").split("-")
    nums = [p for p in parts[1:] if p.isdigit() and len(p) < 4]
    return f"{parts[0].capitalize()} {'.'.join(nums)}".strip()


def claude_full_id(model: str) -> str:
    return (claude_model_id(model) or "확인 실패") if model in _ALIASES else model


def claude_model_label(model: str) -> str:
    resolved = claude_model_id(model) if model in _ALIASES else model
    return pretty_model_id(resolved) if resolved else model.capitalize()


ANTHROPIC_DEEP = ["claude-sonnet-4-6", "claude-opus-4-6", "claude-sonnet-4-5", "claude-opus-4-5"]
ANTHROPIC_QUICK = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-sonnet-4-5"]
OPENAI_DEEP = ["gpt-5.4", "gpt-5.4-mini", "o1"]
OPENAI_QUICK = ["gpt-5.4-mini", "gpt-5.4"]
# Google Gemini — model_catalog.py 기준 유효 모델명. AI Studio 무료티어는 2.5-flash 위주.
GOOGLE_DEEP = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.1-pro-preview", "gemini-3-flash-preview"]
GOOGLE_QUICK = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview"]

LANGUAGE_OPTIONS = {
    "한국어": "Korean",
    "English": "English",
    "日本語": "Japanese",
    "中文": "Chinese",
}

@st.cache_data(ttl=3600, show_spinner=False)
def search_tickers(query: str) -> list:
    return ticker_catalog.search(query)


def ticker_options(query: str) -> dict:
    """드롭다운 후보: 검색 결과 → 최근 분석한 종목 → 자주 쓰는 종목 순서."""
    names = dict(ticker_catalog.POPULAR)
    recent = [(r["ticker"], " · ".join(p for p in (names.get(r["ticker"]), "최근 분석") if p)) for r in history_store.list_runs()]
    return ticker_catalog.merge(search_tickers(query) if query else [], recent, ticker_catalog.POPULAR)


with st.sidebar:
    st.header("⚙️ 설정")

    query = st.text_input(
        "🔎 종목 검색",
        placeholder="영문 회사명·티커 일부 입력 후 Enter",
        help="Yahoo Finance에서 찾습니다 (예: samsung, bitcoin, korea). 한글 이름은 아래 목록에서 바로 입력해 고르세요.",
    ).strip()
    options = ticker_options(query)
    if query and not search_tickers(query):
        st.caption("검색 결과가 없습니다. 아래 목록에서 고르거나 티커를 직접 입력하세요.")
    picked = st.selectbox(
        "종목 티커",
        list(options),
        index=0,  # 검색 첫 결과, 없으면 가장 최근에 분석한 종목
        format_func=lambda s: f"{s} — {options[s]}" if options.get(s) else s,
        accept_new_options=True,
        key=f"ticker::{query}",  # 새로 검색하면 첫 결과가 선택된다
        help="입력하면 후보가 걸러집니다 (티커·영문·한글 이름). 목록에 없는 티커도 그대로 입력할 수 있습니다. "
        "예: NVDA, 005930.KS, BTC-USD",
    )
    ticker = ticker_catalog.normalize(picked or "")
    if not ticker_catalog.is_valid_ticker(ticker):
        st.error("올바른 티커가 아닙니다. 목록에서 고르거나 위 검색을 이용하세요 (예: 005930.KS).")
    trade_date = st.date_input(
        "분석 기준일",
        value=date.today(),
        max_value=date.today(),
        help="이 날짜를 '오늘'로 보고 분석합니다. 시세는 기준일 전날 종가까지 반영됩니다 "
        "(오늘로 두면 가장 최근 종가까지). 과거 날짜를 고르면 그 시점 기준으로 분석합니다.",
    )

    st.divider()
    st.subheader("🤖 LLM 공급자")
    provider = st.selectbox(
        "Provider",
        ["claude-code", "anthropic", "openai", "google"],
        index=0,
        format_func=lambda p: {
            "claude-code": "Claude Code (구독 로그인 · API 키 불필요)",
            "anthropic": "Anthropic API",
            "openai": "OpenAI API",
            "google": "Google Gemini API",
        }[p],
        help="Claude Code는 터미널에서 `claude auth login`으로 로그인된 구독을 사용합니다. "
             "나머지는 .env에 해당 API 키가 설정되어 있어야 합니다.",
    )

    if provider == "claude-code":
        deep_model = st.selectbox("심층 사고 모델 (Deep)", CLAUDE_CODE_DEEP, index=0,
                                   format_func=claude_model_label,
                                   help="복잡한 추론·최종 의사결정에 사용. 설치된 Claude Code CLI가 정한 최신 모델을 씁니다 "
                                        "(CLI를 업데이트하면 따라 바뀜).")
        quick_model = st.selectbox("빠른 사고 모델 (Quick)", CLAUDE_CODE_QUICK, index=0,
                                    format_func=claude_model_label,
                                    help="도구 호출·간단한 판단에 사용. Haiku가 사용량 한도 소모가 가장 적음")
        st.caption(f"실제 모델 ID — 심층 `{claude_full_id(deep_model)}` · 빠른 `{claude_full_id(quick_model)}`")
        st.caption("구독 사용량 한도에서 차감됩니다. 호출마다 CLI를 띄우므로 API 방식보다 다소 느립니다.")
    elif provider == "anthropic":
        deep_model = st.selectbox("심층 사고 모델 (Deep)", ANTHROPIC_DEEP, index=0,
                                   help="복잡한 추론·최종 의사결정에 사용")
        quick_model = st.selectbox("빠른 사고 모델 (Quick)", ANTHROPIC_QUICK, index=0,
                                    help="도구 호출·간단한 판단에 사용")
    elif provider == "openai":
        deep_model = st.selectbox("심층 사고 모델 (Deep)", OPENAI_DEEP, index=0)
        quick_model = st.selectbox("빠른 사고 모델 (Quick)", OPENAI_QUICK, index=0)
    else:  # google
        deep_model = st.selectbox("심층 사고 모델 (Deep)", GOOGLE_DEEP, index=1,
                                   help="AI Studio 무료티어는 gemini-2.5-flash 권장")
        quick_model = st.selectbox("빠른 사고 모델 (Quick)", GOOGLE_QUICK, index=0)

    st.divider()
    st.subheader("🌐 출력 언어")
    lang_label = st.selectbox(
        "분석 리포트 언어",
        list(LANGUAGE_OPTIONS.keys()),
        index=0,
        help="최종 리포트와 의사결정의 언어. 내부 에이전트 토론은 추론 품질을 위해 영어로 진행됩니다.",
    )
    output_language = LANGUAGE_OPTIONS[lang_label]

    st.divider()
    st.subheader("👥 분석가 선택")
    analysts = []
    if st.checkbox("📊 시장 분석가 (기술적 분석)", value=True):
        analysts.append("market")
    if st.checkbox("💬 소셜미디어 분석가", value=True):
        analysts.append("social")
    if st.checkbox("📰 뉴스 분석가", value=True):
        analysts.append("news")
    if st.checkbox("🏢 펀더멘털 분석가", value=True):
        analysts.append("fundamentals")

    st.divider()
    st.subheader("💬 토론 횟수")
    debate_rounds = st.slider("투자 토론 라운드 (Bull vs Bear)", 1, 5, 1,
                               help="값이 클수록 분석은 깊어지지만 시간·비용 증가")
    risk_rounds = st.slider("리스크 토론 라운드", 1, 5, 1)

    st.divider()
    # 보유 여부·평균단가도 종목별로 기억한다. 입력하지 않으면 '미보유' 전제로 결정이 나온다.
    positions_by_ticker = st.session_state.setdefault("positions_by_ticker", {})
    saved_pos = positions_by_ticker.get(ticker, {"held": False, "avg_price": None})
    held_label = st.radio(
        f"💼 보유 여부 — {ticker}",
        ["미보유", "보유 중"],
        index=1 if saved_pos["held"] else 0,
        key=f"held::{ticker}",
        horizontal=True,
        help="결정을 내 상황 기준으로 받습니다. 미보유면 '지금 살지 말지', 보유 중이면 '더 살지·유지할지·팔지'로 답합니다.",
    )
    held = held_label == "보유 중"
    avg_price = None
    if held:
        avg_price = st.number_input(
            "평균단가 (선택 · 종목 표시 통화 기준)",
            min_value=0.0,
            value=float(saved_pos["avg_price"] or 0.0),
            step=1.0,
            format="%.4g",
            key=f"avg::{ticker}",
            help="0이면 평균단가 없이 '보유 중'만 전달합니다. 손익 언급에만 쓰이고 매수가에 얽매인 판단은 하지 않게 지시합니다.",
        ) or None
    positions_by_ticker[ticker] = {"held": held, "avg_price": avg_price}

    # 메모는 종목별로 따로 기억한다 (삼성전자용 메모가 다른 종목 분석에 섞이지 않게).
    notes_by_ticker = st.session_state.setdefault("notes_by_ticker", {})
    notes = st.text_area(
        f"📝 추가 참고 메모 — {ticker} (선택)",
        value=notes_by_ticker.get(ticker, ""),
        key=f"notes::{ticker}",
        max_chars=4000,
        height=120,
        placeholder="예: 2027-01-12 증여세 평가기간 종료, 1월 말 주주환원 정책 발표 예정 …",
        help=(
            "데이터 도구가 못 가져오는 일정·이슈를 적으면 분석가·토론자·매니저 모두에게 전달됩니다. "
            "에이전트는 이 내용을 '사용자 제공·미검증' 정보로 보고 반영 여부를 밝힙니다. "
            "틀린 정보를 넣으면 결론이 그쪽으로 기울 수 있으니 출처가 확실한 내용만 적어 주세요."
        ),
    )
    notes_by_ticker[ticker] = notes
    requester = st.text_input(
        "🙋 요청자 이름 (선택)", key="requester", max_chars=20, placeholder="예: 민수",
        help="여러 사람이 같이 쓸 때 대기열에서 누구 작업인지 표시합니다.",
    )
    q_running, q_waiting, _ = queue.snapshot()
    ahead = len(q_waiting) + (1 if q_running or _legacy_running() else 0)
    run_button = st.button(
        f"➕ 대기열에 추가 (앞에 {ahead}건)" if ahead else "🚀 분석 실행",
        type="primary",
        width="stretch",
        help="분석은 서버에서 한 번에 한 건씩 순서대로 실행됩니다. 창을 닫아도 계속됩니다.",
    )


# ---------------------------------------------------------------------------
# 결과 렌더링
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def _ohlc(ticker: str, days: int) -> pd.DataFrame:
    return price_chart.load_ohlc(ticker, days)


def _render_chart(ticker: str, plan: dict | None, avg_price: float | None, key: str,
                  extra_levels: list | None = None, legend: bool = True) -> None:
    """일봉 캔들 차트 + 분석의 가격 조건."""
    period = st.segmented_control("기간", list(price_chart.PERIOD_DAYS), default="6개월",
                                  key=f"{key}_period", label_visibility="collapsed") or "6개월"
    try:
        df = _ohlc(ticker, price_chart.PERIOD_DAYS[period])
    except Exception as exc:
        st.caption(f"차트 데이터를 불러오지 못했습니다 ({type(exc).__name__}).")
        return
    if df.empty:
        st.caption("차트로 그릴 가격 데이터가 없습니다.")
        return
    levels = price_chart.levels_from_plan(plan, avg_price) + list(extra_levels or [])
    st.altair_chart(price_chart.build_chart(df, levels), width="stretch")
    far = price_chart.hidden_levels(levels, df)
    if far:
        st.caption("차트 범위 밖(현재가에서 35% 넘게 떨어진) 조건: "
                   + " · ".join(f"{lv['label']}" for lv in far))
    st.caption(f"마지막 종가 {df['d'].iloc[-1]} 기준 · 상승 빨강 / 하락 파랑 · "
               "초록=진입·추가 매수, 보라=손절, 주황=목표가, 회색=폐기·보류, 청록=내 평균단가")
    if legend and levels:
        st.markdown("**선 설명** · 차트에는 짧은 이름만 적었습니다")
        st.dataframe(pd.DataFrame(price_chart.legend_rows(levels)), hide_index=True, width="stretch")


def _render_entry_plan(record: dict, key_prefix: str) -> None:
    """최종 의사결정 본문에서 뽑은 진입·청산 가격 조건과 차트."""
    st.markdown("##### 🎯 진입·청산 조건")
    plan = entry_plans.load_plan(record["id"])
    if plan is None:
        st.caption("가격 조건을 아직 정리하지 않았습니다. 관심 종목 점검 프로세스가 최근 분석부터 한 건씩 "
                   "정리하니 몇 분 뒤 새로고침해 주세요.")
    else:
        st.markdown(f"**{plan.get('summary', '')}**")
        check = plan.get("last_check") or {}
        status = {i["label"]: i for i in check.get("triggers", [])}
        if plan["triggers"]:
            st.dataframe(pd.DataFrame({
                "구분": [entry_plans.ROLE_KO.get(t["role"], t["role"]) for t in plan["triggers"]],
                "조건": [t["label"] for t in plan["triggers"]],
                "가격": [entry_plans.level_text(t) for t in plan["triggers"]],
                "연속 종가": [f"{t['days']}일" for t in plan["triggers"]],
                "함께 볼 것": [t.get("other_conditions") or "-" for t in plan["triggers"]],
                "현재": [
                    ("✅ 충족" if status[t["label"]]["met"] else f"{status[t['label']]['distance']:+.1f}%")
                    if t["label"] in status and status[t["label"]]["distance"] is not None else "-"
                    for t in plan["triggers"]
                ],
            }), hide_index=True, width="stretch")
        if plan.get("checks"):
            st.caption("먼저 확인할 일: " + " · ".join(plan["checks"]))
        if check:
            st.caption(f"관심 종목 점검 {check.get('checked_at', '').replace('T', ' ')} · "
                       f"종가 {check.get('last_close')} ({check.get('last_date')}) · {check.get('status_ko')}")
    pos = history_store.position_of(record)
    with st.expander("📈 차트에서 보기", expanded=plan is not None and bool(plan.get("triggers")),
                     key=f"{key_prefix}_{record['id']}_exp", on_change="rerun") as chart_box:
        if chart_box.open:
            _render_chart(record["ticker"], plan, pos.get("avg_price"), key=f"{key_prefix}_{record['id']}", legend=False)


def render_record(record: dict, key_prefix: str) -> None:
    """저장된 분석 1건(history_store 레코드)을 렌더링."""
    s = record["state"]

    meta = [f"**종목** `{record['ticker']}`", f"**기준일** `{record['trade_date']}`"]
    if record.get("provider"):
        meta.append(f"**모델** `{record['provider']}` · 심층 `{record.get('deep_model')}` · 빠른 `{record.get('quick_model')}`")
        if record.get("model_ids"):
            meta.append("**실제 모델** " + ", ".join(f"`{m}`" for m in record["model_ids"]))
    if record.get("analysts"):
        meta.append(f"**분석가** {history_store.analysts_ko(record['analysts'])}")
    if record.get("elapsed_sec"):
        meta.append(f"**소요** {_mmss(record['elapsed_sec'])} · LLM {record.get('llm_calls')}회")
    meta.append(f"**저장** {record.get('saved_at', '').replace('T', ' ')}")
    st.markdown(" · ".join(meta))
    if record.get("notes"):
        with st.expander("📝 추가 참고 메모 (사용자 제공 · 미검증) — 이 내용이 에이전트에게 전달됨"):
            st.text(record["notes"])
    if record.get("warning"):
        st.warning(record["warning"], icon="⚠️")

    st.subheader("🎯 최종 거래 의사결정")
    decision = (record.get("decision") or "").strip().capitalize()
    if decision in history_store.DECISION_KO:
        ko, desc = history_store.DECISION_KO[decision]
        box = {"Buy": st.success, "Overweight": st.success, "Hold": st.info}.get(decision, st.error)
        box(f"**{ko}** ({decision}) — {desc}")
        action = history_store.action_ko(record)
        st.markdown(f"💼 **포지션** {history_store.position_ko(record)} → 👉 **할 일: {action}**")
        if "position" not in record:
            st.caption("보유 여부를 입력하기 전에 실행한 분석이라 본문은 '보유 중' 전제로 쓰였을 수 있습니다. "
                       "위의 '할 일'은 미보유 기준으로 풀어 쓴 것입니다.")
    else:
        st.info(record.get("decision") or "(없음)")
    _render_entry_plan(record, key_prefix)
    with st.expander("최종 의사결정 원문 보기"):
        st.markdown(s["final_trade_decision"] or "(없음)")
    if record.get("step_times"):
        sm = history_store.step_summary(record["step_times"])
        share = sm["analysts"] / sm["total"] * 100 if sm["total"] else 0
        with st.expander(f"⏱ 단계별 소요 시간 — 분석가 구간 {_mmss(sm['analysts'])} ({share:.0f}%) · "
                         f"나머지 {_mmss(sm['rest'])}"):
            st.dataframe(
                pd.DataFrame({
                    "단계": [x["step"] for x in record["step_times"]],
                    "시작 (초)": [x["start"] for x in record["step_times"]],
                    "소요 (초)": [x["sec"] for x in record["step_times"]],
                    "LLM 호출": [x["calls"] for x in record["step_times"]],
                }),
                hide_index=True, width="stretch",
                column_config={"소요 (초)": st.column_config.ProgressColumn(
                    format="%.1f", min_value=0, max_value=max(x["sec"] for x in record["step_times"]) or 1)},
            )
            st.caption("단계 시간 = 그 에이전트의 LLM 호출 + 호출 사이 데이터 도구 실행 시간. "
                       "토론 단계는 라운드마다 다시 나옵니다.")

    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button(
        "📄 마크다운으로 내려받기", history_store.to_markdown(record),
        file_name=f"{record['id']}.md", mime="text/markdown", key=f"{key_prefix}_md",
    )
    c2.download_button(
        "🗂 JSON으로 내려받기", json.dumps(record, ensure_ascii=False, indent=2),
        file_name=f"{record['id']}.json", mime="application/json", key=f"{key_prefix}_json",
    )

    st.subheader("📑 분석가 리포트")
    tab_market, tab_social, tab_news, tab_funda, tab_debate, tab_trader, tab_risk = st.tabs(
        ["시장", "소셜", "뉴스", "펀더멘털", "🐂🐻 토론", "트레이더 계획", "리스크 토론"]
    )

    with tab_market:
        st.markdown(s["market_report"] or "_(시장 분석가가 선택되지 않았습니다)_")
    with tab_social:
        st.markdown(s["sentiment_report"] or "_(소셜 분석가가 선택되지 않았습니다)_")
    with tab_news:
        st.markdown(s["news_report"] or "_(뉴스 분석가가 선택되지 않았습니다)_")
    with tab_funda:
        st.markdown(s["fundamentals_report"] or "_(펀더멘털 분석가가 선택되지 않았습니다)_")

    with tab_debate:
        debate = s["investment_debate_state"]
        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.markdown("### 🐂 강세 연구원 (Bull)")
            st.markdown(debate["bull_history"] or "_(없음)_")
        with col_bear:
            st.markdown("### 🐻 약세 연구원 (Bear)")
            st.markdown(debate["bear_history"] or "_(없음)_")
        st.markdown("### ⚖️ 리서치 매니저 판단")
        st.markdown(debate["judge_decision"] or "_(없음)_")

    with tab_trader:
        st.markdown(s["trader_investment_plan"] or "_(트레이더 계획 없음)_")

    with tab_risk:
        risk = s["risk_debate_state"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🔥 공격적 입장")
            st.markdown(risk["aggressive_history"] or "_(없음)_")
        with c2:
            st.markdown("### 🛡️ 보수적 입장")
            st.markdown(risk["conservative_history"] or "_(없음)_")
        with c3:
            st.markdown("### ⚖️ 중립적 입장")
            st.markdown(risk["neutral_history"] or "_(없음)_")
        st.markdown("### 📋 포트폴리오 매니저 최종 결정")
        st.markdown(risk["judge_decision"] or "_(없음)_")


def _params_line(p: dict) -> str:
    return (
        f"**종목** `{p['ticker']}` · **기준일** `{p['trade_date']}` · **모델** `{p['provider']}` · "
        f"심층 `{p['deep_model']}` · 빠른 `{p['quick_model']}` · **분석가** {history_store.analysts_ko(p['analysts'])} · "
        f"**토론** {p['debate_rounds']}/{p['risk_rounds']}라운드 · **언어** {p['language']}"
        + f" · **💼 포지션** {history_store.position_ko(p)}"
        + (f" · **📝 메모** {len(p['notes'])}자" if p.get("notes") else "")
    )


def _who(job) -> str:
    return getattr(job, "requester", "") or "이름 없음"


def _clock(ts) -> str:
    return time.strftime("%H:%M", time.localtime(ts)) if ts else "-"


def _finished_jobs() -> list:
    """대기열에서 끝난 작업 + (있다면) 대기열 도입 전 방식으로 끝난 작업, 최신순."""
    jobs = queue.snapshot()[2]
    legacy = legacy_slot["job"]
    if legacy is not None and legacy.status != "running":
        jobs = sorted([*jobs, legacy], key=lambda j: j.finished or 0, reverse=True)
    return jobs


def _show_running(job, cancellable: bool) -> None:
    st.markdown(f"▶️ **실행 중** · 요청 {_who(job)}  \n" + _params_line(job.params))
    with st.status(f"{job.label} · 경과 {_mmss(job.elapsed)}", expanded=True):
        for step in job.steps:
            st.write(step)
    if cancellable and not job.cancel_requested:
        if job_queue.can_cancel(job, SESSION_ID):
            if st.button("⏹ 이 분석 중단", key=f"stop_{job.id}",
                         help="지금 진행 중인 LLM 호출이 끝나면 멈춥니다."):
                queue.cancel(job.id, owner=SESSION_ID)
                st.rerun()
        else:
            st.caption("다른 사람이 요청한 분석이라 중단할 수 없습니다.")


@st.fragment(run_every=3)
def queue_panel() -> None:
    """실행 중·대기 중 작업 (3초마다 이 부분만 새로고침)."""
    running, waiting, _ = queue.snapshot()

    # 작업이 끝나면 화면 전체를 다시 그려 '최근 끝난 작업'에 반영
    done_ids = tuple(id(j) for j in _finished_jobs())
    if st.session_state.setdefault("done_ids", done_ids) != done_ids:
        st.session_state["done_ids"] = done_ids
        st.rerun()

    legacy = legacy_slot["job"]
    if _legacy_running():
        _show_running(legacy, cancellable=False)
        st.caption("대기열 도입 전에 시작된 분석입니다. 끝나면 대기열의 다음 작업이 이어서 시작됩니다.")
    if running:
        _show_running(running, cancellable=True)
    if not running and not _legacy_running():
        st.info(
            "지금 실행 중인 분석이 없습니다. 👈 왼쪽에서 조건을 정하고 **🚀 분석 실행**을 누르세요.\n\n"
            "**참고:** 분석가 4명 + 토론 1라운드 기준 Claude Code(구독) 방식은 1건에 약 **10~25분** 걸립니다. "
            "여러 건을 넣으면 순서대로 하나씩 실행됩니다."
        )

    if waiting:
        st.markdown(f"#### ⏳ 대기 중 {len(waiting)}건")
        for n, job in enumerate(waiting, 1):
            p = job.params
            c1, c2 = st.columns([6, 1])
            c1.markdown(
                f"**{n}.** `{p['ticker']}` · {p['trade_date']} · 심층 `{p['deep_model']}` · 빠른 `{p['quick_model']}` · "
                f"토론 {p['debate_rounds']}/{p['risk_rounds']} · 요청 {_who(job)} · {_clock(job.created)} 추가"
            )
            if job_queue.can_cancel(job, SESSION_ID) and c2.button("취소", key=f"cancel_{job.id}"):
                queue.cancel(job.id, owner=SESSION_ID)
                st.rerun()
    st.caption("창을 닫아도 분석은 서버에서 계속되고, 끝나면 📚 분석 이력에 자동 저장됩니다. "
               "서버를 다시 켜면 대기 중이던 작업도 이어서 처리합니다.")


def show_finished(job, key_prefix: str) -> None:
    st.markdown(_params_line(job.params) + f" · **요청** {_who(job)}")
    if job.status == "done":
        st.success(f"✅ 분석 완료 · LLM 호출 {job.calls}회 · 총 {_mmss(job.elapsed)} — 📚 분석 이력에 저장됨")
        with st.expander("단계별 소요 시간"):
            for step in job.steps:
                st.write(step)
        render_record(history_store.load_run(job.record_id), key_prefix=key_prefix)
    elif job.status == "cancelled":
        st.info(f"⏹ 취소됨 · {'경과 ' + _mmss(job.elapsed) if job.started else '시작 전에 취소'}")
    else:
        # 상세 오류(내부 경로가 담긴 traceback)는 화면에 보여주지 않고 서버 로그에만 남긴다.
        if not getattr(job, "_error_logged", False):
            log.error("분석 실패 (작업 %s):\n%s", job.id, job.error)
            job._error_logged = True
        last = (job.error or "").strip().splitlines()[-1:] or [""]
        kind = last[0].split(":", 1)[0].rsplit(".", 1)[-1] if last[0] else "알 수 없는 오류"
        st.error(f"❌ 분석 실패 · 경과 {_mmss(job.elapsed)} · 오류 종류: `{kind}` (자세한 내용은 서버 로그 참고)")


# ---------------------------------------------------------------------------
# 대기열에 추가
# ---------------------------------------------------------------------------
if run_button:
    if not analysts:
        st.sidebar.error("분석가를 최소 한 명 이상 선택해주세요.")
    elif not ticker_catalog.is_valid_ticker(ticker):
        st.sidebar.error("올바른 종목 티커를 선택해주세요.")
    else:
        config = DEFAULT_CONFIG.copy()
        config.update(
            llm_provider=provider,
            deep_think_llm=deep_model,
            quick_think_llm=quick_model,
            max_debate_rounds=debate_rounds,
            max_risk_discuss_rounds=risk_rounds,
            output_language=output_language,
        )
        params = dict(
            ticker=ticker,
            trade_date=str(trade_date),
            provider=provider,
            deep_model=deep_model,
            quick_model=quick_model,
            analysts=analysts,
            debate_rounds=debate_rounds,
            risk_rounds=risk_rounds,
            language=lang_label,
            notes=notes.strip(),
            position=positions_by_ticker[ticker],
        )
        try:
            job, ahead = queue.submit(params, config, requester=requester, owner=SESSION_ID)
        except job_queue.DuplicateJob as dup:
            st.sidebar.warning(f"같은 조건의 분석이 이미 대기열에 있습니다 (요청 {_who(dup.job)}).")
        except job_queue.QueueFull:
            st.sidebar.error(f"대기열이 가득 찼습니다 (최대 {job_queue.MAX_QUEUED}건). 앞의 작업이 끝난 뒤 다시 시도하세요.")
        else:
            st.toast(f"{ticker} 분석을 " + (f"대기열에 넣었습니다 · 앞에 {ahead}건" if ahead else "시작합니다"), icon="✅")


# ---------------------------------------------------------------------------
# 메인 패널
# ---------------------------------------------------------------------------
tab_run, tab_watch, tab_swing, tab_history, tab_telegram, tab_events = st.tabs(
    ["🚀 분석 · 대기열", "👀 관심 종목", "📐 스윙 매매", "📚 분석 이력", "📨 텔레그램", "🧠 이벤트 분석"],
    key="main_tab", on_change="rerun",
)

if tab_swing.open:
    with tab_swing:
        swing_ui.render()

if tab_run.open:
    with tab_run:
        queue_panel()

        finished = _finished_jobs()
        if finished:
            st.divider()
            st.markdown("#### 🏁 최근 끝난 작업")
            picked_job = st.selectbox(
                "결과 보기",
                finished,
                index=0,
                format_func=lambda j: f"{job_queue.STATUS_KO.get(j.status, j.status)} · {j.params['ticker']} · "
                f"{j.params['trade_date']} · 요청 {_who(j)} · {_clock(j.finished)} 종료",
                help=f"서버를 다시 켜기 전까지 최근 {job_queue.KEEP_FINISHED}건을 보여줍니다. 지난 결과는 📚 분석 이력에 있습니다.",
            )
            show_finished(picked_job, key_prefix="latest")


@st.cache_data(show_spinner=False)
def cached_runs(signature: tuple) -> list:
    return history_store.list_runs()


if tab_history.open:
    with tab_history:
        history_store.import_engine_logs(Path(DEFAULT_CONFIG["results_dir"]))
        history_store.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        signature = tuple(sorted((f.name, f.stat().st_mtime) for f in history_store.HISTORY_DIR.glob("*.json")))
        runs = cached_runs(signature)

        if not runs:
            st.info("아직 저장된 분석이 없습니다. 분석을 실행하면 끝난 결과가 여기에 쌓입니다.")
        else:
            tickers = sorted({r["ticker"] for r in runs})
            picked = st.multiselect("종목 필터", tickers, placeholder="전체 종목")
            shown = [r for r in runs if not picked or r["ticker"] in picked]

            df = pd.DataFrame(
                {
                    "저장 시각": [r.get("saved_at", "").replace("T", " ") for r in shown],
                    "종목": [r["ticker"] for r in shown],
                    "기준일": [r["trade_date"] for r in shown],
                    "결정": [history_store.decision_ko(r.get("decision")) or "-" for r in shown],
                    "포지션": [history_store.position_ko(r) for r in shown],
                    "할 일": [history_store.action_ko(r) or "-" for r in shown],
                    "진입·청산 조건": [(entry_plans.load_plan(r["id"]) or {}).get("summary") or "-" for r in shown],
                    "모델": [
                        f"{r['provider']} · {r.get('deep_model')}/{r.get('quick_model')}" if r.get("provider") else "-"
                        for r in shown
                    ],
                    "분석가": [history_store.analysts_ko(r.get("analysts")) or "-" for r in shown],
                    "요청자": [r.get("requester") or "-" for r in shown],
                    "소요": [_mmss(r["elapsed_sec"]) if r.get("elapsed_sec") else "-" for r in shown],
                    "출처": ["앱" if r.get("source") == "app" else "엔진 로그" for r in shown],
                    "비고": ["⚠️ 신뢰 불가" if r.get("warning") else "" for r in shown],
                }
            )
            st.caption(f"총 {len(shown)}건 · 행을 클릭하면 아래에 내용이 열립니다. "
                       f"보관 위치: `{history_store.HISTORY_DIR}`")
            event = st.dataframe(
                df, hide_index=True, width="stretch",
                on_select="rerun", selection_mode="single-row", key="history_table",
            )
            if event.selection.rows:
                st.divider()
                render_record(history_store.load_run(shown[event.selection.rows[0]]["id"]), key_prefix="history")


if tab_telegram.open:
    with tab_telegram:
        data = telegram_collector.read_messages(st.text_input("🔎 메시지 검색", placeholder="내용·작성자 일부"), limit=500)
        if data is None:
            st.info(
                "아직 수집된 텔레그램 메시지가 없습니다. 수집기를 처음 쓰는 순서:\n\n"
                "1. https://my.telegram.org → **API development tools** 에서 `api_id`, `api_hash` 발급\n"
                "2. `.env` 의 `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` 에 입력\n"
                "3. 터미널에서 로그인: `.venv/Scripts/python.exe telegram_collector.py login`\n"
                "4. 수집 시작 (5분마다): `.venv/Scripts/python.exe telegram_collector.py run`"
            )
        else:
            runs = data["runs"]
            last = runs[0] if runs else None
            cols = st.columns(3)
            cols[0].metric("저장된 메시지", f"{data['total']:,}건")
            if last:
                when = pd.Timestamp(last["finished_at"]).tz_convert("Asia/Seoul").strftime("%m-%d %H:%M")
                cols[1].metric("마지막 수집", when)
                cols[2].metric("직전 수집 결과", f"새 글 {last['new']}건" if not last["error"] else "실패")
                if last["error"]:
                    st.warning(f"마지막 수집 실패: {last['error']}", icon="⚠️")
            msgs = data["messages"]
            if msgs:
                df = pd.DataFrame(msgs)
                df["시각"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M")
                view = df.rename(columns={"title": "방", "sender_name": "작성자", "text": "내용", "views": "조회",
                                          "media": "첨부", "link": "링크"})[["시각", "방", "작성자", "내용", "조회", "첨부", "링크"]]
                st.caption(f"최신 {len(view)}건 표시 · 보관 위치 `{telegram_collector.DB_PATH}`")
                st.dataframe(view, hide_index=True, width="stretch",
                             column_config={"링크": st.column_config.LinkColumn("링크", display_text="열기")})
                st.download_button("📄 CSV로 내려받기", view.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="telegram_messages.csv", mime="text/csv")


def _ago(value) -> str:
    """'11-22 22:36 (12분 전)' 형식 (KST)."""
    if not value:
        return "-"
    dt = datetime.fromisoformat(str(value))
    minutes = int((datetime.now(dt.tzinfo) - dt).total_seconds() // 60)
    rel = f"{minutes}분 전" if minutes < 60 else f"{minutes // 60}시간 전" if minutes < 1440 else f"{minutes // 1440}일 전"
    return f"{news_event_agent._kst(dt)} ({rel})"


def _render_event_pipeline() -> None:
    """뉴스가 언제 들어왔고, 각 이벤트 문서가 언제 나오는지."""
    s = news_event_agent.pipeline_status()
    if not s.get("ok"):
        st.info("텔레그램 수집 데이터가 아직 없습니다.")
        return
    run = s.get("last_run") or {}
    st.markdown("##### ⏱ 수집·분석 현황")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("마지막 수집 (5분마다)", _ago(run.get("finished_at")),
              help="텔레그램 채널을 확인한 시각. 새 글이 없어도 5분마다 확인합니다.")
    c2.metric("채널 마지막 새 글", _ago(s.get("last_message")), help=f"오늘 들어온 글 {s.get('messages_today', 0)}건")
    c3.metric("분류 대기", f"{s.get('unclassified', 0)}건",
              help=f"마지막 분류 {_ago(s.get('last_classified'))}. 새 글은 다음 5분 주기에 이벤트인지 분류합니다.")
    c4.metric("오늘 작성한 문서", f"{s.get('reports_today', 0)} / {s.get('daily_limit', 20)}건",
              help="하루(UTC) 문서 작성 상한")
    if run.get("error"):
        st.warning(f"마지막 수집 오류: {run['error']}")

    auto = [e for e in s["events"] if e["auto"]]
    manual = [e for e in s["events"] if not e["auto"]]
    if auto:
        st.markdown(f"**대기 중인 이벤트 {len(auto)}건** — 1차 문서는 뉴스가 반영된 첫 거래일 장 마감 뒤 "
                    f"약 {news_event_agent.REPORT_LAG_MINUTES}분 안에, 최종은 20거래일 뒤에 작성됩니다.")
        st.dataframe(pd.DataFrame({
            "뉴스 일시": [news_event_agent._kst(e["news_date"]) for e in auto],
            "종목": [e["ticker"] for e in auto],
            "유형": [e["event_type"] for e in auto],
            "이벤트": [e["headline"] for e in auto],
            "상태": [e["state"] for e in auto],
            "1차(D0) 예정": [news_event_agent._kst(e["d0_eta"]) if e["d0_eta"] else "완료" for e in auto],
            "최종(D+20) 예정": [news_event_agent._kst(e["d20_eta"]) for e in auto],
        }), hide_index=True, width="stretch")
        st.caption("예정 시각은 한국 시간이며 주말만 건너뛴 추정입니다. 거래소 휴장일이 끼면 그만큼 늦어집니다.")
    else:
        st.caption("대기 중인 이벤트가 없습니다. 새 글이 이벤트로 분류되면 여기에 나타납니다.")
    if manual:
        with st.expander(f"자동 분석 시작({news_event_agent._kst(s.get('since'))}) 전 이벤트 {len(manual)}건 — 문서 없음"):
            st.caption("비용 때문에 과거 글은 자동으로 쓰지 않습니다. 필요하면 소급 분석(backlog)을 요청하세요.")
            st.dataframe(pd.DataFrame({
                "뉴스 일시": [news_event_agent._kst(e["news_date"]) for e in manual],
                "종목": [e["ticker"] for e in manual],
                "유형": [e["event_type"] for e in manual],
                "이벤트": [e["headline"] for e in manual],
            }), hide_index=True, width="stretch")
    st.divider()


if tab_events.open:
    with tab_events:
        reports = news_event_agent.read_reports(limit=500)
        st.caption(
            "텔레그램 뉴스 중 주가를 움직일 만한 이벤트를 골라 '이벤트 드리븐 퀀트 애널리스트' 역할로 분석한 문서입니다. "
            "1차(D0)는 뉴스가 반영된 첫 거래일 장 마감 뒤, 최종(D+20)은 20거래일 뒤에 다시 씁니다. "
            f"보관 위치 `{news_event_agent.REPORT_DIR}`"
        )
        _render_event_pipeline()
        if not reports:
            st.info("아직 작성된 문서가 없습니다. 위 '대기 중인 이벤트'의 1차 예정 시각이 지나면 생깁니다.")
        else:
            c1, c2, c3 = st.columns([2, 2, 3])
            tickers = sorted({r["ticker"] for r in reports})
            pick_t = c1.multiselect("종목", tickers, placeholder="전체")
            kinds = sorted({r["event_type"] for r in reports})
            pick_k = c2.multiselect("이벤트 유형", kinds, placeholder="전체")
            words = c3.text_input("요약 검색", placeholder="예: 자사주, 셀온, 가이던스")
            shown = [
                r for r in reports
                if (not pick_t or r["ticker"] in pick_t) and (not pick_k or r["event_type"] in pick_k)
                and (not words or words in (r["summary"] or "") or words in (r["headline"] or ""))
            ]
            # 같은 문서를 가리키는 글(같은 종목·같은 날 묶음)은 한 줄만
            seen, rows = set(), []
            for r in shown:
                if r["path"] not in seen:
                    seen.add(r["path"])
                    rows.append(r)
            df = pd.DataFrame({
                "뉴스 일시": [pd.Timestamp(r["news_date"]).tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M") for r in rows],
                "종목": [r["ticker"] for r in rows],
                "유형": [r["event_type"] for r in rows],
                "단계": ["최종 D+20" if r["stage"] == "D20" else "1차 D0" for r in rows],
                "이벤트": [r["headline"] for r in rows],
                "요약": [r["summary"] for r in rows],
            })
            st.caption(f"문서 {len(rows)}건 · 행을 클릭하면 아래에 문서가 열립니다.")
            event = st.dataframe(df, hide_index=True, width="stretch", on_select="rerun",
                                 selection_mode="single-row", key="event_reports")
            if event.selection.rows:
                doc = Path(rows[event.selection.rows[0]]["path"])
                st.divider()
                if doc.exists():
                    body = doc.read_text(encoding="utf-8")
                    st.download_button("📄 문서 내려받기 (.md)", body, file_name=doc.name, mime="text/markdown")
                    st.markdown(body)
                else:
                    st.warning(f"문서 파일이 없습니다: {doc}")


# ---------------------------------------------------------------------------
# 관심 종목 — 매일 자동 이슈 점검 (watchlist.py 가 실행)
# ---------------------------------------------------------------------------
_IMPORTANCE_ICON = {"높음": "🔴", "보통": "🟡", "낮음": "⚪"}
_IMPACT_ICON = {"긍정": "🟢", "부정": "🔻", "중립": "▫️"}


def _lookup_name(symbol: str) -> str:
    for sym, label in ticker_catalog.search(symbol, limit=5):
        if sym.upper() == symbol:
            return label
    return ""


def _render_briefing(rec: dict) -> None:
    icon = _IMPORTANCE_ICON.get(rec.get("importance"), "🟡")
    pos = history_store.position_ko({"position": rec.get("position")})
    st.markdown(f"#### {icon} {rec['ticker']} — {rec.get('headline', '')}")
    st.caption(f"{rec.get('name') or ''} · 점검 {rec.get('created_at', '').replace('T', ' ')} · "
               f"가격 기준 {rec.get('price_date') or '-'} · 중요도 {rec.get('importance')} · 포지션 {pos} · "
               f"뉴스 {rec.get('inputs', {}).get('news', 0)}건 · 텔레그램 {rec.get('inputs', {}).get('telegram', 0)}건"
               + "".join(f" · {label} {rec['inputs'][key]}건" for key, label in (("filings", "공시"), ("naver", "네이버"))
                         if rec.get("inputs", {}).get(key)))
    st.markdown(rec.get("summary", ""))
    for issue in rec.get("issues", []):
        st.markdown(f"- {_IMPACT_ICON.get(issue.get('impact'), '▫️')} **{issue.get('title')}** "
                    f"({issue.get('source')}) — {issue.get('detail')}")
    st.markdown(f"📈 {rec.get('price_comment', '')}")
    entry = rec.get("entry_watch")
    if entry:
        line = f"🎯 **{entry['status_ko']}** — {entry.get('summary', '')}"
        if entry.get("fired"):
            line += " · 충족: " + ", ".join(entry["fired"])
        elif entry.get("nearest"):
            line += f" · 가장 가까운 조건: {entry['nearest']}"
        st.markdown(line)
        st.caption(f"기준 분석 {entry.get('trade_date')} ({history_store.decision_ko(entry.get('decision'))}) · "
                   f"종가 {entry.get('last_close')} ({entry.get('last_date')})")
    full = "전체 분석 필요" if rec.get("needs_full_analysis") else "전체 분석 불필요"
    st.caption(f"🧭 {full}: {rec.get('reason', '')}" + (f" → {rec['full_analysis']}" if rec.get("full_analysis") else ""))


if tab_watch.open:
    with tab_watch:
        st.markdown(
            f"매일 **{watchlist.RUN_AT}**에 아래 종목의 뉴스·텔레그램 언급·가격 변화를 모아 이슈 브리핑을 씁니다. "
            f"중요한 새 이슈가 있거나, 최근 전체 분석이 제시한 진입·손절·목표가 조건이 종가로 새로 충족되면 "
            f"전체 분석(매수/매도 판단)을 자동으로 대기열에 넣습니다 "
            f"(하루 최대 {watchlist.MAX_FULL}건, 결과는 📚 분석 이력)."
        )
        items = watchlist.load_watchlist()
        editor_df = pd.DataFrame(
            [{"티커": i["ticker"], "종목명": i["name"], "보유 중": i["held"],
              "평균단가": i["avg_price"], "자동 전체 분석": i["auto_full"]} for i in items],
            columns=["티커", "종목명", "보유 중", "평균단가", "자동 전체 분석"],
        )
        edited = st.data_editor(
            editor_df, num_rows="dynamic", hide_index=True, width="stretch", key="watchlist_editor",
            column_config={
                "티커": st.column_config.TextColumn(help="Yahoo Finance 티커 (예: NVDA, 005930.KS)", required=True),
                "종목명": st.column_config.TextColumn(help="비워 두면 저장할 때 자동으로 찾습니다. 텔레그램 글 검색에도 쓰입니다."),
                "보유 중": st.column_config.CheckboxColumn(default=False),
                "평균단가": st.column_config.NumberColumn(min_value=0.0, format="%.4g"),
                "자동 전체 분석": st.column_config.CheckboxColumn(default=True, help="끄면 브리핑만 쓰고 전체 분석은 맡기지 않습니다."),
            },
        )
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("💾 목록 저장", type="primary", width="stretch"):
            new_items, bad = [], []
            for _, row in edited.iterrows():
                symbol = ticker_catalog.normalize(str(row["티커"] or ""))
                if not symbol:
                    continue
                if not ticker_catalog.is_valid_ticker(symbol):
                    bad.append(symbol)
                    continue
                if symbol in {i["ticker"] for i in new_items}:
                    continue
                name = str(row["종목명"] or "").strip() if pd.notna(row["종목명"]) else ""
                avg = row["평균단가"] if pd.notna(row["평균단가"]) else None
                new_items.append({
                    "ticker": symbol, "name": name or _lookup_name(symbol),
                    **normalize_position({"held": row["보유 중"] is True, "avg_price": avg}),
                    "auto_full": bool(row["자동 전체 분석"]) if pd.notna(row["자동 전체 분석"]) else True,
                })
            if bad:
                st.error("티커 형식이 아닌 항목은 저장하지 않았습니다: " + ", ".join(bad))
            watchlist.save_watchlist(new_items)
            st.toast(f"관심 종목 {len(new_items)}개 저장", icon="✅")
            st.rerun()
        if c2.button("🔄 지금 점검", width="stretch", disabled=not items,
                     help="정기 점검을 기다리지 않고 지금 모든 관심 종목을 점검합니다 (1분 안에 시작, 종목당 약 20초)."):
            watchlist.request_run()
            st.toast("점검을 요청했습니다. 1~2분 뒤 새로고침하세요.", icon="🔄")

        state = watchlist._read_json(watchlist.STATE_FILE, {})
        pending = watchlist.RUN_REQUEST.exists()
        st.caption(f"마지막 정기 점검: {state.get('last_run') or '아직 없음'}"
                   + (" · ⏳ 요청한 점검 대기 중" if pending else ""))

        briefs = watchlist.list_briefings(days=30)
        if briefs:
            st.divider()
            latest = {}
            for rec in briefs:
                latest.setdefault(rec["ticker"], rec)
            order = [i["ticker"] for i in items if i["ticker"] in latest] + [t for t in latest if t not in {i["ticker"] for i in items}]
            st.markdown("### 🗞 종목별 최신 브리핑")
            positions = {i["ticker"]: i for i in items}
            last_runs = {}
            for run in history_store.list_runs():
                if run.get("decision") and not run.get("warning"):
                    last_runs.setdefault(run["ticker"], run["id"])
            for symbol in order:
                with st.container(border=True):
                    rec = latest[symbol]
                    _render_briefing(rec)
                    run_id = (rec.get("entry_watch") or {}).get("run_id") or last_runs.get(symbol)
                    plan = entry_plans.load_plan(run_id) if run_id else None
                    # 펼쳤을 때만 차트를 그림 (닫힌 차트까지 매번 그리면 화면이 느려짐)
                    with st.expander("📈 차트와 진입·청산 가격대", key=f"watch_exp_{symbol}",
                                     on_change="rerun") as chart_box:
                        if chart_box.open:
                            swing_on = st.toggle("📐 스윙 기준선 겹쳐 보기 (오늘 종가 기준 손절·2R·4R)",
                                                 key=f"swing_lv_{symbol}")
                            extra = swing_ui.swing_levels(symbol, swing_rules.load_settings()) if swing_on else []
                            _render_chart(symbol, plan, positions.get(symbol, {}).get("avg_price"),
                                          key=f"watch_{symbol}", extra_levels=extra)

            with st.expander("📅 지난 브리핑 보기"):
                dates = sorted({r["date"] for r in briefs}, reverse=True)
                day = st.selectbox("날짜", dates, key="brief_day")
                for rec in [r for r in briefs if r["date"] == day]:
                    with st.container(border=True):
                        _render_briefing(rec)
        elif items:
            st.info("아직 브리핑이 없습니다. '🔄 지금 점검'을 누르거나 다음 정기 점검을 기다려 주세요.")
