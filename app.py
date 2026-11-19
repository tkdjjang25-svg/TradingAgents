"""TradingAgents Streamlit UI (한글).

실행:
    .venv\\Scripts\\python.exe -m streamlit run app.py

분석은 서버의 백그라운드 스레드에서 돌기 때문에, 실행 중에 화면을 조작하거나
브라우저를 닫아도 중단되지 않는다. 끝난 분석은 history_store가 자동 저장한다.
"""
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# .env에서 API 키 로드 (tradingagents import 전에 수행 — 일부 클라이언트는
# import 시점에 환경변수를 읽음). override=True로 .env 변경이 항상 반영되도록 함.
load_dotenv(override=True)

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

import history_store
import job_queue
import ticker_catalog

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TradingAgents",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 TradingAgents")
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
        runner=lambda job: job_queue.run_job(job, TradingAgentsGraph), busy=_legacy_running
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
    if record.get("warning"):
        st.warning(record["warning"], icon="⚠️")

    st.subheader("🎯 최종 거래 의사결정")
    decision = (record.get("decision") or "").strip().capitalize()
    if decision in history_store.DECISION_KO:
        ko, desc = history_store.DECISION_KO[decision]
        box = {"Buy": st.success, "Overweight": st.success, "Hold": st.info}.get(decision, st.error)
        box(f"**{ko}** ({decision}) — {desc}")
    else:
        st.info(record.get("decision") or "(없음)")
    with st.expander("최종 의사결정 원문 보기"):
        st.markdown(s["final_trade_decision"] or "(없음)")

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
        if st.button("⏹ 이 분석 중단", key=f"stop_{job.id}",
                     help="지금 진행 중인 LLM 호출이 끝나면 멈춥니다. 다른 사람의 작업일 수 있으니 확인하고 누르세요."):
            queue.cancel(job.id)
            st.rerun()


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
            if c2.button("취소", key=f"cancel_{job.id}"):
                queue.cancel(job.id)
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
        st.error(f"❌ 분석 실패 · 경과 {_mmss(job.elapsed)}")
        st.code(job.error)


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
        )
        try:
            job, ahead = queue.submit(params, config, requester=requester)
        except job_queue.DuplicateJob as dup:
            st.sidebar.warning(f"같은 조건의 분석이 이미 대기열에 있습니다 (요청 {_who(dup.job)}).")
        except job_queue.QueueFull:
            st.sidebar.error(f"대기열이 가득 찼습니다 (최대 {job_queue.MAX_QUEUED}건). 앞의 작업이 끝난 뒤 다시 시도하세요.")
        else:
            st.toast(f"{ticker} 분석을 " + (f"대기열에 넣었습니다 · 앞에 {ahead}건" if ahead else "시작합니다"), icon="✅")


# ---------------------------------------------------------------------------
# 메인 패널
# ---------------------------------------------------------------------------
tab_run, tab_history = st.tabs(["🚀 분석 · 대기열", "📚 분석 이력"])

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
