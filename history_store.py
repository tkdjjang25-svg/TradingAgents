"""분석 이력 저장소 (app.py 전용).

분석 1건을 ~/.tradingagents/history/<id>.json 파일 하나로 보관한다.

엔진도 결과를 results_dir/<티커>/TradingAgentsStrategy_logs/full_states_log_<기준일>.json 에
남기지만, 같은 종목·기준일로 다시 분석하면 덮어써지고 모델·소요시간 같은 정보가 없다.
그래서 앱이 실행한 분석은 실행 정보와 함께 따로 저장하고, 콘솔 CLI 등 앱 밖에서 생긴
엔진 로그는 처음 볼 때 이력으로 복사해 둔다.
"""
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tradingagents.agents.utils.rating import parse_rating

HISTORY_DIR = Path.home() / ".tradingagents" / "history"

# 막 쓰인 엔진 로그는 앱의 save_run()이 곧 참조할 수 있으므로 가져오기를 잠시 미룬다.
_IMPORT_GRACE_SEC = 30

_ANALYST_REPORTS = (
    ("market", "market_report"),
    ("social", "sentiment_report"),
    ("news", "news_report"),
    ("fundamentals", "fundamentals_report"),
)


def normalize_state(s: Dict[str, Any]) -> Dict[str, Any]:
    """그래프 최종 state와 엔진 로그를 화면 렌더링용 공통 형태로 맞춘다."""
    inv = s.get("investment_debate_state") or {}
    risk = s.get("risk_debate_state") or {}
    return {
        "market_report": s.get("market_report") or "",
        "sentiment_report": s.get("sentiment_report") or "",
        "news_report": s.get("news_report") or "",
        "fundamentals_report": s.get("fundamentals_report") or "",
        "investment_debate_state": {
            "bull_history": inv.get("bull_history") or "",
            "bear_history": inv.get("bear_history") or "",
            "judge_decision": inv.get("judge_decision") or "",
        },
        "investment_plan": s.get("investment_plan") or "",
        # 엔진 로그는 trader_investment_plan을 trader_investment_decision 키로 저장한다.
        "trader_investment_plan": s.get("trader_investment_plan") or s.get("trader_investment_decision") or "",
        "risk_debate_state": {
            "aggressive_history": risk.get("aggressive_history") or "",
            "conservative_history": risk.get("conservative_history") or "",
            "neutral_history": risk.get("neutral_history") or "",
            "judge_decision": risk.get("judge_decision") or "",
        },
        "final_trade_decision": s.get("final_trade_decision") or "",
    }


def _engine_log_ref(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    return {"path": str(path), "mtime": path.stat().st_mtime}


def _write(record: Dict[str, Any], history_dir: Path) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    base = record["id"]
    path = history_dir / f"{base}.json"
    n = 2
    while path.exists():
        record["id"] = f"{base}-{n}"
        path = history_dir / f"{record['id']}.json"
        n += 1
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def _make_id(saved_at: datetime, ticker: str, trade_date: str) -> str:
    return f"{saved_at:%Y%m%d-%H%M%S}_{re.sub(r'[^A-Za-z0-9._-]', '_', ticker)}_{trade_date}"


def save_run(
    final_state: Dict[str, Any],
    decision: str,
    meta: Dict[str, Any],
    engine_log: Optional[Path] = None,
    history_dir: Optional[Path] = None,
) -> Path:
    """앱에서 실행한 분석 1건을 실행 정보(meta)와 함께 저장한다."""
    history_dir = history_dir or HISTORY_DIR
    saved_at = datetime.now()
    ticker = str(final_state.get("company_of_interest") or meta.get("ticker"))
    trade_date = str(final_state.get("trade_date") or meta.get("trade_date"))
    record = {
        **meta,
        "id": _make_id(saved_at, ticker, trade_date),
        "saved_at": saved_at.isoformat(timespec="seconds"),
        "source": "app",
        "ticker": ticker,
        "trade_date": trade_date,
        "decision": decision,
        "engine_log": _engine_log_ref(engine_log),
        "state": normalize_state(final_state),
    }
    return _write(record, history_dir)


def list_runs(history_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """저장된 분석 목록(본문 state 제외)을 최신순으로 반환한다."""
    history_dir = history_dir or HISTORY_DIR
    runs = []
    for path in history_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        record.pop("state", None)
        runs.append(record)
    return sorted(runs, key=lambda r: r.get("saved_at", ""), reverse=True)


def load_run(run_id: str, history_dir: Optional[Path] = None) -> Dict[str, Any]:
    history_dir = history_dir or HISTORY_DIR
    return json.loads((history_dir / f"{run_id}.json").read_text(encoding="utf-8"))


def import_engine_logs(results_dir: Path, history_dir: Optional[Path] = None) -> int:
    """아직 이력에 없는 엔진 로그를 이력으로 복사하고, 가져온 개수를 반환한다.

    같은 경로라도 수정 시각이 다르면(같은 종목·기준일 재분석으로 덮어쓰기) 새 건으로 본다.
    """
    history_dir = history_dir or HISTORY_DIR
    known = {
        (r["engine_log"]["path"], r["engine_log"]["mtime"])
        for r in list_runs(history_dir)
        if r.get("engine_log")
    }
    imported = 0
    for log in Path(results_dir).glob("*/TradingAgentsStrategy_logs/full_states_log_*.json"):
        mtime = log.stat().st_mtime
        if (str(log), mtime) in known or time.time() - mtime < _IMPORT_GRACE_SEC:
            continue
        try:
            raw = json.loads(log.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        state = normalize_state(raw)
        saved_at = datetime.fromtimestamp(mtime)
        ticker = str(raw.get("company_of_interest") or log.parents[1].name)
        trade_date = str(raw.get("trade_date") or log.stem.removeprefix("full_states_log_"))
        record = {
            "id": _make_id(saved_at, ticker, trade_date),
            "saved_at": saved_at.isoformat(timespec="seconds"),
            "source": "engine-log",
            "ticker": ticker,
            "trade_date": trade_date,
            "decision": parse_rating(state["final_trade_decision"]) if state["final_trade_decision"] else "",
            "analysts": [name for name, key in _ANALYST_REPORTS if state[key]],
            "engine_log": {"path": str(log), "mtime": mtime},
            "state": state,
        }
        _write(record, history_dir)
        imported += 1
    return imported


# 5단계 투자의견 (tradingagents.agents.utils.rating.RATINGS_5_TIER) → (한글, 설명)
DECISION_KO = {
    "Buy": ("매수", "지금 사거나 보유량을 늘리라는 의견"),
    "Overweight": ("비중 확대", "포트폴리오에서 이 종목의 비중을 평소보다 늘리라는 의견"),
    "Hold": ("보유 · 관망", "새로 사거나 팔지 말고 지금 보유량을 유지하라는 의견"),
    "Underweight": ("비중 축소", "포트폴리오에서 이 종목의 비중을 평소보다 줄이라는 의견"),
    "Sell": ("매도", "보유량을 팔라는 의견"),
}


def decision_ko(decision: Optional[str]) -> str:
    """'Underweight' → '비중 축소 (Underweight)'. 모르는 값은 그대로 둔다."""
    if not decision:
        return ""
    ko = DECISION_KO.get(decision.strip().capitalize())
    return f"{ko[0]} ({decision})" if ko else decision


_ANALYST_KO = {"market": "시장", "social": "소셜", "news": "뉴스", "fundamentals": "펀더멘털"}


def analysts_ko(analysts: Optional[List[str]]) -> str:
    return ", ".join(_ANALYST_KO.get(a, a) for a in analysts or [])


def to_markdown(record: Dict[str, Any]) -> str:
    """이력 1건을 내려받기용 마크다운 문서로 만든다."""
    s = record["state"]
    inv, risk = s["investment_debate_state"], s["risk_debate_state"]
    meta = [
        f"- 저장 시각: {record.get('saved_at', '')}",
        f"- 최종 결정: **{decision_ko(record.get('decision')) or '-'}**",
        f"- 분석가: {analysts_ko(record.get('analysts')) or '-'}",
    ]
    if record.get("provider"):
        meta.append(
            f"- 모델: {record['provider']} · 심층 {record.get('deep_model')} · 빠른 {record.get('quick_model')}"
        )
    if record.get("model_ids"):
        meta.append(f"- 실제 모델: {', '.join(record['model_ids'])}")
    if record.get("elapsed_sec"):
        meta.append(f"- 소요: {record['elapsed_sec'] // 60}분 {record['elapsed_sec'] % 60}초 · LLM 호출 {record.get('llm_calls')}회")
    if record.get("warning"):
        meta.insert(0, f"> ⚠️ {record['warning']}\n")

    sections = [
        ("추가 참고 메모 (사용자 제공 · 미검증)", record.get("notes", "")),
        ("최종 거래 의사결정", s["final_trade_decision"]),
        ("시장 분석", s["market_report"]),
        ("소셜 분석", s["sentiment_report"]),
        ("뉴스 분석", s["news_report"]),
        ("펀더멘털 분석", s["fundamentals_report"]),
        ("강세 연구원", inv["bull_history"]),
        ("약세 연구원", inv["bear_history"]),
        ("리서치 매니저 판단", inv["judge_decision"]),
        ("트레이더 계획", s["trader_investment_plan"]),
        ("리스크 토론 — 공격적", risk["aggressive_history"]),
        ("리스크 토론 — 보수적", risk["conservative_history"]),
        ("리스크 토론 — 중립적", risk["neutral_history"]),
    ]
    body = "\n\n".join(f"## {title}\n\n{text}" for title, text in sections if text)
    return f"# {record['ticker']} · 기준일 {record['trade_date']}\n\n" + "\n".join(meta) + "\n\n" + body + "\n"
