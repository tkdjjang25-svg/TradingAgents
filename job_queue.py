"""분석 대기열 (app.py 전용).

서버 프로세스 하나에 대기열 하나가 있고, 작업 스레드 하나가 앞에서부터 한 건씩 분석한다.
여러 사람이 동시에 요청해도 분석은 한 번에 한 건만 돈다. 동시에 돌리면 구독 사용량
한도가 빨리 닳고, 엔진의 데이터 설정(dataflows.config)이 프로세스 전역이라 서로 섞인다.

대기 중인 작업과 실행 중인 작업은 QUEUE_FILE에 저장해 두므로, 서버를 다시 켜면
이어서 처리한다 (실행 중이던 작업은 처음부터 다시 분석한다).
"""
import json
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from tradingagents.default_config import TRADINGAGENTS_HOME
from typing import Callable, List, Optional, Tuple

from langchain_core.callbacks import BaseCallbackHandler

import history_store
from tradingagents.agents.utils.agent_utils import normalize_position

QUEUE_FILE = Path(TRADINGAGENTS_HOME) / "queue.json"
MAX_QUEUED = 30
KEEP_FINISHED = 30

# LangGraph 노드 이름 → 화면 표시명
AGENT_LABELS = {
    "Market Analyst": "📊 시장 분석가",
    "Social Analyst": "💬 소셜 분석가",
    "News Analyst": "📰 뉴스 분석가",
    "Fundamentals Analyst": "🏢 펀더멘털 분석가",
    "Bull Researcher": "🐂 강세 연구원",
    "Bear Researcher": "🐻 약세 연구원",
    "Research Manager": "⚖️ 리서치 매니저",
    "Trader": "💼 트레이더",
    "Aggressive Analyst": "🔥 공격적 리스크 분석가",
    "Neutral Analyst": "⚖️ 중립적 리스크 분석가",
    "Conservative Analyst": "🛡️ 보수적 리스크 분석가",
    "Portfolio Manager": "📋 포트폴리오 매니저",
}

STATUS_KO = {
    "queued": "⏳ 대기",
    "running": "▶️ 실행 중",
    "done": "✅ 완료",
    "error": "❌ 실패",
    "cancelled": "⏹ 취소",
}


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}분 {int(seconds % 60):02d}초"


class Cancelled(Exception):
    """사용자가 실행 중인 분석을 중단했다."""


class QueueFull(Exception):
    pass


class DuplicateJob(Exception):
    def __init__(self, job: "Job"):
        super().__init__(job.id)
        self.job = job


@dataclass
class Job:
    params: dict  # 실행 조건 (화면 표시·이력 저장용)
    config: dict
    requester: str = ""
    owner: str = ""  # 로그인 세션 식별자 (게이트웨이가 준 값). 이 세션만 작업을 취소할 수 있다
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    status: str = "queued"  # queued | running | done | error | cancelled
    cancel_requested: bool = False
    label: str = "대기 중"
    calls: int = 0
    steps: list = field(default_factory=list)
    error: str = ""
    record_id: str = ""
    model_ids: list = field(default_factory=list)  # 실제로 응답한 모델 ID (claude-code)

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def to_saved(self) -> dict:
        return {
            "id": self.id, "params": self.params, "config": self.config,
            "requester": self.requester, "owner": self.owner, "created": self.created,
        }


class ProgressHandler(BaseCallbackHandler):
    """LLM 호출이 시작될 때마다 현재 에이전트를 Job에 기록하고, 중단 요청을 확인한다.

    LangGraph가 콜백 메타데이터에 실행 중인 노드 이름(langgraph_node)을 넣어준다.
    그래프 밖 호출은 실행 초반의 지난 결정 회고(reflection)뿐이다.
    raise_error=True라야 Cancelled가 콜백 밖으로 나가 분석을 멈춘다.
    """

    raise_error = True

    def __init__(self, job: Job):
        self.job = job
        self.current = None
        self.current_start = job.started or time.time()

    def on_chat_model_start(self, serialized, messages, *, metadata=None, **kwargs):
        if self.job.cancel_requested:
            raise Cancelled()
        node = (metadata or {}).get("langgraph_node")
        label = AGENT_LABELS.get(node, node) if node else "🧠 지난 결정 회고"
        now = time.time()
        if label != self.current:
            self._close_step(now)
            self.current, self.current_start = label, now
        self.job.calls += 1
        self.job.label = f"{label} 작업 중 · LLM 호출 {self.job.calls}회째"

    def on_llm_end(self, response, **kwargs):
        try:
            for gens in response.generations:
                for gen in gens:
                    meta = getattr(getattr(gen, "message", None), "response_metadata", None) or {}
                    for model_id in meta.get("model_ids") or []:
                        if model_id not in self.job.model_ids:
                            self.job.model_ids.append(model_id)
        except Exception:
            pass  # 표시용 정보라 분석을 멈출 이유가 없다

    def _close_step(self, now: float) -> None:
        if self.current:
            self.job.steps.append(f"✅ {self.current} — {mmss(now - self.current_start)}")

    def finish(self) -> None:
        self._close_step(time.time())
        self.current = None


def run_job(job: Job, graph_factory: Callable) -> None:
    """Job 하나를 끝까지 실행하고 결과를 이력에 저장한다. 예외는 Job에 기록한다."""
    job.started, job.status, job.label = time.time(), "running", "분석 시작…"
    progress = ProgressHandler(job)
    p = job.params
    try:
        ta = graph_factory(selected_analysts=p["analysts"], debug=False, config=job.config, callbacks=[progress])
        final_state, decision = ta.propagate(p["ticker"], p["trade_date"], user_notes=p.get("notes", ""), position=p.get("position"))
        progress.finish()
        job.finished = time.time()
        engine_log = (
            Path(job.config["results_dir"]) / p["ticker"] / "TradingAgentsStrategy_logs"
            / f"full_states_log_{p['trade_date']}.json"
        )
        meta = {
            k: p[k]
            for k in ("provider", "deep_model", "quick_model", "analysts", "debate_rounds", "risk_rounds", "language")
        }
        meta.update(elapsed_sec=round(job.elapsed), llm_calls=job.calls)
        if job.model_ids:
            meta["model_ids"] = job.model_ids
        if job.requester:
            meta["requester"] = job.requester
        if p.get("notes"):
            meta["notes"] = p["notes"]
        meta["position"] = normalize_position(p.get("position"))
        job.record_id = history_store.save_run(final_state, decision, meta, engine_log).stem
        job.status = "done"
    except Exception as exc:
        progress.finish()
        if isinstance(exc, Cancelled) or job.cancel_requested:
            job.status = "cancelled"
        else:
            job.error = traceback.format_exc()
            job.status = "error"
    finally:
        job.finished = job.finished or time.time()


def can_cancel(job: Job, owner: Optional[str]) -> bool:
    """owner=None 은 내부 호출(제한 없음). 소유자가 기록된 작업은 같은 세션만 취소할 수 있다."""
    if owner is None:
        return True
    job_owner = getattr(job, "owner", "")
    return bool(job_owner) and job_owner == owner


class JobQueue:
    """서버 전체가 공유하는 분석 대기열. 작업 스레드 하나가 순서대로 처리한다.

    runner(job)는 job을 끝까지 실행하는 함수 (보통 run_job). busy()가 True인 동안에는
    다음 작업을 시작하지 않는다 (대기열 밖에서 돌고 있는 분석을 기다릴 때).
    """

    def __init__(
        self,
        runner: Callable[[Job], None],
        path: Optional[Path] = QUEUE_FILE,
        busy: Callable[[], bool] = lambda: False,
        autostart: bool = True,
        poll_sec: float = 2.0,
    ):
        self._runner, self._path, self._busy, self._poll = runner, path, busy, poll_sec
        self._cond = threading.Condition()
        self._jobs: List[Job] = []  # 대기·실행 중인 작업과 최근 끝난 작업, 추가된 순서
        self.version = 0  # 작업이 끝나거나 대기열이 바뀔 때마다 증가
        self._closed = False
        self._load()
        if autostart:
            threading.Thread(target=self._work, daemon=True, name="analysis-queue").start()

    # ----- 조회 -------------------------------------------------------------
    def snapshot(self) -> Tuple[Optional[Job], List[Job], List[Job]]:
        """(실행 중, 대기 중 순서대로, 최근 끝난 것 최신순)"""
        with self._cond:
            running = next((j for j in self._jobs if j.status == "running"), None)
            queued = [j for j in self._jobs if j.status == "queued"]
            finished = sorted(
                (j for j in self._jobs if not j.active), key=lambda j: j.finished or 0, reverse=True
            )
            return running, queued, finished

    def get(self, job_id: str) -> Optional[Job]:
        with self._cond:
            return next((j for j in self._jobs if j.id == job_id), None)

    # ----- 변경 -------------------------------------------------------------
    def submit(self, params: dict, config: dict, requester: str = "", owner: str = "") -> Tuple[Job, int]:
        """작업을 맨 뒤에 넣고 (작업, 앞에 있는 작업 수)를 반환한다."""
        with self._cond:
            active = [j for j in self._jobs if j.active]
            for j in active:
                if j.params == params:
                    raise DuplicateJob(j)
            if sum(j.status == "queued" for j in active) >= MAX_QUEUED:
                raise QueueFull()
            job = Job(params=params, config=config, requester=requester.strip(), owner=owner)
            self._jobs.append(job)
            self._changed()
            return job, len(active)

    def cancel(self, job_id: str, owner: Optional[str] = None) -> bool:
        """대기 중이면 바로 빼고, 실행 중이면 다음 LLM 호출 직전에 멈추도록 표시한다.

        owner 를 주면 그 세션이 넣은 작업만 취소한다 (다른 사람 작업은 False).
        """
        with self._cond:
            job = next((j for j in self._jobs if j.id == job_id), None)
            if job is None or not job.active or not can_cancel(job, owner):
                return False
            if job.status == "queued":
                job.status, job.finished = "cancelled", time.time()
            else:
                job.cancel_requested = True
                job.label = "중단 요청됨 — 지금 진행 중인 LLM 호출이 끝나면 멈춥니다"
            self._changed()
            return True

    def close(self) -> None:
        """작업 스레드를 멈춘다 (지금 실행 중인 작업은 끝까지 돈다). 테스트용."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def clear_finished(self) -> None:
        with self._cond:
            self._jobs = [j for j in self._jobs if j.active]
            self._changed()

    # ----- 내부 -------------------------------------------------------------
    def _changed(self) -> None:
        """잠금을 잡은 상태에서 호출."""
        finished = [j for j in self._jobs if not j.active]
        for old in finished[:-KEEP_FINISHED] if len(finished) > KEEP_FINISHED else []:
            self._jobs.remove(old)
        self.version += 1
        self._save()
        self._cond.notify_all()

    def _next(self) -> Optional[Job]:
        with self._cond:
            while not self._closed:
                job = next((j for j in self._jobs if j.status == "queued"), None)
                if job is not None and not self._busy():
                    job.status = "running"  # run_job이 started 등을 채운다
                    self._changed()
                    return job
                self._cond.wait(timeout=self._poll)
            return None

    def _work(self) -> None:
        while True:
            job = self._next()
            if job is None:
                return
            try:
                self._runner(job)
            except Exception:  # runner가 예외를 삼키지 못했을 때도 대기열은 계속 돈다
                job.status, job.error = "error", traceback.format_exc()
                job.finished = job.finished or time.time()
            with self._cond:
                if self._closed:
                    return
                self._changed()

    def _save(self) -> None:
        if self._path is None:
            return
        pending = [j.to_saved() for j in self._jobs if j.active]
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(pending, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            tmp.replace(self._path)
        except OSError:
            pass  # 저장 실패해도 메모리의 대기열은 그대로 돈다

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            saved = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for item in saved:
            self._jobs.append(
                Job(
                    params=item["params"], config=item["config"], requester=item.get("requester", ""),
                    owner=item.get("owner", ""),
                    id=item.get("id") or uuid.uuid4().hex[:8], created=item.get("created") or time.time(),
                )
            )


def shared_queue(factory: Callable[[], JobQueue]) -> JobQueue:
    """프로세스에 대기열을 하나만 만든다.

    Streamlit은 코드가 바뀌면 이 모듈을 다시 불러오고 st.cache_resource도 새로 만든다.
    그때마다 대기열이 새로 생기면 작업 스레드가 둘이 되어 분석이 동시에 돌므로,
    다시 불러와도 남는 sys에 매달아 둔다.
    """
    lock = sys.__dict__.setdefault("_tradingagents_queue_lock", threading.Lock())
    with lock:
        queue = sys.__dict__.get("_tradingagents_queue")
        if queue is None:
            queue = factory()
            sys._tradingagents_queue = queue
        return queue
