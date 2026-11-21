import json
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import job_queue  # noqa: E402
from job_queue import DuplicateJob, Job, JobQueue, QueueFull, run_job  # noqa: E402


def _params(ticker="NVDA", **kw):
    return {"ticker": ticker, "trade_date": "2026-11-19", "provider": "claude-code", "deep_model": "opus",
            "quick_model": "sonnet", "analysts": ["market"], "debate_rounds": 1, "risk_rounds": 1,
            "language": "한국어", **kw}


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class GatedRunner:
    """runner stand-in: each job blocks until release() so tests control the order."""

    def __init__(self):
        self.ran, self.gate = [], threading.Event()

    def __call__(self, job):
        job.started, job.status = time.time(), "running"
        self.ran.append(job.params["ticker"])
        self.gate.wait(5)
        self.gate.clear()
        job.status, job.finished = "done", time.time()

    def release(self):
        self.gate.set()


@pytest.mark.unit
class TestJobQueue(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = Path(self.tmp.name) / "queue.json"
        self.queues = []

    def tearDown(self):
        for q in self.queues:
            q.close()
        self.tmp.cleanup()

    def queue(self, runner, **kw):
        q = JobQueue(runner, path=self.path, poll_sec=0.01, **kw)
        self.queues.append(q)
        return q

    def test_jobs_run_one_at_a_time_in_order(self):
        runner = GatedRunner()
        q = self.queue(runner)
        _, ahead_a = q.submit(_params("A"), {})
        _, ahead_b = q.submit(_params("B"), {})
        self.assertEqual((ahead_a, ahead_b), (0, 1))
        self.assertTrue(_wait(lambda: runner.ran == ["A"]))
        running, queued, _ = q.snapshot()
        self.assertEqual((running.params["ticker"], [j.params["ticker"] for j in queued]), ("A", ["B"]))
        runner.release()
        self.assertTrue(_wait(lambda: runner.ran == ["A", "B"]))
        runner.release()
        self.assertTrue(_wait(lambda: q.snapshot()[0] is None and len(q.snapshot()[2]) == 2))

    def test_duplicate_and_full_queue_are_rejected(self):
        q = JobQueue(lambda job: None, path=self.path, autostart=False)
        first, _ = q.submit(_params("A"), {})
        with self.assertRaises(DuplicateJob) as ctx:
            q.submit(_params("A"), {})
        self.assertIs(ctx.exception.job, first)
        q.submit(_params("A", deep_model="sonnet"), {})  # different settings are a different job
        with patch.object(job_queue, "MAX_QUEUED", 2):
            with self.assertRaises(QueueFull):
                q.submit(_params("C"), {})

    def test_cancel_queued_job_skips_it(self):
        runner = GatedRunner()
        q = self.queue(runner)
        q.submit(_params("A"), {})
        b, _ = q.submit(_params("B"), {})
        q.submit(_params("C"), {})
        self.assertTrue(q.cancel(b.id))
        self.assertEqual(b.status, "cancelled")
        self.assertTrue(_wait(lambda: runner.ran == ["A"]))
        runner.release()
        self.assertTrue(_wait(lambda: runner.ran == ["A", "C"]))
        runner.release()
        self.assertTrue(_wait(lambda: q.snapshot()[0] is None and not q.snapshot()[1]))
        self.assertEqual(runner.ran, ["A", "C"])
        self.assertFalse(q.cancel(b.id))  # already finished

    def test_worker_waits_while_busy(self):
        busy = threading.Event()
        busy.set()
        runner = GatedRunner()
        q = self.queue(runner, busy=busy.is_set)
        q.submit(_params("A"), {})
        time.sleep(0.1)
        self.assertEqual(runner.ran, [])
        busy.clear()
        self.assertTrue(_wait(lambda: runner.ran == ["A"]))
        runner.release()
        self.assertTrue(_wait(lambda: q.snapshot()[0] is None))

    def test_pending_jobs_survive_a_restart(self):
        q = JobQueue(lambda job: None, path=self.path, autostart=False)
        a, _ = q.submit(_params("A"), {"results_dir": "x"}, requester=" 민수 ")
        q.submit(_params("B"), {})
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual([s["params"]["ticker"] for s in saved], ["A", "B"])

        restored = JobQueue(lambda job: None, path=self.path, autostart=False)
        _, queued, _ = restored.snapshot()
        self.assertEqual([j.params["ticker"] for j in queued], ["A", "B"])
        self.assertEqual((queued[0].id, queued[0].requester, queued[0].config), (a.id, "민수", {"results_dir": "x"}))

    def test_runner_crash_does_not_stop_the_queue(self):
        ran = []

        def runner(job):
            ran.append(job.params["ticker"])
            if job.params["ticker"] == "A":
                raise RuntimeError("boom")
            job.status = "done"

        q = self.queue(runner)
        a, _ = q.submit(_params("A"), {})
        q.submit(_params("B"), {})
        self.assertTrue(_wait(lambda: ran == ["A", "B"] and not q.snapshot()[0]))
        self.assertEqual(a.status, "error")
        self.assertIn("boom", a.error)


class FakeGraph:
    """Calls a real LangChain chat model with the callbacks, the way the agents do."""

    calls_before_return = 3
    hooks = {}

    def __init__(self, selected_analysts, debug, config, callbacks):
        self.callbacks = callbacks
        self.llm = FakeListChatModel(responses=["ok"] * 10)

    def propagate(self, ticker, trade_date, user_notes="", position=None):
        self.hooks["user_notes"] = user_notes
        self.hooks["position"] = position
        for i in range(self.calls_before_return):
            self.llm.invoke("hi", config={"callbacks": self.callbacks, "metadata": {"langgraph_node": "Market Analyst"}})
            if i == 0 and "first_call" in self.hooks:
                self.hooks["first_call"]()
        return {"company_of_interest": ticker, "trade_date": trade_date, "final_trade_decision": "Buy"}, "Buy"



@pytest.mark.unit
class TestRunJob(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.config = {"results_dir": self.tmp.name}

    def tearDown(self):
        self.tmp.cleanup()
        FakeGraph.hooks.clear()

    def test_successful_run_is_saved_to_history(self):
        job = Job(params=_params(), config=self.config, requester="민수")
        with patch.object(job_queue.history_store, "HISTORY_DIR", Path(self.tmp.name) / "h"):
            run_job(job, FakeGraph)
            record = job_queue.history_store.load_run(job.record_id)
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(job.calls, 3)
        self.assertEqual(record["requester"], "민수")
        self.assertEqual(record["decision"], "Buy")
        self.assertNotIn("notes", record)
        steps = record["step_times"]
        self.assertEqual([s["node"] for s in steps], ["Market Analyst"])
        self.assertEqual(steps[0]["calls"], 3)
        self.assertGreaterEqual(steps[0]["sec"], 0)

    def test_step_times_split_agents_and_prep(self):
        job = Job(params=_params(), config=self.config)
        job.started = 100.0
        handler = job_queue.ProgressHandler(job)
        clock = iter([103.0, 104.0, 110.0, 130.0, 131.0])
        with patch.object(job_queue.time, "time", lambda: next(clock)):
            for node in ("Market Analyst", "Market Analyst", "News Analyst", "Trader"):
                handler.on_chat_model_start({}, [], metadata={"langgraph_node": node})
            handler.finish()
        self.assertEqual([(s["node"], s["sec"], s["calls"]) for s in job.step_times], [
            ("", 3.0, 0), ("Market Analyst", 7.0, 2), ("News Analyst", 20.0, 1), ("Trader", 1.0, 1),
        ])
        self.assertEqual(job.step_times[0]["step"], job_queue.PREP_LABEL)
        summary = job_queue.history_store.step_summary(job.step_times)
        self.assertEqual((summary["analysts"], summary["rest"], summary["total"]), (27.0, 4.0, 31.0))
        md = job_queue.history_store.step_table_md(job.step_times)
        self.assertIn("분석가 구간 합계 27초 (87%)", md)

    def test_notes_reach_the_graph_and_the_saved_record(self):
        job = Job(params=_params(notes="1월 말 주주환원 발표 예정"), config=self.config)
        with patch.object(job_queue.history_store, "HISTORY_DIR", Path(self.tmp.name) / "h"):
            run_job(job, FakeGraph)
            record = job_queue.history_store.load_run(job.record_id)
        self.assertEqual(FakeGraph.hooks["user_notes"], "1월 말 주주환원 발표 예정")
        self.assertEqual(record["notes"], "1월 말 주주환원 발표 예정")
        self.assertEqual(record["position"], {"held": False, "avg_price": None})  # default: not held

    def test_position_reaches_the_graph_and_the_saved_record(self):
        pos = {"held": True, "avg_price": 210000.0}
        job = Job(params=_params(position=pos), config=self.config)
        with patch.object(job_queue.history_store, "HISTORY_DIR", Path(self.tmp.name) / "h"):
            run_job(job, FakeGraph)
            record = job_queue.history_store.load_run(job.record_id)
        self.assertEqual(FakeGraph.hooks["position"], pos)
        self.assertEqual(record["position"], pos)

    def test_cancel_request_stops_at_the_next_llm_call(self):
        job = Job(params=_params(), config=self.config)
        FakeGraph.hooks["first_call"] = lambda: setattr(job, "cancel_requested", True)
        run_job(job, FakeGraph)
        self.assertEqual(job.status, "cancelled", job.error)
        self.assertEqual(job.calls, 1)
        self.assertEqual(job.record_id, "")

    def test_graph_error_is_recorded(self):
        def broken(**kwargs):
            raise ValueError("bad ticker")

        job = Job(params=_params(), config=self.config)
        run_job(job, broken)
        self.assertEqual(job.status, "error")
        self.assertIn("bad ticker", job.error)
        self.assertIsNotNone(job.finished)
