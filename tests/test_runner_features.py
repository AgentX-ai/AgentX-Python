"""Runner-level tests for dataset splits, concurrent execution, output reuse, and the
fail-fast batch submission - all against a fake EvaluationsClient, no engine required."""

import threading
import time
from typing import Any, Dict, List, Optional

import pytest

from agentx.evaluations.client import EvaluationSubmissionError
from agentx.evaluations.models import (
    BatchAppendResponse,
    Dataset,
    EvaluationRun,
    EvaluationSubject,
)
from agentx.evaluations.runner import EvaluationRunContext, _build_cases


def make_dataset(**overrides: Any) -> Dataset:
    payload: Dict[str, Any] = {
        "_id": "ds-1",
        "name": "split dataset",
        "questions": [
            {"main_question": {"query": "q0", "splits": ["smoke"]}},
            {"main_question": {"query": "q1"}},
            {"main_question": {"query": "q2", "splits": ["smoke", "full"]}},
        ],
    }
    payload.update(overrides)
    return Dataset(**payload)


def make_run() -> EvaluationRun:
    return EvaluationRun(runId="run-1", datasetId="ds-1")


class FakeClient:
    def __init__(self, prior_run: Optional[Dict[str, Any]] = None, fail_batches: int = 0):
        self.batches: List[List[Any]] = []
        self._prior_run = prior_run
        self._fail_remaining = fail_batches

    def get_submitted_keys(self, run_id: str) -> List[str]:
        return []

    def append_results(self, run_id: str, batch_id: str, results: List[Any]) -> BatchAppendResponse:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise RuntimeError("engine down")
        self.batches.append(list(results))
        return BatchAppendResponse(
            runId=run_id, batchId=batch_id, accepted=len(results), duplicates=0, failedValidation=0
        )

    def get_run(self, run_id: str) -> Dict[str, Any]:
        assert self._prior_run is not None
        return self._prior_run


def make_context(client: FakeClient, split: Optional[str] = None) -> EvaluationRunContext:
    return EvaluationRunContext(
        client,  # type: ignore[arg-type]
        make_dataset(),
        make_run(),
        EvaluationSubject(),
        split=split,
    )


def test_build_cases_filters_by_split_and_keeps_indexes():
    cases = _build_cases(make_dataset(), make_run(), split="smoke")
    assert [c.question_index for c in cases] == [0, 2]
    assert [c.query for c in cases] == ["q0", "q2"]

    all_cases = _build_cases(make_dataset(), make_run())
    assert [c.question_index for c in all_cases] == [0, 1, 2]


def test_execute_runs_only_the_split(monkeypatch):
    monkeypatch.setenv("AGENTX_EVAL_QUIET", "1")
    client = FakeClient()
    ctx = make_context(client, split="smoke")
    seen: List[str] = []

    def agent(case):
        seen.append(case.query)
        return f"answer to {case.query}"

    ctx.execute(agent)
    assert seen == ["q0", "q2"]
    submitted = [r for batch in client.batches for r in batch]
    assert [r.question_index for r in submitted] == [0, 2]


def test_concurrent_execution_preserves_submission_order(monkeypatch):
    monkeypatch.setenv("AGENTX_EVAL_QUIET", "1")
    client = FakeClient()
    ctx = make_context(client)
    threads: List[str] = []

    def agent(case):
        threads.append(threading.current_thread().name)
        # The FIRST case is the slowest - order must still hold.
        time.sleep(0.2 if case.query == "q0" else 0.01)
        return f"answer to {case.query}"

    ctx.execute(agent, concurrency=3)
    submitted = [r for batch in client.batches for r in batch]
    assert [r.question_index for r in submitted] == [0, 1, 2]
    assert any(name != "MainThread" for name in threads)


def test_reuse_outputs_from_replays_matching_queries(monkeypatch):
    monkeypatch.setenv("AGENTX_EVAL_QUIET", "1")
    prior = {
        "results": [
            {"input": {"query": "q0"}, "output": {"text": "cached answer 0"}, "runNumber": 1},
            # q1's prior row errored - must NOT be reused.
            {"input": {"query": "q1"}, "output": {"text": "bad"}, "runNumber": 1, "status": "failed"},
        ]
    }
    client = FakeClient(prior_run=prior)
    ctx = make_context(client)
    ran: List[str] = []

    def agent(case):
        ran.append(case.query)
        return f"fresh answer to {case.query}"

    ctx.execute(agent, reuse_outputs_from="run-0")
    # q0 replayed from cache; q1 (failed before) and q2 (no cache) ran for real.
    assert ran == ["q1", "q2"]
    submitted = [r for batch in client.batches for r in batch]
    assert submitted[0].output == {"text": "cached answer 0"}
    assert submitted[0].metadata.get("reusedFromRun") == "run-0"
    assert submitted[1].output == {"text": "fresh answer to q1"}


def test_flush_batch_failure_raises_after_one_retry(monkeypatch):
    monkeypatch.setenv("AGENTX_EVAL_QUIET", "1")
    client = FakeClient(fail_batches=2)  # first attempt + its retry both fail
    ctx = make_context(client)

    with pytest.raises(EvaluationSubmissionError):
        ctx.execute(lambda case: "x")
