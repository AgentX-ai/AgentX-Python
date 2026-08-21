from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from agentx.util import _DEFAULT_API_BASE as _UTIL_API_BASE, endpoint_missing
from agentx.exceptions import AgentXAPIError, CINotEnabled, DatasetNotFound, EndpointNotAvailable
from agentx.tracing.ci_types import (
    CIRun,
    CIRunResult,
    CIRunStatus,
    CIQuestionScore,
    CITestCase,
    ThresholdViolation,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 2.0, 4.0]
_QUEUE_MAX = 500


class IngestClient:
    """
    Non-blocking client for POST /ingest/traces.

    Traces are queued in memory and drained by a background daemon thread
    so they never block the agent's critical path.
    """

    def __init__(
        self,
        api_key: str,
        sdk_version: str = "unknown",
        base_url: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise ValueError("AGENTX_API_KEY is required")

        self._workspace_id = workspace_id or os.getenv("AGENTX_WORKSPACE_ID")

        self._api_key = api_key
        self._sdk_version = sdk_version

        _base = (base_url or os.getenv("AGENTX_API_BASE_URL", _UTIL_API_BASE)).rstrip("/")
        # Strip the custom-agent-evaluations suffix if someone passes the eval base URL
        if _base.endswith("/custom-agent-evaluations"):
            _base = _base[: -len("/custom-agent-evaluations")]
        self._endpoint = f"{_base}/ingest/traces"

        self._session = requests.Session()
        self._session.headers.update(
            {
                "x-api-key": self._api_key,
                "Content-Type": "application/json",
                "User-Agent": f"agentx-python/{self._sdk_version}",
            }
        )

        self._queue: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue(maxsize=_QUEUE_MAX)
        self._worker = threading.Thread(target=self._drain, daemon=True, name="agentx-ingest")
        self._worker.start()

        # Base URL (without the /ingest/traces suffix) for synchronous calls like evaluate_trace
        self._base_url = _base

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def enqueue(self, payload: Dict[str, Any]) -> None:
        """Add a trace payload to the send queue. Never blocks; drops silently on overflow."""
        if self._workspace_id:
            payload = {**payload, "workspaceId": self._workspace_id}
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            logger.debug("agentx ingest queue full - trace dropped")

    def flush(self, timeout: float = 5.0) -> None:
        """Block until all queued traces have been sent (or timeout elapses)."""
        self._queue.join()

    def send_trace_sync(self, payload: Dict[str, Any]) -> Optional[str]:
        """
        Send a trace payload synchronously and return the ingested trace's id, or ``None`` on
        failure. Used by ``Tracer.trace(..., sync=True)`` when the caller needs the trace_id back
        immediately (e.g. to attach it to an evaluation result) - unlike ``enqueue()``, this blocks
        and does not retry, trading the tracer's usual fire-and-forget guarantee for a same-call
        result. Never raises; a failed send just means no trace_id (never blocks the caller's eval
        run over a tracing hiccup).
        """
        if self._workspace_id:
            payload = {**payload, "workspaceId": self._workspace_id}
        try:
            resp = self._session.post(self._endpoint, json=payload, timeout=10)
        except requests.RequestException as exc:
            logger.debug("agentx ingest sync send error: %s", exc)
            return None
        if not resp.ok:
            logger.debug("agentx ingest sync HTTP %d: %s", resp.status_code, resp.text[:200])
            return None
        try:
            return resp.json().get("trace_id")
        except Exception:
            return None

    def evaluate_trace(
        self,
        trace_id: str,
        dataset_id: str,
        *,
        question_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Synchronously score a previously-ingested trace against a dataset.

        The trace's recorded input/output are used as the pre-computed agent
        result - the agent is NOT re-run.

        Returns a dict with keys: run_id, trace_id, rating, justification, status.
        Raises requests.HTTPError on non-2xx responses.
        """
        url = f"{self._base_url}/ingest/traces/{trace_id}/evaluate"
        payload: Dict[str, Any] = {"datasetId": dataset_id}
        if question_index is not None:
            payload["question_index"] = question_index
        if self._workspace_id:
            payload["workspaceId"] = self._workspace_id

        resp = self._session.post(url, json=payload, timeout=60)
        self._raise_for_ci_status(resp, "evaluate_trace()")
        return resp.json()

    # ------------------------------------------------------------------
    # CI/CD evaluation methods (synchronous)
    # ------------------------------------------------------------------

    def create_ci_run(
        self,
        dataset_id: str,
        *,
        agent_name: Optional[str] = None,
        pass_rate_threshold: Optional[float] = None,
        git_context: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
    ) -> CIRun:
        """Create a CI run and return test cases from the dataset."""
        payload: Dict[str, Any] = {"dataset_id": dataset_id}
        if agent_name:
            payload["agent_name"] = agent_name
        if pass_rate_threshold is not None:
            payload["pass_rate_threshold"] = pass_rate_threshold
        if git_context:
            payload["git_context"] = git_context
        resolved_workspace = workspace_id or self._workspace_id
        if resolved_workspace:
            payload["workspaceId"] = resolved_workspace

        url = f"{self._base_url}/ingest/ci-runs"
        resp = self._session.post(url, json=payload, timeout=30)
        self._raise_for_ci_status(resp, "create_ci_run()")

        data = resp.json()
        return CIRun(
            run_id=data["run_id"],
            dataset_id=data["dataset_id"],
            total_questions=data["total_questions"],
            test_cases=[CITestCase(index=tc["index"], query=tc.get("query")) for tc in data.get("test_cases", [])],
            expires_at=data["expires_at"],
        )

    def submit_ci_result(
        self,
        run_id: str,
        question_index: int,
        output: Any,
        *,
        input: Optional[Any] = None,
        latency_ms: Optional[int] = None,
    ) -> CIQuestionScore:
        """Submit an agent result for one test case and receive the score."""
        payload: Dict[str, Any] = {
            "question_index": question_index,
            "output": output,
        }
        if input is not None:
            payload["input"] = input
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms

        url = f"{self._base_url}/ingest/ci-runs/{run_id}/results"
        resp = self._session.post(url, json=payload, timeout=60)
        self._raise_for_ci_status(resp, "submit_result()")

        data = resp.json()
        return CIQuestionScore(
            question_index=data["question_index"],
            rating=data["rating"],
            justification=data["justification"],
            passed=data["passed"],
            gate_fired=data.get("gate_fired", False),
        )

    def finalize_ci_run(self, run_id: str) -> CIRunResult:
        """Finalize the run and return the gate result."""
        url = f"{self._base_url}/ingest/ci-runs/{run_id}/finalize"
        resp = self._session.post(url, json={}, timeout=60)
        self._raise_for_ci_status(resp, "finalize_ci_run()")
        return self._parse_ci_result(resp.json())

    def get_ci_run(self, run_id: str) -> CIRunStatus:
        """Poll the status of a CI run."""
        url = f"{self._base_url}/ingest/ci-runs/{run_id}"
        resp = self._session.get(url, timeout=15)
        self._raise_for_ci_status(resp, "get_ci_run()")
        data = resp.json()
        return CIRunStatus(
            run_id=data["run_id"],
            status=data["status"],
            gate=data.get("gate"),
            results_submitted=data["results_submitted"],
            total_questions=data["total_questions"],
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            finalized_at=data.get("finalized_at"),
            git_context=data.get("git_context"),
        )

    def get_dataset_test_cases(
        self,
        dataset_id: str,
        *,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch test case queries for a dataset without creating a CI run."""
        params: Dict[str, str] = {}
        resolved_workspace = workspace_id or self._workspace_id
        if resolved_workspace:
            params["workspaceId"] = resolved_workspace
        url = f"{self._base_url}/ingest/datasets/{dataset_id}/test-cases"
        resp = self._session.get(url, params=params, timeout=15)
        self._raise_for_ci_status(resp, "get_dataset_test_cases()")
        return resp.json()

    # ------------------------------------------------------------------
    # CI helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ci_result(data: Dict[str, Any]) -> CIRunResult:
        scores: List[CIQuestionScore] = [
            CIQuestionScore(
                question_index=s["question_index"],
                rating=s["rating"],
                justification=s["justification"],
                passed=s["passed"],
                input=s.get("input"),
                output=s.get("output"),
            )
            for s in data.get("scores", [])
        ]
        violations: List[ThresholdViolation] = [
            ThresholdViolation(
                question_index=v["question_index"],
                metric=v["metric"],
                threshold=v["threshold"],
                actual=v["actual"],
                question_text=v.get("question_text", ""),
            )
            for v in data.get("violations", [])
        ]
        return CIRunResult(
            run_id=data["run_id"],
            gate=data["gate"],
            pass_rate=data["pass_rate"],
            total_questions=data["total_questions"],
            passed_questions=data["passed_questions"],
            scores=scores,
            violations=violations,
            finalized_at=data.get("finalized_at"),
        )

    @staticmethod
    def _raise_for_ci_status(resp: requests.Response, call: str = "This call") -> None:
        if resp.ok:
            return
        try:
            body = resp.json()
            message = body.get("message") or body.get("error") or resp.text[:200]
        except Exception:
            message = resp.text[:200]

        # Every 404 here used to become DatasetNotFound, including the ones that mean the route
        # does not exist on this deployment. Against a self-host engine - which serves none of the
        # CI-run surface - that told callers their dataset was gone seconds after they created it.
        if endpoint_missing(resp):
            raise EndpointNotAvailable(call, resp.request.method or "?", resp.url)
        if resp.status_code == 404:
            raise DatasetNotFound(message)
        if resp.status_code == 400 and "ci" in message.lower():
            raise CINotEnabled(message)
        raise AgentXAPIError(message, status_code=resp.status_code)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drain(self) -> None:
        while True:
            payload = self._queue.get()
            if payload is None:
                self._queue.task_done()
                break
            try:
                self._send(payload)
            except Exception as exc:
                logger.debug("agentx ingest send error: %s", exc)
            finally:
                self._queue.task_done()

    def _send(self, payload: Dict[str, Any]) -> None:
        last_exc: Optional[Exception] = None
        for attempt, wait in enumerate([0.0] + _RETRY_BACKOFF):
            if wait:
                time.sleep(wait)
            try:
                resp = self._session.post(self._endpoint, json=payload, timeout=10)
            except requests.RequestException as exc:
                last_exc = exc
                continue

            if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                last_exc = Exception(f"HTTP {resp.status_code}")
                continue
            if not resp.ok:
                logger.debug("agentx ingest HTTP %d: %s", resp.status_code, resp.text[:200])
                return
            return

        logger.debug("agentx ingest failed after retries: %s", last_exc)
