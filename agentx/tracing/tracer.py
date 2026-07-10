from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from agentx.exceptions import CIGateFailure
from agentx.tracing.ingest_client import IngestClient
from agentx.tracing.ci_types import CIRun, CIRunResult, CIRunStatus, CIQuestionScore

F = TypeVar("F", bound=Callable[..., Any])


def _safe_serialize(value: Any, depth: int = 0) -> Any:
    """Best-effort conversion to a JSON-safe structure, truncated to avoid huge payloads."""
    if depth > 3:
        return str(value)[:200]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v, depth + 1) for k, v in list(value.items())[:30]}
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v, depth + 1) for v in value[:30]]
    # Pydantic models, dataclasses, etc.
    if hasattr(value, "model_dump"):
        try:
            return _safe_serialize(value.model_dump(), depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _safe_serialize(vars(value), depth + 1)
        except Exception:
            pass
    return str(value)[:200]


def _capture_fn_input(fn: Callable, args: tuple, kwargs: dict) -> Optional[Dict[str, Any]]:
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        result = {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}
        return _safe_serialize(result) or None
    except Exception:
        return None


class _TraceSpan:
    """
    Returned by ``Tracer.trace()``.  Works as both a context manager and a
    decorator factory so the same object handles both usage patterns.
    """

    def __init__(
        self,
        tracer: "Tracer",
        name: str,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        framework: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._tracer = tracer
        self.name = name
        # Public: callers may reassign span.input inside the context manager
        self.input: Any = input
        self._metadata = metadata
        self._framework = framework
        self._model = model
        self._session_id = session_id

        # Fields the caller can set while inside the context manager
        self.output: Any = None
        self.tool_calls: list = []

        self._start: Optional[float] = None
        self._error: Optional[str] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "_TraceSpan":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, tb):
        latency_ms = int((time.time() - self._start) * 1000) if self._start else None
        if exc_val is not None and self._error is None:
            self._error = str(exc_val)
        self._tracer._send(
            name=self.name,
            input=_safe_serialize(self.input) if self.input is not None else None,
            output=_safe_serialize(self.output) if self.output is not None else None,
            latency_ms=latency_ms,
            error=self._error,
            metadata=self._metadata,
            framework=self._framework,
            model=self._model,
            tool_calls=self.tool_calls or None,
            session_id=self._session_id,
        )
        return False  # never suppress exceptions

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------

    def add_tool_call(
        self,
        name: str,
        *,
        input: Any = None,
        output: Any = None,
        latency_ms: Optional[int] = None,
    ) -> None:
        """Record a tool call made during this span."""
        self.tool_calls.append({
            "name": name,
            "input": _safe_serialize(input) if input is not None else None,
            "output": _safe_serialize(output) if output is not None else None,
            "latency_ms": latency_ms,
        })

    def set_error(self, message: str) -> None:
        """Mark this span as failed with the given error message."""
        self._error = message

    # ------------------------------------------------------------------
    # Decorator factory: span(fn) wraps fn synchronously or async
    # ------------------------------------------------------------------

    def __call__(self, fn: F) -> F:
        if asyncio.iscoroutinefunction(fn):
            return self._wrap_async(fn)  # type: ignore[return-value]
        return self._wrap_sync(fn)  # type: ignore[return-value]

    def _wrap_sync(self, fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            from agentx.integrations._perf import build_performance_summary
            captured_input = _capture_fn_input(fn, args, kwargs)
            start_t = time.time()
            error: Optional[str] = None
            output = None
            try:
                output = fn(*args, **kwargs)
                return output
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                end_t = time.time()
                latency_ms = int((end_t - start_t) * 1000)
                self._tracer._send(
                    name=self.name,
                    input=captured_input,
                    output=_safe_serialize(output) if output is not None else None,
                    latency_ms=latency_ms,
                    error=error,
                    metadata=self._metadata,
                    framework=self._framework,
                    model=self._model,
                    session_id=self._session_id,
                    performance_summary=build_performance_summary(
                        total_duration_ms=latency_ms,
                        execution_steps=[{
                            "name": "LLM Call 1",
                            "duration_ms": latency_ms,
                            "start_time": start_t,
                            "end_time": end_t,
                        }],
                        has_errors=error is not None,
                    ),
                )

        return wrapper  # type: ignore[return-value]

    def _wrap_async(self, fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            from agentx.integrations._perf import build_performance_summary
            captured_input = _capture_fn_input(fn, args, kwargs)
            start_t = time.time()
            error: Optional[str] = None
            output = None
            try:
                output = await fn(*args, **kwargs)
                return output
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                end_t = time.time()
                latency_ms = int((end_t - start_t) * 1000)
                self._tracer._send(
                    name=self.name,
                    input=captured_input,
                    output=_safe_serialize(output) if output is not None else None,
                    latency_ms=latency_ms,
                    error=error,
                    metadata=self._metadata,
                    framework=self._framework,
                    model=self._model,
                    session_id=self._session_id,
                    performance_summary=build_performance_summary(
                        total_duration_ms=latency_ms,
                        execution_steps=[{
                            "name": "LLM Call 1",
                            "duration_ms": latency_ms,
                            "start_time": start_t,
                            "end_time": end_t,
                        }],
                        has_errors=error is not None,
                    ),
                )

        return wrapper  # type: ignore[return-value]


class Tracer:
    """
    Production tracer attached to ``AgentX.tracer``.

    Usage — decorator::

        @agentx.tracer.trace("my-agent")
        def run(query: str) -> str:
            ...

    Usage — context manager::

        with agentx.tracer.trace("my-agent", input={"query": q}) as span:
            result = call_agent(q)
            span.output = result

    Usage — async::

        @agentx.tracer.trace("my-agent")
        async def run(query: str) -> str:
            ...
    """

    def __init__(self, ingest_client: IngestClient) -> None:
        self._client = ingest_client

    def trace(
        self,
        name: str,
        *,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        framework: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> _TraceSpan:
        """
        Return a :class:`_TraceSpan` that works as both a decorator and a
        context manager.
        """
        return _TraceSpan(
            tracer=self,
            name=name,
            input=input,
            metadata=metadata,
            framework=framework,
            model=model,
            session_id=session_id,
        )

    def flush(self, timeout: float = 5.0) -> None:
        """Block until all queued traces have been delivered."""
        self._client.flush(timeout)

    # ------------------------------------------------------------------
    # CI/CD evaluation
    # ------------------------------------------------------------------

    def run_eval(
        self,
        dataset_id: str,
        agent_fn: Callable[[str], str],
        *,
        agent_name: Optional[str] = None,
        pass_rate_threshold: Optional[float] = None,
        git_context: Optional[Dict[str, Any]] = None,
        concurrency: int = 1,
        fail_on_gate: bool = False,
        timeout_per_question: Optional[float] = None,
    ) -> CIRunResult:
        """
        Run the full CI/CD evaluation lifecycle in one call.

        Creates a CI run, calls ``agent_fn(query)`` for each test case,
        submits results to AgentX for scoring, finalizes the run, and
        returns the gate decision.

        Args:
            dataset_id:             EvaluationSettings ID (must have ci.enabled: true).
            agent_fn:               Function that takes a query string and returns
                                    the agent's response string.
            agent_name:             Label for this agent on the AgentX platform.
            pass_rate_threshold:    Override the dataset's passRateThreshold (0.0–1.0).
            git_context:            Dict with branch, commit_sha, pr_number, etc.
            concurrency:            Max parallel question invocations (default 1).
            fail_on_gate:           Raise CIGateFailure if gate is "fail".
            timeout_per_question:   Seconds to wait for agent_fn per question.

        Returns:
            CIRunResult with gate, pass_rate, scores, and violations.
        """
        run = self.create_ci_run(
            dataset_id,
            agent_name=agent_name,
            pass_rate_threshold=pass_rate_threshold,
            git_context=git_context,
        )

        def _process_case(tc: Any) -> CIQuestionScore:
            query = tc.query or ""
            start = time.time()
            error_output: Optional[str] = None
            output: Optional[str] = None
            try:
                if timeout_per_question:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        future = ex.submit(agent_fn, query)
                        output = future.result(timeout=timeout_per_question)
                else:
                    output = agent_fn(query)
            except Exception as exc:
                error_output = f"ERROR: {exc}"
            latency_ms = int((time.time() - start) * 1000)

            score = self.submit_result(
                run.run_id,
                tc.index,
                error_output if error_output is not None else (output or ""),
                input=query,
                latency_ms=latency_ms,
            )
            return score

        if concurrency <= 1:
            for tc in run.test_cases:
                score = _process_case(tc)
                if score.gate_fired:
                    # failFast triggered — run already finalized server-side
                    result = self.get_ci_run(run.run_id)
                    final = CIRunResult(
                        run_id=run.run_id,
                        gate="fail",
                        pass_rate=0.0,
                        total_questions=run.total_questions,
                        passed_questions=0,
                        finalized_at=result.finalized_at,
                    )
                    if fail_on_gate:
                        raise CIGateFailure(final)
                    return final
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = {ex.submit(_process_case, tc): tc for tc in run.test_cases}
                for future in concurrent.futures.as_completed(futures):
                    score = future.result()
                    if score.gate_fired:
                        ex.shutdown(wait=False, cancel_futures=True)
                        result = self.get_ci_run(run.run_id)
                        final = CIRunResult(
                            run_id=run.run_id,
                            gate="fail",
                            pass_rate=0.0,
                            total_questions=run.total_questions,
                            passed_questions=0,
                            finalized_at=result.finalized_at,
                        )
                        if fail_on_gate:
                            raise CIGateFailure(final)
                        return final

        result = self.finalize_ci_run(run.run_id)
        if fail_on_gate and result.gate == "fail":
            raise CIGateFailure(result)
        return result

    def create_ci_run(
        self,
        dataset_id: str,
        *,
        agent_name: Optional[str] = None,
        pass_rate_threshold: Optional[float] = None,
        git_context: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
    ) -> CIRun:
        """Create a CI run and receive test cases from the dataset."""
        return self._client.create_ci_run(
            dataset_id,
            agent_name=agent_name,
            pass_rate_threshold=pass_rate_threshold,
            git_context=git_context,
            workspace_id=workspace_id,
        )

    def submit_result(
        self,
        run_id: str,
        question_index: int,
        output: Any,
        *,
        input: Optional[Any] = None,
        latency_ms: Optional[int] = None,
    ) -> CIQuestionScore:
        """Submit an agent output for one CI run test case."""
        return self._client.submit_ci_result(
            run_id,
            question_index,
            output,
            input=input,
            latency_ms=latency_ms,
        )

    def finalize_ci_run(self, run_id: str) -> CIRunResult:
        """Finalize a CI run and return the gate result."""
        return self._client.finalize_ci_run(run_id)

    def get_ci_run(self, run_id: str) -> CIRunStatus:
        """Poll the current status of a CI run."""
        return self._client.get_ci_run(run_id)

    def evaluate_trace(
        self,
        trace_id: str,
        dataset_id: str,
        *,
        question_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Score a previously-ingested production trace against a dataset.

        Calls ``POST /ingest/traces/{trace_id}/evaluate`` synchronously and
        returns the scoring result. The trace's recorded input/output are used
        as-is — the agent is NOT re-run.

        Args:
            trace_id:       ID returned by a previous ``trace()`` call
                            (available as ``span._trace_id`` after flush).
            dataset_id:     EvaluationSettings ID to score against.
            question_index: Optional index into the dataset's questions array.
                            When supplied, that question's ``expectedResults``
                            is included in the scoring prompt.

        Returns:
            Dict with keys: ``run_id``, ``trace_id``, ``rating``,
            ``justification``, ``status``.
        """
        return self._client.evaluate_trace(trace_id, dataset_id, question_index=question_index)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send(self, **kwargs) -> None:
        payload = {k: v for k, v in kwargs.items() if v is not None}
        # Remap to snake_case wire format expected by the backend
        wire: Dict[str, Any] = {}
        if "name" in payload:
            wire["name"] = payload["name"]
        if "input" in payload:
            wire["input"] = payload["input"]
        if "output" in payload:
            wire["output"] = payload["output"]
        if "latency_ms" in payload:
            wire["latency_ms"] = payload["latency_ms"]
        if "error" in payload:
            wire["error"] = payload["error"]
        if "framework" in payload:
            wire["framework"] = payload["framework"]
        if "model" in payload:
            wire["model"] = payload["model"]
        if "tool_calls" in payload:
            wire["tool_calls"] = payload["tool_calls"]
        if "metadata" in payload:
            wire["metadata"] = payload["metadata"]
        if "session_id" in payload:
            wire["session_id"] = payload["session_id"]
        if "performance_summary" in payload:
            wire["performance_summary"] = payload["performance_summary"]
        if "input_tokens" in payload:
            wire["input_tokens"] = payload["input_tokens"]
        if "output_tokens" in payload:
            wire["output_tokens"] = payload["output_tokens"]
        self._client.enqueue(wire)
