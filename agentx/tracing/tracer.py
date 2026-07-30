from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

from agentx.exceptions import CIGateFailure
from agentx.tracing.ingest_client import IngestClient
from agentx.tracing.ci_types import CIRun, CIRunResult, CIRunStatus, CIQuestionScore
from agentx.integrations._perf import (
    build_performance_summary,
    merge_retrieval_steps,
    merge_tool_call_steps,
)

F = TypeVar("F", bound=Callable[..., Any])

# Matches the generic "LLM Call N" step names integrations generate (e.g.
# langchain.py numbers steps within one top-level chain run). Used by
# _TraceSpan._merge_child_run to renumber them on merge — see there.
_LLM_CALL_NAME_RE = re.compile(r"^LLM Call \d+$")


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
        sync: bool = False,
        monitor: bool = False,
        pattern_ids: Optional[List[str]] = None,
    ) -> None:
        self._tracer = tracer
        self.name = name
        # Public: callers may reassign span.input inside the context manager
        self.input: Any = input
        self._metadata = metadata
        self._framework = framework
        self._model = model
        self._session_id = session_id
        # When True, __exit__ sends synchronously (blocking) instead of enqueueing, so trace_id
        # is populated by the time the `with` block exits — see Tracer.trace()'s sync param.
        self._sync = sync
        self._trace_id: Optional[str] = None
        # Monitor: check this trace against patterns immediately on ingest, no dashboard profile
        # required. pattern_ids (if given) fully defines what's checked — only those patterns run,
        # the built-in checks are skipped. See Tracer.trace()'s monitor/pattern_ids params.
        self._monitor = monitor
        self._pattern_ids = pattern_ids

        # Fields the caller can set while inside the context manager
        self.output: Any = None
        self.tool_calls: list = []

        self._start: Optional[float] = None
        self._error: Optional[str] = None

        # Populated by auto-instrumented calls (e.g. patched Anthropic
        # client, AgentXCallbackHandler) made while this span is the
        # tracer's active span, so a whole multi-call / multi-chain run
        # collapses into one trace instead of one trace per call. See
        # Tracer.current_span / _record_llm_call / _merge_child_run.
        self._execution_steps: list = []
        self._retrieval_steps: list = []
        self._captured_model: Optional[str] = None
        # Adopted from a merged child run (e.g. AgentXCallbackHandler) when this span itself
        # wasn't opened with an explicit framework= — see _merge_child_run below.
        self._captured_framework: Optional[str] = None
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        # Guards _merge_child_run — with Tracer.use_span(), multiple threads
        # (e.g. a ThreadPoolExecutor) can merge into this span concurrently.
        self._merge_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "_TraceSpan":
        self._start = time.time()
        self._tracer._push_active_span(self)
        return self

    def __exit__(self, exc_type, exc_val, tb):
        self._tracer._pop_active_span(self)
        latency_ms = int((time.time() - self._start) * 1000) if self._start else None
        if exc_val is not None and self._error is None:
            self._error = str(exc_val)

        # Auto-instrumented integrations (patched Anthropic client, AgentXCallbackHandler, ...)
        # populate _execution_steps via _record_llm_call while this span is active. Wrapping a raw
        # API call with no such integration (e.g. a bare `openai` call) never populates it — without
        # this fallback the Execution Timeline would be empty despite the span having real
        # input/output, since nothing else here describes what the wrapped code actually did.
        # Mirrors what the @tracer.trace(...) decorator form has always synthesized for exactly
        # this case (see _wrap_sync/_wrap_async below).
        execution_steps = self._execution_steps or (
            [
                {
                    "name": "LLM Call 1",
                    "duration_ms": latency_ms or 0,
                    "start_time": self._start,
                    "end_time": time.time(),
                    "model": self._model or self._captured_model,
                    "input": _safe_serialize(self.input) if self.input is not None else None,
                    "output": _safe_serialize(self.output) if self.output is not None else None,
                }
            ]
            if self.input is not None or self.output is not None
            else []
        )

        perf = build_performance_summary(
            total_duration_ms=latency_ms or 0,
            execution_steps=execution_steps,
            tool_call_steps=[
                {
                    "name": tc.get("name"),
                    "duration_ms": tc.get("latency_ms") or 0,
                    "start_time": tc.get("start_time"),
                    "end_time": tc.get("end_time"),
                    "input": tc.get("input"),
                    "output": tc.get("output"),
                }
                for tc in self.tool_calls
            ],
            retrieval_steps=self._retrieval_steps,
            has_errors=self._error is not None,
        )

        self._trace_id = self._tracer._send(
            sync=self._sync,
            monitor=self._monitor or None,
            pattern_ids=self._pattern_ids,
            name=self.name,
            input=_safe_serialize(self.input) if self.input is not None else None,
            output=_safe_serialize(self.output) if self.output is not None else None,
            latency_ms=latency_ms,
            error=self._error,
            metadata=self._metadata,
            framework=self._framework or self._captured_framework,
            model=self._model or self._captured_model,
            tool_calls=self.tool_calls or None,
            session_id=self._session_id,
            performance_summary=perf,
            input_tokens=self._input_tokens or None,
            output_tokens=self._output_tokens or None,
        )
        return False  # never suppress exceptions

    @property
    def trace_id(self) -> Optional[str]:
        """The ingested trace's id — only populated once this span has exited AND it was opened
        with ``tracer.trace(..., sync=True)``. ``None`` for the default (async/enqueued) mode,
        since there's nothing to wait on for a same-call id."""
        return self._trace_id

    # ------------------------------------------------------------------
    # Called by auto-instrumented integrations (e.g. patch_anthropic_client)
    # when this span is the tracer's active span, instead of them sending
    # their own independent trace per call.
    # ------------------------------------------------------------------

    def _record_llm_call(
        self,
        *,
        duration_ms: float,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        input: Any = None,
        output: Any = None,
        model: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        """Append a single LLM-call execution step (e.g. one patched Anthropic call)."""
        self._merge_child_run(
            execution_steps=[{
                "name": f"LLM Call {len(self._execution_steps) + 1}",
                "duration_ms": duration_ms,
                "start_time": start_time,
                "end_time": end_time,
                "model": model,
                "input": input,
                "output": output,
                "inputTokenSize": input_tokens,
                "outputTokenSize": output_tokens,
            }],
            input=input,
            output=output,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _merge_child_run(
        self,
        *,
        execution_steps: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        retrieval_steps: Optional[List[Dict[str, Any]]] = None,
        input: Any = None,
        output: Any = None,
        model: Optional[str] = None,
        framework: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        """
        Fold a whole auto-instrumented sub-run (e.g. one top-level LangChain
        chain/agent/retriever invocation) into this span, instead of it
        becoming its own independent trace.
        """
        with self._merge_lock:
            if execution_steps:
                # Step names like "LLM Call N" are numbered locally within
                # whatever sub-run produced them (e.g. one LangChain chain
                # invocation numbers its own calls 1, 2, 3...). When several
                # sub-runs merge into this span — e.g. a sequence of
                # single-call specialist chains all folded into one
                # orchestrator span via use_span() — renumber so the merged
                # trace doesn't end up with several "LLM Call 1" entries.
                next_n = len(self._execution_steps) + 1
                for offset, step in enumerate(execution_steps):
                    name = step.get("name")
                    if isinstance(name, str) and _LLM_CALL_NAME_RE.match(name):
                        step["name"] = f"LLM Call {next_n + offset}"
            self._execution_steps.extend(execution_steps or [])
            self.tool_calls.extend(tool_calls or [])
            self._retrieval_steps.extend(retrieval_steps or [])
            if self.input is None and input is not None:
                self.input = input
            if output is not None:
                self.output = output
            if model and not self._captured_model:
                self._captured_model = model
            if framework and not self._captured_framework:
                self._captured_framework = framework
            if input_tokens:
                self._input_tokens += input_tokens
            if output_tokens:
                self._output_tokens += output_tokens

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
                            "model": self._model,
                            "input": captured_input,
                            "output": _safe_serialize(output) if output is not None else None,
                        }],
                        has_errors=error is not None,
                    ),
                )

        return wrapper  # type: ignore[return-value]

    def _wrap_async(self, fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
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
                            "model": self._model,
                            "input": captured_input,
                            "output": _safe_serialize(output) if output is not None else None,
                        }],
                        has_errors=error is not None,
                    ),
                )

        return wrapper  # type: ignore[return-value]


class _RetrievalRecorder:
    """Handle yielded by ``Tracer.trace_retrieval()`` — set ``doc_count``/``output`` inside the block."""

    def __init__(self) -> None:
        self.doc_count: Optional[int] = None
        self.output: Any = None


class _ToolCallRecorder:
    """Handle yielded by ``Tracer.trace_tool_call()`` — set ``output`` inside the block."""

    def __init__(self) -> None:
        self.output: Any = None


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
        self._pending_retrievals: List[Dict[str, Any]] = []
        self._pending_tool_calls: List[Dict[str, Any]] = []
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Active-span stack (per thread) — lets auto-instrumented integrations
    # (e.g. patch_anthropic_client) detect they're running inside a
    # `with tracer.trace(...)` block and attach to it as an LLM-call step
    # instead of sending their own independent trace.
    # ------------------------------------------------------------------

    def _get_span_stack(self) -> List["_TraceSpan"]:
        stack = getattr(self._local, "span_stack", None)
        if stack is None:
            stack = []
            self._local.span_stack = stack
        return stack

    def _push_active_span(self, span: "_TraceSpan") -> None:
        self._get_span_stack().append(span)

    def _pop_active_span(self, span: "_TraceSpan") -> None:
        stack = self._get_span_stack()
        if stack and stack[-1] is span:
            stack.pop()
        elif span in stack:
            stack.remove(span)

    @property
    def current_span(self) -> Optional["_TraceSpan"]:
        """The innermost ``with tracer.trace(...)`` span active on this thread, if any."""
        stack = self._get_span_stack()
        return stack[-1] if stack else None

    @contextmanager
    def use_span(self, span: "_TraceSpan") -> Iterator["_TraceSpan"]:
        """
        Make ``span`` (created on another thread) the active span for the
        duration of this block, on *this* thread. The active-span stack is
        thread-local, so work submitted to a ``ThreadPoolExecutor`` or run on
        any other thread doesn't automatically see a span opened on the
        calling thread — wrap the worker function body in this to attach it::

            with tracer.trace("orchestrator") as span:
                def worker():
                    with tracer.use_span(span):
                        chain.invoke(..., config={"callbacks": [handler]})

                with ThreadPoolExecutor(max_workers=2) as ex:
                    ex.submit(worker).result()

        Safe to call from multiple threads concurrently for the same span —
        each thread pushes/pops on its own stack.
        """
        self._push_active_span(span)
        try:
            yield span
        finally:
            self._pop_active_span(span)

    def record_tool_call(
        self,
        name: str,
        *,
        input: Any = None,
        output: Any = None,
        latency_ms: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """
        Manually record a tool call that an auto-instrumented framework
        integration can't see — e.g. a hand-rolled Anthropic/OpenAI tool-use
        loop where the tool executes in plain Python between two
        ``messages.create()`` calls. Queued and attached to both the
        ``tool_calls`` list and the ``performance_summary`` of the next
        trace this tracer sends.
        """
        self._pending_tool_calls.append({
            "name": name,
            "input": _safe_serialize(input) if input is not None else None,
            "output": _safe_serialize(output) if output is not None else None,
            "latency_ms": latency_ms,
            "duration_ms": latency_ms if latency_ms is not None else 0,
            "start_time": start_time,
            "end_time": end_time,
        })

    @contextmanager
    def trace_tool_call(self, name: str, *, input: Any = None) -> Iterator["_ToolCallRecorder"]:
        """
        Context manager that times a tool call and records it via
        :meth:`record_tool_call` on exit::

            with tracer.trace_tool_call("policy_lookup", input=topic) as t:
                result = policy_lookup(topic)
                t.output = result
        """
        start_t = time.time()
        recorder = _ToolCallRecorder()
        try:
            yield recorder
        finally:
            end_t = time.time()
            self.record_tool_call(
                name,
                input=input,
                output=recorder.output,
                latency_ms=int((end_t - start_t) * 1000),
                start_time=start_t,
                end_time=end_t,
            )

    def record_retrieval(
        self,
        name: str = "Retrieval",
        *,
        query: Optional[str] = None,
        doc_count: Optional[int] = None,
        output: Any = None,
        duration_ms: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """
        Manually record a knowledge-base / vector-store retrieval that an
        auto-instrumented framework integration can't see on its own — e.g. a
        hand-rolled RAG lookup wrapped around a raw Anthropic/OpenAI call or a
        CrewAI kickoff. Queued and attached to the ``performance_summary`` of
        the next trace this tracer sends.
        """
        self._pending_retrievals.append({
            "name": name,
            "duration_ms": duration_ms or 0,
            "start_time": start_time,
            "end_time": end_time,
            "query": query,
            "doc_count": doc_count,
            "output": _safe_serialize(output) if output is not None else None,
        })

    @contextmanager
    def trace_retrieval(self, name: str = "Retrieval", *, query: Optional[str] = None) -> Iterator["_RetrievalRecorder"]:
        """
        Context manager that times a retrieval and records it via
        :meth:`record_retrieval` on exit::

            with tracer.trace_retrieval("kb_search", query=question) as r:
                docs = retrieve(question)
                r.doc_count = len(docs)
                r.output = docs
        """
        start_t = time.time()
        recorder = _RetrievalRecorder()
        try:
            yield recorder
        finally:
            end_t = time.time()
            self.record_retrieval(
                name,
                query=query,
                doc_count=recorder.doc_count,
                output=recorder.output,
                duration_ms=(end_t - start_t) * 1000,
                start_time=start_t,
                end_time=end_t,
            )

    def trace(
        self,
        name: str,
        *,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        framework: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        sync: bool = False,
        monitor: bool = False,
        pattern_ids: Optional[List[str]] = None,
    ) -> _TraceSpan:
        """
        Return a :class:`_TraceSpan` that works as both a decorator and a
        context manager.

        By default the trace is queued and sent on a background thread — fire-and-forget, never
        blocks the caller, but there's no way to learn the resulting trace_id. Pass ``sync=True``
        to send it synchronously instead (blocks until ingested) so ``span.trace_id`` is populated
        once the ``with`` block exits — e.g. to attach the trace to an evaluation result::

            with client.tracer.trace("support_agent_call", framework="openai", sync=True) as span:
                resp = call_llm(...)
                span.output = resp
            return {"output": resp, "trace_id": span.trace_id}

        Pass ``monitor=True`` to check this trace against Monitor patterns immediately, with no
        dashboard profile required. ``pattern_ids`` (ids from ``client.monitor.patterns.builder(
        ...).publish()``) restricts detection to exactly those patterns; omit it to run the full
        default sweep (built-in checks plus every enabled workspace pattern)::

            pattern = client.monitor.patterns.builder(
                name="Promises a refund", detector_kind="semantic",
                semantic_prompt="The response promises a refund.",
            ).publish()

            with client.tracer.trace("support_agent_call", monitor=True, pattern_ids=[pattern.id]) as span:
                span.output = call_llm(...)
        """
        return _TraceSpan(
            tracer=self,
            name=name,
            input=input,
            metadata=metadata,
            framework=framework,
            model=model,
            session_id=session_id,
            sync=sync,
            monitor=monitor,
            pattern_ids=pattern_ids,
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
            trace_id:       ID of a trace ingested via ``trace(..., sync=True)``
                            (available as ``span.trace_id`` once that `with` block exits).
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

    def _send(self, sync: bool = False, **kwargs) -> Optional[str]:
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
        if "monitor" in payload:
            wire["monitor"] = payload["monitor"]
        if "pattern_ids" in payload:
            wire["pattern_ids"] = payload["pattern_ids"]

        pending_retrievals, self._pending_retrievals = self._pending_retrievals, []
        pending_tool_calls, self._pending_tool_calls = self._pending_tool_calls, []

        if pending_tool_calls:
            wire["tool_calls"] = list(wire.get("tool_calls") or []) + [
                {
                    "name": t["name"],
                    "input": t["input"],
                    "output": t["output"],
                    "latency_ms": t["latency_ms"],
                }
                for t in pending_tool_calls
            ]

        if pending_retrievals or pending_tool_calls:
            if "performance_summary" not in wire:
                wire["performance_summary"] = build_performance_summary(
                    total_duration_ms=wire.get("latency_ms") or 0,
                    has_errors="error" in wire,
                )
            if pending_retrievals:
                wire["performance_summary"] = merge_retrieval_steps(
                    wire["performance_summary"], pending_retrievals
                )
            if pending_tool_calls:
                wire["performance_summary"] = merge_tool_call_steps(
                    wire["performance_summary"], pending_tool_calls
                )

        if sync:
            return self._client.send_trace_sync(wire)
        self._client.enqueue(wire)
        return None
