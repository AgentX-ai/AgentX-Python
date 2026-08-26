from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar
from uuid import uuid4

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
        sync: bool = False,
        monitor: bool = False,
        pattern_ids: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        self._tracer = tracer
        self.name = name
        # Public: callers may reassign span.input inside the context manager
        self.input: Any = input
        self._metadata = metadata
        self._framework = framework
        self._model = model
        self._session_id = session_id
        # Disambiguator for when `name` alone isn't enough - pass an already-known agent id (e.g.
        # from a prior GET /agents lookup) to pin this trace to that exact agent. None (the
        # default) resolves from `name` alone server-side, one stable agent per distinct name.
        self._agent_id = agent_id
        # When True, __exit__ sends synchronously (blocking) instead of enqueueing, so trace_id
        # is populated by the time the `with` block exits - see Tracer.trace()'s sync param.
        self._sync = sync
        self._trace_id: Optional[str] = None

        # Real span hierarchy - every span always gets an id and links to its real parent (if
        # any) and a shared session_id, the same model AgentX's OTel ingestion path uses. A
        # nested `with tracer.trace(...)` block, or any auto-instrumented call made while this
        # span is active (a patched Anthropic/OpenAI/Google GenAI/LiteLLM client, or a framework
        # integration like AgentXCallbackHandler), becomes its own linked child-span row - see
        # __enter__/_merge_child_run/child_span.
        self._span_id = uuid4().hex
        self._parent_span_id: Optional[str] = None
        # Numbers auto-named "LLM Call N"/"Retrieval N" child spans - see _merge_child_run.
        self._child_span_count = 0
        # Monitor: True checks this trace against patterns immediately on ingest, no dashboard
        # profile required; False explicitly OPTS OUT of every ingest-time check (patterns,
        # built-ins, online/custom evaluators, topics) - what eval-run traces send, since the
        # run's own evaluator already judges each case. None (default) leaves the server's
        # standard behavior. pattern_ids (if given) fully defines what's checked. See
        # Tracer.trace()'s monitor/pattern_ids params.
        self._monitor = monitor
        self._pattern_ids = pattern_ids

        # Fields the caller can set while inside the context manager
        self.output: Any = None
        self.tool_calls: list = []

        self._start: Optional[float] = None
        self._error: Optional[str] = None

        self._captured_model: Optional[str] = None
        # Adopted from a merged child run (e.g. AgentXCallbackHandler) when this span itself
        # wasn't opened with an explicit framework= - see _merge_child_run below.
        self._captured_framework: Optional[str] = None
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        # Subsets of _input_tokens (not additional tokens) - a prompt-caching write/read, when the
        # provider reports one. See _record_llm_call/child_span for where these get populated.
        self._cache_read_tokens: int = 0
        self._cache_write_tokens: int = 0
        # Guards _merge_child_run - with Tracer.use_span(), multiple threads
        # (e.g. a ThreadPoolExecutor) can merge into this span concurrently.
        self._merge_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "_TraceSpan":
        self._start = time.time()
        # Resolve real span hierarchy against whatever's currently active on this thread, before
        # pushing self (so `parent` here is the actual enclosing span, not self).
        parent = self._tracer.current_span
        if parent is not None:
            self._parent_span_id = parent._span_id
            if self._session_id is None:
                self._session_id = parent._session_id
        elif self._session_id is None:
            self._session_id = f"sdk_{uuid4().hex}"
        self._tracer._push_active_span(self)
        return self

    def __exit__(self, exc_type, exc_val, tb):
        self._tracer._pop_active_span(self)
        latency_ms = int((time.time() - self._start) * 1000) if self._start else None
        if exc_val is not None and self._error is None:
            self._error = str(exc_val)

        if self._sync and self._parent_span_id is None:
            # sync=True means the WHOLE tree is delivered before this block returns: child
            # spans (tool calls, LLM calls) were enqueued asynchronously during the block, so
            # drain them before the root's own synchronous send. Without this, read-after-trace
            # intermittently misses children (root lands, children still in flight) - the exact
            # race the enterprise assessment reproduced (P0.1). Bounded by the same 5s budget
            # flush() uses; child-only spans keep their async fire-and-forget behavior.
            self._tracer.flush(timeout=5.0)

        self._trace_id = self._tracer._send(
            sync=self._sync,
            monitor=self._monitor,
            pattern_ids=self._pattern_ids,
            name=self.name,
            agent_id=self._agent_id,
            input=_safe_serialize(self.input) if self.input is not None else None,
            output=_safe_serialize(self.output) if self.output is not None else None,
            latency_ms=latency_ms,
            error=self._error,
            metadata=self._metadata,
            framework=self._framework or self._captured_framework,
            model=self._model or self._captured_model,
            tool_calls=self.tool_calls or None,
            session_id=self._session_id,
            input_tokens=self._input_tokens or None,
            output_tokens=self._output_tokens or None,
            cache_read_tokens=self._cache_read_tokens or None,
            cache_write_tokens=self._cache_write_tokens or None,
            span_id=self._span_id,
            parent_span_id=self._parent_span_id,
            started_at_unix_nano=str(int(self._start * 1_000_000_000)) if self._start else None,
        )
        return False  # never suppress exceptions

    @property
    def span_id(self) -> str:
        """This span's id - usable to parent a further span via child_span() or an explicit
        session_id/parent chain."""
        return self._span_id

    @property
    def trace_id(self) -> Optional[str]:
        """The ingested trace's id - only populated once this span has exited AND it was opened
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
        cache_read_tokens: Optional[int] = None,
        cache_write_tokens: Optional[int] = None,
    ) -> None:
        """Record one LLM-call child span (e.g. one patched Anthropic call) under this span -
        name left unset so _merge_child_run auto-numbers it "LLM Call N"."""
        self._merge_child_run(
            execution_steps=[{
                "duration_ms": duration_ms,
                "start_time": start_time,
                "end_time": end_time,
                "model": model,
                "input": input,
                "output": output,
                "inputTokenSize": input_tokens,
                "outputTokenSize": output_tokens,
                "cacheReadTokenSize": cache_read_tokens,
                "cacheWriteTokenSize": cache_write_tokens,
            }],
            input=input,
            output=output,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

    def child_span(
        self,
        name: str,
        *,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration_ms: Optional[float] = None,
        input: Any = None,
        output: Any = None,
        model: Optional[str] = None,
        framework: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cache_read_tokens: Optional[int] = None,
        cache_write_tokens: Optional[int] = None,
        error: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        span_kind: Optional[str] = None,
    ) -> "_TraceSpan":
        """
        Send one real child-span row parented to this span, with explicit timing (the caller's
        own recorded start/end, not wall-clock time.time() at call time) - the
        AgentXCallbackHandler / _merge_child_run counterpart for integrations that already track
        their own real per-step identity and timing (LangChain's run_id/parent_run_id,
        LlamaIndex's parent_id, the OpenAI Agents SDK's own span objects) and want to parent a new
        child under a specific span they're holding a reference to - not just whatever's on top of
        the tracer's thread-local active-span stack.

        Returns the child span (its ``.span_id`` can parent a further-nested grandchild via
        another ``child_span()`` call on it). The returned span is not pushed onto the
        active-span stack and has already been sent by the time this returns, since the caller
        supplies finished timing rather than opening a ``with`` block.
        """
        child = _TraceSpan(
            tracer=self._tracer,
            name=name,
            framework=framework or self._framework or self._captured_framework,
            model=model,
            session_id=self._session_id,
        )
        child._parent_span_id = self._span_id
        child.input = input
        child.output = output
        child.tool_calls = tool_calls or []
        if error:
            child.set_error(error)

        latency_ms = (
            int(duration_ms)
            if duration_ms is not None
            else int((end_time - start_time) * 1000)
            if start_time is not None and end_time is not None
            else None
        )
        # Wire dict built directly (not via Tracer._send()) so this bypasses that method's
        # pending-tool-call drain - see Tracer._dispatch's docstring for why a child send must not
        # consume queue entries meant for a sibling or the outer span.
        wire: Dict[str, Any] = {"name": name}
        if input is not None:
            wire["input"] = _safe_serialize(input)
        if output is not None:
            wire["output"] = _safe_serialize(output)
        if latency_ms is not None:
            wire["latency_ms"] = latency_ms
        if child._error is not None:
            wire["error"] = child._error
        if child._framework:
            wire["framework"] = child._framework
        if model:
            wire["model"] = model
        if child.tool_calls:
            wire["tool_calls"] = child.tool_calls
        if metadata:
            wire["metadata"] = _safe_serialize(metadata)
        # What kind of step this is, stated rather than left for the backend to guess from the
        # span's name and which columns happen to be null. Same idea as LangSmith's run_type and
        # Langfuse's observation type; the engine folds other vocabularies onto its own.
        if span_kind:
            wire["span_kind"] = span_kind
        if child._session_id:
            wire["session_id"] = child._session_id
        wire["span_id"] = child._span_id
        if child._parent_span_id:
            wire["parent_span_id"] = child._parent_span_id
        if start_time is not None:
            wire["started_at_unix_nano"] = str(int(start_time * 1_000_000_000))
        if input_tokens:
            wire["input_tokens"] = input_tokens
        if output_tokens:
            wire["output_tokens"] = output_tokens
        if cache_read_tokens:
            wire["cache_read_tokens"] = cache_read_tokens
        if cache_write_tokens:
            wire["cache_write_tokens"] = cache_write_tokens
        child._trace_id = self._tracer._dispatch(wire)
        return child

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
        cache_read_tokens: Optional[int] = None,
        cache_write_tokens: Optional[int] = None,
        emit_steps: bool = True,
    ) -> None:
        """
        Explode a whole auto-instrumented sub-run (e.g. one top-level LangChain
        chain/agent/retriever invocation) into real child-span rows - one per step, tool call,
        and retrieval - parented to this span, instead of it becoming its own independent trace.
        Also adopts a reasonable input/output/model/token summary onto this span itself, so it
        stays useful even though its own detail now lives entirely in the child rows below it.
        tool_calls entries here carry no start/end timing (only latency_ms, see callers e.g.
        langchain.py) - their child span falls back to offset-0 positioning in the tree panel;
        execution_steps (LLM calls) always carry real timing and position correctly.

        ``emit_steps=False`` skips the flat child-span emission but keeps everything else (the
        flat tool_calls mirror for the Monitor tool-failure check, and the summary adoption) -
        for callers that emit their own HIERARCHICAL span tree instead (langchain.py's
        _emit_span_tree parents steps under the graph node that ran them, rather than flat
        under this span).
        """
        with self._merge_lock:
            for step in [] if not emit_steps else (execution_steps or []):
                self._child_span_count += 1
                self.child_span(
                    step.get("name") or f"LLM Call {self._child_span_count}",
                    start_time=step.get("start_time"),
                    end_time=step.get("end_time"),
                    duration_ms=step.get("duration_ms"),
                    input=step.get("input"),
                    output=step.get("output"),
                    model=step.get("model"),
                    input_tokens=step.get("inputTokenSize"),
                    output_tokens=step.get("outputTokenSize"),
                    cache_read_tokens=step.get("cacheReadTokenSize"),
                    cache_write_tokens=step.get("cacheWriteTokenSize"),
                    # Stated, so a step named anything other than "LLM Call N" still classifies -
                    # the backend's name regex was the only thing holding this together.
                    span_kind="llm",
                )
            for tc in tool_calls or []:
                # Some callers' tool_calls dicts (e.g. langchain.py's, which sets these on the
                # same dict for exactly this reason - see its on_tool_end) already carry real
                # start_time/end_time; prefer those over latency_ms alone when both know duration
                # since they also let this child span position correctly in the tree panel
                # instead of defaulting to offset 0.
                if emit_steps:
                    self.child_span(
                        tc.get("name") or "Tool call",
                        start_time=tc.get("start_time"),
                        end_time=tc.get("end_time"),
                        duration_ms=tc.get("latency_ms"),
                        input=tc.get("input"),
                        output=tc.get("output"),
                        error=None if tc.get("success", True) else str(tc.get("output") or "Tool call failed"),
                        span_kind="tool",
                    )
                # Also mirror onto this span's own flat tool_calls list, sent in this span's own
                # wire payload on __exit__ (see tool_calls=self.tool_calls or None below). The
                # child span above is only for the trace detail view's span tree; the engine's
                # built-in "Tool failure" Monitor check reads *this* flat list specifically,
                # looking for a `success: false` entry - a child span row has no such field, so
                # without this a failed tool call recorded via a framework integration (e.g.
                # langchain.py's on_tool_error) would silently never trip that check.
                self.tool_calls.append({
                    "name": tc.get("name") or "Tool call",
                    "input": tc.get("input"),
                    "output": tc.get("output"),
                    "latency_ms": tc.get("latency_ms"),
                    "success": tc.get("success", True),
                })
            for step in [] if not emit_steps else (retrieval_steps or []):
                self._child_span_count += 1
                self.child_span(
                    step.get("name") or f"Retrieval {self._child_span_count}",
                    start_time=step.get("start_time"),
                    end_time=step.get("end_time"),
                    duration_ms=step.get("duration_ms"),
                    input=step.get("query"),
                    output=step.get("output"),
                    metadata={"kind": "retrieval"},
                    span_kind="retrieval",
                )

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
            if cache_read_tokens:
                self._cache_read_tokens += cache_read_tokens
            if cache_write_tokens:
                self._cache_write_tokens += cache_write_tokens

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
            # A fresh span per call (not `self` reused across every call the decorator wraps) -
            # `self` here is only ever the one template object tracer.trace(...) returned at
            # decoration time, so reusing it directly would give every invocation the same
            # span_id/timing. Built via tracer.trace(...) (not the constructor directly) so it
            # picks up real parent/session linkage the normal way if a decorated function happens
            # to be called from inside another active span.
            span = self._tracer.trace(
                self.name, metadata=self._metadata, framework=self._framework, model=self._model,
                session_id=self._session_id,
            )
            span.__enter__()
            try:
                span.input = _capture_fn_input(fn, args, kwargs)
                output = fn(*args, **kwargs)
                span.output = _safe_serialize(output) if output is not None else None
                return output
            except Exception as exc:
                span.set_error(str(exc))
                raise
            finally:
                span.__exit__(None, None, None)

        return wrapper  # type: ignore[return-value]

    def _wrap_async(self, fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            # See _wrap_sync's comment - same "fresh span per call" reasoning applies here.
            span = self._tracer.trace(
                self.name, metadata=self._metadata, framework=self._framework, model=self._model,
                session_id=self._session_id,
            )
            span.__enter__()
            try:
                span.input = _capture_fn_input(fn, args, kwargs)
                output = await fn(*args, **kwargs)
                span.output = _safe_serialize(output) if output is not None else None
                return output
            except Exception as exc:
                span.set_error(str(exc))
                raise
            finally:
                span.__exit__(None, None, None)

        return wrapper  # type: ignore[return-value]


class _RetrievalRecorder:
    """Handle yielded by ``Tracer.trace_retrieval()`` - set ``doc_count``/``output`` inside the block."""

    def __init__(self) -> None:
        self.doc_count: Optional[int] = None
        self.output: Any = None


class _ToolCallRecorder:
    """Handle yielded by ``Tracer.trace_tool_call()`` - set ``output`` inside the block.
    ``success``/``error`` may be set manually; an exception escaping the block sets them
    automatically (success False, error str(exc)) before propagating."""

    def __init__(self) -> None:
        self.output: Any = None
        self.success: Optional[bool] = None
        self.error: Optional[str] = None


class Tracer:
    """
    Production tracer attached to ``AgentX.tracer``.

    Usage - decorator::

        @agentx.tracer.trace("my-agent")
        def run(query: str) -> str:
            ...

    Usage - context manager::

        with agentx.tracer.trace("my-agent", input={"query": q}) as span:
            result = call_agent(q)
            span.output = result

    Usage - async::

        @agentx.tracer.trace("my-agent")
        async def run(query: str) -> str:
            ...
    """

    def __init__(self, ingest_client: IngestClient) -> None:
        self._client = ingest_client
        self._pending_tool_calls: List[Dict[str, Any]] = []
        self._pending_retrievals: List[Dict[str, Any]] = []
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Active-span stack (per thread) - lets auto-instrumented integrations
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
        calling thread - wrap the worker function body in this to attach it::

            with tracer.trace("orchestrator") as span:
                def worker():
                    with tracer.use_span(span):
                        chain.invoke(..., config={"callbacks": [handler]})

                with ThreadPoolExecutor(max_workers=2) as ex:
                    ex.submit(worker).result()

        Safe to call from multiple threads concurrently for the same span -
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
        success: Optional[bool] = None,
        error: Optional[str] = None,
        latency_ms: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """
        Manually record a tool call that an auto-instrumented framework
        integration can't see - e.g. a hand-rolled Anthropic/OpenAI tool-use
        loop where the tool executes in plain Python between two
        ``messages.create()`` calls. Sent as a real child-span row of the active span (see
        ``current_span``) immediately; queued onto the next trace's plain ``tool_calls`` list if
        there's no active span to attach a child to.

        ``success``/``error`` mark a failed call. ``success=False`` is what the engine's built-in
        "Tool failure" Monitor check and the dashboard's Tool quality column both read; leaving
        ``success`` unset means "unknown" and the dashboard falls back to its output-text
        heuristic instead of assuming the call passed.
        """
        active_span = self.current_span
        if active_span is not None:
            active_span.child_span(
                name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=latency_ms,
                input=input,
                output=output,
                error=error,
                span_kind="tool",
            )
            # The child span above is only for the trace detail's span tree - the engine's
            # built-in "Tool failure" check and the dashboard's Tool quality column read the
            # ROOT trace's flat tool_calls list, so a summary lands there too. Same dual-write
            # _merge_child_run already does for framework-integration tool calls, and the reason
            # a failed trace_tool_call() used to be invisible to both surfaces.
            summary: Dict[str, Any] = {
                "name": name,
                "input": _safe_serialize(input) if input is not None else None,
                "output": _safe_serialize(output) if output is not None else None,
                "latency_ms": latency_ms,
            }
            if success is not None:
                summary["success"] = success
            if error is not None:
                summary["error"] = error
            active_span.tool_calls.append(summary)
            return
        pending: Dict[str, Any] = {
            "name": name,
            "input": _safe_serialize(input) if input is not None else None,
            "output": _safe_serialize(output) if output is not None else None,
            "latency_ms": latency_ms,
        }
        if success is not None:
            pending["success"] = success
        if error is not None:
            pending["error"] = error
        self._pending_tool_calls.append(pending)

    @contextmanager
    def trace_tool_call(self, name: str, *, input: Any = None) -> Iterator["_ToolCallRecorder"]:
        """
        Context manager that times a tool call and records it via
        :meth:`record_tool_call` on exit::

            with tracer.trace_tool_call("policy_lookup", input=topic) as t:
                result = policy_lookup(topic)
                t.output = result

        An exception escaping the block records the call as failed (``success=False`` plus the
        error text - what the engine's "Tool failure" check and the dashboard's Tool quality
        column read) and then propagates unchanged.
        """
        start_t = time.time()
        recorder = _ToolCallRecorder()
        try:
            yield recorder
        except BaseException as exc:
            if recorder.success is None:
                recorder.success = False
            if recorder.error is None:
                recorder.error = str(exc)
            raise
        finally:
            end_t = time.time()
            self.record_tool_call(
                name,
                input=input,
                output=recorder.output,
                success=recorder.success,
                error=recorder.error,
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
        auto-instrumented framework integration can't see on its own - e.g. a
        hand-rolled RAG lookup wrapped around a raw Anthropic/OpenAI call or a
        CrewAI kickoff. Sent as a real child-span row of the active span (see ``current_span``);
        with no active span it queues and merges into the very next trace this tracer sends
        (same behavior as ``record_tool_call``, covering the patched-client flow where the
        retrieval runs just before a standalone ``messages.create()`` /
        ``chat.completions.create()`` call).
        """
        active_span = self.current_span
        if active_span is None:
            latency_ms = (
                int(duration_ms)
                if duration_ms is not None
                else int((end_time - start_time) * 1000)
                if start_time is not None and end_time is not None
                else None
            )
            self._pending_retrievals.append({
                "name": name,
                "query": _safe_serialize(query) if query is not None else None,
                "output": _safe_serialize(output) if output is not None else None,
                "duration_ms": latency_ms,
            })
            return
        # The kind marker is what tells the engine (retrieval-context extraction for RAG
        # judges) and the dashboard timeline that this span is a retrieval regardless of its
        # name - the name-based "Retrieval N" heuristic remains only as a fallback for older
        # traces, so custom names like "kb_search" work everywhere.
        active_span.child_span(
            name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            input=query,
            output=output,
            metadata={"kind": "retrieval"},
            span_kind="retrieval",
        )

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
        monitor: Optional[bool] = None,
        pattern_ids: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
    ) -> _TraceSpan:
        """
        Return a :class:`_TraceSpan` that works as both a decorator and a
        context manager.

        Nested calls link as real parent/child span rows (the same model AgentX's OTel ingestion
        path uses), grouped by a shared session_id - a multi-step run shows up as a real tree in
        the trace dialog's span panel. Nesting is automatic: any `with tracer.trace(...)` opened
        while another is already active becomes its child, and so does every auto-instrumented
        call made while a span is active (a patched Anthropic/OpenAI/Google GenAI/LiteLLM client,
        or a framework integration like ``AgentXCallbackHandler``)::

            with client.tracer.trace("support_agent_call") as span:
                reply = call_llm(...)  # becomes its own child-span row
                span.output = reply

        By default the trace is queued and sent on a background thread - fire-and-forget, never
        blocks the caller, but there's no way to learn the resulting trace_id. Pass ``sync=True``
        to send it synchronously instead (blocks until ingested) so ``span.trace_id`` is populated
        once the ``with`` block exits. On a root span, ``sync=True`` covers the WHOLE tree: any
        child spans recorded inside the block (tool calls, LLM calls) are drained before the
        root is sent, so a read immediately after the block sees every span, not just the root.
        Use it e.g. to attach the trace to an evaluation result::

            with client.tracer.trace("support_agent_call", framework="openai", sync=True) as span:
                resp = call_llm(...)
                span.output = resp
            return {"output": resp, "trace_id": span.trace_id}

        Pass ``monitor=True`` to check this trace against Monitor patterns immediately, with no
        dashboard profile required. Pass ``monitor=False`` to explicitly skip EVERY ingest-time
        check (patterns, built-ins, online/custom evaluators, topics) - use this for traces
        created inside an evaluation run, which the run's own evaluator already judges. ``pattern_ids`` (ids from ``client.monitor.patterns.builder(
        ...).publish()``) restricts detection to exactly those patterns; omit it to run the full
        default sweep (built-in checks plus every enabled workspace pattern)::

            pattern = client.monitor.patterns.builder(
                name="Promises a refund", detector_kind="semantic",
                semantic_prompt="The response promises a refund.",
            ).publish()

            with client.tracer.trace("support_agent_call", monitor=True, pattern_ids=[pattern.id]) as span:
                span.output = call_llm(...)

        ``agent_id`` disambiguates when ``name`` alone isn't enough - pass an already-known agent
        id (e.g. one you've seen in the dashboard's Overview tab, or from a direct ``GET /agents``
        call) to pin this trace to that exact agent instead of resolving by name. Omit it (the
        default) and this resolves from ``name`` alone - one stable agent per distinct name,
        created on first use::

            with client.tracer.trace("support-agent", agent_id="ag_123", sync=True) as span:
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
            agent_id=agent_id,
        )

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until all queued traces have been delivered, or until ``timeout`` seconds
        elapse. Returns ``True`` when everything drained, ``False`` on deadline (a warning is
        logged and undelivered traces keep sending in the background)."""
        return self._client.flush(timeout)

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
                    # failFast triggered - run already finalized server-side
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
        as-is - the agent is NOT re-run.

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
        if "input_tokens" in payload:
            wire["input_tokens"] = payload["input_tokens"]
        if "output_tokens" in payload:
            wire["output_tokens"] = payload["output_tokens"]
        if "cache_read_tokens" in payload:
            wire["cache_read_tokens"] = payload["cache_read_tokens"]
        if "cache_write_tokens" in payload:
            wire["cache_write_tokens"] = payload["cache_write_tokens"]
        if "monitor" in payload:
            wire["monitor"] = payload["monitor"]
        if "pattern_ids" in payload:
            wire["pattern_ids"] = payload["pattern_ids"]
        if "span_id" in payload:
            wire["span_id"] = payload["span_id"]
        if "parent_span_id" in payload:
            wire["parent_span_id"] = payload["parent_span_id"]
        if "started_at_unix_nano" in payload:
            wire["started_at_unix_nano"] = payload["started_at_unix_nano"]
        if "agent_id" in payload:
            wire["agent_id"] = payload["agent_id"]

        pending_tool_calls, self._pending_tool_calls = self._pending_tool_calls, []
        if pending_tool_calls:
            # Passed through whole rather than re-projected field by field: record_tool_call may
            # have attached success/error (the fields the engine's tool-failure check reads), and
            # a projection that predates them would silently strip exactly the failure evidence.
            wire["tool_calls"] = list(wire.get("tool_calls") or []) + [dict(t) for t in pending_tool_calls]

        # record_retrieval entries queued with no active span ride the root's
        # performance_summary.retrieval_steps - the same shape older flat traces used, which the
        # engine's retrieval-context extraction and the dashboard's references panel both read.
        pending_retrievals, self._pending_retrievals = self._pending_retrievals, []
        if pending_retrievals:
            summary = dict(wire.get("performance_summary") or {})
            summary["retrieval_steps"] = list(summary.get("retrieval_steps") or []) + [
                dict(r) for r in pending_retrievals
            ]
            wire["performance_summary"] = summary

        return self._dispatch(wire, sync=sync)

    def _dispatch(self, wire: Dict[str, Any], *, sync: bool = False) -> Optional[str]:
        """
        Enqueue (or, if ``sync``, synchronously send) an already wire-shaped payload - the tail
        end of ``_send()``, split out so ``_TraceSpan.child_span()`` can post a child-span row
        directly without going through ``_send()``'s pending-tool-call drain above. That drain
        attaches ``tracer.record_tool_call()`` queue entries (only reached when there's no active
        span to attach a real child span to) to "whatever this tracer sends next" - correct when
        there's exactly one `_send()` per top-level span, but a span with children triggers
        several child sends during its own lifetime, and a pending item recorded for the outer
        span must not get silently attached to an unrelated child span's row instead.
        """
        if sync:
            return self._client.send_trace_sync(wire)
        self._client.enqueue(wire)
        return None
