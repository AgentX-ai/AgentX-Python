"""
Shared helper for tracing a raw SDK client call that may be sync or async.

Anthropic's and Google GenAI's raw clients expose the same method names for
both their sync and async client variants (``client.messages.create``,
``client.models.generate_content``) - the only way to tell them apart is to
call the method and check whether the result is awaitable.
``inspect.iscoroutinefunction`` is unreliable for this: it returns ``False``
even for ``AsyncAnthropic().messages.create``, since these SDKs don't
implement the async variant as a plain top-level ``async def``.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any, Callable, Dict, Optional

from agentx.tracing.tracer import Tracer, _safe_serialize


def call_and_trace(
    original: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    on_finish: Callable[[Optional[Any], Optional[str]], None],
) -> Any:
    """
    Call ``original(*args, **kwargs)``.

    If the result is awaitable (async client), return a coroutine that awaits
    it and calls ``on_finish(response, error)`` only after the real await
    completes, so timing/output/tokens reflect the actual call rather than
    the moment the coroutine object was constructed. If the result is a
    normal value (sync client), call ``on_finish`` immediately.

    Either way, the original call's own return value / exception behavior is
    unchanged for the caller - this only affects when/how the trace is built.
    """
    try:
        result = original(*args, **kwargs)
    except Exception as exc:
        on_finish(None, str(exc))
        raise

    if asyncio.iscoroutine(result) or inspect.isawaitable(result):
        return _await_and_finish(result, on_finish)

    on_finish(result, None)
    return result


async def _await_and_finish(
    awaitable: Any,
    on_finish: Callable[[Optional[Any], Optional[str]], None],
) -> Any:
    try:
        response = await awaitable
    except Exception as exc:
        on_finish(None, str(exc))
        raise
    on_finish(response, None)
    return response


# The engine's unregistered-tool surfacing (Tools & MCPs -> Unregistered) reads a trace's
# metadata "tools" key to show the REAL definition instead of one inferred from observed
# arguments (see AgentX-trace-eval's toolSchemas.ts draftFromMetadata). Raw-client patches see
# the request's tools=[...] right in kwargs, so capture it - capped so a huge toolbox never
# blows up the trace's metadata budget.
_MAX_TOOL_DEFINITIONS = 20
_MAX_TOOL_DEFINITIONS_BYTES = 12_000


def capture_tool_definitions(tools: Any) -> Optional[list]:
    """Return a metadata-ready copy of a request's ``tools=[...]`` list, or None."""
    if not isinstance(tools, list) or not tools:
        return None
    # A plain JSON round-trip preserves nested schema objects exactly (default=str catches the
    # odd SDK object inside); _safe_serialize would repr-stringify nested dicts, turning a
    # parameters schema into an unusable string.
    try:
        text = json.dumps(tools[:_MAX_TOOL_DEFINITIONS], default=str)
        if len(text) > _MAX_TOOL_DEFINITIONS_BYTES:
            return None
        serialized = json.loads(text)
    except (TypeError, ValueError):
        return None
    return serialized if isinstance(serialized, list) else None


def finish_llm_call(
    tracer: Tracer,
    *,
    name: str,
    framework: str,
    metadata: Optional[Dict[str, Any]],
    session_id: Optional[str],
    start_t: float,
    end_t: float,
    input_repr: Any,
    output: Optional[str],
    model: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    error: Optional[str],
    cache_read_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    tool_definitions: Optional[list] = None,
    call_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Close out one raw-client LLM call - shared by the ``on_finish``/exit
    callbacks of every integration that patches a raw provider client
    (``anthropic.py``, ``google_genai.py``, ``openai.py``, ``litellm.py``) rather than a
    framework-level callback/plugin system.

    If the call happened inside a ``with tracer.trace(...)`` block, it becomes that span's own
    real child span (via _record_llm_call) instead of an independent trace - the same "part of a
    multi-call agentic loop" behavior ``anthropic.py`` already had; folded in here so every
    raw-client integration gets it instead of each having to remember to check
    ``tracer.current_span`` itself. Otherwise it becomes its own real root span, opened/closed
    directly here (not via ``tracer._send()``) so it still gets a real span_id/session_id and the
    call's exact timing rather than wall-clock "now".
    """
    latency_ms = int((end_t - start_t) * 1000)

    if tool_definitions:
        metadata = {**(metadata or {}), "tools": tool_definitions}

    active_span = tracer.current_span
    if active_span is not None:
        # The definitions describe the whole call's toolbox - attach them to the enclosing
        # span's metadata (first capture wins) so the ROOT trace carries them for the
        # unregistered-tool listing, same as the standalone-trace path below.
        if tool_definitions and not (active_span._metadata or {}).get("tools"):
            active_span._metadata = {**(active_span._metadata or {}), "tools": tool_definitions}
        if error is not None:
            active_span.set_error(error)
        active_span._record_llm_call(
            duration_ms=latency_ms,
            start_time=start_t,
            end_time=end_t,
            input=input_repr,
            output=output,
            model=model,
            # Stamp the provider literal on a span opened without one (adoption keeps an
            # explicit framework= or a framework integration's label winning over this).
            framework=framework,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            metadata=call_metadata,
        )
        return

    # A patched provider call outside any active span becomes its own root trace - it is a bare
    # model call, so stamp it "llm" rather than leaving the kind unset.
    span = tracer.trace(
        name,
        metadata={**(metadata or {}), **(call_metadata or {})} if (metadata or call_metadata) else None,
        framework=framework,
        model=model,
        session_id=session_id,
        span_kind="llm",
    )
    span.__enter__()
    span._start = start_t
    span.input = input_repr
    span.output = output
    if error:
        span.set_error(error)
    if input_tokens:
        span._input_tokens = input_tokens
    if output_tokens:
        span._output_tokens = output_tokens
    if cache_read_tokens:
        span._cache_read_tokens = cache_read_tokens
    if cache_write_tokens:
        span._cache_write_tokens = cache_write_tokens
    span.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Streaming: wrap a provider's chunk stream so the trace is built from what
# was actually streamed, without touching the caller's consumption of it.
# ---------------------------------------------------------------------------

class StreamAccumulator:
    """
    What a streaming patch feeds each chunk into. Subclasses collect the
    provider-specific pieces (text deltas, tool-call deltas, the usage block
    that only arrives on the final chunk) and hand back the finished picture
    in ``result()``.
    """

    def feed(self, chunk: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def result(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class TracedStream:
    """
    Transparent proxy over a provider ``Stream``/``AsyncStream``: iterates the
    real object, feeds every chunk to the accumulator, and calls ``on_finish``
    exactly once when the stream is exhausted, raises, is closed (``close()``,
    ``with``/``async with`` exit), or is dropped part-way and garbage
    collected - so an abandoned stream still records what it streamed.

    Latency is measured to the LAST chunk (the response as the caller saw it),
    and the time to the FIRST chunk is reported separately as
    ``time_to_first_token_ms`` - the two numbers a streaming call is judged by.

    Attribute access falls through to the wrapped stream (``.response``,
    provider helpers), and ``__iter__``/``__aiter__`` return ``self`` so early
    ``break`` leaves no half-driven generator behind.
    """

    def __init__(
        self,
        stream: Any,
        accumulator: StreamAccumulator,
        on_finish: Callable[[Dict[str, Any], Optional[str]], None],
    ) -> None:
        self._stream = stream
        self._accumulator = accumulator
        self._on_finish = on_finish
        self._done = False
        self._first_chunk_t: Optional[float] = None
        self._last_chunk_t: Optional[float] = None
        self._sync_iter: Any = None
        self._async_iter: Any = None

    # -- bookkeeping ---------------------------------------------------------

    def _observe(self, chunk: Any) -> None:
        now = time.time()
        if self._first_chunk_t is None:
            self._first_chunk_t = now
        self._last_chunk_t = now
        try:
            self._accumulator.feed(chunk)
        except Exception:
            # A malformed chunk must never break the caller's stream; it just
            # goes uncounted in the trace.
            pass

    def _finish(self, error: Optional[str]) -> None:
        if self._done:
            return
        self._done = True
        try:
            result = self._accumulator.result()
        except Exception:
            result = {}
        start_t = result.pop("_start_t", None)
        result["time_to_first_token_ms"] = (
            int((self._first_chunk_t - start_t) * 1000) if self._first_chunk_t is not None and start_t is not None else None
        )
        # The response "ended" at its last chunk, not at whatever later moment the caller closed
        # or dropped the stream - that is the latency the user experienced.
        result["end_t"] = self._last_chunk_t if self._last_chunk_t is not None else time.time()
        self._on_finish(result, error)

    @property
    def first_chunk_at(self) -> Optional[float]:
        return self._first_chunk_t

    # -- sync iteration ------------------------------------------------------

    def __iter__(self) -> "TracedStream":
        return self

    def __next__(self) -> Any:
        if self._sync_iter is None:
            self._sync_iter = iter(self._stream)
        try:
            chunk = next(self._sync_iter)
        except StopIteration:
            self._finish(None)
            raise
        except BaseException as exc:
            self._finish(str(exc))
            raise
        self._observe(chunk)
        return chunk

    # -- async iteration -----------------------------------------------------

    def __aiter__(self) -> "TracedStream":
        return self

    async def __anext__(self) -> Any:
        if self._async_iter is None:
            self._async_iter = self._stream.__aiter__()
        try:
            chunk = await self._async_iter.__anext__()
        except StopAsyncIteration:
            self._finish(None)
            raise
        except BaseException as exc:
            self._finish(str(exc))
            raise
        self._observe(chunk)
        return chunk

    # -- context managers / close --------------------------------------------

    def __enter__(self) -> "TracedStream":
        enter = getattr(self._stream, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type, exc_val, tb) -> Any:
        exit_ = getattr(self._stream, "__exit__", None)
        result = exit_(exc_type, exc_val, tb) if exit_ is not None else None
        self._finish(str(exc_val) if exc_val else None)
        return result

    async def __aenter__(self) -> "TracedStream":
        enter = getattr(self._stream, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, exc_type, exc_val, tb) -> Any:
        exit_ = getattr(self._stream, "__aexit__", None)
        result = await exit_(exc_type, exc_val, tb) if exit_ is not None else None
        self._finish(str(exc_val) if exc_val else None)
        return result

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        try:
            if close is not None:
                close()
        finally:
            self._finish(None)

    async def aclose(self) -> None:
        # openai's AsyncStream spells its close as `async def close()`; httpx-style streams
        # spell it `aclose()`. Await whichever one answers with an awaitable.
        close = getattr(self._stream, "aclose", None) or getattr(self._stream, "close", None)
        try:
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
        finally:
            self._finish(None)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._stream, item)

    def __del__(self) -> None:
        # Best effort only: a stream the caller stopped reading and dropped still records the
        # chunks it did see. Never raises - a destructor exception is unactionable noise.
        try:
            self._finish(None)
        except Exception:
            pass


def trace_stream(
    result: Any,
    accumulator: StreamAccumulator,
    on_finish: Callable[[Dict[str, Any], Optional[str]], None],
) -> Any:
    """
    Wrap the value a patched ``create(..., stream=True)`` returned. A sync
    client hands back the stream object directly; an async client hands back
    a coroutine that resolves to it, so the wrapping is deferred until the
    real stream exists - the caller's ``await`` is unchanged either way.
    """
    if asyncio.iscoroutine(result) or inspect.isawaitable(result):
        return _await_and_wrap(result, accumulator, on_finish)
    return TracedStream(result, accumulator, on_finish)


async def _await_and_wrap(
    awaitable: Any,
    accumulator: StreamAccumulator,
    on_finish: Callable[[Dict[str, Any], Optional[str]], None],
) -> Any:
    try:
        stream = await awaitable
    except Exception as exc:
        on_finish({}, str(exc))
        raise
    return TracedStream(stream, accumulator, on_finish)
