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
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        return

    span = tracer.trace(name, metadata=metadata, framework=framework, model=model, session_id=session_id)
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
