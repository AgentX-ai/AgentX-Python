"""
Shared helper for tracing a raw SDK client call that may be sync or async.

Anthropic's and Google GenAI's raw clients expose the same method names for
both their sync and async client variants (``client.messages.create``,
``client.models.generate_content``) — the only way to tell them apart is to
call the method and check whether the result is awaitable.
``inspect.iscoroutinefunction`` is unreliable for this: it returns ``False``
even for ``AsyncAnthropic().messages.create``, since these SDKs don't
implement the async variant as a plain top-level ``async def``.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, Optional

from agentx.tracing.tracer import Tracer
from agentx.integrations._perf import build_performance_summary


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
    unchanged for the caller — this only affects when/how the trace is built.
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
) -> None:
    """
    Close out one raw-client LLM call — shared by the ``on_finish``/exit
    callbacks of every integration that patches a raw provider client
    (``anthropic.py``, ``google_genai.py``, ``openai.py``) rather than a
    framework-level callback/plugin system.

    If the call happened inside a ``with tracer.trace(...)`` block, attach it
    as one LLM-call step on that span instead of sending an independent
    trace — the same "part of a multi-call agentic loop" behavior
    ``anthropic.py`` already had; folded in here so every raw-client
    integration gets it instead of each having to remember to check
    ``tracer.current_span`` itself.
    """
    latency_ms = int((end_t - start_t) * 1000)

    active_span = tracer.current_span
    if active_span is not None:
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
        )
        return

    perf = build_performance_summary(
        total_duration_ms=latency_ms,
        execution_steps=[{
            "name": "LLM Call 1",
            "duration_ms": latency_ms,
            "start_time": start_t,
            "end_time": end_t,
            "model": model,
            "input": input_repr,
            "output": output,
            "inputTokenSize": input_tokens,
            "outputTokenSize": output_tokens,
        }],
        has_errors=error is not None,
    )
    tracer._send(
        name=name,
        input=input_repr,
        output=output,
        latency_ms=latency_ms,
        error=error,
        framework=framework,
        model=model,
        metadata=metadata,
        session_id=session_id,
        performance_summary=perf,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
