"""
Anthropic SDK integration for AgentX production tracing.

Usage::

    from agentx.integrations.anthropic import patch_anthropic_client
    import anthropic

    client = anthropic.Anthropic()
    patch_anthropic_client(client, agentx.tracer, name="claude-agent")

    # All subsequent client.messages.create() calls are now traced automatically.

Requires: ``pip install agentx[anthropic]``
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from agentx.tracing.tracer import Tracer, _safe_serialize
from agentx.integrations._perf import build_performance_summary


def patch_anthropic_client(
    client: Any,
    tracer: Tracer,
    name: str = "anthropic-agent",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> None:
    """
    Monkey-patch ``client.messages.create`` and ``client.messages.stream``
    (if present) to automatically send a trace for every call.

    The original method is still called and its return value is passed through
    unchanged so nothing in the caller needs to change.
    """
    messages = getattr(client, "messages", None)
    if messages is None:
        raise ValueError("Provided client does not have a .messages attribute")

    _patch_create(messages, tracer, name, metadata, session_id)

    # stream is optional (not present in all versions)
    if hasattr(messages, "stream"):
        _patch_stream(messages, tracer, name, metadata, session_id)


def _patch_create(
    messages_resource: Any,
    tracer: Tracer,
    name: str,
    metadata: Optional[Dict[str, Any]],
    session_id: Optional[str],
) -> None:
    original = messages_resource.create
    if getattr(original, "_agentx_patched", False):
        return  # already patched

    def patched_create(*args, **kwargs):
        start_t = time.time()
        error: Optional[str] = None
        response = None
        try:
            response = original(*args, **kwargs)
            return response
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            end_t = time.time()
            latency_ms = int((end_t - start_t) * 1000)
            input_messages = kwargs.get("messages") or (args[0] if args else None)
            model = kwargs.get("model")
            output = None
            input_tokens = None
            output_tokens = None
            if response is not None:
                try:
                    output = response.content[0].text if response.content else None
                except Exception:
                    output = str(response)
                try:
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        input_tokens = getattr(usage, "input_tokens", None)
                        output_tokens = getattr(usage, "output_tokens", None)
                except Exception:
                    pass

            active_span = tracer.current_span
            if active_span is not None:
                # Part of a `with tracer.trace(...)` block (e.g. a multi-call
                # agentic loop) — attach as one LLM-call step on that span's
                # trace instead of sending an independent trace per call.
                if error is not None:
                    active_span.set_error(error)
                active_span._record_llm_call(
                    duration_ms=latency_ms,
                    start_time=start_t,
                    end_time=end_t,
                    input=_safe_serialize(input_messages),
                    output=output,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            else:
                perf = build_performance_summary(
                    total_duration_ms=latency_ms,
                    execution_steps=[{
                        "name": "LLM Call 1",
                        "duration_ms": latency_ms,
                        "start_time": start_t,
                        "end_time": end_t,
                        "model": model,
                        "input": _safe_serialize(input_messages),
                        "output": output,
                        "inputTokenSize": input_tokens,
                        "outputTokenSize": output_tokens,
                    }],
                    has_errors=error is not None,
                )
                tracer._send(
                    name=name,
                    input=_safe_serialize(input_messages),
                    output=output,
                    latency_ms=latency_ms,
                    error=error,
                    framework="anthropic",
                    model=model,
                    metadata=metadata,
                    session_id=session_id,
                    performance_summary=perf,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

    patched_create._agentx_patched = True
    messages_resource.create = patched_create


def _patch_stream(
    messages_resource: Any,
    tracer: Tracer,
    name: str,
    metadata: Optional[Dict[str, Any]],
    session_id: Optional[str],
) -> None:
    original_stream = messages_resource.stream
    if getattr(original_stream, "_agentx_patched", False):
        return

    def patched_stream(*args, **kwargs):
        start_t = time.time()
        ctx = original_stream(*args, **kwargs)

        class _TracedStream:
            """Thin wrapper that records timing when the stream context exits."""

            def __enter__(self_inner):
                return ctx.__enter__()

            def __exit__(self_inner, exc_type, exc_val, tb):
                result = ctx.__exit__(exc_type, exc_val, tb)
                end_t = time.time()
                latency_ms = int((end_t - start_t) * 1000)
                error = str(exc_val) if exc_val else None
                output = None
                input_tokens = None
                output_tokens = None
                try:
                    final = ctx.get_final_message()
                    output = final.content[0].text if final.content else None
                    usage = getattr(final, "usage", None)
                    if usage is not None:
                        input_tokens = getattr(usage, "input_tokens", None)
                        output_tokens = getattr(usage, "output_tokens", None)
                except Exception:
                    pass
                perf = build_performance_summary(
                    total_duration_ms=latency_ms,
                    execution_steps=[{
                        "name": "LLM Call 1",
                        "duration_ms": latency_ms,
                        "start_time": start_t,
                        "end_time": end_t,
                        "model": kwargs.get("model"),
                        "input": _safe_serialize(kwargs.get("messages")),
                        "output": output,
                        "inputTokenSize": input_tokens,
                        "outputTokenSize": output_tokens,
                    }],
                    has_errors=error is not None,
                )
                tracer._send(
                    name=name,
                    input=_safe_serialize(kwargs.get("messages")),
                    output=output,
                    latency_ms=latency_ms,
                    error=error,
                    framework="anthropic",
                    model=kwargs.get("model"),
                    metadata=metadata,
                    session_id=session_id,
                    performance_summary=perf,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                return result

            def __iter__(self_inner):
                return iter(ctx)

            def __aiter__(self_inner):
                return aiter(ctx)

            def __getattr__(self_inner, item):
                return getattr(ctx, item)

        return _TracedStream()

    patched_stream._agentx_patched = True
    messages_resource.stream = patched_stream
