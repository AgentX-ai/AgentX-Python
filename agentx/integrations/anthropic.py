"""
Anthropic SDK integration for AgentX production tracing.

Usage::

    from agentx.integrations.anthropic import patch_anthropic_client
    import anthropic

    client = anthropic.Anthropic()
    patch_anthropic_client(client, agentx.tracer, name="claude-agent")

    # All subsequent client.messages.create() calls are now traced automatically.

Works with both ``anthropic.Anthropic`` and ``anthropic.AsyncAnthropic`` clients.

Requires: ``pip install "agentx-python[anthropic]"``
"""
from __future__ import annotations

import inspect
import time
from typing import Any, Dict, Optional, Tuple

from agentx.tracing.tracer import Tracer, _safe_serialize
from agentx.integrations._traced_call import capture_tool_definitions, call_and_trace, finish_llm_call


def _extract_output_text(response: Any) -> Optional[str]:
    """
    Extract the assistant's text reply from a Messages API response, falling
    back to a description of any tool_use blocks when the response is a pure
    tool call with no accompanying text.
    """
    content = getattr(response, "content", None) if response is not None else None
    if not content:
        return None
    texts = []
    tool_calls = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
        elif block_type == "tool_use":
            name = getattr(block, "name", "unknown")
            tool_input = getattr(block, "input", None)
            tool_calls.append(f"{name}({tool_input})")
    if texts:
        return "\n".join(texts)
    if tool_calls:
        return "[tool call] " + ", ".join(tool_calls)
    return None


def _prepend_system(messages: Any, system: Any) -> Any:
    """
    Fold the ``system`` kwarg (a separate top-level parameter in the Anthropic
    SDK, not part of ``messages``) into the traced input as a leading
    system-role entry - the same shape trace consumers already expect from
    other frameworks' captured input.
    """
    if not system:
        return messages
    system_entry = {"role": "system", "content": system}
    if isinstance(messages, list):
        return [system_entry] + list(messages)
    if messages is None:
        return [system_entry]
    return [system_entry, messages]


def _extract_usage_tokens(
    usage: Any,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Pull input/output/cache token counts off a ``response.usage`` object.
    ``input_tokens`` stays the *total* (base + cache_creation + cache_read) -
    still real input tokens for cost/context-window purposes - while
    ``cache_read``/``cache_write`` are reported alongside as the subset of
    that total the provider actually billed at a different (cache) rate, so
    the backend can price them separately instead of at the full input rate.
    """
    if usage is None:
        return None, None, None, None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cache_creation = getattr(usage, "cache_creation_input_tokens", None)
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    if cache_creation or cache_read:
        input_tokens = (input_tokens or 0) + (cache_creation or 0) + (cache_read or 0)
    return input_tokens, output_tokens, cache_read, cache_creation


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
    unchanged so nothing in the caller needs to change. Works with both sync
    (``Anthropic``) and async (``AsyncAnthropic``) clients.
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
        input_messages = _prepend_system(
            kwargs.get("messages") or (args[0] if args else None),
            kwargs.get("system"),
        )
        model = kwargs.get("model")
        tool_definitions = capture_tool_definitions(kwargs.get("tools"))

        input_repr = _safe_serialize(input_messages)

        def on_finish(response: Optional[Any], error: Optional[str]) -> None:
            end_t = time.time()
            output = None
            input_tokens = None
            output_tokens = None
            cache_read_tokens = None
            cache_write_tokens = None
            if response is not None:
                output = _extract_output_text(response)
                try:
                    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens = _extract_usage_tokens(
                        getattr(response, "usage", None)
                    )
                except Exception:
                    pass

            finish_llm_call(
                tracer,
                name=name,
                framework="anthropic",
                metadata=metadata,
                session_id=session_id,
                start_t=start_t,
                end_t=end_t,
                input_repr=input_repr,
                output=output,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                error=error,
                tool_definitions=tool_definitions,
            )

        return call_and_trace(original, args, kwargs, on_finish)

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
        # `.stream()` itself returns a context-manager object synchronously
        # for both `Anthropic` and `AsyncAnthropic` - the async/sync split
        # only shows up in whether `with`/`async with` and
        # `get_final_message()` are used, handled inside `_TracedStream`.
        start_t = time.time()
        ctx = original_stream(*args, **kwargs)
        input_repr = _safe_serialize(_prepend_system(kwargs.get("messages"), kwargs.get("system")))
        model = kwargs.get("model")
        tool_definitions = capture_tool_definitions(kwargs.get("tools"))

        def build_and_send(end_t: float, error: Optional[str], final_message: Optional[Any]) -> None:
            output = None
            input_tokens = None
            output_tokens = None
            cache_read_tokens = None
            cache_write_tokens = None
            if final_message is not None:
                output = _extract_output_text(final_message)
                try:
                    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens = _extract_usage_tokens(
                        getattr(final_message, "usage", None)
                    )
                except Exception:
                    pass
            finish_llm_call(
                tracer,
                name=name,
                framework="anthropic",
                metadata=metadata,
                session_id=session_id,
                start_t=start_t,
                end_t=end_t,
                input_repr=input_repr,
                output=output,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                error=error,
                tool_definitions=tool_definitions,
            )

        class _TracedStream:
            """Thin wrapper that records timing when the stream context exits."""

            def __enter__(self_inner):
                return ctx.__enter__()

            def __exit__(self_inner, exc_type, exc_val, tb):
                result = ctx.__exit__(exc_type, exc_val, tb)
                end_t = time.time()
                error = str(exc_val) if exc_val else None
                final_message = None
                try:
                    final_message = ctx.get_final_message()
                except Exception:
                    pass
                build_and_send(end_t, error, final_message)
                return result

            async def __aenter__(self_inner):
                return await ctx.__aenter__()

            async def __aexit__(self_inner, exc_type, exc_val, tb):
                result = await ctx.__aexit__(exc_type, exc_val, tb)
                end_t = time.time()
                error = str(exc_val) if exc_val else None
                final_message = None
                try:
                    raw = ctx.get_final_message()
                    final_message = await raw if inspect.isawaitable(raw) else raw
                except Exception:
                    pass
                build_and_send(end_t, error, final_message)
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
