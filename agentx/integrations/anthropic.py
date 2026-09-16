"""
Anthropic SDK integration for AgentX production tracing.

Usage::

    from agentx.integrations.anthropic import patch_anthropic_client
    import anthropic

    client = anthropic.Anthropic()
    patch_anthropic_client(client, agentx.tracer, name="claude-agent")

    # All subsequent client.messages.create() calls are now traced automatically.

Works with both ``anthropic.Anthropic`` and ``anthropic.AsyncAnthropic`` clients.

Both streaming shapes are traced: the ``client.messages.stream(...)`` helper
(a context manager with ``get_final_message()``) and the raw
``messages.create(..., stream=True)`` event stream, which is wrapped in a
transparent proxy that assembles the reply, tool-use blocks, and token usage
from the events as the caller consumes them.

Requires: ``pip install "agentx-python[anthropic]"``
"""
from __future__ import annotations

import inspect
import time
from typing import Any, Dict, Optional, Tuple

from agentx.tracing.tracer import Tracer, _safe_serialize
from agentx.integrations._traced_call import (
    StreamAccumulator,
    capture_tool_definitions,
    call_and_trace,
    finish_llm_call,
    trace_stream,
)


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


class _MessageEventStreamAccumulator(StreamAccumulator):
    """
    Rebuild a ``Message`` from the raw ``create(stream=True)`` event sequence:
    ``message_start`` carries the input-side usage, ``content_block_start`` opens
    a text or tool_use block, ``content_block_delta`` appends ``text_delta`` /
    ``input_json_delta`` fragments to it, ``message_delta`` carries the
    output-token count. Token accounting mirrors ``_extract_usage_tokens``.
    """

    def __init__(self, start_t: float) -> None:
        self._start_t = start_t
        self._blocks: Dict[int, Dict[str, Any]] = {}
        self._input_tokens: Optional[int] = None
        self._output_tokens: Optional[int] = None
        self._cache_read: Optional[int] = None
        self._cache_write: Optional[int] = None
        self._model: Optional[str] = None

    def feed(self, event: Any) -> None:
        event_type = getattr(event, "type", None)
        if event_type == "message_start":
            message = getattr(event, "message", None)
            self._model = getattr(message, "model", None) or self._model
            input_tokens, output_tokens, cache_read, cache_write = _extract_usage_tokens(getattr(message, "usage", None))
            self._input_tokens = input_tokens
            self._cache_read = cache_read
            self._cache_write = cache_write
            if output_tokens:
                self._output_tokens = output_tokens
        elif event_type == "content_block_start":
            index = getattr(event, "index", 0) or 0
            block = getattr(event, "content_block", None)
            self._blocks[index] = {
                "type": getattr(block, "type", None),
                "name": getattr(block, "name", None),
                "text": [getattr(block, "text", None) or ""] if getattr(block, "type", None) == "text" else [],
                "json": [],
            }
        elif event_type == "content_block_delta":
            index = getattr(event, "index", 0) or 0
            delta = getattr(event, "delta", None)
            entry = self._blocks.setdefault(index, {"type": None, "name": None, "text": [], "json": []})
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
                entry["type"] = entry["type"] or "text"
                entry["text"].append(getattr(delta, "text", None) or "")
            elif delta_type == "input_json_delta":
                entry["type"] = entry["type"] or "tool_use"
                entry["json"].append(getattr(delta, "partial_json", None) or "")
        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
            if output_tokens is not None:
                self._output_tokens = output_tokens

    def result(self) -> Dict[str, Any]:
        texts = []
        tool_calls = []
        for _, block in sorted(self._blocks.items()):
            if block["type"] == "text":
                text = "".join(block["text"])
                if text:
                    texts.append(text)
            elif block["type"] == "tool_use":
                tool_calls.append(f"{block['name'] or 'unknown'}({''.join(block['json'])})")
        output: Optional[str] = "\n".join(texts) if texts else None
        if output is None and tool_calls:
            output = "[tool call] " + ", ".join(tool_calls)
        return {
            "_start_t": self._start_t,
            "output": output,
            "model": self._model,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cache_read_tokens": self._cache_read,
            "cache_write_tokens": self._cache_write,
        }


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

        if kwargs.get("stream"):
            # Parent fixed at call time - see openai.py's patched_create for why.
            parent = tracer.current_span

            def on_stream_finish(collected: Dict[str, Any], error: Optional[str]) -> None:
                ttft = collected.get("time_to_first_token_ms")
                call_metadata: Dict[str, Any] = {"streaming": True}
                if ttft is not None:
                    call_metadata["timeToFirstTokenMs"] = ttft
                finish_llm_call(
                    tracer,
                    name=name,
                    framework="anthropic",
                    metadata=metadata,
                    call_metadata=call_metadata,
                    active_span=parent,
                    session_id=session_id,
                    start_t=start_t,
                    end_t=collected.get("end_t") or time.time(),
                    input_repr=input_repr,
                    output=collected.get("output"),
                    model=collected.get("model") or model,
                    input_tokens=collected.get("input_tokens"),
                    output_tokens=collected.get("output_tokens"),
                    cache_read_tokens=collected.get("cache_read_tokens"),
                    cache_write_tokens=collected.get("cache_write_tokens"),
                    error=error,
                    tool_definitions=tool_definitions,
                )

            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                on_stream_finish({}, str(exc))
                raise
            return trace_stream(result, _MessageEventStreamAccumulator(start_t), on_stream_finish)

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
        parent = tracer.current_span
        ctx = original_stream(*args, **kwargs)
        input_repr = _safe_serialize(_prepend_system(kwargs.get("messages") or (args[0] if args else None), kwargs.get("system")))
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
            """
            Thin wrapper that records the final message when the stream context exits.
            ``ctx`` is the SDK's stream *manager*; the ``MessageStream`` it yields on enter is
            what carries ``get_final_message()``, and it must be read BEFORE the manager's exit
            closes it - reading it off the manager after close silently yielded no output.
            """

            _inner: Any = None
            _sent: bool = False

            # What streamed so far, WITHOUT draining the rest of the response: the SDK's
            # get_final_message() calls until_done(), which would turn an early `break` into a
            # blocking read of every remaining token. The snapshot is the final message once the
            # stream was consumed, and honestly partial when the caller stopped early.
            def _snapshot(self_inner):
                inner = self_inner._inner
                if inner is None:
                    return None
                try:
                    return getattr(inner, "current_message_snapshot", None)
                except Exception:
                    return None

            def _send_once(self_inner, end_t: float, error: Optional[str], snapshot: Any) -> None:
                if self_inner._sent:
                    return
                self_inner._sent = True
                try:
                    build_and_send(end_t, error, snapshot)
                except Exception:
                    pass  # tracing never raises into the caller

            def __enter__(self_inner):
                self_inner._inner = ctx.__enter__()
                return self_inner._inner

            def __exit__(self_inner, exc_type, exc_val, tb):
                end_t = time.time()
                error = str(exc_val) if exc_val else None
                # Snapshot BEFORE the manager closes the stream (the earlier bug read it after).
                snapshot = self_inner._snapshot()
                try:
                    return ctx.__exit__(exc_type, exc_val, tb)
                finally:
                    self_inner._send_once(end_t, error, snapshot)

            async def __aenter__(self_inner):
                self_inner._inner = await ctx.__aenter__()
                return self_inner._inner

            async def __aexit__(self_inner, exc_type, exc_val, tb):
                end_t = time.time()
                error = str(exc_val) if exc_val else None
                snapshot = self_inner._snapshot()
                try:
                    return await ctx.__aexit__(exc_type, exc_val, tb)
                finally:
                    self_inner._send_once(end_t, error, snapshot)

            def __del__(self_inner):
                # A helper stream that was entered but never exited still records what it saw.
                try:
                    if self_inner._inner is not None:
                        self_inner._send_once(time.time(), None, self_inner._snapshot())
                except Exception:
                    pass

            def __iter__(self_inner):
                return iter(ctx)

            def __aiter__(self_inner):
                return ctx.__aiter__()  # aiter() builtin is 3.10+; python_requires is >=3.9

            def __getattr__(self_inner, item):
                return getattr(ctx, item)

        return _TracedStream()

    patched_stream._agentx_patched = True
    messages_resource.stream = patched_stream
