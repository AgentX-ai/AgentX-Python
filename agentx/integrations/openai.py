"""
Raw OpenAI SDK integration for AgentX production tracing.

This patches the plain ``openai`` Python client directly - for tracing
agents built on the higher-level OpenAI Agents SDK instead, see
``agentx.integrations.openai_agents``.

Usage::

    from agentx.integrations.openai import patch_openai_client
    import openai

    client = openai.OpenAI()
    patch_openai_client(client, agentx.tracer, name="my-agent")

    # All subsequent client.chat.completions.create() calls are now traced.

Works with both ``openai.OpenAI`` and ``openai.AsyncOpenAI`` clients.

Streaming calls (``stream=True``) are traced too: the returned stream is
wrapped in a transparent proxy that assembles the reply from the chunks as the
caller consumes them, so the trace carries the full text, tool calls, and
(with ``stream_options={"include_usage": True}``) token usage, plus the time
to first token.

Requires: ``pip install "agentx-python[openai]"``
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import logging

from agentx.tracing.tracer import Tracer, _safe_serialize

logger = logging.getLogger(__name__)
# Warn once per process, not per call: a streamed OpenAI call carries no usage unless the caller
# asked for it, and a silent zero would under-report every streaming app's spend.
_warned_stream_usage = False
from agentx.integrations._traced_call import (
    StreamAccumulator,
    capture_tool_definitions,
    call_and_trace,
    finish_llm_call,
    trace_stream,
)


def _extract_output_text(response: Any) -> Optional[str]:
    """
    Extract the assistant's text reply from a ``ChatCompletion``, falling
    back to a description of any tool calls when the response is a pure
    tool call with no accompanying text.
    """
    choices = getattr(response, "choices", None) or []
    texts = []
    tool_call_descriptions = []
    for choice in choices:
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content:
            texts.append(content)
        tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        for tc in tool_calls or []:
            fn = getattr(tc, "function", None)
            fn_name = getattr(fn, "name", "unknown") if fn is not None else "unknown"
            fn_args = getattr(fn, "arguments", None) if fn is not None else None
            tool_call_descriptions.append(f"{fn_name}({fn_args})")
    if texts:
        return "\n".join(texts)
    if tool_call_descriptions:
        return "[tool call] " + ", ".join(tool_call_descriptions)
    return None


def _extract_usage_tokens(usage: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Pull input/output/cached token counts off a ``response.usage`` object.
    ``prompt_tokens`` already includes cached tokens (``prompt_tokens_details
    .cached_tokens`` is a discount breakdown, not an addition), so - unlike
    Anthropic's cache accounting - no extra folding is needed for the input
    total; ``cached_tokens`` is reported alongside it so the backend can
    price that subset at its own (cheaper) cache rate instead of the full
    input rate. OpenAI has no cache-*write* concept to report.
    """
    if usage is None:
        return None, None, None
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", None) if details is not None else None
    return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None), cached_tokens


class _ChatCompletionStreamAccumulator(StreamAccumulator):
    """
    Rebuild a ``ChatCompletion``-shaped result from ``ChatCompletionChunk``s:
    text deltas concatenate per choice, tool-call deltas merge by index (name
    arrives once, arguments arrive as fragments), and the ``usage`` block -
    present only on the final chunk, and only when the caller asked for it
    with ``stream_options={"include_usage": True}`` - is kept when it appears.
    """

    def __init__(self, start_t: float) -> None:
        self._start_t = start_t
        self._texts: Dict[int, list] = {}
        self._tool_calls: Dict[int, Dict[str, Any]] = {}
        self._usage: Any = None
        self._model: Optional[str] = None

    def feed(self, chunk: Any) -> None:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._usage = usage
        model = getattr(chunk, "model", None)
        if model and not self._model:
            self._model = model
        for choice in getattr(chunk, "choices", None) or []:
            index = getattr(choice, "index", 0) or 0
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                self._texts.setdefault(index, []).append(content)
            for tc in getattr(delta, "tool_calls", None) or []:
                key = getattr(tc, "index", 0) or 0
                entry = self._tool_calls.setdefault(key, {"name": None, "arguments": []})
                fn = getattr(tc, "function", None)
                fn_name = getattr(fn, "name", None) if fn is not None else None
                fn_args = getattr(fn, "arguments", None) if fn is not None else None
                if fn_name:
                    entry["name"] = fn_name
                if fn_args:
                    entry["arguments"].append(fn_args)

    def result(self) -> Dict[str, Any]:
        texts = ["".join(parts) for _, parts in sorted(self._texts.items())]
        output: Optional[str] = "\n".join(t for t in texts if t) or None
        if output is None and self._tool_calls:
            described = [
                f"{entry['name'] or 'unknown'}({''.join(entry['arguments'])})"
                for _, entry in sorted(self._tool_calls.items())
            ]
            output = "[tool call] " + ", ".join(described)
        input_tokens, output_tokens, cache_read_tokens = _extract_usage_tokens(self._usage)
        return {
            "_start_t": self._start_t,
            "output": output,
            "model": self._model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
        }


def patch_openai_client(
    client: Any,
    tracer: Tracer,
    name: str = "openai-agent",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> None:
    """
    Monkey-patch ``client.chat.completions.create`` to automatically send a
    trace for every non-streaming call.

    The original method is still called and its return value is passed
    through unchanged so nothing in the caller needs to change. Works with
    both sync (``OpenAI``) and async (``AsyncOpenAI``) clients - the async
    client's ``create()`` returns a coroutine, which is detected and awaited
    before the trace is built.

    Calls made with ``stream=True`` return a transparent proxy over the
    provider's stream (see ``_traced_call.TracedStream``): iteration, ``with``,
    ``close()`` and attribute access all pass through to the real stream, and
    the trace is built from the chunks the caller actually consumed - text and
    tool calls assembled from the deltas, token usage from the final chunk
    when ``stream_options={"include_usage": True}`` was requested (OpenAI omits
    usage from streams otherwise), latency to the last chunk, and the time to
    first token in the trace metadata.
    """
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    if completions is None:
        raise ValueError("Provided client does not have a .chat.completions attribute")

    _patch_chat_completions_create(completions, tracer, name, metadata, session_id)


def _patch_chat_completions_create(
    completions_resource: Any,
    tracer: Tracer,
    name: str,
    metadata: Optional[Dict[str, Any]],
    session_id: Optional[str],
    framework: str = "openai",
) -> None:
    # `framework` exists for OpenAI-compatible endpoints served by other vendors
    # (agentx.integrations.nvidia_nim stamps "nvidia-nim" through here) - the request/response
    # shapes are identical, so they share this machinery instead of duplicating it.
    original = completions_resource.create
    if getattr(original, "_agentx_patched", False):
        return  # already patched

    def patched_create(*args, **kwargs):
        start_t = time.time()
        input_messages = kwargs.get("messages") or (args[0] if args else None)
        model = kwargs.get("model")
        input_repr = _safe_serialize(input_messages)
        tool_definitions = capture_tool_definitions(kwargs.get("tools"))

        if kwargs.get("stream"):
            # The parent is fixed at call time: the stream finalizes later, possibly inside an
            # unrelated span (or none), and must not attach to whatever is active then.
            parent = tracer.current_span

            def on_stream_finish(collected: Dict[str, Any], error: Optional[str]) -> None:
                global _warned_stream_usage
                ttft = collected.get("time_to_first_token_ms")
                call_metadata: Dict[str, Any] = {"streaming": True}
                if ttft is not None:
                    call_metadata["timeToFirstTokenMs"] = ttft
                if error is None and collected.get("input_tokens") is None and not _warned_stream_usage:
                    _warned_stream_usage = True
                    logger.warning(
                        "Streamed %s call carried no token usage - pass stream_options={\"include_usage\": True} "
                        "so traces (and cost) reflect streamed traffic.",
                        framework,
                    )
                finish_llm_call(
                    tracer,
                    name=name,
                    framework=framework,
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
                    error=error,
                    tool_definitions=tool_definitions,
                )

            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                on_stream_finish({}, str(exc))
                raise
            return trace_stream(result, _ChatCompletionStreamAccumulator(start_t), on_stream_finish)

        def on_finish(response: Optional[Any], error: Optional[str]) -> None:
            end_t = time.time()
            output = None
            input_tokens = None
            output_tokens = None
            cache_read_tokens = None
            if response is not None:
                output = _extract_output_text(response)
                try:
                    input_tokens, output_tokens, cache_read_tokens = _extract_usage_tokens(
                        getattr(response, "usage", None)
                    )
                except Exception:
                    pass

            finish_llm_call(
                tracer,
                name=name,
                framework=framework,
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
                error=error,
                tool_definitions=tool_definitions,
            )

        return call_and_trace(original, args, kwargs, on_finish)

    patched_create._agentx_patched = True
    completions_resource.create = patched_create
