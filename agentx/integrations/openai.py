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

Streaming calls (``stream=True``) are passed through untouched and are not
currently traced - see ``patch_openai_client``'s docstring.

Requires: ``pip install "agentx-python[openai]"``
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from agentx.tracing.tracer import Tracer, _safe_serialize
from agentx.integrations._traced_call import capture_tool_definitions, call_and_trace, finish_llm_call


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

    Calls made with ``stream=True`` are passed through untouched and are not
    traced by this function: safely wrapping a (sync or async) chunk
    iterator without disrupting the caller's own consumption of it needs
    different handling than a single request/response call, so it's left
    unpatched rather than risking a partially-consumed or double-consumed
    stream for the caller.
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
) -> None:
    original = completions_resource.create
    if getattr(original, "_agentx_patched", False):
        return  # already patched

    def patched_create(*args, **kwargs):
        if kwargs.get("stream"):
            # Not traced - see patch_openai_client's docstring. Passed
            # through completely untouched, sync or async.
            return original(*args, **kwargs)

        start_t = time.time()
        input_messages = kwargs.get("messages") or (args[0] if args else None)
        model = kwargs.get("model")
        input_repr = _safe_serialize(input_messages)
        tool_definitions = capture_tool_definitions(kwargs.get("tools"))

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
                framework="openai",
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
