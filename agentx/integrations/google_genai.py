"""
Google Gen AI SDK integration for AgentX production tracing.

Usage::

    from agentx.integrations.google_genai import patch_genai_client
    from google import genai

    client = genai.Client(api_key="GEMINI_API_KEY")
    patch_genai_client(client, agentx.tracer, name="gemini-agent")

    # All subsequent client.models.generate_content() calls are now traced.

Requires: ``pip install "agentx-python[google-genai]"``
"""
from __future__ import annotations

import inspect
import time
from typing import Any, Dict, List, Optional

from agentx.tracing.tracer import Tracer, _safe_serialize
from agentx.integrations._traced_call import call_and_trace, finish_llm_call


def patch_genai_client(
    client: Any,
    tracer: Tracer,
    name: str = "gemini-agent",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> None:
    """
    Monkey-patch ``client.models.generate_content`` (and
    ``client.models.generate_content_stream`` if present) to automatically
    send a trace for every call.

    The original methods are still called and their return values are passed
    through unchanged — nothing in the caller needs to change. Works with
    both the sync client (``client.models``) and the async client
    (``client.aio.models``) — pass whichever ``.models`` resource you use;
    ``generate_content`` on the async side returns a coroutine, which is
    detected and awaited before the trace is built.

    Streaming (``client.models.generate_content_stream`` /
    ``client.aio.models.generate_content_stream``) is patched too, sync or
    async — detected via ``inspect.iscoroutinefunction``, which (unlike
    ``generate_content``/``create`` on this and other raw SDK clients) is
    reliable here since the SDK implements the async variant as a plain
    top-level ``async def``.

    Calling this function on an already-patched client is a no-op.
    """
    models = getattr(client, "models", None)
    if models is None:
        raise ValueError("Provided client does not have a .models attribute")

    _patch_generate_content(models, tracer, name, metadata, session_id)
    if hasattr(models, "generate_content_stream"):
        _patch_generate_content_stream(models, tracer, name, metadata, session_id)


def _extract_response_text(response: Any) -> Optional[str]:
    """
    Pull the generated text out of a GenerateContentResponse, falling back to
    a description of any function_call parts when the response is a pure
    tool call with no text (Gemini function calling).
    """
    # Convenience .text property (available on non-streaming responses)
    text = getattr(response, "text", None)
    if text and isinstance(text, str):
        return text
    # Fallback: walk candidates → content → parts
    candidates = getattr(response, "candidates", None) or []
    function_calls: List[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            t = getattr(part, "text", None)
            if t and isinstance(t, str):
                return t
            fc = getattr(part, "function_call", None)
            if fc is not None:
                name = getattr(fc, "name", "unknown")
                args = getattr(fc, "args", None)
                function_calls.append(f"{name}({args})")
    if function_calls:
        return "[tool call] " + ", ".join(function_calls)
    return None


def _patch_generate_content(
    models: Any,
    tracer: Tracer,
    name: str,
    metadata: Optional[Dict[str, Any]],
    session_id: Optional[str],
) -> None:
    original = models.generate_content
    if getattr(original, "_agentx_patched", False):
        return

    def patched(*args, **kwargs):
        start_t = time.time()
        model = kwargs.get("model") or (args[0] if args else None)
        contents = kwargs.get("contents") or (args[1] if len(args) > 1 else None)
        input_repr = contents if isinstance(contents, str) else _safe_serialize(contents)

        def on_finish(response: Optional[Any], error: Optional[str]) -> None:
            end_t = time.time()
            output = _extract_response_text(response) if response is not None else None
            input_tokens = None
            output_tokens = None
            cache_read_tokens = None
            if response is not None:
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    input_tokens = getattr(usage, "prompt_token_count", None)
                    output_tokens = getattr(usage, "candidates_token_count", None)
                    # prompt_token_count already includes this — a discount breakdown, same
                    # "total unchanged, cache portion reported alongside" posture as OpenAI's
                    # prompt_tokens_details.cached_tokens.
                    cache_read_tokens = getattr(usage, "cached_content_token_count", None)
            finish_llm_call(
                tracer,
                name=name,
                framework="google-genai",
                metadata=metadata,
                session_id=session_id,
                start_t=start_t,
                end_t=end_t,
                input_repr=input_repr,
                output=output,
                model=str(model) if model else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                error=error,
            )

        return call_and_trace(original, args, kwargs, on_finish)

    patched._agentx_patched = True
    models.generate_content = patched


def _patch_generate_content_stream(
    models: Any,
    tracer: Tracer,
    name: str,
    metadata: Optional[Dict[str, Any]],
    session_id: Optional[str],
) -> None:
    original_stream = models.generate_content_stream
    if getattr(original_stream, "_agentx_patched", False):
        return

    if inspect.iscoroutinefunction(original_stream):
        _patch_async_generate_content_stream(models, original_stream, tracer, name, metadata, session_id)
    else:
        _patch_sync_generate_content_stream(models, original_stream, tracer, name, metadata, session_id)


def _stream_input_and_model(args: tuple, kwargs: dict) -> tuple:
    model = kwargs.get("model") or (args[0] if args else None)
    contents = kwargs.get("contents") or (args[1] if len(args) > 1 else None)
    input_repr = contents if isinstance(contents, str) else _safe_serialize(contents)
    return str(model) if model else None, input_repr


def _patch_sync_generate_content_stream(
    models: Any,
    original_stream: Any,
    tracer: Tracer,
    name: str,
    metadata: Optional[Dict[str, Any]],
    session_id: Optional[str],
) -> None:
    def patched_stream(*args, **kwargs):
        start_t = time.time()
        model, input_repr = _stream_input_and_model(args, kwargs)
        accumulated_text: List[str] = []
        last_usage_metadata = None
        error: Optional[str] = None

        try:
            for chunk in original_stream(*args, **kwargs):
                text = getattr(chunk, "text", None)
                if text:
                    accumulated_text.append(text)
                # Track usage_metadata from the last chunk (Gemini includes it there)
                chunk_usage = getattr(chunk, "usage_metadata", None)
                if chunk_usage is not None:
                    last_usage_metadata = chunk_usage
                yield chunk
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            end_t = time.time()
            input_tokens = None
            output_tokens = None
            cache_read_tokens = None
            if last_usage_metadata is not None:
                input_tokens = getattr(last_usage_metadata, "prompt_token_count", None)
                output_tokens = getattr(last_usage_metadata, "candidates_token_count", None)
                cache_read_tokens = getattr(last_usage_metadata, "cached_content_token_count", None)
            output_repr = "".join(accumulated_text) or None
            finish_llm_call(
                tracer,
                name=name,
                framework="google-genai",
                metadata=metadata,
                session_id=session_id,
                start_t=start_t,
                end_t=end_t,
                input_repr=input_repr,
                output=output_repr,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                error=error,
            )

    patched_stream._agentx_patched = True
    models.generate_content_stream = patched_stream


def _patch_async_generate_content_stream(
    models: Any,
    original_stream: Any,
    tracer: Tracer,
    name: str,
    metadata: Optional[Dict[str, Any]],
    session_id: Optional[str],
) -> None:
    async def patched_stream(*args, **kwargs):
        # The real SDK method is itself `async def` and returns an async
        # iterable — callers use `async for chunk in await client.aio.models
        # .generate_content_stream(...)`. To preserve that exact shape,
        # `patched_stream` is also `async def`: awaiting it runs this setup
        # (including the real `await original_stream(...)` call) and
        # resolves to `traced_agen()`, an async generator object — not yet
        # iterated, so no chunk is consumed until the caller's `async for`
        # drives it.
        start_t = time.time()
        model, input_repr = _stream_input_and_model(args, kwargs)
        inner = await original_stream(*args, **kwargs)

        async def traced_agen():
            accumulated_text: List[str] = []
            last_usage_metadata = None
            error: Optional[str] = None
            try:
                async for chunk in inner:
                    text = getattr(chunk, "text", None)
                    if text:
                        accumulated_text.append(text)
                    chunk_usage = getattr(chunk, "usage_metadata", None)
                    if chunk_usage is not None:
                        last_usage_metadata = chunk_usage
                    yield chunk
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                end_t = time.time()
                input_tokens = None
                output_tokens = None
                cache_read_tokens = None
                if last_usage_metadata is not None:
                    input_tokens = getattr(last_usage_metadata, "prompt_token_count", None)
                    output_tokens = getattr(last_usage_metadata, "candidates_token_count", None)
                    cache_read_tokens = getattr(last_usage_metadata, "cached_content_token_count", None)
                output_repr = "".join(accumulated_text) or None
                finish_llm_call(
                    tracer,
                    name=name,
                    framework="google-genai",
                    metadata=metadata,
                    session_id=session_id,
                    start_t=start_t,
                    end_t=end_t,
                    input_repr=input_repr,
                    output=output_repr,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    error=error,
                )

        return traced_agen()

    patched_stream._agentx_patched = True
    models.generate_content_stream = patched_stream
