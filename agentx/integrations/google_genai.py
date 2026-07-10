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

import time
from typing import Any, Dict, Optional

from agentx.tracing.tracer import Tracer, _safe_serialize
from agentx.integrations._perf import build_performance_summary


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
    through unchanged — nothing in the caller needs to change.

    Calling this function on an already-patched client is a no-op.
    """
    models = getattr(client, "models", None)
    if models is None:
        raise ValueError("Provided client does not have a .models attribute")

    _patch_generate_content(models, tracer, name, metadata, session_id)
    if hasattr(models, "generate_content_stream"):
        _patch_generate_content_stream(models, tracer, name, metadata, session_id)


def _extract_response_text(response: Any) -> Optional[str]:
    """Pull the generated text out of a GenerateContentResponse."""
    # Convenience .text property (available on non-streaming responses)
    text = getattr(response, "text", None)
    if text and isinstance(text, str):
        return text
    # Fallback: walk candidates → content → parts
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            t = getattr(part, "text", None)
            if t and isinstance(t, str):
                return t
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
            model = kwargs.get("model") or (args[0] if args else None)
            contents = kwargs.get("contents") or (args[1] if len(args) > 1 else None)
            output = _extract_response_text(response) if response is not None else None
            input_tokens = None
            output_tokens = None
            if response is not None:
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    input_tokens = getattr(usage, "prompt_token_count", None)
                    output_tokens = getattr(usage, "candidates_token_count", None)
            perf = build_performance_summary(
                total_duration_ms=latency_ms,
                execution_steps=[{
                    "name": "LLM Call 1",
                    "duration_ms": latency_ms,
                    "start_time": start_t,
                    "end_time": end_t,
                }],
                has_errors=error is not None,
            )
            tracer._send(
                name=name,
                input=contents if isinstance(contents, str) else _safe_serialize(contents),
                output=output,
                latency_ms=latency_ms,
                error=error,
                framework="google-genai",
                model=str(model) if model else None,
                metadata=metadata,
                session_id=session_id,
                performance_summary=perf,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

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

    def patched_stream(*args, **kwargs):
        start_t = time.time()
        accumulated_text: list[str] = []
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
            latency_ms = int((end_t - start_t) * 1000)
            model = kwargs.get("model") or (args[0] if args else None)
            contents = kwargs.get("contents") or (args[1] if len(args) > 1 else None)
            input_tokens = None
            output_tokens = None
            if last_usage_metadata is not None:
                input_tokens = getattr(last_usage_metadata, "prompt_token_count", None)
                output_tokens = getattr(last_usage_metadata, "candidates_token_count", None)
            perf = build_performance_summary(
                total_duration_ms=latency_ms,
                execution_steps=[{
                    "name": "LLM Call 1",
                    "duration_ms": latency_ms,
                    "start_time": start_t,
                    "end_time": end_t,
                }],
                has_errors=error is not None,
            )
            tracer._send(
                name=name,
                input=contents if isinstance(contents, str) else _safe_serialize(contents),
                output="".join(accumulated_text) or None,
                latency_ms=latency_ms,
                error=error,
                framework="google-genai",
                model=str(model) if model else None,
                metadata=metadata,
                session_id=session_id,
                performance_summary=perf,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    patched_stream._agentx_patched = True
    models.generate_content_stream = patched_stream
