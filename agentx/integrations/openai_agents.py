"""
OpenAI Agents SDK integration for AgentX production tracing.

Usage::

    from agentx.integrations.openai_agents import AgentXTracingProcessor

    processor = AgentXTracingProcessor(agentx.tracer)

    # Register once at startup — affects all agent runs in the process
    from agents import add_trace_processor
    add_trace_processor(processor)

Requires: ``pip install "agentx-python[openai-agents]"``
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agentx.tracing.tracer import Tracer, _safe_serialize
from agentx.integrations._perf import build_performance_summary


def _iso_to_ts(iso: Optional[str]) -> Optional[float]:
    """Parse an ISO-8601 timestamp string to a Unix timestamp float."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _span_latency_ms(span: Any) -> Optional[int]:
    """Return span duration in ms from ISO started_at / ended_at strings."""
    t0 = _iso_to_ts(getattr(span, "started_at", None))
    t1 = _iso_to_ts(getattr(span, "ended_at", None))
    if t0 is not None and t1 is not None:
        return max(0, int((t1 - t0) * 1000))
    return None


def _extract_text(output: Any) -> Optional[str]:
    """
    Best-effort extraction of a human-readable string from a GenerationSpanData
    or ResponseSpanData output, which can be a list of message dicts in either
    the Chat Completions or Responses API format.
    """
    if output is None:
        return None
    if isinstance(output, str):
        return output
    if not isinstance(output, (list, tuple)):
        return _safe_serialize(output)

    for item in reversed(output):
        if not isinstance(item, dict):
            # Could be a pydantic model — try .text or .content
            text = getattr(item, "text", None) or getattr(item, "content", None)
            if text and isinstance(text, str):
                return text
            continue

        # Responses API: {"type": "message", "content": [{"type": "output_text", "text": "..."}]}
        if item.get("type") == "message":
            content = item.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        return part.get("text")

        # Chat Completions API: {"role": "assistant", "content": "..."}
        role = item.get("role", "")
        if role == "assistant":
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text")

    return _safe_serialize(output)


def _extract_input_text(input_data: Any) -> Optional[str]:
    """Extract the user's input query from a generation span's input messages."""
    if input_data is None:
        return None
    if isinstance(input_data, str):
        return input_data
    if not isinstance(input_data, (list, tuple)):
        return _safe_serialize(input_data)

    # Walk messages and grab the last user message content
    user_text = None
    for item in input_data:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str):
                user_text = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                        user_text = part.get("text")

    return user_text or _safe_serialize(input_data)


class AgentXTracingProcessor:
    """
    Implements the ``TracingProcessor`` interface expected by the OpenAI Agents SDK
    (``agents.add_trace_processor``).

    Sends one AgentX trace per top-level agent run.
    """

    def __init__(
        self,
        tracer: Tracer,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._tracer = tracer
        self._metadata = metadata
        self._session_id = session_id
        # trace_id → accumulated state
        self._spans: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # TracingProcessor protocol
    # ------------------------------------------------------------------

    def on_trace_start(self, trace: Any) -> None:
        trace_id = getattr(trace, "trace_id", None) or str(id(trace))
        self._spans[trace_id] = {
            "start": time.time(),
            "name": getattr(trace, "name", "openai-agent"),
            "input": None,
            "output": None,
            "tool_calls": [],
            "model": None,
            "execution_steps": [],
            "perf_tool_calls": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "error": None,
        }

    def on_trace_end(self, trace: Any) -> None:
        trace_id = getattr(trace, "trace_id", None) or str(id(trace))
        state = self._spans.pop(trace_id, None)
        if state is None:
            return
        latency_ms = int((time.time() - state["start"]) * 1000)
        perf = build_performance_summary(
            total_duration_ms=latency_ms,
            execution_steps=state["execution_steps"],
            tool_call_steps=state["perf_tool_calls"],
        )
        self._tracer._send(
            name=state["name"],
            input=state.get("input"),
            output=state.get("output"),
            latency_ms=latency_ms,
            error=state.get("error"),
            framework="openai-agents",
            model=state.get("model"),
            tool_calls=state["tool_calls"] or None,
            metadata=self._metadata,
            session_id=self._session_id,
            performance_summary=perf,
            input_tokens=state["input_tokens"] or None,
            output_tokens=state["output_tokens"] or None,
        )

    def on_span_start(self, span: Any) -> None:
        pass  # All data is captured in on_span_end when fields are fully populated

    def on_span_end(self, span: Any) -> None:
        span_data = getattr(span, "span_data", None)
        if span_data is None:
            return

        trace_id = getattr(span, "trace_id", None)
        if not trace_id or trace_id not in self._spans:
            return

        state = self._spans[trace_id]
        span_type = getattr(span_data, "type", None)

        # `TracingProcessor` has no dedicated error callback — a span's
        # failure lives on `span.error` (a `SpanError | None`) instead.
        # First error wins: one failed span is enough to flag the trace.
        span_error = getattr(span, "error", None)
        if span_error is not None and state.get("error") is None:
            error_message = getattr(span_error, "message", None) or str(span_error)
            error_data = getattr(span_error, "data", None)
            state["error"] = f"{error_message} ({error_data})" if error_data else error_message

        t0 = _iso_to_ts(getattr(span, "started_at", None))
        t1 = _iso_to_ts(getattr(span, "ended_at", None))

        if span_type == "generation":
            call_input = _extract_input_text(span_data.input) if span_data.input else None
            call_output = _extract_text(span_data.output) if span_data.output else None
            call_model = str(span_data.model) if span_data.model else None

            # Capture input from the first generation span
            if state["input"] is None and call_input:
                state["input"] = call_input
            # Always update output to the latest generation (last one wins = final reply)
            if call_output:
                state["output"] = call_output
            # Capture model name
            if not state["model"] and call_model:
                state["model"] = call_model

            # Token counts — usage is a dict with "input_tokens" / "output_tokens"
            usage = getattr(span_data, "usage", None)
            call_input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
            call_output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
            if call_input_tokens is not None:
                state["input_tokens"] += int(call_input_tokens)
            if call_output_tokens is not None:
                state["output_tokens"] += int(call_output_tokens)

            # Execution step
            if t0 is not None and t1 is not None:
                steps = state["execution_steps"]
                steps.append({
                    "name": f"LLM Call {len(steps) + 1}",
                    "duration_ms": (t1 - t0) * 1000,
                    "start_time": t0,
                    "end_time": t1,
                    "model": call_model,
                    "input": call_input,
                    "output": call_output,
                    "inputTokenSize": call_input_tokens,
                    "outputTokenSize": call_output_tokens,
                })

        elif span_type == "response":
            # Responses API path — extract from the response object
            response = getattr(span_data, "response", None)
            call_input = None
            call_output = None
            call_model = None
            call_input_tokens = None
            call_output_tokens = None
            if response is not None:
                raw_input = getattr(span_data, "input", None)
                if raw_input:
                    call_input = _extract_input_text(raw_input) if isinstance(raw_input, list) else str(raw_input)
                if state["input"] is None and call_input:
                    state["input"] = call_input
                output_items = getattr(response, "output", None)
                if output_items:
                    call_output = _extract_text(output_items)
                    state["output"] = call_output
                model = getattr(response, "model", None)
                if model:
                    call_model = str(model)
                if not state["model"] and call_model:
                    state["model"] = call_model
                # Token counts from response.usage or span_data.usage
                usage = getattr(response, "usage", None) or getattr(span_data, "usage", None)
                if isinstance(usage, dict):
                    call_input_tokens = usage.get("input_tokens")
                    call_output_tokens = usage.get("output_tokens")
                elif usage is not None:
                    call_input_tokens = getattr(usage, "input_tokens", None)
                    call_output_tokens = getattr(usage, "output_tokens", None)
                if call_input_tokens is not None:
                    state["input_tokens"] += int(call_input_tokens)
                if call_output_tokens is not None:
                    state["output_tokens"] += int(call_output_tokens)
            # Execution step
            if t0 is not None and t1 is not None:
                steps = state["execution_steps"]
                steps.append({
                    "name": f"LLM Call {len(steps) + 1}",
                    "duration_ms": (t1 - t0) * 1000,
                    "start_time": t0,
                    "end_time": t1,
                    "model": call_model,
                    "input": call_input,
                    "output": call_output,
                    "inputTokenSize": call_input_tokens,
                    "outputTokenSize": call_output_tokens,
                })

        elif span_type == "function":
            # Tool / function call
            latency = _span_latency_ms(span)
            tool_output = str(span_data.output) if span_data.output is not None else None
            tool_entry: Dict[str, Any] = {
                "name": span_data.name,
                "input": span_data.input,
                "output": tool_output,
            }
            if latency is not None:
                tool_entry["latency_ms"] = latency
            state["tool_calls"].append(tool_entry)
            # Perf tool call with timestamps
            if t0 is not None and t1 is not None:
                state["perf_tool_calls"].append({
                    "name": span_data.name,
                    "duration_ms": (t1 - t0) * 1000,
                    "start_time": t0,
                    "end_time": t1,
                    "input": span_data.input,
                    "output": tool_output,
                })

    def force_flush(self) -> None:
        self._tracer.flush()

    def shutdown(self) -> None:
        self._tracer.flush()
