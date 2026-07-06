"""
OpenAI Agents SDK integration for AgentX production tracing.

Usage::

    from agentx.integrations.openai_agents import AgentXTracingProcessor

    processor = AgentXTracingProcessor(agentx.tracer)

    # Register once at startup — affects all agent runs in the process
    from agents import add_trace_processor
    add_trace_processor(processor)

Requires: ``pip install agentx[openai-agents]``
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from agentx.tracing.tracer import Tracer, _safe_serialize


class AgentXTracingProcessor:
    """
    Implements the ``TracingProcessor`` interface expected by the OpenAI Agents SDK
    (``agents.add_trace_processor``).

    Sends one AgentX trace per top-level agent run span.
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
        # span_id → state for in-flight agent spans
        self._spans: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # TracingProcessor protocol
    # ------------------------------------------------------------------

    def on_trace_start(self, trace: Any) -> None:
        """Called when a new trace (top-level run) begins."""
        trace_id = getattr(trace, "trace_id", None) or str(id(trace))
        self._spans[trace_id] = {
            "start": time.time(),
            "name": getattr(trace, "name", "openai-agent"),
            "input": None,
            "tool_calls": [],
            "model": None,
        }

    def on_trace_end(self, trace: Any) -> None:
        """Called when the trace ends (agent run complete)."""
        trace_id = getattr(trace, "trace_id", None) or str(id(trace))
        state = self._spans.pop(trace_id, None)
        if state is None:
            return
        latency_ms = int((time.time() - state["start"]) * 1000)
        self._tracer._send(
            name=state["name"],
            input=state.get("input"),
            output=_safe_serialize(getattr(trace, "output", None)),
            latency_ms=latency_ms,
            framework="openai-agents",
            model=state.get("model"),
            tool_calls=state["tool_calls"] or None,
            metadata=self._metadata,
            session_id=self._session_id,
        )

    def on_span_start(self, span: Any) -> None:
        """Called for each nested span (LLM call, tool call, etc.)."""
        span_type = getattr(span, "span_data", None)
        if span_type is None:
            return

        # Capture model from LLM generation spans
        if hasattr(span_type, "model"):
            trace_id = getattr(span, "trace_id", None)
            if trace_id and trace_id in self._spans and not self._spans[trace_id].get("model"):
                self._spans[trace_id]["model"] = str(span_type.model)

        # Capture input on the root span
        if hasattr(span_type, "input") and span_type.input:
            trace_id = getattr(span, "trace_id", None)
            if trace_id and trace_id in self._spans and not self._spans[trace_id].get("input"):
                self._spans[trace_id]["input"] = _safe_serialize(span_type.input)

    def on_span_end(self, span: Any) -> None:
        """Called when a nested span ends; captures tool-call results."""
        span_data = getattr(span, "span_data", None)
        if span_data is None:
            return

        # Tool / function calls
        if hasattr(span_data, "tool_name") or hasattr(span_data, "name"):
            trace_id = getattr(span, "trace_id", None)
            if trace_id and trace_id in self._spans:
                tool_entry: Dict[str, Any] = {
                    "name": getattr(span_data, "tool_name", None)
                    or getattr(span_data, "name", "tool"),
                    "input": _safe_serialize(getattr(span_data, "input", None)),
                    "output": str(getattr(span_data, "output", ""))[:500],
                }
                if hasattr(span, "started_at") and hasattr(span, "ended_at"):
                    try:
                        tool_entry["latency_ms"] = int(
                            (span.ended_at - span.started_at) * 1000
                        )
                    except Exception:
                        pass
                self._spans[trace_id]["tool_calls"].append(tool_entry)

    def force_flush(self) -> None:
        self._tracer.flush()

    def shutdown(self) -> None:
        self._tracer.flush()
