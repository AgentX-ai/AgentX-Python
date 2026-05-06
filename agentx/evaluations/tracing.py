from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agentx.evaluations.models import ObservableTrace, TraceEvent

# Hard ceiling on trace payload (bytes when serialised to JSON)
MAX_TRACE_BYTES = 20_000


def build_trace(raw: Any) -> Optional[ObservableTrace]:
    """Convert whatever a user returns as 'trace' into an ObservableTrace."""
    if raw is None:
        return None
    if isinstance(raw, ObservableTrace):
        return _bound(raw)
    if isinstance(raw, dict):
        events_raw = raw.get("events", [])
        events = [_coerce_event(e) for e in events_raw if e is not None]
        return _bound(ObservableTrace(events=events))
    if isinstance(raw, list):
        events = [_coerce_event(e) for e in raw if e is not None]
        return _bound(ObservableTrace(events=events))
    return None


def _coerce_event(e: Any) -> TraceEvent:
    if isinstance(e, TraceEvent):
        return e
    if isinstance(e, dict):
        return TraceEvent(
            type=e.get("type", "unknown"),
            name=e.get("name"),
            summary=e.get("summary"),
            latency_ms=e.get("latency_ms") or e.get("latencyMs"),
            metadata=e.get("metadata"),
        )
    return TraceEvent(type="unknown", summary=str(e))


def _bound(trace: ObservableTrace) -> ObservableTrace:
    """Drop events until the serialised payload fits within MAX_TRACE_BYTES."""
    if len(json.dumps(trace.model_dump()).encode()) <= MAX_TRACE_BYTES:
        return trace
    kept: List[TraceEvent] = []
    size = 2  # for "[]"
    for event in trace.events:
        chunk = len(json.dumps(event.model_dump()).encode()) + 1
        if size + chunk > MAX_TRACE_BYTES:
            break
        kept.append(event)
        size += chunk
    return ObservableTrace(events=kept)
