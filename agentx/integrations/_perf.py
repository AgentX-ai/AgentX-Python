"""
Shared helper for building a performance_summary that matches the AgentX
platform agent format.

External framework integrations collect:
  - execution_steps  — LLM inference calls (name, duration_ms, start_time, end_time)
  - tool_call_steps  — function/tool invocations (same shape)

``build_performance_summary`` merges them into the full structure expected by
the backend ingest endpoint and the AgentX UI.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_performance_summary(
    total_duration_ms: float,
    execution_steps: Optional[List[Dict[str, Any]]] = None,
    tool_call_steps: Optional[List[Dict[str, Any]]] = None,
    retrieval_steps: Optional[List[Dict[str, Any]]] = None,
    has_errors: bool = False,
) -> Dict[str, Any]:
    """
    Return a ``performance_summary`` dict compatible with the AgentX platform
    agent format.

    Args:
        total_duration_ms:  Wall-clock duration of the entire run.
        execution_steps:    LLM call entries — each must have ``name``,
                            ``duration_ms``, and optionally ``start_time`` /
                            ``end_time`` as Unix floats, plus ``model``,
                            ``input``, ``output``, ``inputTokenSize``,
                            ``outputTokenSize`` for per-call drill-down.
        tool_call_steps:    Tool / function call entries — same shape, plus
                            optional ``input`` / ``output``.
        retrieval_steps:    RAG / vector-store retrieval entries — same shape,
                            plus optional ``query`` (str), ``doc_count`` (int)
                            and ``output`` (the retrieved content).
        has_errors:         Set to True when the run ended with an exception.

    None of these fields are truncated here — callers decide what to capture.
    """
    execution_steps = execution_steps or []
    tool_call_steps = tool_call_steps or []
    retrieval_steps = retrieval_steps or []

    # Merge all items with their phase type, sort by start_time
    all_items: List[tuple[Dict[str, Any], str]] = (
        [(s, "execution_step") for s in execution_steps]
        + [(t, "tool_call") for t in tool_call_steps]
        + [(r, "retrieval") for r in retrieval_steps]
    )
    all_items.sort(key=lambda x: x[0].get("start_time") or 0)

    unified: List[Dict[str, Any]] = []
    out_steps: List[Dict[str, Any]] = []
    out_tools: List[Dict[str, Any]] = []
    out_retrievals: List[Dict[str, Any]] = []

    for order, (item, phase_type) in enumerate(all_items, start=1):
        entry: Dict[str, Any] = {
            "name": item["name"],
            "duration_ms": round(float(item["duration_ms"]), 3),
            "start_order": order,
            "phase_type": phase_type,
        }
        if item.get("start_time") is not None:
            entry["start_time"] = item["start_time"]
        if item.get("end_time") is not None:
            entry["end_time"] = item["end_time"]
        # Per-step drill-down detail — untruncated, present when the caller
        # captured it (e.g. an LLM call's model/prompt/completion, a tool
        # call's args/result, or a retrieval's returned documents).
        if item.get("model"):
            entry["model"] = item["model"]
        if item.get("input") is not None:
            entry["input"] = item["input"]
        if item.get("output") is not None:
            entry["output"] = item["output"]
        # Per-call token usage — populated for LLM call execution_steps by
        # integrations that read it off the provider response (e.g. Anthropic's
        # response.usage), so performance_summary carries a token breakdown
        # per call, not just the trace-level input_tokens/output_tokens total.
        # Named to match PromptTrace.inputTokenSize/outputTokenSize on the backend.
        if item.get("inputTokenSize") is not None:
            entry["inputTokenSize"] = item["inputTokenSize"]
        if item.get("outputTokenSize") is not None:
            entry["outputTokenSize"] = item["outputTokenSize"]
        # Retrieval-specific fields
        if phase_type == "retrieval":
            if item.get("query"):
                entry["query"] = item["query"]
            if item.get("doc_count") is not None:
                entry["doc_count"] = item["doc_count"]
        unified.append(entry)

        flat = {k: v for k, v in entry.items() if k != "phase_type"}
        if phase_type == "execution_step":
            out_steps.append(flat)
        elif phase_type == "tool_call":
            out_tools.append(flat)
        else:
            out_retrievals.append(flat)

    tools_total_ms = round(sum(t["duration_ms"] for t in out_tools), 3)
    retrievals_total_ms = round(sum(r["duration_ms"] for r in out_retrievals), 3)

    return {
        "total_duration_ms": round(float(total_duration_ms), 3),
        "todo_tasks_enabled": None,
        "main_phases": [],
        "tool_calls": out_tools,
        "mcp_tool_calls": [],
        "action_tool_calls": [],
        "delegate_calls": [],
        "execution_steps": out_steps,
        "knowledge_retrievals": out_retrievals,
        "todo_operations": [],
        "detailed_phases": [],
        "active_phases": [],
        "has_errors": has_errors,
        "memory_actions": None,
        "unified_timeline": unified,
        "statistics": {
            "total_main_phases": 0,
            "total_tool_calls": len(out_tools),
            "total_mcp_tool_calls": 0,
            "total_action_tool_calls": 0,
            "total_delegate_calls": 0,
            "total_execution_steps": len(out_steps),
            "total_knowledge_retrievals": len(out_retrievals),
            "total_todo_operations": 0,
            "main_phases_total_ms": 0,
            "tool_calls_total_ms": tools_total_ms,
            "knowledge_retrievals_total_ms": retrievals_total_ms,
            "mcp_tool_calls_total_ms": 0,
            "todo_operations_total_ms": 0,
        },
    }


def _merge_steps(
    summary: Dict[str, Any],
    steps: List[Dict[str, Any]],
    *,
    phase_type: str,
    target_key: str,
    stats_count_key: str,
    stats_total_ms_key: str,
    extra_fields: tuple = (),
    default_name_prefix: str = "Step",
) -> Dict[str, Any]:
    """Fold externally-recorded steps into an already-built ``performance_summary``."""
    if not steps:
        return summary

    next_order = len(summary.get("unified_timeline", [])) + 1
    for i, item in enumerate(steps):
        entry: Dict[str, Any] = {
            "name": item.get("name") or f"{default_name_prefix} {i + 1}",
            "duration_ms": round(float(item.get("duration_ms") or 0), 3),
            "start_order": next_order + i,
            "phase_type": phase_type,
        }
        if item.get("start_time") is not None:
            entry["start_time"] = item["start_time"]
        if item.get("end_time") is not None:
            entry["end_time"] = item["end_time"]
        for field in extra_fields:
            value = item.get(field)
            if value not in (None, ""):
                entry[field] = value

        summary["unified_timeline"].append(entry)
        summary[target_key].append({k: v for k, v in entry.items() if k != "phase_type"})

    summary["unified_timeline"].sort(key=lambda x: x.get("start_time") or 0)
    summary["statistics"][stats_count_key] = len(summary[target_key])
    summary["statistics"][stats_total_ms_key] = round(
        sum(r["duration_ms"] for r in summary[target_key]), 3
    )
    return summary


def merge_retrieval_steps(
    summary: Dict[str, Any],
    retrieval_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Fold externally-recorded retrieval steps (e.g. from ``Tracer.record_retrieval``,
    used for hand-rolled RAG lookups the framework integration can't see) into an
    already-built ``performance_summary``.
    """
    return _merge_steps(
        summary,
        retrieval_steps,
        phase_type="retrieval",
        target_key="knowledge_retrievals",
        stats_count_key="total_knowledge_retrievals",
        stats_total_ms_key="knowledge_retrievals_total_ms",
        extra_fields=("query", "doc_count", "output"),
        default_name_prefix="Retrieval",
    )


def merge_tool_call_steps(
    summary: Dict[str, Any],
    tool_call_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Fold externally-recorded tool calls (e.g. from ``Tracer.record_tool_call``,
    used when a tool executes outside a framework integration's visibility —
    such as a manual Anthropic tool-use loop) into an already-built
    ``performance_summary``.
    """
    return _merge_steps(
        summary,
        tool_call_steps,
        phase_type="tool_call",
        target_key="tool_calls",
        stats_count_key="total_tool_calls",
        stats_total_ms_key="tool_calls_total_ms",
        extra_fields=("input", "output"),
        default_name_prefix="Tool Call",
    )
