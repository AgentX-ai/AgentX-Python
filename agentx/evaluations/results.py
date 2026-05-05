from __future__ import annotations

import time
from typing import Any, Optional

from agentx.evaluations.models import (
    EvaluationCase,
    EvaluationResult,
    ResultError,
    ResultTimings,
)
from agentx.evaluations.tracing import build_trace
from agentx.evaluations.redaction import redact_dict


def normalize_result(
    case: EvaluationCase,
    raw: Any,
    latency_ms: Optional[int] = None,
) -> EvaluationResult:
    """Turn whatever the user's callable returned into a normalised EvaluationResult."""
    if isinstance(raw, EvaluationResult):
        raw.case_id = case.case_id
        raw.question_index = case.question_index
        raw.run_number = case.run_number
        return raw

    output: Optional[dict] = None
    trace = None
    metadata: Optional[dict] = None
    error: Optional[ResultError] = None

    if isinstance(raw, str):
        output = {"text": raw}

    elif isinstance(raw, dict):
        text = raw.get("output") or raw.get("text") or raw.get("response") or ""
        if isinstance(text, dict):
            output = text
        else:
            output = {"text": str(text)} if text else None

        trace = build_trace(raw.get("trace") or raw.get("observable_trace"))
        meta_raw = raw.get("metadata")
        if isinstance(meta_raw, dict):
            metadata = redact_dict(meta_raw)

        err_raw = raw.get("error")
        if err_raw:
            if isinstance(err_raw, dict):
                error = ResultError(
                    type=err_raw.get("type", "Exception"),
                    message=err_raw.get("message", str(err_raw)),
                    retryable=err_raw.get("retryable", False),
                )
            else:
                error = ResultError(type="Exception", message=str(err_raw))
            if output is None:
                output = {"text": ""}
    else:
        output = {"text": str(raw)} if raw is not None else {"text": ""}

    return EvaluationResult(
        case_id=case.case_id,
        question_index=case.question_index,
        run_number=case.run_number,
        input={"query": case.query},
        output=output,
        observableTrace=trace,
        error=error,
        timings=ResultTimings(latencyMs=latency_ms) if latency_ms is not None else None,
        metadata=metadata,
    )


def normalize_error(case: EvaluationCase, exc: Exception, latency_ms: Optional[int] = None) -> EvaluationResult:
    """Build a failed-case result from an exception."""
    return EvaluationResult(
        case_id=case.case_id,
        question_index=case.question_index,
        run_number=case.run_number,
        input={"query": case.query},
        output={"text": ""},
        observableTrace=None,
        error=ResultError(
            type=type(exc).__name__,
            message=str(exc),
            retryable=False,
        ),
        timings=ResultTimings(latencyMs=latency_ms) if latency_ms is not None else None,
    )
