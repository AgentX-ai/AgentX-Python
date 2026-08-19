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


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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
        raw.is_smoke_test_variant = case.is_smoke_test_variant
        raw.smoke_test_variant_text = case.smoke_test_variant_text
        return raw

    output: Optional[dict] = None
    trace = None
    trace_id: Optional[str] = None
    retrieval_context = None
    metadata: Optional[dict] = None
    error: Optional[ResultError] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    if isinstance(raw, str):
        output = {"text": raw}

    elif isinstance(raw, dict):
        text = raw.get("output") or raw.get("text") or raw.get("response") or ""
        if isinstance(text, dict):
            output = text
        else:
            output = {"text": str(text)} if text else None

        trace = build_trace(raw.get("trace") or raw.get("observable_trace"))
        trace_id_raw = raw.get("trace_id") or raw.get("traceId")
        trace_id = str(trace_id_raw) if trace_id_raw else None
        retrieval_context = raw.get("retrieval_context") or raw.get("retrievalContext")
        meta_raw = raw.get("metadata")
        if isinstance(meta_raw, dict):
            metadata = redact_dict(meta_raw)

        # Extract token counts - top-level keys take priority, fall back to metadata
        input_tokens = _to_int(raw.get("input_tokens"))
        output_tokens = _to_int(raw.get("output_tokens"))
        if input_tokens is None and isinstance(meta_raw, dict):
            input_tokens = _to_int(
                meta_raw.get("input_tokens") or meta_raw.get("prompt_tokens")
            )
        if output_tokens is None and isinstance(meta_raw, dict):
            output_tokens = _to_int(
                meta_raw.get("output_tokens") or meta_raw.get("completion_tokens")
            )

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

    has_timings = (
        latency_ms is not None or input_tokens is not None or output_tokens is not None
    )
    return EvaluationResult(
        case_id=case.case_id,
        question_index=case.question_index,
        run_number=case.run_number,
        input={"query": case.query},
        output=output,
        observableTrace=trace,
        error=error,
        timings=(
            ResultTimings(
                latencyMs=latency_ms,
                inputTokens=input_tokens,
                outputTokens=output_tokens,
            )
            if has_timings
            else None
        ),
        metadata=metadata,
        traceId=trace_id,
        retrievalContext=retrieval_context,
        isSmokeTestVariant=case.is_smoke_test_variant,
        smokeTestVariantText=case.smoke_test_variant_text,
    )


def normalize_error(
    case: EvaluationCase, exc: Exception, latency_ms: Optional[int] = None
) -> EvaluationResult:
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
        isSmokeTestVariant=case.is_smoke_test_variant,
        smokeTestVariantText=case.smoke_test_variant_text,
    )
