from __future__ import annotations

from typing import Any, Callable

from agentx.evaluations.models import EvaluationCase, EvaluationResult
from agentx.evaluations.results import normalize_error, normalize_result


class RawCallableAdapter:
    """
    Wraps any Python callable that accepts an EvaluationCase and returns
    str | dict | EvaluationResult.

    Usage::

        def my_agent(case: EvaluationCase) -> str:
            return my_llm.invoke(case.query)

        adapter = RawCallableAdapter(my_agent)
        result = adapter.run(case)
    """

    def __init__(self, fn: Callable[[EvaluationCase], Any]):
        self._fn = fn

    def run(
        self, case: EvaluationCase, latency_ms: int | None = None
    ) -> EvaluationResult:
        import time

        start = time.monotonic()
        try:
            raw = self._fn(case)
        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return normalize_error(case, exc, latency_ms=elapsed)
        elapsed = int((time.monotonic() - start) * 1000)
        return normalize_result(case, raw, latency_ms=elapsed)
