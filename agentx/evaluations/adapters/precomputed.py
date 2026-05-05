from __future__ import annotations

from typing import Any, Dict, List, Union

from agentx.evaluations.models import EvaluationCase, EvaluationResult
from agentx.evaluations.results import normalize_result


class PrecomputedAdapter:
    """
    Adapter for pre-computed outputs — useful when you already have agent
    responses and just want AgentX to score them.

    Accepts a list or dict keyed by case_id::

        outputs = {
            "case-0": "You can reset your password from account settings.",
            "case-1": {"output": "Contact support@example.com", "metadata": {...}},
        }
        adapter = PrecomputedAdapter(outputs)
    """

    def __init__(self, outputs: Union[List[Any], Dict[str, Any]]):
        if isinstance(outputs, list):
            self._lookup: Dict[str, Any] = {str(i): v for i, v in enumerate(outputs)}
        else:
            self._lookup = {str(k): v for k, v in outputs.items()}

    def run(self, case: EvaluationCase) -> EvaluationResult:
        raw = self._lookup.get(case.case_id) or self._lookup.get(str(case.question_index))
        if raw is None:
            raw = ""
        return normalize_result(case, raw, latency_ms=0)
