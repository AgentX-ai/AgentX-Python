from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from agentx.evaluations.models import EvaluationCase, EvaluationResult
from agentx.evaluations.results import normalize_error, normalize_result


class HttpEndpointAdapter:
    """
    Calls a user-hosted HTTP endpoint for each evaluation case.
    The SDK (running locally) makes the request — the AgentX API never
    touches the customer's endpoint.

    The endpoint receives a POST with::

        {"query": "...", "case_id": "...", "metadata": {...}}

    It should respond with a JSON body containing at least ``output``
    (or ``text``) and optionally ``trace`` and ``metadata``.

    Usage::

        adapter = HttpEndpointAdapter(
            url="http://localhost:8080/eval",
            headers={"Authorization": "Bearer my-token"},
            timeout=30,
        )
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        method: str = "POST",
    ):
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout
        self._method = method.upper()

    def run(self, case: EvaluationCase) -> EvaluationResult:
        payload = {
            "query": case.query,
            "case_id": case.case_id,
            "question_index": case.question_index,
            "run_number": case.run_number,
        }
        start = time.monotonic()
        try:
            resp = requests.request(
                self._method,
                self._url,
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            elapsed = int((time.monotonic() - start) * 1000)
            raw = resp.json()
        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return normalize_error(case, exc, latency_ms=elapsed)

        return normalize_result(case, raw, latency_ms=elapsed)
