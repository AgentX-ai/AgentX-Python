from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

from agentx.evaluations.models import (
    BatchAppendResponse,
    Dataset,
    EvaluationResult,
    EvaluationRun,
    EvaluationSubject,
    Report,
)

logger = logging.getLogger(__name__)

from agentx.util import _DEFAULT_API_BASE as _UTIL_API_BASE

_DEFAULT_BASE_URL = f"{_UTIL_API_BASE}/custom-agent-evaluations"
SDK_NAME = "agentx-python"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 2.0, 4.0]


class AgentXEvaluationsError(Exception):
    pass


class AgentXAuthError(AgentXEvaluationsError):
    pass


class AgentXValidationError(AgentXEvaluationsError):
    pass


class EvaluationsClient:
    def __init__(
        self, api_key: str, sdk_version: str = "unknown", base_url: str = None
    ):
        if not api_key:
            raise AgentXAuthError("AGENTX_API_KEY is required")
        self._api_key = api_key
        self._sdk_version = sdk_version
        # Priority: constructor arg > env var > SDK default
        # Always append /custom-agent-evaluations so users only need to provide /api/v1
        _api_base = (
            base_url or os.getenv("AGENTX_API_BASE_URL", _UTIL_API_BASE)
        ).rstrip("/")
        if not _api_base.endswith("/custom-agent-evaluations"):
            _api_base = f"{_api_base}/custom-agent-evaluations"
        self._base_url = _api_base
        self._session = requests.Session()
        self._session.headers.update(
            {
                "x-api-key": self._api_key,
                "Content-Type": "application/json",
                "User-Agent": f"{SDK_NAME}/{self._sdk_version}",
                "accept": "*/*",
            }
        )
        # Expose dataset builder factory
        from agentx.evaluations.datasets import DatasetClient

        self.datasets = DatasetClient(self)

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, timeout: int = 30, **kwargs) -> Any:
        url = f"{self._base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt, wait in enumerate([0.0] + _RETRY_BACKOFF):
            if wait:
                time.sleep(wait)
            try:
                resp = self._session.request(method, url, timeout=timeout, **kwargs)
            except requests.RequestException as e:
                last_exc = e
                logger.debug("Request error (attempt %d): %s", attempt + 1, e)
                continue

            if resp.status_code == 401:
                raise AgentXAuthError("Invalid or missing API key")
            if resp.status_code == 422:
                raise AgentXValidationError(resp.text)
            if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                logger.debug(
                    "Retryable status %d (attempt %d)", resp.status_code, attempt + 1
                )
                last_exc = AgentXEvaluationsError(f"HTTP {resp.status_code}")
                continue
            if not resp.ok:
                raise AgentXEvaluationsError(f"HTTP {resp.status_code}: {resp.text}")
            try:
                return resp.json()
            except Exception:
                return resp.text
        raise AgentXEvaluationsError(f"Request failed after retries: {last_exc}")

    # ------------------------------------------------------------------
    # Dataset endpoints
    # ------------------------------------------------------------------

    def create_dataset(self, payload: dict) -> Dataset:
        data = self._request("POST", "/datasets", json=payload)
        return Dataset(**data)

    def list_datasets(self) -> List[Dataset]:
        data = self._request("GET", "/datasets")
        return [
            Dataset(**d)
            for d in (data if isinstance(data, list) else data.get("datasets", []))
        ]

    def get_dataset(self, dataset_id: str) -> Dataset:
        data = self._request("GET", f"/datasets/{dataset_id}")
        return Dataset(**data)

    # ------------------------------------------------------------------
    # Run endpoints
    # ------------------------------------------------------------------

    def init_run(
        self,
        dataset_id: str,
        subject: EvaluationSubject,
        python_version: Optional[str] = None,
    ) -> EvaluationRun:
        from agentx.version import VERSION

        payload = {
            "datasetId": dataset_id,
            "evaluationSubject": subject.model_dump(by_alias=True, exclude_none=True),
            "runSource": "sdk",
            "sdk": {
                "name": SDK_NAME,
                "version": VERSION,
                "runnerVersion": "1",
                "pythonVersion": python_version or _python_version(),
            },
        }
        data = self._request("POST", "/runs", json=payload)
        return EvaluationRun(**data)

    def append_results(
        self, run_id: str, batch_id: str, results: List[EvaluationResult]
    ) -> BatchAppendResponse:
        payload = {
            "batchId": batch_id,
            "results": [_result_to_payload(r) for r in results],
        }
        data = self._request("POST", f"/runs/{run_id}/results", json=payload)
        return BatchAppendResponse(**data)

    def finalize_run(self, run_id: str) -> Dict[str, Any]:
        return self._request(
            "POST", f"/runs/{run_id}/finalize", json={"status": "completed"}
        )

    def analyze_run(self, run_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/runs/{run_id}/analyze", json={}, timeout=300)

    def get_run(self, run_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/runs/{run_id}")

    def get_report(self, run_id: str) -> Report:
        data = self._request("GET", f"/runs/{run_id}/report")
        return Report(**data)

    def get_missing_results(self, run_id: str) -> List[Dict[str, Any]]:
        data = self._request("GET", f"/runs/{run_id}/missing-results")
        return data if isinstance(data, list) else data.get("missing", [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_payload(r: EvaluationResult) -> dict:
    d = r.model_dump(by_alias=True, exclude_none=True)
    # Rename snake_case fields the model may have stored locally
    d["caseId"] = d.pop("case_id", d.get("caseId"))
    d["questionIndex"] = d.pop("question_index", d.get("questionIndex"))
    d["runNumber"] = d.pop("run_number", d.get("runNumber"))
    d["idempotencyKey"] = d.pop("idempotency_key", d.get("idempotencyKey"))
    return {k: v for k, v in d.items() if v is not None}


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}"
