from __future__ import annotations

import logging
import os
import time
from typing import Any, List, Optional

import requests

from agentx.monitor.models import MonitorPattern, MonitorProfile, MonitorSignal

logger = logging.getLogger(__name__)

from agentx.util import _DEFAULT_API_BASE as _UTIL_API_BASE

SDK_NAME = "agentx-python"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 2.0, 4.0]


class AgentXMonitorError(Exception):
    pass


class AgentXAuthError(AgentXMonitorError):
    pass


class AgentXValidationError(AgentXMonitorError):
    pass


class MonitorClient:
    """Low-level HTTP client for the Monitor API (``/monitor``). Accessed via
    ``client.monitor`` on the top-level :class:`agentx.AgentX` instance; most callers
    should use ``client.monitor.patterns.builder(...)`` instead of this directly."""

    def __init__(
        self,
        api_key: str,
        sdk_version: str = "unknown",
        base_url: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ):
        if not api_key:
            raise AgentXAuthError("AGENTX_API_KEY is required")
        self._api_key = api_key
        self._sdk_version = sdk_version
        # Falls back to the caller's default workspace server-side when unset — mirrors
        # EvaluationsClient. Without this, pattern creation silently lands in whatever
        # workspace the API key's user defaults to, not the one the caller intended.
        self._workspace_id = workspace_id
        _api_base = (
            base_url or os.getenv("AGENTX_API_BASE_URL", _UTIL_API_BASE)
        ).rstrip("/")
        if not _api_base.endswith("/monitor"):
            _api_base = f"{_api_base}/monitor"
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

        from agentx.monitor.patterns import MonitorPatternClient
        from agentx.monitor.signals import MonitorSignalClient
        from agentx.monitor.profile import MonitorProfileClient

        self.patterns = MonitorPatternClient(self)
        self.signals = MonitorSignalClient(self)
        self.profile = MonitorProfileClient(self)

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _with_workspace(self, payload: dict) -> dict:
        if self._workspace_id and not payload.get("workspaceId"):
            return {**payload, "workspaceId": self._workspace_id}
        return payload

    def _workspace_params(self) -> Optional[dict]:
        return {"workspaceId": self._workspace_id} if self._workspace_id else None

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
                last_exc = AgentXMonitorError(f"HTTP {resp.status_code}")
                continue
            if not resp.ok:
                raise AgentXMonitorError(f"HTTP {resp.status_code}: {resp.text}")
            try:
                return resp.json()
            except Exception:
                return resp.text
        raise AgentXMonitorError(f"Request failed after retries: {last_exc}")

    # ------------------------------------------------------------------
    # Pattern endpoints
    # ------------------------------------------------------------------

    def create_pattern(self, payload: dict) -> MonitorPattern:
        data = self._request("POST", "/patterns", json=self._with_workspace(payload))
        return MonitorPattern(**data["pattern"])

    def list_patterns(self) -> List[MonitorPattern]:
        data = self._request("GET", "/patterns", params=self._workspace_params())
        return [MonitorPattern(**p) for p in data.get("patterns", [])]

    def get_pattern(self, pattern_id: str) -> MonitorPattern:
        data = self._request(
            "GET", f"/patterns/{pattern_id}", params=self._workspace_params()
        )
        return MonitorPattern(**data["pattern"])

    # ------------------------------------------------------------------
    # Signal endpoints
    # ------------------------------------------------------------------

    def list_signals(
        self,
        polarity: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[MonitorSignal]:
        params = {**(self._workspace_params() or {})}
        if polarity is not None:
            params["polarity"] = polarity
        if status is not None:
            params["status"] = status
        if severity is not None:
            params["severity"] = severity
        if agent_id is not None:
            params["agentId"] = agent_id
        if limit is not None:
            params["limit"] = limit
        data = self._request("GET", "/signals", params=params)
        return [MonitorSignal(**s) for s in data.get("signals", [])]

    def get_signal(self, signal_id: str) -> MonitorSignal:
        data = self._request(
            "GET", f"/signals/{signal_id}", params=self._workspace_params()
        )
        return MonitorSignal(**data["signal"])

    # ------------------------------------------------------------------
    # Profile endpoints
    # ------------------------------------------------------------------

    def get_profile(self, agent_id: str) -> Optional[MonitorProfile]:
        data = self._request(
            "GET", f"/profiles/{agent_id}", params=self._workspace_params()
        )
        profile = data.get("profile")
        return MonitorProfile(**profile) if profile else None

    def update_profile(self, agent_id: str, payload: dict) -> MonitorProfile:
        data = self._request(
            "PUT", f"/profiles/{agent_id}", json=self._with_workspace(payload)
        )
        return MonitorProfile(**data["profile"])
