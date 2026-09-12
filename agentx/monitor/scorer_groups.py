"""Scorer groups (self-host): scorers of any kind - LLM judges, patterns, custom code/external
scorers - composed into ONE 0-10 score via per-member weights and optional must-pass gates.
Members are references: ``{"kind": "judge" | "pattern" | "custom", "refId": ..., "weight": ...,
"gate": ...}``. Grade a dataset run with a group by passing its id as ``scorer_group_id`` to
``client.evaluations.run(...)``; give it an ``online`` profile to score sampled live traffic and
raise Signals below the alert threshold."""

from typing import Any, Dict, List, Optional

import requests

from agentx.exceptions import AgentXError, AgentXAuthError, AgentXValidationError


class AgentXScorerGroupsError(AgentXError):
    """``status_code`` carries the HTTP status when the error came from a server
    response; ``None`` for transport-level failures."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ScorerGroup(dict):
    """Wire object (dict subclass so unknown fields round-trip)."""

    @property
    def id(self) -> str:
        return self["_id"]

    @property
    def name(self) -> str:
        return self["name"]

    @property
    def members(self) -> List[Dict[str, Any]]:
        return list(self.get("members") or [])

    @property
    def online(self) -> Optional[Dict[str, Any]]:
        return self.get("online")


class ScorerGroupsClient:
    def __init__(self, api_key: str, base_url: str, workspace_id: Optional[str] = None):
        self._api_key = api_key
        # Same workspace pinning MonitorClient does - without it, group CRUD silently lands
        # in whatever workspace the API key's user defaults to.
        self._workspace_id = workspace_id
        self._base = base_url.rstrip("/") + "/agent-monitoring/scorer-groups"

    def _request(self, method: str, url: str, json: Optional[Dict[str, Any]] = None) -> Any:
        params = None
        if self._workspace_id:
            # Mirrors MonitorClient._workspace_params/_with_workspace: GETs (and DELETEs)
            # carry workspaceId as a query param, write bodies carry it as a field.
            if method.upper() in ("POST", "PUT", "PATCH"):
                if json is None:
                    json = {"workspaceId": self._workspace_id}
                elif not json.get("workspaceId"):
                    json = {**json, "workspaceId": self._workspace_id}
            else:
                params = {"workspaceId": self._workspace_id}
        response = requests.request(
            method,
            url,
            headers={"x-api-key": self._api_key, "content-type": "application/json"},
            json=json,
            params=params,
            timeout=30,
        )
        # Canonical taxonomy (evaluations/monitor client precedent): auth and validation
        # failures raise the top-level typed errors, so `except agentx.AgentXAuthError`
        # works whichever sub-client raised.
        if response.status_code == 401:
            raise AgentXAuthError("Invalid or missing API key", status_code=401)
        if response.status_code == 422:
            raise AgentXValidationError(response.text, status_code=422)
        if response.status_code >= 400:
            raise AgentXScorerGroupsError(
                f"HTTP {response.status_code}: {response.text}", status_code=response.status_code
            )
        # DELETE (and any other empty 2xx) has no body - .json() on it raises.
        return response.json() if response.text else {}

    def list(self) -> List[ScorerGroup]:
        return [ScorerGroup(g) for g in self._request("GET", self._base).get("scorerGroups", [])]

    def get(self, group_id: str) -> ScorerGroup:
        return ScorerGroup(self._request("GET", f"{self._base}/{group_id}")["scorerGroup"])

    def create(
        self,
        name: str,
        members: List[Dict[str, Any]],
        description: Optional[str] = None,
        online: Optional[Dict[str, Any]] = None,
    ) -> ScorerGroup:
        """``members``: [{"kind": "judge"|"pattern"|"custom", "refId": ..., "weight": 1, "gate": False}].
        ``online``: {"enabled": True, "sampleRate": 0.1, "alertThreshold": 5, "severity": "medium"}.
        Add ``"scope": "session", "idleSeconds": 120`` to score whole multi-turn sessions once
        idle, instead of each sampled trace. Pass ``online=None`` (the default) for a group
        that only grades offline dataset runs."""
        payload: Dict[str, Any] = {"name": name, "members": members}
        if description is not None:
            payload["description"] = description
        if online is not None:
            payload["online"] = online
        return ScorerGroup(self._request("POST", self._base, json=payload)["scorerGroup"])

    def update(self, group_id: str, **fields: Any) -> ScorerGroup:
        """Sparse update - pass any of name/description/members/online (online=None detaches
        live scoring)."""
        return ScorerGroup(self._request("PUT", f"{self._base}/{group_id}", json=fields)["scorerGroup"])

    def delete(self, group_id: str) -> None:
        self._request("DELETE", f"{self._base}/{group_id}")

    def ratings(self, group_id: str, window: str = "7d") -> Dict[str, Any]:
        """Live score history for a group - ``{"window", "points": [{ts, averageRating, count}]}``,
        the same shape online-evaluator ratings use. ``window``: "24h" | "7d" | "30d"."""
        return self._request("GET", f"{self._base}/{group_id}/ratings?window={window}")
