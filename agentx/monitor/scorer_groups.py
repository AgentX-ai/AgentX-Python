"""Scorer groups (self-host): scorers of any kind - LLM judges, patterns, custom code/external
scorers - composed into ONE 0-10 score via per-member weights and optional must-pass gates.
Members are references: ``{"kind": "judge" | "pattern" | "custom", "refId": ..., "weight": ...,
"gate": ...}``. Grade a dataset run with a group by passing its id as ``scorer_group_id`` to
``client.evaluations.run(...)``; give it an ``online`` profile to score sampled live traffic and
raise Signals below the alert threshold."""

from typing import Any, Dict, List, Optional

import requests

from agentx.exceptions import AgentXError, AgentXAuthError, AgentXValidationError
from agentx.monitor._transport import request_with_retries


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

    def _request(
        self, method: str, url: str, json: Optional[Dict[str, Any]] = None, retry: bool = True
    ) -> Any:
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
        # retry=False for ANY non-idempotent write (creates, deletes) -
        # MonitorClient._request's posture, via the shared monitor transport.
        response = request_with_retries(
            method,
            url,
            retry=retry,
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
        # Server-side create: a timeout + transport retry would create the group twice.
        return ScorerGroup(
            self._request("POST", self._base, json=payload, retry=False)["scorerGroup"]
        )

    def update(self, group_id: str, **fields: Any) -> ScorerGroup:
        """Sparse update - pass any of name/description/members/online. ``online`` itself may be
        partial: ``online={"enabled": False}`` pauses live scoring, the engine merges the patch
        over the stored profile. ``online=None`` detaches live scoring entirely. A partial
        ``online=`` patch on a group with NO stored live profile is rejected by the engine
        (400) - send the full profile the first time (the shape ``create`` documents)."""
        # Same guard patterns.update/rules.update carry: the engine's schema strips keys it
        # does not recognize, so a snake_case key would 200 with the group unchanged.
        for key in fields:
            if "_" in key:
                first, *rest = key.split("_")
                camel = first + "".join(part.capitalize() for part in rest)
                raise ValueError(
                    f"Unknown scorer group field {key!r} - the engine reads camelCase keys and "
                    f"would silently ignore this; send {camel!r} instead."
                )
        return ScorerGroup(self._request("PUT", f"{self._base}/{group_id}", json=fields)["scorerGroup"])

    def delete(self, group_id: str) -> None:
        # retry=False: a lost response + transport retry would turn a successful delete
        # into a spurious 404.
        self._request("DELETE", f"{self._base}/{group_id}", retry=False)

    def ratings(self, group_id: str, window: str = "7d") -> Dict[str, Any]:
        """Live score history for a group - ``{"window", "points": [{ts, averageRating, count}]}``,
        the same shape online-evaluator ratings use. ``window``: "24h" | "7d" | "30d"."""
        return self._request("GET", f"{self._base}/{group_id}/ratings?window={window}")
