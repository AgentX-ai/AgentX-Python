"""Scorer groups (self-host): scorers of any kind - LLM judges, patterns, custom code/external
scorers - composed into ONE 0-10 score via per-member weights and optional must-pass gates.
Members are references: ``{"kind": "judge" | "pattern" | "custom", "refId": ..., "weight": ...,
"gate": ...}``. Grade a dataset run with a group by passing its id as ``scorer_group_id`` to
``client.evaluations.run(...)``; give it an ``online`` profile to score sampled live traffic and
raise Signals below the alert threshold."""

from typing import Any, Dict, List, Optional

import requests


class AgentXScorerGroupsError(Exception):
    pass


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
    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base = base_url.rstrip("/") + "/agent-monitoring/scorer-groups"

    def _request(self, method: str, url: str, json: Optional[Dict[str, Any]] = None) -> Any:
        response = requests.request(
            method,
            url,
            headers={"x-api-key": self._api_key, "content-type": "application/json"},
            json=json,
            timeout=30,
        )
        if response.status_code >= 400:
            raise AgentXScorerGroupsError(f"HTTP {response.status_code}: {response.text}")
        return response.json()

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
        ``online``: {"enabled": True, "sampleRate": 0.1, "alertThreshold": 5, "severity": "medium"}
        or None for offline-only."""
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
