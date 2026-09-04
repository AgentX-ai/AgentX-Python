from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from agentx.util import api_base, get_headers


class AgentXImprovementGroupsError(Exception):
    pass


class ImprovementGroupsClient:
    """Surfaced as ``client.monitor.improvement_groups``: the auto-improve loop's accumulator.

    Batch lifecycle: one COLLECTING group at a time. Every Confirm verdict in signal review
    automatically lands the confirmed failure there - accumulation is free, declining is
    choosing Ignore. ``generate_report`` SPENDS the batch: one LLM pass clusters the confirmed
    failures into issues with recommendations, the group is sealed onto that report (keeping
    exactly its source cases), and the pending accumulator is thereby cleared - the next
    Confirm starts a fresh batch, and the next generate makes a new report from it. The report's id is the
    hand-off: paste it into the AgentX-Eval-Skill ``auto-improve`` skill, which fetches the
    report (``get_report``) and triages the fixes against your agent's actual source code.

    Evidence here is exclusively ONLINE - production verdicts a human confirmed - never
    offline dataset runs. Self-host only.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._api_key = api_key
        self._base_url = (base_url or api_base()).rstrip("/")

    def _request(self, method: str, path: str, json: Any = None, timeout: int = 120) -> Any:
        resp = requests.request(
            method,
            f"{self._base_url}/agent-monitoring{path}",
            headers={**get_headers(self._api_key), "Content-Type": "application/json"},
            json=json,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.reason)
            except ValueError:
                detail = resp.reason
            raise AgentXImprovementGroupsError(f"Improvement group request failed ({resp.status_code}): {detail}")
        return resp.json() if resp.text else {}

    def list(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/improvement-groups").get("improvementGroups", [])

    def get(self, group_id: str) -> Dict[str, Any]:
        """The group with its members - each a confirmed failure's evidence snapshot."""
        return self._request("GET", f"/improvement-groups/{group_id}")["improvementGroup"]

    def remove_member(self, group_id: str, member_id: str) -> None:
        """Prune a member before spending the group (a confirm that turned out uninteresting)."""
        self._request("DELETE", f"/improvement-groups/{group_id}/members/{member_id}")

    def generate_report(self, group_id: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Spend the group: one real LLM call clustering the confirmed failures into issues
        with recommendations. Returns the report; its ``_id`` is what the auto-improve skill
        takes. Explicit and billed - never called implicitly."""
        payload: Dict[str, Any] = {}
        if model is not None:
            payload["model"] = model
        return self._request("POST", f"/improvement-groups/{group_id}/report", json=payload, timeout=300)["report"]

    def list_reports(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/improvement-reports").get("improvementReports", [])

    def get_report(self, report_id: str) -> Dict[str, Any]:
        """Fetch a report by the id the dashboard (or generate_report) handed out - the exact
        call the auto-improve skill makes."""
        return self._request("GET", f"/improvement-reports/{report_id}")["report"]
