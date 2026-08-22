from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import requests

from agentx.util import api_base, get_headers

logger = logging.getLogger(__name__)


class AgentXScorersError(Exception):
    pass


class ScorersClient:
    """Surfaced as ``client.monitor.scorers``: administer the Scorers catalog as code.

    Covers what the dashboard's Scorers page does:

    - **Template scorers** (the shipped zero-LLM detectors): ``templates()`` lists them with
      enablement, ``enable()``/``disable()`` flip them. Everything is opt-in - a fresh project
      runs nothing until a scorer is enabled.
    - **Code scorers**: ``create_code()`` deploys your own Python/JavaScript
      ``handler(input, output, expected, metadata, trace)`` run in-engine per sampled trace.
    - **External scorers**: ``create_external()`` registers your HTTP endpoint (contract v2:
      the full trace record plus its span subtree).
    - Shared CRUD: ``list()``, ``update()``, ``delete()``, and ``dry_run()`` (executes a code
      scorer, or POSTs the sample payload to an external URL, without persisting anything).

    The engine resource name for code/external scorers remains ``custom-evaluators`` on the
    wire.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key

    def _request(self, method: str, path: str, json: Any = None, params: Any = None) -> Any:
        resp = requests.request(
            method,
            f"{api_base()}/agent-monitoring{path}",
            headers={**get_headers(self._api_key), "Content-Type": "application/json"},
            json=json,
            params=params,
            timeout=20,
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.reason)
            except ValueError:
                detail = resp.reason
            raise AgentXScorersError(f"Scorer request failed ({resp.status_code}): {detail}")
        return resp.json() if resp.text else {}

    # ------------------------------------------------------------------
    # Template scorers (built-in, opt-in)
    # ------------------------------------------------------------------

    def templates(self) -> List[Dict[str, Any]]:
        """The shipped template scorers with their keys, rules, and ``enabled`` state."""
        patterns = self._request("GET", "/patterns").get("patterns", [])
        return [p for p in patterns if p.get("source") == "builtIn"]

    def _enabled_template_keys(self) -> List[str]:
        return [p["key"] for p in self.templates() if p.get("enabled")]

    def enable(self, keys: Sequence[str]) -> List[str]:
        """Enable template scorers by key (e.g. ``["pii-in-response"]``), preserving what is
        already on. Returns the resulting enabled-key list."""
        merged = sorted(set(self._enabled_template_keys()) | set(keys))
        self._request("PUT", "/settings/monitoring-defaults", json={"enabledBuiltinPatterns": merged})
        return merged

    def disable(self, keys: Sequence[str]) -> List[str]:
        """Disable template scorers by key, preserving the rest. Returns the resulting list."""
        merged = sorted(set(self._enabled_template_keys()) - set(keys))
        self._request("PUT", "/settings/monitoring-defaults", json={"enabledBuiltinPatterns": merged})
        return merged

    # ------------------------------------------------------------------
    # Code / external scorers
    # ------------------------------------------------------------------

    def list(self) -> List[Dict[str, Any]]:
        """All code and external scorers (wire kind: ``"code"`` / ``"external"``)."""
        return self._request("GET", "/custom-evaluators").get("evaluators", [])

    def create_code(
        self,
        name: str,
        script: str,
        *,
        language: str = "python",
        alert_below: float = 0.5,
        sample_rate: float = 0.1,
        severity: str = "medium",
        enabled: bool = True,
        scope_mode: str = "all",
        agent_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Deploy a code scorer. ``script`` defines ``handler(input, output, expected,
        metadata, trace)`` returning a 0..1 score, ``{"score", "name"?, "metadata"?}``, or
        ``None`` to skip; a score below ``alert_below`` raises a signal."""
        if language not in ("python", "javascript"):
            raise AgentXScorersError('language must be "python" or "javascript"')
        return self._request("POST", "/custom-evaluators", json={
            "name": name,
            "kind": "code",
            "language": language,
            "script": script,
            "alertBelow": alert_below,
            "sampleRate": sample_rate,
            "severity": severity,
            "enabled": enabled,
            "scopeMode": scope_mode,
            "agentIds": list(agent_ids) if agent_ids else [],
        })["evaluator"]

    def create_external(
        self,
        name: str,
        url: str,
        *,
        sample_rate: float = 0.1,
        severity: str = "medium",
        enabled: bool = True,
        invert_match: bool = False,
        scope_mode: str = "all",
        agent_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Register an external scorer endpoint (POSTed the v2 payload per sampled trace)."""
        return self._request("POST", "/custom-evaluators", json={
            "name": name,
            "url": url,
            "sampleRate": sample_rate,
            "severity": severity,
            "enabled": enabled,
            "invertMatch": invert_match,
            "scopeMode": scope_mode,
            "agentIds": list(agent_ids) if agent_ids else [],
        })["evaluator"]

    def update(self, scorer_id: str, **fields: Any) -> Dict[str, Any]:
        """Update a code/external scorer. snake_case kwargs are converted (``alert_below`` ->
        ``alertBelow`` etc.); kind is immutable."""
        wire = {_SNAKE_TO_WIRE.get(k, k): v for k, v in fields.items()}
        return self._request("PUT", f"/custom-evaluators/{scorer_id}", json=wire)["evaluator"]

    def delete(self, scorer_id: str) -> None:
        self._request("DELETE", f"/custom-evaluators/{scorer_id}")

    def events(self, scorer_id: str, window: str = "24h") -> List[Dict[str, Any]]:
        """The scorer's per-check history (score, matched, justification, trace ids)."""
        return self._request("GET", f"/custom-evaluators/{scorer_id}/events", params={"window": window}).get("events", [])

    def dry_run(self, **payload: Any) -> Dict[str, Any]:
        """Execute a scorer against the built-in sample without persisting: pass either
        ``url=...`` (external) or ``kind="code", language=..., script=...`` (code)."""
        wire = {_SNAKE_TO_WIRE.get(k, k): v for k, v in payload.items()}
        return self._request("POST", "/custom-evaluators/dry-run", json=wire)


_SNAKE_TO_WIRE = {
    "alert_below": "alertBelow",
    "sample_rate": "sampleRate",
    "scope_mode": "scopeMode",
    "agent_ids": "agentIds",
    "invert_match": "invertMatch",
}
