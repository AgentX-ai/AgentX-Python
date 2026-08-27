from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agentx.monitor.client import MonitorClient

logger = logging.getLogger(__name__)


class MonitorRule(dict):
    """Wire object for one automation rule (dict subclass so unknown fields round-trip)."""

    @property
    def id(self) -> str:
        return self["_id"]

    @property
    def enabled(self) -> bool:
        return bool(self.get("enabled"))


class MonitorRulesClient:
    """Surfaced as ``client.monitor.rules``: automation rules, evaluated on every ingested root
    trace. A RULE routes traffic somewhere (it never scores): ``action`` is one of

    - ``"review"`` - sample matching traces into the human-review queue (the stream that feeds
      judge calibration and tuning),
    - ``"dataset"`` - append matching traces as cases on a dataset (``action_config
      {"datasetId": ...}``),
    - ``"webhook"`` - POST the matching trace to your URL (``action_config {"url": ...}``).

    ``filter`` narrows what matches: ``{"model": ..., "status": "error"|"any", "contains": ...,
    "scopeMode": "all"|"selected", "agentIds": [...]}``; ``sample_rate`` (0-1, default 1)
    down-samples the matches.
    """

    def __init__(self, client: "MonitorClient"):
        self._client = client

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._client._request(method, path, base=self._client._api_root(), **kwargs)

    def list(self) -> List[MonitorRule]:
        data = self._request("GET", "/agent-monitoring/rules")
        return [MonitorRule(r) for r in data.get("rules", [])]

    def create(
        self,
        name: str,
        action: str,
        *,
        filter: Optional[Dict[str, Any]] = None,
        sample_rate: Optional[float] = None,
        action_config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> MonitorRule:
        payload: Dict[str, Any] = {"name": name, "action": action, "enabled": enabled}
        if filter is not None:
            payload["filter"] = filter
        if sample_rate is not None:
            payload["sampleRate"] = sample_rate
        if action_config is not None:
            payload["actionConfig"] = action_config
        data = self._request("POST", "/agent-monitoring/rules", json=payload)
        return MonitorRule(data.get("rule", data))

    def update(self, rule_id: str, **fields: Any) -> MonitorRule:
        """Sparse update. snake_case keys are mapped to the wire (``sample_rate`` ->
        ``sampleRate``, ``action_config`` -> ``actionConfig``)."""
        aliases = {"sample_rate": "sampleRate", "action_config": "actionConfig"}
        payload = {aliases.get(k, k): v for k, v in fields.items()}
        data = self._request("PUT", f"/agent-monitoring/rules/{rule_id}", json=payload)
        return MonitorRule(data.get("rule", data))

    def delete(self, rule_id: str) -> None:
        self._request("DELETE", f"/agent-monitoring/rules/{rule_id}")
