from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agentx.monitor.models import MonitorPattern

if TYPE_CHECKING:
    from agentx.monitor.client import MonitorClient

logger = logging.getLogger(__name__)


class MonitorPatternBuilder:
    """Fluent builder for creating a custom pattern. ``detector_kind`` selects which of
    ``include_terms``/``regex``/``semantic_prompt`` is used:

    - ``"contains"`` (default): ``include_terms`` - a match if any (or all, with
      ``match_mode="all"``) phrase appears in the target text.
    - ``"regex"``: ``regex`` - a single regular expression.
    - ``"semantic"``: ``semantic_prompt`` - an LLM judges whether the response violates the
      described rubric.

    ``conditions`` (self-host) is the engine's full N-condition model - a list of condition
    dicts, each with its own detector kind and match settings - passed through verbatim to
    the create payload; when set, the engine honors it as the pattern's whole rule set and
    the flat fields above are only legacy display metadata.
    """

    def __init__(
        self,
        client: "MonitorClient",
        name: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        detector_kind: str = "contains",
        match_target: Optional[List[str]] = None,
        match_mode: str = "any",
        include_terms: Optional[List[str]] = None,
        exclude_terms: Optional[List[str]] = None,
        regex: Optional[str] = None,
        semantic_prompt: Optional[str] = None,
        severity: str = "medium",
        polarity: str = "failure",
        enabled: bool = True,
        sample_rate: float = 1.0,
        scope_mode: str = "all",
        agent_ids: Optional[List[str]] = None,
        conditions: Optional[List[dict]] = None,
    ):
        self._client = client
        self._payload: Dict[str, Any] = {
            "name": name,
            "description": description,
            "category": category,
            "detectorKind": detector_kind,
            "matchTarget": match_target or ["response"],
            "matchMode": match_mode,
            "includeTerms": include_terms or [],
            "excludeTerms": exclude_terms or [],
            "regex": regex,
            "semanticPrompt": semantic_prompt,
            "severity": severity,
            # A "failure" pattern (default) raises a signal to triage; a "proper" pattern logs
            # a healthy tally instead.
            "polarity": polarity,
            "enabled": enabled,
            "sampleRate": sample_rate,
            "scopeMode": scope_mode,
            "agentIds": agent_ids or [],
        }
        # Passed through verbatim - the engine honors body.conditions as the full
        # N-condition model (see the class docstring).
        if conditions is not None:
            self._payload["conditions"] = conditions

    def publish(self) -> MonitorPattern:
        logger.info("Publishing monitor pattern '%s'", self._payload["name"])
        return self._client.create_pattern(self._payload)


class MonitorPatternClient:
    """Thin wrapper surfaced as ``client.monitor.patterns``."""

    def __init__(self, client: "MonitorClient"):
        self._client = client

    def builder(
        self,
        name: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        detector_kind: str = "contains",
        match_target: Optional[List[str]] = None,
        match_mode: str = "any",
        include_terms: Optional[List[str]] = None,
        exclude_terms: Optional[List[str]] = None,
        regex: Optional[str] = None,
        semantic_prompt: Optional[str] = None,
        severity: str = "medium",
        polarity: str = "failure",
        enabled: bool = True,
        sample_rate: float = 1.0,
        scope_mode: str = "all",
        agent_ids: Optional[List[str]] = None,
        conditions: Optional[List[dict]] = None,
    ) -> MonitorPatternBuilder:
        return MonitorPatternBuilder(
            self._client,
            name=name,
            description=description,
            category=category,
            detector_kind=detector_kind,
            match_target=match_target,
            match_mode=match_mode,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
            regex=regex,
            semantic_prompt=semantic_prompt,
            severity=severity,
            polarity=polarity,
            enabled=enabled,
            sample_rate=sample_rate,
            scope_mode=scope_mode,
            agent_ids=agent_ids,
            conditions=conditions,
        )

    def delete(self, pattern_id: str) -> None:
        """Delete a pattern. Its historical signals remain as history."""
        # retry=False: a lost response + transport retry would turn a successful
        # delete into a spurious 404.
        self._client._request(
            "DELETE",
            f"/agent-monitoring/patterns/{pattern_id}",
            base=self._client._api_root(),
            retry=False,
        )

    def update(self, pattern_id: str, **fields: Any) -> MonitorPattern:
        """Update a pattern's fields in place (wire camelCase keys, passed through
        verbatim - e.g. ``enabled=False``, ``conditions=[...]``) and return the
        updated :class:`MonitorPattern`. retry=False: a non-idempotent server-side
        write must not be re-fired on a lost response."""
        data = self._client._request(
            "PUT",
            f"/agent-monitoring/patterns/{pattern_id}",
            base=self._client._api_root(),
            json=fields,
            retry=False,
        )
        return MonitorPattern(**data["pattern"])

    def get(self, pattern_id: str) -> MonitorPattern:
        return self._client.get_pattern(pattern_id)

    def list(self) -> List[MonitorPattern]:
        return self._client.list_patterns()
