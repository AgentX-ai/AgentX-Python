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

    - ``"contains"`` (default): ``include_terms`` — a match if any (or all, with
      ``match_mode="all"``) phrase appears in the target text.
    - ``"regex"``: ``regex`` — a single regular expression.
    - ``"semantic"``: ``semantic_prompt`` — an LLM judges whether the response violates the
      described rubric.
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
        )

    def get(self, pattern_id: str) -> MonitorPattern:
        return self._client.get_pattern(pattern_id)

    def list(self) -> List[MonitorPattern]:
        return self._client.list_patterns()
