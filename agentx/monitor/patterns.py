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

    # snake_case -> wire camelCase, same posture as rules.update: the engine reads only the
    # camelCase key and silently keeps the stored value for anything it does not recognize -
    # update(sample_rate=0.05) used to 200 with the rate unchanged.
    _UPDATE_ALIASES = {
        "sample_rate": "sampleRate",
        "scope_mode": "scopeMode",
        "agent_ids": "agentIds",
        "detector_kind": "detectorKind",
        "include_terms": "includeTerms",
        "exclude_terms": "excludeTerms",
        "regex": "regex",
        "semantic_prompt": "semanticPrompt",
        "match_mode": "matchMode",
        "match_target": "matchTarget",
    }

    # The engine's PUT rebuilds the pattern's WHOLE conditions array (legacyPayloadToConditions:
    # "a full replace, not a sparse patch") exactly when the body carries one of these - its own
    # sentConditionFields set, minus "conditions". Bodies without any of them keep the stored
    # conditions untouched.
    _TRIGGER_FIELDS = ("includeTerms", "regex", "semanticPrompt")

    # Sent alone, these look sparse but cannot land: without a trigger field the engine never
    # rebuilds conditions (the values are silently ignored). For excludeTerms/matchMode there is
    # also nothing to merge them over client-side - the wire GET reports display-only
    # placeholders (includeTerms/excludeTerms always [], matchMode always "any"; conditions is
    # the only truth). matchTarget IS reported faithfully, but the engine only reads it during a
    # rebuild, so alone it is the same silent no-op.
    _UNMERGEABLE_ALONE = ("excludeTerms", "matchMode", "matchTarget")

    def update(self, pattern_id: str, **fields: Any) -> MonitorPattern:
        """Update a pattern and return the updated :class:`MonitorPattern`. Accepts snake_case
        kwargs (``sample_rate=0.05``) or the wire's camelCase; an unrecognized snake_case key
        raises instead of silently changing nothing.

        The real contract on self-host: the stored truth is the pattern's ``conditions`` array,
        and the flat fields the wire GET returns are display-only placeholders (``includeTerms``/
        ``excludeTerms`` always ``[]``, ``matchMode`` always ``"any"``, ``regex``/
        ``semanticPrompt`` omitted). The engine's PUT rebuilds the WHOLE conditions array
        whenever the body carries ``include_terms``, ``regex``, or ``semantic_prompt`` (or an
        explicit ``conditions`` list, which wins outright). Consequences:

        - ``update(pid, regex=...)`` (or include_terms/semantic_prompt) is a full detector
          rewrite; the detector kind follows the trigger field actually sent (``regex`` ->
          ``"regex"``, ``semantic_prompt`` -> ``"semantic"``, ``include_terms`` ->
          ``"contains"``), so cross-kind updates work, and this client back-fills only
          ``matchTarget`` from the stored pattern so the rebuild keeps its target. It never
          back-fills ``includeTerms``/``excludeTerms``/``matchMode`` - the GET values are
          placeholders, and copying them in would destroy real conditions.
        - ``exclude_terms=``, ``match_mode=``, or ``match_target=`` alone raises ValueError:
          the engine silently ignores them without a rebuild. Pass ``conditions=[...]`` (built
          from ``get(pattern_id).conditions``) instead, or pass them alongside the
          include_terms/regex/semantic_prompt they should be rebuilt with.
        - Everything else stays a sparse metadata edit that leaves the stored conditions
          untouched."""
        payload: Dict[str, Any] = {}
        for key, value in fields.items():
            wire_key = self._UPDATE_ALIASES.get(key, key)
            if "_" in wire_key:
                raise ValueError(
                    f"Unknown pattern field {key!r} - the engine reads camelCase keys and would "
                    "silently ignore this (see MonitorPattern for the field names)."
                )
            payload[wire_key] = value
        sends_conditions = "conditions" in payload
        triggered = any(k in payload for k in self._TRIGGER_FIELDS)
        if not sends_conditions and not triggered:
            offending = [k for k in self._UNMERGEABLE_ALONE if k in payload]
            if offending:
                raise ValueError(
                    f"{' and '.join(offending)} cannot be updated on their own: the engine only "
                    "rebuilds a pattern's conditions when includeTerms/regex/semanticPrompt is "
                    "sent (alone they are silently ignored), and the wire GET returns display-"
                    "only placeholders (includeTerms/excludeTerms always [], matchMode always "
                    "'any'), so there is no stored value to merge them over. Pass "
                    "conditions=[...] built from get(pattern_id).conditions instead, or send "
                    "them alongside the trigger field they should be rebuilt with."
                )
        if triggered and not sends_conditions:
            stored = self.get(pattern_id)
            # The rebuild's detector kind follows the trigger field actually sent - back-filling
            # the STORED kind 400s a kind change (regex= on a "contains" pattern) and silently
            # corrupts the inverse (include_terms= on a regex pattern would write phrase
            # conditions while keeping detectorKind "regex").
            if "regex" in payload:
                inferred_kind = "regex"
            elif "semanticPrompt" in payload:
                inferred_kind = "semantic"
            else:
                inferred_kind = "contains"
            payload.setdefault("detectorKind", inferred_kind)
            # matchTarget is the one field the wire reports faithfully - it rides along so the
            # server-side full-replace rebuild keeps the stored target. NEVER includeTerms/
            # excludeTerms/matchMode - see _UNMERGEABLE_ALONE's comment.
            payload.setdefault("matchTarget", stored.match_target)
        data = self._client._request(
            "PUT",
            f"/agent-monitoring/patterns/{pattern_id}",
            base=self._client._api_root(),
            json=payload,
            # An idempotent merge server-side (same payload, same result) - but the read-
            # merge-write above is not atomic, so keep the single-shot posture.
            retry=False,
        )
        return MonitorPattern(**data["pattern"])

    def get(self, pattern_id: str) -> MonitorPattern:
        return self._client.get_pattern(pattern_id)

    def list(self) -> List[MonitorPattern]:
        return self._client.list_patterns()
