from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MonitorPattern(BaseModel):
    """A detection rule checked against production traces. Built via
    ``client.monitor.patterns.builder(...).publish()`` and referenced by id as a
    ``pattern_ids`` entry in ``tracer.trace(..., monitor=True, pattern_ids=[...])``.

    A "failure" pattern (the default) raises a signal to triage; a "proper" pattern logs a
    healthy tally instead. Only one of ``include_terms``/``regex``/``semantic_prompt`` is
    meaningful at a time, selected by ``detector_kind``.
    """

    id: str = Field(alias="_id")
    key: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    detector_kind: str = Field(default="contains", alias="detectorKind")
    match_target: List[str] = Field(default_factory=lambda: ["response"], alias="matchTarget")
    match_mode: str = Field(default="any", alias="matchMode")
    include_terms: List[str] = Field(default_factory=list, alias="includeTerms")
    exclude_terms: List[str] = Field(default_factory=list, alias="excludeTerms")
    regex: Optional[str] = None
    semantic_prompt: Optional[str] = Field(default=None, alias="semanticPrompt")
    severity: str = "medium"
    polarity: str = "failure"
    enabled: bool = True
    sample_rate: float = Field(default=1.0, alias="sampleRate")
    scope_mode: str = Field(default="all", alias="scopeMode")
    agent_ids: List[str] = Field(default_factory=list, alias="agentIds")

    class Config:
        extra = "ignore"


class SignalOccurrence(BaseModel):
    """One hit behind a signal, capped at the server's most recent N per signal."""

    agent_id: Optional[Dict[str, Any]] = Field(default=None, alias="agentId")
    conversation_id: Optional[str] = Field(default=None, alias="conversationId")
    message_id: Optional[str] = Field(default=None, alias="messageId")
    trace_id: Optional[str] = Field(default=None, alias="traceId")
    seen_at: Optional[str] = Field(default=None, alias="seenAt")

    class Config:
        populate_by_name = True
        extra = "ignore"


class MonitorProfile(BaseModel):
    """A single agent's Monitor coverage/detection settings: what gets sampled, how long it's
    kept, whether info (clean-run) signals are logged, per-check threshold overrides (e.g.
    ``threshold_overrides["latencyMs"]`` for the built-in "Latency regression" pattern), and the
    approval policy gating autotune actions. See ``client.monitor.profile.get()/update()``.
    ``None`` from ``get()`` means this agent has never been configured and is running on
    platform defaults (e.g. the built-in latency threshold defaults to 20000ms).
    """

    id: Optional[str] = Field(default=None, alias="_id")
    workspace_id: Optional[str] = Field(default=None, alias="workspaceId")
    agent_id: Optional[Any] = Field(default=None, alias="agentId")
    enabled: bool = True
    failure_detection_enabled: bool = Field(default=True, alias="failureDetectionEnabled")
    info_detection_enabled: bool = Field(default=True, alias="infoDetectionEnabled")
    coverage_mode: str = Field(default="all", alias="coverageMode")
    sample_rate: float = Field(default=0.1, alias="sampleRate")
    channels: List[str] = Field(default_factory=list)
    threshold_overrides: Optional[Dict[str, Any]] = Field(default=None, alias="thresholdOverrides")
    retention_days: int = Field(default=30, alias="retentionDays")
    redaction_mode: str = Field(default="standard", alias="redactionMode")
    approval_policy: Optional[Dict[str, str]] = Field(default=None, alias="approvalPolicy")
    created_at: Optional[str] = Field(default=None, alias="createdAt")
    updated_at: Optional[str] = Field(default=None, alias="updatedAt")

    class Config:
        populate_by_name = True
        extra = "ignore"


class MonitorSignal(BaseModel):
    """An alert/finding produced when a trace matched a pattern (or, for a "proper" pattern,
    a healthy tally). Read-only from the SDK, see ``client.monitor.signals.list()/get()``, a
    signal is the system's output from checking traces against patterns, not something an SDK
    caller constructs."""

    id: str = Field(alias="_id")
    workspace_id: Optional[str] = Field(default=None, alias="workspaceId")
    # Populated to {"_id", "name", "avatar"} when set, since the server populates this field.
    agent_id: Optional[Dict[str, Any]] = Field(default=None, alias="agentId")
    conversation_id: Optional[str] = Field(default=None, alias="conversationId")
    message_id: Optional[str] = Field(default=None, alias="messageId")
    type: str
    severity: str = "medium"
    polarity: str = "failure"
    status: str = "open"
    score: Optional[float] = None
    threshold: Optional[float] = None
    summary: str
    pattern_key: str = Field(alias="patternKey")
    evidence: Optional[Dict[str, Any]] = None
    root_cause: Optional[str] = Field(default=None, alias="rootCause")
    recommended_actions: List[str] = Field(default_factory=list, alias="recommendedActions")
    review_status: Optional[str] = Field(default=None, alias="reviewStatus")
    first_seen_at: Optional[str] = Field(default=None, alias="firstSeenAt")
    last_seen_at: Optional[str] = Field(default=None, alias="lastSeenAt")
    occurrence_count: int = Field(default=1, alias="occurrenceCount")
    occurrences: List[SignalOccurrence] = Field(default_factory=list)

    class Config:
        populate_by_name = True
        extra = "ignore"
