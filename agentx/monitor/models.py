from __future__ import annotations

from typing import List, Optional
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
