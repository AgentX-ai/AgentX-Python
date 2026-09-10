"""Dataclasses for CI/CD evaluation run responses.

Annotations deliberately use typing.Optional/List (not PEP 604/585 syntax):
``from __future__ import annotations`` only defers evaluation, and
``typing.get_type_hints`` on these classes still has to resolve the strings
at runtime, which fails for ``str | None`` / ``list[...]`` on Python 3.9.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class CITestCase:
    index: int
    query: Optional[str] = None  # None when ci.exposeTestInputs is false


@dataclass
class CIRun:
    run_id: str
    dataset_id: str
    total_questions: int
    test_cases: List[CITestCase]
    expires_at: str


@dataclass
class CIQuestionScore:
    question_index: int
    rating: int
    justification: str
    passed: bool
    gate_fired: bool = False
    input: Any = None
    output: Any = None


@dataclass
class ThresholdViolation:
    question_index: int
    metric: str
    threshold: float
    actual: float
    question_text: str


@dataclass
class CIRunResult:
    run_id: str
    gate: Literal["pass", "fail"]
    pass_rate: float
    total_questions: int
    passed_questions: int
    scores: List[CIQuestionScore] = field(default_factory=list)
    violations: List[ThresholdViolation] = field(default_factory=list)
    git_context: Optional[Dict[str, Any]] = None
    finalized_at: Optional[str] = None


@dataclass
class CIRunStatus:
    run_id: str
    status: Literal["in_progress", "completed", "failed"]
    gate: Optional[Literal["pass", "fail"]]
    results_submitted: int
    total_questions: int
    created_at: str
    expires_at: str
    finalized_at: Optional[str] = None
    git_context: Optional[Dict[str, Any]] = None
