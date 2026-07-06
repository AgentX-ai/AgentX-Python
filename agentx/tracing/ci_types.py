"""Dataclasses for CI/CD evaluation run responses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class CITestCase:
    index: int
    query: str | None = None  # None when ci.exposeTestInputs is false


@dataclass
class CIRun:
    run_id: str
    dataset_id: str
    total_questions: int
    test_cases: list[CITestCase]
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
    scores: list[CIQuestionScore] = field(default_factory=list)
    violations: list[ThresholdViolation] = field(default_factory=list)
    git_context: dict | None = None
    finalized_at: str | None = None


@dataclass
class CIRunStatus:
    run_id: str
    status: Literal["in_progress", "completed", "failed"]
    gate: Literal["pass", "fail"] | None
    results_submitted: int
    total_questions: int
    created_at: str
    expires_at: str
    finalized_at: str | None = None
    git_context: dict | None = None
