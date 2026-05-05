from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observable trace
# ---------------------------------------------------------------------------

class TraceEvent(BaseModel):
    type: str
    name: Optional[str] = None
    summary: Optional[str] = None
    latency_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = "ignore"


class ObservableTrace(BaseModel):
    events: List[TraceEvent] = Field(default_factory=list)

    class Config:
        extra = "ignore"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestCase(BaseModel):
    query: str
    expected_results: Optional[str] = None
    expected_capabilities: Optional[List[str]] = None
    expected_knowledge_base: Optional[List[str]] = None
    expected_delegations: Optional[List[str]] = None

    class Config:
        extra = "ignore"


class DatasetQuestion(BaseModel):
    main_question: TestCase
    follow_up_questions: List[TestCase] = Field(default_factory=list)


class Dataset(BaseModel):
    id: str = Field(alias="_id")
    name: str
    description: Optional[str] = None
    number_of_requests: int = Field(default=1, alias="numberOfRequests")
    acceptance_criteria: Optional[str] = Field(default=None, alias="acceptanceCriteria")
    rejection_criteria: Optional[str] = Field(default=None, alias="rejectionCriteria")
    evaluation_criteria: Optional[str] = Field(default=None, alias="evaluationCriteria")
    questions: List[DatasetQuestion] = Field(default_factory=list)
    status: str = "published"
    version_id: Optional[str] = Field(default=None, alias="versionId")

    class Config:
        populate_by_name = True
        extra = "ignore"


# ---------------------------------------------------------------------------
# Evaluation subject
# ---------------------------------------------------------------------------

FrameworkKind = Literal[
    "raw_python", "openai", "anthropic", "langchain", "llamaindex",
    "crewai", "autogen", "n8n", "flowise", "other"
]

RuntimeKind = Literal["local", "ci", "customer_hosted", "low_code"]


class EvaluationSubject(BaseModel):
    kind: Literal["custom_agent", "agentx_agent", "agentx_team"] = "custom_agent"
    display_name: Optional[str] = Field(default=None, alias="displayName")
    framework: Optional[FrameworkKind] = None
    framework_version: Optional[str] = Field(default=None, alias="frameworkVersion")
    runtime: Optional[RuntimeKind] = "local"
    metadata: Optional[Dict[str, Union[str, int, bool]]] = None

    class Config:
        populate_by_name = True
        extra = "ignore"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

class ServerLimits(BaseModel):
    max_batch_size: int = Field(default=10, alias="maxBatchSize")
    max_trace_bytes_per_result: int = Field(default=20000, alias="maxTraceBytesPerResult")
    max_metadata_bytes_per_result: int = Field(default=4000, alias="maxMetadataBytesPerResult")

    class Config:
        populate_by_name = True
        extra = "ignore"


class EvaluationRun(BaseModel):
    run_id: str = Field(alias="runId")
    dataset_id: str = Field(alias="datasetId")
    dataset_version_id: Optional[str] = Field(default=None, alias="datasetVersionId")
    status: str = "in_progress"
    limits: ServerLimits = Field(default_factory=ServerLimits)

    class Config:
        populate_by_name = True
        extra = "ignore"


# ---------------------------------------------------------------------------
# Evaluation case (one item from the dataset, passed to the user's callable)
# ---------------------------------------------------------------------------

class EvaluationCase(BaseModel):
    case_id: str
    question_index: int
    run_number: int
    query: str
    expected_results: Optional[str] = None
    expected_capabilities: Optional[List[str]] = None
    expected_knowledge_base: Optional[List[str]] = None
    expected_delegations: Optional[List[str]] = None

    class Config:
        extra = "ignore"


# ---------------------------------------------------------------------------
# Result produced by the user's callable and normalised by the SDK
# ---------------------------------------------------------------------------

class ResultError(BaseModel):
    type: str
    message: str
    retryable: bool = False

    class Config:
        extra = "ignore"


class ResultTimings(BaseModel):
    latency_ms: Optional[int] = Field(default=None, alias="latencyMs")
    input_tokens: Optional[int] = Field(default=None, alias="inputTokens")
    output_tokens: Optional[int] = Field(default=None, alias="outputTokens")

    class Config:
        populate_by_name = True
        extra = "ignore"


class EvaluationResult(BaseModel):
    case_id: str
    question_index: int
    run_number: int
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    observable_trace: Optional[ObservableTrace] = Field(default=None, alias="observableTrace")
    error: Optional[ResultError] = None
    timings: Optional[ResultTimings] = None
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = Field(default=None, alias="idempotencyKey")

    class Config:
        populate_by_name = True
        extra = "ignore"


# ---------------------------------------------------------------------------
# Scored result (returned by API after scoring)
# ---------------------------------------------------------------------------

class ScoredResult(BaseModel):
    idempotency_key: str = Field(alias="idempotencyKey")
    rating: Optional[float] = None
    justification: Optional[str] = None

    class Config:
        populate_by_name = True
        extra = "ignore"


# ---------------------------------------------------------------------------
# Batch append response
# ---------------------------------------------------------------------------

class BatchAppendResponse(BaseModel):
    run_id: str = Field(alias="runId")
    batch_id: str = Field(alias="batchId")
    accepted: int = 0
    duplicates: int = 0
    failed_validation: int = Field(default=0, alias="failedValidation")
    status: str = "in_progress"
    scored_results: List[ScoredResult] = Field(default_factory=list, alias="scoredResults")

    class Config:
        populate_by_name = True
        extra = "ignore"


# ---------------------------------------------------------------------------
# Analysis / report
# ---------------------------------------------------------------------------

class ReportStatistics(BaseModel):
    number_of_runs: int = Field(default=0, alias="numberOfRuns")
    average_rating: float = Field(default=0.0, alias="averageRating")
    min_rating: float = Field(default=0.0, alias="minRating")
    max_rating: float = Field(default=0.0, alias="maxRating")

    class Config:
        populate_by_name = True
        extra = "ignore"


class ReportInstructionAdherence(BaseModel):
    score: Optional[float] = None
    analysis: Optional[str] = None
    deviations: List[str] = Field(default_factory=list)
    rating: Optional[str] = None

    class Config:
        extra = "ignore"


class ReportResponsePatterns(BaseModel):
    similarities: List[str] = Field(default_factory=list)
    differences: List[str] = Field(default_factory=list)
    outliers: List[str] = Field(default_factory=list)
    rating: Optional[str] = None

    class Config:
        extra = "ignore"


class ReportReasoningAnalysis(BaseModel):
    cot_quality: Optional[str] = Field(default=None, alias="cotQuality")
    reasoning_patterns: List[str] = Field(default_factory=list, alias="reasoningPatterns")
    reasoning_gaps: List[str] = Field(default_factory=list, alias="reasoningGaps")
    rating: Optional[str] = None

    class Config:
        populate_by_name = True
        extra = "ignore"


class ReportToolUsageAnalysis(BaseModel):
    effectiveness: Optional[str] = None
    patterns: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    rating: Optional[str] = None

    class Config:
        extra = "ignore"


class ReportRecommendation(BaseModel):
    category: Optional[str] = None
    priority: Optional[str] = None
    recommendation: Optional[str] = None
    reasoning: Optional[str] = None

    class Config:
        extra = "ignore"


class Report(BaseModel):
    run_id: str = Field(alias="runId")
    dataset_id: str = Field(alias="datasetId")
    status: str = "completed"
    statistics: Optional[ReportStatistics] = None
    summary: Optional[str] = None
    consistency_score: Optional[float] = Field(default=None, alias="consistencyScore")
    instruction_adherence: Optional[ReportInstructionAdherence] = Field(default=None, alias="instructionAdherence")
    response_patterns: Optional[ReportResponsePatterns] = Field(default=None, alias="responsePatterns")
    reasoning_analysis: Optional[ReportReasoningAnalysis] = Field(default=None, alias="reasoningAnalysis")
    tool_usage_analysis: Optional[ReportToolUsageAnalysis] = Field(default=None, alias="toolUsageAnalysis")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    overall_rating: Optional[str] = Field(default=None, alias="overallRating")
    recommendations: List[ReportRecommendation] = Field(default_factory=list)
    low_scoring_cases: List[Dict[str, Any]] = Field(default_factory=list, alias="lowScoringCases")
    dashboard_url: Optional[str] = Field(default=None, alias="dashboardUrl")

    class Config:
        populate_by_name = True
        extra = "ignore"
