from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator

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


class SmokeTestSettings(BaseModel):
    """Only meaningful on a question's main_question. See DatasetBuilder.add_case's
    smoke_test_count/smoke_test_guidance for how to set this."""

    enabled: bool = False
    count: int = 1
    guidance: Optional[str] = None

    class Config:
        populate_by_name = True
        extra = "ignore"


class TestCase(BaseModel):
    query: str
    expected_results: Optional[str] = Field(default=None, alias="expectedResults")
    expected_capabilities: Optional[List[str]] = Field(default=None, alias="expectedCapabilities")
    expected_knowledge_base: Optional[List[str]] = Field(default=None, alias="expectedKnowledgeBase")
    expected_delegations: Optional[List[str]] = Field(default=None, alias="expectedDelegations")
    judge_guideline: Optional[str] = Field(default=None, alias="judgeGuideline")
    smoke_test: Optional[SmokeTestSettings] = Field(default=None, alias="smokeTest")

    class Config:
        populate_by_name = True
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
    # Sovereignty & Portability — models selected to compare on this dataset.
    # Hoisted from the nested ``sovereigntyIndex`` object when enabled.
    sovereignty_models: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _extract_sovereignty_models(cls, data: Any) -> Any:
        if isinstance(data, dict):
            sov = data.get("sovereigntyIndex") or data.get("sovereignty_index") or {}
            if isinstance(sov, dict) and sov.get("enabled") and sov.get("models"):
                data = {**data, "sovereignty_models": list(sov.get("models") or [])}
        return data

    class Config:
        populate_by_name = True
        extra = "ignore"


class EvaluationSettings(BaseModel):
    """A standalone, reusable grading config — no dataset/questions attached.
    Created via ``client.evaluations.settings.builder(...).publish()`` and run
    against any dataset by passing its id as ``evaluation_settings_id`` to
    ``client.evaluations.run(...)``."""

    id: str = Field(alias="_id")
    name: str
    description: Optional[str] = None
    number_of_requests: int = Field(default=1, alias="numberOfRequests")
    acceptance_criteria: Optional[str] = Field(default=None, alias="acceptanceCriteria")
    rejection_criteria: Optional[str] = Field(default=None, alias="rejectionCriteria")
    evaluation_criteria: Optional[str] = Field(default=None, alias="evaluationCriteria")
    # LLM-as-judge overrides. None means "use the server default" (raw prompt template / OpenAI
    # gpt-5.5). See client.evaluations.settings.builder(judge_prompt=..., judge_model=...).
    judge_prompt: Optional[str] = Field(default=None, alias="judgePrompt")
    judge_model: Optional[str] = Field(default=None, alias="judgeModel")
    status: str = "published"
    # Sovereignty & Portability — models selected to compare when this config runs.
    # Hoisted from the nested ``sovereigntyIndex`` object when enabled.
    sovereignty_models: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _extract_sovereignty_models(cls, data: Any) -> Any:
        if isinstance(data, dict):
            sov = data.get("sovereigntyIndex") or data.get("sovereignty_index") or {}
            if isinstance(sov, dict) and sov.get("enabled") and sov.get("models"):
                data = {**data, "sovereignty_models": list(sov.get("models") or [])}
        return data

    class Config:
        populate_by_name = True
        extra = "ignore"


# ---------------------------------------------------------------------------
# Prompt registry — the external-agent analog to AgentX's native autotune. AgentX doesn't own
# your agent's code, so instead of branching/applying a config it becomes the prompt's source of
# truth (same shape as LangSmith's Prompt Hub / Langfuse's Prompt Management): pull a version at
# runtime with ``client.evaluations.prompts.get(name_or_id)``, use ``prompt.text`` as your agent's
# system prompt, and tag your eval runs so the existing version-comparison view can tell you
# which published version scored higher — see ``client.evaluations.prompts`` docs.
# ---------------------------------------------------------------------------


class Prompt(BaseModel):
    id: str = Field(alias="_id")
    name: str
    description: Optional[str] = None
    version: int
    text: str
    created_at: Optional[str] = Field(default=None, alias="createdAt")
    updated_at: Optional[str] = Field(default=None, alias="updatedAt")

    class Config:
        populate_by_name = True
        extra = "ignore"


# ---------------------------------------------------------------------------
# Evaluation subject
# ---------------------------------------------------------------------------

FrameworkKind = Literal[
    "raw_python",
    "openai",
    "anthropic",
    "google",
    "langchain",
    "llamaindex",
    "crewai",
    "autogen",
    "n8n",
    "flowise",
    "other",
]

RuntimeKind = Literal["local", "ci", "customer_hosted", "low_code"]


class EvaluationSubject(BaseModel):
    kind: Literal["custom_agent", "agentx_agent", "agentx_team"] = "custom_agent"
    display_name: Optional[str] = Field(default=None, alias="displayName")
    framework: Optional[FrameworkKind] = None
    framework_version: Optional[str] = Field(default=None, alias="frameworkVersion")
    runtime: Optional[RuntimeKind] = "local"
    agent_instructions: Optional[str] = Field(default=None, alias="agentInstructions")
    metadata: Optional[Dict[str, Union[str, int, bool]]] = None

    class Config:
        populate_by_name = True
        extra = "ignore"


# ---------------------------------------------------------------------------
# Supported models (the AgentX model registry / portability set)
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    """One entry from the AgentX supported-model registry — the set selectable
    for the Sovereignty & Portability Index. Returned by ``list_models()``."""

    name: str
    display: Optional[str] = None
    provider: Optional[str] = None
    context_window: Optional[int] = Field(default=None, alias="contextWindow")
    max_output_tokens: Optional[int] = Field(default=None, alias="maxOutputTokens")
    input_cost: Optional[float] = Field(default=None, alias="inputCost")
    output_cost: Optional[float] = Field(default=None, alias="outputCost")
    knowledge_cutoff: Optional[str] = Field(default=None, alias="knowledgeCutOff")
    legacy: Optional[bool] = None

    class Config:
        populate_by_name = True
        extra = "ignore"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class ServerLimits(BaseModel):
    max_batch_size: int = Field(default=10, alias="maxBatchSize")
    max_trace_bytes_per_result: int = Field(
        default=20000, alias="maxTraceBytesPerResult"
    )
    max_metadata_bytes_per_result: int = Field(
        default=4000, alias="maxMetadataBytesPerResult"
    )

    class Config:
        populate_by_name = True
        extra = "ignore"


class LiveStatistics(BaseModel):
    """Rating aggregate computed server-side from submitted results — available
    as soon as results are scored, independent of the `.analyze()` step (which
    only adds the LLM-driven qualitative report). Returned on the run resource
    (``GET /runs/:runId``) as ``liveStatistics``."""

    average_rating: Optional[float] = Field(default=None, alias="averageRating")
    min_rating: Optional[float] = Field(default=None, alias="minRating")
    max_rating: Optional[float] = Field(default=None, alias="maxRating")
    rated_count: int = Field(default=0, alias="ratedCount")

    class Config:
        populate_by_name = True
        extra = "ignore"


class SmokeTestVariantGroup(BaseModel):
    """Paraphrased variants for one question, generated server-side (reusing the same
    generation the dashboard's native runs use) and frozen for the lifetime of the run.
    The SDK never generates or counts these itself, it only consumes what's returned here."""

    question_index: int = Field(alias="questionIndex")
    variants: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True
        extra = "ignore"


class EvaluationRun(BaseModel):
    run_id: str = Field(alias="runId")
    dataset_id: str = Field(alias="datasetId")
    dataset_version_id: Optional[str] = Field(default=None, alias="datasetVersionId")
    status: str = "in_progress"
    limits: ServerLimits = Field(default_factory=ServerLimits)
    # Present only when at least one question in the dataset has smokeTest.enabled. See
    # SmokeTestVariantGroup.
    smoke_test_variants: Optional[List[SmokeTestVariantGroup]] = Field(
        default=None, alias="smokeTestVariants"
    )

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
    # Sovereignty & Portability: the model this case should run on. Set when the
    # dataset selects comparison models; your callable can read it to pick the
    # model. The SDK also tags the submitted result with it.
    model: Optional[str] = None
    # Smoke test: True when `query` is a server-generated paraphrase variant rather than the
    # dataset's original question text (see SmokeTestVariantGroup). Your callable doesn't need
    # to branch on this, `query` is already the text to ask, but it's available if you want to
    # log or handle variants differently.
    is_smoke_test_variant: bool = False
    smoke_test_variant_text: Optional[str] = None

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
    observable_trace: Optional[ObservableTrace] = Field(
        default=None, alias="observableTrace"
    )
    error: Optional[ResultError] = None
    timings: Optional[ResultTimings] = None
    metadata: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = Field(default=None, alias="idempotencyKey")
    # Links this result to a PromptTrace ingested via client.tracer.trace(..., sync=True) — lets
    # the dashboard's "Message Trace Details -> Execution Timeline" render the full execution
    # trace for this case, not just the lightweight observable_trace events above.
    trace_id: Optional[str] = Field(default=None, alias="traceId")
    # Smoke test: set by execute() from the originating EvaluationCase, not something you need to
    # set yourself when returning a plain str/dict from your callable.
    is_smoke_test_variant: Optional[bool] = Field(default=None, alias="isSmokeTestVariant")
    smoke_test_variant_text: Optional[str] = Field(default=None, alias="smokeTestVariantText")

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
    scored_results: List[ScoredResult] = Field(
        default_factory=list, alias="scoredResults"
    )
    # Server-computed rating aggregate, refreshed after this batch — see LiveStatistics.
    live_statistics: Optional[LiveStatistics] = Field(default=None, alias="liveStatistics")

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
    cosine_similarity: Optional[float] = Field(default=None, alias="cosineSimilarity")
    jaccard_similarity: Optional[float] = Field(default=None, alias="jaccardSimilarity")
    bleu_score: Optional[float] = Field(default=None, alias="bleuScore")
    rouge_score: Optional[float] = Field(default=None, alias="rougeScore")

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
    reasoning_patterns: List[str] = Field(
        default_factory=list, alias="reasoningPatterns"
    )
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


class SovereigntyModelMetrics(BaseModel):
    """Per-model row of the Sovereignty & Portability matrix."""

    model: str
    provider: Optional[str] = None
    is_baseline: bool = Field(default=False, alias="isBaseline")
    run_count: int = Field(default=0, alias="runCount")
    average_rating: Optional[float] = Field(default=None, alias="averageRating")
    min_rating: Optional[float] = Field(default=None, alias="minRating")
    max_rating: Optional[float] = Field(default=None, alias="maxRating")
    rating_variance: Optional[float] = Field(default=None, alias="ratingVariance")
    average_vector_similarity: Optional[float] = Field(
        default=None, alias="averageVectorSimilarity"
    )
    average_jaccard_similarity: Optional[float] = Field(
        default=None, alias="averageJaccardSimilarity"
    )
    average_bleu_score: Optional[float] = Field(
        default=None, alias="averageBleuScore"
    )
    average_rouge_score: Optional[float] = Field(
        default=None, alias="averageRougeScore"
    )
    average_latency_ms: Optional[float] = Field(default=None, alias="averageLatencyMs")
    total_input_tokens: Optional[int] = Field(default=None, alias="totalInputTokens")
    total_output_tokens: Optional[int] = Field(default=None, alias="totalOutputTokens")

    class Config:
        populate_by_name = True
        extra = "ignore"


class SovereigntyIndex(BaseModel):
    """Sovereignty & Portability matrix — side-by-side per-model performance,
    grouped from the model-tagged results of a run."""

    enabled: bool = False
    models: List[SovereigntyModelMetrics] = Field(default_factory=list)

    class Config:
        extra = "ignore"


class AnalysisResult(BaseModel):
    """Shared qualitative-report fields, produced by ``client.evaluations.run(...).analyze()``."""

    summary: Optional[str] = None
    consistency_score: Optional[float] = Field(default=None, alias="consistencyScore")
    instruction_adherence: Optional[ReportInstructionAdherence] = Field(
        default=None, alias="instructionAdherence"
    )
    response_patterns: Optional[ReportResponsePatterns] = Field(
        default=None, alias="responsePatterns"
    )
    reasoning_analysis: Optional[ReportReasoningAnalysis] = Field(
        default=None, alias="reasoningAnalysis"
    )
    tool_usage_analysis: Optional[ReportToolUsageAnalysis] = Field(
        default=None, alias="toolUsageAnalysis"
    )
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    overall_rating: Optional[str] = Field(default=None, alias="overallRating")
    recommendations: List[ReportRecommendation] = Field(default_factory=list)

    class Config:
        populate_by_name = True
        extra = "ignore"


class Report(AnalysisResult):
    run_id: str = Field(alias="runId")
    dataset_id: str = Field(alias="datasetId")
    status: str = "completed"
    statistics: Optional[ReportStatistics] = None
    low_scoring_cases: List[Dict[str, Any]] = Field(
        default_factory=list, alias="lowScoringCases"
    )
    sovereignty_index: Optional[SovereigntyIndex] = Field(
        default=None, alias="sovereigntyIndex"
    )
    dashboard_url: Optional[str] = Field(default=None, alias="dashboardUrl")

    class Config:
        populate_by_name = True
        extra = "ignore"

    @model_validator(mode="before")
    @classmethod
    def _hoist_similarity_into_statistics(cls, data: Any) -> Any:
        """Backend may send similarity metrics either at the top level (e.g.
        ``cosineSimilarity``) or nested under ``statistics``. Normalize so the
        nested form is always populated when either is present."""
        if not isinstance(data, dict):
            return data
        stats = data.get("statistics")
        stats = dict(stats) if isinstance(stats, dict) else {}
        for top_key, nested_key, marker in (
            ("cosineSimilarity", "cosineSimilarity", "cosine"),
            ("cosine_similarity", "cosine_similarity", "cosine"),
            ("jaccardSimilarity", "jaccardSimilarity", "jaccard"),
            ("jaccard_similarity", "jaccard_similarity", "jaccard"),
            ("bleuScore", "bleuScore", "bleu"),
            ("bleu_score", "bleu_score", "bleu"),
            ("rougeScore", "rougeScore", "rouge"),
            ("rouge_score", "rouge_score", "rouge"),
        ):
            top_val = data.get(top_key)
            if top_val is None:
                continue
            if (
                stats.get("cosineSimilarity") is None
                and stats.get("cosine_similarity") is None
                and marker == "cosine"
            ):
                stats[nested_key] = top_val
            if (
                stats.get("jaccardSimilarity") is None
                and stats.get("jaccard_similarity") is None
                and marker == "jaccard"
            ):
                stats[nested_key] = top_val
            if stats.get("bleuScore") is None and stats.get("bleu_score") is None and marker == "bleu":
                stats[nested_key] = top_val
            if stats.get("rougeScore") is None and stats.get("rouge_score") is None and marker == "rouge":
                stats[nested_key] = top_val
        if stats:
            data["statistics"] = stats
        return data

    @property
    def cosine_similarity(self) -> Optional[float]:
        """Average cosine similarity across scored results, or ``None`` if the
        metric was not enabled for the dataset or no result has a value yet."""
        return (
            self.statistics.cosine_similarity if self.statistics is not None else None
        )

    @property
    def jaccard_similarity(self) -> Optional[float]:
        """Average Jaccard similarity across scored results, or ``None`` if the
        metric was not enabled for the dataset or no result has a value yet."""
        return (
            self.statistics.jaccard_similarity if self.statistics is not None else None
        )

    @property
    def bleu_score(self) -> Optional[float]:
        """Average BLEU score across scored results, or ``None`` if the metric
        was not enabled for the dataset or no result has a value yet."""
        return self.statistics.bleu_score if self.statistics is not None else None

    @property
    def rouge_score(self) -> Optional[float]:
        """Average ROUGE-L (F1) score across scored results, or ``None`` if the
        metric was not enabled for the dataset or no result has a value yet."""
        return self.statistics.rouge_score if self.statistics is not None else None

    @property
    def average_rating(self) -> Optional[float]:
        """Convenience accessor matching cosine_similarity / jaccard_similarity."""
        return self.statistics.average_rating if self.statistics is not None else None


class AnalysisLevelProgress(BaseModel):
    total: int = 0
    completed: int = 0
    failed: int = 0
    percentage: int = 0

    class Config:
        extra = "ignore"


class AnalysisProgress(BaseModel):
    overall_percentage: int = Field(default=0, alias="overallPercentage")
    current_level: Optional[str] = Field(default=None, alias="currentLevel")
    levels: Dict[str, AnalysisLevelProgress] = Field(default_factory=dict)

    class Config:
        populate_by_name = True
        extra = "ignore"


class AnalysisFailureReason(BaseModel):
    code: str
    message: str
    retryable: bool = False

    class Config:
        extra = "ignore"


class AnalysisStatus(BaseModel):
    """Returned by ``client.evaluations.run(...).analyze()``'s polling loop
    (``EvaluationsClient.get_analysis_status``). ``status`` is terminal once it's one of
    "completed", "partially_failed", or "failed"."""

    job_id: Optional[str] = Field(default=None, alias="jobId")
    status: str = "not_started"
    progress: AnalysisProgress = Field(default_factory=AnalysisProgress)
    failure_reason: Optional[AnalysisFailureReason] = Field(default=None, alias="failureReason")
    warnings: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        populate_by_name = True
        extra = "ignore"

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "partially_failed", "failed")
