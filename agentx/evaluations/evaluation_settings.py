from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agentx.evaluations.models import EvaluationSettings

if TYPE_CHECKING:
    from agentx.evaluations.client import EvaluationsClient

logger = logging.getLogger(__name__)


class EvaluationSettingsBuilder:
    """Fluent builder for creating a standalone, reusable grading config (no
    dataset/questions attached)."""

    def __init__(
        self,
        client: "EvaluationsClient",
        name: str,
        description: Optional[str] = None,
        number_of_requests: int = 1,
        acceptance_criteria: Optional[str] = None,
        rejection_criteria: Optional[str] = None,
        evaluation_criteria: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        judge_model: Optional[str] = None,
        vector_similarity: bool = False,
        jaccard_similarity: bool = False,
        bleu_score: bool = False,
        rouge_score: bool = False,
        similarity_model: Optional[str] = None,
        sovereignty_models: Optional[List[str]] = None,
        code_scorers: Optional[List[Dict[str, Any]]] = None,
    ):
        self._client = client
        self._payload: Dict[str, Any] = {
            "name": name,
            "description": description,
            "numberOfRequests": number_of_requests,
            "acceptanceCriteria": acceptance_criteria,
            "rejectionCriteria": rejection_criteria,
            "evaluationCriteria": evaluation_criteria,
        }
        # LLM-as-judge overrides. Omit either to keep the server default (raw prompt template /
        # OpenAI gpt-5.5, see EVALUATIONS.md). judge_model must be one of
        # client.evaluations.list_models() (OpenAI or Anthropic).
        if judge_prompt is not None:
            self._payload["judgePrompt"] = judge_prompt
        if judge_model is not None:
            self._payload["judgeModel"] = judge_model
        # Opt-in similarity metrics, mirrors DatasetBuilder's config kwargs.
        if vector_similarity:
            vs: Dict[str, Any] = {"enabled": True}
            if similarity_model:
                vs["model"] = similarity_model
            self._payload["vectorSimilarity"] = vs
        if jaccard_similarity:
            self._payload["jaccardSimilarity"] = {"enabled": True}
        if bleu_score:
            self._payload["bleuScore"] = {"enabled": True}
        if rouge_score:
            self._payload["rougeScore"] = {"enabled": True}
        # Sovereignty & Portability - the models to compare when this config runs
        # (use client.evaluations.list_models() to discover valid ids).
        if sovereignty_models:
            self._payload["sovereigntyIndex"] = {
                "enabled": True,
                "models": list(sovereignty_models),
            }
        # Sandboxed JS scorers run per result alongside the judge - each entry is
        # {"name": ..., "enabled": True, "code": "..."} where the code is a JS function body
        # receiving (input, output, expected, toolCalls) and returning {score, reasoning}.
        if code_scorers:
            self._payload["codeScorers"] = list(code_scorers)

    def publish(self) -> EvaluationSettings:
        logger.info("Publishing evaluation settings '%s'", self._payload["name"])
        return self._client.create_evaluation_settings(self._payload)


class EvaluationSettingsClient:
    """Thin wrapper surfaced as ``client.evaluations.settings``.

    Note: an evaluation-settings record is the judge rubric + OFFLINE profile of an
    **LLM Judge Scorer** - the unified entity at ``client.monitor.judge_scorers``, which also
    carries the optional online (live-traffic) profile. This client keeps working unchanged;
    prefer ``judge_scorers`` for new code so both profiles live in one place."""

    def __init__(self, client: "EvaluationsClient"):
        self._client = client

    def builder(
        self,
        name: str,
        description: Optional[str] = None,
        number_of_requests: int = 1,
        acceptance_criteria: Optional[str] = None,
        rejection_criteria: Optional[str] = None,
        evaluation_criteria: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        judge_model: Optional[str] = None,
        vector_similarity: bool = False,
        jaccard_similarity: bool = False,
        bleu_score: bool = False,
        rouge_score: bool = False,
        similarity_model: Optional[str] = None,
        sovereignty_models: Optional[List[str]] = None,
        code_scorers: Optional[List[Dict[str, Any]]] = None,
    ) -> EvaluationSettingsBuilder:
        return EvaluationSettingsBuilder(
            self._client,
            name=name,
            description=description,
            number_of_requests=number_of_requests,
            acceptance_criteria=acceptance_criteria,
            rejection_criteria=rejection_criteria,
            evaluation_criteria=evaluation_criteria,
            judge_prompt=judge_prompt,
            judge_model=judge_model,
            vector_similarity=vector_similarity,
            jaccard_similarity=jaccard_similarity,
            bleu_score=bleu_score,
            rouge_score=rouge_score,
            similarity_model=similarity_model,
            sovereignty_models=sovereignty_models,
            code_scorers=code_scorers,
        )

    def get(self, evaluation_settings_id: str) -> EvaluationSettings:
        return self._client.get_evaluation_settings(evaluation_settings_id)

    def list(self) -> List[EvaluationSettings]:
        return self._client.list_evaluation_settings()
