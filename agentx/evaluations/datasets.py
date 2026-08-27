from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from agentx.evaluations.models import Dataset

if TYPE_CHECKING:
    from agentx.evaluations.client import EvaluationsClient

logger = logging.getLogger(__name__)

_REQUIRED_CSV_COLS = {"query"}


class DatasetBuilder:
    """Fluent builder for creating a Custom Agent Evaluations dataset."""

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
            "questions": [],
        }
        # LLM-as-judge overrides for this dataset's own grading config. Omit either to keep the
        # server default (raw prompt template / OpenAI gpt-5.5, see EVALUATIONS.md). judge_model
        # must be one of client.evaluations.list_models() (OpenAI or Anthropic).
        if judge_prompt is not None:
            self._payload["judgePrompt"] = judge_prompt
        if judge_model is not None:
            self._payload["judgeModel"] = judge_model
        # Opt-in similarity metrics, surfaced on the report as cosine_similarity /
        # jaccard_similarity / bleu_score / rouge_score (computed against each
        # case's expected_results).
        if vector_similarity:
            vs: Dict[str, Any] = {"enabled": True}
            if similarity_model:
                vs["model"] = similarity_model
            self._payload["vectorSimilarity"] = vs
        if jaccard_similarity:
            self._payload["jaccardSimilarity"] = {"enabled": True}
        # Offline code scorers, versioned in the repo next to the dataset they guard (P1.4):
        # each entry is {"name", "code"} (a JS function body invoked as
        # score({input, output, expected, toolCalls})), optional "enabled" (default True).
        if code_scorers:
            import uuid as _uuid

            self._payload["codeScorers"] = [
                {
                    "id": scorer.get("id") or _uuid.uuid4().hex[:12],
                    "name": scorer["name"],
                    "code": scorer["code"],
                    "enabled": scorer.get("enabled", True),
                }
                for scorer in code_scorers
            ]
        if bleu_score:
            self._payload["bleuScore"] = {"enabled": True}
        if rouge_score:
            self._payload["rougeScore"] = {"enabled": True}
        # Sovereignty & Portability - the models to compare on this dataset (use
        # client.evaluations.list_models() to discover valid ids).
        if sovereignty_models:
            self._payload["sovereigntyIndex"] = {
                "enabled": True,
                "models": list(sovereignty_models),
            }

    def add_case(
        self,
        query: str,
        expected_results: Optional[str] = None,
        expected_capabilities: Optional[List[str]] = None,
        expected_knowledge_base: Optional[List[str]] = None,
        expected_delegations: Optional[List[str]] = None,
        follow_up_questions: Optional[List[Dict[str, Any]]] = None,
        judge_guideline: Optional[str] = None,
        smoke_test_count: Optional[int] = None,
        smoke_test_guidance: Optional[str] = None,
        expected_tools: Optional[List[str]] = None,
        trajectory_match_mode: str = "strict",
        expected_retrieval_context: Optional[Union[str, List[str]]] = None,
        splits: Optional[List[str]] = None,
    ) -> "DatasetBuilder":
        """Add a case. `judge_guideline` is optional grading guidance specific to this question.

        `splits` tags this case with named subsets (e.g. ``["smoke"]``): a run started with
        ``client.evaluations.run(dataset_id, subject, split="smoke")`` executes only the tagged
        cases (original case indexes are preserved, so per-case comparisons still line up with
        full runs). An untagged case belongs to no split and only runs in full runs.

        `expected_tools` declares the tool calls a correct run of this case should make. When a
        result links its trace (return `{"output": ..., "trace_id": span.trace_id}` from the
        agent function), the engine matches the trace's actual tool-call sequence against it and
        reports a pass/fail "Trajectory match" scorer row on the result. `trajectory_match_mode`
        follows agentevals semantics: "strict" (same calls, same order), "unordered" (same calls,
        any order), "superset" (all expected present, extras allowed), or "subset" (no unexpected
        calls, missing allowed).

        `expected_retrieval_context` (string or list of chunk strings) declares what a correct
        retriever should have fetched for this case. When the run's result carries actual
        retrieved context (a `retrieval_context` return value, or a linked trace with retrieval
        spans), the engine compares the two with token-level Jaccard similarity and reports a
        deterministic "Context match (jaccard)" scorer row (0-1) - a cheap retriever regression
        check with no LLM judge call.

        `smoke_test_count`, when set (1-10), asks this question that many extra ways each
        evaluation run, LLM-paraphrased server-side, to catch agents that are brittle to phrasing
        rather than genuinely wrong. `smoke_test_guidance` optionally steers what kind of variants
        get generated (e.g. tone, adversarial phrasing, different languages); the SDK never
        generates or counts variants itself, both fields are only ever consumed server-side.
        Ignored on `follow_up_questions`, only the opening question of a case can be smoke-tested.
        """
        main: Dict[str, Any] = {"query": query}
        if expected_results:
            main["expectedResults"] = expected_results
        if expected_capabilities:
            main["expectedCapabilities"] = expected_capabilities
        if expected_knowledge_base:
            main["expectedKnowledgeBase"] = expected_knowledge_base
        if expected_delegations:
            main["expectedDelegations"] = expected_delegations
        if judge_guideline:
            main["judgeGuideline"] = judge_guideline
        if smoke_test_count:
            main["smokeTest"] = {"enabled": True, "count": smoke_test_count}
            if smoke_test_guidance:
                main["smokeTest"]["guidance"] = smoke_test_guidance
        if expected_tools:
            main["expectedTrajectory"] = {"tools": expected_tools, "mode": trajectory_match_mode}
        if expected_retrieval_context:
            main["expectedRetrievalContext"] = expected_retrieval_context
        if splits:
            main["splits"] = splits
        self._payload["questions"].append(
            {
                "main_question": main,
                "follow_up_questions": follow_up_questions or [],
            }
        )
        return self

    def publish(self) -> Dataset:
        if not self._payload["questions"]:
            raise ValueError("Dataset must have at least one case before publishing")
        logger.info(
            "Publishing dataset '%s' with %d case(s)",
            self._payload["name"],
            len(self._payload["questions"]),
        )
        return self._client.create_dataset(self._payload)

    # ------------------------------------------------------------------
    # Class-level importers
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        client: "EvaluationsClient",
        path: str,
        name: str,
        **kwargs,
    ) -> "DatasetBuilder":
        """
        Load cases from a CSV file.
        Required column: query
        Optional columns: expected_results, expected_capabilities (semicolon-separated),
                          expected_knowledge_base (semicolon-separated), expected_delegations (semicolon-separated)
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        builder = cls(client, name=name, **kwargs)
        invalid: List[Dict[str, Any]] = []

        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
            missing = _REQUIRED_CSV_COLS - cols
            if missing:
                raise ValueError(f"CSV is missing required column(s): {missing}")

            for i, row in enumerate(reader, start=2):
                query = row.get("query", "").strip()
                if not query:
                    invalid.append({"row": i, "reason": "empty query"})
                    continue
                builder.add_case(
                    query=query,
                    expected_results=row.get("expected_results", "").strip() or None,
                    expected_capabilities=_split_semi(
                        row.get("expected_capabilities", "")
                    ),
                    expected_knowledge_base=_split_semi(
                        row.get("expected_knowledge_base", "")
                    ),
                    expected_delegations=_split_semi(
                        row.get("expected_delegations", "")
                    ),
                )

        if invalid:
            logger.warning("Skipped %d invalid row(s) from CSV:", len(invalid))
            for inv in invalid:
                logger.warning("  Row %d: %s", inv["row"], inv["reason"])

        return builder

    @classmethod
    def from_dataframe(
        cls,
        client: "EvaluationsClient",
        df: Any,
        name: str,
        **kwargs,
    ) -> "DatasetBuilder":
        """
        Load cases from a pandas DataFrame.
        Required column: query
        """
        cols = set(df.columns.tolist())
        missing = _REQUIRED_CSV_COLS - cols
        if missing:
            raise ValueError(f"DataFrame is missing required column(s): {missing}")

        builder = cls(client, name=name, **kwargs)
        invalid: List[Dict[str, Any]] = []

        for i, row in df.iterrows():
            query = str(row.get("query", "")).strip()
            if not query or query == "nan":
                invalid.append({"row": i, "reason": "empty query"})
                continue
            builder.add_case(
                query=query,
                expected_results=_str_or_none(row.get("expected_results")),
                expected_capabilities=_split_semi(row.get("expected_capabilities", "")),
                expected_knowledge_base=_split_semi(
                    row.get("expected_knowledge_base", "")
                ),
                expected_delegations=_split_semi(row.get("expected_delegations", "")),
            )

        if invalid:
            logger.warning("Skipped %d invalid row(s) from DataFrame:", len(invalid))
            for inv in invalid:
                logger.warning("  Row %s: %s", inv["row"], inv["reason"])

        return builder


class DatasetClient:
    """Thin wrapper surfaced as client.evaluations.datasets."""

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
    ) -> DatasetBuilder:
        return DatasetBuilder(
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
            # Was documented (evaluation/code-scorers.mdx) but not forwarded - DatasetBuilder
            # itself always accepted it. Fixed with the judge-scorer unification.
            code_scorers=code_scorers,
        )

    def from_csv(self, path: str, name: str, **kwargs) -> DatasetBuilder:
        return DatasetBuilder.from_csv(self._client, path=path, name=name, **kwargs)

    def from_dataframe(self, df: Any, name: str, **kwargs) -> DatasetBuilder:
        return DatasetBuilder.from_dataframe(self._client, df=df, name=name, **kwargs)

    def get(self, dataset_id: str) -> Dataset:
        return self._client.get_dataset(dataset_id)

    def list(self) -> List[Dataset]:
        return self._client.list_datasets()

    def delete(self, dataset_id: str) -> None:
        """Delete a dataset (and its grading config + version histories; past runs are kept)."""
        self._client.delete_dataset(dataset_id)

    def import_dataset(self, source: Any, name: Optional[str] = None) -> Dataset:
        """Create a NEW dataset from an exported/fetched one (a ``Dataset`` from ``get()``, or
        the engine's wire/NDJSON-export dict). Always a copy with a fresh id - never a
        restore-in-place. ``name`` optionally renames the copy."""
        wire: Dict[str, Any] = (
            source.model_dump(by_alias=True) if hasattr(source, "model_dump") else dict(source)
        )
        payload: Dict[str, Any] = {
            "name": name or wire.get("name") or "Imported dataset",
            "questions": wire.get("questions") or [],
        }
        for key in (
            "description",
            "numberOfRequests",
            "acceptanceCriteria",
            "rejectionCriteria",
            "evaluationCriteria",
            "vectorSimilarity",
            "jaccardSimilarity",
            "bleuScore",
            "rougeScore",
            "codeScorers",
        ):
            if wire.get(key) is not None:
                payload[key] = wire[key]
        return self._client.create_dataset(payload)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_semi(value: Any) -> Optional[List[str]]:
    if not value:
        return None
    s = str(value).strip()
    if not s or s == "nan":
        return None
    return [v.strip() for v in s.split(";") if v.strip()]


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s != "nan" else None
