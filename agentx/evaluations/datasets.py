from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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
        vector_similarity: bool = False,
        jaccard_similarity: bool = False,
        bleu_score: bool = False,
        rouge_score: bool = False,
        similarity_model: Optional[str] = None,
        sovereignty_models: Optional[List[str]] = None,
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
        if bleu_score:
            self._payload["bleuScore"] = {"enabled": True}
        if rouge_score:
            self._payload["rougeScore"] = {"enabled": True}
        # Sovereignty & Portability — the models to compare on this dataset (use
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
    ) -> "DatasetBuilder":
        main = {"query": query}
        if expected_results:
            main["expectedResults"] = expected_results
        if expected_capabilities:
            main["expectedCapabilities"] = expected_capabilities
        if expected_knowledge_base:
            main["expectedKnowledgeBase"] = expected_knowledge_base
        if expected_delegations:
            main["expectedDelegations"] = expected_delegations
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
        vector_similarity: bool = False,
        jaccard_similarity: bool = False,
        bleu_score: bool = False,
        rouge_score: bool = False,
        similarity_model: Optional[str] = None,
        sovereignty_models: Optional[List[str]] = None,
    ) -> DatasetBuilder:
        return DatasetBuilder(
            self._client,
            name=name,
            description=description,
            number_of_requests=number_of_requests,
            acceptance_criteria=acceptance_criteria,
            rejection_criteria=rejection_criteria,
            evaluation_criteria=evaluation_criteria,
            vector_similarity=vector_similarity,
            jaccard_similarity=jaccard_similarity,
            bleu_score=bleu_score,
            rouge_score=rouge_score,
            similarity_model=similarity_model,
            sovereignty_models=sovereignty_models,
        )

    def from_csv(self, path: str, name: str, **kwargs) -> DatasetBuilder:
        return DatasetBuilder.from_csv(self._client, path=path, name=name, **kwargs)

    def from_dataframe(self, df: Any, name: str, **kwargs) -> DatasetBuilder:
        return DatasetBuilder.from_dataframe(self._client, df=df, name=name, **kwargs)

    def get(self, dataset_id: str) -> Dataset:
        return self._client.get_dataset(dataset_id)

    def list(self) -> List[Dataset]:
        return self._client.list_datasets()


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
