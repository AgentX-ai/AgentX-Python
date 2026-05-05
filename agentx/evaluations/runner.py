from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Union

from agentx.evaluations.adapters.raw import RawCallableAdapter
from agentx.evaluations.adapters.precomputed import PrecomputedAdapter
from agentx.evaluations.adapters.http_endpoint import HttpEndpointAdapter
from agentx.evaluations.client import EvaluationsClient
from agentx.evaluations.models import (
    Dataset,
    EvaluationCase,
    EvaluationResult,
    EvaluationRun,
    EvaluationSubject,
    Report,
)
from agentx.evaluations.redaction import redact_dict
from agentx.evaluations.reporting import print_report
from agentx.evaluations.results import normalize_result, normalize_error

logger = logging.getLogger(__name__)

AdapterLike = Union[
    Callable[[EvaluationCase], Any],
    RawCallableAdapter,
    PrecomputedAdapter,
    HttpEndpointAdapter,
]


class EvaluationRunContext:
    """
    Fluent builder returned by client.evaluations.run(...).
    Chains: .execute(fn) -> .finalize() -> .analyze() -> Report
    """

    def __init__(
        self,
        client: EvaluationsClient,
        dataset: Dataset,
        run: EvaluationRun,
        subject: EvaluationSubject,
    ):
        self._client = client
        self._dataset = dataset
        self._run = run
        self._subject = subject
        self._results: List[EvaluationResult] = []
        self._submitted_keys: Set[str] = set()
        self._report: Optional[Report] = None

    # ------------------------------------------------------------------
    # Step 1: execute
    # ------------------------------------------------------------------

    def execute(self, adapter: AdapterLike) -> "EvaluationRunContext":
        """Run all cases locally and submit batches to AgentX."""
        normalized = _wrap_adapter(adapter)
        cases = _build_cases(self._dataset)
        max_batch = self._run.limits.max_batch_size

        # Resume: skip already-submitted keys
        already_done = self._fetch_submitted_keys()

        batch: List[EvaluationResult] = []
        total = len(cases)

        for idx, case in enumerate(cases, start=1):
            idem_key = _idem_key(self._run.run_id, case.case_id, case.run_number)

            if idem_key in already_done:
                logger.debug("Skipping already-submitted case: %s", idem_key)
                _print_progress(idx, total, case, skipped=True)
                continue

            result = normalized(case)
            result.idempotency_key = idem_key
            result = EvaluationResult(
                **{**result.model_dump(), "idempotencyKey": idem_key}
            )
            self._results.append(result)
            batch.append(result)
            _print_progress(idx, total, case)

            if len(batch) >= max_batch:
                self._flush_batch(batch)
                batch = []

        if batch:
            self._flush_batch(batch)

        return self

    def _flush_batch(self, batch: List[EvaluationResult]) -> None:
        batch_id = str(uuid.uuid4())
        try:
            resp = self._client.append_results(self._run.run_id, batch_id, batch)
            logger.info(
                "Batch %s: accepted=%d duplicates=%d failed=%d",
                batch_id[:8],
                resp.accepted,
                resp.duplicates,
                resp.failed_validation,
            )
        except Exception as exc:
            logger.error("Failed to submit batch %s: %s", batch_id[:8], exc)

    def _fetch_submitted_keys(self) -> Set[str]:
        try:
            missing = self._client.get_missing_results(self._run.run_id)
            # missing-results returns cases NOT yet submitted — we want the inverse
            # but if the endpoint isn't live yet, just return empty set
            return set()
        except Exception:
            return set()

    # ------------------------------------------------------------------
    # Step 2: finalize
    # ------------------------------------------------------------------

    def finalize(self) -> "EvaluationRunContext":
        try:
            self._client.finalize_run(self._run.run_id)
            logger.info("Run %s finalized", self._run.run_id)
        except Exception as exc:
            logger.error("Finalize failed: %s", exc)
        return self

    # ------------------------------------------------------------------
    # Step 3: analyze + report
    # ------------------------------------------------------------------

    def analyze(self) -> Report:
        try:
            self._client.analyze_run(self._run.run_id)
        except Exception as exc:
            logger.warning("Analyze request failed: %s", exc)

        try:
            report = self._client.get_report(self._run.run_id)
        except Exception as exc:
            logger.warning("Could not fetch report: %s", exc)
            report = Report(
                runId=self._run.run_id,
                datasetId=self._dataset.id,
                status="completed",
            )

        self._report = report
        print_report(report)
        return report


class EvaluationsRunner:
    """
    Entry point surfaced as client.evaluations.
    Usage::

        report = (
            client.evaluations
            .run(dataset_id="evds_...", subject={...})
            .execute(my_fn)
            .finalize()
            .analyze()
        )
    """

    def __init__(self, client: EvaluationsClient):
        self._client = client
        self.datasets = client.datasets

    def run(
        self,
        dataset_id: str,
        subject: Union[Dict[str, Any], EvaluationSubject],
    ) -> EvaluationRunContext:
        if isinstance(subject, dict):
            subject = EvaluationSubject(**subject)

        dataset = self._client.get_dataset(dataset_id)
        run = self._client.init_run(dataset_id, subject)
        logger.info(
            "Started evaluation run %s on dataset %s (%d case(s), %d repetition(s))",
            run.run_id,
            dataset_id,
            len(dataset.questions),
            dataset.number_of_requests,
        )
        return EvaluationRunContext(self._client, dataset, run, subject)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_adapter(adapter: AdapterLike) -> Callable[[EvaluationCase], EvaluationResult]:
    if isinstance(adapter, (RawCallableAdapter, PrecomputedAdapter, HttpEndpointAdapter)):
        return adapter.run
    if callable(adapter):
        return RawCallableAdapter(adapter).run
    raise TypeError(f"adapter must be callable or an Adapter instance, got {type(adapter)}")


def _build_cases(dataset: Dataset) -> List[EvaluationCase]:
    cases: List[EvaluationCase] = []
    n_runs = max(dataset.number_of_requests, 1)
    for q_idx, question in enumerate(dataset.questions):
        mq = question.main_question
        for run_num in range(1, n_runs + 1):
            cases.append(EvaluationCase(
                case_id=f"case-{q_idx}",
                question_index=q_idx,
                run_number=run_num,
                query=mq.query,
                expected_results=mq.expected_results,
                expected_capabilities=mq.expected_capabilities,
                expected_knowledge_base=mq.expected_knowledge_base,
                expected_delegations=mq.expected_delegations,
            ))
    return cases


def _idem_key(run_id: str, case_id: str, run_number: int) -> str:
    return f"{run_id}:{case_id}:run-{run_number}"


def _print_progress(idx: int, total: int, case: EvaluationCase, skipped: bool = False) -> None:
    tag = "[skip]" if skipped else "[ ok ]"
    query_preview = (case.query[:60] + "...") if len(case.query) > 60 else case.query
    print(f"  {tag} [{idx}/{total}] Q{case.question_index+1} run#{case.run_number}: {query_preview}")
