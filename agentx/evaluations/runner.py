from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Union

from agentx.evaluations.adapters.raw import RawCallableAdapter
from agentx.evaluations.adapters.precomputed import PrecomputedAdapter
from agentx.evaluations.adapters.http_endpoint import HttpEndpointAdapter
from agentx.evaluations.client import EvaluationsClient, EvaluationSubmissionError
from agentx.evaluations.models import (
    AnalysisStatus,
    Dataset,
    EvaluationCase,
    EvaluationResult,
    EvaluationRun,
    EvaluationSettings,
    EvaluationSubject,
    LiveStatistics,
    ModelInfo,
    Report,
)
from agentx.evaluations.reporting import print_report
from agentx.evaluations.results import normalize_result, normalize_error
from agentx.evaluations._term import (
    bold,
    cyan,
    green,
    yellow,
    red,
    dim,
    BOLD,
    RESET,
    Spinner,
)

logger = logging.getLogger(__name__)


def _eval_quiet() -> bool:
    """AGENTX_EVAL_QUIET=1 silences the interactive progress UI (spinners, per-case lines) for
    CI logs - results, gate verdicts, and errors still print. Read per call so tests can toggle."""
    return os.getenv("AGENTX_EVAL_QUIET", "").lower() in ("1", "true", "yes")


def _say(*args, **kwargs) -> None:
    if not _eval_quiet():
        print(*args, **kwargs)

AdapterLike = Union[
    Callable[[EvaluationCase], Any],
    RawCallableAdapter,
    PrecomputedAdapter,
    HttpEndpointAdapter,
]

_ANALYSIS_LEVEL_LABELS = {
    "l1_score": "scoring responses",
    "l2_question_reduce": "reducing questions",
    "l3_cluster_reduce": "reducing clusters",
    "l4_final_reduce": "writing final report",
}

_DEFAULT_JUDGE_MODEL = "gpt-5.5"


class GateResult:
    """Wire result of the CI gate (GET /runs/:id/gate) with attribute access for the fields a
    CI script actually branches on."""

    def __init__(self, data: Dict[str, Any]):
        self.raw = data
        self.passed: bool = bool(data.get("passed"))
        self.average_rating: Optional[float] = data.get("averageRating")
        self.baseline_average: Optional[float] = data.get("baselineAverage")
        self.baseline_run_id: Optional[str] = data.get("baselineRunId")
        self.checks: List[Dict[str, Any]] = data.get("checks", [])

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1


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
        evaluation_settings: Optional[EvaluationSettings] = None,
        split: Optional[str] = None,
    ):
        self._client = client
        self._dataset = dataset
        self._run = run
        self._subject = subject
        # Named case subset for this run - only cases tagged with it are built/executed.
        self._split = split
        # When set, this run was started with an independently chosen grading
        # config (evaluation_settings_id) - its fields take precedence over the
        # dataset's own for anything execution-time reads (see _build_cases).
        self._evaluation_settings = evaluation_settings
        self._results: List[EvaluationResult] = []
        self._submitted_keys: Set[str] = set()
        self._report: Optional[Report] = None
        # Server-computed rating aggregate (Evaluate.liveStatistics) - refreshed
        # from the response of each append_results()/finalize_run() call. The
        # API is the single source of truth for this number (same value the
        # dashboard UI reads), so the SDK does not average results itself.
        self._live_stats: Optional[LiveStatistics] = None

    # ------------------------------------------------------------------
    # Step 1: execute
    # ------------------------------------------------------------------

    def execute(
        self,
        adapter: AdapterLike,
        concurrency: int = 1,
        reuse_outputs_from: Optional[str] = None,
    ) -> "EvaluationRunContext":
        """Run all cases locally and submit batches to AgentX.

        The whole loop runs inside the eval-run scope (tracing/eval_scope.py): any trace the
        agent function creates is stamped source="eval-run" + monitor=False automatically, so
        eval traffic never skews production monitoring and no one has to remember a flag.

        ``concurrency`` > 1 runs the agent callable across a thread pool (results are still
        submitted in case order, and the eval-run scope is propagated into the workers).
        ``reuse_outputs_from`` replays a previous run's recorded outputs for cases whose query
        text is unchanged instead of re-running (and re-paying for) the agent - the judge still
        re-scores them, which makes iterating on scorers cheap. Changed or new cases run
        normally.
        """
        from agentx.tracing.eval_scope import enter_eval_run, exit_eval_run

        scope_token = enter_eval_run(self._run.run_id)
        try:
            return self._execute_inner(adapter, concurrency=concurrency, reuse_outputs_from=reuse_outputs_from)
        finally:
            exit_eval_run(scope_token)

    def _fetch_reusable_outputs(self, run_id: str) -> Dict[tuple, str]:
        """(query, run_number, is_smoke_variant) -> output text, from a previous run's rows.
        Keyed on the query TEXT so a reworded case never silently reuses a stale answer."""
        try:
            prior = self._client.get_run(run_id)
        except Exception as exc:
            logger.warning("reuse_outputs_from: could not load run %s (%s) - running everything", run_id, exc)
            return {}
        reusable: Dict[tuple, str] = {}
        for row in prior.get("results", []) or []:
            if row.get("error") or row.get("status") == "failed":
                continue
            output = (row.get("output") or {}).get("text")
            query = (row.get("input") or {}).get("query")
            if not output or not query:
                continue
            key = (query, row.get("runNumber") or 1, bool(row.get("isSmokeTestVariant")))
            reusable[key] = output
        return reusable

    def _execute_inner(
        self,
        adapter: AdapterLike,
        concurrency: int = 1,
        reuse_outputs_from: Optional[str] = None,
    ) -> "EvaluationRunContext":
        normalized = _wrap_adapter(adapter)
        cases = _build_cases(self._dataset, self._run, self._evaluation_settings, split=self._split)
        max_batch = self._run.limits.max_batch_size
        reusable = self._fetch_reusable_outputs(reuse_outputs_from) if reuse_outputs_from else {}

        # Banner
        sep = "─" * 60
        name = self._dataset.name or self._dataset.id
        framework = self._subject.framework or "custom"
        runtime = self._subject.runtime or "local"
        display = self._subject.display_name or ""
        n_q = len({c.question_index for c in cases if not c.is_smoke_test_variant})
        n_r = (
            self._evaluation_settings.number_of_requests
            if self._evaluation_settings
            else self._dataset.number_of_requests
        )
        n_smoke = sum(1 for c in cases if c.is_smoke_test_variant)

        _say(cyan(sep))
        _say(f"  {bold('AgentX Evaluation')}  {dim(' - ')}  {name}")
        _say(cyan(sep))
        _say(f"  {dim('Run   :')} {dim(self._run.run_id)}")
        if display:
            _say(f"  {dim('Agent :')} {display}  {dim(f'({framework} / {runtime})')}")
        _say()
        exec_line = f"{bold('Executing')}  {n_q} question{'s' if n_q != 1 else ''} × {n_r} run{'s' if n_r != 1 else ''}"
        if n_smoke:
            variant_word = "variant" if n_smoke == 1 else "variants"
            exec_line += f"  {dim(f'(+{n_smoke} smoke-test {variant_word})')}"
        _say(exec_line)

        # Resume: skip already-submitted keys
        already_done = self._fetch_submitted_keys()

        batch: List[EvaluationResult] = []
        total = len(cases)

        def produce(case: EvaluationCase) -> EvaluationResult:
            # Cached replay: same query text at the same repetition reuses the recorded output
            # (the server still re-scores it with THIS run's grading config).
            cached = reusable.get((case.query, case.run_number, case.is_smoke_test_variant))
            if cached is not None:
                return normalize_result(
                    case, {"output": cached, "metadata": {"reusedFromRun": reuse_outputs_from}}
                )
            return normalized(case)

        if concurrency > 1:
            import concurrent.futures
            import contextvars

            def in_scope(case: EvaluationCase) -> EvaluationResult:
                # ContextVars (the eval-run scope) do not cross thread boundaries on their own -
                # each worker task runs inside a copy of the submitting thread's context so the
                # agent's traces still get stamped source="eval-run".
                return contextvars.copy_context().run(produce, case)

            pending = [
                case
                for case in cases
                if _idem_key(self._run.run_id, case.case_id, case.run_number) not in already_done
            ]
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
            # map() yields in submission order, so batching/submission below stays deterministic.
            mapped = executor.map(in_scope, pending)

            def ordered() -> "Iterator[EvaluationResult]":
                try:
                    yield from mapped
                finally:
                    executor.shutdown(wait=True)

            results_iter = ordered()
        else:
            results_iter = None  # sequential path below produces inline

        for idx, case in enumerate(cases, start=1):
            idem_key = _idem_key(self._run.run_id, case.case_id, case.run_number)

            if idem_key in already_done:
                logger.debug("Skipping already-submitted case: %s", idem_key)
                _print_progress(idx, total, case, skipped=True)
                continue

            result = next(results_iter) if results_iter is not None else produce(case)
            result.idempotency_key = idem_key
            # Tag the result with the case's model so the server can group it into
            # the Sovereignty & Portability matrix (the callable may also set it).
            if case.model:
                meta = dict(result.metadata or {})
                meta.setdefault("model", case.model)
                result.metadata = meta
            result = EvaluationResult(
                **{**result.model_dump(), "idempotencyKey": idem_key}
            )
            self._results.append(result)
            batch.append(result)
            _print_progress(idx, total, case, result=result)

            if len(batch) >= max_batch:
                self._flush_batch(batch)
                batch = []

        if batch:
            self._flush_batch(batch)

        return self

    def _flush_batch(self, batch: List[EvaluationResult]) -> None:
        batch_id = str(uuid.uuid4())
        n = len(batch)
        with Spinner(f"Scoring - AI is rating {n} result{'s' if n != 1 else ''}"):
            last_exc: Optional[Exception] = None
            for attempt in (1, 2):
                try:
                    resp = self._client.append_results(self._run.run_id, batch_id, batch)
                    if resp.live_statistics is not None:
                        self._live_stats = resp.live_statistics
                    _say(
                        f"  {green('✓')}  Scored {resp.accepted} result{'s' if resp.accepted != 1 else ''}"
                    )
                    logger.info(
                        "Batch %s: accepted=%d duplicates=%d failed=%d",
                        batch_id[:8],
                        resp.accepted,
                        resp.duplicates,
                        resp.failed_validation,
                    )
                    return
                except Exception as exc:
                    last_exc = exc
                    if attempt == 1:
                        logger.warning("Batch %s submission failed, retrying once: %s", batch_id[:8], exc)
        # A batch that cannot be submitted must FAIL the run, not print a red line and carry on:
        # execute() used to finish "successfully" having uploaded nothing. Failing fast also
        # stops paying for agent calls whose results would hit the same broken engine; a
        # re-execute() of this context resumes past everything already accepted (idempotency
        # keys are deterministic and the engine returns the submitted set).
        _say(f"  {red('✗')}  Scoring failed: {dim(str(last_exc))}")
        logger.error("Failed to submit batch %s after retry: %s", batch_id[:8], last_exc)
        raise EvaluationSubmissionError(
            f"Failed to submit a batch of {n} result(s) to the engine after a retry: {last_exc}. "
            "The run was left unfinalized; re-running execute() resumes past already-submitted cases."
        ) from last_exc

    def _fetch_submitted_keys(self) -> Set[str]:
        """Keys already accepted by this run - the engine's /missing-results route returns them
        so a re-execute() after a crash skips (and never re-pays for) finished cases."""
        try:
            return set(self._client.get_submitted_keys(self._run.run_id))
        except Exception:
            # Older engines without the route: no resume, identical to the historical behavior.
            return set()

    # ------------------------------------------------------------------
    # Step 2: finalize
    # ------------------------------------------------------------------

    def finalize(self) -> "EvaluationRunContext":
        _say()
        with Spinner("Finalizing - submitting results"):
            try:
                data = self._client.finalize_run(self._run.run_id)
                if isinstance(data, dict) and data.get("liveStatistics") is not None:
                    self._live_stats = LiveStatistics(**data["liveStatistics"])
                _say(f"  {green('✓')}  Finalized")
                logger.info("Run %s finalized", self._run.run_id)
            except Exception as exc:
                _say(f"  {red('✗')}  Finalize failed: {dim(str(exc))}")
                logger.error("Finalize failed: %s", exc)
        return self

    def gate(
        self,
        *,
        fail_under: Optional[float] = None,
        no_regression: bool = False,
        tolerance: Optional[float] = None,
        caller: str = "sdk",
    ) -> "GateResult":
        """CI gate (self-host): pass/fail this finalized run so a CI job can block a merge.

        ``fail_under`` fails the gate when the run's average rating is below the floor;
        ``no_regression=True`` fails it when the average dropped more than ``tolerance``
        (default 0.5, judge scores are noisy) below the dataset's previous completed run.
        At least one check is required. Prints a CI-log-friendly verdict and returns a
        :class:`GateResult` - the caller decides the exit code::

            report = client.evaluations.run(...).execute(my_agent).finalize()
            gate = report.gate(fail_under=7, no_regression=True)
            if not gate.passed:
                sys.exit(1)
        """
        data = self._client.gate_run(
            self._run.run_id,
            fail_under=fail_under,
            no_regression=no_regression,
            tolerance=tolerance,
            caller=caller,
        )
        result = GateResult(data)
        _say()
        for check in result.checks:
            mark = green("✓") if check.get("passed") else red("✗")
            _say(f"  {mark}  [{check.get('check')}] {check.get('detail')}")
        _say(f"  {green('✓  GATE PASSED') if result.passed else red('✗  GATE FAILED')}")
        return result

    # ------------------------------------------------------------------
    # Live rating stats - server-computed (Evaluate.liveStatistics), refreshed
    # from the response of each append_results()/finalize_run() call. Available
    # as soon as .execute() submits batches, no .analyze() required. The SDK
    # does not average ratings itself - this mirrors exactly what the dashboard
    # UI reads, computed once in the API.
    # ------------------------------------------------------------------

    def results(self) -> list:
        """Per-result rows for this run (rating, justification, code scorer rows, trace ids,
        latency/tokens, similarity metrics) - what the dashboard's run detail table shows,
        fetched fresh from the engine. Returns typed ``RunResultRow`` objects (snake_case
        attributes; ``.raw`` is the wire dict; dict-style access warns for one cycle - P1.5)."""
        from agentx.evaluations.models import RunResultRow

        detail = self._client.get_run(self.run_id)
        rows = detail.get("results", []) if isinstance(detail, dict) else []
        return [RunResultRow.from_wire(r) for r in rows]

    @property
    def run_id(self) -> str:
        """The server-side run id - handy for fetching the run's full results afterwards."""
        return self._run.run_id

    @property
    def rated_count(self) -> int:
        """Number of submitted results that have received a rating so far."""
        return self._live_stats.rated_count if self._live_stats else 0

    @property
    def average_rating(self) -> Optional[float]:
        """Live average rating across all results scored so far. Populated as
        soon as .execute() submits batches - unlike Report.average_rating,
        does not require .analyze()."""
        return self._live_stats.average_rating if self._live_stats else None

    @property
    def min_rating(self) -> Optional[float]:
        return self._live_stats.min_rating if self._live_stats else None

    @property
    def max_rating(self) -> Optional[float]:
        return self._live_stats.max_rating if self._live_stats else None

    # ------------------------------------------------------------------
    # Step 3: analyze + report
    # ------------------------------------------------------------------

    def analyze(
        self,
        mode: Optional[str] = None,
        quality_mode: Optional[str] = None,
        judges: Optional[List[str]] = None,
        poll_interval: float = 5.0,
        timeout: float = 1800.0,
    ) -> Report:
        """Generate the qualitative AI analysis report.

        Runs the same durable, multi-stage pipeline as the dashboard's "Analyze" button: each
        response is scored by 1-3 LLM judges (``judges``), then reduced into the final report.
        This starts the job and polls until it finishes, which can take noticeably longer than a
        single LLM call for larger runs.

        Args:
            mode: "auto" (default), "sync", or "batch" - how item scoring executes server-side.
            quality_mode: "quality_first" or "balanced" - how many items get a second/third judge.
            judges: 1-3 model ids, e.g. ``["gpt-5.5", "claude-opus-4-8"]``. Defaults to a single
                judge, ``["gpt-5.5"]``, rather than the dashboard's 3-judge default - SDK runs are
                typically lighter-weight, quick-start evaluations.
            poll_interval: seconds between status checks while waiting.
            timeout: give up waiting after this many seconds (the job keeps running server-side;
                call ``get_report()`` later to check on it).
        """
        if judges is not None and not (1 <= len(judges) <= 3):
            raise ValueError("judges must contain 1-3 model ids")
        resolved_judges = judges if judges is not None else [_DEFAULT_JUDGE_MODEL]

        _say()
        with Spinner("Analyzing - AI is reviewing your results") as spinner:
            try:
                self._client.analyze_run(
                    self._run.run_id,
                    mode=mode,
                    quality_mode=quality_mode,
                    judges=resolved_judges,
                )
                deadline = time.monotonic() + timeout
                status = self._client.get_analysis_status(self._run.run_id)
                while not status.is_terminal and time.monotonic() < deadline:
                    level = _ANALYSIS_LEVEL_LABELS.get(status.progress.current_level, "")
                    spinner.update(
                        f"Analyzing, {level + ' ' if level else ''}{status.progress.overall_percentage}%"
                    )
                    time.sleep(poll_interval)
                    status = self._client.get_analysis_status(self._run.run_id)

                if not status.is_terminal:
                    _say(f"  {yellow('!')}  Still running after {int(timeout)}s, check the dashboard for status")
                elif status.status == "failed":
                    reason = status.failure_reason.message if status.failure_reason else "unknown error"
                    _say(f"  {red('✗')}  Analyze failed: {dim(reason)}")
                else:
                    _say(f"  {green('✓')}  Analysis complete")
            except Exception as exc:
                _say(f"  {red('✗')}  Analyze failed: {dim(str(exc))}")
                logger.warning("Analyze request failed: %s", exc)

        try:
            report = self._client.get_report(self._run.run_id)
        except Exception as exc:
            # Deliberately not status="completed". A placeholder that claims completion is
            # indistinguishable from a real report of an evaluation that scored nothing -
            # print_report renders empty statistics and no recommendations either way - so
            # the one signal that something went wrong used to be a logger.warning that is
            # invisible unless the caller configured logging. Say it on stdout, and let the
            # status carry the truth for anything reading the object.
            _say(f"  {red('✗')}  Could not fetch the report: {dim(str(exc))}")
            logger.warning("Could not fetch report: %s", exc)
            report = Report(
                runId=self._run.run_id,
                datasetId=self._dataset.id,
                status="unavailable",
            )

        self._report = report
        _say()
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
        self.prompts = client.prompts
        self.tool_schemas = client.tool_schemas

    @property
    def settings(self):
        # Deferred: touching client.settings eagerly here would fire its legacy-surface
        # DeprecationWarning for every runner, used or not.
        return self._client.settings

    def list_models(self, provider: Optional[str] = None) -> List[ModelInfo]:
        """List the LLM models AgentX supports - the same set selectable for
        the Sovereignty & Portability Index. Pass ``provider`` (e.g. "Google")
        to filter. Useful for discovering valid model identifiers to compare
        against."""
        return self._client.list_models(provider)

    def list_gates(self) -> list:
        """Recorded CI gate verdicts, newest first (the dashboard's CI Gates history)."""
        return self._client.list_gates()

    def simulate_conversation(self, **kwargs) -> dict:
        """Persona-driven multi-turn simulation against a prompt - see
        EvaluationsClient.simulate_conversation for parameters."""
        return self._client.simulate_conversation(**kwargs)

    def compare_pairwise(self, run_a_id: str, run_b_id: str, **kwargs):
        """Head-to-head judging between two runs of the same dataset - "which answer is better"
        rather than two absolute scores. See EvaluationsClient.compare_pairwise for the options,
        and agentx.testing.assert_pairwise for the pytest-side check."""
        return self._client.compare_pairwise(run_a_id, run_b_id, **kwargs)

    def get_pairwise(self, batch_id: str):
        """Read back a stored head-to-head by its batch id."""
        return self._client.get_pairwise(batch_id)

    def list_pairwise(self, **kwargs) -> list:
        """Summaries of past head-to-heads, newest first, optionally narrowed to one run."""
        return self._client.list_pairwise(**kwargs)

    def get_run(self, run_id: str) -> dict:
        """Run summary + per-result rows by id, without needing the EvaluationRunContext that
        created it (e.g. from a separate process)."""
        return self._client.get_run(run_id)

    def get_analysis_status(self, run_id: str) -> AnalysisStatus:
        """Check on an in-progress ``.analyze()`` job by run id, without needing
        the ``EvaluationRunContext`` that started it (e.g. from a separate
        script execution)."""
        return self._client.get_analysis_status(run_id)

    def gate_run(
        self,
        run_id: str,
        *,
        fail_under: Optional[float] = None,
        no_regression: bool = False,
        tolerance: Optional[float] = None,
        record: bool = True,
        caller: Optional[str] = "sdk",
    ) -> GateResult:
        """CI-gate any finalized run by id - the standalone form of
        ``EvaluationRunContext.gate()``, for gating a run created elsewhere or
        re-checking one without re-running it (self-host only). Pass
        ``record=False`` for a check that stays out of the dashboard's CI Gates
        history."""
        return GateResult(
            self._client.gate_run(
                run_id,
                fail_under=fail_under,
                no_regression=no_regression,
                tolerance=tolerance,
                record=record,
                caller=caller,
            )
        )

    def run(
        self,
        dataset_id: str,
        subject: Union[Dict[str, Any], EvaluationSubject],
        scorer_id: Optional[str] = None,
        evaluation_settings_id: Optional[str] = None,
        split: Optional[str] = None,
    ) -> EvaluationRunContext:
        """Start a run of ``dataset_id`` against ``subject``. Pass ``scorer_id`` (an LLM Judge
        Scorer's id, e.g. from ``client.monitor.judge_scorers``) to grade with a specific
        scorer instead of the dataset's default. ``evaluation_settings_id`` is the
        pre-consolidation alias for the same id and keeps working.

        ``split`` runs only the cases tagged with that named subset (``add_case(...,
        splits=["smoke"])``) - the cheap-PR-run vs nightly-full-run workflow. Original case
        indexes are preserved so per-case comparisons line up with full runs."""
        from agentx.evaluations.client import _resolve_scorer_id

        if isinstance(subject, dict):
            subject = EvaluationSubject(**subject)

        grader_id = _resolve_scorer_id(scorer_id, evaluation_settings_id)
        dataset = self._client.get_dataset(dataset_id)
        evaluation_settings = (
            self._client.get_evaluation_settings(grader_id) if grader_id else None
        )
        run = self._client.init_run(dataset_id, subject, scorer_id=grader_id, split=split)
        case_count = (
            sum(1 for q in dataset.questions if split in (q.main_question.splits or []))
            if split
            else len(dataset.questions)
        )
        logger.info(
            "Started evaluation run %s on dataset %s (%d case(s)%s, %d repetition(s))",
            run.run_id,
            dataset_id,
            case_count,
            f' in split "{split}"' if split else "",
            evaluation_settings.number_of_requests
            if evaluation_settings
            else dataset.number_of_requests,
        )
        return EvaluationRunContext(
            self._client, dataset, run, subject, evaluation_settings=evaluation_settings, split=split
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_adapter(adapter: AdapterLike) -> Callable[[EvaluationCase], EvaluationResult]:
    if isinstance(
        adapter, (RawCallableAdapter, PrecomputedAdapter, HttpEndpointAdapter)
    ):
        return adapter.run
    if callable(adapter):
        return RawCallableAdapter(adapter).run
    raise TypeError(
        f"adapter must be callable or an Adapter instance, got {type(adapter)}"
    )


def _build_cases(
    dataset: Dataset,
    run: EvaluationRun,
    evaluation_settings: Optional[EvaluationSettings] = None,
    split: Optional[str] = None,
) -> List[EvaluationCase]:
    cases: List[EvaluationCase] = []
    # When an independent evaluation_settings was chosen (evaluation_settings_id
    # passed to .run()), its numberOfRequests/sovereigntyIndex take precedence
    # over the dataset's own - that's the whole point of decoupling them. With
    # no evaluation_settings, this reproduces today's exact behavior.
    n_runs = max(
        (evaluation_settings.number_of_requests if evaluation_settings else dataset.number_of_requests),
        1,
    )
    # Sovereignty & Portability: when the config selects comparison models, run
    # every question/run once per model in this single run so the report groups
    # results into a per-model portability matrix (mirrors the native route).
    # ``[None]`` keeps legacy single-model behavior (case.model stays unset).
    sovereignty_models = (
        evaluation_settings.sovereignty_models if evaluation_settings else dataset.sovereignty_models
    )
    models: List[Optional[str]] = list(sovereignty_models) or [None]
    # Smoke test: variant text is generated and counted entirely server-side (POST /runs, reusing
    # the same generation the dashboard's native runs use) and handed back on `run`. The SDK never
    # re-derives eligibility, count, or text itself, it only turns what the server already decided
    # into extra cases. Not multiplied across sovereignty_models, matching the server's counting in
    # finalize/missing-results (one case per variant per question, regardless of comparison models).
    smoke_variants_by_question = {
        group.question_index: group.variants for group in (run.smoke_test_variants or [])
    }
    for q_idx, question in enumerate(dataset.questions):
        mq = question.main_question
        # Split filtering preserves q_idx: a "smoke" run's case 7 is the same case 7 a full run
        # scores, so per-case comparisons line up across the two.
        if split and split not in (mq.splits or []):
            continue
        for run_num in range(1, n_runs + 1):
            for model in models:
                suffix = f"::{model}" if model else ""
                cases.append(
                    EvaluationCase(
                        case_id=f"case-{q_idx}{suffix}",
                        question_index=q_idx,
                        run_number=run_num,
                        query=mq.query,
                        expected_results=mq.expected_results,
                        expected_capabilities=mq.expected_capabilities,
                        expected_knowledge_base=mq.expected_knowledge_base,
                        expected_delegations=mq.expected_delegations,
                        model=model,
                    )
                )
        for variant_idx, variant_text in enumerate(smoke_variants_by_question.get(q_idx, [])):
            cases.append(
                EvaluationCase(
                    case_id=f"case-{q_idx}",
                    question_index=q_idx,
                    run_number=n_runs + variant_idx + 1,
                    query=variant_text,
                    expected_results=mq.expected_results,
                    expected_capabilities=mq.expected_capabilities,
                    expected_knowledge_base=mq.expected_knowledge_base,
                    expected_delegations=mq.expected_delegations,
                    is_smoke_test_variant=True,
                    smoke_test_variant_text=variant_text,
                )
            )
    return cases


def _idem_key(run_id: str, case_id: str, run_number: int) -> str:
    return f"{run_id}:{case_id}:run-{run_number}"


def _print_progress(
    idx: int,
    total: int,
    case: EvaluationCase,
    skipped: bool = False,
    result: Optional[EvaluationResult] = None,
) -> None:
    if skipped:
        tag = yellow("↷")
        suffix = dim("skipped")
    elif result and result.error:
        tag = red("✗")
        suffix = red(f"error: {result.error.message[:60]}")
    else:
        tag = green("✓")
        parts = []
        if result and result.timings:
            t = result.timings
            if t.latency_ms is not None:
                parts.append(dim(f"{t.latency_ms}ms"))
            if t.input_tokens is not None and t.output_tokens is not None:
                parts.append(dim(f"{t.input_tokens}→{t.output_tokens} tok"))
        suffix = "  ".join(parts)

    counter = dim(f"[{idx}/{total}]")
    label = dim(f"Q{case.question_index + 1} run #{case.run_number}")
    query_preview = (case.query[:55] + "…") if len(case.query) > 55 else case.query
    line = f"  {tag}  {counter} {label}  {query_preview}"
    if suffix:
        line += f"  {suffix}"
    _say(line)
