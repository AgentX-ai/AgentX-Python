from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

from agentx.evaluations.models import (
    AnalysisStatus,
    BatchAppendResponse,
    Dataset,
    EvaluationResult,
    EvaluationRun,
    EvaluationSettings,
    EvaluationSubject,
    ModelInfo,
    PairwiseComparison,
    Prompt,
    Report,
)

logger = logging.getLogger(__name__)

from agentx.util import _DEFAULT_API_BASE as _UTIL_API_BASE

_DEFAULT_BASE_URL = f"{_UTIL_API_BASE}/custom-agent-evaluations"
SDK_NAME = "agentx-python"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 2.0, 4.0]

# The self-host analyze route judges every result before it responds, so the client has to
# wait out the whole job on one connection. Matches EvaluationRunContext.analyze()'s own
# default timeout.
_SELF_HOST_ANALYZE_TIMEOUT = 1800


class AgentXEvaluationsError(Exception):
    """An evaluations API call failed.

    ``status_code`` carries the HTTP status when the failure came from a response rather
    than from the transport, so callers can branch on it instead of matching on the message
    text. It is ``None`` for connection errors and for retry exhaustion.
    """

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AgentXAuthError(AgentXEvaluationsError):
    pass


class AgentXValidationError(AgentXEvaluationsError):
    pass


class EvaluationSubmissionError(AgentXEvaluationsError):
    """A result batch could not be submitted (after one retry). The run is left unfinalized;
    re-running execute() on the same context resumes past already-submitted cases."""

    pass


def _resolve_scorer_id(scorer_id: Optional[str], evaluation_settings_id: Optional[str]) -> Optional[str]:
    """One grader, two spellings: ``scorer_id`` is the post-consolidation name for what the wire
    still calls ``evaluationSettingsId`` (the ids are identical by design). Both kwargs are
    accepted everywhere a run picks its grader; passing both with different values is a bug."""
    if scorer_id and evaluation_settings_id and scorer_id != evaluation_settings_id:
        raise ValueError("Pass either scorer_id or evaluation_settings_id (they are the same id), not two different ids")
    return scorer_id or evaluation_settings_id


class EvaluationsClient:
    def __init__(
        self,
        api_key: str,
        sdk_version: str = "unknown",
        base_url: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ):
        if not api_key:
            raise AgentXAuthError("AGENTX_API_KEY is required")
        self._api_key = api_key
        self._sdk_version = sdk_version
        # Falls back to the caller's default workspace server-side when unset - see
        # _with_workspace(). Without this, dataset/settings/run creation silently land in
        # whatever workspace the API key's user defaults to, not the one the caller intended.
        self._workspace_id = workspace_id
        # Priority: constructor arg > env var > SDK default
        # Always append /custom-agent-evaluations so users only need to provide /api/v1
        _api_base = (
            base_url or os.getenv("AGENTX_API_BASE_URL", _UTIL_API_BASE)
        ).rstrip("/")
        if not _api_base.endswith("/custom-agent-evaluations"):
            _api_base = f"{_api_base}/custom-agent-evaluations"
        self._base_url = _api_base
        # None until an analysis call tells us which engine this is; see _api_root.
        self._analysis_on_dashboard_router: Optional[bool] = None
        self._session = requests.Session()
        self._session.headers.update(
            {
                "x-api-key": self._api_key,
                "Content-Type": "application/json",
                "User-Agent": f"{SDK_NAME}/{self._sdk_version}",
                "accept": "*/*",
            }
        )
        # Expose dataset / evaluation-settings / prompt builder factories
        from agentx.evaluations.datasets import DatasetClient
        from agentx.evaluations.evaluation_settings import EvaluationSettingsClient
        from agentx.evaluations.prompts import PromptClient

        self.datasets = DatasetClient(self)
        # Legacy view of an LLM Judge Scorer's offline profile - constructed lazily so its
        # DeprecationWarning fires on first USE, not for every client that never touches it.
        self._settings: "EvaluationSettingsClient | None" = None
        self.prompts = PromptClient(self)
        from agentx.evaluations.tool_schemas import ToolSchemaClient
        self.tool_schemas = ToolSchemaClient(self)

    @property
    def settings(self) -> "EvaluationSettingsClient":
        if self._settings is None:
            from agentx.evaluations.evaluation_settings import EvaluationSettingsClient

            self._settings = EvaluationSettingsClient(self)
        return self._settings

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _with_workspace(self, payload: dict) -> dict:
        """Injects the client's workspace_id into a request payload, unless the caller already
        set one explicitly. Without this, requests silently fall back to the API key user's
        default workspace server-side, which may not be the workspace the caller intended."""
        if self._workspace_id and not payload.get("workspaceId"):
            return {**payload, "workspaceId": self._workspace_id}
        return payload

    def _workspace_params(self) -> Optional[dict]:
        """Same as _with_workspace(), for GET requests that take workspaceId as a query param."""
        return {"workspaceId": self._workspace_id} if self._workspace_id else None

    def _request(
        self,
        method: str,
        path: str,
        timeout: int = 30,
        base: Optional[str] = None,
        retry: bool = True,
        **kwargs,
    ) -> Any:
        """Call the evaluations API.

        ``base`` overrides the ``/custom-agent-evaluations`` prefix for the handful of
        routes that live on a different router (see ``_api_root``).

        ``retry=False`` disables the backoff loop entirely. Use it for any request that is
        both slow and billable: the loop retries on ``requests.RequestException``, which
        includes read timeouts, so a synchronous endpoint that outlives its timeout would
        otherwise be re-invoked, and paid for, up to four times.
        """
        url = f"{base or self._base_url}{path}"
        schedule = [0.0] + (_RETRY_BACKOFF if retry else [])
        last_exc: Optional[Exception] = None
        for attempt, wait in enumerate(schedule):
            if wait:
                time.sleep(wait)
            try:
                resp = self._session.request(method, url, timeout=timeout, **kwargs)
            except requests.RequestException as e:
                last_exc = e
                logger.debug("Request error (attempt %d): %s", attempt + 1, e)
                continue

            if resp.status_code == 401:
                raise AgentXAuthError("Invalid or missing API key")
            if resp.status_code == 422:
                raise AgentXValidationError(resp.text)
            if (
                resp.status_code in _RETRYABLE_STATUS
                and retry
                and attempt < _MAX_RETRIES - 1
            ):
                logger.debug(
                    "Retryable status %d (attempt %d)", resp.status_code, attempt + 1
                )
                last_exc = AgentXEvaluationsError(
                    f"HTTP {resp.status_code}", status_code=resp.status_code
                )
                continue
            if not resp.ok:
                raise AgentXEvaluationsError(
                    f"HTTP {resp.status_code}: {resp.text}", status_code=resp.status_code
                )
            try:
                return resp.json()
            except Exception:
                return resp.text
        raise AgentXEvaluationsError(f"Request failed after retries: {last_exc}")

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def list_models(self, provider: Optional[str] = None) -> List[ModelInfo]:
        """List the LLM models AgentX supports - the same set selectable for
        the Sovereignty & Portability Index. Pass ``provider`` (e.g. "Google")
        to filter."""
        params = {"provider": provider} if provider else None
        data = self._request("GET", "/models", params=params)
        items = data if isinstance(data, list) else data.get("models", [])
        return [ModelInfo(**m) for m in items]

    # ------------------------------------------------------------------
    # Dataset endpoints
    # ------------------------------------------------------------------

    def create_dataset(self, payload: dict) -> Dataset:
        data = self._request("POST", "/datasets", json=self._with_workspace(payload))
        return Dataset(**data)

    def list_datasets(self) -> List[Dataset]:
        data = self._request("GET", "/datasets", params=self._workspace_params())
        return [
            Dataset(**d)
            for d in (data if isinstance(data, list) else data.get("datasets", []))
        ]

    def get_dataset(self, dataset_id: str) -> Dataset:
        data = self._request(
            "GET", f"/datasets/{dataset_id}", params=self._workspace_params()
        )
        return Dataset(**data)

    # ------------------------------------------------------------------
    # Evaluation Settings endpoints - standalone grading config, reusable
    # across datasets.
    # ------------------------------------------------------------------

    def create_evaluation_settings(self, payload: dict) -> EvaluationSettings:
        data = self._request(
            "POST", "/evaluation-settings", json=self._with_workspace(payload)
        )
        return EvaluationSettings(**data)

    def list_evaluation_settings(self) -> List[EvaluationSettings]:
        data = self._request(
            "GET", "/evaluation-settings", params=self._workspace_params()
        )
        return [
            EvaluationSettings(**e)
            for e in (
                data if isinstance(data, list) else data.get("evaluationSettings", [])
            )
        ]

    def get_evaluation_settings(self, evaluation_settings_id: str) -> EvaluationSettings:
        data = self._request(
            "GET",
            f"/evaluation-settings/{evaluation_settings_id}",
            params=self._workspace_params(),
        )
        return EvaluationSettings(**data)

    # ------------------------------------------------------------------
    # Prompt registry endpoints - see agentx.evaluations.prompts.PromptClient for the concept
    # (the external-agent analog to native autotune). Deliberately read-mostly: no publish here,
    # a new version only ever comes from the dashboard's human-approved propose/publish flow.
    # ------------------------------------------------------------------

    def create_prompt(self, payload: dict) -> Prompt:
        data = self._request("POST", "/prompts", json=self._with_workspace(payload))
        return Prompt(**data)

    def list_prompts(self) -> List[Prompt]:
        data = self._request("GET", "/prompts", params=self._workspace_params())
        return [Prompt(**p) for p in (data if isinstance(data, list) else data.get("prompts", []))]

    def get_prompt(self, name: str, version: Optional[int] = None) -> Prompt:
        # `name` doubles as an id: the backend route tries a name match first, then falls back to
        # an id match (see engine's getPromptRowByNameOrId), so callers can pass either.
        params = self._workspace_params() or {}
        if version is not None:
            params = {**params, "version": version}
        data = self._request("GET", f"/prompts/{name}", params=params or None)
        return Prompt(**data)

    # ------------------------------------------------------------------
    # Run endpoints
    # ------------------------------------------------------------------

    def init_run(
        self,
        dataset_id: str,
        subject: EvaluationSubject,
        python_version: Optional[str] = None,
        scorer_id: Optional[str] = None,
        evaluation_settings_id: Optional[str] = None,
        split: Optional[str] = None,
    ) -> EvaluationRun:
        """``scorer_id`` names the LLM Judge Scorer grading this run (its id doubles as the
        wire's ``evaluationSettingsId``). ``evaluation_settings_id`` is the pre-consolidation
        alias and keeps working. ``split`` records the named case subset this run covers."""
        from agentx.version import VERSION

        grader_id = _resolve_scorer_id(scorer_id, evaluation_settings_id)

        payload = {
            "datasetId": dataset_id,
            "evaluationSubject": subject.model_dump(by_alias=True, exclude_none=True),
            "runSource": "sdk",
            "sdk": {
                "name": SDK_NAME,
                "version": VERSION,
                "runnerVersion": "1",
                "pythonVersion": python_version or _python_version(),
            },
        }
        if grader_id:
            payload["evaluationSettingsId"] = grader_id
        if split:
            payload["split"] = split
        data = self._request("POST", "/runs", json=self._with_workspace(payload))
        return EvaluationRun(**data)

    def append_results(
        self, run_id: str, batch_id: str, results: List[EvaluationResult]
    ) -> BatchAppendResponse:
        payload = {
            "batchId": batch_id,
            "results": [_result_to_payload(r) for r in results],
        }
        data = self._request("POST", f"/runs/{run_id}/results", json=payload)
        return BatchAppendResponse(**data)

    def finalize_run(self, run_id: str) -> Dict[str, Any]:
        return self._request(
            "POST", f"/runs/{run_id}/finalize", json={"status": "completed"}
        )

    def gate_run(
        self,
        run_id: str,
        *,
        fail_under: Optional[float] = None,
        no_regression: bool = False,
        tolerance: Optional[float] = None,
        record: bool = True,
        caller: Optional[str] = "sdk",
    ) -> Dict[str, Any]:
        # CI gate (self-host): pass/fail a finalized run against an absolute rating floor and/or
        # the dataset's previous completed run. Recorded into gate history by default (the
        # dashboard's CI page lists these); pass record=False for a preview that leaves no trace.
        # `caller` is a free label shown in that history ("sdk", "github-actions", ...). See
        # EvaluationRunContext.gate() for the CI-facing wrapper with printed verdicts.
        params: Dict[str, Any] = {}
        if fail_under is not None:
            params["failUnder"] = fail_under
        if no_regression:
            params["noRegression"] = "true"
        if tolerance is not None:
            params["tolerance"] = tolerance
        if record:
            params["record"] = "true"
            if caller:
                params["caller"] = caller
        return self._request("GET", f"/runs/{run_id}/gate", params=params)

    def analyze_run(
        self,
        run_id: str,
        mode: Optional[str] = None,
        quality_mode: Optional[str] = None,
        judges: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Starts the durable analysis job and returns immediately (e.g. {"jobId": ..., "status":
        # "pending"}); poll get_analysis_status() until it reaches a terminal status, then call
        # get_report(). mode/quality_mode/judges mirror the dashboard's AnalyzeEvaluationRequest.
        #
        # On self-host the fallback route runs the analysis synchronously and returns only when
        # it is done, so the caller's poll loop sees a terminal status on its first check. The
        # request is therefore given the full analysis timeout and, critically, no retries.
        payload: Dict[str, Any] = {}
        if mode is not None:
            payload["mode"] = mode
        if quality_mode is not None:
            payload["qualityMode"] = quality_mode
        if judges is not None:
            payload["judges"] = [{"model": m} for m in judges]

        if not self._analysis_on_dashboard_router:
            try:
                return self._request(
                    "POST", f"/runs/{run_id}/analyze", json=payload, timeout=30
                )
            except AgentXEvaluationsError as exc:
                if not self._note_missing_analysis_route(exc, "analyze"):
                    raise

        return self._request(
            "POST",
            f"/evaluate/analyze/{run_id}",
            base=self._api_root,
            json=payload,
            timeout=_SELF_HOST_ANALYZE_TIMEOUT,
            retry=False,
        )

    def get_analysis_status(self, run_id: str) -> AnalysisStatus:
        if not self._analysis_on_dashboard_router:
            try:
                return AnalysisStatus(
                    **self._request("GET", f"/runs/{run_id}/analyze-status")
                )
            except AgentXEvaluationsError as exc:
                if not self._note_missing_analysis_route(exc, "analyze-status"):
                    raise

        data = self._request(
            "GET", f"/evaluate/analyze/{run_id}/status", base=self._api_root
        )
        return AnalysisStatus(**data)

    def get_run(self, run_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/runs/{run_id}")

    def get_report(self, run_id: str) -> Report:
        if not self._analysis_on_dashboard_router:
            try:
                return Report(**self._request("GET", f"/runs/{run_id}/report"))
            except AgentXEvaluationsError as exc:
                if not self._note_missing_analysis_route(exc, "report"):
                    raise

        return self._report_from_dashboard(run_id)

    def get_missing_results(self, run_id: str) -> List[Dict[str, Any]]:
        data = self._request("GET", f"/runs/{run_id}/missing-results")
        return data if isinstance(data, list) else data.get("missing", [])

    def get_submitted_keys(self, run_id: str) -> List[str]:
        """Idempotency keys this run has already accepted - what execute() uses to resume a
        crashed or interrupted run without re-running (and re-paying for) finished cases."""
        data = self._request("GET", f"/runs/{run_id}/missing-results")
        keys = data.get("submittedKeys", []) if isinstance(data, dict) else []
        return [k for k in keys if isinstance(k, str)]

    # ------------------------------------------------------------------
    # Self-host analysis fallback
    #
    # The self-host engine (AgentX-trace-eval) mounts two routers: the SDK's
    # /custom-agent-evaluations, and /evaluate for the dashboard. Its
    # /custom-agent-evaluations router implements the run lifecycle - runs, results,
    # finalize, gate - but not /analyze, /analyze-status or /report, which exist only on
    # /evaluate. Hosted AgentX serves all of them from the SDK's own router.
    #
    # So the three analysis calls try the SDK route first and fall back to the dashboard
    # route on a 404, which means hosted behaviour is byte-for-byte unchanged: it never
    # 404s, so it never falls back. The outcome is cached on the client so the probe costs
    # one request per process, not one per call.
    # ------------------------------------------------------------------

    @property
    def _api_root(self) -> str:
        """The API base with the ``/custom-agent-evaluations`` suffix removed."""
        suffix = "/custom-agent-evaluations"
        if self._base_url.endswith(suffix):
            return self._base_url[: -len(suffix)]
        return self._base_url

    def _note_missing_analysis_route(
        self, exc: AgentXEvaluationsError, route: str
    ) -> bool:
        """Return True if ``exc`` is the 404 that means "this engine is self-host".

        Only a 404 qualifies. Anything else - auth, validation, a 500, a dead connection -
        is a real failure on a route that does exist, and must propagate rather than be
        retried against a different endpoint that would mask it.
        """
        if exc.status_code != 404:
            return False
        if self._analysis_on_dashboard_router is None:
            logger.info(
                "%s is not served from %s; using the dashboard router at %s "
                "(self-host engine)",
                route,
                self._base_url,
                self._api_root,
            )
        self._analysis_on_dashboard_router = True
        return True

    def _report_from_dashboard(self, run_id: str) -> Report:
        """Assemble a Report from the dashboard's evaluation record.

        Self-host has no /report route; it returns the analysis nested inside the
        evaluation itself, under ``analysis.analysis`` with its statistics one level up.
        The field names already match the Report models, so this is a reshape, not a
        translation.
        """
        record = self._request("GET", f"/evaluate/{run_id}", base=self._api_root)
        envelope = record.get("analysis") or {}
        body = envelope.get("analysis") or {}

        if not envelope:
            raise AgentXEvaluationsError(
                f"Run {run_id} has no analysis to report. Nothing has analyzed it yet, "
                "or the analysis failed - call analyze_run() first."
            )

        dataset_id = record.get("datasetId")
        if isinstance(dataset_id, dict):  # populated reference, not a bare id
            dataset_id = dataset_id.get("_id") or dataset_id.get("id")

        return Report(
            runId=run_id,
            datasetId=dataset_id or "",
            status=envelope.get("status") or "completed",
            statistics=envelope.get("statistics"),
            **body,
        )

    # ------------------------------------------------------------------
    # Prompt improvement loop (examples -> propose -> publish). These ride the engine's
    # /evaluate dialect via _api_root(), same precedent get_report/_analysis already use for
    # routes that live on the dashboard router (self-host only).
    # ------------------------------------------------------------------

    def get_prompt_examples(self, prompt_id: str, window: Optional[str] = None) -> dict:
        params = {"window": window} if window else None
        return self._request(
            "GET", f"/evaluate/prompts/{prompt_id}/examples", base=self._api_root, params=params
        )

    def propose_prompt(self, prompt_id: str) -> dict:
        # One real judge call - no retry (see _request's retry note).
        return self._request(
            "POST", f"/evaluate/prompts/{prompt_id}/propose", base=self._api_root, timeout=180, retry=False
        )

    def publish_prompt_version(
        self, prompt_id: str, *, text: str, source: str = "proposed",
        reasoning: Optional[str] = None, based_on_version: Optional[int] = None,
    ) -> dict:
        payload: dict = {"text": text, "source": source}
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if based_on_version is not None:
            payload["basedOnVersion"] = based_on_version
        return self._request(
            "POST", f"/evaluate/prompts/{prompt_id}/versions", base=self._api_root, json=payload
        )

    # ------------------------------------------------------------------
    # Head-to-head (pairwise) judging. Rides the /evaluate dialect via _api_root(), same
    # precedent as get_report and the prompt loop above.
    # ------------------------------------------------------------------

    def compare_pairwise(
        self,
        run_a_id: str,
        run_b_id: str,
        *,
        criteria: Optional[str] = None,
        judge_model: Optional[str] = None,
        both_orders: bool = False,
    ) -> PairwiseComparison:
        """Ask a judge which of two runs answered each question better.

        Both runs must be of the same dataset. Absolute ratings answer "is this above the bar";
        this answers "did the change help", which is the question a diff between two runs is
        actually asking. The judge sees the two answers as "Answer 1"/"Answer 2" with the order
        alternating case by case, so it never learns which run is the candidate.

        ``both_orders=True`` judges every pair twice with the sides swapped. It doubles the judge
        cost and is the only real defense against position bias: a pair whose winner reverses is
        recorded as a tie, and the batch reports its ``flip_rate``.

        ``criteria`` and ``judge_model`` default to the dataset's own evaluation criteria and
        judge model, so a head-to-head grades on the same terms a normal run of it does.
        """
        payload: Dict[str, Any] = {"runAId": run_a_id, "runBId": run_b_id}
        if criteria:
            payload["criteria"] = criteria
        if judge_model:
            payload["judgeModel"] = judge_model
        if both_orders:
            payload["bothOrders"] = True
        response = self._request("POST", "/evaluate/runs/pairwise", json=payload, base=self._api_root)
        return PairwiseComparison(**response["comparison"])

    def get_pairwise(self, batch_id: str) -> PairwiseComparison:
        """Read back a stored head-to-head by its batch id."""
        response = self._request("GET", f"/evaluate/runs/pairwise/{batch_id}", base=self._api_root)
        return PairwiseComparison(**response["comparison"])

    def list_pairwise(
        self, *, run_a_id: Optional[str] = None, run_b_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Summaries of past head-to-heads, newest first, optionally narrowed to one run."""
        params: Dict[str, Any] = {}
        if run_a_id:
            params["runAId"] = run_a_id
        if run_b_id:
            params["runBId"] = run_b_id
        response = self._request(
            "GET", "/evaluate/runs/pairwise", params=params or None, base=self._api_root
        )
        return response.get("comparisons", [])

    # ------------------------------------------------------------------
    # Tool schema registry (same version-scoped propose/publish loop as prompts)
    # ------------------------------------------------------------------

    def list_tool_schemas(self) -> List[dict]:
        data = self._request("GET", "/evaluate/tool-schemas", base=self._api_root)
        return data.get("toolSchemas", []) if isinstance(data, dict) else data

    def create_tool_schema(self, *, name: str, definition: str, description: Optional[str] = None) -> dict:
        payload: dict = {"name": name, "definition": definition}
        if description is not None:
            payload["description"] = description
        return self._request("POST", "/evaluate/tool-schemas", base=self._api_root, json=payload)

    def get_tool_schema_examples(self, tool_schema_id: str, window: Optional[str] = None) -> dict:
        params = {"window": window} if window else None
        return self._request(
            "GET", f"/evaluate/tool-schemas/{tool_schema_id}/examples", base=self._api_root, params=params
        )

    def propose_tool_schema(self, tool_schema_id: str, window: Optional[str] = None) -> dict:
        payload = {"window": window} if window else {}
        return self._request(
            "POST", f"/evaluate/tool-schemas/{tool_schema_id}/propose",
            base=self._api_root, json=payload, timeout=180, retry=False,
        )

    def publish_tool_schema_version(
        self, tool_schema_id: str, *, definition: str, source: str = "proposed",
        reasoning: Optional[str] = None, based_on_version: Optional[int] = None,
    ) -> dict:
        payload: dict = {"definition": definition, "source": source}
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if based_on_version is not None:
            payload["basedOnVersion"] = based_on_version
        return self._request(
            "POST", f"/evaluate/tool-schemas/{tool_schema_id}/versions", base=self._api_root, json=payload
        )

    # ------------------------------------------------------------------
    # CI gate history + conversation simulation (self-host)
    # ------------------------------------------------------------------

    def list_gates(self) -> List[dict]:
        """Recorded CI gate verdicts, newest first - the dashboard's CI Gates history."""
        data = self._request("GET", "/evaluate/ci/gates", base=self._api_root)
        return data.get("gates", []) if isinstance(data, dict) else data

    def simulate_conversation(
        self, *, model: str, system_prompt: str, persona: str, goal: str,
        max_turns: int = 5, tools: Optional[List[dict]] = None, agent_name: Optional[str] = None,
    ) -> dict:
        """Run a persona-driven multi-turn simulation against a prompt (the Playground's
        "Simulate conversation"). Blocking and judge-billed: one LLM call per simulated turn
        plus the closing judgment, so expect it to take tens of seconds."""
        payload: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}],
            "persona": persona,
            "goal": goal,
            "maxTurns": max_turns,
        }
        if tools:
            payload["tools"] = tools
        if agent_name:
            payload["agentName"] = agent_name
        return self._request(
            "POST", "/evaluate/playground/simulate", base=self._api_root, json=payload,
            timeout=600, retry=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_payload(r: EvaluationResult) -> dict:
    d = r.model_dump(by_alias=True, exclude_none=True)
    # Rename snake_case fields the model may have stored locally
    d["caseId"] = d.pop("case_id", d.get("caseId"))
    d["questionIndex"] = d.pop("question_index", d.get("questionIndex"))
    d["runNumber"] = d.pop("run_number", d.get("runNumber"))
    d["idempotencyKey"] = d.pop("idempotency_key", d.get("idempotencyKey"))
    d["traceId"] = d.pop("trace_id", d.get("traceId"))
    return {k: v for k, v in d.items() if v is not None}


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}"
