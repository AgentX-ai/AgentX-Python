from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from agentx.monitor.models import (
    MonitorPattern,
    MonitorProfile,
    MonitorSignal,
    MonitorOnlineEvaluator,
    OnlineEvaluatorRatingPoint,
    OnlineEvaluatorEvent,
)

logger = logging.getLogger(__name__)

from agentx.util import _DEFAULT_API_BASE as _UTIL_API_BASE

SDK_NAME = "agentx-python"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 2.0, 4.0]


class AgentXMonitorError(Exception):
    pass


class AgentXAuthError(AgentXMonitorError):
    pass


class AgentXValidationError(AgentXMonitorError):
    pass


class CalibrationSummary(dict):
    """Judge Calibration numbers (dict subclass, so existing key access keeps working).
    Properties mirror the wire's exact camelCase keys."""

    @property
    def compared_count(self) -> int:
        return int(self.get("comparedCount") or 0)

    @property
    def agreement_rate(self):
        return self.get("agreementRate")

    @property
    def false_positive_rate(self):
        return self.get("falsePositiveRate")

    @property
    def false_negative_rate(self):
        return self.get("falseNegativeRate")

    @property
    def reported_count(self) -> int:
        return int(self.get("reportedCount") or 0)

    @property
    def review_label_count(self) -> int:
        return int(self.get("reviewLabelCount") or 0)


class MonitorClient:
    """Low-level HTTP client for the Monitor API (``/monitor``). Accessed via
    ``client.monitor`` on the top-level :class:`agentx.AgentX` instance; most callers
    should use ``client.monitor.patterns.builder(...)`` instead of this directly."""

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
        # Falls back to the caller's default workspace server-side when unset - mirrors
        # EvaluationsClient. Without this, pattern creation silently lands in whatever
        # workspace the API key's user defaults to, not the one the caller intended.
        self._workspace_id = workspace_id
        _api_base = (
            base_url or os.getenv("AGENTX_API_BASE_URL", _UTIL_API_BASE)
        ).rstrip("/")
        if not _api_base.endswith("/monitor"):
            _api_base = f"{_api_base}/monitor"
        self._base_url = _api_base
        self._session = requests.Session()
        self._session.headers.update(
            {
                "x-api-key": self._api_key,
                "Content-Type": "application/json",
                "User-Agent": f"{SDK_NAME}/{self._sdk_version}",
                "accept": "*/*",
            }
        )

        from agentx.monitor.patterns import MonitorPatternClient
        from agentx.monitor.signals import MonitorSignalClient
        from agentx.monitor.profile import MonitorProfileClient
        from agentx.monitor.online_evaluators import MonitorOnlineEvaluatorClient

        self.patterns = MonitorPatternClient(self)
        self.signals = MonitorSignalClient(self)
        from agentx.monitor.review_queue import ReviewQueueClient

        # The human-review queue (list / queue / label / dismiss) - what makes the
        # label-and-calibrate loop scriptable instead of dashboard-only.
        self.review_queue = ReviewQueueClient(self)
        from agentx.monitor.rules import MonitorRulesClient

        # Automation rules: route matching traffic into review / a dataset / a webhook.
        self.rules = MonitorRulesClient(self)
        from agentx.monitor.scorers import ScorersClient
        # Scorers-catalog administration as code: template enable/disable, code/external scorer
        # CRUD and dry runs - full parity with the dashboard's Scorers page (P1.3).
        # Handed this client's own resolved API root, never the process-global default, so a
        # second AgentX() with a different base_url can't re-point it (deep-dive bug #1).
        self.scorers = ScorersClient(api_key=api_key, base_url=self._api_root())
        from agentx.monitor.judge_scorers import JudgeScorersClient
        # The unified LLM Judge Scorer (rubric + offline/online profiles in one entity) - the
        # surface that matches the product; evaluations.settings and online_evaluators below
        # remain as its profile-level views.
        self.judge_scorers = JudgeScorersClient(api_key=api_key, base_url=self._api_root())
        self.profile = MonitorProfileClient(self)
        # Legacy view of an LLM Judge Scorer's online profile - constructed lazily so its
        # DeprecationWarning fires on first USE, not for every client that never touches it.
        self._online_evaluators: "MonitorOnlineEvaluatorClient | None" = None
        from agentx.monitor.sessions import MonitorSessionClient
        from agentx.monitor.agents import MonitorAgentClient
        self.sessions = MonitorSessionClient(self)
        self.agents = MonitorAgentClient(self)

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _with_workspace(self, payload: dict) -> dict:
        if self._workspace_id and not payload.get("workspaceId"):
            return {**payload, "workspaceId": self._workspace_id}
        return payload

    def _workspace_params(self) -> Optional[dict]:
        return {"workspaceId": self._workspace_id} if self._workspace_id else None

    def _api_root(self) -> str:
        """The API base with the ``/monitor`` suffix removed - for the handful of self-host
        routes that live on the engine's other routers (ingest sessions, agent-monitoring
        calibration/tuning/portability), same precedent EvaluationsClient._api_root sets."""
        suffix = "/monitor"
        if self._base_url.endswith(suffix):
            return self._base_url[: -len(suffix)]
        return self._base_url

    def _request(self, method: str, path: str, timeout: int = 30, base: Optional[str] = None, **kwargs) -> Any:
        url = f"{base or self._base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt, wait in enumerate([0.0] + _RETRY_BACKOFF):
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
            if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                logger.debug(
                    "Retryable status %d (attempt %d)", resp.status_code, attempt + 1
                )
                last_exc = AgentXMonitorError(f"HTTP {resp.status_code}")
                continue
            if not resp.ok:
                raise AgentXMonitorError(f"HTTP {resp.status_code}: {resp.text}")
            try:
                return resp.json()
            except Exception:
                return resp.text
        raise AgentXMonitorError(f"Request failed after retries: {last_exc}")

    # ------------------------------------------------------------------
    # Pattern endpoints
    # ------------------------------------------------------------------

    def create_pattern(self, payload: dict) -> MonitorPattern:
        data = self._request("POST", "/patterns", json=self._with_workspace(payload))
        return MonitorPattern(**data["pattern"])

    def list_patterns(self) -> List[MonitorPattern]:
        data = self._request("GET", "/patterns", params=self._workspace_params())
        return [MonitorPattern(**p) for p in data.get("patterns", [])]

    def get_pattern(self, pattern_id: str) -> MonitorPattern:
        data = self._request(
            "GET", f"/patterns/{pattern_id}", params=self._workspace_params()
        )
        return MonitorPattern(**data["pattern"])

    # ------------------------------------------------------------------
    # Online evaluator endpoints
    # ------------------------------------------------------------------

    def create_online_evaluator(self, payload: dict) -> MonitorOnlineEvaluator:
        data = self._request("POST", "/online-evaluators", json=self._with_workspace(payload))
        return MonitorOnlineEvaluator(**data["evaluator"])

    @property
    def online_evaluators(self) -> "MonitorOnlineEvaluatorClient":
        if self._online_evaluators is None:
            from agentx.monitor.online_evaluators import MonitorOnlineEvaluatorClient

            self._online_evaluators = MonitorOnlineEvaluatorClient(self)
        return self._online_evaluators

    def list_online_evaluators(self) -> List[MonitorOnlineEvaluator]:
        data = self._request("GET", "/online-evaluators", params=self._workspace_params())
        return [MonitorOnlineEvaluator(**e) for e in data.get("evaluators", [])]

    def get_online_evaluator(self, evaluator_id: str) -> MonitorOnlineEvaluator:
        data = self._request(
            "GET", f"/online-evaluators/{evaluator_id}", params=self._workspace_params()
        )
        return MonitorOnlineEvaluator(**data["evaluator"])

    def update_online_evaluator(self, evaluator_id: str, payload: dict) -> MonitorOnlineEvaluator:
        data = self._request(
            "PUT", f"/online-evaluators/{evaluator_id}", json=self._with_workspace(payload)
        )
        return MonitorOnlineEvaluator(**data["evaluator"])

    def delete_online_evaluator(self, evaluator_id: str) -> None:
        self._request(
            "DELETE", f"/online-evaluators/{evaluator_id}", params=self._workspace_params()
        )

    def get_online_evaluator_ratings(self, evaluator_id: str, window: str) -> List[OnlineEvaluatorRatingPoint]:
        params = {**(self._workspace_params() or {}), "window": window}
        data = self._request("GET", f"/online-evaluators/{evaluator_id}/ratings", params=params)
        return [OnlineEvaluatorRatingPoint(**p) for p in data.get("points", [])]

    def get_online_evaluator_events(self, evaluator_id: str, window: str) -> List[OnlineEvaluatorEvent]:
        params = {**(self._workspace_params() or {}), "window": window}
        data = self._request("GET", f"/online-evaluators/{evaluator_id}/events", params=params)
        return [OnlineEvaluatorEvent(**e) for e in data.get("events", [])]

    # ------------------------------------------------------------------
    # KPI endpoint
    # ------------------------------------------------------------------

    def kpis(self, window: str = "7d") -> dict:
        """Production health metrics over a window ("24h", "7d", or "30d") - the same numbers
        the dashboard's Overview KPI strip shows: totalRuns, healthRate, failureRate,
        downvoteRate (share of end-user votes that were "down"), toolFailureRate, p95LatencyMs,
        plus deltas vs the prior window and the run-outcome breakdown."""
        return self._request("GET", "/kpis", params={"window": window})

    def metrics(
        self,
        window: str = "1d",
        *,
        agent: Optional[str] = None,
        model: Optional[str] = None,
        tool: Optional[str] = None,
        framework: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict:
        """The Monitor metrics grid's data over a window ("1h".."90d") - bucketed spans by kind,
        latency percentiles, tokens/cost, tool executions and failures, and platform attribution
        (`frameworks` window totals + per-bucket `byFramework` - the Platforms chart). Optional
        filters scope every number the way the dashboard's filter chips do; `framework` matches
        the platform label traces carry (see TRACING.md's Platform detection), with "other"
        selecting unlabeled traffic."""
        params = {"window": window}
        for key, value in (("agent", agent), ("model", model), ("tool", tool), ("framework", framework), ("status", status)):
            if value is not None:
                params[key] = value
        return self._request("GET", "/metrics", params=params)

    def topics(self, window: str = "7d") -> dict:
        """The Topics view's data over a window ("24h", "7d", "30d"): LLM-classified themes of
        sampled production traffic with per-topic counts and sentiment. Empty until Topics is
        enabled project-wide via ``set_topics(True)`` - classification spends one judge call
        per sampled trace, so it is off by default."""
        return self._request(
            "GET", "/agent-monitoring/topics",
            base=self._api_root(), params={"window": window},
        )

    def set_topics(self, enabled: bool, sample_rate: Optional[float] = None) -> dict:
        """Turn Topics classification on/off for the whole project (Platform Settings >
        Monitoring Defaults). ``sample_rate`` (0-1) optionally bounds what fraction of traffic
        is classified - each classified trace costs one judge call."""
        payload: Dict[str, Any] = {"topicsEnabled": enabled}
        if sample_rate is not None:
            payload["topicsSampleRate"] = sample_rate
        return self._request(
            "PUT", "/agent-monitoring/settings/monitoring-defaults",
            base=self._api_root(), json=payload,
        )

    def calibration(self, window: str = "7d") -> "CalibrationSummary":
        """Project-level judge calibration over a window ("24h", "7d", or "30d"): how often
        AgentX's own verdicts agreed with real-world ground truth reported later (ops outcomes
        via ``client.outcomes``, end-user downvotes, and human review labels). Returns the
        dashboard's Judge Calibration numbers with these exact keys: ``comparedCount``,
        ``agreementRate``, ``falsePositiveRate``, ``falseNegativeRate`` (plus
        ``reportedCount``/``reviewLabelCount``/``noVerdictCount``). Per-scorer calibration
        lives on ``client.monitor.judge_scorers.calibration(scorer_id)``."""
        return CalibrationSummary(
            self._request(
                "GET", "/agent-monitoring/calibration",
                base=self._api_root(), params={"window": window},
            )
        )

    # ------------------------------------------------------------------
    # Signal endpoints
    # ------------------------------------------------------------------

    def list_signals(
        self,
        polarity: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[MonitorSignal]:
        params = {**(self._workspace_params() or {})}
        if polarity is not None:
            params["polarity"] = polarity
        if status is not None:
            params["status"] = status
        if severity is not None:
            params["severity"] = severity
        if agent_id is not None:
            params["agentId"] = agent_id
        if limit is not None:
            params["limit"] = limit
        data = self._request("GET", "/signals", params=params)
        return [MonitorSignal(**s) for s in data.get("signals", [])]

    def get_signal(self, signal_id: str) -> MonitorSignal:
        data = self._request(
            "GET", f"/signals/{signal_id}", params=self._workspace_params()
        )
        return MonitorSignal(**data["signal"])

    # ------------------------------------------------------------------
    # Profile endpoints
    # ------------------------------------------------------------------

    def get_profile(self, agent_id: str) -> Optional[MonitorProfile]:
        data = self._request(
            "GET", f"/profiles/{agent_id}", params=self._workspace_params()
        )
        profile = data.get("profile")
        return MonitorProfile(**profile) if profile else None

    # ------------------------------------------------------------------
    # Agents (the engine's SDK-facing /agents router - self-host)
    # ------------------------------------------------------------------

    def list_agents(self) -> List[dict]:
        data = self._request("GET", "/agents", base=self._api_root())
        return data.get("agents", []) if isinstance(data, dict) else data

    def create_agent(self, name: str) -> dict:
        data = self._request("POST", "/agents", base=self._api_root(), json={"name": name})
        return data.get("agent", data) if isinstance(data, dict) else data

    # ------------------------------------------------------------------
    # Sessions (self-host)
    # ------------------------------------------------------------------

    def run_session_coherence_check(self, session_id: str) -> dict:
        """One judge call over the assembled session - the dashboard's "Check coherence"
        button. Raises AgentXMonitorError if the engine has no judge key configured."""
        data = self._request(
            "POST", f"/agent-monitoring/sessions/{session_id}/coherence-check",
            base=self._api_root(), timeout=180,
        )
        return data.get("score", data) if isinstance(data, dict) else data

    def list_session_spans(self, session_id: str) -> List[dict]:
        data = self._request("GET", f"/ingest/sessions/{session_id}/spans", base=self._api_root())
        return data.get("spans", []) if isinstance(data, dict) else data

    # ------------------------------------------------------------------
    # Model portability (self-host): replay a trace's input against other models
    # ------------------------------------------------------------------

    def run_model_portability(self, trace_id: str, model_ids: List[str]) -> dict:
        """Replay the trace's captured input against ``model_ids`` and judge each output -
        the dashboard trace detail's "Compare models" action. One LLM call per candidate
        plus judging, so expect tens of seconds."""
        return self._request(
            "POST", f"/agent-monitoring/traces/{trace_id}/portability",
            base=self._api_root(), json={"modelIds": model_ids}, timeout=300,
        )

    # ------------------------------------------------------------------
    # Judge tuning (self-host): calibrate an online evaluator against recorded
    # ground truth, then rewrite/validate/publish its criteria
    # ------------------------------------------------------------------

    def get_online_evaluator_calibration(self, evaluator_id: str, window: str = "7d") -> dict:
        return self._request(
            "GET", f"/agent-monitoring/online-evaluators/{evaluator_id}/calibration",
            base=self._api_root(), params={"window": window},
        )

    def propose_online_evaluator_tuning(self, evaluator_id: str, window: str = "7d") -> dict:
        data = self._request(
            "POST", f"/agent-monitoring/online-evaluators/{evaluator_id}/tune",
            base=self._api_root(), json={"window": window}, timeout=300,
        )
        return data.get("proposal", data) if isinstance(data, dict) else data

    def validate_online_evaluator_tuning(
        self, evaluator_id: str, criteria: dict, window: str = "7d"
    ) -> dict:
        return self._request(
            "POST", f"/agent-monitoring/online-evaluators/{evaluator_id}/tune/validate",
            base=self._api_root(), json={**criteria, "window": window}, timeout=600,
        )

    def publish_online_evaluator_tuning(
        self, evaluator_id: str, criteria: dict, *, validation: Optional[dict] = None, force: bool = False
    ) -> dict:
        # The engine gates publish on validation provenance (and refuses a measured regression)
        # unless forced - see judge_scorers.publish_tuning for the full story.
        payload = dict(criteria)
        if validation is not None:
            payload["validation"] = validation
        if force:
            payload["force"] = True
        return self._request(
            "POST", f"/agent-monitoring/online-evaluators/{evaluator_id}/tune/publish",
            base=self._api_root(), json=payload, timeout=60,
        )

    def update_profile(self, agent_id: str, payload: dict) -> MonitorProfile:
        data = self._request(
            "PUT", f"/profiles/{agent_id}", json=self._with_workspace(payload)
        )
        return MonitorProfile(**data["profile"])
