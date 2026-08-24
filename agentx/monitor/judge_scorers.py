from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from agentx.util import api_base, get_headers

logger = logging.getLogger(__name__)

_SENTINEL: Any = object()


class AgentXJudgeScorersError(Exception):
    pass


class JudgeScorer(dict):
    """Wire object for the unified LLM Judge Scorer (dict subclass so unknown fields
    round-trip). Convenience properties expose the pieces scripts reach for most."""

    @property
    def id(self) -> str:
        return self["_id"]

    @property
    def name(self) -> str:
        return self["name"]

    @property
    def judge(self) -> Dict[str, Any]:
        return self.get("judge", {})

    @property
    def offline(self) -> Dict[str, Any]:
        return self.get("offline", {})

    @property
    def online(self) -> Optional[Dict[str, Any]]:
        return self.get("online")

    @property
    def online_profile_id(self) -> Optional[str]:
        online = self.online
        return online.get("profileId") if online else None


class JudgeScorersClient:
    """Surfaced as ``client.monitor.judge_scorers``: the unified LLM Judge Scorer.

    One scorer = one judge rubric (``judge``: acceptance/rejection/evaluation criteria, judge
    prompt, judge model) + two setting profiles:

    - ``offline`` - how dataset runs grade with it (repetitions, similarity metrics, code
      scorers, default/status). The scorer's ``id`` is exactly what
      ``client.evaluations.run(dataset_id, scorer_id=scorer.id)`` takes.
    - ``online`` - whether/how it scores live traffic (enabled, sample rate, scope,
      alert threshold). ``None`` means offline-only.

    Strictly one online profile per scorer. This supersedes the split between
    ``client.evaluations.settings`` (the offline half) and ``client.monitor.online_evaluators``
    (the online half) - both keep working, but this is the surface that matches the product.

    Example::

        scorer = client.monitor.judge_scorers.create(
            "Support quality",
            judge={"acceptanceCriteria": "Concrete, correct, cites the policy."},
            online={"enabled": True, "sampleRate": 0.2, "alertThreshold": 6},
        )
        client.evaluations.run(dataset_id, subject, scorer_id=scorer.id)
        cal = client.monitor.judge_scorers.calibration(scorer.id)
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._api_key = api_key
        # Captured once at construction so two clients with different bases can coexist.
        self._base_url = (base_url or api_base()).rstrip("/")

    def _request(self, method: str, path: str, json: Any = None, timeout: int = 60) -> Any:
        resp = requests.request(
            method,
            f"{self._base_url}/agent-monitoring{path}",
            headers={**get_headers(self._api_key), "Content-Type": "application/json"},
            json=json,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.reason)
            except ValueError:
                detail = resp.reason
            raise AgentXJudgeScorersError(f"Judge scorer request failed ({resp.status_code}): {detail}")
        return resp.json() if resp.text else {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def builder(
        self,
        name: str,
        description: Optional[str] = None,
        *,
        # Judge rubric (shared by every surface that runs this scorer)
        acceptance_criteria: Optional[str] = None,
        rejection_criteria: Optional[str] = None,
        evaluation_criteria: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        judge_model: Optional[str] = None,
        tool_context: Optional[str] = None,
        # Offline profile (dataset-run grading)
        number_of_requests: int = 1,
        vector_similarity: bool = False,
        jaccard_similarity: bool = False,
        bleu_score: bool = False,
        rouge_score: bool = False,
        similarity_model: Optional[str] = None,
        sovereignty_models: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, Any]] = None,
        code_scorers: Optional[List[Dict[str, Any]]] = None,
        is_default: bool = False,
        # Online profile (live traffic). live=True enables scoring immediately - every check is
        # a real judge call on your own key.
        live: bool = False,
        sample_rate: float = 0.1,
        scope: str = "trace",
        alert_threshold: Optional[float] = 5,
        severity: str = "medium",
        agent_ids: Optional[List[str]] = None,
        idle_seconds: int = 120,
    ) -> "JudgeScorerBuilder":
        """Snake_case builder with ``.publish()``, the unified successor of
        ``client.evaluations.settings.builder(...)`` - same offline fields (plus ``thresholds``,
        ``tool_context``) and, new here, the online profile in the same call. The scorer the
        builder publishes is one entity: its ``.id`` is what ``client.evaluations.run(...,
        scorer_id=...)`` takes, and its live profile is what online scoring keys on."""
        judge: Dict[str, Any] = {}
        for key, value in (
            ("acceptanceCriteria", acceptance_criteria),
            ("rejectionCriteria", rejection_criteria),
            ("evaluationCriteria", evaluation_criteria),
            ("judgePrompt", judge_prompt),
            ("judgeModel", judge_model),
            ("toolContext", tool_context),
        ):
            if value is not None:
                judge[key] = value
        offline: Dict[str, Any] = {"numberOfRequests": number_of_requests, "isDefault": is_default}
        if vector_similarity:
            offline["vectorSimilarity"] = (
                {"enabled": True, "model": similarity_model} if similarity_model else {"enabled": True}
            )
        if jaccard_similarity:
            offline["jaccardSimilarity"] = {"enabled": True}
        if bleu_score:
            offline["bleuScore"] = {"enabled": True}
        if rouge_score:
            offline["rougeScore"] = {"enabled": True}
        if sovereignty_models is not None:
            offline["sovereigntyIndex"] = {"enabled": bool(sovereignty_models), "models": sovereignty_models}
        if thresholds is not None:
            offline["thresholds"] = thresholds
        if code_scorers is not None:
            offline["codeScorers"] = code_scorers
        online: Optional[Dict[str, Any]] = None
        if live:
            online = {
                "enabled": True,
                "sampleRate": sample_rate,
                "scope": scope,
                "alertThreshold": alert_threshold,
                "severity": severity,
                "idleSeconds": idle_seconds,
            }
            if agent_ids:
                online["scopeMode"] = "selected"
                online["agentIds"] = agent_ids
        return JudgeScorerBuilder(self, name=name, description=description, judge=judge, offline=offline, online=online)

    def create(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        judge: Optional[Dict[str, Any]] = None,
        offline: Optional[Dict[str, Any]] = None,
        online: Optional[Dict[str, Any]] = None,
    ) -> JudgeScorer:
        """Create a scorer. ``online=None`` (default) creates it offline-only; pass e.g.
        ``online={"enabled": True, "sampleRate": 0.1, "alertThreshold": 5}`` to score live
        traffic from the start. Section dicts use the wire's camelCase keys."""
        payload: Dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if judge is not None:
            payload["judge"] = judge
        if offline is not None:
            payload["offline"] = offline
        if online is not None:
            payload["online"] = online
        return JudgeScorer(self._request("POST", "/judge-scorers", json=payload)["judgeScorer"])

    def get(self, scorer_id: str) -> JudgeScorer:
        return JudgeScorer(self._request("GET", f"/judge-scorers/{scorer_id}")["judgeScorer"])

    def list(self) -> List[JudgeScorer]:
        return [JudgeScorer(s) for s in self._request("GET", "/judge-scorers").get("judgeScorers", [])]

    def update(
        self,
        scorer_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        judge: Optional[Dict[str, Any]] = None,
        offline: Optional[Dict[str, Any]] = None,
        online: Any = _SENTINEL,
    ) -> JudgeScorer:
        """Sparse update: only the sections you pass change. ``online={...}`` upserts the
        online profile (this is how an offline-only scorer goes live); ``online=None``
        detaches it (refused for the built-in Session Baseline Judge)."""
        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if judge is not None:
            payload["judge"] = judge
        if offline is not None:
            payload["offline"] = offline
        if online is not _SENTINEL:
            payload["online"] = online
        return JudgeScorer(self._request("PUT", f"/judge-scorers/{scorer_id}", json=payload)["judgeScorer"])

    def delete(self, scorer_id: str) -> None:
        """Delete the scorer: rubric, version history, and online profile together.
        Irreversible; refused for the built-in Session Baseline Judge."""
        self._request("DELETE", f"/judge-scorers/{scorer_id}")

    # ------------------------------------------------------------------
    # Online-profile pass-throughs (calibration / tuning / ratings / events)
    # These endpoints key on the online profile's id, resolved here so callers only ever
    # handle the scorer id.
    # ------------------------------------------------------------------

    def _profile_id(self, scorer_id: str) -> str:
        scorer = self.get(scorer_id)
        profile_id = scorer.online_profile_id
        if not profile_id:
            raise AgentXJudgeScorersError(
                f"Judge scorer {scorer_id!r} has no online profile - calibration/tuning/ratings "
                "cover live-traffic scoring. Give it one first: "
                "update(scorer_id, online={'enabled': True})."
            )
        return profile_id

    def calibration(self, scorer_id: str, window: str = "7d") -> dict:
        """How this scorer's verdicts compare against recorded ground truth (triage
        corrections, outcomes, end-user votes) over the window."""
        return self._request("GET", f"/online-evaluators/{self._profile_id(scorer_id)}/calibration?window={window}")

    def tune(self, scorer_id: str, window: str = "7d") -> dict:
        """Propose a rewrite of the rubric from calibration disagreements (LLM call, slow)."""
        data = self._request(
            "POST", f"/online-evaluators/{self._profile_id(scorer_id)}/tune", json={"window": window}, timeout=300
        )
        # The wire wraps the proposal ({"proposal": {...}}); unwrap like the legacy client so
        # proposal["reasoning"] / the criteria fields are directly addressable.
        return data.get("proposal", data) if isinstance(data, dict) else data

    def validate_tuning(self, scorer_id: str, criteria: Dict[str, Any], window: str = "7d") -> dict:
        """Re-judge the disagreement + control cases with candidate criteria (LLM calls, slow)."""
        # The wire takes the criteria fields at the TOP level of the body, not nested.
        return self._request(
            "POST",
            f"/online-evaluators/{self._profile_id(scorer_id)}/tune/validate",
            json={**criteria, "window": window},
            timeout=600,
        )

    def publish_tuning(self, scorer_id: str, criteria: Dict[str, Any]) -> dict:
        """Write tuned criteria onto the scorer's rubric - it applies everywhere the scorer is
        used: online scoring, offline dataset runs, and the playground."""
        return self._request(
            "POST", f"/online-evaluators/{self._profile_id(scorer_id)}/tune/publish", json=dict(criteria)
        )

    def ratings(self, scorer_id: str, window: str = "7d") -> "List[OnlineEvaluatorRatingPoint]":
        """Bucketed average-rating-over-time for this scorer's live checks - same typed points
        the legacy online_evaluators client returns, so scripts migrate without shape changes."""
        from agentx.monitor.models import OnlineEvaluatorRatingPoint

        data = self._request("GET", f"/online-evaluators/{self._profile_id(scorer_id)}/ratings?window={window}")
        return [OnlineEvaluatorRatingPoint(**p) for p in data.get("points", [])]

    def events(self, scorer_id: str, window: str = "7d") -> "List[OnlineEvaluatorEvent]":
        """Individually scored traces behind the ratings series, worst-rated first - typed, same
        as the legacy online_evaluators client."""
        from agentx.monitor.models import OnlineEvaluatorEvent

        data = self._request("GET", f"/online-evaluators/{self._profile_id(scorer_id)}/events?window={window}")
        return [OnlineEvaluatorEvent(**e) for e in data.get("events", [])]


class JudgeScorerBuilder:
    """Assembled by :meth:`JudgeScorersClient.builder`; ``.publish()`` creates the scorer."""

    def __init__(
        self,
        client: "JudgeScorersClient",
        *,
        name: str,
        description: Optional[str],
        judge: Dict[str, Any],
        offline: Dict[str, Any],
        online: Optional[Dict[str, Any]],
    ):
        self._client = client
        self._name = name
        self._description = description
        self._judge = judge
        self._offline = offline
        self._online = online

    def publish(self) -> JudgeScorer:
        return self._client.create(
            self._name,
            description=self._description,
            judge=self._judge or None,
            offline=self._offline or None,
            online=self._online,
        )
