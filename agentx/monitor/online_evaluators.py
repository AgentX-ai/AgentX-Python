from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agentx.monitor.models import MonitorOnlineEvaluator, OnlineEvaluatorRatingPoint, OnlineEvaluatorEvent

if TYPE_CHECKING:
    from agentx.monitor.client import MonitorClient

logger = logging.getLogger(__name__)


class MonitorOnlineEvaluatorBuilder:
    """Fluent builder for creating an online evaluator: continuous LLM-judge scoring of a
    sample of live production traffic, distinct from a MonitorPattern's rule matching.
    ``evaluation_settings_id`` must reference an existing Evaluator config (criteria, judge
    prompt, judge model), the same config datasets/Evaluate runs use, see
    ``client.evaluations.settings.builder(...)``.

    Note: an online evaluator is the ONLINE profile of an **LLM Judge Scorer** - the unified
    entity at ``client.monitor.judge_scorers``. Strictly one profile per config since the
    unification: binding a config that is already another evaluator's profile transparently
    binds a fresh copy of it instead (the response's ``evaluationSettingsId`` is the copy).
    This surface keeps working unchanged; prefer ``judge_scorers`` for new code."""

    def __init__(
        self,
        client: "MonitorClient",
        name: str,
        evaluation_settings_id: str,
        sample_rate: float = 0.1,
        scope_mode: str = "all",
        agent_ids: Optional[List[str]] = None,
        enabled: bool = True,
        alert_threshold: Optional[float] = 5,
        severity: str = "medium",
        scope: str = "trace",
        idle_seconds: int = 120,
    ):
        self._client = client
        self._payload: Dict[str, Any] = {
            "name": name,
            "evaluationSettingsId": evaluation_settings_id,
            # Every check here is a real LLM call against your own API key: keep this low
            # unless you want to score every trace.
            "sampleRate": sample_rate,
            "scopeMode": scope_mode,
            "agentIds": agent_ids or [],
            "enabled": enabled,
            # A score below this raises/updates a Signal, same triage surface a failing
            # MonitorPattern already lands on. None scores without ever raising one.
            "alertThreshold": alert_threshold,
            "severity": severity,
            # "trace" (default) judges each sampled trace at ingest. "session" (self-host only)
            # judges whole conversations once they've been idle for idle_seconds, re-judging if
            # the conversation resumes - see the engine's idle-session sweep.
            "scope": scope,
            "idleSeconds": idle_seconds,
        }

    def publish(self) -> MonitorOnlineEvaluator:
        logger.info("Publishing online evaluator '%s'", self._payload["name"])
        return self._client.create_online_evaluator(self._payload)


class MonitorOnlineEvaluatorClient:
    """Thin wrapper surfaced as ``client.monitor.online_evaluators``.

    Note: an online evaluator is the ONLINE (live-traffic) profile of an **LLM Judge Scorer** -
    the unified entity at ``client.monitor.judge_scorers``, which also carries the judge rubric
    and the offline profile. This client keeps working unchanged; prefer ``judge_scorers`` for
    new code so both profiles live in one place."""

    def __init__(self, client: "MonitorClient"):
        self._client = client
        # Soft deprecation: hidden by default (DeprecationWarning), visible under -W or pytest.
        warnings.warn(
            "client.monitor.online_evaluators is the legacy view of an LLM Judge Scorer's "
            "online profile; prefer client.monitor.judge_scorers, which manages the judge "
            "rubric, offline profile, and online profile as one entity.",
            DeprecationWarning,
            stacklevel=3,
        )

    def builder(
        self,
        name: str,
        evaluation_settings_id: str,
        sample_rate: float = 0.1,
        scope_mode: str = "all",
        agent_ids: Optional[List[str]] = None,
        enabled: bool = True,
        alert_threshold: Optional[float] = 5,
        severity: str = "medium",
        scope: str = "trace",
        idle_seconds: int = 120,
    ) -> MonitorOnlineEvaluatorBuilder:
        return MonitorOnlineEvaluatorBuilder(
            self._client,
            name=name,
            evaluation_settings_id=evaluation_settings_id,
            sample_rate=sample_rate,
            scope_mode=scope_mode,
            agent_ids=agent_ids,
            enabled=enabled,
            alert_threshold=alert_threshold,
            severity=severity,
            scope=scope,
            idle_seconds=idle_seconds,
        )

    def get(self, evaluator_id: str) -> MonitorOnlineEvaluator:
        return self._client.get_online_evaluator(evaluator_id)

    def list(self) -> List[MonitorOnlineEvaluator]:
        return self._client.list_online_evaluators()

    def update(self, evaluator_id: str, **fields: Any) -> MonitorOnlineEvaluator:
        """Partial update: pass only the fields you want to change, e.g.
        ``client.monitor.online_evaluators.update(id, enabled=False)`` to pause one, or
        ``sample_rate=0.25`` to change its sampling. Field names match the builder's
        (``evaluation_settings_id``, ``sample_rate``, ``scope_mode``, ``agent_ids``, ``enabled``,
        ``alert_threshold``, ``severity``, ``scope``, ``idle_seconds``). Pass
        ``alert_threshold=None`` to stop this evaluator from ever raising a Signal;
        ``scope="session"`` (self-host only) switches it to judging whole idle conversations.
        """
        alias_map = {
            "evaluation_settings_id": "evaluationSettingsId",
            "sample_rate": "sampleRate",
            "scope_mode": "scopeMode",
            "agent_ids": "agentIds",
            "alert_threshold": "alertThreshold",
            "idle_seconds": "idleSeconds",
        }
        payload = {alias_map.get(key, key): value for key, value in fields.items()}
        return self._client.update_online_evaluator(evaluator_id, payload)

    def delete(self, evaluator_id: str) -> None:
        self._client.delete_online_evaluator(evaluator_id)

    def calibration(self, evaluator_id: str, window: str = "7d") -> dict:
        """How often this evaluator's verdicts agreed with recorded ground truth (outcomes,
        user feedback, human re-scores) in the window - including the disagreement cases a
        tune() proposal would rewrite from."""
        return self._client.get_online_evaluator_calibration(evaluator_id, window)

    def tune(self, evaluator_id: str, window: str = "7d") -> dict:
        """Judge-written rewrite of this evaluator's own criteria, grounded in calibration
        disagreements. Returns the proposal (criteria + reasoning + changes); publishes nothing."""
        return self._client.propose_online_evaluator_tuning(evaluator_id, window)

    def validate_tuning(self, evaluator_id: str, criteria: dict, window: str = "7d") -> dict:
        """Exact re-judging: candidate criteria re-judge the cases the current criteria got
        wrong plus a control set they got right, measured against recorded reality. `criteria`
        is {acceptanceCriteria, rejectionCriteria, evaluationCriteria} from tune()."""
        return self._client.validate_online_evaluator_tuning(evaluator_id, criteria, window)

    def publish_tuning(
        self, evaluator_id: str, criteria: dict, *, validation: Optional[dict] = None, force: bool = False
    ) -> dict:
        """Publish tuned criteria onto the evaluator's config (the human-approval step).
        Pass ``validation`` (the ``validate_tuning`` result) - the engine refuses an unvalidated
        publish, and a ``regressed`` verdict, unless ``force=True``."""
        return self._client.publish_online_evaluator_tuning(evaluator_id, criteria, validation=validation, force=force)

    def ratings(self, evaluator_id: str, window: str = "7d") -> List[OnlineEvaluatorRatingPoint]:
        """Bucketed average-rating-over-time for this evaluator. ``window`` is one of
        ``"24h"``, ``"7d"``, ``"30d"``."""
        return self._client.get_online_evaluator_ratings(evaluator_id, window)

    def events(self, evaluator_id: str, window: str = "7d") -> List[OnlineEvaluatorEvent]:
        """Individually scored traces behind the ratings series, worst-rated first and capped
        (see the dashboard's Online Evaluators tab for the same view), lets a low point on the
        ratings series be traced back to exactly which conversation(s) caused it and why."""
        return self._client.get_online_evaluator_events(evaluator_id, window)
