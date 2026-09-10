"""Session helpers, surfaced as ``client.monitor.sessions`` (self-host)."""
from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from agentx.monitor.client import MonitorClient


class MonitorSessionClient:
    def __init__(self, client: "MonitorClient"):
        self._client = client

    def coherence_check(self, session_id: str) -> dict:
        """Judge the assembled multi-turn session for consistency/drift (one judge call).
        Returns the score row: rating, justification, spanCount, driftSpanId."""
        return self._client.run_session_coherence_check(session_id)

    def spans(self, session_id: str) -> List[dict]:
        """Every span in the session (roots and children), oldest first."""
        return self._client.list_session_spans(session_id)

    def scores(self, session_id: str) -> List[dict]:
        """Session-level verdicts, newest first. ``kind`` says who scored: a session-scoped
        online evaluator (``online-eval:<id>``), a session-scoped scorer group
        (``scorer-group:<id>``), or legacy ``"coherence"`` rows written before the Session
        Baseline Judge existed - branch defensively on unknown kinds."""
        return self._client.list_session_scores(session_id)

    def judge(self, session_id: str, evaluator_id: str, *, if_stale: bool = False) -> dict:
        """Judge one session with one session-scoped evaluator, now (one judge call).
        ``if_stale=True`` skips re-judging a session that was already scored since its
        last activity - the engine then answers ``{"skipped": True}`` without spending
        another judge call. Returns the score row (or that skip marker)."""
        data = self._client._request(
            "POST",
            f"/agent-monitoring/sessions/{session_id}/judge/{evaluator_id}",
            base=self._client._api_root(),
            timeout=120,
            retry=False,
            params={"ifStale": "true"} if if_stale else None,
        )
        return data.get("score", data) if isinstance(data, dict) else data

    def run_sweep(self) -> dict:
        """Trigger the idle-session sweep once (normally automatic, every minute) - scores
        idle multi-turn sessions with every enabled session-scoped evaluator and scorer
        group. Returns ``{"judged": n}``; the response may instead carry ``skipped: true``
        when another sweep is already in flight."""
        return self._client.run_session_sweep()
