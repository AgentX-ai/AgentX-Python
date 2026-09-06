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
        online evaluator (``online-eval:<id>``) or a session-scoped scorer group
        (``scorer-group:<id>``)."""
        return self._client.list_session_scores(session_id)

    def run_sweep(self) -> dict:
        """Trigger the idle-session sweep once (normally automatic, every minute) - scores
        idle multi-turn sessions with every enabled session-scoped evaluator and scorer
        group. Returns ``{"judged": n}``."""
        return self._client.run_session_sweep()
