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
