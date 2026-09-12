"""Agent registry helpers, surfaced as ``client.monitor.agents`` (self-host).

Agents in self-host are lightweight name rows - normally auto-created the first time a trace
arrives under a name. These helpers exist for flows that need the agent id before any traffic
(e.g. enabling a monitoring profile up front).
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agentx.monitor.client import MonitorClient


class MonitorAgentClient:
    def __init__(self, client: "MonitorClient"):
        self._client = client

    def list(self) -> List[dict]:
        return self._client.list_agents()

    def create(self, name: str) -> dict:
        return self._client.create_agent(name)

    def ensure(self, name: str) -> dict:
        """Best-effort get-or-create - concurrent callers converge on the oldest row."""
        existing = next((a for a in self.list() if a.get("name") == name), None)
        if existing is not None:
            return existing
        created = self.create(name)
        # The engine's POST /agents always creates a new row, so two concurrent ensure()
        # calls can both create. Re-list and return the oldest row among same-name rows
        # (createdAt is ISO-8601, so lexicographic min is chronological min) - the same
        # row the engine's own name resolution picks - so every caller converges on the
        # same agent.
        matches = [a for a in self.list() if a.get("name") == name]
        if not matches:
            return created
        return min(matches, key=lambda a: str(a.get("createdAt") or ""))
