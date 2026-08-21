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
        """Get-or-create by name - idempotent, safe to re-run."""
        existing = next((a for a in self.list() if a.get("name") == name), None)
        return existing if existing is not None else self.create(name)
