from __future__ import annotations

import logging
from typing import List, Optional, TYPE_CHECKING

from agentx.monitor.models import MonitorSignal

if TYPE_CHECKING:
    from agentx.monitor.client import MonitorClient

logger = logging.getLogger(__name__)


class MonitorSignalClient:
    """Thin wrapper surfaced as ``client.monitor.signals``. Read-only: a signal is produced by
    checking a trace against patterns (see ``tracer.trace(..., monitor=True, pattern_ids=[...])``),
    not something the SDK creates directly."""

    def __init__(self, client: "MonitorClient"):
        self._client = client

    def list(
        self,
        polarity: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[MonitorSignal]:
        """List this workspace's signals, most recently seen first.

        `polarity` defaults server-side to failures only ("proper", the healthy tally, is
        excluded); pass ``polarity="all"`` to include both, or ``"proper"``/``"failure"`` to
        narrow to one kind. `limit` is capped at 100 server-side.
        """
        return self._client.list_signals(
            polarity=polarity, status=status, severity=severity, agent_id=agent_id, limit=limit
        )

    def get(self, signal_id: str) -> MonitorSignal:
        return self._client.get_signal(signal_id)
