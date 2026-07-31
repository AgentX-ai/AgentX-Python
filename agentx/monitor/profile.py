from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agentx.monitor.models import MonitorProfile

if TYPE_CHECKING:
    from agentx.monitor.client import MonitorClient

logger = logging.getLogger(__name__)


class MonitorProfileClient:
    """Thin wrapper surfaced as ``client.monitor.profile``: get/update a single agent's Monitor
    coverage and detection settings (coverage mode, sample rate, retention, redaction, approval
    policy, and threshold_overrides, e.g. the built-in "Latency regression" pattern's threshold).

    Unlike patterns/signals, a profile is scoped to one agent per call, since that mirrors how
    the dashboard's per-agent monitoring settings dialog works: ``get(agent_id)`` returns ``None``
    for an agent that's never been configured (still on platform defaults), and ``update(agent_id,
    ...)`` upserts one.
    """

    def __init__(self, client: "MonitorClient"):
        self._client = client

    def get(self, agent_id: str) -> Optional[MonitorProfile]:
        return self._client.get_profile(agent_id)

    def update(
        self,
        agent_id: str,
        *,
        enabled: Optional[bool] = None,
        failure_detection_enabled: Optional[bool] = None,
        info_detection_enabled: Optional[bool] = None,
        coverage_mode: Optional[str] = None,
        sample_rate: Optional[float] = None,
        channels: Optional[List[str]] = None,
        dataset_id: Optional[str] = None,
        threshold_overrides: Optional[Dict[str, Any]] = None,
        retention_days: Optional[int] = None,
        redaction_mode: Optional[str] = None,
        approval_policy: Optional[Dict[str, str]] = None,
    ) -> MonitorProfile:
        """Update (and enable, if not already) this agent's Monitor profile. Only fields passed
        here are changed; everything else on the existing profile is left as is.

        Example, overriding the built-in "Latency regression" pattern's threshold for one agent::

            client.monitor.profile.update("agent_123", threshold_overrides={"latencyMs": 15000})
        """
        payload: Dict[str, Any] = {
            "enabled": enabled,
            "failureDetectionEnabled": failure_detection_enabled,
            "infoDetectionEnabled": info_detection_enabled,
            "coverageMode": coverage_mode,
            "sampleRate": sample_rate,
            "channels": channels,
            "datasetId": dataset_id,
            "thresholdOverrides": threshold_overrides,
            "retentionDays": retention_days,
            "redactionMode": redaction_mode,
            "approvalPolicy": approval_policy,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        return self._client.update_profile(agent_id, payload)
