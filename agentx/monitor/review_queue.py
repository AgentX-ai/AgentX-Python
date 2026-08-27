from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agentx.monitor.client import MonitorClient

logger = logging.getLogger(__name__)


class ReviewQueueItem(dict):
    """Wire object for one human-review item (dict subclass so unknown fields round-trip)."""

    @property
    def id(self) -> str:
        return self["_id"]

    @property
    def trace_id(self) -> Optional[str]:
        return self.get("traceId")

    @property
    def status(self) -> Optional[str]:
        return self.get("status")

    @property
    def label(self) -> Optional[str]:
        return self.get("label")

    @property
    def judge_score_at_queue(self) -> Optional[float]:
        return self.get("judgeScoreAtQueue")


class ReviewQueueClient:
    """Surfaced as ``client.monitor.review_queue``: the human-review queue behind the dashboard's
    Review tab, scriptable - so the label-and-calibrate loop (sample traces, label them
    good/bad, optionally re-score) can run end to end from code. Labels feed judge calibration
    and become judge-tuning evidence.

    The engine refuses duplicates (409, a trace already pending) and a full queue (429, pending
    cap reached); both surface as raised errors with the engine's reason.
    """

    def __init__(self, client: "MonitorClient"):
        self._client = client

    def list(self, status: Optional[str] = None, source: Optional[str] = None, limit: int = 100) -> List[ReviewQueueItem]:
        """Queue items, newest first. ``status``: "pending" | "labeled" | "skipped" | "all"
        (server default: pending). ``source``: "manual" | "rule" | "all"."""
        params: Dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        if source is not None:
            params["source"] = source
        data = self._client._request("GET", "/agent-monitoring/review-queue", base=self._client._api_root(), params=params)
        return [ReviewQueueItem(item) for item in data.get("items", [])]

    def queue(self, trace_id: str, note: Optional[str] = None) -> ReviewQueueItem:
        """Send a trace to human review (the SDK-side twin of the dashboard's "Send to review")."""
        payload: Dict[str, Any] = {"traceId": trace_id, "source": "manual"}
        if note:
            payload["note"] = note
        data = self._client._request("POST", "/agent-monitoring/review-queue", base=self._client._api_root(), json=payload)
        return ReviewQueueItem(data.get("item", data))

    def label(
        self,
        item_id: str,
        label: str,
        *,
        corrected_score: Optional[float] = None,
        note: Optional[str] = None,
    ) -> ReviewQueueItem:
        """Record the human verdict on a queued item. ``label`` is "good" or "bad";
        ``corrected_score`` (0-10) optionally re-scores the judge's own rating for the trace -
        the pair that calibration consumes."""
        if label not in ("good", "bad"):
            raise ValueError('label must be "good" or "bad"')
        payload: Dict[str, Any] = {"label": label}
        if corrected_score is not None:
            payload["correctedScore"] = corrected_score
        if note is not None:
            payload["note"] = note
        data = self._client._request(
            "PATCH", f"/agent-monitoring/review-queue/{item_id}", base=self._client._api_root(), json=payload
        )
        return ReviewQueueItem(data.get("item", data))

    def dismiss(self, item_id: str) -> None:
        """Remove an item from the queue without a verdict (does not feed calibration)."""
        self._client._request("DELETE", f"/agent-monitoring/review-queue/{item_id}", base=self._client._api_root())
