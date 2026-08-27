"""Unit tests for client.monitor.review_queue (list / queue / label / dismiss) - wire-level,
no engine required. The engine-side contract is pinned by its review-queue routes."""

from typing import Any, Dict, List

import pytest

from agentx.monitor.review_queue import ReviewQueueClient, ReviewQueueItem


class FakeMonitorClient:
    def __init__(self, responses: List[Any]):
        self.calls: List[Dict[str, Any]] = []
        self._responses = responses

    def _api_root(self) -> str:
        return "http://engine:4700/api/v1"

    def _request(self, method: str, path: str, base: str = "", **kwargs: Any) -> Any:
        self.calls.append({"method": method, "path": path, "base": base, **kwargs})
        return self._responses.pop(0) if self._responses else {}


def test_list_hits_the_queue_with_filters():
    fake = FakeMonitorClient([{"items": [{"_id": "r1", "traceId": "t1", "status": "pending"}], "pending": 1}])
    items = ReviewQueueClient(fake).list(status="pending", source="rule", limit=25)  # type: ignore[arg-type]
    call = fake.calls[0]
    assert call["method"] == "GET"
    assert call["path"] == "/agent-monitoring/review-queue"
    assert call["base"] == "http://engine:4700/api/v1"
    assert call["params"] == {"limit": 25, "status": "pending", "source": "rule"}
    assert isinstance(items[0], ReviewQueueItem)
    assert items[0].id == "r1"
    assert items[0].trace_id == "t1"


def test_queue_sends_trace_and_note():
    fake = FakeMonitorClient([{"item": {"_id": "r2", "traceId": "t9"}}])
    item = ReviewQueueClient(fake).queue("t9", note="looks off")  # type: ignore[arg-type]
    call = fake.calls[0]
    assert call["method"] == "POST"
    assert call["json"] == {"traceId": "t9", "source": "manual", "note": "looks off"}
    assert item.id == "r2"


def test_label_validates_and_sends_the_calibration_pair():
    fake = FakeMonitorClient([{"item": {"_id": "r3", "label": "bad", "judgeScoreAtQueue": 8.0}}])
    client = ReviewQueueClient(fake)  # type: ignore[arg-type]
    item = client.label("r3", "bad", corrected_score=2, note="hallucinated policy")
    call = fake.calls[0]
    assert call["method"] == "PATCH"
    assert call["path"] == "/agent-monitoring/review-queue/r3"
    assert call["json"] == {"label": "bad", "correctedScore": 2, "note": "hallucinated policy"}
    assert item.label == "bad"
    assert item.judge_score_at_queue == 8.0

    with pytest.raises(ValueError):
        client.label("r3", "meh")


def test_dismiss_deletes_the_item():
    fake = FakeMonitorClient([""])
    ReviewQueueClient(fake).dismiss("r4")  # type: ignore[arg-type]
    call = fake.calls[0]
    assert call["method"] == "DELETE"
    assert call["path"] == "/agent-monitoring/review-queue/r4"


def test_registered_on_the_monitor_client():
    from agentx.monitor.client import MonitorClient

    monitor = MonitorClient(api_key="k", base_url="http://engine:4700/api/v1/monitor")
    assert isinstance(monitor.review_queue, ReviewQueueClient)
