"""Unit tests for client.monitor.alert_rules - wire-level, no engine required. The engine-side
contract (metric evaluation, the firing/resolved lifecycle, channel delivery) is pinned by the
engine's alertRules integration suite; this pins the SDK's half: camelCase wire keys, the
no-retry posture on writes, argument validation, and the channel helpers."""

from typing import Any, Dict, List

import pytest

from agentx.monitor.alert_rules import AlertEvent, AlertRule, AlertRulesClient, email, pagerduty, slack, teams, webhook


class FakeMonitorClient:
    def __init__(self, responses: List[Any]):
        self.calls: List[Dict[str, Any]] = []
        self._responses = responses

    def _api_root(self) -> str:
        return "http://engine:4700/api/v1"

    def _request(self, method: str, path: str, base: str = "", **kwargs: Any) -> Any:
        self.calls.append({"method": method, "path": path, "base": base, **kwargs})
        return self._responses.pop(0) if self._responses else {}


def test_channel_helpers_build_wire_dicts():
    assert slack("https://hooks.slack.com/x") == {"kind": "slack", "target": "https://hooks.slack.com/x"}
    assert teams("https://t.example/hook") == {"kind": "teams", "target": "https://t.example/hook"}
    assert pagerduty("R0123456789abcdef0123456789abcdef") == {"kind": "pagerduty", "target": "R0123456789abcdef0123456789abcdef"}
    assert email("oncall@example.com") == {"kind": "email", "target": "oncall@example.com"}
    assert webhook("https://ops.example/agentx") == {"kind": "webhook", "target": "https://ops.example/agentx"}


def test_create_sends_camelcase_wire_without_retry():
    fake = FakeMonitorClient([{"rule": {"_id": "a1", "state": "ok", "enabled": True, "firedCount": 0}}])
    rule = AlertRulesClient(fake).create(  # type: ignore[arg-type]
        "Failure rate above 10%",
        metric="failureRate",
        operator="gt",
        threshold=0.1,
        window_minutes=15,
        channels=[slack("https://hooks.slack.com/x")],
        agent_id="agent-7",
        severity="critical",
        cooldown_minutes=30,
    )
    call = fake.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/agent-monitoring/alert-rules"
    assert call["base"] == "http://engine:4700/api/v1"
    assert call["retry"] is False
    assert call["json"] == {
        "name": "Failure rate above 10%",
        "metric": "failureRate",
        "operator": "gt",
        "threshold": 0.1,
        "windowMinutes": 15,
        "channels": [{"kind": "slack", "target": "https://hooks.slack.com/x"}],
        "severity": "critical",
        "cooldownMinutes": 30,
        "enabled": True,
        "agentId": "agent-7",
    }
    assert isinstance(rule, AlertRule)
    assert rule.id == "a1"
    assert rule.state == "ok"
    assert rule.fired_count == 0


def test_create_validates_metric_operator_and_channel_kind_locally():
    client = AlertRulesClient(FakeMonitorClient([]))  # type: ignore[arg-type]
    common = dict(operator="gt", threshold=1, window_minutes=5, channels=[slack("https://h/x")])
    with pytest.raises(ValueError, match="metric"):
        client.create("x", metric="vibes", **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="operator"):
        client.create("x", metric="traceCount", operator="ge", threshold=1, window_minutes=5, channels=[slack("https://h/x")])
    with pytest.raises(ValueError, match="channel kind"):
        client.create("x", metric="traceCount", operator="lt", threshold=1, window_minutes=5, channels=[{"kind": "sms", "target": "1"}])


def test_update_maps_snake_case_and_refuses_unknown_keys():
    fake = FakeMonitorClient([{"rule": {"_id": "a1", "state": "ok"}}])
    AlertRulesClient(fake).update("a1", window_minutes=60, cooldown_minutes=15, enabled=False)  # type: ignore[arg-type]
    call = fake.calls[0]
    assert call["method"] == "PUT"
    assert call["path"] == "/agent-monitoring/alert-rules/a1"
    assert call["json"] == {"windowMinutes": 60, "cooldownMinutes": 15, "enabled": False}
    with pytest.raises(ValueError, match="Unknown alert rule field"):
        AlertRulesClient(fake).update("a1", sample_rate=0.5)  # type: ignore[arg-type]


def test_update_applies_the_same_local_checks_as_create():
    fake = FakeMonitorClient([])
    client = AlertRulesClient(fake)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metric"):
        client.update("a1", metric="vibes")
    with pytest.raises(ValueError, match="operator"):
        client.update("a1", operator="ge")
    with pytest.raises(ValueError, match="channel kind"):
        client.update("a1", channels=[{"kind": "sms", "target": "1"}])
    assert fake.calls == []


def test_events_test_preview_and_sweep_paths():
    fake = FakeMonitorClient(
        [
            {"events": [{"kind": "triggered", "deliveries": [{"kind": "slack", "ok": True}, {"kind": "email", "ok": False}]}]},
            {"event": {"kind": "test", "deliveries": [{"kind": "slack", "ok": True}]}, "delivered": True},
            {"metric": "p95LatencyMs", "value": 812, "valueLabel": "812 ms"},
            {"evaluated": 2, "results": []},
        ]
    )
    client = AlertRulesClient(fake)  # type: ignore[arg-type]

    history = client.events("a1", limit=10)
    assert fake.calls[0]["path"] == "/agent-monitoring/alert-rules/a1/events"
    assert fake.calls[0]["params"] == {"limit": 10}
    assert isinstance(history[0], AlertEvent)
    assert history[0].kind == "triggered"
    assert history[0].delivered is False  # one channel failed

    sent = client.test("a1")
    assert fake.calls[1]["method"] == "POST"
    assert fake.calls[1]["path"] == "/agent-monitoring/alert-rules/a1/test"
    assert fake.calls[1]["retry"] is False
    assert sent.delivered is True

    preview = client.preview("p95LatencyMs", 15, agent_id="agent-7")
    assert fake.calls[2]["json"] == {"metric": "p95LatencyMs", "windowMinutes": 15, "agentId": "agent-7"}
    assert preview["value"] == 812

    client.run_sweep()
    assert fake.calls[3]["path"] == "/agent-monitoring/alert-rules/sweep/run"
    assert fake.calls[3]["retry"] is False


def test_delete_never_retries():
    fake = FakeMonitorClient([{}])
    AlertRulesClient(fake).delete("a1")  # type: ignore[arg-type]
    assert fake.calls[0] == {"method": "DELETE", "path": "/agent-monitoring/alert-rules/a1", "base": "http://engine:4700/api/v1", "retry": False}
