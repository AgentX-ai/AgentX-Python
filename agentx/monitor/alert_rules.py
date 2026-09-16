from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agentx.monitor.client import MonitorClient

ALERT_METRICS = ("failureRate", "toolFailureRate", "p95LatencyMs", "estimatedCostUsd", "judgeFailures", "traceCount")
ALERT_CHANNEL_KINDS = ("slack", "teams", "pagerduty", "email", "webhook")

# snake_case kwargs -> wire keys. The engine reads camelCase only (the wire convention); an
# unknown snake_case key would be silently ignored, so update() refuses it instead.
_ALIASES = {
    "window_minutes": "windowMinutes",
    "agent_id": "agentId",
    "cooldown_minutes": "cooldownMinutes",
}


class AlertRule(dict):
    """Wire object for one KPI alert rule (dict subclass so unknown fields round-trip)."""

    @property
    def id(self) -> str:
        return self["_id"]

    @property
    def enabled(self) -> bool:
        return bool(self.get("enabled"))

    @property
    def state(self) -> str:
        """``"ok"`` or ``"firing"``."""
        return str(self.get("state") or "ok")

    @property
    def last_value(self) -> Optional[float]:
        return self.get("lastValue")

    @property
    def fired_count(self) -> int:
        return int(self.get("firedCount") or 0)


class AlertEvent(dict):
    """One row of a rule's notification history: ``kind`` is ``triggered`` / ``repeat`` /
    ``resolved`` / ``test``; ``deliveries`` lists each channel's outcome."""

    @property
    def kind(self) -> str:
        return str(self.get("kind"))

    @property
    def delivered(self) -> bool:
        deliveries = self.get("deliveries") or []
        return bool(deliveries) and all(bool(d.get("ok")) for d in deliveries)


def slack(url: str) -> Dict[str, str]:
    """A Slack incoming-webhook channel."""
    return {"kind": "slack", "target": url}


def teams(url: str) -> Dict[str, str]:
    """A Microsoft Teams incoming-webhook (or Workflows) channel."""
    return {"kind": "teams", "target": url}


def pagerduty(routing_key: str) -> Dict[str, str]:
    """A PagerDuty Events API v2 integration - ``routing_key`` is the integration key."""
    return {"kind": "pagerduty", "target": routing_key}


def email(address: str) -> Dict[str, str]:
    """An email recipient (needs a mailer configured on the engine)."""
    return {"kind": "email", "target": address}


def webhook(url: str) -> Dict[str, str]:
    """A generic JSON webhook receiving the full structured notification."""
    return {"kind": "webhook", "target": url}


class AlertRulesClient:
    """Surfaced as ``client.monitor.alert_rules``: KPI alert rules.

    An alert rule watches an AGGREGATE over a sliding window - one of
    ``failureRate``, ``toolFailureRate``, ``p95LatencyMs``, ``estimatedCostUsd``,
    ``judgeFailures``, ``traceCount`` - and pages typed channels (Slack, Teams,
    PagerDuty, email, generic webhook) when it crosses a threshold. The engine
    evaluates every enabled rule once a minute with an Alertmanager-style
    lifecycle: one ``triggered`` notification when the rule starts breaching, a
    ``repeat`` every ``cooldown_minutes`` while it keeps breaching, and a
    ``resolved`` notification when it recovers. This is distinct from a scorer's
    per-verdict alert threshold and from an automation rule's per-trace routing.

    Example::

        from agentx.monitor.alert_rules import slack, pagerduty

        rule = client.monitor.alert_rules.create(
            "Failure rate above 10%",
            metric="failureRate", operator="gt", threshold=0.10, window_minutes=15,
            severity="high",
            channels=[slack("https://hooks.slack.com/services/..."), pagerduty("R0123...")],
        )
        client.monitor.alert_rules.test(rule.id)   # sends a TEST page to every channel
    """

    def __init__(self, client: "MonitorClient"):
        self._client = client

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._client._request(method, path, base=self._client._api_root(), **kwargs)

    def list(self) -> List[AlertRule]:
        data = self._request("GET", "/agent-monitoring/alert-rules")
        return [AlertRule(r) for r in data.get("rules", [])]

    def get(self, rule_id: str) -> AlertRule:
        data = self._request("GET", f"/agent-monitoring/alert-rules/{rule_id}")
        return AlertRule(data.get("rule", data))

    def create(
        self,
        name: str,
        *,
        metric: str,
        operator: str,
        threshold: float,
        window_minutes: int,
        channels: List[Dict[str, str]],
        agent_id: Optional[str] = None,
        severity: str = "high",
        cooldown_minutes: int = 60,
        enabled: bool = True,
    ) -> AlertRule:
        """Create a rule. ``operator`` is ``"gt"`` (above) or ``"lt"`` (below); rates are
        fractions (``0.10`` = 10%), latency is milliseconds, cost is USD. ``channels`` takes the
        dicts the module-level helpers build (``slack(url)``, ``pagerduty(key)``, ...)."""
        if metric not in ALERT_METRICS:
            raise ValueError(f"metric must be one of {ALERT_METRICS}, got {metric!r}")
        if operator not in ("gt", "lt"):
            raise ValueError(f"operator must be 'gt' or 'lt', got {operator!r}")
        for channel in channels:
            if channel.get("kind") not in ALERT_CHANNEL_KINDS:
                raise ValueError(f"channel kind must be one of {ALERT_CHANNEL_KINDS}, got {channel.get('kind')!r}")
        payload: Dict[str, Any] = {
            "name": name,
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "windowMinutes": window_minutes,
            "channels": channels,
            "severity": severity,
            "cooldownMinutes": cooldown_minutes,
            "enabled": enabled,
        }
        if agent_id is not None:
            payload["agentId"] = agent_id
        # Server-side write: a timeout retry would create a duplicate rule that pages twice on
        # every incident - no transport retry (same posture as rules.create).
        data = self._request("POST", "/agent-monitoring/alert-rules", json=payload, retry=False)
        return AlertRule(data.get("rule", data))

    def update(self, rule_id: str, **fields: Any) -> AlertRule:
        """Sparse update; snake_case kwargs are mapped to the wire. Changing the metric,
        operator, threshold, window, or agent resets the rule's firing state."""
        payload: Dict[str, Any] = {}
        for key, value in fields.items():
            wire_key = _ALIASES.get(key, key)
            if "_" in wire_key:
                raise ValueError(
                    f"Unknown alert rule field {key!r} - the engine reads camelCase keys and would "
                    "silently ignore this (see AlertRulesClient.create for the field names)."
                )
            payload[wire_key] = value
        data = self._request("PUT", f"/agent-monitoring/alert-rules/{rule_id}", json=payload)
        return AlertRule(data.get("rule", data))

    def delete(self, rule_id: str) -> None:
        """Deletes the rule and its history. A PagerDuty incident the rule opened is not
        resolved by this - close it in PagerDuty."""
        self._request("DELETE", f"/agent-monitoring/alert-rules/{rule_id}", retry=False)

    def events(self, rule_id: str, limit: int = 50) -> List[AlertEvent]:
        """The rule's notification history, newest first, with per-channel delivery results."""
        data = self._request("GET", f"/agent-monitoring/alert-rules/{rule_id}/events", params={"limit": limit})
        return [AlertEvent(e) for e in data.get("events", [])]

    def test(self, rule_id: str) -> AlertEvent:
        """Send a TEST notification to the rule's channels with the metric's live value and
        return the recorded event - ``event.delivered`` says whether every channel accepted it.
        Never changes the rule's firing state."""
        data = self._request("POST", f"/agent-monitoring/alert-rules/{rule_id}/test", json={}, retry=False)
        return AlertEvent(data.get("event", data))

    def preview(self, metric: str, window_minutes: int, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """What ``metric`` reads right now over the last ``window_minutes`` - the same
        computation the sweep runs. Returns ``{"value": float | None, "valueLabel": str, ...}``;
        ``value`` is ``None`` when the window has no data for a rate metric."""
        payload: Dict[str, Any] = {"metric": metric, "windowMinutes": window_minutes}
        if agent_id is not None:
            payload["agentId"] = agent_id
        return self._request("POST", "/agent-monitoring/alert-rules/preview", json=payload)

    def run_sweep(self) -> Dict[str, Any]:
        """Evaluate this project's rules now instead of waiting for the next minute tick."""
        return self._request("POST", "/agent-monitoring/alert-rules/sweep/run", json={}, retry=False)
