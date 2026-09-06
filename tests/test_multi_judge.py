"""Wire-level tests for multi-judge dataset runs: additional_scorer_ids on run creation and
the named-scorer CI gate (gate_run(scorer=...)). The session/HTTP layer is monkeypatched; the
engine-side behavior is pinned by the engine's multiJudge.integration.test.ts."""

from typing import Any, Dict, List

import pytest

from agentx.evaluations.client import EvaluationsClient
from agentx.evaluations.models import EvaluationSubject


class FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = "x"

    def json(self) -> Dict[str, Any]:
        return self._payload


@pytest.fixture()
def recorded(monkeypatch):
    calls: List[Dict[str, Any]] = []

    def fake_request(method, url, timeout=None, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse(
            {
                "runId": "r1",
                "datasetId": "ds1",
                "status": "in_progress",
                "passed": True,
                "checks": [],
                "gatedScorer": None,
            }
        )

    client = EvaluationsClient(api_key="k", base_url="http://engine:4700/api/v1")
    monkeypatch.setattr(client._session, "request", fake_request)
    return client, calls


def test_init_run_sends_additional_scorer_ids_camel_case(recorded):
    client, calls = recorded
    client.init_run(
        "ds1",
        EvaluationSubject(kind="custom_agent"),
        scorer_id="primary",
        additional_scorer_ids=["safety", "tone"],
    )
    payload = calls[0]["json"]
    assert payload["evaluationSettingsId"] == "primary"
    assert payload["additionalScorerIds"] == ["safety", "tone"]


def test_init_run_omits_the_key_when_no_additional_scorers(recorded):
    client, calls = recorded
    client.init_run("ds1", EvaluationSubject(kind="custom_agent"), scorer_id="primary")
    assert "additionalScorerIds" not in calls[0]["json"]


def test_gate_run_forwards_the_named_scorer(recorded):
    client, calls = recorded
    client.gate_run("r1", fail_under=5, scorer="Safety")
    params = calls[0]["params"]
    assert params["failUnder"] == 5
    assert params["scorer"] == "Safety"


def test_gate_run_leaves_scorer_off_for_primary_gates(recorded):
    client, calls = recorded
    client.gate_run("r1", fail_under=5)
    assert "scorer" not in calls[0]["params"]
