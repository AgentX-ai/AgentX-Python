"""Unit tests for the unified LLM Judge Scorer surface (client.monitor.judge_scorers) and the
dataset-builder code_scorers forwarding fix - wire-level, no engine required (requests is
monkeypatched; the engine-side contract is pinned by the engine's own
judgeScorers.integration.test.ts)."""

from typing import Any, Dict, List, Optional

import pytest

from agentx.monitor.judge_scorers import AgentXJudgeScorersError, JudgeScorersClient


class FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = "x"
        self.reason = "reason"

    def json(self) -> Dict[str, Any]:
        return self._payload


@pytest.fixture()
def recorded(monkeypatch):
    calls: List[Dict[str, Any]] = []
    responses: List[FakeResponse] = []

    def fake_request(method, url, headers=None, json=None, timeout=None):
        calls.append({"method": method, "url": url, "json": json})
        return responses.pop(0) if responses else FakeResponse({"judgeScorer": {"_id": "s1", "name": "n"}})

    monkeypatch.setattr("agentx.monitor.judge_scorers.requests.request", fake_request)
    return calls, responses


def make_client() -> JudgeScorersClient:
    return JudgeScorersClient(api_key="k", base_url="http://engine:4700/api/v1")


def test_create_hits_the_unified_route_with_sections(recorded):
    calls, _ = recorded
    make_client().create(
        "Support quality",
        judge={"acceptanceCriteria": "Concrete."},
        online={"enabled": True, "sampleRate": 0.2},
    )
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"].endswith("/agent-monitoring/judge-scorers")
    assert calls[0]["json"] == {
        "name": "Support quality",
        "judge": {"acceptanceCriteria": "Concrete."},
        "online": {"enabled": True, "sampleRate": 0.2},
    }


def test_update_distinguishes_online_none_from_online_absent(recorded):
    calls, _ = recorded
    client = make_client()
    client.update("s1", judge={"judgePrompt": "p"})
    assert "online" not in calls[0]["json"]  # absent section untouched

    client.update("s1", online=None)
    assert calls[1]["json"] == {"online": None}  # explicit detach


def test_online_passthroughs_resolve_the_profile_id(recorded):
    calls, responses = recorded
    responses.append(FakeResponse({"judgeScorer": {"_id": "s1", "name": "n", "online": {"profileId": "prof-9"}}}))
    responses.append(FakeResponse({"agreementRate": 0.5}))
    result = make_client().calibration("s1", window="24h")
    assert result == {"agreementRate": 0.5}
    assert calls[1]["url"].endswith("/agent-monitoring/online-evaluators/prof-9/calibration?window=24h")


def test_offline_only_scorers_get_a_clear_error_for_online_surfaces(recorded):
    _, responses = recorded
    responses.append(FakeResponse({"judgeScorer": {"_id": "s1", "name": "n", "online": None}}))
    with pytest.raises(AgentXJudgeScorersError, match="no online profile"):
        make_client().ratings("s1")


def test_http_errors_surface_the_engine_message(recorded):
    _, responses = recorded
    responses.append(FakeResponse({"error": "built-in judge scorer"}, status_code=409))
    with pytest.raises(AgentXJudgeScorersError, match="409.*built-in judge scorer"):
        make_client().delete("baseline")


def test_dataset_builder_forwards_code_scorers(monkeypatch):
    # The documented kwarg (evaluation/code-scorers.mdx) raised TypeError before the fix.
    from agentx.evaluations.datasets import DatasetClient

    class FakeEvalClient:
        def create_dataset(self, payload):
            self.payload = payload
            return {"_id": "d1"}

    fake = FakeEvalClient()
    builder = DatasetClient(fake).builder(
        "ds", code_scorers=[{"name": "tool order", "code": "def handler(...): pass"}]
    )
    payload = builder._payload  # noqa: SLF001 - asserting the wire body the builder assembled
    assert "codeScorers" in payload
    assert payload["codeScorers"][0]["name"] == "tool order"
    assert payload["codeScorers"][0]["enabled"] is True
    assert payload["codeScorers"][0]["id"]


def test_legacy_profile_clients_warn_once_on_first_access():
    """The legacy views (evaluations.settings / monitor.online_evaluators) stay functional but
    emit a DeprecationWarning pointing at judge_scorers - lazily, so a client that never touches
    them never warns."""
    import warnings

    from agentx import AgentX

    with warnings.catch_warnings(record=True) as during_init:
        warnings.simplefilter("always")
        client = AgentX(api_key="agtx_local_test", base_url="http://localhost:1")
        _ = client.monitor.judge_scorers  # the unified surface never warns
    assert not [w for w in during_init if "judge_scorers" in str(w.message)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = client.evaluations.settings
        _ = client.monitor.online_evaluators
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("judge_scorers" in m for m in messages), messages
    assert len(messages) >= 2
