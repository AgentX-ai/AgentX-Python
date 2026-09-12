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

    def fake_request(method, url, headers=None, json=None, params=None, timeout=None):
        calls.append({"method": method, "url": url, "json": json, "params": params})
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


def test_scorer_id_is_the_preferred_run_kwarg():
    """Runs pick their grader as scorer_id (post-consolidation name); the legacy
    evaluation_settings_id kwarg maps to the same wire field, and passing two DIFFERENT ids is
    rejected."""
    import pytest

    from agentx.evaluations.client import _resolve_scorer_id

    assert _resolve_scorer_id("s1", None) == "s1"
    assert _resolve_scorer_id(None, "s1") == "s1"
    assert _resolve_scorer_id("s1", "s1") == "s1"
    assert _resolve_scorer_id(None, None) is None
    with pytest.raises(ValueError):
        _resolve_scorer_id("s1", "s2")


def test_init_run_sends_scorer_id_as_evaluationSettingsId(monkeypatch):
    from agentx.evaluations.client import EvaluationsClient
    from agentx.evaluations.models import EvaluationSubject

    client = EvaluationsClient(api_key="agtx_local_test", base_url="http://localhost:1")
    captured = {}

    def fake_request(method, path, **kwargs):
        captured["payload"] = kwargs.get("json")
        return {
            "runId": "r1",
            "datasetId": "d1",
            "status": "running",
            "numberOfRequests": 1,
        }

    monkeypatch.setattr(client, "_request", fake_request)
    client.init_run("d1", EvaluationSubject(type="external"), scorer_id="scorer-123")
    assert captured["payload"]["evaluationSettingsId"] == "scorer-123"


def test_judge_scorers_builder_matches_legacy_builder_ergonomics(monkeypatch):
    """The unified successor of evaluations.settings.builder: snake_case kwargs, .publish(),
    plus what the legacy builder never had - tool_context, thresholds, and the live profile in
    the same call. The payload it assembles is plain judge-scorers wire."""
    from agentx.monitor.judge_scorers import JudgeScorersClient

    client = JudgeScorersClient(api_key="agtx_local_test", base_url="http://localhost:1")
    captured = {}

    def fake_request(method, path, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = kwargs.get("json")
        return {"judgeScorer": {"_id": "s1", "name": "Support quality", "judge": {}, "offline": {}, "online": None}}

    monkeypatch.setattr(client, "_request", fake_request)
    scorer = (
        client.builder(
            "Support quality",
            acceptance_criteria="Concrete and correct.",
            judge_model="gpt-4.1-mini",
            tool_context="detailed",
            number_of_requests=2,
            vector_similarity=True,
            thresholds={"enabled": True, "gates": [{"metric": "rating", "operator": "lt", "value": 5}]},
            live=True,
            sample_rate=0.25,
            agent_ids=["support-agent"],
        ).publish()
    )
    assert scorer.id == "s1"
    payload = captured["payload"]
    assert captured["path"] == "/judge-scorers"
    assert payload["judge"] == {
        "acceptanceCriteria": "Concrete and correct.",
        "judgeModel": "gpt-4.1-mini",
        "toolContext": "detailed",
    }
    assert payload["offline"]["numberOfRequests"] == 2
    assert payload["offline"]["vectorSimilarity"] == {"enabled": True}
    assert payload["offline"]["thresholds"]["gates"][0]["value"] == 5
    assert payload["online"]["enabled"] is True
    assert payload["online"]["sampleRate"] == 0.25
    assert payload["online"]["scopeMode"] == "selected"
    assert payload["online"]["agentIds"] == ["support-agent"]


def test_builder_rejects_idle_seconds_without_session_scope():
    """idleSeconds only applies to session scope - the engine silently ignores it with trace
    scope, so an explicit idle_seconds without scope="session" is a hard error, not an inert
    wire field."""
    client = JudgeScorersClient(api_key="agtx_local_test", base_url="http://localhost:1")
    with pytest.raises(ValueError, match="idle_seconds requires scope='session'"):
        client.builder("Support quality", live=True, idle_seconds=300)


def test_builder_rejects_online_kwargs_without_live():
    """agent_ids and a non-default scope only reach the wire when live=True - explicit
    values with live=False were silently discarded, so they are now hard errors, same
    posture as the idle_seconds guard."""
    client = JudgeScorersClient(api_key="agtx_local_test", base_url="http://localhost:1")
    with pytest.raises(ValueError, match="agent_ids requires live=True"):
        client.builder("Support quality", agent_ids=["support-agent"])
    with pytest.raises(ValueError, match="scope requires live=True"):
        client.builder("Support quality", scope="session")
    with pytest.raises(ValueError, match="agent_ids and scope require live=True"):
        client.builder("Support quality", agent_ids=["support-agent"], scope="session")


def test_builder_sends_idle_seconds_and_requires_expected(monkeypatch):
    """With scope="session", an explicit idle_seconds reaches the wire, and requires_expected
    lands in the judge section as requiresExpected."""
    client = JudgeScorersClient(api_key="agtx_local_test", base_url="http://localhost:1")
    captured = {}

    def fake_request(method, path, **kwargs):
        captured["payload"] = kwargs.get("json")
        return {"judgeScorer": {"_id": "s1", "name": "Support quality", "judge": {}, "offline": {}, "online": None}}

    monkeypatch.setattr(client, "_request", fake_request)
    client.builder(
        "Support quality",
        requires_expected=True,
        live=True,
        scope="session",
        idle_seconds=300,
    ).publish()
    payload = captured["payload"]
    assert payload["judge"]["requiresExpected"] is True
    assert payload["online"]["scope"] == "session"
    assert payload["online"]["idleSeconds"] == 300


def test_from_env_honors_selfhost_base_url_conventions(monkeypatch):
    """from_env silently targeting the hosted default while the shell exports the self-host
    conventions (AGENTX_SELFHOST_BASE_URL / BASE_URL) produced confusing auth errors - it now
    picks up the first convention that is set."""
    from agentx import AgentX

    monkeypatch.setenv("AGENTX_API_KEY", "agtx_local_test")
    monkeypatch.delenv("AGENTX_API_BASE_URL", raising=False)
    monkeypatch.setenv("AGENTX_SELFHOST_BASE_URL", "http://localhost:4999/api/v1")
    client = AgentX.from_env()
    assert client.base_url == "http://localhost:4999/api/v1"

    monkeypatch.setenv("AGENTX_API_BASE_URL", "http://localhost:5000/api/v1")
    assert AgentX.from_env().base_url == "http://localhost:5000/api/v1"  # explicit name wins


def test_tune_unwraps_the_proposal_envelope(monkeypatch):
    """The tune wire wraps its result in {"proposal": {...}} - judge_scorers.tune must unwrap it
    like the legacy client does, so proposal["reasoning"]/criteria are directly addressable
    (selfhost_demo/11 crashed on the wrapped form)."""
    from agentx.monitor.judge_scorers import JudgeScorersClient

    client = JudgeScorersClient(api_key="agtx_local_test", base_url="http://localhost:1")
    monkeypatch.setattr(client, "_profile_id", lambda scorer_id: "prof-1")
    monkeypatch.setattr(
        client,
        "_request",
        lambda *a, **k: {"proposal": {"reasoning": "why", "acceptanceCriteria": "a"}},
    )
    proposal = client.tune("s1")
    assert proposal["reasoning"] == "why"


def test_validate_and_publish_send_criteria_at_top_level(monkeypatch):
    """The tuning wire takes the criteria fields at the TOP level of the body (the legacy client
    always did) - nesting them under "criteria" 400s with 'acceptanceCriteria is required'."""
    from agentx.monitor.judge_scorers import JudgeScorersClient

    client = JudgeScorersClient(api_key="agtx_local_test", base_url="http://localhost:1")
    monkeypatch.setattr(client, "_profile_id", lambda scorer_id: "prof-1")
    captured = {}

    def fake_request(method, path, json=None, timeout=60):
        captured[path.rsplit("/", 1)[-1]] = json
        return {}

    monkeypatch.setattr(client, "_request", fake_request)
    criteria = {"acceptanceCriteria": "a", "rejectionCriteria": "r", "evaluationCriteria": "e"}
    client.validate_tuning("s1", criteria, window="24h")
    client.publish_tuning("s1", criteria)
    assert captured["validate"]["acceptanceCriteria"] == "a"
    assert captured["validate"]["window"] == "24h"
    assert "criteria" not in captured["validate"]
    assert captured["publish"]["acceptanceCriteria"] == "a"


def test_code_scorers_are_retrievable_from_the_wire_object():
    """The wire rows may lack ids (SDK-created scorers) - retrieval must hand them back as-is."""
    from agentx.monitor.judge_scorers import JudgeScorer

    scorer = JudgeScorer(
        {
            "_id": "s1",
            "name": "Blend",
            "offline": {"codeScorers": [{"name": "Final score", "code": "return 1;", "enabled": True}]},
        }
    )
    assert scorer.code_scorers == [{"name": "Final score", "code": "return 1;", "enabled": True}]
    # And an offline profile without any stays an empty list, not a KeyError.
    assert JudgeScorer({"_id": "s2", "name": "Plain", "offline": {}}).code_scorers == []


def test_dataset_model_round_trips_code_scorers():
    """extra="ignore" used to silently drop codeScorers on read - import_dataset lost them."""
    from agentx.evaluations.models import Dataset

    wire = {
        "_id": "d1",
        "name": "Guarded",
        "questions": [],
        "codeScorers": [{"id": "cs1", "name": "gate", "code": "return 0;", "enabled": True}],
    }
    parsed = Dataset(**wire)
    assert parsed.code_scorers == wire["codeScorers"]
    assert parsed.model_dump(by_alias=True)["codeScorers"] == wire["codeScorers"]


def test_publish_tuning_preserves_the_validation_token(recorded):
    """P1 regression: publish_tuning re-projected the validation dict and dropped the signed
    validationToken from validate_tuning's response, so the engine's version history stamped
    every publish client-asserted instead of measured."""
    calls, responses = recorded
    responses.append(
        FakeResponse({"judgeScorer": {"_id": "s1", "name": "n", "online": {"profileId": "prof-9"}}})
    )
    responses.append(FakeResponse({"published": True}))
    make_client().publish_tuning(
        "s1",
        {"acceptanceCriteria": "a", "rejectionCriteria": "r", "evaluationCriteria": "e"},
        validation={
            "verdict": "improved",
            "netAgreementGain": 0.12,
            "validationToken": "tok-123",
            "fixed": 3,
        },
    )
    body = calls[1]["json"]
    assert body["validation"] == {"verdict": "improved", "netAgreementGain": 0.12, "token": "tok-123"}


def test_legacy_monitor_publish_lifts_validation_token(monkeypatch):
    """Same fix on the legacy MonitorClient path: the dict passes through whole AND the
    validationToken is lifted into the `token` key the engine's publish route verifies."""
    from agentx.monitor.client import MonitorClient

    client = MonitorClient(api_key="k", base_url="http://engine:1/api/v1")
    captured = {}

    def fake_request(method, path, base=None, json=None, **kwargs):
        captured.update({"method": method, "path": path, "json": json})
        return {}

    monkeypatch.setattr(client, "_request", fake_request)
    client.publish_online_evaluator_tuning(
        "ev-1",
        {"acceptanceCriteria": "a"},
        validation={"verdict": "improved", "validationToken": "tok-9", "fixed": 2},
    )
    validation = captured["json"]["validation"]
    assert validation["token"] == "tok-9"
    assert validation["validationToken"] == "tok-9"  # passed through whole
    assert validation["fixed"] == 2
