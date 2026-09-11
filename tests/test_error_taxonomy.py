"""P1 regression: one error taxonomy. Every per-client error subclasses
agentx.exceptions.AgentXError, and the auth/validation classes are the SAME canonical classes
everywhere - so `except agentx.AgentXError` / `except agentx.AgentXAuthError` work no matter
which sub-client raised."""

import pytest

import agentx
from agentx.exceptions import AgentXError


def test_every_client_error_subclasses_agentx_error():
    from agentx.evaluations.client import (
        AgentXEvaluationsError,
        EvaluationSubmissionError,
    )
    from agentx.monitor.client import AgentXMonitorError
    from agentx.monitor.judge_scorers import AgentXJudgeScorersError
    from agentx.monitor.scorer_groups import AgentXScorerGroupsError
    from agentx.monitor.scorers import AgentXScorersError
    from agentx.monitor.improvement_groups import AgentXImprovementGroupsError
    from agentx.outcomes import AgentXOutcomesError
    from agentx.feedback import AgentXFeedbackError
    from agentx.export import AgentXExportError
    from agentx.traces import AgentXTracesError
    from agentx.projects import AgentXProjectsError

    for cls in (
        AgentXEvaluationsError,
        EvaluationSubmissionError,
        AgentXMonitorError,
        AgentXJudgeScorersError,
        AgentXScorerGroupsError,
        AgentXScorersError,
        AgentXImprovementGroupsError,
        AgentXOutcomesError,
        AgentXFeedbackError,
        AgentXExportError,
        AgentXTracesError,
        AgentXProjectsError,
    ):
        assert issubclass(cls, AgentXError), cls.__name__
        assert issubclass(cls, agentx.AgentXError), cls.__name__


def test_auth_and_validation_errors_are_the_canonical_classes():
    from agentx.evaluations import client as eval_client
    from agentx.monitor import client as monitor_client

    # The shadow definitions are gone - both modules re-export the canonical classes, so the
    # names still import from where they always did.
    assert eval_client.AgentXAuthError is agentx.AgentXAuthError
    assert eval_client.AgentXValidationError is agentx.AgentXValidationError
    assert monitor_client.AgentXAuthError is agentx.AgentXAuthError
    assert monitor_client.AgentXValidationError is agentx.AgentXValidationError


def test_evaluations_401_is_catchable_as_top_level_auth_error(monkeypatch):
    from agentx.evaluations.client import EvaluationsClient

    client = EvaluationsClient(api_key="k", base_url="http://engine:1/api/v1")

    class FakeResponse:
        status_code = 401
        ok = False
        text = "nope"

    monkeypatch.setattr(client._session, "request", lambda *a, **kw: FakeResponse())
    with pytest.raises(agentx.AgentXAuthError):
        client.get_run("run-1")


def test_monitor_422_is_catchable_as_top_level_validation_error(monkeypatch):
    from agentx.monitor.client import MonitorClient

    client = MonitorClient(api_key="k", base_url="http://engine:1/api/v1")

    class FakeResponse:
        status_code = 422
        ok = False
        text = "bad payload"

    monkeypatch.setattr(client._session, "request", lambda *a, **kw: FakeResponse())
    with pytest.raises(agentx.AgentXValidationError) as caught:
        client.kpis()
    assert caught.value.status_code == 422
