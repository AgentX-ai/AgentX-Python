"""The analysis calls' self-host fallback.

Self-host (AgentX-trace-eval) serves whole-run analysis from its dashboard router, not from
the /custom-agent-evaluations router the SDK targets. Older engines lack the SDK routes
entirely, so the three analysis calls try the SDK route and fall back on a 404.

What these tests pin down is the part that is easy to get wrong: the fallback must be
invisible to hosted AgentX, must not swallow anything other than a 404, and must not retry a
billable synchronous request.

No network and no API key - the HTTP session is replaced with a recorder.
"""

import pytest

from agentx.evaluations.client import (
    AgentXAuthError,
    AgentXEvaluationsError,
    EvaluationsClient,
)

API_ROOT = "https://example.test/api/v1"
SDK_ROOT = f"{API_ROOT}/custom-agent-evaluations"
RUN = "run-123"


class FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = str(self._payload)

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self):
        return self._payload


class FakeSession:
    """Records every call and answers from a {(method, url): [responses]} routing table."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []
        self.headers = {}

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        key = (method, url)
        if key not in self.routes:
            return FakeResponse(404, {"message": "Not found"})
        answer = self.routes[key]
        if isinstance(answer, list):
            return answer.pop(0) if len(answer) > 1 else answer[0]
        return answer

    def urls(self, method=None):
        return [u for m, u, _ in self.calls if method is None or m == method]


def make_client(routes):
    client = EvaluationsClient(api_key="k", base_url=API_ROOT)
    session = FakeSession(routes)
    client._session = session
    return client, session


STATUS_BODY = {
    "evaluationId": RUN,
    "jobId": RUN,
    "status": "completed",
    "progress": {"overallPercentage": 100, "currentLevel": None, "levels": {}},
}

REPORT_BODY = {
    "runId": RUN,
    "datasetId": "ds-1",
    "status": "completed",
    "summary": "It went fine.",
    "recommendations": [{"category": "instructions", "priority": "high"}],
    "statistics": {"numberOfRuns": 4, "averageRating": 7.5, "minRating": 5, "maxRating": 9},
}


# ---------------------------------------------------------------------------
# Hosted AgentX: the SDK routes answer, so nothing may change
# ---------------------------------------------------------------------------


def test_hosted_uses_the_sdk_routes_and_never_probes_the_dashboard():
    client, session = make_client(
        {
            ("POST", f"{SDK_ROOT}/runs/{RUN}/analyze"): FakeResponse(200, {"status": "pending"}),
            ("GET", f"{SDK_ROOT}/runs/{RUN}/analyze-status"): FakeResponse(200, STATUS_BODY),
            ("GET", f"{SDK_ROOT}/runs/{RUN}/report"): FakeResponse(200, REPORT_BODY),
        }
    )

    client.analyze_run(RUN)
    assert client.get_analysis_status(RUN).status == "completed"
    assert client.get_report(RUN).summary == "It went fine."

    assert all("/custom-agent-evaluations/" in url for url in session.urls())
    assert client._analysis_on_dashboard_router is None, "hosted must never flip the flag"


# ---------------------------------------------------------------------------
# Self-host: the SDK routes 404, the dashboard routes answer
# ---------------------------------------------------------------------------


def test_analysis_status_falls_back_to_the_dashboard_router_on_404():
    client, session = make_client(
        {("GET", f"{API_ROOT}/evaluate/analyze/{RUN}/status"): FakeResponse(200, STATUS_BODY)}
    )

    assert client.get_analysis_status(RUN).is_terminal is True
    assert session.urls("GET") == [
        f"{SDK_ROOT}/runs/{RUN}/analyze-status",
        f"{API_ROOT}/evaluate/analyze/{RUN}/status",
    ]


def test_the_fallback_is_probed_once_then_remembered():
    client, session = make_client(
        {("GET", f"{API_ROOT}/evaluate/analyze/{RUN}/status"): FakeResponse(200, STATUS_BODY)}
    )

    client.get_analysis_status(RUN)
    client.get_analysis_status(RUN)

    # One 404 probe, not one per call.
    assert session.urls("GET").count(f"{SDK_ROOT}/runs/{RUN}/analyze-status") == 1
    assert client._analysis_on_dashboard_router is True


def test_report_is_assembled_from_the_dashboard_evaluation_record():
    client, _ = make_client(
        {
            ("GET", f"{API_ROOT}/evaluate/{RUN}"): FakeResponse(
                200,
                {
                    # datasetId arrives as a populated reference here, not a bare id.
                    "datasetId": {"_id": "ds-1", "name": "Support"},
                    "analysis": {
                        "status": "completed",
                        "statistics": {"numberOfRuns": 16, "averageRating": 6.21875,
                                       "minRating": 1, "maxRating": 9.5},
                        "analysis": {
                            "summary": "Uneven.",
                            "recommendations": [
                                {"category": "instructions", "priority": "high",
                                 "recommendation": "Cite the doc id.", "reasoning": "None cited."}
                            ],
                            # A key the SDK's Report does not model; must not raise.
                            "overallAssessment": "mixed",
                        },
                    },
                },
            )
        }
    )

    report = client.get_report(RUN)

    assert report.run_id == RUN
    assert report.dataset_id == "ds-1", "the populated reference must be unwrapped to its id"
    assert report.status == "completed"
    assert report.summary == "Uneven."
    assert report.statistics.average_rating == 6.21875
    assert report.statistics.min_rating == 1
    assert len(report.recommendations) == 1
    assert report.recommendations[0].category == "instructions"


def test_report_says_so_when_nothing_has_analyzed_the_run():
    client, _ = make_client(
        {("GET", f"{API_ROOT}/evaluate/{RUN}"): FakeResponse(200, {"datasetId": "ds-1"})}
    )

    with pytest.raises(AgentXEvaluationsError, match="no analysis"):
        client.get_report(RUN)


def test_the_synchronous_fallback_analyze_is_never_retried():
    """A read timeout on a billable sync endpoint must not re-run the judges."""
    import requests

    client, session = make_client({})

    def always_times_out(method, url, **kwargs):
        session.calls.append((method, url, kwargs))
        if url.endswith("/analyze"):
            return FakeResponse(404, {"message": "Not found"})
        raise requests.exceptions.ReadTimeout("too slow")

    session.request = always_times_out

    with pytest.raises(AgentXEvaluationsError):
        client.analyze_run(RUN)

    dashboard_posts = [u for u in session.urls("POST") if "/evaluate/analyze/" in u]
    assert len(dashboard_posts) == 1, f"retried a billable request: {dashboard_posts}"


def test_the_fallback_request_gets_the_long_analysis_timeout():
    client, session = make_client(
        {("POST", f"{API_ROOT}/evaluate/analyze/{RUN}"): FakeResponse(200, {"status": "completed"})}
    )

    client.analyze_run(RUN, judges=["gpt-5.5"])

    method, url, kwargs = session.calls[-1]
    assert url == f"{API_ROOT}/evaluate/analyze/{RUN}"
    assert kwargs["timeout"] > 60, "a synchronous judge pass needs more than the 30s default"
    assert kwargs["json"]["judges"] == [{"model": "gpt-5.5"}]


# ---------------------------------------------------------------------------
# Only a 404 means "wrong engine"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 403, 500])
def test_failures_that_are_not_404_propagate_untouched(status):
    client, session = make_client(
        {
            ("GET", f"{SDK_ROOT}/runs/{RUN}/analyze-status"): FakeResponse(status, {"e": "boom"}),
            # Present, and must not be reached.
            ("GET", f"{API_ROOT}/evaluate/analyze/{RUN}/status"): FakeResponse(200, STATUS_BODY),
        }
    )

    with pytest.raises(AgentXEvaluationsError) as caught:
        client.get_analysis_status(RUN)

    assert caught.value.status_code == status
    assert not [u for u in session.urls() if "/evaluate/" in u], "masked a real failure"
    assert client._analysis_on_dashboard_router is None


def test_auth_errors_are_not_mistaken_for_a_missing_route():
    client, session = make_client(
        {("GET", f"{SDK_ROOT}/runs/{RUN}/analyze-status"): FakeResponse(401, {"e": "nope"})}
    )

    with pytest.raises(AgentXAuthError):
        client.get_analysis_status(RUN)
    assert not [u for u in session.urls() if "/evaluate/" in u]
