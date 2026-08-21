"""
What the SDK does when the deployment it is pointed at has no route for the call.

Pointing AGENTX_API_BASE_URL at a self-host engine (AgentX-ai/AgentX-trace-eval) is a documented
setup, and that engine implements Trace, Evaluate and Monitor but not the hosted API's whole
surface. Calls to the rest used to fail two ways, both bad: a raw ``HTTP 404: {...}`` dump, or -
worse - ``DatasetNotFound``, because every 404 on the CI path was mapped to it. That last one is
an outright lie; it sends someone hunting for a dataset that is sitting right there.

So: a 404 with no handler behind it raises EndpointNotAvailable, naming the call and the URL. A
404 that a handler actually wrote keeps meaning what it meant. These tests pin both halves - the
second matters as much as the first, since a detector that swallows real "not found" answers just
moves the lie somewhere else.

No network: the transport is replaced with canned responses.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pytest
import requests

from agentx import AgentX, EndpointNotAvailable
from agentx.util import endpoint_missing

BASE_URL = "https://selfhost.test/api/v1"

# What each backend actually puts in the body, copied from the real responses.
NO_ROUTE_JSON = {"statusCode": 404, "message": "Not found"}      # engine's /api catch-all
DATASET_GONE_ENGINE = {"error": "Dataset not found"}             # an engine handler
DATASET_GONE_HOSTED = {"message": "Dataset not found"}           # a hosted ErrorHandler


def _response(status: int, body: Any, *, url: str = BASE_URL, method: str = "GET") -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp.url = url
    resp.request = requests.Request(method=method, url=url).prepare()
    if isinstance(body, (dict, list)):
        resp.headers["content-type"] = "application/json"
        resp._content = json.dumps(body).encode()
    else:
        resp.headers["content-type"] = "text/html"
        resp._content = str(body).encode()
    return resp


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], AgentX]:
    """Returns a factory: give it a body, get a client whose every request 404s with it."""

    def build(body: Any, status: int = 404) -> AgentX:
        def fake_request(self: requests.Session, method: str, url: str, **kwargs: Any) -> requests.Response:
            return _response(status, body, url=url, method=method)

        monkeypatch.setattr(requests.Session, "request", fake_request, raising=True)
        monkeypatch.setenv("AGENTX_API_KEY", "selfhost-test-key")
        monkeypatch.setenv("AGENTX_API_BASE_URL", BASE_URL)
        return AgentX(api_key="selfhost-test-key", base_url=BASE_URL)

    return build


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (404, NO_ROUTE_JSON, True),                       # engine catch-all
        (404, "<!DOCTYPE html><title>Error</title>", True),  # bare Express 404
        (404, "", True),                                  # empty body, nothing wrote it
        (404, {"message": "Not Found"}, True),            # same thing, different casing
        (404, DATASET_GONE_ENGINE, False),                # a handler answered
        (404, DATASET_GONE_HOSTED, False),                # a handler answered
        (404, {"error": "Run not found"}, False),
        (403, NO_ROUTE_JSON, False),                      # only 404 is ambiguous
        (500, NO_ROUTE_JSON, False),
        (200, {}, False),
    ],
)
def test_endpoint_missing_separates_no_route_from_no_object(status, body, expected) -> None:
    assert endpoint_missing(_response(status, body)) is expected


# ---------------------------------------------------------------------------
# Every entry point a self-host engine has no route for
# ---------------------------------------------------------------------------

# Each of these 404s against AgentX-trace-eval today. The point is not that they work - they
# cannot - but that they say so.
UNROUTED_ON_SELF_HOST: list[tuple[str, Callable[[AgentX], Any]]] = [
    ("evaluations.list_models()", lambda c: c.evaluations.list_models()),
    ("tracer.evaluate_trace()", lambda c: c.tracer.evaluate_trace("trace-1", "ds-1")),
    ("tracer.create_ci_run()", lambda c: c.tracer.create_ci_run("ds-1")),
    ("tracer.submit_result()", lambda c: c.tracer.submit_result("run-1", 0, "answer")),
    ("tracer.finalize_ci_run()", lambda c: c.tracer.finalize_ci_run("run-1")),
    ("tracer.get_ci_run()", lambda c: c.tracer.get_ci_run("run-1")),
    ("get_dataset_test_cases()", lambda c: c.tracer._client.get_dataset_test_cases("ds-1")),
    ("tracer.run_eval()", lambda c: c.tracer.run_eval("ds-1", lambda q: "answer")),
]


@pytest.mark.parametrize("label,call", UNROUTED_ON_SELF_HOST, ids=[label for label, _ in UNROUTED_ON_SELF_HOST])
def test_unrouted_call_says_the_endpoint_is_missing(client, label: str, call) -> None:
    c = client(NO_ROUTE_JSON)
    with pytest.raises(EndpointNotAvailable) as excinfo:
        call(c)
    message = str(excinfo.value)
    # The two things someone needs to act on: which call, and against which deployment.
    assert BASE_URL.split("//")[1].split("/")[0] in message, message
    assert "404" in message, message


@pytest.mark.parametrize("label,call", UNROUTED_ON_SELF_HOST, ids=[label for label, _ in UNROUTED_ON_SELF_HOST])
def test_unrouted_call_survives_a_bare_html_404(client, label: str, call) -> None:
    """A proxy or a plain Express 404 in front of the engine returns HTML, not JSON. Parsing that
    as a body used to be how a confusing error got even more confusing."""
    c = client("<!DOCTYPE html><html><body>Cannot POST /api/v1/ingest/ci-runs</body></html>")
    with pytest.raises(EndpointNotAvailable):
        call(c)


# ---------------------------------------------------------------------------
# A real "not found" must keep meaning that
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", [DATASET_GONE_ENGINE, DATASET_GONE_HOSTED])
def test_a_handlers_own_404_is_not_reported_as_a_missing_endpoint(client, body) -> None:
    c = client(body)
    for call in (
        lambda: c.evaluations.datasets.get("ds-1"),
        lambda: c.monitor.signals.get("sig-1"),
        lambda: c.tracer.create_ci_run("ds-1"),
    ):
        with pytest.raises(Exception) as excinfo:
            call()
        assert not isinstance(excinfo.value, EndpointNotAvailable), (
            f"{body} is a handler saying the object is gone, not a missing route: {excinfo.value}"
        )


def test_missing_dataset_on_the_ci_path_is_still_DatasetNotFound(client) -> None:
    """The behaviour that was right before, and has to stay right: CI against a dataset that
    genuinely does not exist."""
    from agentx.exceptions import DatasetNotFound

    c = client(DATASET_GONE_HOSTED)
    with pytest.raises(DatasetNotFound):
        c.tracer.create_ci_run("ds-1")


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def test_existing_except_clauses_still_catch_it(client) -> None:
    """EndpointNotAvailable is a new condition, not a rename. Code that already catches the
    per-module error keeps working."""
    from agentx.evaluations.client import AgentXEvaluationsError
    from agentx.monitor.client import AgentXMonitorError

    c = client(NO_ROUTE_JSON)
    with pytest.raises(AgentXEvaluationsError):
        c.evaluations.list_models()
    with pytest.raises(AgentXMonitorError):
        c.monitor.patterns.list()


def test_endpoint_not_available_carries_the_details_programmatically(client) -> None:
    c = client(NO_ROUTE_JSON)
    with pytest.raises(EndpointNotAvailable) as excinfo:
        c.evaluations.list_models()
    err = excinfo.value
    assert err.call == "list_models()"
    assert err.method == "GET"
    assert err.url.endswith("/custom-agent-evaluations/models")
