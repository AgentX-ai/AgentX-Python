"""
Contract test: the exact set of HTTP endpoints this SDK calls.

Every path here is answered by one of two backends we don't ship in this repo — AgentX's hosted
API, and the self-hostable governance engine (AgentX-ai/AgentX-trace-eval). Renaming a path, or
adding a call to one nobody implemented, is invisible from inside this package: unit tests mock
the transport and the integration suite only touches the handful of endpoints it exercises. What
users get instead is a 404 at runtime.

So the endpoint list lives in one checked-in file, ``contracts/sdk-endpoints.json`` in the
AgentX-trace-eval repo, which also records which backend implements each one. This test drives
every public SDK method through a recording transport and asserts the paths it emits are exactly
the rows in that file — no more (a new call nobody has implemented), no fewer (a row that is now
dead). Two sibling suites in that repo assert the other halves: that the engine mounts its rows,
and that the hosted API routes its own.

The file lives one repo over, so this suite is opt-in:

    AGENTX_SDK_CONTRACT=../AgentX-trace-eval/contracts/sdk-endpoints.json pytest tests/test_endpoint_contract.py

It also finds a sibling checkout automatically, and skips when there is neither. Nothing here
touches the network — the transport is replaced before the client is built.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest
import requests

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase, EvaluationSubject
from agentx.resources.agent import Agent
from agentx.resources.conversation import Conversation
from agentx.resources.workforce import Workforce

BASE_URL = "https://contract.test/api/v1"
# One sentinel for every path parameter, so a recorded URL normalizes back to its templated form
# without this test having to know which segments are ids.
ID = "contract-probe"


# ---------------------------------------------------------------------------
# The contract file
# ---------------------------------------------------------------------------


def _contract_path() -> Path:
    override = os.getenv("AGENTX_SDK_CONTRACT")
    candidates = [Path(override)] if override else []
    repo = Path(__file__).resolve().parents[1]
    candidates += [
        repo.parent / "AgentX-trace-eval" / "contracts" / "sdk-endpoints.json",
        repo.parent / "AgentX-Trace-Eval" / "contracts" / "sdk-endpoints.json",
    ]
    return next((c for c in candidates if c.is_file()), Path())


CONTRACT_FILE = _contract_path()

pytestmark = pytest.mark.skipif(
    not CONTRACT_FILE.is_file(),
    reason="contracts/sdk-endpoints.json not found — set AGENTX_SDK_CONTRACT or check out AgentX-trace-eval alongside this repo",
)


def _load_contract() -> Dict[str, Any]:
    return json.loads(CONTRACT_FILE.read_text())


def _normalize(path: str) -> str:
    """"/runs/{runId}/results" and "/runs/contract-probe/results" both become "/runs/*/results"."""
    segments = [
        "*" if (segment == ID or (segment.startswith("{") and segment.endswith("}"))) else segment
        for segment in path.split("/")
    ]
    return "/".join(segments).rstrip("/") or "/"


# ---------------------------------------------------------------------------
# Recording transport
# ---------------------------------------------------------------------------

# Just enough of a response body for calls that feed a later call in the same driver — a run
# needs its dataset's questions to build cases, and .analyze() polls until the status is terminal.
# Everything else gets {}: the drivers below swallow the resulting parse errors, because what is
# under test is which request went out, not what the SDK made of the answer.
_BODIES: Dict[str, Any] = {
    "GET /custom-agent-evaluations/datasets/*": {
        "_id": ID,
        "name": "contract dataset",
        "numberOfRequests": 1,
        "questions": [{"main_question": {"query": "ping", "expectedResults": "pong"}}],
    },
    "POST /custom-agent-evaluations/runs": {"runId": ID, "datasetId": ID},
    "GET /custom-agent-evaluations/runs/*/analyze-status": {"status": "completed"},
}


class _Recorder:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, str]] = []

    def record(self, method: str, url: str) -> requests.Response:
        path = url.split("/api/v1", 1)[-1].split("?")[0]
        normalized = _normalize(path)
        self.calls.append((method.upper(), normalized))

        response = requests.Response()
        response.status_code = 200
        response.url = url
        response.headers["content-type"] = "application/json"
        response._content = json.dumps(_BODIES.get(f"{method.upper()} {normalized}", {})).encode()
        return response

    def paths(self) -> set:
        return set(self.calls)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()

    # A plain function, not the recorder object: only functions bind as methods, and a non-descriptor
    # here would be called without `self` and silently record nothing.
    def fake_request(self: requests.Session, method: str, url: str, **kwargs: Any) -> requests.Response:
        return rec.record(method, url)

    # requests.get/post/... funnel through Session.request too, so this one patch covers both the
    # pooled clients (evaluations/monitor/ingest) and the module-level calls the resource objects
    # and AgentX itself make.
    monkeypatch.setattr(requests.Session, "request", fake_request, raising=True)
    # AgentX.__init__ writes its arguments back into the environment for the sub-clients to read.
    monkeypatch.setenv("AGENTX_API_KEY", "contract-test-key")
    monkeypatch.setenv("AGENTX_API_BASE_URL", BASE_URL)
    monkeypatch.delenv("AGENTX_WORKSPACE_ID", raising=False)
    return rec


@pytest.fixture
def client(recorder: _Recorder) -> AgentX:
    return AgentX(api_key="contract-test-key", base_url=BASE_URL)


# ---------------------------------------------------------------------------
# One driver per public method that makes an HTTP call
# ---------------------------------------------------------------------------


def _drivers(client: AgentX) -> Dict[str, Callable[[], Any]]:
    """Keyed by the contract's ``sdk`` field, so a renamed row fails loudly rather than silently
    dropping its coverage."""
    subject = EvaluationSubject(kind="custom_agent", display_name="contract-probe")
    agent = Agent.model_construct(id=ID, name="contract-probe")
    conversation = Conversation.model_construct(agent_id=ID, id=ID)
    workforce = Workforce.model_construct(id=ID, manager=agent)

    # .run() is the entry point for the whole run lifecycle; the later stages need its context.
    run_context: Dict[str, Any] = {}

    def start_run() -> Any:
        run_context["ctx"] = client.evaluations.run(ID, subject)
        return run_context["ctx"]

    return {
        "client.evaluations.list_models()": lambda: client.evaluations.list_models(),
        "client.evaluations.datasets.builder(...).publish()": lambda: client.evaluations.datasets.builder(
            name="contract dataset"
        )
        .add_case(query="ping", expected_results="pong")
        .publish(),
        "client.evaluations.datasets.list()": lambda: client.evaluations.datasets.list(),
        "client.evaluations.datasets.get(dataset_id)": lambda: client.evaluations.datasets.get(ID),
        "client.evaluations.settings.builder(...).publish()": lambda: client.evaluations.settings.builder(
            name="contract settings"
        ).publish(),
        "client.evaluations.settings.list()": lambda: client.evaluations.settings.list(),
        "client.evaluations.settings.get(evaluation_settings_id)": lambda: client.evaluations.settings.get(ID),
        "client.evaluations.prompts.create(name, text)": lambda: client.evaluations.prompts.create(
            name="contract-prompt", text="You are helpful."
        ),
        "client.evaluations.prompts.list()": lambda: client.evaluations.prompts.list(),
        "client.evaluations.prompts.get(name)": lambda: client.evaluations.prompts.get(ID),
        "client.evaluations.run(dataset_id, subject)": start_run,
        "run.execute(adapter)": lambda: run_context["ctx"].execute(lambda case: "pong"),
        "run.finalize()": lambda: run_context["ctx"].finalize(),
        "run.analyze()": lambda: run_context["ctx"].analyze(),
        "client.evaluations.get_analysis_status(run_id)": lambda: client.evaluations.get_analysis_status(ID),
        "client.evaluations._client.get_report(run_id)": lambda: client.evaluations._client.get_report(ID),
        "client.evaluations._client.get_run(run_id)": lambda: client.evaluations._client.get_run(ID),
        "run.gate(...) / client.evaluations.gate_run(run_id, ...)": lambda: (
            run_context["ctx"].gate(fail_under=7),
            client.evaluations.gate_run(ID, fail_under=7),
        ),
        # The analysis routes have a self-host fallback: on a 404 from the primary path the client
        # latches onto the dashboard router and stays there. Latching it by hand drives that half
        # without having to make the recorder answer 404 for one path only.
        "client.evaluations._client.analyze_run(run_id) [dashboard-router fallback]": lambda: _on_fallback_router(
            client
        ).analyze_run(ID),
        "client.evaluations._client.get_analysis_status(run_id) [dashboard-router fallback]": lambda: _on_fallback_router(
            client
        ).get_analysis_status(ID),
        "client.evaluations._client.get_report(run_id) [dashboard-router fallback]": lambda: _on_fallback_router(
            client
        ).get_report(ID),
        "client.monitor.patterns.builder(...).publish()": lambda: client.monitor.patterns.builder(
            name="contract pattern", detector_kind="contains", include_terms=["nope"]
        ).publish(),
        "client.monitor.patterns.list()": lambda: client.monitor.patterns.list(),
        "client.monitor.patterns.get(pattern_id)": lambda: client.monitor.patterns.get(ID),
        "client.monitor.signals.list()": lambda: client.monitor.signals.list(),
        "client.monitor.signals.get(signal_id)": lambda: client.monitor.signals.get(ID),
        "client.monitor.profile.get(agent_id)": lambda: client.monitor.profile.get(ID),
        "client.monitor.profile.update(agent_id, ...)": lambda: client.monitor.profile.update(
            ID, threshold_overrides={"latencyMs": 500}
        ),
        "client.monitor.online_evaluators.builder(...).publish()": lambda: client.monitor.online_evaluators.builder(
            name="contract evaluator", evaluation_settings_id=ID
        ).publish(),
        "client.monitor.online_evaluators.list()": lambda: client.monitor.online_evaluators.list(),
        "client.monitor.online_evaluators.get(evaluator_id)": lambda: client.monitor.online_evaluators.get(ID),
        "client.monitor.online_evaluators.update(evaluator_id, ...)": lambda: client.monitor.online_evaluators.update(
            ID, sample_rate=0.5
        ),
        "client.monitor.online_evaluators.delete(evaluator_id)": lambda: client.monitor.online_evaluators.delete(ID),
        "client.monitor.online_evaluators.ratings(evaluator_id)": lambda: client.monitor.online_evaluators.ratings(ID),
        "client.monitor.online_evaluators.events(evaluator_id)": lambda: client.monitor.online_evaluators.events(ID),
        # sync=True so the POST happens on this thread; the default queues it on the ingest worker
        # and there would be nothing to record by the time the assertions run.
        "client.tracer.trace(...)": lambda: _traced(client),
        "client.tracer.evaluate_trace(trace_id, dataset_id)": lambda: client.tracer.evaluate_trace(ID, ID),
        "client.tracer.create_ci_run(dataset_id)": lambda: client.tracer.create_ci_run(ID),
        "client.tracer.submit_result(run_id, ...)": lambda: client.tracer.submit_result(ID, 0, "pong"),
        "client.tracer.finalize_ci_run(run_id)": lambda: client.tracer.finalize_ci_run(ID),
        "client.tracer.get_ci_run(run_id)": lambda: client.tracer.get_ci_run(ID),
        "client.tracer._client.get_dataset_test_cases(dataset_id)": lambda: client.tracer._client.get_dataset_test_cases(
            ID
        ),
        "client.outcomes.report(...)": lambda: client.outcomes.report(
            trace_id=ID, outcome="reopened", is_negative=True
        ),
        "client.feedback.report(trace_id, rating)": lambda: client.feedback.report(ID, "down"),
        "AgentX.list_agents()": lambda: client.list_agents(),
        "AgentX.get_agent(id)": lambda: client.get_agent(ID),
        "AgentX.list_workforces()": lambda: client.list_workforces(),
        "AgentX.get_profile()": lambda: client.get_profile(),
        "Agent.new_conversation()": lambda: agent.new_conversation(),
        "Agent.list_conversations()": lambda: agent.list_conversations(),
        "Conversation.list_messages()": lambda: conversation.list_messages(),
        "Conversation.chat(message)": lambda: conversation.chat("ping"),
        "Conversation.chat_stream(message)": lambda: list(conversation.chat_stream("ping")),
        "Workforce.new_conversation()": lambda: workforce.new_conversation(),
        "Workforce.list_conversations()": lambda: workforce.list_conversations(),
        "Workforce.chat_stream(conversation_id, message)": lambda: list(workforce.chat_stream(ID, "ping")),
    }


def _on_fallback_router(client: AgentX) -> Any:
    """Flip the evaluations client into the mode it enters after a 404 from the primary analysis
    routes, so the fallback paths are exercised in the same sweep as the primary ones."""
    inner = client.evaluations._client
    inner._analysis_on_dashboard_router = True
    return inner


def _traced(client: AgentX) -> Any:
    with client.tracer.trace("contract-probe", sync=True) as span:
        span.input = "ping"
        span.output = "pong"
    return span


# Public methods that introduce no endpoint of their own, so the coverage check below doesn't
# demand a contract row for them. Some make no request at all; the tracer's recorders send on
# POST /ingest/traces, which is already a row. Anything not listed here and not driven above is a
# genuine hole.
# Sub-clients hung off AgentX that the contract has rows for.
_CLIENT_SURFACES = ("evaluations", "monitor", "tracer", "outcomes", "feedback")

_NO_NEW_ENDPOINT = {
    "AgentX.from_env",
    "Agent.get_conversation",  # filters the result of list_conversations()
    "DatasetClient.builder",
    "DatasetClient.from_csv",
    "DatasetClient.from_dataframe",
    "EvaluationSettingsClient.builder",
    "MonitorOnlineEvaluatorClient.builder",
    "MonitorPatternClient.builder",
    "EvaluationsRunner.run",  # driven as "client.evaluations.run(dataset_id, subject)"
    "Tracer.flush",
    "Tracer.trace",
    "Tracer.current_span",
    "Tracer.use_span",
    # Child spans and recorders: all send POST /ingest/traces, the row tracer.trace() covers.
    "Tracer.record_tool_call",
    "Tracer.trace_tool_call",
    "Tracer.record_retrieval",
    "Tracer.trace_retrieval",
    "Tracer.run_eval",  # composes create_ci_run/submit_result/finalize_ci_run, all driven above
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_client_exposes_every_surface_the_contract_covers(client: AgentX) -> None:
    """First line of diagnosis. A checkout behind origin/main fails everything below with
    AttributeErrors that look like bugs; this says what is actually going on."""
    missing = [name for name in _CLIENT_SURFACES if not hasattr(client, name)]
    assert missing == [], (
        f"contracts/sdk-endpoints.json covers client.{{{','.join(missing)}}}, which this checkout "
        "doesn't have. Usually that means the checkout is behind origin/main — pull, then re-run. "
        "If the surface was removed on purpose, drop its rows from the contract."
    )


def test_every_contract_endpoint_has_a_driver(client: AgentX) -> None:
    contract = _load_contract()
    documented = {e["sdk"] for e in contract["endpoints"]}
    driven = set(_drivers(client))
    assert documented - driven == set(), "contract rows with no driver in this test"
    assert driven - documented == set(), "drivers here for endpoints the contract doesn't list"


def test_sdk_calls_exactly_the_documented_endpoints(client: AgentX, recorder: _Recorder) -> None:
    contract = _load_contract()

    for label, drive in _drivers(client).items():
        try:
            drive()
        except Exception:
            # A canned {} body is not a valid model for most endpoints. The request still went out,
            # and that is the whole assertion — see _BODIES.
            pass

    documented = {(e["method"], _normalize(e["path"])) for e in contract["endpoints"]}
    called = recorder.paths()

    undocumented = sorted(f"{m} {p}" for m, p in called - documented)
    assert undocumented == [], (
        "the SDK calls endpoints that contracts/sdk-endpoints.json doesn't list. Add them there "
        "with the backends that implement them, or the call 404s for whoever isn't running the "
        f"one that does: {undocumented}"
    )

    uncalled = sorted(f"{m} {p}" for m, p in documented - called)
    assert uncalled == [], (
        "contracts/sdk-endpoints.json lists endpoints this SDK no longer calls. Drop the rows, or "
        f"restore the calls: {uncalled}"
    )


def test_no_public_client_method_is_unaccounted_for(client: AgentX) -> None:
    """Guards the drivers above: a new public method that calls the API is caught here rather than
    quietly going untested until someone notices the 404."""
    targets = [
        (AgentX, client),
        *[
            (type(sub), sub)
            for sub in (
                client.evaluations,
                client.evaluations.datasets,
                client.evaluations.settings,
                client.evaluations.prompts,
                client.monitor.patterns,
                client.monitor.signals,
                client.monitor.profile,
                client.monitor.online_evaluators,
                client.tracer,
                # Absent on a checkout behind origin/main; the diagnosis test above is the one
                # that should fail for that, not this.
                getattr(client, "outcomes", None),
                getattr(client, "feedback", None),
            )
            if sub is not None
        ],
        (Agent, None),
        (Conversation, None),
        (Workforce, None),
    ]

    driven = set(_drivers(client))
    uncovered = []
    for cls, _instance in targets:
        for name, attr in vars(cls).items():
            # Inner classes (pydantic's `Config`) are callable too, and are not API surface.
            if name.startswith("_") or not callable(attr) or isinstance(attr, type):
                continue
            qualified = f"{cls.__name__}.{name}"
            if qualified in _NO_NEW_ENDPOINT:
                continue
            if not any(f".{name}(" in label or label.endswith(f".{name}()") for label in driven):
                uncovered.append(qualified)

    assert uncovered == [], (
        "public SDK methods with no contract coverage — drive them in _drivers() and add their "
        f"endpoints to contracts/sdk-endpoints.json, or list them in _NO_NEW_ENDPOINT: {uncovered}"
    )
