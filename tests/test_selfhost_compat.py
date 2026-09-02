"""SDK-to-self-host compatibility matrix.

Every public SDK surface is exercised against a LIVE self-host engine and must land in
exactly one of two tables:

- SELF_HOST: the surface must answer without raising. Any exception fails the test with
  the surface's name, so a "fictional" surface (one the engine never grew) can't ship.
- HOSTED_ONLY: the surface must KEEP failing against the engine AND carry a documented
  reason (plus a docs-file banner that says so). If the engine grows the surface, the
  test fails loudly telling us to promote the entry to SELF_HOST.

Opt-in, like the engine's own backend suites: the whole module skips unless both env
vars below are set.

How to run:
    1. Boot a scratch engine (any free port, throwaway home dir):
           cd AgentX-trace-eval/engine
           PORT=4799 AGENTX_HOME=$(mktemp -d) yarn dev
    2. Copy the "Default project API key: agtx_local_..." line from its startup log.
    3. Run the suite:
           AGENTX_COMPAT_BASE_URL=http://localhost:4799/api/v1 \
           AGENTX_COMPAT_API_KEY=agtx_local_... \
           pytest tests/test_selfhost_compat.py -q

The scratch engine usually has no judge/provider keys. That is fine and deliberate:
judge-dependent steps (eval-run scoring) then record results as skipped/unrated, and this
suite only asserts that every surface ANSWERS, never that the judge liked the answer.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest

BASE_URL = os.getenv("AGENTX_COMPAT_BASE_URL")
API_KEY = os.getenv("AGENTX_COMPAT_API_KEY")

pytestmark = pytest.mark.skipif(
    not (BASE_URL and API_KEY),
    reason="self-host compat suite is opt-in: set AGENTX_COMPAT_BASE_URL and AGENTX_COMPAT_API_KEY",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _tag() -> str:
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Shared state: one client, plus lazily created artifacts (a sync trace, one
# full eval run) reused across parametrized tests so the suite stays fast.
# ---------------------------------------------------------------------------


class CompatContext:
    def __init__(self) -> None:
        # Quiet the runner's interactive spinner/banner in test output.
        os.environ.setdefault("AGENTX_EVAL_QUIET", "1")
        from agentx import AgentX

        self.client = AgentX(api_key=API_KEY, base_url=BASE_URL)
        self.session_id = f"compat-{_tag()}"
        self._trace_id: str | None = None
        self._eval: dict | None = None
        self.created_dataset_ids: list[str] = []

    # -- lazy shared artifacts ------------------------------------------------

    def trace_id(self) -> str:
        """One sync-ingested trace, created on first use (sync=True so the id exists)."""
        if self._trace_id is None:
            with self.client.tracer.trace(
                "compat-check-agent",
                input={"query": "compat ping"},
                session_id=self.session_id,
                sync=True,
                monitor=False,
            ) as span:
                span.output = "compat pong"
            assert span.trace_id, "tracer.trace(sync=True) exited without a trace_id"
            self._trace_id = span.trace_id
        return self._trace_id

    def eval_artifacts(self) -> dict:
        """One full evaluation lifecycle, run once: dataset publish -> init_run (via
        client.evaluations.run) -> execute/submit one result -> finalize -> gate.
        A keyless engine records the result unrated; every surface must still answer."""
        if self._eval is None:
            ds = (
                self.client.evaluations.datasets.builder(
                    name=f"compat-ds-{_tag()}",
                    description="scratch dataset for the self-host compat matrix",
                )
                .add_case("What is 2 + 2?", expected_results="4")
                .publish()
            )
            self.created_dataset_ids.append(ds.id)
            run_ctx = self.client.evaluations.run(
                ds.id, {"kind": "custom_agent", "displayName": "compat-check"}
            )
            run_ctx.execute(lambda case: "4")
            run_ctx.finalize()
            gate = run_ctx.gate(fail_under=0.0)
            self._eval = {"dataset_id": ds.id, "run_id": run_ctx.run_id, "gate": gate}
        return self._eval

    # -- cleanup --------------------------------------------------------------

    def cleanup(self) -> None:
        for dataset_id in self.created_dataset_ids:
            try:
                self.client.evaluations.datasets.delete(dataset_id)
            except Exception:
                pass


@pytest.fixture(scope="module")
def compat():
    ctx = CompatContext()
    yield ctx
    ctx.cleanup()


# ---------------------------------------------------------------------------
# SELF_HOST checks - each must answer without raising
# ---------------------------------------------------------------------------


def _check_ping(ctx: CompatContext) -> None:
    result = ctx.client.ping()
    assert result.get("ok") is True


def _check_tracer_sync_trace_and_flush(ctx: CompatContext) -> None:
    assert ctx.trace_id()
    assert ctx.client.tracer.flush(timeout=10) is True


def _check_traces_get(ctx: CompatContext) -> None:
    detail = ctx.client.traces.get(ctx.trace_id())
    assert isinstance(detail, dict) and detail


def _check_traces_list(ctx: CompatContext) -> None:
    ctx.trace_id()  # make sure at least one trace exists
    page = ctx.client.traces.list(limit=5)
    assert isinstance(page.get("traces"), list) and page["traces"]


def _check_monitor_kpis(ctx: CompatContext) -> None:
    assert isinstance(ctx.client.monitor.kpis(), dict)


def _check_monitor_metrics(ctx: CompatContext) -> None:
    assert isinstance(ctx.client.monitor.metrics(window="1h"), dict)


def _check_monitor_topics(ctx: CompatContext) -> None:
    assert isinstance(ctx.client.monitor.topics(), dict)


def _check_monitor_list_agents(ctx: CompatContext) -> None:
    ctx.trace_id()  # tracing auto-creates the agent
    agents = ctx.client.monitor.list_agents()
    assert isinstance(agents, list)


def _check_monitor_patterns(ctx: CompatContext) -> None:
    # builder + publish + get + list. The SDK exposes no pattern delete, so the
    # published pattern stays behind on the scratch engine (throwaway by design).
    pattern = ctx.client.monitor.patterns.builder(
        name=f"compat-pattern-{_tag()}",
        detector_kind="contains",
        include_terms=["compat-term-that-never-matches"],
        enabled=False,
    ).publish()
    assert pattern.id
    assert ctx.client.monitor.patterns.get(pattern.id).id == pattern.id
    assert any(p.id == pattern.id for p in ctx.client.monitor.patterns.list())


def _check_monitor_judge_scorers_round_trip(ctx: CompatContext) -> None:
    scorer = ctx.client.monitor.judge_scorers.builder(
        name=f"compat-scorer-{_tag()}",
        acceptance_criteria="The answer is correct.",
    ).publish()
    try:
        assert ctx.client.monitor.judge_scorers.get(scorer.id).id == scorer.id
        updated = ctx.client.monitor.judge_scorers.update(
            scorer.id,
            online={"enabled": False, "sampleRate": 0.1, "alertThreshold": 5},
        )
        assert updated.id == scorer.id
    finally:
        ctx.client.monitor.judge_scorers.delete(scorer.id)


def _check_monitor_scorers_list(ctx: CompatContext) -> None:
    assert isinstance(ctx.client.monitor.scorers.list(), list)


def _check_monitor_review_queue_list(ctx: CompatContext) -> None:
    assert isinstance(ctx.client.monitor.review_queue.list(status="all"), list)


def _check_monitor_rules_list(ctx: CompatContext) -> None:
    assert isinstance(ctx.client.monitor.rules.list(), list)


def _check_monitor_sessions_spans(ctx: CompatContext) -> None:
    ctx.trace_id()  # ingests one span into ctx.session_id
    spans = ctx.client.monitor.sessions.spans(ctx.session_id)
    assert isinstance(spans, list) and spans


def _check_evaluations_dataset_round_trip(ctx: CompatContext) -> None:
    ds = (
        ctx.client.evaluations.datasets.builder(
            name=f"compat-ds-roundtrip-{_tag()}",
            description="round-trip dataset (deleted by this test)",
        )
        .add_case("Name a prime number.", expected_results="Any prime, e.g. 7")
        .publish()
    )
    assert ds.id
    fetched = ctx.client.evaluations.datasets.get(ds.id)
    assert fetched.id == ds.id and len(fetched.questions) == 1
    ctx.client.evaluations.datasets.delete(ds.id)


def _check_evaluations_run_lifecycle(ctx: CompatContext) -> None:
    artifacts = ctx.eval_artifacts()
    assert artifacts["run_id"]
    # Keyless judge => unrated results => gate answers but may not pass. Both fine.
    assert isinstance(artifacts["gate"].passed, bool)


def _check_evaluations_get_run(ctx: CompatContext) -> None:
    run = ctx.client.evaluations.get_run(ctx.eval_artifacts()["run_id"])
    assert isinstance(run, dict) and run


def _check_evaluations_list_gates(ctx: CompatContext) -> None:
    ctx.eval_artifacts()  # records one gate verdict
    assert isinstance(ctx.client.evaluations.list_gates(), list)


def _check_evaluations_prompts_registry(ctx: CompatContext) -> None:
    name = f"compat-prompt-{_tag()}"
    created = ctx.client.evaluations.prompts.create(
        name, "You are a compat-check assistant.", description="compat matrix scratch prompt"
    )
    assert created.version >= 1
    fetched = ctx.client.evaluations.prompts.get(name)
    assert fetched.name == name and fetched.text
    assert any(p.name == name for p in ctx.client.evaluations.prompts.list())


def _check_feedback_report(ctx: CompatContext) -> None:
    report = ctx.client.feedback.report(
        trace_id=ctx.trace_id(), rating="up", end_user_id="compat-suite"
    )
    assert isinstance(report, dict)


def _check_outcomes_report(ctx: CompatContext) -> None:
    report = ctx.client.outcomes.report(
        trace_id=ctx.trace_id(),
        outcome="confirmed_good",
        is_negative=False,
        reported_by="compat-suite",
    )
    assert isinstance(report, dict)


def _check_export_manifest_and_iter(ctx: CompatContext) -> None:
    ctx.trace_id()  # at least one exportable row
    manifest = ctx.client.export.manifest()
    assert isinstance(manifest, list) and manifest
    rows = list(ctx.client.export.iter("traces"))
    assert rows and all(isinstance(r, dict) for r in rows)


SELF_HOST = [
    ("client.ping", _check_ping),
    ("tracer.trace(sync=True) + tracer.flush", _check_tracer_sync_trace_and_flush),
    ("traces.get", _check_traces_get),
    ("traces.list", _check_traces_list),
    ("monitor.kpis", _check_monitor_kpis),
    ("monitor.metrics", _check_monitor_metrics),
    ("monitor.topics", _check_monitor_topics),
    ("monitor.list_agents", _check_monitor_list_agents),
    ("monitor.patterns builder/publish/get/list", _check_monitor_patterns),
    ("monitor.judge_scorers create/get/update/delete", _check_monitor_judge_scorers_round_trip),
    ("monitor.scorers.list", _check_monitor_scorers_list),
    ("monitor.review_queue.list", _check_monitor_review_queue_list),
    ("monitor.rules.list", _check_monitor_rules_list),
    ("monitor.sessions.spans", _check_monitor_sessions_spans),
    ("evaluations.datasets builder/publish/get/delete", _check_evaluations_dataset_round_trip),
    ("evaluations run/execute/finalize/gate", _check_evaluations_run_lifecycle),
    ("evaluations.get_run", _check_evaluations_get_run),
    ("evaluations.list_gates", _check_evaluations_list_gates),
    ("evaluations.prompts create/get/list", _check_evaluations_prompts_registry),
    ("feedback.report", _check_feedback_report),
    ("outcomes.report", _check_outcomes_report),
    ("export.manifest + export.iter", _check_export_manifest_and_iter),
]


@pytest.mark.parametrize(
    "surface,check", SELF_HOST, ids=[name for name, _ in SELF_HOST]
)
def test_self_host_surface(surface, check, compat):
    try:
        check(compat)
    except AssertionError:
        raise
    except Exception as exc:
        pytest.fail(
            f"self-host surface {surface!r} raised {type(exc).__name__}: {exc} "
            "(either the SDK or the engine drifted - this surface is supposed to work "
            "against a self-host engine)"
        )


# ---------------------------------------------------------------------------
# HOSTED_ONLY checks - each must KEEP failing against the engine, and the docs
# must say so. If one starts working, promote it to SELF_HOST above.
# ---------------------------------------------------------------------------

CI_BANNER_NEEDLE = "Hosted platform only."
CI_ROUTE_NEEDLE = "/ingest/ci-runs"


def _call_run_eval(ctx: CompatContext):
    return ctx.client.tracer.run_eval("evds_compat_missing", lambda q: "answer")


def _call_create_ci_run(ctx: CompatContext):
    return ctx.client.tracer.create_ci_run("evds_compat_missing")


def _call_get_ci_run(ctx: CompatContext):
    return ctx.client.tracer.get_ci_run("cirun_compat_missing")


def _call_finalize_ci_run(ctx: CompatContext):
    return ctx.client.tracer.finalize_ci_run("cirun_compat_missing")


def _call_list_models(ctx: CompatContext):
    return ctx.client.evaluations.list_models()


HOSTED_ONLY = [
    (
        "tracer.run_eval",
        _call_run_eval,
        "Targets the hosted /ingest/ci-runs API; the self-host engine does not serve it. "
        "Self-host CI gating is evaluations.run(...).execute(...).finalize().gate(...).",
        "CICD_EVAL.md",
        (CI_BANNER_NEEDLE, CI_ROUTE_NEEDLE),
    ),
    (
        "tracer.create_ci_run",
        _call_create_ci_run,
        "Low-level hosted /ingest/ci-runs call; 404 on self-host, surfaced as DatasetNotFound.",
        "CICD_EVAL.md",
        (CI_BANNER_NEEDLE, CI_ROUTE_NEEDLE),
    ),
    (
        "tracer.get_ci_run",
        _call_get_ci_run,
        "Low-level hosted /ingest/ci-runs call; 404 on self-host, surfaced as DatasetNotFound.",
        "CICD_EVAL.md",
        (CI_BANNER_NEEDLE, CI_ROUTE_NEEDLE),
    ),
    (
        "tracer.finalize_ci_run",
        _call_finalize_ci_run,
        "Low-level hosted /ingest/ci-runs call; 404 on self-host, surfaced as DatasetNotFound.",
        "CICD_EVAL.md",
        (CI_BANNER_NEEDLE, CI_ROUTE_NEEDLE),
    ),
    (
        "evaluations.list_models",
        _call_list_models,
        "Targets the hosted /custom-agent-evaluations/models registry; the engine explicitly "
        "has not ported it (engine routes/evaluations.ts: 'Still not ported: list_models').",
        "EVALUATIONS.md",
        ("`list_models()` is **hosted platform only**",),
    ),
]


@pytest.mark.parametrize(
    "surface,call,reason,doc_file,doc_needles",
    HOSTED_ONLY,
    ids=[name for name, *_ in HOSTED_ONLY],
)
def test_hosted_only_surface(surface, call, reason, doc_file, doc_needles, compat):
    assert reason and reason.strip(), f"hosted-only entry {surface!r} must document why"

    try:
        call(compat)
    except Exception:
        pass  # expected: the engine does not serve this surface
    else:
        pytest.fail(
            f"hosted-only surface {surface!r} SUCCEEDED against the self-host engine. "
            "The engine grew this surface: move the entry to SELF_HOST and update the docs."
        )

    doc_text = (REPO_ROOT / doc_file).read_text(encoding="utf-8")
    for needle in doc_needles:
        assert needle in doc_text, (
            f"hosted-only surface {surface!r}: expected {doc_file} to contain {needle!r} "
            "so the limitation stays documented"
        )
