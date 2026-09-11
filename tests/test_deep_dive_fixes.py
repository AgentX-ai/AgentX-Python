"""Regressions for deep-dive round 3's SDK bugs (sample-scripts/eval_framework_deep_dive/
DEEP_DIVE_REPORT.md): bug #1 (base_url was process-global via os.environ, so the
last-constructed client silently re-pointed every other client) and bug #5 (flush(timeout)
called queue.join() with no deadline and dropped traces silently)."""

import logging
import os
import time

import pytest
import requests

from agentx import AgentX
from agentx.tracing.ingest_client import IngestClient


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("AGENTX_API_BASE_URL", raising=False)
    monkeypatch.setenv("AGENTX_API_KEY", "agtx_test_key")
    yield


def test_second_client_does_not_repoint_the_first():
    a = AgentX(api_key="k1", base_url="http://engine-a:1111/api/v1")
    b = AgentX(api_key="k2", base_url="http://engine-b:2222/api/v1")

    # The constructor must not leak its base into process-global state...
    assert os.environ.get("AGENTX_API_BASE_URL") is None

    # ...and every per-call-resolving surface of client A must still point at engine A.
    for sub in (a.projects, a.traces, a.export, a.feedback, a.outcomes, a.monitor.scorers):
        assert "engine-a:1111" in sub._base_url, type(sub).__name__
    for sub in (b.projects, b.traces, b.export, b.feedback, b.outcomes, b.monitor.scorers):
        assert "engine-b:2222" in sub._base_url, type(sub).__name__


def test_env_var_still_works_as_a_default(monkeypatch):
    monkeypatch.setenv("AGENTX_API_BASE_URL", "http://from-env:3333/api/v1")
    c = AgentX(api_key="k")
    assert "from-env:3333" in c.projects._base_url


def _stalled_client(monkeypatch, per_call_delay: float) -> IngestClient:
    client = IngestClient(api_key="k", sdk_version="test", base_url="http://localhost:9/api/v1")

    def slow_post(*args, **kwargs):
        time.sleep(per_call_delay)
        raise ConnectionError("engine down")

    monkeypatch.setattr(client._session, "post", slow_post)
    return client


def test_flush_timeout_is_a_wall_clock_bound(monkeypatch):
    client = _stalled_client(monkeypatch, per_call_delay=2.0)
    for i in range(5):
        client.enqueue({"name": f"t{i}"})

    t0 = time.time()
    drained = client.flush(timeout=1.0)
    elapsed = time.time() - t0

    assert drained is False
    # The old queue.join() implementation blocked for the full retry schedule of every queued
    # item (measured 140s for 20 items); the deadline must now bind within a small margin.
    assert elapsed < 3.0, f"flush(timeout=1.0) blocked for {elapsed:.1f}s"


def test_dropped_traces_warn_instead_of_vanishing(monkeypatch, caplog):
    client = IngestClient(api_key="k", sdk_version="test", base_url="http://localhost:9/api/v1")

    def refuse(*args, **kwargs):
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(client._session, "post", refuse)
    with caplog.at_level(logging.WARNING, logger="agentx.tracing.ingest_client"):
        client.enqueue({"name": "doomed"})
        for _ in range(200):  # wait out the retry backoff
            if client._dropped >= 1:
                break
            time.sleep(0.1)

    assert client._dropped >= 1
    assert any("dropped a trace" in rec.message for rec in caplog.records)


def test_evaluations_shaped_base_url_works_for_every_subclient():
    """P0 regression: a base_url carrying the /custom-agent-evaluations suffix (what users copy
    out of eval env files) used to 404 every non-eval sub-client - AgentX.__init__ now
    normalizes it (trailing slash + suffix stripped) before handing it out."""
    c = AgentX(api_key="k", base_url="http://engine:4700/api/v1/custom-agent-evaluations/")

    assert c.base_url == "http://engine:4700/api/v1"
    assert c.tracer._client._endpoint == "http://engine:4700/api/v1/ingest/traces"
    assert c.monitor._base_url == "http://engine:4700/api/v1/monitor"
    for sub in (c.projects, c.traces, c.export, c.feedback, c.outcomes):
        assert sub._base_url == "http://engine:4700/api/v1", type(sub).__name__
    # The evaluations client re-appends its own suffix exactly once.
    assert c.evaluations._client._base_url == "http://engine:4700/api/v1/custom-agent-evaluations"


def test_env_var_with_evaluations_suffix_is_normalized_too(monkeypatch):
    monkeypatch.setenv("AGENTX_API_BASE_URL", "http://from-env:3333/api/v1/custom-agent-evaluations")
    c = AgentX(api_key="k")
    assert c.base_url == "http://from-env:3333/api/v1"
    assert c.tracer._client._endpoint == "http://from-env:3333/api/v1/ingest/traces"


def test_ingest_client_close_stops_the_worker():
    """P2: close() enqueues the sentinel and joins the worker so a torn-down client leaves no
    thread behind; AgentX exposes it (and context-manager form) on top."""
    client = IngestClient(api_key="k", sdk_version="test", base_url="http://localhost:9/api/v1")
    assert client.close(timeout=2.0) is True
    assert not client._worker.is_alive()
    # Idempotent.
    assert client.close(timeout=1.0) is True


def test_agentx_context_manager_closes_the_ingest_worker():
    with AgentX(api_key="k", base_url="http://localhost:9/api/v1") as c:
        worker = c._ingest_client._worker
        assert worker.is_alive()
    assert not worker.is_alive()


def test_constructor_no_longer_writes_the_api_key_into_the_environment(monkeypatch):
    monkeypatch.delenv("AGENTX_API_KEY", raising=False)
    AgentX(api_key="k-secret", base_url="http://localhost:9/api/v1")
    assert os.environ.get("AGENTX_API_KEY") is None
