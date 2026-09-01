"""
Platform-agnostic framework capture: explicit framework= always wins, integrations stamp their
literal, and - new - the SDK auto-detects the single unambiguous orchestration framework
imported in the process (agentx/tracing/framework_detect.py). Also pins the two capture-gap
fixes: _merge_child_run adopts the framework BEFORE emitting child spans (CrewAI/AutoGen
children used to go out unlabeled), and _record_llm_call forwards the patched client's provider
literal onto a user-opened span that has no label of its own.

Same harness as test_span_tree.py: only the ingest_client boundary is mocked, so the real
_send/_dispatch/child_span paths run and every wire dict is inspectable.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from agentx.tracing.framework_detect import _ORCHESTRATOR_MODULES, detect_framework
from agentx.tracing.tracer import Tracer


def make_tracer() -> Tracer:
    return Tracer(ingest_client=MagicMock())


def enqueued_wires(tracer: Tracer) -> list:
    return [call.args[0] for call in tracer._client.enqueue.call_args_list]


@pytest.fixture()
def clean_modules(monkeypatch):
    """Remove every known orchestrator from sys.modules so each test states its own world."""
    for module in list(_ORCHESTRATOR_MODULES) + ["agents.run", "agents.tracing"]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    return monkeypatch


def fake_import(monkeypatch, name: str) -> None:
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


def test_detects_single_imported_orchestrator(clean_modules):
    fake_import(clean_modules, "crewai")
    assert detect_framework() == "crewai"
    tracer = make_tracer()
    with tracer.trace("agent"):
        pass
    assert enqueued_wires(tracer)[0]["framework"] == "crewai"


def test_no_orchestrator_means_no_label(clean_modules):
    assert detect_framework() is None
    tracer = make_tracer()
    with tracer.trace("agent"):
        pass
    assert "framework" not in enqueued_wires(tracer)[0]


def test_ambiguous_imports_stay_unlabeled(clean_modules):
    fake_import(clean_modules, "crewai")
    fake_import(clean_modules, "llama_index")
    assert detect_framework() is None


def test_langchain_family_collapses_to_one_literal(clean_modules):
    # langgraph + langchain_core together are ONE framework, not an ambiguity.
    fake_import(clean_modules, "langchain_core")
    fake_import(clean_modules, "langgraph")
    assert detect_framework() == "langchain"


def test_agents_module_needs_sdk_submodules(clean_modules):
    fake_import(clean_modules, "agents")  # could be anyone's package named "agents"
    assert detect_framework() is None
    fake_import(clean_modules, "agents.run")
    assert detect_framework() == "openai-agents"


def test_explicit_framework_beats_detection(clean_modules):
    fake_import(clean_modules, "crewai")
    tracer = make_tracer()
    with tracer.trace("agent", framework="my-inhouse-runner"):
        pass
    assert enqueued_wires(tracer)[0]["framework"] == "my-inhouse-runner"


def test_children_inherit_detected_framework(clean_modules):
    fake_import(clean_modules, "llama_index")
    tracer = make_tracer()
    with tracer.trace("agent") as span:
        span.child_span("step", duration_ms=5)
    wires = enqueued_wires(tracer)
    assert [w.get("framework") for w in wires] == ["llamaindex", "llamaindex"]


def test_merge_child_run_adopts_before_emitting_children(clean_modules):
    # The CrewAI/AutoGen shape: root opened with no framework, the merged sub-run carries it.
    # Children must go out labeled too - adoption used to happen after emission.
    tracer = make_tracer()
    with tracer.trace("crew") as span:
        span._merge_child_run(
            framework="crewai",
            execution_steps=[{"duration_ms": 10, "input": "q", "output": "a"}],
        )
    wires = enqueued_wires(tracer)
    assert len(wires) == 2
    assert all(w.get("framework") == "crewai" for w in wires)


def test_record_llm_call_stamps_provider_on_unlabeled_span(clean_modules):
    tracer = make_tracer()
    with tracer.trace("agent") as span:
        span._record_llm_call(duration_ms=7, model="claude-x", framework="anthropic")
    root = enqueued_wires(tracer)[-1]
    assert root["framework"] == "anthropic"


def test_record_llm_call_never_overrides_explicit_framework(clean_modules):
    tracer = make_tracer()
    with tracer.trace("agent", framework="langchain") as span:
        span._record_llm_call(duration_ms=7, model="gpt-x", framework="openai")
    root = enqueued_wires(tracer)[-1]
    assert root["framework"] == "langchain"
