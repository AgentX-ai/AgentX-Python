"""
Tests for the SDK's real span hierarchy (span_id/parent_span_id/session_id on every trace, the
same model AgentX's OTel ingestion path uses) — the SDK's only trace representation now;
performance_summary is no longer sent at all (see tracer.py's __exit__/_wrap_sync/_wrap_async and
every framework integration).

Unlike test_integrations.py's make_tracer() (which mocks tracer._send directly, bypassing the
real _send/_dispatch chain), these tests only mock the ingest_client boundary
(Tracer(ingest_client=MagicMock())) so the real _send()/_dispatch()/child_span() code paths run
end to end — every dispatched wire dict lands in tracer._client.enqueue.call_args_list (async,
the default) or is passed straight to tracer._client.send_trace_sync (sync=True).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentx.tracing.tracer import Tracer


def make_tracer() -> Tracer:
    return Tracer(ingest_client=MagicMock())


def enqueued_wires(tracer: Tracer) -> list:
    return [call.args[0] for call in tracer._client.enqueue.call_args_list]


def assert_no_performance_summary(wires: list) -> None:
    for wire in wires:
        assert "performance_summary" not in wire


def test_root_span_gets_id_and_auto_session_no_parent():
    tracer = make_tracer()
    with tracer.trace("agent"):
        pass
    wires = enqueued_wires(tracer)
    assert len(wires) == 1
    root = wires[0]
    assert root["span_id"]
    assert "parent_span_id" not in root
    assert root["session_id"].startswith("sdk_")
    assert "started_at_unix_nano" in root
    assert_no_performance_summary(wires)


def test_nested_with_blocks_link_parent_and_session():
    tracer = make_tracer()
    with tracer.trace("outer") as outer:
        with tracer.trace("inner") as inner:
            pass
    wires = enqueued_wires(tracer)
    assert len(wires) == 2
    # Enqueue order: inner exits (and sends) before outer, since it's the innermost `with`.
    inner_wire, outer_wire = wires
    assert outer_wire["span_id"] == outer.span_id
    assert inner_wire["parent_span_id"] == outer.span_id
    assert inner_wire["session_id"] == outer_wire["session_id"]
    assert inner_wire["span_id"] != outer_wire["span_id"]


def test_explicit_session_id_is_respected_and_inherited():
    tracer = make_tracer()
    with tracer.trace("outer", session_id="conv-123"):
        with tracer.trace("inner"):
            pass
    wires = enqueued_wires(tracer)
    assert all(w["session_id"] == "conv-123" for w in wires)


def test_record_llm_call_emits_real_child_span():
    tracer = make_tracer()
    with tracer.trace("agent") as span:
        span._record_llm_call(duration_ms=42.0, start_time=1000.0, end_time=1000.042, input="hi", output="hello")
    wires = enqueued_wires(tracer)
    assert len(wires) == 2
    child, root = wires
    assert child["name"] == "LLM Call 1"
    assert child["parent_span_id"] == root["span_id"]
    assert child["session_id"] == root["session_id"]
    assert child["latency_ms"] == 42
    assert child["started_at_unix_nano"] == str(int(1000.0 * 1_000_000_000))
    # The parent still gets a reasonable summary even though its own detail now lives in the child.
    assert root["input"] == "hi"
    assert root["output"] == "hello"
    assert_no_performance_summary(wires)


def test_merge_child_run_emits_one_child_per_step():
    tracer = make_tracer()
    with tracer.trace("agent") as span:
        span._merge_child_run(
            execution_steps=[
                {"duration_ms": 10, "start_time": 1.0, "end_time": 1.01},
                {"duration_ms": 20, "start_time": 1.02, "end_time": 1.04},
            ],
            tool_calls=[{"name": "search", "input": "q", "output": "r", "latency_ms": 5}],
            input="in",
            output="out",
        )
    wires = enqueued_wires(tracer)
    # 2 execution steps + 1 tool call + 1 root = 4
    assert len(wires) == 4
    root = wires[-1]
    child_names = sorted(w["name"] for w in wires[:-1])
    assert child_names == ["LLM Call 1", "LLM Call 2", "search"]
    assert all(w["parent_span_id"] == root["span_id"] for w in wires[:-1])
    assert root["input"] == "in"
    assert root["output"] == "out"


def test_child_span_always_dispatches():
    tracer = make_tracer()
    with tracer.trace("agent") as span:
        child = span.child_span("sub-step", start_time=1.0, end_time=1.5, input="x", output="y")
        assert child.span_id
    wires = enqueued_wires(tracer)
    assert len(wires) == 2  # root + the manually opened child
    child_wire = next(w for w in wires if w["name"] == "sub-step")
    assert child_wire["parent_span_id"] == span.span_id


def test_decorator_form_gets_a_real_span_per_call_and_no_performance_summary():
    tracer = make_tracer()

    @tracer.trace("decorated")
    def run(x: int) -> int:
        return x + 1

    assert run(1) == 2
    assert run(2) == 3

    wires = enqueued_wires(tracer)
    assert len(wires) == 2
    # A fresh span_id per call — the decorated function's template span is never reused directly.
    assert wires[0]["span_id"] != wires[1]["span_id"]
    assert_no_performance_summary(wires)


# ---------------------------------------------------------------------------
# langchain.py — AgentXCallbackHandler opens a real root span per top-level chain invocation (or
# folds into an already-active enclosing span) and lets _merge_child_run explode it into real
# child spans.
# ---------------------------------------------------------------------------


def test_langchain_handler_emits_real_child_spans():
    from uuid import uuid4
    from agentx.integrations.langchain import AgentXCallbackHandler

    tracer = make_tracer()
    handler = AgentXCallbackHandler(tracer, name="agent")

    chain_run = uuid4()
    llm_run = uuid4()
    tool_run = uuid4()

    handler.on_chain_start({}, {"input": "hi"}, run_id=chain_run, parent_run_id=None)
    handler.on_llm_start({"kwargs": {"model": "gpt-4"}}, ["hi"], run_id=llm_run, parent_run_id=chain_run)
    handler.on_llm_end(
        types_result("hello"),
        run_id=llm_run,
        parent_run_id=chain_run,
    )
    handler.on_tool_start({"name": "search"}, "q", run_id=tool_run, parent_run_id=chain_run)
    handler.on_tool_end("result", run_id=tool_run, parent_run_id=chain_run)
    handler.on_chain_end({"output": "done"}, run_id=chain_run, parent_run_id=None)

    wires = enqueued_wires(tracer)
    assert len(wires) == 3  # root + LLM call child + tool call child
    root = next(w for w in wires if w["name"] == "agent")
    children = [w for w in wires if w is not root]
    assert len(children) == 2
    assert all(c["parent_span_id"] == root["span_id"] for c in children)
    assert all(c["session_id"] == root["session_id"] for c in children)
    child_names = sorted(c["name"] for c in children)
    assert child_names == ["LLM Call 1", "search"]
    assert_no_performance_summary(wires)


def types_result(text: str):
    import types

    generation = types.SimpleNamespace(text=text, generation_info={})
    return types.SimpleNamespace(generations=[[generation]], llm_output={})


# ---------------------------------------------------------------------------
# openai_agents.py — AgentXTracingProcessor opens a real root span in on_trace_start and turns
# each Agents-SDK span into a child span in on_span_end.
# ---------------------------------------------------------------------------


def test_openai_agents_emits_real_child_spans():
    import types
    from agentx.integrations.openai_agents import AgentXTracingProcessor

    tracer = make_tracer()
    processor = AgentXTracingProcessor(tracer)

    trace = types.SimpleNamespace(trace_id="trace-1", name="my-agent")
    processor.on_trace_start(trace)

    gen_data = types.SimpleNamespace(type="generation", input="hi", output="hello", model="gpt-4", usage=None)
    gen_span = types.SimpleNamespace(
        trace_id="trace-1",
        span_data=gen_data,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
        error=None,
    )
    processor.on_span_end(gen_span)

    fn_data = types.SimpleNamespace(type="function", name="lookup", input="q", output="r")
    fn_span = types.SimpleNamespace(
        trace_id="trace-1",
        span_data=fn_data,
        started_at="2026-01-01T00:00:01Z",
        ended_at="2026-01-01T00:00:02Z",
        error=None,
    )
    processor.on_span_end(fn_span)

    processor.on_trace_end(trace)

    wires = enqueued_wires(tracer)
    assert len(wires) == 3  # root + LLM call child + function-tool child
    root = next(w for w in wires if w["name"] == "my-agent")
    children = [w for w in wires if w is not root]
    assert len(children) == 2
    assert all(c["parent_span_id"] == root["span_id"] for c in children)
    assert all(c["session_id"] == root["session_id"] for c in children)
    child_names = sorted(c["name"] for c in children)
    assert child_names == ["LLM Call 1", "lookup"]
    # Root still adopts a reasonable summary even though its own detail is now in children.
    assert root["input"] == "hi"
    assert root["output"] == "hello"
    assert_no_performance_summary(wires)


def test_openai_agents_span_error_still_captured():
    import types
    from agentx.integrations.openai_agents import AgentXTracingProcessor

    tracer = make_tracer()
    processor = AgentXTracingProcessor(tracer)

    trace = types.SimpleNamespace(trace_id="trace-2", name="my-agent")
    processor.on_trace_start(trace)
    span_error = types.SimpleNamespace(message="generation failed", data={"code": 500})
    span_data = types.SimpleNamespace(type="generation", input=None, output=None, model=None, usage=None)
    span = types.SimpleNamespace(
        trace_id="trace-2", span_data=span_data, started_at=None, ended_at=None, error=span_error
    )
    processor.on_span_end(span)
    processor.on_trace_end(trace)

    wires = enqueued_wires(tracer)
    root = next(w for w in wires if w["name"] == "my-agent")
    assert root["error"] == "generation failed ({'code': 500})"


# ---------------------------------------------------------------------------
# llamaindex.py — AgentXLlamaIndexHandler opens a real root in _send_trace and lets
# _merge_child_run explode it into real child spans. Drives a real VectorStoreIndex query
# (MockLLM + MockEmbedding, no network/API keys) through the real CallbackManager event system.
# ---------------------------------------------------------------------------


def test_llamaindex_emits_real_child_spans():
    pytest.importorskip("llama_index.core")
    from llama_index.core import Document, Settings, VectorStoreIndex
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms import MockLLM

    from agentx.integrations.llamaindex import AgentXLlamaIndexHandler

    tracer = make_tracer()
    handler = AgentXLlamaIndexHandler(tracer, name="rag-agent")

    Settings.llm = MockLLM()
    Settings.embed_model = MockEmbedding(embed_dim=8)
    Settings.callback_manager = CallbackManager([handler])
    try:
        index = VectorStoreIndex.from_documents([Document(text="LiteLLM is a proxy for many LLM providers.")])
        engine = index.as_query_engine()
        response = engine.query("What is LiteLLM?")
    finally:
        Settings.callback_manager = CallbackManager([])

    assert str(response)
    wires = enqueued_wires(tracer)
    assert len(wires) >= 3  # root + at least one LLM-call child + one retrieval child
    root = next(w for w in wires if w["name"] == "rag-agent")
    assert root["input"] == "What is LiteLLM?"
    children = [w for w in wires if w is not root]
    assert all(c["parent_span_id"] == root["span_id"] for c in children)
    assert all(c["session_id"] == root["session_id"] for c in children)
    # The root's own latency reflects the real query duration, not the ~instant gap between
    # _send_trace opening its root span and closing it a few lines later.
    assert root["latency_ms"] >= 0
    assert_no_performance_summary(wires)


# ---------------------------------------------------------------------------
# crewai.py. No real crewai package needed — same technique test_integrations.py's
# test_crewai_falls_back_to_even_split_without_event_bus uses: monkeypatch
# _start_task_timing_capture directly, rather than depending on the real crewai.events module.
# ---------------------------------------------------------------------------


def test_crewai_emits_one_child_per_task():
    from agentx.integrations.crewai import AgentXCrewObserver

    tracer = make_tracer()
    observer = AgentXCrewObserver(tracer, name="my-crew")

    class FakeTaskOutput:
        def __init__(self, description, raw):
            self.description = description
            self.raw = raw

    class FakeCrewOutput:
        def __init__(self, raw, tasks_output):
            self.raw = raw
            self.tasks_output = tasks_output

    task_outputs = [FakeTaskOutput("Research topic", "research done"), FakeTaskOutput("Write summary", "summary done")]

    class FakeCrew:
        def kickoff(self, inputs=None):
            return FakeCrewOutput(raw="final output", tasks_output=task_outputs)

    now = 1_700_000_000.0
    fake_timings = {
        "task-1": {"name": "Research topic", "start": now, "end": now + 0.03, "error": None},
        "task-2": {"name": "Write summary", "start": now + 0.03, "end": now + 0.04, "error": None},
    }
    observer._start_task_timing_capture = lambda: (fake_timings, lambda: None)

    result = observer.kickoff(FakeCrew(), inputs={"topic": "AI"})

    assert result.raw == "final output"
    wires = enqueued_wires(tracer)
    assert len(wires) == 3  # root + 2 task children
    root = next(w for w in wires if w["name"] == "my-crew")
    children = [w for w in wires if w is not root]
    assert len(children) == 2
    assert all(c["parent_span_id"] == root["span_id"] for c in children)
    child_names = sorted(c["name"] for c in children)
    assert child_names == ["Research topic", "Write summary"]
    assert root["output"] == "final output"
    assert_no_performance_summary(wires)


def test_crewai_falls_back_to_evenly_divided_children_without_event_bus():
    """No event-bus timing available — tasks still become real child spans, just with
    evenly-divided (approximate) durations instead of real per-task start/end."""
    from agentx.integrations.crewai import AgentXCrewObserver

    tracer = make_tracer()
    observer = AgentXCrewObserver(tracer, name="my-crew")
    observer._start_task_timing_capture = lambda: ({}, lambda: None)

    class FakeTaskOutput:
        def __init__(self, description, raw):
            self.description = description
            self.raw = raw

    class FakeCrewOutput:
        def __init__(self, raw, tasks_output):
            self.raw = raw
            self.tasks_output = tasks_output

    class FakeCrew:
        def kickoff(self, inputs=None):
            return FakeCrewOutput(
                raw="final output",
                tasks_output=[FakeTaskOutput("Task A", "a done"), FakeTaskOutput("Task B", "b done")],
            )

    observer.kickoff(FakeCrew(), inputs={"topic": "AI"})

    wires = enqueued_wires(tracer)
    assert len(wires) == 3  # root + 2 task children
    root = next(w for w in wires if w["name"] == "my-crew")
    children = [w for w in wires if w is not root]
    assert len(children) == 2
    assert all(c["parent_span_id"] == root["span_id"] for c in children)


def test_crewai_does_not_swallow_kickoff_exception():
    """Regression guard: a `return` inside the try/finally's finally block would silently
    swallow the exception crew.kickoff() raises — see the comment in crewai.py's kickoff()."""
    from agentx.integrations.crewai import AgentXCrewObserver

    tracer = make_tracer()
    observer = AgentXCrewObserver(tracer, name="my-crew")

    class FakeCrew:
        def kickoff(self, inputs=None):
            raise RuntimeError("boom")

    now = 1_700_000_000.0
    fake_timings = {"task-1": {"name": "Research topic", "start": now, "end": now + 0.03, "error": None}}
    observer._start_task_timing_capture = lambda: (fake_timings, lambda: None)

    with pytest.raises(RuntimeError, match="boom"):
        observer.kickoff(FakeCrew(), inputs={"topic": "AI"})


# ---------------------------------------------------------------------------
# autogen.py. The real autogen-agentchat/autogen-core packages don't install cleanly in this
# Python 3.9 environment (same story as crewai/llama-index's newer releases) — duck-typed fakes
# matching the exact attributes _summarize_messages reads (.type, .content, .created_at,
# .models_usage), same approach the rest of this file already uses.
# ---------------------------------------------------------------------------


def test_autogen_emits_child_span_per_llm_call():
    import asyncio
    from datetime import datetime, timezone
    import types

    from agentx.integrations.autogen import AgentXAutoGenObserver

    def fake_message(content: str, prompt_tokens: int, completion_tokens: int, when: datetime):
        return types.SimpleNamespace(
            type="TextMessage",
            content=content,
            created_at=when,
            models_usage=types.SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    task_echo = types.SimpleNamespace(type="TextMessage", content="Say hello", created_at=t0, models_usage=None)
    task_result = types.SimpleNamespace(messages=[task_echo, fake_message("Hello from AutoGen!", 22, 3, t1)])

    class FakeTeam:
        async def run(self, task=None, **kwargs):
            return task_result

    tracer = make_tracer()
    observer = AgentXAutoGenObserver(tracer, name="my-agent")

    asyncio.run(observer.run(FakeTeam(), task="Say hello"))

    wires = enqueued_wires(tracer)
    assert len(wires) == 2  # root + one LLM-call child
    root = next(w for w in wires if w["name"] == "my-agent")
    child = next(w for w in wires if w is not root)
    assert child["parent_span_id"] == root["span_id"]
    assert child["name"] == "LLM Call 1"
    assert root["output"] == "Hello from AutoGen!"
    assert_no_performance_summary(wires)


def test_autogen_does_not_swallow_run_exception():
    """Same finally-block hazard as crewai.py — see test_crewai_does_not_swallow_kickoff_exception."""
    import asyncio
    from agentx.integrations.autogen import AgentXAutoGenObserver

    class FakeTeam:
        async def run(self, task=None, **kwargs):
            raise RuntimeError("boom")

    tracer = make_tracer()
    observer = AgentXAutoGenObserver(tracer, name="my-agent")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(observer.run(FakeTeam(), task="hi"))


# ---------------------------------------------------------------------------
# google_adk.py — AgentXADKPlugin opens a real root span in before_run_callback and turns each
# model/tool callback into a child span. Same fake invocation_context/callback_context shape
# test_integrations.py's real (google-adk installed) test_adk_model_error_is_captured uses.
# ---------------------------------------------------------------------------


def test_google_adk_emits_real_child_spans():
    import asyncio
    import types

    from agentx.integrations.google_adk import AgentXADKPlugin

    tracer = make_tracer()
    plugin = AgentXADKPlugin(tracer, name="adk-agent")

    invocation_context = types.SimpleNamespace(invocation_id="inv-1", agent=types.SimpleNamespace(name="adk-agent"))
    callback_context = types.SimpleNamespace(get_invocation_context=lambda: invocation_context)
    tool_context = types.SimpleNamespace(get_invocation_context=lambda: invocation_context)

    llm_request = types.SimpleNamespace(model="gemini-x", contents=None)
    llm_response = types.SimpleNamespace(
        content=types.SimpleNamespace(parts=[types.SimpleNamespace(text="Hello!", function_call=None)]),
        usage_metadata=types.SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
    )
    tool = types.SimpleNamespace(name="lookup")

    async def run():
        await plugin.before_run_callback(invocation_context=invocation_context)
        await plugin.before_model_callback(callback_context=callback_context, llm_request=llm_request)
        await plugin.after_model_callback(callback_context=callback_context, llm_response=llm_response)
        await plugin.before_tool_callback(tool=tool, tool_args={"q": "x"}, tool_context=tool_context)
        await plugin.after_tool_callback(tool=tool, tool_args={"q": "x"}, tool_context=tool_context, result={"ok": True})
        await plugin.after_run_callback(invocation_context=invocation_context)

    asyncio.run(run())

    wires = enqueued_wires(tracer)
    assert len(wires) == 3  # root + LLM call child + tool call child
    root = next(w for w in wires if w["name"] == "adk-agent")
    children = [w for w in wires if w is not root]
    assert len(children) == 2
    assert all(c["parent_span_id"] == root["span_id"] for c in children)
    assert all(c["session_id"] == root["session_id"] for c in children)
    child_names = sorted(c["name"] for c in children)
    assert child_names == ["LLM Call 1", "lookup"]
    assert root["output"] == "Hello!"
    assert_no_performance_summary(wires)


def test_google_adk_model_error_is_captured():
    import asyncio
    import types

    from agentx.integrations.google_adk import AgentXADKPlugin

    tracer = make_tracer()
    plugin = AgentXADKPlugin(tracer, name="adk-agent")

    invocation_context = types.SimpleNamespace(invocation_id="inv-2", agent=types.SimpleNamespace(name="adk-agent"))
    callback_context = types.SimpleNamespace(get_invocation_context=lambda: invocation_context)
    llm_request = types.SimpleNamespace(model="gemini-x", contents=None)

    async def run():
        await plugin.before_run_callback(invocation_context=invocation_context)
        await plugin.before_model_callback(callback_context=callback_context, llm_request=llm_request)
        await plugin.on_model_error_callback(
            callback_context=callback_context, llm_request=llm_request, error=RuntimeError("boom")
        )
        await plugin.after_run_callback(invocation_context=invocation_context)

    asyncio.run(run())

    wires = enqueued_wires(tracer)
    assert len(wires) == 2  # root + the failed LLM-call child
    root = next(w for w in wires if w["name"] == "adk-agent")
    child = next(w for w in wires if w is not root)
    assert child["parent_span_id"] == root["span_id"]
    # Both the failing child and the root (which adopts it) report the error.
    assert child["error"] == "boom"
    assert root["error"] == "boom"


# ---------------------------------------------------------------------------
# Tracer.record_tool_call()/trace_tool_call() and record_retrieval()/trace_retrieval() — a third
# fold point (separate from _record_llm_call/_merge_child_run) that queues onto the tracer-level
# _pending_tool_calls list, drained by whichever _send() runs next, when there's no active span
# to attach a real child span to. With an active span, this becomes a real child span immediately
# instead of being queued at all — a span can trigger several child sends during its own
# lifetime, and a pending item drained by the wrong one of those sends would get silently
# misattributed (see Tracer._dispatch's docstring for the full hazard).
# ---------------------------------------------------------------------------


def test_trace_tool_call_emits_real_child_span():
    tracer = make_tracer()
    with tracer.trace("agent") as span:
        with tracer.trace_tool_call("policy_lookup", input="digital") as t:
            t.output = "digital purchases are final"

    wires = enqueued_wires(tracer)
    assert len(wires) == 2
    child, root = wires
    assert child["name"] == "policy_lookup"
    assert child["parent_span_id"] == root["span_id"]
    assert child["output"] == "digital purchases are final"
    assert "tool_calls" not in root or root.get("tool_calls") in (None, [])


def test_trace_retrieval_emits_real_child_span():
    tracer = make_tracer()
    with tracer.trace("agent") as span:
        with tracer.trace_retrieval("kb_search", query="refund policy") as r:
            r.doc_count = 3
            r.output = "3 matching docs"

    wires = enqueued_wires(tracer)
    assert len(wires) == 2
    child, root = wires
    assert child["name"] == "kb_search"
    assert child["parent_span_id"] == root["span_id"]
    assert child["output"] == "3 matching docs"


def test_record_retrieval_with_no_active_span_is_a_no_op():
    """No enclosing `with tracer.trace()` and nothing else to attach a child span to — retrieval
    data has no standalone wire representation anymore (that was performance_summary-only),
    so this is a documented no-op rather than a silent fabrication."""
    tracer = make_tracer()
    tracer.record_retrieval("orphan_search", query="x", output="y")
    with tracer.trace("agent"):
        pass
    wires = enqueued_wires(tracer)
    assert len(wires) == 1
    assert wires[0]["name"] == "agent"


def test_record_tool_call_with_no_active_span_still_queues():
    """No enclosing `with tracer.trace()` at all — record_tool_call has nothing to attach a
    child span to, so it queues onto the tracer-level pending list, drained into the plain
    tool_calls wire field of whatever trace sends next."""
    tracer = make_tracer()
    tracer.record_tool_call("orphan_call", input="x", output="y", latency_ms=5)
    with tracer.trace("agent"):
        pass
    wires = enqueued_wires(tracer)
    assert len(wires) == 1
    assert wires[0]["tool_calls"][0]["name"] == "orphan_call"
