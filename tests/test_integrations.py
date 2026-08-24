"""
Regression tests for gaps fixed in agentx/integrations/*.py:

1. google_adk.py / openai_agents.py silently dropping `error` on a failed call.
2. anthropic.py / google_genai.py / openai.py building an empty/instant trace
   for async clients because the raw client call returns an unawaited coroutine.
3. langchain.py's on_retriever_error silently dropping a failed RAG lookup.
4. anthropic.py not capturing prompt-caching tokens.
5. anthropic.py not capturing the `system` kwarg.
6. No tracing integration existed at all for the raw openai client (only the
   higher-level OpenAI Agents SDK was covered).
7. langchain.py's AgentXCallbackHandler leaking a `_top_level`/`_parents`
   entry per nested chain run forever (on_chain_end/on_chain_error only
   cleaned those dicts up for top-level runs), plus a TTL safety net for the
   rarer case of a start callback that never gets a matching end/error at all.
8. google_genai.py not patching async streaming at all
   (client.aio.models.generate_content_stream).
9. crewai.py approximating per-task timing by evenly dividing total latency
   instead of using CrewAI's real event bus timestamps.

Each test uses a hand-built duck-typed fake client/context (no real API keys,
no network) — the same approach the integration modules themselves use
(``client: Any``), so these run anywhere the relevant extra is installed.
"""
from __future__ import annotations

import asyncio
import time
import types
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from agentx.tracing.tracer import Tracer


def make_tracer() -> Tracer:
    tracer = Tracer(ingest_client=MagicMock())
    tracer._send = MagicMock(return_value="trace-id")
    return tracer


# ---------------------------------------------------------------------------
# 1a. google_adk.py — on_model_error_callback
# ---------------------------------------------------------------------------

def test_adk_model_error_is_captured():
    # google-adk is an optional extra; skip like the crewai/litellm/llamaindex/autogen tests
    # below rather than failing on ImportError (importing the module raises when it is absent).
    pytest.importorskip("google.adk")
    from agentx.integrations.google_adk import AgentXADKPlugin

    tracer = make_tracer()
    plugin = AgentXADKPlugin(tracer, name="adk-agent")

    invocation_context = types.SimpleNamespace(
        invocation_id="inv-1",
        agent=types.SimpleNamespace(name="adk-agent"),
    )
    callback_context = types.SimpleNamespace(
        get_invocation_context=lambda: invocation_context
    )
    llm_request = types.SimpleNamespace(model="gemini-x", contents=None)

    async def run():
        await plugin.before_run_callback(invocation_context=invocation_context)
        await plugin.before_model_callback(
            callback_context=callback_context, llm_request=llm_request
        )
        await plugin.on_model_error_callback(
            callback_context=callback_context,
            llm_request=llm_request,
            error=RuntimeError("model call failed"),
        )
        await plugin.after_run_callback(invocation_context=invocation_context)

    asyncio.run(run())

    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["error"] == "model call failed"


# ---------------------------------------------------------------------------
# 1b. openai_agents.py — span.error capture
# ---------------------------------------------------------------------------

def test_openai_agents_span_error_is_captured():
    from agentx.integrations.openai_agents import AgentXTracingProcessor

    tracer = make_tracer()
    processor = AgentXTracingProcessor(tracer)

    trace = types.SimpleNamespace(trace_id="trace-1", name="openai-agent")
    processor.on_trace_start(trace)

    span_error = types.SimpleNamespace(message="generation failed", data={"code": 500})
    span_data = types.SimpleNamespace(type="generation", input=None, output=None, model=None, usage=None)
    span = types.SimpleNamespace(
        trace_id="trace-1",
        span_data=span_data,
        started_at=None,
        ended_at=None,
        error=span_error,
    )
    processor.on_span_end(span)
    processor.on_trace_end(trace)

    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["error"] == "generation failed ({'code': 500})"


def test_openai_agents_no_error_when_span_clean():
    from agentx.integrations.openai_agents import AgentXTracingProcessor

    tracer = make_tracer()
    processor = AgentXTracingProcessor(tracer)

    trace = types.SimpleNamespace(trace_id="trace-2", name="openai-agent")
    processor.on_trace_start(trace)

    span_data = types.SimpleNamespace(type="generation", input=None, output=None, model=None, usage=None)
    span = types.SimpleNamespace(
        trace_id="trace-2", span_data=span_data, started_at=None, ended_at=None, error=None
    )
    processor.on_span_end(span)
    processor.on_trace_end(trace)

    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    # Tracer._send filters out None values before it hits the wire — here we
    # only need to confirm on_trace_end didn't fabricate an error.
    assert kwargs.get("error") is None


# ---------------------------------------------------------------------------
# 2a. anthropic.py — async client support
# ---------------------------------------------------------------------------

class _FakeAnthropicUsage:
    def __init__(self, input_tokens=10, output_tokens=5,
                 cache_creation_input_tokens=0, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _FakeAnthropicBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeAnthropicMessage:
    def __init__(self, text, usage=None):
        self.content = [_FakeAnthropicBlock(text)]
        self.usage = usage or _FakeAnthropicUsage()


def test_anthropic_async_client_traces_the_real_response():
    from agentx.integrations.anthropic import patch_anthropic_client

    response = _FakeAnthropicMessage("hello from async claude")

    class FakeAsyncMessages:
        async def create(self, **kwargs):
            await asyncio.sleep(0.01)
            return response

    client = types.SimpleNamespace(messages=FakeAsyncMessages())
    tracer = make_tracer()
    patch_anthropic_client(client, tracer, name="claude-agent")

    result = asyncio.run(
        client.messages.create(model="claude-x", messages=[{"role": "user", "content": "hi"}])
    )

    assert result is response
    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    # Before the fix: response was an unawaited coroutine, so output/tokens
    # would silently be None and latency would read as near-instant.
    assert kwargs["output"] == "hello from async claude"
    assert kwargs["input_tokens"] == 10
    assert kwargs["output_tokens"] == 5
    assert kwargs["latency_ms"] >= 5


def test_anthropic_async_client_propagates_and_records_errors():
    from agentx.integrations.anthropic import patch_anthropic_client

    class FakeAsyncMessages:
        async def create(self, **kwargs):
            await asyncio.sleep(0.01)
            raise ValueError("api error")

    client = types.SimpleNamespace(messages=FakeAsyncMessages())
    tracer = make_tracer()
    patch_anthropic_client(client, tracer, name="claude-agent")

    async def run():
        await client.messages.create(model="claude-x", messages=[{"role": "user", "content": "hi"}])

    with pytest.raises(ValueError, match="api error"):
        asyncio.run(run())

    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["error"] == "api error"


# ---------------------------------------------------------------------------
# 2b + 5 + 4. anthropic.py — system prompt + cache tokens (sync client)
# ---------------------------------------------------------------------------

def test_anthropic_system_and_cache_tokens_captured():
    from agentx.integrations.anthropic import patch_anthropic_client

    usage = _FakeAnthropicUsage(
        input_tokens=100, output_tokens=20,
        cache_creation_input_tokens=30, cache_read_input_tokens=15,
    )
    response = _FakeAnthropicMessage("answer", usage=usage)

    class FakeSyncMessages:
        def create(self, **kwargs):
            return response

    client = types.SimpleNamespace(messages=FakeSyncMessages())
    tracer = make_tracer()
    patch_anthropic_client(client, tracer, name="claude-agent")

    client.messages.create(
        model="claude-x",
        system="You are a helpful assistant",
        messages=[{"role": "user", "content": "hi"}],
    )

    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    # cache tokens folded into the input total (100 + 30 + 15)
    assert kwargs["input_tokens"] == 145
    sent_input = kwargs["input"]
    assert isinstance(sent_input, list)
    assert sent_input[0] == {"role": "system", "content": "You are a helpful assistant"}
    assert sent_input[1] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# 2c. google_genai.py — async client support
# ---------------------------------------------------------------------------

class _FakeGenaiUsage:
    def __init__(self, prompt_token_count=8, candidates_token_count=4):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class _FakeGenaiResponse:
    def __init__(self, text, usage=None):
        self.text = text
        self.candidates = []
        self.usage_metadata = usage or _FakeGenaiUsage()


def test_google_genai_async_client_traces_the_real_response():
    from agentx.integrations.google_genai import patch_genai_client

    response = _FakeGenaiResponse("hello from async gemini")

    class FakeAsyncModels:
        async def generate_content(self, **kwargs):
            await asyncio.sleep(0.01)
            return response

    client = types.SimpleNamespace(models=FakeAsyncModels())
    tracer = make_tracer()
    patch_genai_client(client, tracer, name="gemini-agent")

    result = asyncio.run(client.models.generate_content(model="gemini-x", contents="hi"))

    assert result is response
    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["output"] == "hello from async gemini"
    assert kwargs["input_tokens"] == 8
    assert kwargs["output_tokens"] == 4
    assert kwargs["latency_ms"] >= 5


def test_google_genai_async_streaming_traces_accumulated_output():
    """
    The real SDK's async generate_content_stream is used as `async for chunk
    in await client.aio.models.generate_content_stream(...)` — a coroutine
    that resolves to an async iterable, not an async generator function
    itself. Before the fix this was entirely unpatched.
    """
    from agentx.integrations.google_genai import patch_genai_client

    class FakeAsyncChunk:
        def __init__(self, text, usage=None):
            self.text = text
            self.usage_metadata = usage

    async def fake_async_gen():
        yield FakeAsyncChunk("Hello ")
        yield FakeAsyncChunk("world", usage=_FakeGenaiUsage(prompt_token_count=3, candidates_token_count=2))

    class FakeAsyncModels:
        async def generate_content(self, **kwargs):
            raise NotImplementedError("not used in this test")

        async def generate_content_stream(self, **kwargs):
            await asyncio.sleep(0.01)
            return fake_async_gen()

    client = types.SimpleNamespace(models=FakeAsyncModels())
    tracer = make_tracer()
    patch_genai_client(client, tracer, name="gemini-agent")

    async def run():
        chunks = []
        async for chunk in await client.models.generate_content_stream(model="gemini-x", contents="hi"):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())

    assert len(chunks) == 2
    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["output"] == "Hello world"
    assert kwargs["input_tokens"] == 3
    assert kwargs["output_tokens"] == 2


# ---------------------------------------------------------------------------
# 3. langchain.py — on_retriever_error
# ---------------------------------------------------------------------------

def test_langchain_retriever_error_is_recorded():
    from agentx.integrations.langchain import AgentXCallbackHandler

    tracer = make_tracer()
    handler = AgentXCallbackHandler(tracer, name="rag-chain")

    chain_run_id = uuid4()
    retriever_run_id = uuid4()

    handler.on_chain_start({}, {"input": "what's the weather"}, run_id=chain_run_id, parent_run_id=None)
    handler.on_retriever_start({}, "weather query", run_id=retriever_run_id, parent_run_id=chain_run_id)
    handler.on_retriever_error(
        RuntimeError("vector store unreachable"), run_id=retriever_run_id, parent_run_id=chain_run_id
    )
    handler.on_chain_end({"output": "I don't know"}, run_id=chain_run_id, parent_run_id=None)

    # The retrieval is now its own real child span (see tracer.py's _merge_child_run), sent via
    # _dispatch/enqueue rather than the root's own _send() call — tracer._send stays mocked here
    # for the root only, so the child is observed via the ingest_client mock directly instead.
    tracer._send.assert_called_once()
    retrieval_calls = [c.args[0] for c in tracer._client.enqueue.call_args_list]
    assert len(retrieval_calls) == 1
    assert retrieval_calls[0]["output"] == "ERROR: vector store unreachable"
    assert retrieval_calls[0]["input"] == "weather query"


def test_langchain_retriever_error_before_chain_start_is_buffered():
    """Pre-run RAG pattern: retriever fails before the top-level chain starts."""
    from agentx.integrations.langchain import AgentXCallbackHandler

    tracer = make_tracer()
    handler = AgentXCallbackHandler(tracer, name="rag-chain")

    retriever_run_id = uuid4()
    chain_run_id = uuid4()

    handler.on_retriever_start({}, "pre-run query", run_id=retriever_run_id, parent_run_id=None)
    handler.on_retriever_error(RuntimeError("timeout"), run_id=retriever_run_id, parent_run_id=None)

    handler.on_chain_start({}, {"input": "hi"}, run_id=chain_run_id, parent_run_id=None)
    handler.on_chain_end({"output": "done"}, run_id=chain_run_id, parent_run_id=None)

    tracer._send.assert_called_once()
    retrieval_calls = [c.args[0] for c in tracer._client.enqueue.call_args_list]
    assert len(retrieval_calls) == 1
    assert retrieval_calls[0]["output"] == "ERROR: timeout"


# ---------------------------------------------------------------------------
# 6. openai.py — new raw OpenAI SDK integration
# ---------------------------------------------------------------------------

class _FakeOpenAIUsage:
    def __init__(self, prompt_tokens=12, completion_tokens=6):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeOpenAIMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeOpenAIChoice:
    def __init__(self, message):
        self.message = message


class _FakeOpenAIChatCompletion:
    def __init__(self, content, usage=None):
        self.choices = [_FakeOpenAIChoice(_FakeOpenAIMessage(content))]
        self.usage = usage or _FakeOpenAIUsage()


def _fake_openai_client(completions):
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


def test_openai_sync_client_traces_call():
    from agentx.integrations.openai import patch_openai_client

    response = _FakeOpenAIChatCompletion("hello from gpt")

    class FakeCompletions:
        def create(self, **kwargs):
            return response

    client = _fake_openai_client(FakeCompletions())
    tracer = make_tracer()
    patch_openai_client(client, tracer, name="gpt-agent")

    result = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    assert result is response
    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["output"] == "hello from gpt"
    assert kwargs["input_tokens"] == 12
    assert kwargs["output_tokens"] == 6


def test_openai_async_client_traces_the_real_response():
    from agentx.integrations.openai import patch_openai_client

    response = _FakeOpenAIChatCompletion("hello from async gpt")

    class FakeAsyncCompletions:
        async def create(self, **kwargs):
            await asyncio.sleep(0.01)
            return response

    client = _fake_openai_client(FakeAsyncCompletions())
    tracer = make_tracer()
    patch_openai_client(client, tracer, name="gpt-agent")

    result = asyncio.run(
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    )

    assert result is response
    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    # Before the fix (same bug as anthropic.py/google_genai.py): response
    # would be an unawaited coroutine, so output/tokens would silently be None.
    assert kwargs["output"] == "hello from async gpt"
    assert kwargs["input_tokens"] == 12
    assert kwargs["output_tokens"] == 6
    assert kwargs["latency_ms"] >= 5


def test_openai_streaming_calls_are_passed_through_untraced():
    from agentx.integrations.openai import patch_openai_client

    sentinel_stream = object()

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs.get("stream") is True
            return sentinel_stream

    client = _fake_openai_client(FakeCompletions())
    tracer = make_tracer()
    patch_openai_client(client, tracer, name="gpt-agent")

    result = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], stream=True
    )

    assert result is sentinel_stream
    tracer._send.assert_not_called()


def test_openai_sync_client_records_errors():
    from agentx.integrations.openai import patch_openai_client

    class FakeCompletions:
        def create(self, **kwargs):
            raise ValueError("rate limited")

    client = _fake_openai_client(FakeCompletions())
    tracer = make_tracer()
    patch_openai_client(client, tracer, name="gpt-agent")

    with pytest.raises(ValueError, match="rate limited"):
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["error"] == "rate limited"


# ---------------------------------------------------------------------------
# 7. langchain.py — nested-run state cleanup + TTL safety net
# ---------------------------------------------------------------------------

def test_langchain_nested_chain_end_does_not_leak_state():
    from agentx.integrations.langchain import AgentXCallbackHandler

    tracer = make_tracer()
    handler = AgentXCallbackHandler(tracer, name="agent")

    top_run_id = uuid4()
    nested_run_id = uuid4()

    handler.on_chain_start({}, {"input": "hi"}, run_id=top_run_id, parent_run_id=None)
    handler.on_chain_start({}, {"input": "hi"}, run_id=nested_run_id, parent_run_id=top_run_id)
    handler.on_chain_end({"output": "nested done"}, run_id=nested_run_id, parent_run_id=top_run_id)
    handler.on_chain_end({"output": "done"}, run_id=top_run_id, parent_run_id=None)

    # Before the fix: on_chain_end returned early for non-top-level runs
    # without ever popping their _top_level/_parents entries — a guaranteed
    # per-request leak proportional to how many nested chain steps a run has.
    assert nested_run_id not in handler._top_level
    assert nested_run_id not in handler._parents
    assert top_run_id not in handler._top_level
    assert top_run_id not in handler._parents
    tracer._send.assert_called_once()


def test_langchain_nested_chain_error_does_not_leak_state():
    from agentx.integrations.langchain import AgentXCallbackHandler

    tracer = make_tracer()
    handler = AgentXCallbackHandler(tracer, name="agent")

    top_run_id = uuid4()
    nested_run_id = uuid4()

    handler.on_chain_start({}, {"input": "hi"}, run_id=top_run_id, parent_run_id=None)
    handler.on_chain_start({}, {"input": "hi"}, run_id=nested_run_id, parent_run_id=top_run_id)
    handler.on_chain_error(RuntimeError("nested failure"), run_id=nested_run_id, parent_run_id=top_run_id)
    handler.on_chain_error(RuntimeError("top failure"), run_id=top_run_id, parent_run_id=None)

    assert nested_run_id not in handler._top_level
    assert nested_run_id not in handler._parents
    assert top_run_id not in handler._top_level
    assert top_run_id not in handler._parents
    tracer._send.assert_called_once()


def test_langchain_prune_sweeps_orphaned_entries_with_no_matching_end():
    """
    Safety net for a start callback that never gets a matching end/error at
    all (hard crash, cancellation bypassing LangChain's own dispatch, etc.).
    """
    from agentx.integrations.langchain import AgentXCallbackHandler

    tracer = make_tracer()
    handler = AgentXCallbackHandler(tracer, name="agent", max_run_age_seconds=0.01)

    orphaned_run_id = uuid4()
    handler.on_chain_start({}, {"input": "hi"}, run_id=orphaned_run_id, parent_run_id=None)
    # No matching on_chain_end/on_chain_error ever fires for this run_id.

    time.sleep(0.02)

    new_run_id = uuid4()
    handler.on_chain_start({}, {"input": "hi again"}, run_id=new_run_id, parent_run_id=None)

    assert orphaned_run_id not in handler._runs
    assert orphaned_run_id not in handler._top_level
    assert orphaned_run_id not in handler._parents
    assert new_run_id in handler._runs


# ---------------------------------------------------------------------------
# 9. crewai.py — real per-task timing via CrewAI's event bus
# ---------------------------------------------------------------------------

def test_crewai_captures_real_per_task_timing_via_event_bus():
    """
    Drives CrewAI's actual event bus (crewai_event_bus.emit with real
    TaskStartedEvent/TaskCompletedEvent/TaskOutput instances) around a fake
    Crew.kickoff() that sleeps different amounts per task, so the two tasks'
    durations are provably unequal — the old "divide latency evenly across
    tasks" approximation would have reported them as identical.
    """
    crewai = pytest.importorskip("crewai")
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.task_events import TaskCompletedEvent, TaskStartedEvent
    from crewai.tasks.task_output import TaskOutput

    from agentx.integrations.crewai import AgentXCrewObserver

    class FakeTask:
        def __init__(self, task_id, name):
            self.id = task_id
            self.name = name
            self.description = name
            self.fingerprint = None

    task1 = FakeTask("task-1", "Research topic")
    task2 = FakeTask("task-2", "Write summary")

    def emit(event):
        future = crewai_event_bus.emit(object(), event)
        if future is not None:
            future.result(timeout=5.0)

    class FakeCrewOutput:
        def __init__(self, raw, tasks_output):
            self.raw = raw
            self.tasks_output = tasks_output

    class FakeCrew:
        def kickoff(self, inputs=None):
            emit(TaskStartedEvent(context="", task=task1))
            time.sleep(0.03)
            output1 = TaskOutput(description="Research topic", raw="research done", agent="tester")
            emit(TaskCompletedEvent(output=output1, task=task1))

            emit(TaskStartedEvent(context="", task=task2))
            time.sleep(0.01)
            output2 = TaskOutput(description="Write summary", raw="summary done", agent="tester")
            emit(TaskCompletedEvent(output=output2, task=task2))

            return FakeCrewOutput(raw="final output", tasks_output=[output1, output2])

    tracer = make_tracer()
    observer = AgentXCrewObserver(tracer, name="my-crew")

    result = observer.kickoff(FakeCrew(), inputs={"topic": "AI"})

    assert result.raw == "final output"
    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    steps = kwargs["performance_summary"]["execution_steps"]
    assert len(steps) == 2
    assert steps[0]["name"] == "Research topic"
    assert steps[1]["name"] == "Write summary"
    # Real timing, not an even split — task 1 slept ~3x longer than task 2.
    assert steps[0]["duration_ms"] > steps[1]["duration_ms"] * 1.5
    assert steps[0]["output"] == "research done"
    assert steps[1]["output"] == "summary done"


def test_crewai_falls_back_to_even_split_without_event_bus():
    """When CrewAI's events module isn't importable, the old approximation still works."""
    import agentx.integrations.crewai as crewai_integration

    tracer = make_tracer()
    observer = crewai_integration.AgentXCrewObserver(tracer, name="my-crew")

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

    # Force the "no events module" path regardless of whether crewai is installed.
    original = observer._start_task_timing_capture
    observer._start_task_timing_capture = lambda: ({}, lambda: None)
    try:
        result = observer.kickoff(FakeCrew(), inputs={"topic": "AI"})
    finally:
        observer._start_task_timing_capture = original

    assert result.raw == "final output"
    # Each task is now a real child span (see tracer.py's _merge_child_run), sent via
    # _dispatch/enqueue rather than the root's own _send() call.
    tracer._send.assert_called_once()
    steps = [c.args[0] for c in tracer._client.enqueue.call_args_list]
    assert len(steps) == 2
    assert steps[0]["latency_ms"] == steps[1]["latency_ms"]


# ---------------------------------------------------------------------------
# 10. litellm.py — new LiteLLM integration
# ---------------------------------------------------------------------------

def _make_litellm_logger():
    from agentx.integrations.litellm import AgentXLiteLLMLogger

    tracer = make_tracer()
    logger = AgentXLiteLLMLogger(tracer, name="litellm-agent")
    return tracer, logger


def _reset_litellm_callback_state(litellm) -> None:
    """
    LiteLLM derives internal dispatch lists (``_async_success_callback`` etc.)
    from ``litellm.callbacks`` the first time a call runs them, and doesn't
    appear to fully re-derive them on a later reassignment within the same
    process — real long-lived apps only ever set ``litellm.callbacks`` once at
    startup, so this doesn't come up there. Only matters for test isolation
    here (this suite exercises several call shapes back to back), not for the
    actual integration in agentx/integrations/litellm.py, which never touches
    this private state.
    """
    litellm.callbacks = []
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.input_callback = []
    litellm._async_success_callback = []
    litellm._async_failure_callback = []
    litellm._async_input_callback = []


def test_litellm_sync_completion_traces_call():
    litellm = pytest.importorskip("litellm")
    _reset_litellm_callback_state(litellm)

    tracer, logger = _make_litellm_logger()
    litellm.callbacks = [logger]
    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello there!",
        )
    finally:
        litellm.callbacks = []

    assert response.choices[0].message.content == "Hello there!"
    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["output"] == "Hello there!"
    assert kwargs["input_tokens"] is not None
    assert kwargs["output_tokens"] is not None
    assert kwargs["model"] == "gpt-3.5-turbo"


def test_litellm_async_completion_traces_call():
    litellm = pytest.importorskip("litellm")
    _reset_litellm_callback_state(litellm)

    tracer, logger = _make_litellm_logger()
    litellm.callbacks = [logger]

    async def run():
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello async!",
        )
        # LiteLLM dispatches async_log_success_event as a fire-and-forget
        # background task that isn't awaited before acompletion() returns —
        # give it a beat to actually run before asserting.
        await asyncio.sleep(0.2)
        return response

    try:
        response = asyncio.run(run())
    finally:
        litellm.callbacks = []

    assert response.choices[0].message.content == "Hello async!"
    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["output"] == "Hello async!"


def test_litellm_streaming_traces_aggregated_response():
    """LiteLLM reassembles a streamed response before firing the success callback."""
    litellm = pytest.importorskip("litellm")
    _reset_litellm_callback_state(litellm)

    tracer, logger = _make_litellm_logger()
    litellm.callbacks = [logger]
    try:
        stream = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello streamed!",
            stream=True,
        )
        chunks = list(stream)
        # Even for this sync call, LiteLLM dispatches the success callback on
        # a background thread after the stream is exhausted rather than
        # inline — same as the async fire-and-forget path, give it a beat.
        time.sleep(0.3)
    finally:
        litellm.callbacks = []

    assert len(chunks) > 1
    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["output"] == "Hello streamed!"


def test_litellm_failure_records_error():
    litellm = pytest.importorskip("litellm")
    _reset_litellm_callback_state(litellm)

    tracer, logger = _make_litellm_logger()
    litellm.callbacks = [logger]
    try:
        with pytest.raises(Exception):
            litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hi"}],
                mock_response=Exception("boom"),
            )
    finally:
        litellm.callbacks = []

    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert "boom" in kwargs["error"]


# ---------------------------------------------------------------------------
# 11. llamaindex.py — new LlamaIndex integration
# ---------------------------------------------------------------------------

def test_llamaindex_query_engine_traces_retrieval_and_llm_steps():
    """
    Drives a real VectorStoreIndex query (MockLLM + MockEmbedding, no network/
    API keys) through the real CallbackManager event system, and checks the
    resulting trace has both a retrieval step and an LLM step with real text.
    """
    pytest.importorskip("llama_index.core")
    from llama_index.core import Document, Settings, VectorStoreIndex
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms import MockLLM

    from agentx.integrations.llamaindex import AgentXLlamaIndexHandler

    tracer = make_tracer()
    handler = AgentXLlamaIndexHandler(tracer, name="rag-agent")

    # Assigning fresh values directly rather than reading Settings.llm/
    # embed_model first — Settings.llm is a lazily-resolved property that
    # tries to construct a real default OpenAI LLM the moment it's read if
    # nothing has been assigned yet, which would fail with no API key.
    Settings.llm = MockLLM()
    Settings.embed_model = MockEmbedding(embed_dim=8)
    Settings.callback_manager = CallbackManager([handler])
    try:
        index = VectorStoreIndex.from_documents([Document(text="LiteLLM is a proxy for many LLM providers.")])
        # Index construction itself has no QUERY/LLM root event, so it must
        # not have produced a trace yet.
        tracer._send.assert_not_called()

        engine = index.as_query_engine()
        response = engine.query("What is LiteLLM?")
    finally:
        Settings.callback_manager = CallbackManager([])

    assert str(response)
    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["input"] == "What is LiteLLM?"
    assert kwargs["output"]
    # The LLM call and the retrieval are now each their own real child span (see tracer.py's
    # _merge_child_run), sent via _dispatch/enqueue rather than folded into the root's own send.
    children = [c.args[0] for c in tracer._client.enqueue.call_args_list]
    assert len(children) >= 2  # at least one LLM call child + one retrieval child
    retrieval_children = [c for c in children if c["name"].startswith("Retrieval")]
    assert len(retrieval_children) == 1
    assert "LiteLLM is a proxy" in retrieval_children[0]["output"]


def test_llamaindex_llm_error_is_captured():
    """A bare llm.complete() failure (no query engine wrapper) still traces with an error."""
    pytest.importorskip("llama_index.core")
    from typing import Any

    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse
    from llama_index.core.llms.callbacks import llm_completion_callback

    from agentx.integrations.llamaindex import AgentXLlamaIndexHandler

    class BrokenLLM(CustomLLM):
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata()

        @llm_completion_callback()
        def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
            raise RuntimeError("LLM backend unreachable")

        @llm_completion_callback()
        def stream_complete(self, prompt: str, **kwargs: Any):
            raise NotImplementedError

    tracer = make_tracer()
    handler = AgentXLlamaIndexHandler(tracer, name="rag-agent")
    llm = BrokenLLM(callback_manager=CallbackManager([handler]))

    with pytest.raises(RuntimeError, match="LLM backend unreachable"):
        llm.complete("hello")

    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["error"] == "LLM backend unreachable"


# ---------------------------------------------------------------------------
# 12. autogen.py — new Microsoft AutoGen integration
# ---------------------------------------------------------------------------

def test_autogen_agent_run_traces_text_reply():
    """Drives a real AssistantAgent.run() via AutoGen's own ReplayChatCompletionClient (no network/API keys)."""
    pytest.importorskip("autogen_agentchat")
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.replay import ReplayChatCompletionClient

    from agentx.integrations.autogen import AgentXAutoGenObserver

    model_client = ReplayChatCompletionClient(["Hello from AutoGen!"])
    agent = AssistantAgent("assistant", model_client=model_client)

    tracer = make_tracer()
    observer = AgentXAutoGenObserver(tracer, name="my-agent")

    result = asyncio.run(observer.run(agent, task="Say hello"))

    assert result.messages[-1].content == "Hello from AutoGen!"
    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["input"] == "Say hello"
    assert kwargs["output"] == "Hello from AutoGen!"
    assert kwargs["input_tokens"] == 22
    assert kwargs["output_tokens"] == 3
    steps = kwargs["performance_summary"]["execution_steps"]
    assert len(steps) == 1
    assert steps[0]["output"] == "Hello from AutoGen!"


def test_autogen_agent_run_traces_tool_call():
    pytest.importorskip("autogen_agentchat")
    import json

    from autogen_agentchat.agents import AssistantAgent
    from autogen_core import FunctionCall
    from autogen_core.models import CreateResult, ModelInfo, RequestUsage
    from autogen_core.tools import FunctionTool
    from autogen_ext.models.replay import ReplayChatCompletionClient

    from agentx.integrations.autogen import AgentXAutoGenObserver

    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    tool = FunctionTool(get_weather, description="Get the weather for a city")
    model_info = ModelInfo(vision=False, function_calling=True, json_output=False, family="unknown", structured_output=False)
    model_client = ReplayChatCompletionClient(
        [
            CreateResult(
                finish_reason="function_calls",
                content=[FunctionCall(id="call_1", name="get_weather", arguments=json.dumps({"city": "NYC"}))],
                usage=RequestUsage(prompt_tokens=15, completion_tokens=5),
                cached=False,
            ),
            "The weather in NYC is sunny.",
        ],
        model_info=model_info,
    )
    agent = AssistantAgent("assistant", model_client=model_client, tools=[tool])

    tracer = make_tracer()
    observer = AgentXAutoGenObserver(tracer, name="my-agent")

    asyncio.run(observer.run(agent, task="What is the weather in NYC?"))

    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    perf = kwargs["performance_summary"]
    assert len(perf["tool_calls"]) == 1
    tool_call = perf["tool_calls"][0]
    assert tool_call["name"] == "get_weather"
    assert "NYC" in tool_call["input"]
    assert tool_call["output"] == "sunny in NYC"


def test_autogen_run_failure_records_error():
    class BrokenAgentOrTeam:
        async def run(self, task=None, **kwargs):
            raise RuntimeError("agent crashed")

    from agentx.integrations.autogen import AgentXAutoGenObserver

    tracer = make_tracer()
    observer = AgentXAutoGenObserver(tracer, name="my-agent")

    with pytest.raises(RuntimeError, match="agent crashed"):
        asyncio.run(observer.run(BrokenAgentOrTeam(), task="do something"))

    tracer._send.assert_called_once()
    _, kwargs = tracer._send.call_args
    assert kwargs["error"] == "agent crashed"
