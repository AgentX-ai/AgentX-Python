# Production Tracing - Python SDK

## Overview

The AgentX Python SDK lets you record production agent runs into your workspace with a single decorator or context manager - no changes to your agent's logic. Traces appear in the **Live Traces** tab and can be evaluated against your test datasets.

Works with LangChain, CrewAI, OpenAI Agents, Anthropic, or any Python function.

---

## Installation

```bash
pip install agentx-python
```

---

## Quick start

```python
from agentx import AgentX

client = AgentX(api_key="ax_live_xxxxxxxxxxxxxxxx")
tracer = client.tracer

@tracer.trace("customer-support-agent", framework="langchain")
def handle_query(query: str) -> str:
    # your LangChain / LLM code here
    return chain.invoke(query)

# Every call to handle_query() is automatically traced
response = handle_query("How do I reset my password?")
```

---

## Authentication

```python
# Explicit API key
client = AgentX(api_key="ax_live_xxxxxxxxxxxxxxxx")

# From environment variables
client = AgentX.from_env()
```

`from_env()` reads `AGENTX_API_KEY`, plus - for the base URL - the first of `AGENTX_API_BASE_URL` / `AGENTX_SELFHOST_BASE_URL` / `BASE_URL` that is set:

```bash
export AGENTX_API_KEY=ax_live_xxxxxxxxxxxxxxxx
# self-host only:
export AGENTX_API_BASE_URL=http://localhost:4700/api/v1
```

The tracer is fire-and-forget: a wrong key or URL surfaces only as a one-time log warning while traces silently go nowhere. For a long-running service, call `client.ping()` once at startup - it raises immediately (`AgentXConnectionError` / `AgentXAuthError`) on a bad URL or key.

---

## The `Tracer`

Access the tracer via `client.tracer`:

```python
tracer = client.tracer
```

All tracing methods are on the `tracer` object.

---

## `tracer.trace()` - decorator / context manager

The primary tracing interface. Captures the wrapped function's arguments as `input`, return value as `output`, wall-clock time as `latency_ms`, and any exception as `error`.

### As a decorator

```python
@tracer.trace(
    "agent-name",
    framework="crewai",      # optional
    model="gpt-4o",          # optional
    session_id="...",        # optional
)
def my_agent(query: str) -> str:
    ...
```

### As a context manager

```python
with tracer.trace("agent-name", framework="langchain") as span:
    span.input = {"query": query, "context": context}
    result = chain.invoke(query)
    span.output = result
    span.add_tool_call("search_kb", input=query, output=kb_result, latency_ms=210)
```

### Parameters

All parameters work in both decorator and context-manager form - the decorator forwards every one of them (including `sync`, `monitor`, `pattern_ids`, `agent_id`, and `span_kind`) onto the span it opens per call.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | ✓ | Agent or operation label shown in the UI. One stable agent is resolved per distinct name |
| `input` | any | - | Initial input value (context manager form; the decorator captures the function's arguments) |
| `framework` | `str` | - | Platform label - any string, including custom platform names. Auto-filled when omitted: see [Platform detection](#platform-detection) |
| `model` | `str` | - | LLM model used, e.g. `"gpt-4o"`, `"claude-sonnet-4-6"` |
| `session_id` | `str` | - | Groups traces from the same user session or thread |
| `metadata` | `dict` | - | Arbitrary key-value metadata |
| `sync` | `bool` | - | `True` sends the trace synchronously on exit so `span.trace_id` is populated. See [Getting the trace id back](#getting-the-trace-id-back-synctrue) |
| `monitor` | `bool` | - | `True` checks this trace against Monitor patterns immediately; `False` opts out of every ingest-time check. Default (`None`) leaves the server's standard behavior. See [Monitor](#monitor) |
| `pattern_ids` | `list[str]` | - | With `monitor=True`: restrict detection to exactly these pattern ids |
| `agent_id` | `str` | - | Pin this trace to a known agent id instead of resolving by `name` - a disambiguator for when the name alone isn't enough |
| `span_kind` | `str` | - | What kind of step this span is (`"agent"`, `"llm"`, `"tool"`, `"retrieval"`, ...), stated instead of left to the backend's classification fallback |

### `_TraceSpan` methods and attributes (context manager form)

| Method / Attribute | Description |
|---|---|
| `span.input = value` | Override the captured input |
| `span.output = value` | Set the output |
| `span.add_tool_call(name, *, input, output, latency_ms)` | Record a tool call made during the span |
| `span.set_error(message)` | Mark the span as failed with the given error message |
| `span.trace_id` | The ingested trace's id - populated after the `with` block exits, and only when the span was opened with `sync=True` (otherwise `None`) |
| `span.span_id` | This span's id, usable to parent further spans |
| `span.child_span(name, *, start_time, end_time, input, output, ...)` | Send one already-finished child-span row with explicit timing, parented to this span. Returns the child (its `.span_id` can parent grandchildren) |

---

## Span trees and nesting

Nested `with tracer.trace(...)` blocks link as real parent/child span rows sharing one session, so a multi-step run shows up as a tree in the trace dialog's span panel. Nesting is automatic: any span opened while another is active on the same thread becomes its child, and so does every auto-instrumented call made inside the block (a patched Anthropic/OpenAI/Google GenAI/LiteLLM client, or a framework integration like `AgentXCallbackHandler`):

```python
with tracer.trace("orchestrator") as root:
    with tracer.trace("plan") as plan:          # child span of "orchestrator"
        plan.output = make_plan(query)
    reply = claude.messages.create(...)          # patched client: its own child-span row
    root.output = reply
```

The active-span stack is **thread-local**. Work submitted to a `ThreadPoolExecutor` (or any other thread) doesn't see a span opened on the calling thread - wrap the worker body in `tracer.use_span(span)` to attach it:

```python
with tracer.trace("orchestrator") as span:
    def worker():
        with tracer.use_span(span):
            chain.invoke(..., config={"callbacks": [handler]})

    with ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(worker).result()
```

`use_span` is safe to use from multiple threads concurrently for the same span.

---

## Getting the trace id back (`sync=True`)

By default a trace is queued and delivered by a background thread - it never blocks the caller, but there is no way to learn the resulting trace id. Pass `sync=True` to send synchronously on block exit instead, so `span.trace_id` is populated:

```python
with tracer.trace("support_agent_call", sync=True) as span:
    resp = call_llm(query)
    span.output = resp

print(span.trace_id)   # ready - e.g. to link this trace to an evaluation result
```

On a root span, `sync=True` covers the whole tree: child spans recorded inside the block are drained before the root is sent, so a read immediately afterwards sees every span. This is exactly what evaluation harnesses use to link a case to its trace: return `{"output": resp, "trace_id": span.trace_id}` from the agent function (see EVALUATIONS.md).

`tracer.flush(timeout=5.0)` is the companion for the default async mode: it blocks until all queued traces are delivered (or the timeout elapses, returning `False` and leaving delivery running in the background). Call it before a short-lived process exits.

---

## Platform detection

Tracing is **platform agnostic**: every trace carries a platform label, and any agent runtime
works. The label resolves in priority order:

1. **Explicit** - `framework="..."` on `trace()`. Any string is valid, including platforms
   AgentX has no integration for: `framework="my-inhouse-runner"` charts and filters like any
   built-in name. (The engine folds labels to lowercase, so `"LangChain"` and `"langchain"`
   are one platform.)
2. **Integration** - every AgentX integration stamps its literal automatically, no parameter
   needed:

   | Integration | Label |
   |---|---|
   | `AgentXCallbackHandler` (LangChain/LangGraph) | `langchain` |
   | `AgentXCrewObserver` | `crewai` |
   | `AgentXTracingProcessor` (OpenAI Agents SDK) | `openai-agents` |
   | `patch_openai_client` | `openai` |
   | `patch_anthropic_client` | `anthropic` |
   | `patch_genai_client` | `google-genai` |
   | `AgentXADKPlugin` | `google-adk` |
   | `AgentXLiteLLMLogger` | `litellm` |
   | `AgentXLlamaIndexHandler` | `llamaindex` |
   | `AgentXAutoGenObserver` | `autogen` |
   | `MoveworksImporter` | `moveworks` |
   | `DatabricksTraceImporter` | `databricks` |

3. **Auto-detection** - a plain `@tracer.trace(...)` with neither of the above looks at which
   known orchestration framework is actually imported in the process (LangChain/LangGraph,
   CrewAI, LlamaIndex, AutoGen, OpenAI Agents SDK, Google ADK, Semantic Kernel, Haystack,
   Pydantic AI, smolagents, DSPy) and labels the span when exactly one is loaded. Ambiguous or
   unknown means no label - the trace still ingests fine and buckets as "Other / custom" in the
   dashboard, never mislabeled.

The label powers the Live Traces framework filter and Monitor's **Platforms** chart. From the
SDK, the same numbers come from `client.monitor.metrics()` (`GET /monitor/metrics`): bucketed
spans by kind, latency percentiles, tokens/cost, tool executions/failures, and platform
attribution (`frameworks` window totals plus per-bucket `byFramework`). It takes a `window`
(`"1h"` up to `"90d"`, default `"1d"`) and optional `agent`/`model`/`tool`/`framework`/`status`
filters matching the dashboard's filter chips - `framework="other"` selects unlabeled traffic.

## Framework examples

### LangChain

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from agentx import AgentX

client = AgentX.from_env()
tracer = client.tracer

llm = ChatOpenAI(model="gpt-4o")

@tracer.trace("support-agent", framework="langchain", model="gpt-4o")
def handle(query: str) -> str:
    response = llm.invoke([HumanMessage(content=query)])
    return response.content

handle("How do I cancel my subscription?")
```

### CrewAI

```python
from crewai import Agent, Task, Crew
from agentx import AgentX

client = AgentX.from_env()
tracer = client.tracer

@tracer.trace("research-crew", framework="crewai")
def run_crew(topic: str) -> str:
    agent = Agent(role="Researcher", goal=f"Research {topic}", backstory="...")
    task = Task(description=f"Research {topic}", agent=agent, expected_output="...")
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    return str(result)

run_crew("quantum computing trends 2026")
```

### OpenAI Agents SDK

```python
from agents import Agent, Runner
from agentx import AgentX

client = AgentX.from_env()
tracer = client.tracer

agent = Agent(name="Support", instructions="You are a helpful support agent.")

@tracer.trace("openai-support-agent", framework="openai-agents", model="gpt-4o")
async def run(query: str) -> str:
    result = await Runner.run(agent, query)
    return result.final_output

import asyncio
asyncio.run(run("What is your refund policy?"))
```

### Anthropic (direct)

```python
import anthropic
from agentx import AgentX

client = AgentX.from_env()
tracer = client.tracer
claude = anthropic.Anthropic()

@tracer.trace("claude-agent", framework="anthropic", model="claude-sonnet-4-6")
def run(query: str) -> str:
    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": query}]
    )
    return message.content[0].text

run("Summarise the key points of our refund policy.")
```

### Plain Python function

```python
from agentx import AgentX

client = AgentX.from_env()
tracer = client.tracer

@tracer.trace("my-llm-wrapper")
def call_llm(prompt: str) -> str:
    # any custom logic
    return my_llm_client.complete(prompt)
```

---

## Recording tool calls

Use `span.add_tool_call()` in context manager mode to record individual tool invocations:

```python
with tracer.trace("rag-agent", framework="langchain") as span:
    span.input = query

    # First tool call
    kb_result = search_knowledge_base(query)
    span.add_tool_call("search_knowledge_base", input=query, output=kb_result, latency_ms=190)

    # Second tool call
    web_result = web_search(query)
    span.add_tool_call("web_search", input=query, output=web_result, latency_ms=850)

    # Final LLM call
    answer = llm.invoke(f"Context: {kb_result}\n\nQuery: {query}")
    span.output = answer
```

Tool calls are displayed in the expanded trace view in the AgentX UI with per-call latency.

### Capturing tool failures: `tracer.trace_tool_call()`

`add_tool_call()` reports a call after the fact; `tracer.trace_tool_call()` wraps the execution itself, timing it and capturing failures automatically:

```python
with tracer.trace("support-agent") as span:
    span.input = query

    with tracer.trace_tool_call("search_orders", input=order_id) as t:
        t.output = search_orders(order_id)   # raising here records a failed call

    span.output = answer
```

An exception escaping the block records the call with `success=False` plus the error text, then propagates unchanged so your own error handling still runs. That `success: false` is what Monitor's built-in "Tool failure" check and the dashboard's Tool quality column read, so a flaky tool shows up in triage without any extra wiring. To set the outcome yourself instead (an API that returned a well-formed error payload, say), use `tracer.record_tool_call(name, input=..., output=..., success=False, error="...")`. Both attach to the innermost active span; with no active span, the call is queued onto the next trace this tracer sends.

### Recording retrievals

The retrieval twins of the tool-call helpers mark a span as a knowledge-base / vector-store lookup, which feeds the engine's retrieval-context extraction (used by RAG judges) and the dashboard's references panel:

```python
with tracer.trace("rag-agent") as span:
    with tracer.trace_retrieval("kb_search", query=question) as r:
        docs = retrieve(question)
        r.doc_count = len(docs)
        r.output = docs
    span.output = answer_from(docs)
```

`tracer.record_retrieval(name, query=..., output=..., duration_ms=...)` is the after-the-fact form. Custom names like `"kb_search"` work - the span carries an explicit retrieval marker, not a name heuristic.

---

## Session grouping

Use `session_id` to group multiple traces that belong to the same user conversation or workflow:

```python
import uuid

session_id = str(uuid.uuid4())   # generated once per user session

@tracer.trace("support-agent", session_id=session_id)
def handle(query: str) -> str:
    ...

# All traces with the same session_id are linked in the UI
handle("First question")
handle("Follow-up question")
```

On self-host, sessions are a first-class surface: Governance > Observe > **Sessions** lists each conversation with its turn count and latest coherence score, and opening one shows every turn in order. A built-in **session coherence** judge (and any [online evaluator](#clientmonitoronline_evaluators-self-host-only) created with `scope="session"`) scores the conversation as a whole, catching the failure mode where every individual reply looks fine but the conversation goes in circles.

When you don't pass `session_id`, each trace gets its own auto-generated session, so passing it is only about grouping, never required.

---

## Error handling

Exceptions raised inside a traced function or context manager are automatically captured as the `error` field and re-raised - they do not affect the trace submission:

```python
@tracer.trace("my-agent")
def risky_agent(query: str) -> str:
    raise ValueError("Model unavailable")

try:
    risky_agent("test")
except ValueError:
    pass
# Trace is submitted with error: "Model unavailable"
```

To set an error manually in context manager mode:

```python
with tracer.trace("my-agent") as span:
    try:
        result = call_agent(query)
        span.output = result
    except Exception as e:
        span.set_error(str(e))
        raise
```

---

## `tracer.evaluate_trace()`

Score a previously ingested trace against a dataset, without re-running the agent (`POST /ingest/traces/{trace_id}/evaluate`, synchronous - it blocks for one judge call). The trace's recorded input/output are used as-is.

```python
with tracer.trace("support-agent", sync=True) as span:
    span.output = call_llm(query)

result = tracer.evaluate_trace(
    trace_id=span.trace_id,
    dataset_id="6876ddd222bbb333ccc444ee",
    question_index=0,   # optional - which question to score against
)

print(result["rating"])         # 0-10 (None if the judge could not score)
print(result["justification"])  # LLM explanation
print(result["run_id"])         # id of the one-result eval run this created
```

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `trace_id` | `str` | ✓ | Id of an ingested trace - from `span.trace_id` after a `sync=True` block |
| `dataset_id` | `str` | ✓ | Evaluation dataset to score against |
| `question_index` | `int` | - | 0-based question index; when supplied, that question's `expectedResults` is included in the scoring prompt. Omit to score against the dataset's general criteria only |

### Return value

A plain `dict` with keys `run_id`, `trace_id`, `rating` (0-10, `None` when scoring failed), `justification`, and `status` (`"completed"`). Raises `requests.HTTPError` on a non-2xx response.

---

## Monitor

Automatic production monitoring: check traces against detection **patterns** and get back triage-ready **signals**, readable from the dashboard (Governance > Observe) or straight from the SDK via `client.monitor.signals`. This works the same way for an agent built natively in AgentX and for an external agent traced entirely through this SDK.

There are two ways to trigger it, and they can be combined.

### Trace-time (explicit, no dashboard setup required)

Pass `monitor=True` on `tracer.trace(...)` to check that specific trace immediately. `pattern_ids` (ids returned by `client.monitor.patterns.builder(...).publish()`) restricts detection to exactly those patterns; omit it to run the full default sweep (built-in checks plus every pattern enabled for the workspace) instead.

```python
pattern = client.monitor.patterns.builder(
    name="Promises a refund",
    detector_kind="semantic",
    semantic_prompt="The response promises a refund.",
    severity="high",
).publish()

with client.tracer.trace("support-agent", monitor=True, pattern_ids=[pattern.id]) as span:
    span.output = call_llm(query)
```

Works with the decorator form too: `@tracer.trace("support-agent", monitor=True, pattern_ids=[pattern.id])`.

### Dashboard toggle (automatic, every trace from an agent)

Enable monitoring once per agent in the dashboard, and every subsequent trace from that agent is checked automatically, with no `monitor=True` needed on any individual call:

1. Send at least one trace. The first `tracer.trace(...)` call for a given agent name auto-creates a reference agent in your workspace.
2. Open **Governance > Observe > Agents**. Your SDK-traced agent appears in the list with an **External** badge.
3. Turn on its monitoring profile and pick a coverage mode (sample a percentage of traffic, or check every trace).

### What gets checked

Built-in detectors: empty response, trace/tool errors, latency regressions, and negative user feedback (native chat votes; on self-host, votes forwarded via `client.feedback.report(...)` too). Custom patterns: keyword, regex, or an LLM-judged semantic rubric, created via the dashboard or `client.monitor.patterns`. A match becomes a signal, deduped against repeat occurrences of the same issue. A trace that matches nothing counts toward the agent's health rate instead.

### `client.monitor.patterns`

```python
pattern = client.monitor.patterns.builder(
    name="Promises a refund",
    detector_kind="semantic",
    semantic_prompt="The response promises a refund.",
).publish()

print(pattern.id)

client.monitor.patterns.get(pattern.id)   # -> MonitorPattern
client.monitor.patterns.list()            # -> list[MonitorPattern]
```

#### `builder()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Pattern display name |
| `description` | `str` | `None` | Human-readable description |
| `detector_kind` | `str` | `"contains"` | `"contains"`, `"regex"`, or `"semantic"`; selects which field below is used |
| `match_target` | `list[str]` | `["response"]` | Where to look: `"response"`, `"userMessage"`, `"trace"` |
| `match_mode` | `str` | `"any"` | For `detector_kind="contains"`: `"any"` or `"all"` of `include_terms` must match |
| `include_terms` / `exclude_terms` | `list[str]` | `[]` | Phrases to require / exclude, for `detector_kind="contains"` |
| `regex` | `str` | `None` | Regular expression body, for `detector_kind="regex"` |
| `semantic_prompt` | `str` | `None` | Rubric an LLM judges the response against, for `detector_kind="semantic"` |
| `severity` | `str` | `"medium"` | `"low"`, `"medium"`, `"high"`, or `"critical"` |
| `polarity` | `str` | `"failure"` | `"failure"` raises a signal to triage; `"proper"` logs a healthy tally instead |
| `enabled` | `bool` | `True` | Whether the pattern is checked at all |
| `sample_rate` | `float` | `1.0` | Fraction of matching traces to actually check, `0.0`-`1.0` |
| `scope_mode` / `agent_ids` | `str` / `list[str]` | `"all"` / `[]` | Restrict this pattern to specific agents instead of the whole workspace |

`publish()` returns a `MonitorPattern` with `.id`, which you pass in `pattern_ids` at trace time.

### `client.monitor.signals`

Read back the alerts/findings a pattern match (or a built-in detector) produced, without opening the dashboard. Read-only: a signal is the system's output from checking traces against patterns, not something you create directly.

```python
signals = client.monitor.signals.list(severity="high", limit=20)
for s in signals:
    print(s.id, s.severity, s.summary, s.occurrence_count)

signal = client.monitor.signals.get(signals[0].id)
print(signal.recommended_actions)
```

#### `list()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `polarity` | `str` | server default (failures only) | `"failure"`, `"proper"` (healthy tally), or `"all"` for both |
| `status` | `str` | `None` | e.g. `"open"`, filter to one status |
| `severity` | `str` | `None` | `"low"`, `"medium"`, `"high"`, or `"critical"` |
| `agent_id` | `str` | `None` | Restrict to one agent (matches either the signal's representative agent or its occurrence trail) |
| `limit` | `int` | `50` | Capped at 100 server-side |

`list()`/`get()` both return `MonitorSignal` (`.id`, `.type`, `.severity`, `.polarity`, `.status`, `.summary`, `.pattern_key`, `.occurrence_count`, `.occurrences`, `.recommended_actions`, `.root_cause`, and more), matching the fields shown in the dashboard's triage queue.

### `client.monitor.profile`

Get/update one agent's Monitor settings, the same settings shown in the dashboard's per-agent monitoring settings dialog.

```python
profile = client.monitor.profile.get("agent_123")
print(profile.enabled if profile else "never configured, on defaults")

# Opt this agent out of info (clean-run) signals.
client.monitor.profile.update("agent_123", info_detection_enabled=False)
```

`get()` returns `None` when the agent has never been configured (still on platform defaults, e.g. a built-in latency threshold of 20000ms). `update()` upserts and only changes the fields you pass, everything else on an existing profile is left as is:

| Parameter | Type | Description |
|---|---|---|
| `enabled` | `bool` | Turn Monitor on/off for this agent |
| `failure_detection_enabled` / `info_detection_enabled` | `bool` | Opt a whole detection category out |
| `coverage_mode` | `str` | `"all"` (every trace) or `"sampled"` |
| `sample_rate` | `float` | Fraction of traffic monitored when `coverage_mode="sampled"` |
| `channels` | `list[str]` | Notification channels |
| `dataset_id` | `str` | Evaluation dataset this agent's signals feed into |
| `threshold_overrides` | `dict` | Per-check threshold overrides, e.g. `{"latencyMs": 15000}` |
| `retention_days` | `int` | How long monitored traces are kept |
| `approval_policy` | `dict[str, str]` | Per-action approval mode for autotune actions |

**Self-host:** `coverage_mode`, `sample_rate`, `retention_days`, and `threshold_overrides["latencyMs"]` are project-level defaults there (set once for every agent in the dashboard's Platform Settings). `update()` still accepts them for wire compatibility, but the self-host engine doesn't read the stored per-agent values - `enabled`, the two detection toggles, and `channels` remain real per-agent settings everywhere.

### `client.monitor.online_evaluators` (self-host only)

> **Legacy view.** An online evaluator is the *online profile* of an **LLM Judge Scorer**, and this client is the half-view of it that predates the consolidation. It keeps working unchanged and its ids are the same ids, but it emits a `DeprecationWarning` on first use. Prefer [`client.monitor.judge_scorers`](EVALUATIONS.md#llm-judge-scorers---reusable-grading-configs), which manages the judge rubric, the offline (dataset-run) profile and this online profile as one entity - and note that route needs a self-host engine build that serves it, where this one works on every engine.

A real LLM judge scoring a sample of live production traffic continuously, distinct from a pattern's rule-matching: the same judge-scoring logic Evaluate's offline runs use, just pointed at production instead of a golden dataset. References an `evaluation_settings_id` (an Evaluator config: criteria, judge prompt, judge model) rather than storing its own copy, the same config datasets/Evaluate runs use - post-consolidation that id is a judge scorer's id, and the two names address the same record.

```python
evaluator = client.monitor.online_evaluators.builder(
    name="Helpfulness",
    evaluation_settings_id=settings.id,
    sample_rate=0.1,
    alert_threshold=5,
    severity="medium",
).publish()

client.monitor.online_evaluators.get(evaluator.id)      # -> MonitorOnlineEvaluator
client.monitor.online_evaluators.list()                 # -> list[MonitorOnlineEvaluator]
client.monitor.online_evaluators.update(evaluator.id, alert_threshold=None)  # score only, never raise a signal
client.monitor.online_evaluators.delete(evaluator.id)

client.monitor.online_evaluators.ratings(evaluator.id, window="7d")  # -> list[OnlineEvaluatorRatingPoint]
client.monitor.online_evaluators.events(evaluator.id, window="7d")   # -> list[OnlineEvaluatorEvent], worst-rated first
```

A score below `alert_threshold` raises/updates a signal the same way a failing pattern does, readable from `client.monitor.signals` alongside pattern-raised ones, deduped by evaluator and agent so a recurring low score accumulates one `occurrence_count` instead of a new signal per trace.

#### `builder()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Evaluator display name |
| `evaluation_settings_id` | `str` | required | Id of an existing Evaluator config (criteria, judge prompt, judge model) |
| `sample_rate` | `float` | `0.1` | Fraction of traffic actually scored. Every check is a real LLM call against your own API key, keep this low unless you want to score everything |
| `scope_mode` / `agent_ids` | `str` / `list[str]` | `"all"` / `[]` | Restrict this evaluator to specific agents instead of the whole workspace |
| `enabled` | `bool` | `True` | Whether the evaluator scores at all |
| `alert_threshold` | `float \| None` | `5` | A score below this raises/updates a signal. `None` scores without ever raising one |
| `severity` | `str` | `"medium"` | `"low"`, `"medium"`, `"high"`, or `"critical"`, applied to signals this evaluator raises |
| `scope` | `str` | `"trace"` | `"trace"` scores individual traces at ingest; `"session"` scores whole conversations instead |
| `idle_seconds` | `int` | `120` | For `scope="session"`: how long a session must be quiet before the engine judges it. A session that resumes after being scored gets re-scored on its next idle |

`publish()` returns a `MonitorOnlineEvaluator` with `.id`. `update()` is a partial update, same field names as the builder, pass only what changes.

A `scope="session"` evaluator is judged by the engine's background sweep rather than at ingest: only sessions with 2+ turns are considered, and a verdict below `alert_threshold` raises a signal exactly like the trace-scoped case. Verdicts appear in the dashboard's session detail view alongside the built-in coherence score.

---

## Async support

The decorator wraps both sync and async functions (it detects coroutine functions and awaits them):

```python
@tracer.trace("async-agent", framework="openai-agents")
async def handle_async(query: str) -> str:
    response = await async_llm_client.complete(query)
    return response
```

The context-manager form is a regular (synchronous) context manager - use plain `with` inside async code, not `async with`:

```python
async def handle(query: str) -> str:
    with tracer.trace("async-agent") as span:
        span.input = query
        result = await async_llm_client.complete(query)
        span.output = result
        return result
```

---

## Configuration reference

```python
from agentx import AgentX

client = AgentX(
    api_key="ax_live_xxxxxxxxxxxxxxxx",         # or set AGENTX_API_KEY
    workspace_id="...",                          # optional - or set AGENTX_WORKSPACE_ID
    base_url="http://localhost:4700/api/v1",    # optional - or set AGENTX_API_BASE_URL;
                                                 # defaults to the hosted API
)
```

Constructing the client makes no network call; `client.ping()` is the fail-fast startup check.

---

## Delivery behavior and limits

- **Queueing** - traces are enqueued (up to 500 in flight) and drained by a background daemon thread. On overflow, or when retries are exhausted, the trace is dropped **with a logged warning** (first drop, then every 50th, with a cumulative count) - never silently.
- **Retries** - each queued trace is retried up to 3 times with backoff on connection errors, 429, and 5xx responses; a 429's `Retry-After` header is honored. `sync=True` sends block once with a 10s timeout and do not retry - a failed sync send just means `span.trace_id` stays `None`.
- **Payload truncation** - `input`, `output`, and `metadata` are serialized best-effort before sending: nesting deeper than 4 levels, dicts/lists beyond 30 entries, and unserializable objects are truncated/stringified (long fallback strings cut to 200 chars) to keep payloads bounded.
- **First failure warns** - the first delivery failure per client logs at WARNING with a hint (bad key vs. bad URL); repeats log at DEBUG. `client.ping()` at startup fails fast instead.
