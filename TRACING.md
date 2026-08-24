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

# From environment variable (AGENTX_API_KEY)
client = AgentX.from_env()
```

Set the environment variable:

```bash
export AGENTX_API_KEY=ax_live_xxxxxxxxxxxxxxxx
```

---

## The `Tracer`

Access the tracer via `client.tracer`:

```python
tracer = client.tracer
```

All tracing methods are on the `tracer` object.

---

## `tracer.trace()` - decorator / context manager

The primary tracing interface. Captures the wrapped function's arguments as `input`, return value as `output`, wall-clock time as `latencyMs`, and any exception as `error`.

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

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | ✓ | Agent or operation label shown in the UI |
| `framework` | `str` | - | Framework identifier: `"langchain"`, `"crewai"`, `"openai-agents"`, `"anthropic"`, or custom |
| `model` | `str` | - | LLM model used, e.g. `"gpt-4o"`, `"claude-sonnet-4-6"` |
| `session_id` | `str` | - | Groups traces from the same user session or thread |
| `metadata` | `dict` | - | Arbitrary key-value metadata (not indexed, max 16 KB) |

### `_TraceSpan` methods (context manager only)

| Method / Attribute | Description |
|---|---|
| `span.input = value` | Override the captured input |
| `span.output = value` | Set the output (required in context manager mode) |
| `span.add_tool_call(name, *, input, output, latency_ms)` | Record a tool call made during the span |
| `span.set_error(message)` | Mark the span as failed with the given error message |

---

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

An exception escaping the block records the call with `success=False` plus the error text, then propagates unchanged so your own error handling still runs. That `success: false` is what Monitor's built-in "Tool failure" check and the dashboard's Tool quality column read, so a flaky tool shows up in triage without any extra wiring. To set the outcome yourself instead (an API that returned a well-formed error payload, say), use `tracer.record_tool_call(name, input=..., output=..., success=False, error="...")`.

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

Evaluate a previously submitted trace against a dataset, without re-running the agent. Returns a score and justification.

```python
result = tracer.evaluate_trace(
    trace_id="6876abc123def456789abc01",
    dataset_id="6876ddd222bbb333ccc444ee",
    question_index=0,   # optional - which question to score against
)

print(result.rating)         # 1–5
print(result.justification)  # LLM explanation
print(result.run_id)         # ID of the created eval run
```

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `trace_id` | `str` | ✓ | Trace ID returned by `POST /ingest/traces` |
| `dataset_id` | `str` | ✓ | Evaluation dataset to score against |
| `question_index` | `int` | - | 0-based question index. Omit to score against general criteria only |

### Return type: `TraceEvalResult`

```python
@dataclass
class TraceEvalResult:
    run_id: str
    trace_id: str
    rating: int          # 1–5
    justification: str
    status: str          # "completed"
```

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

Built-in detectors: empty response, trace/tool errors, latency regressions, and (native chat agents only) negative user feedback. Custom patterns: keyword, regex, or an LLM-judged semantic rubric, created via the dashboard or `client.monitor.patterns`. A match becomes a signal, deduped against repeat occurrences of the same issue. A trace that matches nothing counts toward the agent's health rate instead.

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
| `sample_rate` | `float` | `1.0` | Fraction of matching traces to actually check, `0.0`–`1.0` |
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

Get/update one agent's Monitor coverage and detection settings, the same settings shown in the dashboard's per-agent monitoring settings dialog (Observe > Patterns > Agents view): coverage mode, sample rate, retention, redaction, approval policy, and `threshold_overrides` for built-in detectors that take a configurable threshold (e.g. the "Latency regression" pattern's threshold, which otherwise defaults to 20000ms).

```python
profile = client.monitor.profile.get("agent_123")
print(profile.coverage_mode if profile else "never configured, on defaults")

# Override just the latency-regression threshold, e.g. 15s instead of the 20s default.
client.monitor.profile.update("agent_123", threshold_overrides={"latencyMs": 15000})
```

`get()` returns `None` when the agent has never been configured (still on platform defaults). `update()` upserts and only changes the fields you pass, everything else on an existing profile is left as is:

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
| `redaction_mode` | `str` | `"none"`, `"standard"`, or `"strict"` |
| `approval_policy` | `dict[str, str]` | Per-action approval mode for autotune actions |

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

All tracing methods work with both sync and async functions:

```python
@tracer.trace("async-agent", framework="openai-agents")
async def handle_async(query: str) -> str:
    response = await async_llm_client.complete(query)
    return response

# Or in async context manager:
async with tracer.trace("async-agent") as span:
    span.input = query
    result = await async_llm_client.complete(query)
    span.output = result
```

---

## Configuration reference

```python
from agentx import AgentX

client = AgentX(
    api_key="ax_live_xxxxxxxxxxxxxxxx",   # Required (or use AGENTX_API_KEY env var)
    workspace_id="...",                    # Optional - explicit workspace override
    base_url="https://api.agentx.so",     # Optional - for self-hosted deployments
    timeout=10,                            # HTTP timeout in seconds (default 10)
)
```

---

## Limits

| Limit | Value |
|---|---|
| Traces per minute | 300 |
| Max tool calls per trace | 50 (excess silently truncated) |
| Max `input` / `output` size | 1 MB each |
| Max `metadata` size | 16 KB |

Traces that exceed size limits are submitted with the oversized field truncated and a warning logged to stderr.
