# Production Tracing — Python SDK

## Overview

The AgentX Python SDK lets you record production agent runs into your workspace with a single decorator or context manager — no changes to your agent's logic. Traces appear in the **Live Traces** tab and can be evaluated against your test datasets.

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

## `tracer.trace()` — decorator / context manager

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
| `framework` | `str` | — | Framework identifier: `"langchain"`, `"crewai"`, `"openai-agents"`, `"anthropic"`, or custom |
| `model` | `str` | — | LLM model used, e.g. `"gpt-4o"`, `"claude-sonnet-4-6"` |
| `session_id` | `str` | — | Groups traces from the same user session or thread |
| `metadata` | `dict` | — | Arbitrary key-value metadata (not indexed, max 16 KB) |

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

---

## Error handling

Exceptions raised inside a traced function or context manager are automatically captured as the `error` field and re-raised — they do not affect the trace submission:

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
    question_index=0,   # optional — which question to score against
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
| `question_index` | `int` | — | 0-based question index. Omit to score against general criteria only |

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
    workspace_id="...",                    # Optional — explicit workspace override
    base_url="https://api.agentx.so",     # Optional — for self-hosted deployments
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
