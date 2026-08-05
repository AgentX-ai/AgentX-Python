![Logo](https://agentx-resources.s3.us-west-1.amazonaws.com/AgentX-logo-387x60.png)

[![PyPI version](https://img.shields.io/pypi/v/agentx-python)](https://pypi.org/project/agentx-python/)
[![Python versions](https://img.shields.io/pypi/pyversions/agentx-python)](https://pypi.org/project/agentx-python/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

The official Python SDK for **[AgentX](https://app.agentx.so/)** — build, chat with, orchestrate, and trace AI agents in a few lines of code.

Also see [SDK Developer Docs](https://developers.agentx.so), [API Reference Docs](https://docs.agentx.so/reference)

---

## Contents

- [Why AgentX](#why-agentx)
- [Installation](#installation)
- [Authentication](#authentication)
- [Quick start](#quick-start)
- [Working with agents](#working-with-agents)
  - [List agents](#list-agents)
  - [Start a conversation](#start-a-conversation)
  - [Chat (streaming and non-streaming)](#chat-streaming-and-non-streaming)
- [Workforce (multi-agent orchestration)](#workforce-multi-agent-orchestration) — teams of agents with a designated manager
- [Production tracing](#production-tracing) — record live agent runs from any framework
- [Monitor](#monitor) — automatic production monitoring, patterns and signals
- [Custom agent evaluations](#custom-agent-evaluations) — LLM-as-a-judge, cosine / Jaccard similarity
- [Self-host](#self-host) — run Trace/Evaluate/Monitor on your own machine instead of the hosted dashboard
- [Links](#links)

---

## Why AgentX

- **Simple mental model** — `Agent → Conversation → Message`.
- **Chain-of-thought** is built in, no extra plumbing.
- **Bring any LLM** — works across major open and closed-source vendors.
- **Batteries included** — voice (ASR/TTS), image generation, document/CSV/Excel/OCR, RAG with built-in re-ranking.
- **MCP support** — connect any Model Context Protocol server.
- **Multi-agent orchestration** — workforces of agents with a designated manager, across LLM vendors.
- **Production tracing** — one decorator or context manager records every agent run (input, output, latency, tool calls, token usage) into your workspace, for any framework.
- **Agent Evaluations** — score any agent (LangChain, CrewAI, OpenAI, Anthropic, HTTP, …) with LLM-as-a-judge ratings plus optional cosine and Jaccard similarity metrics. Configurable judge prompt/model (OpenAI or Anthropic), per-question judge guidelines, and smoke testing for phrasing robustness.
- **A2A** — Each agent can be published with agent-to-agent protocol compatible.

---

## Installation

```bash
pip install --upgrade agentx-python
```

Requires Python 3.9 or newer.

---

## Authentication

Get your API key at [app.agentx.so](https://app.agentx.so), then either pass it inline or expose it as an environment variable.

```python
# Option A — pass the key inline
from agentx import AgentX
client = AgentX(api_key="your-api-key-here")

# Option B — set AGENTX_API_KEY in your environment, then:
client = AgentX.from_env()
```

---

## Quick start

```python
from agentx import AgentX

client = AgentX.from_env()

# Pick an existing agent and chat with it
agent = client.list_agents()[0]
conversation = agent.new_conversation()
print(conversation.chat("Hello! What can you help me with?"))
```

That's it. The remaining sections show the same primitives in more detail.

---

## Working with agents

### List agents

```python
agents = client.list_agents()
print(f"You have {len(agents)} agents")
```

### Start a conversation

```python
agent = client.get_agent(id="<agent-id>")

# Either resume an existing conversation…
existing = agent.list_conversations()
last = existing[-1]
for msg in last.list_messages():
    print(msg)

# …or start a fresh one
conversation = agent.new_conversation()
```

### Chat (streaming and non-streaming)

```python
# Blocking — returns the full response once it's ready
response = conversation.chat("What is your name?")
print(response)

# Streaming — yields ChatResponse objects as the model produces them
for chunk in conversation.chat_stream("Hello, what is your name?"):
    if chunk.text:
        print(chunk.text, end="")
```

Each `ChatResponse` chunk exposes the agent's `text` and, where applicable, its `cot` (chain-of-thought) reasoning, along with any retrieved references and tasks.

---

## Workforce (multi-agent orchestration)

A **workforce** is a team of agents coordinated by a designated manager agent. Workforces can mix LLM vendors and route work between specialists.

```python
workforces = client.list_workforces()
workforce = workforces[0]

print(f"Workforce: {workforce.name}")
print(f"Manager:   {workforce.manager.name}")
print(f"Agents:    {[a.name for a in workforce.agents]}")

# Chat with the workforce — the manager decides which agent(s) to delegate to
conversation = workforce.new_conversation()
for chunk in workforce.chat_stream(conversation.id, "How can you help me with this project?"):
    if chunk.text:
        print(chunk.text, end="")
```

---

## Production tracing

Record live agent runs into your workspace with a single decorator or context manager — no changes to your agent's logic. Traces appear in the **Live Traces** tab and can be evaluated against your test datasets with [`tracer.evaluate_trace()`](TRACING.md#tracerevaluate_trace).

```python
from agentx import AgentX

client = AgentX.from_env()
tracer = client.tracer

@tracer.trace("customer-support-agent", framework="langchain", model="gpt-4o")
def handle_query(query: str) -> str:
    return chain.invoke(query)

# Every call is automatically traced: input, output, latency, tool calls, token usage
handle_query("How do I reset my password?")
tracer.flush(timeout=10)  # ensure delivery before the process exits
```

Prefer full control over what gets captured? Use the context manager instead:

```python
with tracer.trace("rag-agent", framework="langchain") as span:
    span.input = {"query": query, "user_id": user_id}

    kb_result = search_knowledge_base(query)
    span.add_tool_call("search_knowledge_base", input=query, output=kb_result, latency_ms=190)

    span.output = llm.invoke(f"Context: {kb_result}\n\nQuery: {query}")
```

### Framework integrations

Each integration auto-captures LLM calls, tool calls, and token usage — install the matching extra:

| Framework             | Install                                      | Integration              |
| --------------------- | -------------------------------------------- | ------------------------ |
| LangChain             | `pip install "agentx-python[langchain]"`     | `AgentXCallbackHandler`  |
| CrewAI                | `pip install "agentx-python[crewai]"`        | `AgentXCrewObserver`     |
| OpenAI Agents SDK     | `pip install "agentx-python[openai-agents]"` | `AgentXTracingProcessor` |
| OpenAI (raw client)   | `pip install "agentx-python[openai]"`        | `patch_openai_client`    |
| Anthropic             | `pip install "agentx-python[anthropic]"`     | `patch_anthropic_client` |
| Google ADK            | `pip install "agentx-python[google-adk]"`    | `AgentXADKPlugin`        |
| Google GenAI (Gemini) | `pip install "agentx-python[google-genai]"`  | `patch_genai_client`     |
| LiteLLM               | `pip install "agentx-python[litellm]"`       | `AgentXLiteLLMLogger`    |
| LlamaIndex             | `pip install "agentx-python[llamaindex]"`    | `AgentXLlamaIndexHandler`|
| AutoGen                | `pip install "agentx-python[autogen]"`       | `AgentXAutoGenObserver`  |

Or plain Python — wrap any function with `@tracer.trace(...)` and it just works, no framework required.

Running specialist agents in parallel with a `ThreadPoolExecutor`? Wrap each worker body in `tracer.use_span(span)` so their steps land on the parent trace instead of becoming independent traces — see [TRACING.md](TRACING.md) for the full pattern.

See **[TRACING.md](TRACING.md)** for the complete guide — session grouping, error handling, async support, and the full API reference.

---

## Monitor

Automatic production monitoring: check traces against detection patterns and get back triage-ready signals. A **pattern** is a first-class SDK resource with a real id, just like a `Dataset` or `EvaluationSettings`: create one once, then reference it by id at trace time.

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

`monitor=True` checks the trace immediately, no dashboard setup required. `pattern_ids` restricts detection to exactly those patterns; omit it to run the full default sweep instead (built-in checks like empty response, trace error, and latency regression, plus every pattern enabled for the workspace).

This works independently of the dashboard's per-agent monitoring toggle (Governance > Observe > Agents), which still auto-checks every trace from an agent once enabled there, with no code changes needed either way.

Read the resulting alerts back with `client.monitor.signals.list()`/`.get()`, no dashboard required:

```python
for signal in client.monitor.signals.list(severity="high"):
    print(signal.summary, signal.occurrence_count)
```

Per-agent coverage/threshold settings (sample rate, retention, and threshold overrides like the built-in "Latency regression" pattern's threshold) are `client.monitor.profile.get()`/`.update()`:

```python
client.monitor.profile.update("agent_123", threshold_overrides={"latencyMs": 15000})
```

Self-host also has **online evaluators**: a real LLM judge scoring a sample of live traffic continuously, distinct from a pattern's rule-matching. A score below `alert_threshold` raises a signal the same way a failing pattern does, deduped and triage-ready in `client.monitor.signals`.

```python
evaluator = client.monitor.online_evaluators.builder(
    name="Helpfulness",
    evaluation_settings_id=settings.id,
    sample_rate=0.1,
    alert_threshold=5,
).publish()

client.monitor.online_evaluators.ratings(evaluator.id, window="7d")
```

See **[TRACING.md](TRACING.md)** for the complete Monitor guide.

---

## Custom agent evaluations

Evaluate **any** AI agent — LangChain, CrewAI, AutoGen, LlamaIndex, OpenAI, Anthropic, HTTP endpoints, or plain Python — using AgentX as the scoring and reporting backend. Includes optional **cosine** and **Jaccard** similarity metrics alongside LLM-graded ratings.

```python
report = (
    client.evaluations
    .run(dataset_id="evds_…", subject={"kind": "custom_agent", "framework": "raw_python"})
    .execute(my_agent_fn)
    .finalize()
    .analyze()
)

print(report.average_rating)       # LLM-graded score, 0–10
print(report.cosine_similarity)    # embedding cosine, 0–1 (None if not enabled)
print(report.jaccard_similarity)   # token-set overlap, 0–1 (None if not enabled)

print(report.summary)              # AI-generated narrative from .analyze()
print(report.recommendations)      # list of prioritized, actionable fixes
```

`.analyze()` also generates a full qualitative report (strengths, weaknesses, instruction adherence, reasoning quality, and recommendations), running the same durable, multi-judge pipeline as the dashboard's "Analyze" button. `analyze(mode=..., quality_mode=..., judges=[...])` controls how items are scored and by which models. See [AI analysis report](EVALUATIONS.md#ai-analysis-report) in the full guide for the complete field and parameter reference.

Ask a case's question several extra ways each run, LLM-paraphrased server-side, to catch agents that break on phrasing rather than substance, and override the judge's prompt/model per config, see [Smoke testing](EVALUATIONS.md#smoke-testing-phrasing-robustness) and [Configuring the judge](EVALUATIONS.md#configuring-the-judge) in the full guide.

Since AgentX doesn't own your agent's code, `client.evaluations.prompts` lets AgentX become your prompt's *source of truth* instead — the same problem LangSmith's Prompt Hub and Langfuse's Prompt Management solve. Pull a version at runtime, tag your eval runs (or live traces) with it, and let a judge propose a rewrite from your real worst-rated results — a human always has to approve before it publishes:

```python
prompt = client.evaluations.prompts.get("support-agent-system-prompt")  # or prompt.id
# use prompt.text as your own agent's system prompt

client.evaluations.run(
    dataset_id="evds_…",
    subject={"kind": "custom_agent", "metadata": {"promptName": prompt.name}},
).execute(my_agent_fn)
```

See [Prompt registry](EVALUATIONS.md#prompt-registry) in the full guide, or [self-host's docs](https://docs.agentx.so/self-host#prompt-registry) for the "Suggest improvement" dashboard flow (self-host only — no hosted-SaaS equivalent yet).

See **[EVALUATIONS.md](EVALUATIONS.md)** for the full guide — dataset builder, framework adapters, similarity metrics, smoke testing, judge configuration, prompt registry, and the complete API reference.

---

## Self-host

Prefer to run Trace/Evaluate/Monitor locally instead of the hosted dashboard — no account, bring your own LLM keys? This SDK ships a launcher for [AgentX-trace-eval](https://github.com/AgentX-ai/AgentX-trace-eval), a separate, portable governance engine:

```bash
agentx-trace-eval --dev
```

The first run downloads the engine (and dashboard) into `~/.agentx/bin` and prints a local API key; every run after that just starts it. Point this SDK at it instead of the hosted API:

```bash
export AGENTX_API_BASE_URL=http://localhost:4700/api/v1
export AGENTX_API_KEY=<printed by agentx-trace-eval on first run>
```

`agentx-trace-eval` isn't this SDK's own code — the engine itself is a separate, compiled binary, downloaded on demand rather than bundled into this package, so installing `agentx-python` doesn't get any heavier for the (much more common) case of just talking to the hosted AgentX API. See that repo's README for what's included, and `AGENTX_INSTALL_DIR`/`AGENTX_TRACE_EVAL_VERSION`/`AGENTX_TRACE_EVAL_SKIP_WEB` env vars to control where/what it installs.

---

## Links

- **Dashboard** — [app.agentx.so](https://app.agentx.so)
- **Website** — [agentx.so](https://www.agentx.so/)
- **PyPI** — [agentx-python](https://pypi.org/project/agentx-python/)
- **Tracing docs** — [TRACING.md](TRACING.md)
- **Evaluations docs** — [EVALUATIONS.md](EVALUATIONS.md)
- **Monitor docs** — [docs.agentx.so/sdk/monitor](https://docs.agentx.so/sdk/monitor)
- **Self-host** — [AgentX-trace-eval](https://github.com/AgentX-ai/AgentX-trace-eval)
