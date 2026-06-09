![Logo](https://agentx-resources.s3.us-west-1.amazonaws.com/AgentX-logo-387x60.png)

[![PyPI version](https://img.shields.io/pypi/v/agentx-python)](https://pypi.org/project/agentx-python/)

The official Python SDK for **[AgentX](https://www.agentx.so/)** — build, chat with, and orchestrate AI agents in a few lines of code.

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
- [Workforce (multi-agent orchestration)](#workforce-multi-agent-orchestration)
- [Agent Evaluations](#custom-agent-evaluations) — LLM-as-a-judge, cosine / Jaccard similarity
- [Links](#links)

---

## Why AgentX

- **Simple mental model** — `Agent → Conversation → Message`.
- **Chain-of-thought** is built in, no extra plumbing.
- **Bring any LLM** — works across major open and closed-source vendors.
- **Batteries included** — voice (ASR/TTS), image generation, document/CSV/Excel/OCR, RAG with built-in re-ranking.
- **MCP support** — connect any Model Context Protocol server.
- **Multi-agent orchestration** — workforces of agents with a designated manager, across LLM vendors.
- **Agent Evaluations** — score any agent (LangChain, CrewAI, OpenAI, Anthropic, HTTP, …) with LLM-as-a-judge ratings plus optional cosine and Jaccard similarity metrics.
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
```

See **[EVALUATIONS.md](EVALUATIONS.md)** for the full guide — dataset builder, framework adapters, similarity metrics, and the complete API reference.

---

## Links

- **Dashboard** — [app.agentx.so](https://app.agentx.so)
- **Website** — [agentx.so](https://www.agentx.so/)
- **PyPI** — [agentx-python](https://pypi.org/project/agentx-python/)
- **Evaluations docs** — [EVALUATIONS.md](EVALUATIONS.md)
