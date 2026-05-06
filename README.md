![Logo](https://agentx-resources.s3.us-west-1.amazonaws.com/AgentX-logo-387x60.png)

[![PyPI version](https://img.shields.io/pypi/v/agentx-python)](https://pypi.org/project/agentx-python/)

---

## A fast way to build AI Agents and create agent workforces

The official AgentX Python SDK for [AgentX](https://www.agentx.so/)

Why build AI agents with AgentX?

- Simplicity — Agent → Conversation → Message structure.
- Chain-of-thought built in.
- Choose from most open and closed-source LLM vendors.
- Built-in Voice (ASR, TTS), Image Gen, Document, CSV/Excel, OCR, and more.
- Support for all running MCP (Model Context Protocol) servers.
- RAG with built-in re-rank.
- Multi-agent workforce orchestration.
- Multiple agents working together with a designated manager agent.
- Cross-LLM-vendor, multi-agent orchestration.
- A2A — agent-to-agent protocol (coming soon)

## Installation

```bash
pip install --upgrade agentx-python
```

## Quick Start

```python
from agentx import AgentX

client = AgentX(api_key="your-api-key-here")

agents = client.list_agents()
print(f"You have {len(agents)} agents")

if agents:
    agent = agents[0]
    conversation = agent.new_conversation()
    response = conversation.chat("Hello! What can you help me with?")
    print(response)
```

## Usage

Provide an `api_key` inline or set `AGENTX_API_KEY` as an environment variable.
Get your API key at https://app.agentx.so

### Agent

```python
from agentx import AgentX

client = AgentX(api_key="<your api key here>")
print(client.list_agents())
```

### Conversation

```python
my_agent = client.get_agent(id="<agent id here>")

existing_conversations = my_agent.list_conversations()
last_conversation = existing_conversations[-1]
msgs = last_conversation.list_messages()
print(msgs)
```

### Chat

```python
a_conversation = my_agent.get_conversation(id="<conversation id here>")

response = a_conversation.chat_stream("Hello, what is your name?")
for chunk in response:
    print(chunk)
```

\*`cot` stands for chain-of-thought

### Workforce

```python
from agentx import AgentX

client = AgentX(api_key="<your api key here>")

workforces = client.list_workforces()
workforce = workforces[0]
print(f"Workforce: {workforce.name}")
print(f"Manager: {workforce.manager.name}")
print(f"Agents: {[agent.name for agent in workforce.agents]}")
```

#### Chat with Workforce

```python
conversation = workforce.new_conversation()
response = workforce.chat_stream(conversation.id, "How can you help me with this project?")
for chunk in response:
    if chunk.text:
        print(chunk.text, end="")
```

---

## Custom Agent Evaluations

See **[EVALUATIONS.md](EVALUATIONS.md)** for full documentation — installation, dataset builder, framework examples (OpenAI, Anthropic, Google, LangChain, CrewAI, AutoGen, LlamaIndex, HTTP endpoints), and the complete API reference.
