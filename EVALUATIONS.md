## Custom Agent Evaluations — Installation

Install only what you need, or everything at once:

```bash
# Minimum (AgentX SDK + env file support)
pip install agentx-python python-dotenv==1.2.2

# OpenAI
pip install openai==2.34.0

# Anthropic
pip install anthropic==0.99.0

# Google Gemini  (use google-genai, NOT the deprecated google-generativeai)
pip install google-genai==1.75.0

# LangChain
pip install langchain==1.2.17 langchain-openai==1.2.1

# LlamaIndex
pip install llama-index==0.14.21

# CrewAI
pip install crewai==1.14.4

# AutoGen
pip install pyautogen==0.10.0

# Everything at once (from the examples directory)
pip install -r examples/evaluations/requirements.txt
```

---

## Custom Agent Evaluations

Evaluate **any AI agent** — LangChain, CrewAI, AutoGen, LlamaIndex, OpenAI, Anthropic, HTTP endpoints, or plain Python — using AgentX as a scoring and reporting backend. Your agent runs locally; AgentX scores results and generates a full analysis report.

### How it works

1. **Build a dataset** — define cases (queries + acceptance/rejection criteria).
2. **Run your agent** — the SDK calls your function or endpoint for each case.
3. **Finalize + analyze** — AgentX scores every response and generates a report.
4. **View results** — in the terminal and on the AgentX dashboard.

```
Your agent (local)  →  AgentX SDK  →  AgentX API (scores + analyzes)  →  Report
```

### Quick start

```python
from agentx import AgentX

client = AgentX(api_key="your-key")   # or AgentX.from_env()

def my_agent(case):
    return f"Answer to: {case.query}"

report = (
    client.evaluations
    .run(
        dataset_id="existing-dataset-id",
        subject={"kind": "custom_agent", "displayName": "My Agent", "framework": "raw_python"},
    )
    .execute(my_agent)
    .finalize()
    .analyze()
)

print(f"Average rating:    {report.average_rating:.2f}")

# Optional similarity metrics — present only when enabled on the dataset.
if report.cosine_similarity is not None:
    print(f"Cosine similarity: {report.cosine_similarity:.3f}")   # 0–1
if report.jaccard_similarity is not None:
    print(f"Jaccard similarity:{report.jaccard_similarity:.3f}")  # 0–1

print(f"Dashboard: {report.dashboard_url}")
```

### Environment variables

| Variable | Description |
|---|---|
| `AGENTX_API_KEY` | Required. Your AgentX API key. |
| `AGENTX_API_BASE_URL` | Optional. Override the API base URL (useful for local dev). |

You can also pass `base_url` directly to the constructor:

```python
# Point at your local dev server
client = AgentX(api_key="your-key", base_url="http://localhost:3000/api/v1/custom-agent-evaluations")
```

---

### Dataset builder

Create datasets with the fluent builder API:

```python
dataset = (
    client.evaluations.datasets
    .builder(
        name="Support Agent v2",
        number_of_requests=3,          # runs per case
        acceptance_criteria="Accurate, concise, grounded in docs.",
        rejection_criteria="No hallucinated policies.",
    )
    .add_case(
        query="How do I reset my password?",
        expected_results="Explain the password reset process step by step.",
    )
    .add_case(
        query="What payment methods do you accept?",
        expected_results="List supported payment methods clearly.",
    )
    .publish()
)

print(dataset.id)   # use this id in .run()
```

#### Import from CSV

```python
dataset = client.evaluations.datasets.from_csv(
    path="cases.csv",
    name="My Dataset",
    number_of_requests=2,
    acceptance_criteria="...",
    rejection_criteria="...",
)
```

CSV format:
```
query,expected_results
"How do I reset my password?","Explain the steps clearly."
"What is your refund policy?","Describe refund terms."
```

Optional `case_id` column for stable idempotency keys across re-runs.

---

### Framework examples

All examples follow the same pattern: wrap your framework's output in a function that accepts an `EvaluationCase` and returns a `str`, `dict`, or `EvaluationResult`.

#### Raw Python callable

```python
from agentx.evaluations.models import EvaluationCase

def my_agent(case: EvaluationCase) -> str:
    return f"Answer to: {case.query}"

report = (
    client.evaluations
    .run(dataset_id="...", subject={"kind": "custom_agent", "displayName": "My Bot", "framework": "raw_python"})
    .execute(my_agent)
    .finalize()
    .analyze()
)
```

Full example: [`examples/evaluations/basic_callable_eval.py`](examples/evaluations/basic_callable_eval.py)

---

#### OpenAI SDK

```python
from openai import OpenAI
oai = OpenAI()

def openai_agent(case):
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful support agent."},
            {"role": "user", "content": case.query},
        ],
    )
    return {"output": resp.choices[0].message.content, "metadata": {"model": resp.model}}

report = (
    client.evaluations
    .run(dataset_id="...", subject={"kind": "custom_agent", "displayName": "GPT-4o-mini", "framework": "openai"})
    .execute(openai_agent)
    .finalize()
    .analyze()
)
```

Full example: [`examples/evaluations/openai_eval.py`](examples/evaluations/openai_eval.py)

---

#### Anthropic SDK

```python
import anthropic
ant = anthropic.Anthropic()

def claude_agent(case):
    msg = ant.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system="You are a helpful customer support agent.",
        messages=[{"role": "user", "content": case.query}],
    )
    return {"output": msg.content[0].text, "metadata": {"model": msg.model}}

report = (
    client.evaluations
    .run(dataset_id="...", subject={"kind": "custom_agent", "displayName": "Claude Haiku", "framework": "anthropic"})
    .execute(claude_agent)
    .finalize()
    .analyze()
)
```

Full example: [`examples/evaluations/anthropic_eval.py`](examples/evaluations/anthropic_eval.py)

---

#### LangChain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful support agent."),
    ("human", "{query}"),
]) | llm

def langchain_agent(case):
    result = chain.invoke({"query": case.query})
    return {"output": result.content, "metadata": {"framework": "langchain"}}

report = (
    client.evaluations
    .run(dataset_id="...", subject={"kind": "custom_agent", "displayName": "LangChain Bot", "framework": "langchain"})
    .execute(langchain_agent)
    .finalize()
    .analyze()
)
```

Full example: [`examples/evaluations/langchain_eval.py`](examples/evaluations/langchain_eval.py)

---

#### CrewAI

```python
from crewai import Agent, Task, Crew, Process

agent = Agent(role="Support Specialist", goal="Answer customer questions accurately.", backstory="...", verbose=False)

def crewai_agent(case):
    task = Task(description=case.query, agent=agent, expected_output="A concise customer support response.")
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return {"output": str(crew.kickoff()), "metadata": {"framework": "crewai"}}

report = (
    client.evaluations
    .run(dataset_id="...", subject={"kind": "custom_agent", "displayName": "CrewAI Agent", "framework": "crewai"})
    .execute(crewai_agent)
    .finalize()
    .analyze()
)
```

Full example: [`examples/evaluations/crewai_eval.py`](examples/evaluations/crewai_eval.py)

---

#### AutoGen / AG2

```python
from autogen import ConversableAgent   # or: from ag2 import ConversableAgent

assistant = ConversableAgent("SupportAgent", system_message="Answer concisely.", llm_config={...}, human_input_mode="NEVER")
user_proxy = ConversableAgent("User", llm_config=False, human_input_mode="NEVER", max_consecutive_auto_reply=0)

def autogen_agent(case):
    user_proxy.initiate_chat(assistant, message=case.query, max_turns=1, silent=True)
    messages = assistant.chat_messages.get(user_proxy, [])
    output = next((m["content"] for m in reversed(messages) if m.get("role") == "assistant"), "")
    return {"output": output, "metadata": {"framework": "autogen"}}

report = (
    client.evaluations
    .run(dataset_id="...", subject={"kind": "custom_agent", "displayName": "AutoGen Agent", "framework": "autogen"})
    .execute(autogen_agent)
    .finalize()
    .analyze()
)
```

Full example: [`examples/evaluations/autogen_eval.py`](examples/evaluations/autogen_eval.py)

---

#### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI

docs = [Document(text="Our refund policy allows returns within 30 days.")]
index = VectorStoreIndex.from_documents(docs)
engine = index.as_query_engine(llm=OpenAI(model="gpt-4o-mini"))

def llamaindex_agent(case):
    response = engine.query(case.query)
    return {"output": str(response), "metadata": {"framework": "llamaindex"}}

report = (
    client.evaluations
    .run(dataset_id="...", subject={"kind": "custom_agent", "displayName": "LlamaIndex RAG", "framework": "llamaindex"})
    .execute(llamaindex_agent)
    .finalize()
    .analyze()
)
```

Full example: [`examples/evaluations/llamaindex_eval.py`](examples/evaluations/llamaindex_eval.py)

---

#### HTTP endpoint

Use `HttpEndpointAdapter` to evaluate any agent exposed as an HTTP service — FastAPI, LangServe, Flask, n8n webhooks, etc.

```python
from agentx.evaluations.adapters.http_endpoint import HttpEndpointAdapter

adapter = HttpEndpointAdapter(
    url="http://localhost:8000/agent/invoke",
    headers={"Authorization": "Bearer your-token"},
    timeout=30,
)

report = (
    client.evaluations
    .run(dataset_id="...", subject={"kind": "custom_agent", "displayName": "My API Agent", "framework": "other"})
    .execute(adapter)
    .finalize()
    .analyze()
)
```

Expected endpoint contract:
```
POST /your-endpoint
Content-Type: application/json
Body: { "query": "..." }
Response: { "output": "..." }
```

Full example: [`examples/evaluations/http_endpoint_eval.py`](examples/evaluations/http_endpoint_eval.py)

---

#### Precomputed results (n8n, Flowise, batch jobs)

Submit outputs you already have without running any agent during evaluation:

```python
from agentx.evaluations.adapters.precomputed import PrecomputedAdapter

outputs = {
    0: "To reset your password, go to Login → Forgot Password.",
    1: "We accept Visa, Mastercard, PayPal, and bank transfers.",
}

adapter = PrecomputedAdapter(outputs)

report = (
    client.evaluations
    .run(dataset_id="...", subject={"kind": "custom_agent", "displayName": "n8n Batch", "framework": "n8n", "runtime": "low_code"})
    .execute(adapter)
    .finalize()
    .analyze()
)
```

Full example: [`examples/evaluations/csv_import_eval.py`](examples/evaluations/csv_import_eval.py)

---

### EvaluationSubject fields

| Field | Values | Description |
|---|---|---|
| `kind` | `custom_agent` | Always `custom_agent` for external agents |
| `displayName` | any string | Human-readable name shown in the dashboard |
| `framework` | `raw_python`, `openai`, `anthropic`, `langchain`, `llamaindex`, `crewai`, `autogen`, `n8n`, `flowise`, `other` | Framework used |
| `runtime` | `local`, `ci`, `customer_hosted`, `low_code` | Where the agent runs |
| `version` | any string | Optional version tag for the agent |
| `endpoint` | URL | Optional, for HTTP-based agents |

### Similarity metrics (optional)

Each scored result can be enriched with two reference-based similarity scores comparing your agent's response to the `expected_results` of the case:

| Metric | What it measures | Cost |
|---|---|---|
| **Cosine** (vector similarity) | Cosine of OpenAI embeddings of `expected_results` vs the actual response. Captures semantic similarity. | One embedding API call per case. |
| **Jaccard** | Token-set overlap `|A ∩ B| / |A ∪ B|` over lowercased word tokens. Pure lexical match. | Free — no API calls. |

Both metrics are returned in the range `[0, 1]` and averaged across all scored results in the report.

**Enable them on the dataset** (via the AgentX dashboard or the dataset API):

```jsonc
{
  "vectorSimilarity":  { "enabled": true, "model": "text-embedding-3-small" },
  "jaccardSimilarity": { "enabled": true }
}
```

**Read them on the report:**

```python
report = client.evaluations.run(...).execute(my_agent).finalize().analyze()

# Top-level convenience accessors — return None when the metric was not
# enabled on the dataset, or no case has a value yet.
report.average_rating       # float | None  — same as report.statistics.average_rating
report.cosine_similarity    # float | None  — averaged across cases (0–1)
report.jaccard_similarity   # float | None  — averaged across cases (0–1)

# Same values are also available nested under the statistics block:
report.statistics.cosine_similarity
report.statistics.jaccard_similarity
```

Cases where `expected_results` is empty or the agent returned an error are skipped from the average, so a sparse dataset still produces a meaningful score. If neither toggle was on for the dataset, both properties return `None`.

### Return value from your agent function

Your callable can return any of:

| Return type | Behavior |
|---|---|
| `str` | Used directly as the output text |
| `dict` with `"output"` key | Output text from `output`, rest stored as metadata |
| `EvaluationResult` | Full control — pass rating, justification, trace, timings |

### Security and redaction

The SDK automatically scrubs secrets from outputs and metadata before uploading:
- `sk-...` API keys
- Bearer tokens
- Authorization headers
- Password-like fields

Raw agent outputs, prompts, and CoT reasoning are **never uploaded** — only the text response, metadata you explicitly include, and optional observable trace summaries.
