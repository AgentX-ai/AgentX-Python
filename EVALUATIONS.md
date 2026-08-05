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
if report.bleu_score is not None:
    print(f"BLEU score:        {report.bleu_score:.3f}")          # 0–1
if report.rouge_score is not None:
    print(f"ROUGE-L score:     {report.rouge_score:.3f}")         # 0–1

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

### Smoke testing (phrasing robustness)

Ask a case's question several extra ways each run, LLM-paraphrased server-side, to catch agents that are brittle to *how* something is asked rather than genuinely wrong. Every variant is graded against the same `expected_results` as the original question.

```python
dataset = (
    client.evaluations.datasets
    .builder(name="Support Agent v2", number_of_requests=3)
    .add_case(
        query="How do I reset my password?",
        expected_results="Explain the password reset process step by step.",
        smoke_test_count=3,                              # ask it 3 extra ways
        smoke_test_guidance="try terse, frustrated, and non-native-speaker phrasing",
    )
    .publish()
)
```

`smoke_test_guidance` is optional free text steering *what kind* of variants get generated (tone, adversarial phrasing, different languages, ...); leave it out for natural rewording only. Both the paraphrase text and the count are decided entirely server-side, reusing the same generation the AgentX dashboard's native runs use, `.execute()` just asks the extra variants and submits them for you, nothing to configure on the SDK side beyond these two kwargs. Ignored on `follow_up_questions`, only a case's opening question can be smoke-tested.

Variants show up as extra entries in `EvaluationCase`/`EvaluationResult`:

```python
def my_agent(case):
    if case.is_smoke_test_variant:
        logger.info("smoke-test variant: %s", case.query)
    return f"Answer to: {case.query}"
```

Smoke-test results are scored like any other run but reported as a **separate robustness signal**, they're excluded from `report.average_rating`/`live_stats` so a single hard phrasing doesn't skew your headline score.

---

### Per-question judge guideline

Extra grading instructions for one specific case, layered on top of the dataset's `acceptance_criteria`/`rejection_criteria`/`evaluation_criteria`:

```python
client.evaluations.datasets.builder(name="Support Agent v2").add_case(
    query="What's your refund policy?",
    expected_results="Refunds within 30 days, no restocking fee.",
    judge_guideline="Tone doesn't matter here, only check the 30-day window and no-fee terms are both present.",
)
```

---

### Evaluation Settings builder — reusable grading configs

By default, a dataset runs against the grading config it was created with (`number_of_requests`, `acceptance_criteria`, similarity metrics, etc. — see above). If you want to grade the **same dataset** against **different configs** (e.g. a strict config vs. a lenient one, or reuse one config across many datasets), create a standalone `EvaluationSettings` and pass its id to `.run()`:

```python
strict_settings = (
    client.evaluations.settings
    .builder(
        name="Strict grading",
        number_of_requests=5,
        acceptance_criteria="Must cite the exact policy clause.",
        rejection_criteria="Any hallucinated or paraphrased policy text.",
    )
    .publish()
)

report = (
    client.evaluations
    .run(dataset_id=dataset.id, subject={...}, evaluation_settings_id=strict_settings.id)
    .execute(my_agent)
    .finalize()
    .analyze()
)
```

Omit `evaluation_settings_id` to keep using the dataset's own config, exactly as before — this is fully additive, no existing code needs to change. The builder accepts the same config kwargs as `datasets.builder(...)` (`number_of_requests`, the three criteria fields, `vector_similarity`/`jaccard_similarity`/`bleu_score`/`rouge_score`, `sovereignty_models`, `judge_prompt`/`judge_model` below) but no `questions` — it's config-only and reusable.

```python
client.evaluations.settings.get(strict_settings.id)   # fetch one
client.evaluations.settings.list()                     # list all
```

#### Configuring the judge

Both `datasets.builder(...)` and `settings.builder(...)` accept `judge_prompt`/`judge_model` to override how the LLM-as-judge grades responses, applying to every scoring path (native dashboard runs and SDK/custom-agent runs alike):

```python
settings = (
    client.evaluations.settings
    .builder(
        name="Strict grading",
        judge_model="claude-opus-4-8",                    # any id from list_models()
        judge_prompt="""You are grading a customer support response.

**User Query:** {input}

**Agent Response:**
{output}

**Expected Results:**
{expected}

Score strictly: any missing policy detail is a failing response.""",
    )
    .publish()
)
```

- `judge_prompt` is a raw template. `{input}`, `{output}`, and `{expected}` are substituted in; everything else (chain of thought, capabilities/references, criteria, per-question `judge_guideline`, delegation notes) is appended automatically after it, so a custom prompt can restructure the grading philosophy without ever losing that context. Omit it to keep the default rubric.
- `judge_model` accepts any OpenAI or Anthropic model id (`client.evaluations.list_models(provider="Anthropic")` to discover valid ones). Omit it to keep the default (`gpt-5.5`).

---

### Prompt registry

**Self-host only** (see [Self-host](README.md#self-host)) — no hosted-SaaS equivalent yet.

AgentX doesn't own your agent's code, so it can't do what native Autotune does — branch and merge a config directly. `client.evaluations.prompts` solves the same "how do I close the loop" problem the way LangSmith's Prompt Hub and Langfuse's Prompt Management do instead: become the prompt's *source of truth*. Your agent pulls a version at runtime, you tag your evaluation evidence with which version it used, and "improvement" becomes propose → a human approves → publish a new version — never a direct edit to your deployed code.

```python
prompt = client.evaluations.prompts.create(
    name="support-agent-system-prompt",
    text="You are a helpful, empathetic customer support agent...",
)
# or, once it exists:
prompt = client.evaluations.prompts.get("support-agent-system-prompt")   # latest version
prompt = client.evaluations.prompts.get("support-agent-system-prompt", version=1)  # a specific one
prompt = client.evaluations.prompts.get(prompt.id)                       # by id instead of name
client.evaluations.prompts.list()                                        # every registered prompt

# use prompt.text as your own agent's actual system prompt, however you call your LLM
```

Tag whichever evidence you want the dashboard's judge (or the `improve-prompt` Claude Code skill)
to learn from by setting `metadata.promptName` — on a deliberate eval run:

```python
client.evaluations.run(
    dataset_id="evds_…",
    subject={"kind": "custom_agent", "metadata": {
        "promptName": prompt.name,
        "version": f"{prompt.name}@v{prompt.version}",   # ties into version comparison too, see below
    }},
).execute(my_agent_fn)
```

or on a live trace from real production traffic, scored continuously by a self-host Online Evaluator:

```python
with client.tracer.trace("support-agent", metadata={"promptName": prompt.name}) as span:
    ...  # your agent's own call
```

From the self-host dashboard: Governance → Improve → **Prompt Management** → a prompt's row menu → **Suggest
improvement**. It merges both kinds of evidence — deliberate eval runs (defaulting to the *current
published version only*, auto-widening to every version if there isn't enough recent evidence yet)
and worst-scoring Online Evaluator ratings from a recent time window — feeds the worst-rated
examples to a judge, and shows a full rewrite plus reasoning. **Nothing is saved until a human
clicks Publish as new version** — there is no `publish()` on this client; a rewrite only ever
reaches your agent through that one explicit, dashboard-only write. Your agent's next
`client.evaluations.prompts.get(name)` call picks up the new version immediately. Tagging
`metadata.version` as `<promptName>@v<N>` (shown above) means the dataset's **Compare versions**
dialog also tells you whether the published rewrite actually scored better, no separate comparison
view needed. Full details, including the judge-key-free `improve-prompt` Claude Code skill: see
[self-host's Prompt registry docs](https://docs.agentx.so/self-host#prompt-registry).

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

#### Google Gemini SDK

```python
# Use google-genai, NOT the deprecated google-generativeai
from google import genai
from google.genai import types

gclient = genai.Client()  # reads GOOGLE_API_KEY

def gemini_agent(case):
    resp = gclient.models.generate_content(
        model="gemini-2.5-flash",
        contents=case.query,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful support agent.",
        ),
    )
    return {
        "output": resp.text,
        "input_tokens": resp.usage_metadata.prompt_token_count,
        "output_tokens": resp.usage_metadata.candidates_token_count,
        "metadata": {"framework": "google", "model": "gemini-2.5-flash"},
    }

report = (
    client.evaluations
    .run(dataset_id="...", subject={"kind": "custom_agent", "displayName": "Gemini Flash", "framework": "google"})
    .execute(gemini_agent)
    .finalize()
    .analyze()
)
```

Full example: [`examples/evaluations/google_eval.py`](examples/evaluations/google_eval.py)

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
| `framework` | `raw_python`, `openai`, `anthropic`, `google`, `langchain`, `llamaindex`, `crewai`, `autogen`, `n8n`, `flowise`, `other` | Framework used |
| `runtime` | `local`, `ci`, `customer_hosted`, `low_code` | Where the agent runs |
| `version` | any string | Optional version tag for the agent |
| `endpoint` | URL | Optional, for HTTP-based agents |

### Similarity metrics (optional)

Each scored result can be enriched with reference-based similarity scores comparing your agent's response to the `expected_results` of the case:

| Metric | What it measures | Cost |
|---|---|---|
| **Cosine** (vector similarity) | Cosine of OpenAI embeddings of `expected_results` vs the actual response. Captures semantic similarity. | One embedding API call per case. |
| **Jaccard** | Token-set overlap `|A ∩ B| / |A ∪ B|` over lowercased word tokens. Pure lexical match. | Free — no API calls. |
| **BLEU** | Sentence-level BLEU-4 (n-gram precision, up to 4-grams, with brevity penalty). Standard machine-translation-style metric — rewards responses that reuse the expected result's exact phrasing. | Free — no API calls. |
| **ROUGE-L** | F1 over the longest common (in-order) subsequence of tokens. Standard summarization-style metric — more tolerant of reordering/insertions than BLEU. | Free — no API calls. |

All four metrics are returned in the range `[0, 1]` and averaged across all scored results in the report. BLEU and ROUGE-L are computed server-side in the same way as Jaccard (no external API call, pure token-based math) — they're a good default choice when you want a similarity signal without embedding cost.

**Enable them on the dataset** (via the AgentX dashboard or the dataset API):

```jsonc
{
  "vectorSimilarity":  { "enabled": true, "model": "text-embedding-3-small" },
  "jaccardSimilarity": { "enabled": true },
  "bleuScore":         { "enabled": true },
  "rougeScore":        { "enabled": true }
}
```

Or via `DatasetBuilder` / `client.evaluations.datasets.builder(...)`:

```python
builder = client.evaluations.datasets.builder(
    name="support-agent-eval",
    jaccard_similarity=True,
    bleu_score=True,
    rouge_score=True,
)
```

**Read them on the report:**

```python
report = client.evaluations.run(...).execute(my_agent).finalize().analyze()

# Top-level convenience accessors — return None when the metric was not
# enabled on the dataset, or no case has a value yet.
report.average_rating       # float | None  — same as report.statistics.average_rating
report.cosine_similarity    # float | None  — averaged across cases (0–1)
report.jaccard_similarity   # float | None  — averaged across cases (0–1)
report.bleu_score           # float | None  — averaged across cases (0–1)
report.rouge_score          # float | None  — averaged across cases (0–1), ROUGE-L F1

# Same values are also available nested under the statistics block:
report.statistics.cosine_similarity
report.statistics.jaccard_similarity
report.statistics.bleu_score
report.statistics.rouge_score
```

Cases where `expected_results` is empty or the agent returned an error are skipped from the average, so a sparse dataset still produces a meaningful score. If a toggle wasn't on for the dataset, that property returns `None`.

These four metrics also appear per-model (as `average_bleu_score`/`average_rouge_score` alongside `average_vector_similarity`/`average_jaccard_similarity`) when a dataset selects multiple comparison models, in each model's row of `report.sovereignty_index.models`.

### AI analysis report

`.analyze()` is the last step in the chain. It runs the same durable, multi-stage pipeline as the dashboard's "Analyze" button: each response is scored by 1-3 LLM judges, then reduced through question- and cluster-level summaries into one final qualitative report, returned as the `Report` object. Because of this, `.analyze()` polls until the job finishes rather than returning instantly, and can take noticeably longer than a single LLM call for larger runs (progress is shown in the terminal while it waits).

```python
report = client.evaluations.run(...).execute(my_agent).finalize().analyze(
    mode="auto",                                    # "auto" (default) | "sync" | "batch"
    quality_mode="quality_first",                   # "quality_first" (default) | "balanced"
    judges=["gpt-5.5", "claude-opus-4-8"],           # 1-3 model ids; omit for a single gpt-5.5 judge
)

report.summary                 # str | None, overall narrative summary
report.consistency_score       # float | None, 0-10, run-to-run consistency
report.instruction_adherence   # ReportInstructionAdherence | None
report.response_patterns       # ReportResponsePatterns | None
report.reasoning_analysis      # ReportReasoningAnalysis | None
report.tool_usage_analysis     # ReportToolUsageAnalysis | None
report.strengths               # list[str]
report.weaknesses              # list[str]
report.overall_rating          # str | None, "high" | "medium" | "low"
report.recommendations         # list[ReportRecommendation]
```

| Field | Shape | Description |
|---|---|---|
| `instruction_adherence` | `{ score, analysis, deviations: [str], rating }` | How well responses followed `subject.agentInstructions` (see [EvaluationSubject fields](#evaluationsubject-fields)) |
| `response_patterns` | `{ similarities: [str], differences: [str], outliers: [str], rating }` | Cross-run consistency patterns |
| `reasoning_analysis` | `{ cot_quality, reasoning_patterns: [str], reasoning_gaps: [str], rating }` | Quality of the agent's chain-of-thought, when traced |
| `tool_usage_analysis` | `{ effectiveness, patterns: [str], issues: [str], rating }` | How well the agent used its tools, when tool calls were traced |
| `recommendations` | `[{ category, priority, recommendation, reasoning }]` | Actionable, prioritized fixes |

This is separate from, and available even without, the numeric `average_rating`/similarity scores, which are ready right after `.finalize()`, before `.analyze()` runs.

#### Controlling the analysis

| Parameter | Values | Description |
|---|---|---|
| `mode` | `"auto"` (default), `"sync"`, `"batch"` | How item scoring executes server-side; `"auto"` picks based on run size |
| `quality_mode` | `"quality_first"`, `"balanced"` | `"quality_first"` runs a second judge on every item; `"balanced"` samples based on risk |
| `judges` | 1-3 model ids | Which LLM(s) score each response. The first always runs; a second confirms, a third only breaks a tie between the first two. Defaults to a single judge, `["gpt-5.5"]`, if omitted. |
| `poll_interval` | seconds, default `5.0` | How often to check job status while waiting |
| `timeout` | seconds, default `1800.0` | Give up waiting after this long (the job keeps running server-side; call `get_report()` later to check on it) |

```python
# Check on a long-running analysis without calling .analyze() again, even from a
# separate script execution:
status = client.evaluations.get_analysis_status(run_id)
print(status.status, status.progress.overall_percentage)
```

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
