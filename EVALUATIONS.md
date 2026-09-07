## Custom Agent Evaluations - Installation

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

Evaluate **any AI agent** - LangChain, CrewAI, AutoGen, LlamaIndex, OpenAI, Anthropic, HTTP endpoints, or plain Python - using AgentX as a scoring and reporting backend. Your agent runs locally; AgentX scores results and generates a full analysis report.

### How it works

1. **Build a dataset** - define cases (queries + acceptance/rejection criteria).
2. **Run your agent** - the SDK calls your function or endpoint for each case.
3. **Finalize + analyze** - AgentX scores every response and generates a report.
4. **View results** - in the terminal and on the AgentX dashboard.

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

# Optional similarity metrics - present only when enabled on the dataset.
if report.cosine_similarity is not None:
    print(f"Cosine similarity: {report.cosine_similarity:.3f}")   # 0-1
if report.jaccard_similarity is not None:
    print(f"Jaccard similarity:{report.jaccard_similarity:.3f}")  # 0-1
if report.bleu_score is not None:
    print(f"BLEU score:        {report.bleu_score:.3f}")          # 0-1
if report.rouge_score is not None:
    print(f"ROUGE-L score:     {report.rouge_score:.3f}")         # 0-1

if report.dashboard_url:   # hosted runs; self-host doesn't send one
    print(f"Dashboard: {report.dashboard_url}")
```

### Environment variables

| Variable | Description |
|---|---|
| `AGENTX_API_KEY` | Required. Your AgentX API key. |
| `AGENTX_API_BASE_URL` | Optional. Override the API base URL, e.g. `http://localhost:4700/api/v1` for self-host. |
| `AGENTX_WORKSPACE_ID` | Optional. Explicit workspace instead of the API key's default. |
| `AGENTX_EVAL_QUIET` | Optional. `1` silences the interactive progress UI (spinners, per-case lines) for CI logs; results, gate verdicts, and errors still print. |

You can also pass `base_url` directly to the constructor (the `/custom-agent-evaluations` suffix is appended automatically, so `/api/v1` is enough):

```python
# Point at a local self-host engine
client = AgentX(api_key="your-key", base_url="http://localhost:4700/api/v1")
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

`from_csv()` returns a builder (add more cases if you like), so finish with `.publish()`:

```python
dataset = client.evaluations.datasets.from_csv(
    path="cases.csv",
    name="My Dataset",
    number_of_requests=2,
    acceptance_criteria="...",
    rejection_criteria="...",
).publish()
```

CSV format:
```
query,expected_results
"How do I reset my password?","Explain the steps clearly."
"What is your refund policy?","Describe refund terms."
```

`query` is the only required column. Optional columns: `expected_results`, plus semicolon-separated `expected_capabilities`, `expected_knowledge_base`, and `expected_delegations`. Rows with an empty query are skipped with a logged warning. `from_dataframe(df, name, ...)` does the same from a pandas DataFrame.

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

`smoke_test_count` accepts 1-10 extra variants per case. `smoke_test_guidance` is optional free text steering *what kind* of variants get generated (tone, adversarial phrasing, different languages, ...); leave it out for natural rewording only. Both the paraphrase text and the count are decided entirely server-side, reusing the same generation the AgentX dashboard's native runs use, `.execute()` just asks the extra variants and submits them for you, nothing to configure on the SDK side beyond these two kwargs. Ignored on `follow_up_questions`, only a case's opening question can be smoke-tested.

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

### Agent trajectory and retrieval-context checks (deterministic, no judge spend)

Two per-case expectations are scored server-side against the case's **linked trace** (return
`{"output": ..., "trace_id": span.trace_id}` from your agent function, with the call wrapped in
`client.tracer.trace(..., sync=True)`):

```python
client.evaluations.datasets.builder(name="Agent checks").add_case(
    query="Refund order 4412 and email a confirmation.",
    expected_results="Refund issued and confirmation sent.",
    # The tool calls a correct run should make, matched against the trace's REAL calls.
    # Modes (agentevals semantics): strict | unordered | superset | subset.
    expected_tools=["lookup_order", "issue_refund", "send_email"],
    trajectory_match_mode="strict",
    # What a correct retriever should have fetched - compared to the actual retrieved context
    # with token Jaccard. Catches retriever regressions even when the answer text is identical.
    expected_retrieval_context=["Refunds: 30 days, no restocking fee."],
)
```

Each produces a scorer row on the result (`Trajectory match (<mode>)` pass/fail, and
`Context match (jaccard)` 0-1). Both are deterministic: no LLM judge call, no spend, and they
run on every result that carries the needed evidence.

---

### Dataset splits (cheap PR runs vs. nightly full runs)

Tag cases with named subsets and run just one subset:

```python
builder.add_case(query="smoke case", splits=["smoke"])
builder.add_case(query="full-only case")

client.evaluations.run(dataset_id, subject, split="smoke").execute(my_fn).finalize()
```

Original case indexes are preserved, so per-case comparisons line up between a split run and a
full run. The connector-driven dashboard run accepts the same `split`.

---

### Concurrency and output reuse

```python
run.execute(my_fn, concurrency=4)                    # thread-pooled agent calls, ordered submission
run.execute(my_fn, reuse_outputs_from="run_abc123")  # replay a previous run's outputs
```

`reuse_outputs_from` replays the recorded output for every case whose query text is unchanged
(errored rows and changed/new cases run normally) and the judge re-scores everything with THIS
run's grading config - which makes iterating on scorers essentially free. Replayed results are
submitted with `metadata.reusedFromRun` set to the source run id (submission-side metadata; the
stored row keeps the replayed output itself).

Interrupted runs resume: `execute()` asks the engine which idempotency keys were already
accepted and skips those cases, so a crash or a failed batch (which now raises
`EvaluationSubmissionError` instead of finishing silently empty) never re-pays for finished
work - just call `execute()` again on the same context.

---

### Human review queue (label-and-calibrate from code)

```python
item = client.monitor.review_queue.queue(trace_id, note="spot-check this")
for item in client.monitor.review_queue.list(status="pending"):
    client.monitor.review_queue.label(item.id, "bad", corrected_score=2, note="hallucinated")
```

Labels (with the judge's own score for the same trace) feed per-scorer calibration
(`client.monitor.judge_scorers.calibration(scorer_id)`) and become judge-tuning evidence.
Project-level numbers come from `client.monitor.calibration(window="7d")`, whose exact wire
keys are `comparedCount`, `agreementRate`, `falsePositiveRate`, `falseNegativeRate`.

---

### LLM Judge Scorers - reusable grading configs

By default, a dataset runs against the grading config it was created with (`number_of_requests`, `acceptance_criteria`, similarity metrics, etc. - see above). If you want to grade the **same dataset** against **different configs** (e.g. a strict config vs. a lenient one, or reuse one config across many datasets), create a standalone **LLM Judge Scorer** and pass its id to `.run()`.

One scorer is one entity: a judge rubric plus two setting profiles.

| Section | What it holds |
|---|---|
| `judge` | The rubric every surface grades with - acceptance / rejection / evaluation criteria, judge prompt, judge model |
| `offline` | How dataset runs grade with it - repetitions, similarity metrics, code scorers, default flag |
| `online` | Whether it *also* scores live production traffic - enabled, sample rate, scope, alert threshold. `None` means offline-only |

```python
strict = (
    client.monitor.judge_scorers
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
    .run(dataset_id=dataset.id, subject={...}, scorer_id=strict.id)
    .execute(my_agent)
    .finalize()
    .analyze()
)
```

**The scorer's id is the `scorer_id` a run takes** - there is no second id to keep track of. Omit `scorer_id` to keep using the dataset's own config, exactly as before - this is fully additive, no existing code needs to change. The builder accepts the same config kwargs as `datasets.builder(...)` (`number_of_requests`, the three criteria fields, `vector_similarity`/`jaccard_similarity`/`bleu_score`/`rouge_score`, `sovereignty_models`, `judge_prompt`/`judge_model` below) but no `questions` - it's config-only and reusable, plus `thresholds`, `tool_context`, `code_scorers` and the online profile below.

```python
client.monitor.judge_scorers.get(strict.id)      # -> JudgeScorer
client.monitor.judge_scorers.list()              # -> list[JudgeScorer]
client.monitor.judge_scorers.update(strict.id, judge={"acceptanceCriteria": "..."})
client.monitor.judge_scorers.delete(strict.id)   # rubric, version history and online profile together
```

`update()` is sparse - only the sections you pass change. `online={...}` upserts the online profile (this is how an offline-only scorer goes live), and `online=None` detaches it. Section dicts use the wire's camelCase keys; `create(name, judge=..., offline=..., online=...)` takes the same three sections directly if you would rather not go through the builder. Deleting, and detaching the online profile, are both refused for the built-in Session Baseline Judge.

A `JudgeScorer` is a `dict` subclass, so unknown fields round-trip untouched, with `.id`, `.name`, `.judge`, `.offline`, `.online` and `.online_profile_id` as conveniences.

#### Naming: `scorer_id` and `evaluation_settings_id`

`scorer_id` is the current name for the kwarg naming a run's grader. `evaluation_settings_id` is the pre-consolidation spelling of **the same id** - the wire still calls the field `evaluationSettingsId` - and it keeps working on `client.evaluations.run(...)` and `init_run(...)`. Passing both with *different* values raises `ValueError`.

Which to write depends on the SDK versions your code has to run under:

| Kwarg | `agentx-python` < 0.6.36 | >= 0.6.36 |
|---|---|---|
| `evaluation_settings_id` | works | works |
| `scorer_id` | `TypeError` | works |

So prefer `scorer_id` in new code, and keep `evaluation_settings_id` where a script may be executed against a pinned older client - CI gates and committed evaluation harnesses that get re-run for a controlled before-and-after comparison are the usual cases.

#### Scoring live traffic with the same scorer

Pass `live=True` to give the scorer an online profile at creation, and the same rubric that grades your dataset runs also scores a sample of production traffic. See [`client.monitor.online_evaluators`](TRACING.md#clientmonitoronline_evaluators-self-host-only) for what the online profile does and what its fields mean.

```python
scorer = client.monitor.judge_scorers.builder(
    name="Support quality",
    acceptance_criteria="Concrete, correct, cites the policy.",
    live=True,             # every check is a real judge call on your own provider key
    sample_rate=0.2,
    alert_threshold=6,
).publish()
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `live` | `bool` | `False` | Create the online profile and start scoring live traffic |
| `sample_rate` | `float` | `0.1` | Fraction of traffic actually scored |
| `scope` | `str` | `"trace"` | `"trace"` scores individual traces at ingest; `"session"` scores whole conversations |
| `alert_threshold` | `float \| None` | `5` | A score below this raises a signal. `None` scores without ever raising one |
| `severity` | `str` | `"medium"` | `"low"`, `"medium"`, `"high"` or `"critical"`, applied to signals it raises |
| `agent_ids` | `list[str]` | `None` | Restrict scoring to specific agents instead of the whole workspace |
| `idle_seconds` | `int` | `120` | For `scope="session"`: how long a session must be quiet before it is judged |

Calibration, tuning and the live-scoring history hang off the same scorer id - the SDK resolves the online profile for you:

```python
client.monitor.judge_scorers.calibration(scorer.id, window="7d")   # verdicts vs. recorded ground truth
proposal = client.monitor.judge_scorers.tune(scorer.id)            # LLM call, slow
verdict = client.monitor.judge_scorers.validate_tuning(scorer.id, proposal)  # re-judge with candidate criteria
client.monitor.judge_scorers.publish_tuning(scorer.id, proposal, validation=verdict)  # write it onto the rubric
client.monitor.judge_scorers.ratings(scorer.id, window="7d")       # -> list[OnlineEvaluatorRatingPoint]
client.monitor.judge_scorers.events(scorer.id, window="7d")        # -> list[OnlineEvaluatorEvent]
```

Those six cover live-traffic scoring, so calling them on an offline-only scorer raises `AgentXJudgeScorersError` naming the fix (`update(scorer_id, online={"enabled": True})`). `publish_tuning` writes to the shared rubric, so it applies everywhere the scorer is used: online scoring, offline dataset runs and the playground alike. Publish is provenance-gated: the engine refuses an unvalidated publish, and a measured regression, unless you pass `force=True`; the validation verdict is stamped into the rubric's version history.

#### Engine compatibility (self-host)

`client.monitor.judge_scorers` calls `/agent-monitoring/judge-scorers`, which needs a self-host engine build that serves it. **Older engines return 404 on that route** while the rest of `/agent-monitoring/` works normally, and the SDK surfaces that as `AgentXJudgeScorersError: ... (404)`. Check before building on it:

```python
try:
    client.monitor.judge_scorers.list()
except Exception as exc:
    print("unified surface unavailable on this engine:", exc)
    # the legacy views below work on every engine
```

The legacy views are not affected - they call the long-standing `/custom-agent-evaluations/evaluation-settings` and `/agent-monitoring/online-evaluators` routes - so they remain the portable choice for code that must run against engines you do not control. This does not affect the `scorer_id` kwarg on `.run()`, which is client-side naming over a field the wire has always had.

#### Legacy views

Before the consolidation the same entity was reached through two half-views, and both keep working:

| Legacy | Covers | Successor |
|---|---|---|
| `client.evaluations.settings` | the offline profile | `client.monitor.judge_scorers` |
| `client.monitor.online_evaluators` | the online profile | `client.monitor.judge_scorers` |

Both emit a `DeprecationWarning` on first use (hidden by default; visible under `-W` or pytest) pointing at `judge_scorers`. Nothing breaks, and by design they address the same records under the same ids - an evaluation-settings id, an online-evaluator's `evaluation_settings_id` and a `scorer_id` are all the one id. They keep their pre-consolidation kwarg names deliberately: renaming compatibility surfaces would defeat their purpose.

```python
settings = client.evaluations.settings.builder(name="Strict grading", ...).publish()
client.evaluations.run(dataset_id=dataset.id, subject={...}, scorer_id=settings.id)
```

#### Configuring the judge

`datasets.builder(...)`, `judge_scorers.builder(...)` and the legacy `settings.builder(...)` all accept `judge_prompt`/`judge_model` to override how the LLM-as-judge grades responses, applying to every scoring path (native dashboard runs and SDK/custom-agent runs alike):

```python
scorer = (
    client.monitor.judge_scorers
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
- `judge_model` accepts any OpenAI or Anthropic model id (`client.evaluations.list_models(provider="Anthropic")` to discover valid ones). Omit it to keep the engine default (`gpt-5.6-luna`).
- `list_models()` is **hosted platform only**: it calls the hosted API's `/custom-agent-evaluations/models` registry, which the self-host engine does not serve (404, surfaced as `AgentXEvaluationsError`). On self-host, pass any model id your engine's judge keys can reach.

---

### Prompt registry

**Self-host only** (see [Self-host](README.md#self-host)) - no hosted-SaaS equivalent yet.

AgentX doesn't own your agent's code, so it can't do what native Autotune does - branch and merge a config directly. `client.evaluations.prompts` solves the same "how do I close the loop" problem the way LangSmith's Prompt Hub and Langfuse's Prompt Management do instead: become the prompt's *source of truth*. Your agent pulls a version at runtime, you tag your evaluation evidence with which version it used, and "improvement" becomes propose → a human approves → publish a new version - never a direct edit to your deployed code.

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
to learn from by setting `metadata.promptName` - on a deliberate eval run:

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

From the self-host dashboard: Governance > Manage > **Prompts** > a prompt's row menu > **Suggest
improvement**. It merges both kinds of evidence - deliberate eval runs (defaulting to the *current
published version only*, auto-widening to every version if there isn't enough recent evidence yet)
and worst-scoring Online Evaluator ratings from a recent time window - feeds the worst-rated
examples to a judge, and shows a full rewrite plus reasoning. **Nothing is saved until a human
approves it as a new version.** The same propose loop is scriptable: `prompts.examples(prompt.id)`
returns the evidence, `prompts.propose(prompt.id)` asks the judge for a rewrite (returns
`revisedText`/`reasoning` without saving anything), and `prompts.publish_version(prompt.id,
text=...)` is the explicit human-approval write. Your agent's next
`client.evaluations.prompts.get(name)` call picks up the new version immediately. Tagging
`metadata.version` as `<promptName>@v<N>` (shown above) means the dataset's **Compare versions**
dialog also tells you whether the published rewrite actually scored better, no separate comparison
view needed. Full details, including the judge-key-free `improve-prompt` Claude Code skill: see
[self-host's Prompt registry docs](https://docs.agentx.so/improve/prompt-management).

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

Use `HttpEndpointAdapter` to evaluate any agent exposed as an HTTP service - FastAPI, LangServe, Flask, n8n webhooks, etc.

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
POST /your-endpoint            (method="..." on the adapter overrides the verb)
Content-Type: application/json
Body:     { "query": "...", "case_id": "...", "question_index": 0, "run_number": 1 }
Response: { "output": "..." }  (or "text"; optional "metadata" and "trace")
```

A non-2xx response or a timeout is recorded as that case's error instead of aborting the run.

Full example: [`examples/evaluations/http_endpoint_eval.py`](examples/evaluations/http_endpoint_eval.py)

---

#### Precomputed results (n8n, Flowise, batch jobs)

Submit outputs you already have without running any agent during evaluation:

```python
from agentx.evaluations.adapters.precomputed import PrecomputedAdapter

outputs = {
    0: "To reset your password, go to Login > Forgot Password.",
    1: "We accept Visa, Mastercard, PayPal, and bank transfers.",
}

adapter = PrecomputedAdapter(outputs)
```

Keys are matched against the case's `case_id` first, then its question index; a plain list works
too (one entry per question, in order). Values can be strings or `{"output": ..., "metadata":
...}` dicts:

```python
adapter = PrecomputedAdapter(["First answer.", "Second answer."])

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
| `kind` | `custom_agent` (default), `agentx_agent`, `agentx_team` | `custom_agent` for external agents |
| `displayName` | any string | Human-readable name shown in the dashboard |
| `framework` | `raw_python`, `openai`, `anthropic`, `google`, `langchain`, `llamaindex`, `crewai`, `autogen`, `n8n`, `flowise`, `other` | Framework used |
| `frameworkVersion` | any string | Optional framework version tag |
| `runtime` | `local` (default), `ci`, `customer_hosted`, `low_code` | Where the agent runs |
| `agentInstructions` | any string | The agent's own system instructions - what the report's instruction-adherence section grades against |
| `metadata` | `dict[str, str \| int \| bool]` | Free-form tags, e.g. `{"promptName": ..., "version": ...}` (see [Prompt registry](#prompt-registry)) |

Unknown fields are silently dropped, so a typo here fails quietly - stick to the fields above.

### Similarity metrics (optional)

Each scored result can be enriched with reference-based similarity scores comparing your agent's response to the `expected_results` of the case:

| Metric | What it measures | Cost |
|---|---|---|
| **Cosine** (vector similarity) | Cosine of OpenAI embeddings of `expected_results` vs the actual response. Captures semantic similarity. | One embedding API call per case. |
| **Jaccard** | Token-set overlap `\|A ∩ B\| / \|A ∪ B\|` over lowercased word tokens. Pure lexical match. | Free - no API calls. |
| **BLEU** | Sentence-level BLEU-4 (n-gram precision, up to 4-grams, with brevity penalty). Standard machine-translation-style metric - rewards responses that reuse the expected result's exact phrasing. | Free - no API calls. |
| **ROUGE-L** | F1 over the longest common (in-order) subsequence of tokens. Standard summarization-style metric - more tolerant of reordering/insertions than BLEU. | Free - no API calls. |

All four metrics are returned in the range `[0, 1]` and averaged across all scored results in the report. BLEU and ROUGE-L are computed server-side in the same way as Jaccard (no external API call, pure token-based math) - they're a good default choice when you want a similarity signal without embedding cost.

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

# Top-level convenience accessors - return None when the metric was not
# enabled on the dataset, or no case has a value yet.
report.average_rating       # float | None  - same as report.statistics.average_rating
report.cosine_similarity    # float | None  - averaged across cases (0-1)
report.jaccard_similarity   # float | None  - averaged across cases (0-1)
report.bleu_score           # float | None  - averaged across cases (0-1)
report.rouge_score          # float | None  - averaged across cases (0-1), ROUGE-L F1

# Same values are also available nested under the statistics block:
report.statistics.cosine_similarity
report.statistics.jaccard_similarity
report.statistics.bleu_score
report.statistics.rouge_score
```

Cases where `expected_results` is empty or the agent returned an error are skipped from the average, so a sparse dataset still produces a meaningful score. If a toggle wasn't on for the dataset, that property returns `None`.

These four metrics also appear per-model (as `average_bleu_score`/`average_rouge_score` alongside `average_vector_similarity`/`average_jaccard_similarity`) when a dataset selects multiple comparison models, in each model's row of `report.sovereignty_index.models`.

### CI gate (self-host)

A finalized run can pass/fail a CI job. `run.gate(...)` (on the run context `.execute()` returns - not on the `Report` model from `get_report()`) checks the run's average rating, prints per-check verdicts into the CI log, and returns a `GateResult` - the caller decides the exit code:

```python
import sys

run = (
    client.evaluations
    .run(dataset_id="evds_…", subject={"kind": "custom_agent", "framework": "raw_python"})
    .execute(my_agent)   # in CI, this is the PR's version of your agent
    .finalize()
)

gate = run.gate(fail_under=7, no_regression=True, caller="github-actions")
sys.exit(gate.exit_code)   # 0 = merge, 1 = block
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fail_under` | `float` | none | Fail when the run's average rating is below this floor. Works from the very first run |
| `no_regression` | `bool` | `False` | Fail when the average dropped more than `tolerance` below the dataset's previous completed run. Needs run history, so point CI at a persistent self-host instance |
| `tolerance` | `float` | `0.5` | Slack for `no_regression` - judge scores are noisy, and an exact comparison would flake builds on variance rather than regressions |
| `caller` | `str` | `"sdk"` | Free label shown in the dashboard's CI Gates history ("github-actions", ...) |

At least one of `fail_under` / `no_regression` is required. `GateResult` exposes `.passed`, `.exit_code` (0/1), `.average_rating`, `.baseline_average`, `.baseline_run_id`, and `.checks` (the per-check verdict list). Every `gate()` call is recorded into the dashboard's CI Gates tab by default; use the lower-level `client.evaluations.gate_run(run_id, ..., record=False)` for an unrecorded check, or to gate a run created elsewhere by id. Set `AGENTX_EVAL_QUIET=1` in CI to silence the interactive progress UI while keeping results and gate verdicts. See [self-host's CI docs](https://docs.agentx.so/integrations/self-host-ci) for the GitHub Actions recipe.

This run + gate flow is the **self-host CI path**. The separate CI-runs API in [CICD_EVAL.md](CICD_EVAL.md) (`tracer.run_eval()` and friends) targets the hosted platform only.

### AI analysis report

`.analyze()` is the last step in the chain. It runs the same durable, multi-stage pipeline as the dashboard's "Analyze" button: each response is scored by 1-3 LLM judges, then reduced through question- and cluster-level summaries into one final qualitative report, returned as the `Report` object. Because of this, `.analyze()` polls until the job finishes rather than returning instantly, and can take noticeably longer than a single LLM call for larger runs (progress is shown in the terminal while it waits).

```python
report = client.evaluations.run(...).execute(my_agent).finalize().analyze(
    mode="auto",                                    # "auto" | "sync" | "batch"
    quality_mode="quality_first",                   # "quality_first" | "balanced"
    judges=["gpt-5.6-luna", "claude-opus-4-8"],      # 1-3 model ids; omit for the platform default judge
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
| `judges` | 1-3 model ids | Which LLM(s) score each response. The first always runs; a second confirms, a third only breaks a tie between the first two. Omit to score with a single judge, the engine's platform default model. |
| `poll_interval` | seconds, default `5.0` | How often to check job status while waiting |
| `timeout` | seconds, default `1800.0` | Give up waiting after this long (the job keeps running server-side; call `get_report()` later to check on it) |

`mode` and `quality_mode` default server-side (omitting them keeps the server's behavior). On self-host, the analysis runs synchronously on the engine: `judges` is honored, `mode`/`quality_mode` are accepted but ignored, and the polling loop sees a terminal status on its first check.

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
| `dict` | Recognized keys: `output` (or `text`/`response`), `metadata` (a dict), `trace_id`, `input_tokens`/`output_tokens` (also read from metadata), `retrieval_context`, `trace`, `error`. Unrecognized keys are dropped - put extra data under `metadata` |
| `EvaluationResult` | Full control - pass error, trace, timings, metadata, retrieval context |

### What gets uploaded

The SDK uploads exactly what your callable returns: the output text, any metadata you include,
and (when tracing is enabled) the trace you instrumented. **The SDK does not scrub or redact
anything** - if your agent's output or metadata can contain secrets, redact them in your own
code before returning, or keep them out of the returned payload entirely.
