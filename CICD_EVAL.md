# Agent CI/CD Evaluation - Python SDK

## Overview

The AgentX CI/CD evaluation SDK lets you **gate agent releases** in any CI pipeline. You define test cases in an AgentX dataset, run your agent against each case during CI, submit the results to AgentX for scoring, and receive a `"pass"` or `"fail"` gate decision.

If the gate fails, the SDK raises `CIGateFailure` (or returns the result so you can `sys.exit(1)`), blocking the pipeline.

> **Status:** This feature is in the design phase. The API and SDK interfaces described here reflect the planned implementation.

---

## Installation

```bash
pip install agentx-python
```

---

## Quick start

```python
# ci_eval.py
import os, sys
from agentx import AgentX

client = AgentX(api_key=os.environ["AGENTX_API_KEY"])

def my_agent(query: str) -> str:
    """Replace with your actual agent call."""
    return call_my_llm_agent(query)

result = client.tracer.run_eval(
    dataset_id=os.environ["AGENTX_DATASET_ID"],
    agent_fn=my_agent,
    agent_name="customer-support-agent",
)

print(f"Gate: {result.gate}  ({result.passed_questions}/{result.total_questions} passed)")
sys.exit(0 if result.gate == "pass" else 1)
```

---

## `tracer.run_eval()` - high-level helper

Runs the full CI/CD evaluation lifecycle in one call:
1. Creates a CI run and fetches test cases from the dataset
2. Calls `agent_fn(query)` for each test case
3. Submits each result to AgentX for scoring
4. Finalizes the run and returns the gate decision

```python
result = tracer.run_eval(
    dataset_id: str,
    agent_fn: Callable[[str], str],
    *,
    agent_name: str | None = None,
    pass_rate_threshold: float | None = None,
    git_context: dict | None = None,
    concurrency: int = 1,
    fail_on_gate: bool = False,
    timeout_per_question: float | None = None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset_id` | `str` | - | ID of the evaluation dataset (must have `ci.enabled: true`) |
| `agent_fn` | `Callable[[str], str]` | - | Function that takes a query string and returns the agent's response |
| `agent_name` | `str` | `None` | Label for this agent (used to create/reuse a reference agent in AgentX) |
| `pass_rate_threshold` | `float` | `None` | Override the dataset's `passRateThreshold` for this run (0.0–1.0) |
| `git_context` | `dict` | `None` | Git metadata (see [Git context](#git-context)) |
| `concurrency` | `int` | `1` | Number of questions to run in parallel (use with caution for rate-limited agents) |
| `fail_on_gate` | `bool` | `False` | Raise `CIGateFailure` if gate is `"fail"` instead of returning the result |
| `timeout_per_question` | `float` | `None` | Seconds to wait for `agent_fn` per question before treating it as an error |

### Return type: `CIRunResult`

```python
@dataclass
class CIRunResult:
    run_id: str
    gate: Literal["pass", "fail"]
    pass_rate: float                    # 0.0–1.0
    total_questions: int
    passed_questions: int
    scores: list[CIQuestionScore]
    violations: list[ThresholdViolation]
    git_context: dict | None
    finalized_at: str                   # ISO 8601
```

```python
@dataclass
class CIQuestionScore:
    question_index: int
    rating: int            # 1–5
    justification: str
    passed: bool
    input: Any | None
    output: Any | None
```

```python
@dataclass
class ThresholdViolation:
    question_index: int
    metric: str            # "rating" | "vectorSimilarity" | "jaccardSimilarity"
    threshold: float
    actual: float
    question_text: str
```

### Example with full output

```python
result = tracer.run_eval(
    dataset_id="6876ddd222bbb333ccc444ee",
    agent_fn=my_agent,
    agent_name="customer-support-v2",
    git_context={
        "branch": "feat/new-retrieval",
        "commit_sha": "a1b2c3d",
        "pr_number": 42,
    },
)

print(f"\n{'='*50}")
print(f"Gate result : {result.gate.upper()}")
print(f"Pass rate   : {result.pass_rate:.0%}  ({result.passed_questions}/{result.total_questions})")
print(f"Run ID      : {result.run_id}")
print(f"{'='*50}\n")

for score in result.scores:
    icon = "✓" if score.passed else "✗"
    print(f"  [{icon}] Q{score.question_index}  rating={score.rating}/5")
    print(f"       {score.justification[:100]}")

if result.violations:
    print("\nThreshold violations:")
    for v in result.violations:
        print(f"  Q{v.question_index}: {v.metric} = {v.actual:.2f} (threshold: {v.threshold:.2f})")
        print(f"    \"{v.question_text}\"")
```

---

## Low-level API

Use the low-level methods when you need more control: custom input formatting, async execution, streaming, or multi-step agent pipelines.

### `tracer.create_ci_run()`

Create a CI run and receive the test cases.

```python
run = tracer.create_ci_run(
    dataset_id: str,
    *,
    agent_name: str | None = None,
    pass_rate_threshold: float | None = None,
    git_context: dict | None = None,
    workspace_id: str | None = None,
)
```

**Returns: `CIRun`**

```python
@dataclass
class CIRun:
    run_id: str
    dataset_id: str
    total_questions: int
    test_cases: list[CITestCase]    # empty if ci.exposeTestInputs is false
    expires_at: str                 # ISO 8601
```

```python
@dataclass
class CITestCase:
    index: int
    query: str | None    # None when ci.exposeTestInputs is false
```

**Example:**

```python
run = tracer.create_ci_run(
    dataset_id="6876ddd222bbb333ccc444ee",
    agent_name="my-agent",
    git_context={"branch": "main", "commit_sha": "abc123"},
)

print(f"Created run {run.run_id} with {run.total_questions} questions")
for case in run.test_cases:
    print(f"  [{case.index}] {case.query}")
```

---

### `tracer.submit_result()`

Submit the agent's output for one test case. Can be called concurrently. AgentX scores the result immediately.

```python
score = tracer.submit_result(
    run_id: str,
    question_index: int,
    output: Any,
    *,
    input: Any | None = None,      # Actual input sent to agent (defaults to test case query)
    latency_ms: int | None = None,
)
```

**Returns: `CIQuestionScore`**

```python
@dataclass
class CIQuestionScore:
    question_index: int
    rating: int
    justification: str
    passed: bool
    gate_fired: bool   # True if failFast triggered early finalization
```

**Example:**

```python
for case in run.test_cases:
    output = my_agent(case.query)
    score = tracer.submit_result(
        run_id=run.run_id,
        question_index=case.index,
        output=output,
        input=case.query,
    )
    print(f"Q{case.index}: {score.rating}/5 - {score.justification[:60]}")
    if score.gate_fired:
        print("Gate fired (failFast) - stopping early")
        break
```

---

### `tracer.finalize_ci_run()`

Finalize the run and get the gate result. Call after all results are submitted.

```python
result = tracer.finalize_ci_run(run_id: str)
```

**Returns: `CIRunResult`** (same as `run_eval()`)

**Example:**

```python
result = tracer.finalize_ci_run(run_id=run.run_id)
print(f"Gate: {result.gate} - {result.pass_rate:.0%} passed")
```

---

### `tracer.get_ci_run()`

Poll the status of a CI run without finalizing it.

```python
status = tracer.get_ci_run(run_id: str)
```

**Returns: `CIRunStatus`**

```python
@dataclass
class CIRunStatus:
    run_id: str
    status: Literal["in_progress", "completed", "failed"]
    gate: Literal["pass", "fail"] | None   # None while in_progress
    results_submitted: int
    total_questions: int
    created_at: str
    expires_at: str
    finalized_at: str | None
```

---

## Git context

Pass git metadata to link CI runs to specific commits and pull requests. This data appears in the AgentX CI Runs history and is included in webhook payloads.

```python
git_context = {
    "branch": "feat/new-retrieval",
    "commit_sha": "a1b2c3d4e5f6",
    "pr_number": 42,
    "repo_url": "https://github.com/your-org/your-repo",
    "triggered_by": "robin",
}
```

### Auto-populate from GitHub Actions

```python
import os

def github_git_context() -> dict:
    ref = os.environ.get("GITHUB_REF", "")
    pr_number = None
    if "/pull/" in ref:
        try:
            pr_number = int(ref.split("/pull/")[1].split("/")[0])
        except (IndexError, ValueError):
            pass
    return {
        "branch": os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME"),
        "commit_sha": os.environ.get("GITHUB_SHA"),
        "pr_number": pr_number,
        "repo_url": f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', '')}",
        "triggered_by": os.environ.get("GITHUB_ACTOR"),
    }

result = tracer.run_eval(
    dataset_id=DATASET_ID,
    agent_fn=my_agent,
    git_context=github_git_context(),
)
```

---

## Framework examples

### LangChain

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from agentx import AgentX
import os, sys

client = AgentX.from_env()
llm = ChatOpenAI(model="gpt-4o")

def langchain_agent(query: str) -> str:
    response = llm.invoke([HumanMessage(content=query)])
    return response.content

result = client.tracer.run_eval(
    dataset_id=os.environ["DATASET_ID"],
    agent_fn=langchain_agent,
    agent_name="langchain-support-agent",
)
sys.exit(0 if result.gate == "pass" else 1)
```

### CrewAI

```python
from crewai import Agent, Task, Crew
from agentx import AgentX
import os, sys

client = AgentX.from_env()

def crewai_agent(query: str) -> str:
    agent = Agent(
        role="Support Specialist",
        goal="Resolve customer queries accurately",
        backstory="Expert in product support"
    )
    task = Task(description=query, agent=agent, expected_output="A helpful response")
    crew = Crew(agents=[agent], tasks=[task])
    return str(crew.kickoff())

result = client.tracer.run_eval(
    dataset_id=os.environ["DATASET_ID"],
    agent_fn=crewai_agent,
    agent_name="crewai-support-crew",
)
sys.exit(0 if result.gate == "pass" else 1)
```

### OpenAI Agents SDK (async)

```python
import asyncio, os, sys
from agents import Agent, Runner
from agentx import AgentX

client = AgentX.from_env()
agent = Agent(name="Support", instructions="You are a helpful support agent.")

async def openai_agent_async(query: str) -> str:
    result = await Runner.run(agent, query)
    return result.final_output

def openai_agent(query: str) -> str:
    return asyncio.run(openai_agent_async(query))

result = client.tracer.run_eval(
    dataset_id=os.environ["DATASET_ID"],
    agent_fn=openai_agent,
    agent_name="openai-support-agent",
)
sys.exit(0 if result.gate == "pass" else 1)
```

### HTTP endpoint agent

Use this pattern for agents deployed as a REST API (any language/platform):

```python
import requests, os, sys
from agentx import AgentX

client = AgentX.from_env()
AGENT_URL = os.environ["AGENT_ENDPOINT_URL"]

def http_agent(query: str) -> str:
    response = requests.post(
        AGENT_URL,
        json={"query": query},
        headers={"Authorization": f"Bearer {os.environ['AGENT_TOKEN']}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["answer"]

result = client.tracer.run_eval(
    dataset_id=os.environ["DATASET_ID"],
    agent_fn=http_agent,
    agent_name="http-deployed-agent",
)
sys.exit(0 if result.gate == "pass" else 1)
```

---

## pytest assertions

`agentx.testing` turns a run into a plain pytest failure, so quality checks live in the same
suite as everything else. Both helpers raise `AssertionError` subclasses - no plugin, no
registration, works in any runner.

### `assert_evaluation` - does it clear the bar

```python
from agentx import AgentX
from agentx.testing import assert_evaluation

def test_support_agent_quality():
    client = AgentX.from_env()
    report = (
        client.evaluations
        .run(dataset_id=DATASET_ID, scorer_id=SCORER_ID, subject=SUBJECT)
        .execute(my_agent)
        .finalize()
    )
    assert_evaluation(report, min_rating=7.0, no_regression=True)
```

`min_rating` is an absolute floor; `no_regression` compares against the dataset's previous
completed run (`tolerance` defaults to 0.5, since judge scores are noisy). The check rides the
engine's CI gate, so every pytest verdict also appears in the dashboard's gate history with
`caller="pytest"` - the red test and the dashboard row are one event, not two systems drifting.

### `assert_pairwise` - is it better than what it replaces

An average clearing a floor does not mean a change helped. For that, compare the two runs
head to head and assert the comparison went the candidate's way:

```python
from agentx.testing import assert_pairwise

def test_new_prompt_beats_the_old_one():
    comparison = client.evaluations.compare_pairwise(
        candidate_run_id,
        baseline_run_id,
        both_orders=True,
    )
    assert_pairwise(
        comparison,
        must_win=True,
        max_losses=2,
        max_flip_rate=0.2,
    )
```

| Check | Fails when |
|---|---|
| `must_win` | Run A did not win more cases than run B. A tie fails - "no worse than before" is not the claim being made. |
| `max_losses` | Run A lost more individual cases than allowed, even if it won overall. Catches a change that lifts the average by improving easy cases while breaking hard ones. |
| `max_flip_rate` | Too many verdicts reversed when the answers were swapped. That is position bias, and an inconclusive comparison must not read as a pass. |

`max_flip_rate` needs `both_orders=True` to mean anything: without it there is no flip rate, and
the check is skipped rather than passing on a fabricated zero. The failure message names the
cases that lost, so the test output is actionable without opening the dashboard.

---

## Exceptions

| Exception | When raised |
|---|---|
| `CIGateFailure` | `fail_on_gate=True` and gate is `"fail"` |
| `CIRunExpired` | Run was not finalized within 2 hours |
| `DatasetNotFound` | `dataset_id` does not exist or is not accessible |
| `CINotEnabled` | Dataset exists but `ci.enabled` is `false` |
| `AgentXAuthError` | Invalid or missing API key |
| `AgentXAPIError` | Unexpected API error (includes status code and message) |

### Handling gate failure gracefully

```python
from agentx.exceptions import CIGateFailure

try:
    result = tracer.run_eval(
        dataset_id=DATASET_ID,
        agent_fn=my_agent,
        fail_on_gate=True,
    )
except CIGateFailure as e:
    print(f"Gate failed: {e.result.pass_rate:.0%} passed")
    for v in e.result.violations:
        print(f"  Violation: Q{v.question_index} {v.metric}={v.actual:.2f} < {v.threshold:.2f}")
    sys.exit(1)
```

---

## Complete GitHub Actions workflow

```yaml
name: Agent Eval Gate

on:
  pull_request:
    branches: [main, staging]
  push:
    branches: [main]

jobs:
  eval-gate:
    name: AgentX Eval Gate
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install agentx-python
          pip install -r requirements.txt

      - name: Run eval gate
        env:
          AGENTX_API_KEY: ${{ secrets.AGENTX_API_KEY }}
          AGENTX_DATASET_ID: ${{ vars.AGENTX_DATASET_ID }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python ci_eval.py

      - name: Upload eval results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval_results.json
          retention-days: 30
```

### `ci_eval.py` with JSON output

```python
import os, sys, json
from agentx import AgentX
from my_agent import build_agent

client = AgentX(api_key=os.environ["AGENTX_API_KEY"])

def github_git_context():
    ref = os.environ.get("GITHUB_REF", "")
    pr_number = None
    if "/pull/" in ref:
        try:
            pr_number = int(ref.split("/pull/")[1].split("/")[0])
        except (IndexError, ValueError):
            pass
    return {
        "branch": os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME"),
        "commit_sha": os.environ.get("GITHUB_SHA"),
        "pr_number": pr_number,
        "repo_url": f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', '')}",
        "triggered_by": os.environ.get("GITHUB_ACTOR"),
    }

agent = build_agent()

result = client.tracer.run_eval(
    dataset_id=os.environ["AGENTX_DATASET_ID"],
    agent_fn=agent.run,
    agent_name="my-agent",
    git_context=github_git_context(),
)

# Write results for artifact upload
with open("eval_results.json", "w") as f:
    json.dump({
        "gate": result.gate,
        "run_id": result.run_id,
        "pass_rate": result.pass_rate,
        "total_questions": result.total_questions,
        "passed_questions": result.passed_questions,
        "scores": [
            {
                "question_index": s.question_index,
                "rating": s.rating,
                "passed": s.passed,
                "justification": s.justification,
            }
            for s in result.scores
        ],
    }, f, indent=2)

# Print summary
print(f"\nAgentX Eval Gate: {result.gate.upper()}")
print(f"Pass rate: {result.pass_rate:.0%}  ({result.passed_questions}/{result.total_questions})\n")
for score in result.scores:
    icon = "✓" if score.passed else "✗"
    print(f"  [{icon}] Q{score.question_index}: {score.rating}/5  {score.justification[:80]}")

sys.exit(0 if result.gate == "pass" else 1)
```

---

## Local development

Run CI evals locally before pushing to get fast feedback:

```bash
export AGENTX_API_KEY=ax_live_xxxxxxxxxxxxxxxx
export AGENTX_DATASET_ID=6876ddd222bbb333ccc444ee
python ci_eval.py
```

To run against a specific dataset question only (for faster iteration):

```python
# Low-level: test a single question
run = tracer.create_ci_run(dataset_id=DATASET_ID)
test_case = run.test_cases[2]   # pick question index 2

output = my_agent(test_case.query)
score = tracer.submit_result(run.run_id, test_case.index, output)
print(f"Q{test_case.index}: {score.rating}/5 - {score.justification}")

# No need to finalize - just abandon the run (it expires automatically)
```

---

## Configuration reference

```python
client = AgentX(
    api_key="ax_live_xxxxxxxxxxxxxxxx",   # or set AGENTX_API_KEY
    workspace_id="...",                    # optional workspace override
    base_url="https://api.agentx.so",     # optional for self-hosted
    timeout=30,                            # HTTP timeout per request (seconds)
)
```
