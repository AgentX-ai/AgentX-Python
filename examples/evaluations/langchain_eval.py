"""
LangChain Agent Evaluation
---------------------------
Covers two modes selectable via LANGCHAIN_MODE env var:

  LANGCHAIN_MODE=chain  (default) — LLM chain with token tracking via usage_metadata
  LANGCHAIN_MODE=agent             — ReAct agent with tools and step-level tracing

Run:
    pip install langchain langchain-openai
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... python examples/evaluations/langchain_eval.py
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... LANGCHAIN_MODE=agent python examples/evaluations/langchain_eval.py
"""

import os

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase


# ---------------------------------------------------------------------------
# Shared: mock tool for policy lookups
# ---------------------------------------------------------------------------

_POLICY_DB = {
    "cancel": "To cancel: go to Account → Subscription → Cancel. Effective at end of billing period.",
    "trial": "We offer a 14-day free trial. No credit card required to start.",
    "refund": "Full refunds available within 30 days. Email support@example.com to request.",
    "export": "Export data from Settings → Data → Export (CSV or JSON).",
}


def _policy_lookup(topic: str) -> str:
    for key, val in _POLICY_DB.items():
        if key in topic.lower():
            return val
    return "No policy found for that topic. Please contact support."


# ---------------------------------------------------------------------------
# Mode: chain (LCEL) with usage_metadata token tracking
# ---------------------------------------------------------------------------

def build_chain():
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful support agent. Answer concisely."),
            ("human", "{query}"),
        ])
        return prompt | llm
    except ImportError:
        return None


def make_chain_fn(chain):
    def eval_subject(case: EvaluationCase) -> dict:
        if chain is None:
            return {"output": f"[stub] Chain response to: {case.query}", "metadata": {"framework": "langchain", "mode": "chain"}}

        result = chain.invoke({"query": case.query})
        usage = getattr(result, "usage_metadata", None) or {}

        return {
            "output": result.content if hasattr(result, "content") else str(result),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "metadata": {"framework": "langchain", "mode": "chain", "model": "gpt-4o-mini"},
        }

    return eval_subject


# ---------------------------------------------------------------------------
# Mode: ReAct agent with tools + step-level trace events
# ---------------------------------------------------------------------------

def build_agent():
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain_core.prompts import PromptTemplate

        @tool
        def policy_lookup(topic: str) -> str:
            """Look up a company policy by topic (cancel, trial, refund, export)."""
            return _policy_lookup(topic)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        tools = [policy_lookup]

        prompt = PromptTemplate.from_template(
            "You are a helpful support agent.\n\n"
            "Tools available:\n{tools}\n\nTool names: {tool_names}\n\n"
            "Question: {input}\nThought: {agent_scratchpad}"
        )

        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, return_intermediate_steps=True)
    except ImportError:
        return None


def make_agent_fn(agent_executor):
    def eval_subject(case: EvaluationCase) -> dict:
        if agent_executor is None:
            return {"output": f"[stub] Agent response to: {case.query}", "metadata": {"framework": "langchain", "mode": "agent"}}

        result = agent_executor.invoke({"input": case.query})
        output = result.get("output", "")

        # Build trace events from intermediate steps (tool calls)
        trace_events = []
        for action, observation in result.get("intermediate_steps", []):
            trace_events.append({
                "type": "tool_call",
                "name": action.tool,
                "summary": f"input={action.tool_input!r} → {str(observation)[:100]}",
            })

        # Token counts not easily available from AgentExecutor — use callback if needed
        return {
            "output": output,
            "trace": {"events": trace_events} if trace_events else None,
            "metadata": {"framework": "langchain", "mode": "agent", "model": "gpt-4o-mini"},
        }

    return eval_subject


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    client = AgentX.from_env()
    mode = os.getenv("LANGCHAIN_MODE", "chain").lower()

    if mode == "agent":
        runner = build_agent()
        eval_fn = make_agent_fn(runner)
        display = "LangChain ReAct Agent (tools)"
    else:
        runner = build_chain()
        eval_fn = make_chain_fn(runner)
        display = "LangChain LCEL Chain"

    dataset = (
        client.evaluations.datasets
        .builder(
            name=f"LangChain Support Agent Dataset ({mode})",
            description="Evaluates a LangChain-based support agent on subscription and trial queries.",
            number_of_requests=2,
            acceptance_criteria="Accurate, concise, grounded in policy.",
            rejection_criteria="No hallucinated policy details.",
        )
        .add_case(
            query="How do I cancel my subscription?",
            expected_results="Explain the cancellation steps clearly.",
        )
        .add_case(
            query="Is there a free trial available?",
            expected_results="Accurately describe free trial availability and length.",
        )
        .add_case(
            query="What is your refund policy?",
            expected_results="State the refund window and how to request one.",
        )
        .publish()
    )

    report = (
        client.evaluations
        .run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": display,
                "framework": "langchain",
                "runtime": "local",
            },
        )
        .execute(eval_fn)
        .finalize()
        .analyze()
    )

    print(f"\nDashboard: {report.dashboard_url}")


if __name__ == "__main__":
    main()
