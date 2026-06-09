"""
LlamaIndex Agent Evaluation
-----------------------------
Covers two modes selectable via LLAMA_MODE env var:

  LLAMA_MODE=rag    (default) — VectorStore RAG query engine with token tracking
  LLAMA_MODE=agent             — ReAct agent with tool use and step-level tracing

Run:
    pip install llama-index llama-index-llms-openai
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... python examples/evaluations/llamaindex_eval.py
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... LLAMA_MODE=agent python examples/evaluations/llamaindex_eval.py
"""

import os

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase

# ---------------------------------------------------------------------------
# Mode: RAG query engine
# ---------------------------------------------------------------------------


def build_rag_engine():
    try:
        from llama_index.core import VectorStoreIndex, Document
        from llama_index.llms.openai import OpenAI

        documents = [
            Document(
                text="Our refund policy allows full refunds within 30 days of purchase. Contact support@example.com."
            ),
            Document(
                text="We support credit cards, PayPal, and bank transfers as payment methods."
            ),
            Document(
                text="Team plans support up to 50 seats. Contact sales for enterprise pricing."
            ),
            Document(
                text="Technical support is available 24/7 via live chat and email."
            ),
            Document(
                text="To export data go to Settings → Data → Export. CSV and JSON formats are available."
            ),
            Document(
                text="Free trial is 14 days, no credit card required. Upgrade anytime from the billing page."
            ),
        ]

        llm = OpenAI(model="gpt-4o-mini", temperature=0)
        index = VectorStoreIndex.from_documents(documents)
        engine = index.as_query_engine(llm=llm, similarity_top_k=2)
        return engine
    except ImportError:
        return None


def make_rag_fn(engine):
    def eval_subject(case: EvaluationCase) -> dict:
        if engine is None:
            return {
                "output": f"[stub] RAG response to: {case.query}",
                "metadata": {"framework": "llamaindex", "mode": "rag"},
            }

        response = engine.query(case.query)

        # Token counts from the raw LLM response
        raw = getattr(response, "raw", None) or {}
        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")

        # Trace: sources used
        source_nodes = getattr(response, "source_nodes", [])
        trace_events = []
        for node in source_nodes:
            trace_events.append(
                {
                    "type": "retrieval",
                    "name": "vector_search",
                    "summary": (
                        f"score={node.score:.3f} text={node.text[:80]}…"
                        if hasattr(node, "score")
                        else node.text[:80]
                    ),
                }
            )

        return {
            "output": str(response),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "trace": {"events": trace_events} if trace_events else None,
            "metadata": {
                "framework": "llamaindex",
                "mode": "rag",
                "sources_used": len(source_nodes),
            },
        }

    return eval_subject


# ---------------------------------------------------------------------------
# Mode: ReAct agent with tools
# ---------------------------------------------------------------------------


def build_react_agent():
    try:
        from llama_index.core.agent import ReActAgent
        from llama_index.core.tools import FunctionTool
        from llama_index.llms.openai import OpenAI

        _POLICY_DB = {
            "refund": "Full refunds within 30 days. Email support@example.com.",
            "payment": "We accept Visa, Mastercard, PayPal, and bank transfer.",
            "support": "24/7 live chat and email support available.",
            "export": "Export via Settings → Data → Export (CSV/JSON).",
            "trial": "14-day free trial, no credit card required.",
        }

        def lookup_policy(topic: str) -> str:
            """Look up company policies by topic."""
            for key, val in _POLICY_DB.items():
                if key in topic.lower():
                    return val
            return "No specific policy found. Please contact support."

        tool = FunctionTool.from_defaults(fn=lookup_policy)
        llm = OpenAI(model="gpt-4o-mini", temperature=0)
        agent = ReActAgent.from_tools([tool], llm=llm, verbose=False)
        return agent
    except ImportError:
        return None


def make_agent_fn(agent):
    def eval_subject(case: EvaluationCase) -> dict:
        if agent is None:
            return {
                "output": f"[stub] Agent response to: {case.query}",
                "metadata": {"framework": "llamaindex", "mode": "agent"},
            }

        response = agent.chat(case.query)
        output = str(response)

        # Extract tool steps from agent sources
        trace_events = []
        sources = getattr(response, "sources", [])
        for source in sources:
            tool_name = getattr(source, "tool_name", "unknown_tool")
            content = str(getattr(source, "content", ""))[:100]
            trace_events.append(
                {"type": "tool_call", "name": tool_name, "summary": content}
            )

        return {
            "output": output,
            "trace": {"events": trace_events} if trace_events else None,
            "metadata": {
                "framework": "llamaindex",
                "mode": "agent",
                "model": "gpt-4o-mini",
            },
        }

    return eval_subject


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    client = AgentX.from_env()
    mode = os.getenv("LLAMA_MODE", "rag").lower()

    if mode == "agent":
        runner = build_react_agent()
        eval_fn = make_agent_fn(runner)
        display = "LlamaIndex ReAct Agent"
    else:
        runner = build_rag_engine()
        eval_fn = make_rag_fn(runner)
        display = "LlamaIndex RAG Engine"

    dataset = (
        client.evaluations.datasets.builder(
            name=f"LlamaIndex Agent Dataset ({mode})",
            description="Evaluates a LlamaIndex agent on product documentation queries.",
            number_of_requests=2,
            acceptance_criteria="Answers must be grounded in the provided documents.",
            rejection_criteria="No answers that contradict or go beyond the source documents.",
        )
        .add_case(
            query="What payment methods do you accept?",
            expected_results="List all supported payment methods as described in the docs.",
        )
        .add_case(
            query="Is there a free trial and do I need a credit card?",
            expected_results="Confirm trial length and whether a credit card is required.",
        )
        .add_case(
            query="How do I export my data?",
            expected_results="Explain the export steps and available formats.",
        )
        .publish()
    )

    report = (
        client.evaluations.run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": display,
                "framework": "llamaindex",
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
