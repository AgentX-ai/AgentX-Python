"""
LlamaIndex Agent Evaluation
-----------------------------
Evaluates a LlamaIndex query engine or agent without adding llama-index
as a hard SDK dependency.

Run:
    pip install llama-index llama-index-llms-openai
    AGENTX_API_KEY=your_key OPENAI_API_KEY=sk-... python examples/evaluations/llamaindex_eval.py
"""

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase


def build_llamaindex_engine():
    """Build and return a LlamaIndex query engine. Returns None if not installed."""
    try:
        from llama_index.core import VectorStoreIndex, Document
        from llama_index.llms.openai import OpenAI

        # In production, load your real documents here
        documents = [
            Document(text="Our refund policy allows full refunds within 30 days of purchase."),
            Document(text="We support credit cards, PayPal, and bank transfers as payment methods."),
            Document(text="Team plans support up to 50 seats. Contact sales for enterprise pricing."),
            Document(text="Technical support is available 24/7 via live chat and email."),
        ]

        llm = OpenAI(model="gpt-4o-mini", temperature=0)
        index = VectorStoreIndex.from_documents(documents)
        engine = index.as_query_engine(llm=llm, similarity_top_k=2)
        return engine

    except ImportError:
        return None


def make_eval_fn(engine):
    def eval_subject(case: EvaluationCase) -> dict:
        if engine is None:
            return {
                "output": f"[stub] LlamaIndex response to: {case.query}",
                "metadata": {"framework": "llamaindex"},
            }

        response = engine.query(case.query)
        source_nodes = getattr(response, "source_nodes", [])
        raw = getattr(response, "raw", None) or {}
        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        return {
            "output": str(response),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "metadata": {
                "framework": "llamaindex",
                "sources_used": len(source_nodes),
            },
        }
    return eval_subject


def main():
    client = AgentX.from_env()
    engine = build_llamaindex_engine()

    dataset = (
        client.evaluations.datasets
        .builder(
            name="LlamaIndex RAG Agent Dataset",
            description="Evaluates a LlamaIndex RAG engine answering questions grounded in product documentation.",
            number_of_requests=2,
            acceptance_criteria="Answers must be grounded in the provided documents. Cite relevant information.",
            rejection_criteria="No answers that contradict or go beyond the source documents.",
        )
        .add_case(
            query="What payment methods do you accept?",
            expected_results="List all supported payment methods as described in the docs.",
        )
        .add_case(
            query="How many seats does the team plan include?",
            expected_results="State the exact seat limit from the documentation.",
        )
        .publish()
    )

    print(f"Dataset: {dataset.id}")

    report = (
        client.evaluations
        .run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": "LlamaIndex RAG Engine",
                "framework": "llamaindex",
                "runtime": "local",
            },
        )
        .execute(make_eval_fn(engine))
        .finalize()
        .analyze()
    )

    print(f"\nDashboard: {report.dashboard_url}")


if __name__ == "__main__":
    main()
