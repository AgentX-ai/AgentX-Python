"""
LangChain Agent Evaluation
---------------------------
Wraps a LangChain agent without adding langchain as a hard SDK dependency.
The adapter pattern works identically for CrewAI, AutoGen, LlamaIndex, etc.
— just swap the inner call.

Run:
    pip install langchain openai
    AGENTX_API_KEY=your_key OPENAI_API_KEY=sk-... python examples/evaluations/langchain_eval.py
"""

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase


def build_langchain_agent():
    """Build your LangChain agent here. Returns an object with .invoke()."""
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


def make_eval_fn(agent):
    def eval_subject(case: EvaluationCase) -> dict:
        if agent is None:
            return {"output": f"[stub] Response to: {case.query}", "metadata": {"framework": "langchain"}}

        result = agent.invoke({"query": case.query})
        usage = getattr(result, "usage_metadata", None) or {}
        return {
            "output": result.content if hasattr(result, "content") else str(result),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "metadata": {
                "framework": "langchain",
                "model": "gpt-4o-mini",
            },
        }
    return eval_subject


def main():
    client = AgentX.from_env()
    agent = build_langchain_agent()

    dataset = (
        client.evaluations.datasets
        .builder(
            name="LangChain Support Agent Dataset",
            description="Evaluates a LangChain-based support agent on subscription and trial queries.",
            number_of_requests=1,
            acceptance_criteria="Accurate, concise, grounded in docs.",
            rejection_criteria="No hallucinated policy details.",
        )
        .add_case(
            query="How do I cancel my subscription?",
            expected_results="Explain the cancellation steps clearly.",
        )
        .add_case(
            query="Is there a free trial available?",
            expected_results="Accurately describe free trial availability.",
        )
        .publish()
    )

    report = (
        client.evaluations
        .run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": "LangChain Support Bot",
                "framework": "langchain",
                "runtime": "local",
            },
        )
        .execute(make_eval_fn(agent))
        .finalize()
        .analyze()
    )

    print(f"\nDashboard: {report.dashboard_url}")


if __name__ == "__main__":
    main()
