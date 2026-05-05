"""
Anthropic SDK Agent Evaluation
--------------------------------
Evaluates an agent built with the Anthropic Python SDK (Claude models).

Run:
    pip install anthropic
    AGENTX_API_KEY=your_key ANTHROPIC_API_KEY=sk-ant-... python examples/evaluations/anthropic_eval.py
"""

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase


def build_anthropic_client():
    try:
        import anthropic
        return anthropic.Anthropic()
    except ImportError:
        return None


def make_eval_fn(anthropic_client):
    def eval_subject(case: EvaluationCase) -> dict:
        if anthropic_client is None:
            return {"output": f"[stub] Response to: {case.query}", "metadata": {"framework": "anthropic"}}

        message = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system="You are a helpful customer support agent. Answer concisely and accurately.",
            messages=[{"role": "user", "content": case.query}],
        )
        return {
            "output": message.content[0].text,
            "metadata": {
                "framework": "anthropic",
                "model": message.model,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        }
    return eval_subject


def main():
    client = AgentX.from_env()
    anthropic_client = build_anthropic_client()

    dataset = (
        client.evaluations.datasets
        .builder(
            name="Anthropic Claude Agent Dataset",
            description="Evaluates a Claude Haiku customer support agent on refund and support contact queries.",
            number_of_requests=2,
            acceptance_criteria="Responses must be accurate, helpful, and appropriately concise.",
            rejection_criteria="No hallucinations, no harmful or misleading content.",
        )
        .add_case(
            query="What is your refund policy?",
            expected_results="Clearly explain refund eligibility, timeframes, and the process.",
        )
        .add_case(
            query="How do I contact technical support?",
            expected_results="Provide clear instructions for reaching technical support with expected response times.",
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
                "displayName": "Claude Haiku Support Agent",
                "framework": "anthropic",
                "runtime": "local",
                "version": "claude-haiku-4-5",
            },
        )
        .execute(make_eval_fn(anthropic_client))
        .finalize()
        .analyze()
    )

    print(f"\nDashboard: {report.dashboard_url}")


if __name__ == "__main__":
    main()
