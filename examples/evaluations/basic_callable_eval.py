"""
Basic Custom Agent Evaluation
------------------------------
Demonstrates the minimal setup: a plain Python function as the evaluation
subject, a dataset created via the builder, and a terminal + dashboard report.

Run:
    AGENTX_API_KEY=your_key python examples/evaluations/basic_callable_eval.py
"""

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase


def my_agent(case: EvaluationCase) -> str:
    """Replace this with your real agent call."""
    return f"This is a placeholder response to: {case.query}"


def main():
    client = AgentX.from_env()

    # 1. Create (or reuse) a dataset
    dataset = (
        client.evaluations.datasets
        .builder(
            name="Basic Regression Dataset",
            description="Regression dataset for a plain Python callable agent handling common support queries.",
            number_of_requests=2,
            acceptance_criteria="Answer must be accurate and concise.",
            rejection_criteria="Do not hallucinate.",
        )
        .add_case(
            query="How do I reset my account password?",
            expected_results="Explain the password reset process step by step.",
        )
        .add_case(
            query="What payment methods do you accept?",
            expected_results="List supported payment methods clearly.",
        )
        .publish()
    )

    print(f"Dataset created: {dataset.id}")

    # 2. Run the evaluation
    report = (
        client.evaluations
        .run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": "My Support Bot v1",
                "framework": "raw_python",
                "runtime": "local",
            },
        )
        .execute(my_agent)
        .finalize()
        .analyze()
    )

    print(f"\nDashboard: {report.dashboard_url}")


if __name__ == "__main__":
    main()
