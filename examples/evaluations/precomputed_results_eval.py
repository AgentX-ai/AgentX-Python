"""
Precomputed Results Evaluation
-------------------------------
Useful when you already have agent outputs (from a batch run, a log file,
n8n / Flowise exports, etc.) and just want AgentX to score them.

Run:
    AGENTX_API_KEY=your_key python examples/evaluations/precomputed_results_eval.py
"""

from agentx import AgentX
from agentx.evaluations.adapters.precomputed import PrecomputedAdapter


# Outputs keyed by case index (0-based) or case_id string
PRECOMPUTED_OUTPUTS = {
    "case-0": "To reset your password, go to Settings > Security > Reset Password.",
    "case-1": {
        "output": "We accept Visa, Mastercard, PayPal, and bank transfers.",
        "metadata": {"source": "n8n-export", "model": "gpt-4o"},
    },
}


def main():
    client = AgentX.from_env()

    dataset = (
        client.evaluations.datasets
        .builder(
            name="Precomputed Outputs Dataset",
            number_of_requests=1,
        )
        .add_case(
            query="How do I reset my account password?",
            expected_results="Explain the password reset process.",
        )
        .add_case(
            query="What payment methods do you accept?",
            expected_results="List accepted payment methods.",
        )
        .publish()
    )

    adapter = PrecomputedAdapter(PRECOMPUTED_OUTPUTS)

    report = (
        client.evaluations
        .run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": "n8n Export — batch 2024-Q1",
                "framework": "n8n",
                "runtime": "low_code",
            },
        )
        .execute(adapter)
        .finalize()
        .analyze()
    )

    print(f"\nDashboard: {report.dashboard_url}")


if __name__ == "__main__":
    main()
