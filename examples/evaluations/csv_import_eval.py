"""
CSV Dataset Import + Precomputed Results Evaluation
-----------------------------------------------------
Demonstrates two things:
  1. Loading a dataset from a CSV file instead of building it case-by-case.
  2. Submitting precomputed outputs (from n8n, Flowise, a batch job, etc.)
     without running any agent callable during the evaluation.

CSV format (required columns):
    query,expected_results
    "How do I reset my password?","Explain the password reset steps."
    "What is your refund policy?","Describe the refund terms clearly."

Optionally add a `case_id` column for stable idempotency across re-runs.

Run:
    AGENTX_API_KEY=your_key python examples/evaluations/csv_import_eval.py
"""

import csv
import os
import tempfile

from agentx import AgentX
from agentx.evaluations.adapters.precomputed import PrecomputedAdapter


def create_sample_csv() -> str:
    """Write a temporary sample CSV and return its path."""
    rows = [
        {
            "query": "How do I reset my password?",
            "expected_results": "Explain the password reset steps clearly.",
        },
        {
            "query": "What payment methods do you accept?",
            "expected_results": "List all supported payment methods.",
        },
        {
            "query": "How do I cancel my subscription?",
            "expected_results": "Describe the cancellation process step by step.",
        },
    ]
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    writer = csv.DictWriter(tmp, fieldnames=["query", "expected_results"])
    writer.writeheader()
    writer.writerows(rows)
    tmp.close()
    return tmp.name


def main():
    client = AgentX.from_env()

    # --- Step 1: Import dataset from CSV ---
    csv_path = create_sample_csv()
    print(f"Using CSV: {csv_path}")

    dataset = client.evaluations.datasets.from_csv(
        path=csv_path,
        name="CSV Import Demo Dataset",
        number_of_requests=1,
        acceptance_criteria="Answer must directly address the question and be accurate.",
        rejection_criteria="No vague or evasive answers.",
    )
    os.unlink(csv_path)  # cleanup temp file

    print(f"Dataset created: {dataset.id} ({len(dataset.cases)} cases)")

    # --- Step 2: Submit precomputed outputs ---
    # These come from a previous batch job, n8n workflow, Flowise run, etc.
    # Keys are case IDs or zero-based indices; values are the agent's output.
    precomputed_outputs = {
        0: "To reset your password: go to Login > Forgot Password, enter your email, and follow the link sent to you.",
        1: "We accept Visa, Mastercard, American Express, PayPal, and bank transfers.",
        2: "To cancel your subscription: go to Account Settings > Billing > Cancel Subscription, then confirm.",
    }

    adapter = PrecomputedAdapter(precomputed_outputs)

    report = (
        client.evaluations.run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": "n8n Batch Agent (Precomputed)",
                "framework": "n8n",
                "runtime": "low_code",
            },
        )
        .execute(adapter)
        .finalize()
        .analyze()
    )

    print(f"\nAverage rating: {report.average_rating:.2f}")
    print(f"Dashboard: {report.dashboard_url}")


if __name__ == "__main__":
    main()
