"""
OpenAI SDK Agent Evaluation
----------------------------
Evaluates an agent built directly with the OpenAI Python SDK
(chat completions or the Assistants API).

Run:
    pip install openai
    AGENTX_API_KEY=your_key OPENAI_API_KEY=sk-... python examples/evaluations/openai_eval.py
"""

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase


def build_openai_agent():
    try:
        from openai import OpenAI
        return OpenAI()
    except ImportError:
        return None


def make_eval_fn(oai_client):
    def eval_subject(case: EvaluationCase) -> dict:
        if oai_client is None:
            return {"output": f"[stub] Response to: {case.query}", "metadata": {"framework": "openai"}}

        response = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful customer support agent. Answer concisely."},
                {"role": "user", "content": case.query},
            ],
            temperature=0,
        )
        return {
            "output": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "metadata": {
                "framework": "openai",
                "model": response.model,
            },
        }
    return eval_subject


def main():
    client = AgentX.from_env()
    oai_client = build_openai_agent()

    dataset = (
        client.evaluations.datasets
        .builder(
            name="OpenAI SDK Agent Dataset",
            description="Evaluates a GPT-4o-mini based customer support agent on common billing and account queries.",
            number_of_requests=2,
            acceptance_criteria="Accurate, concise, and professional customer support responses.",
            rejection_criteria="No hallucinated facts, no harmful content.",
        )
        .add_case(
            query="How do I upgrade my plan?",
            expected_results="Describe the upgrade steps clearly, including where to find billing settings.",
        )
        .add_case(
            query="Can I export my data?",
            expected_results="Explain the data export process and available formats.",
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
                "displayName": "OpenAI GPT-4o-mini Agent",
                "framework": "openai",
                "runtime": "local",
                "version": "gpt-4o-mini",
            },
        )
        .execute(make_eval_fn(oai_client))
        .finalize()
        .analyze()
    )

    print(f"\nDashboard: {report.dashboard_url}")


if __name__ == "__main__":
    main()
