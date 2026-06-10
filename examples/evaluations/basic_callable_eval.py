"""
Basic Custom Agent Evaluation
------------------------------
Demonstrates the minimal setup: a plain Python function as the evaluation
subject, optional agentInstructions so the AI evaluator can check adherence,
and trace events for multi-step reasoning.

Run:
    AGENTX_API_KEY=your_key python examples/evaluations/basic_callable_eval.py
"""

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase

# The system prompt / instructions your agent actually uses.
# Pass this in the subject so the AI evaluator can assess whether
# responses follow the stated rules.
AGENT_INSTRUCTIONS = (
    "You are a helpful customer support agent for AcmeCorp. "
    "Always be concise (max 3 sentences). "
    "Never mention competitor products. "
    "If you don't know the answer, say: 'I'll escalate this to our team.'"
)


def my_agent(case: EvaluationCase) -> dict:
    """
    Replace this with your real agent call.
    Return a dict with 'output' and optionally 'trace' and timing fields.
    """
    # Simulate a simple response
    response = f"Thank you for your question about '{case.query}'. Our team will assist you shortly."

    # Optionally attach a trace — useful for multi-step agents
    # trace_events can include: thinking, reasoning, tool_call, agent_step, retrieval
    return {
        "output": response,
        "trace": {
            "events": [
                {
                    "type": "agent_step",
                    "name": "response_generator",
                    "summary": f"Generated response for: {case.query[:60]}",
                },
            ]
        },
        "metadata": {"framework": "raw_python", "version": "1.0"},
    }


def main():
    client = AgentX.from_env()

    dataset = (
        client.evaluations.datasets.builder(
            name="Basic Regression Dataset",
            description="Regression dataset for a plain Python callable agent handling common support queries.",
            number_of_requests=2,
            acceptance_criteria="Answer must be accurate, concise (max 3 sentences), and professional.",
            rejection_criteria="Do not hallucinate. Do not mention competitor products.",
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

    report = (
        client.evaluations.run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": "My Support Bot v1",
                "framework": "raw_python",
                "runtime": "local",
                # agentInstructions: pass your agent's system prompt here so the
                # AI evaluator can score instruction adherence properly.
                "agentInstructions": AGENT_INSTRUCTIONS,
            },
        )
        .execute(my_agent)
        .finalize()
        .analyze()
    )

    print(f"\nDashboard: {report.dashboard_url}")


if __name__ == "__main__":
    main()
