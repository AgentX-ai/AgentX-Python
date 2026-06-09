"""
Anthropic SDK Agent Evaluation
--------------------------------
Covers two modes selectable via ANTHROPIC_MODE env var:

  ANTHROPIC_MODE=standard  (default) — claude-haiku-4-5, fast and cheap
  ANTHROPIC_MODE=thinking             — claude-opus-4-7 with extended thinking

Run:
    pip install anthropic
    AGENTX_API_KEY=key ANTHROPIC_API_KEY=sk-ant-... python examples/evaluations/anthropic_eval.py
    AGENTX_API_KEY=key ANTHROPIC_API_KEY=sk-ant-... ANTHROPIC_MODE=thinking python examples/evaluations/anthropic_eval.py
"""

import os

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase

# ---------------------------------------------------------------------------
# Mode: standard (claude-haiku-4-5)
# ---------------------------------------------------------------------------


def make_standard_fn(anthropic_client):
    def eval_subject(case: EvaluationCase) -> dict:
        if anthropic_client is None:
            return {
                "output": f"[stub] Response to: {case.query}",
                "metadata": {"framework": "anthropic", "mode": "standard"},
            }

        message = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system="You are a helpful customer support agent. Answer concisely and accurately.",
            messages=[{"role": "user", "content": case.query}],
        )

        return {
            "output": message.content[0].text,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "metadata": {
                "framework": "anthropic",
                "mode": "standard",
                "model": message.model,
            },
        }

    return eval_subject


# ---------------------------------------------------------------------------
# Mode: extended thinking (claude-opus-4-7)
# ---------------------------------------------------------------------------


def make_thinking_fn(anthropic_client):
    def eval_subject(case: EvaluationCase) -> dict:
        if anthropic_client is None:
            return {
                "output": f"[stub] Thinking response to: {case.query}",
                "metadata": {"framework": "anthropic", "mode": "thinking"},
            }

        message = anthropic_client.messages.create(
            model="claude-opus-4-7",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 8000},
            system="You are a helpful customer support agent. Think carefully before answering.",
            messages=[{"role": "user", "content": case.query}],
        )

        # Separate thinking blocks from answer blocks
        thinking_text = ""
        answer_text = ""
        for block in message.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                answer_text = block.text

        trace_events = []
        if thinking_text:
            trace_events.append(
                {
                    "type": "thinking",
                    "name": "claude-opus-4-7",
                    "summary": thinking_text[:500]
                    + ("…" if len(thinking_text) > 500 else ""),
                }
            )

        return {
            "output": answer_text,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "trace": {"events": trace_events} if trace_events else None,
            "metadata": {
                "framework": "anthropic",
                "mode": "thinking",
                "model": message.model,
            },
        }

    return eval_subject


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    client = AgentX.from_env()
    mode = os.getenv("ANTHROPIC_MODE", "standard").lower()

    try:
        import anthropic

        anthropic_client = anthropic.Anthropic()
    except ImportError:
        anthropic_client = None

    eval_fn = (
        make_thinking_fn(anthropic_client)
        if mode == "thinking"
        else make_standard_fn(anthropic_client)
    )
    model_label = (
        "claude-opus-4-7 (extended thinking)"
        if mode == "thinking"
        else "claude-haiku-4-5 (standard)"
    )

    dataset = (
        client.evaluations.datasets.builder(
            name=f"Anthropic Claude Agent Dataset ({mode})",
            description="Evaluates a Claude customer support agent on refund, support, and account queries.",
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
        .add_case(
            query="Can I downgrade my subscription mid-cycle?",
            expected_results="Explain the downgrade process, any proration, and when changes take effect.",
        )
        .publish()
    )

    report = (
        client.evaluations.run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": f"Claude Support Agent ({model_label})",
                "framework": "anthropic",
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
