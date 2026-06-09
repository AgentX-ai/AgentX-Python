"""
OpenAI SDK Agent Evaluation
----------------------------
Covers two modes selectable via OPENAI_MODE env var:

  OPENAI_MODE=chat   (default) — gpt-4o-mini with tool use
  OPENAI_MODE=reason           — o4-mini via Responses API with reasoning summary

Run:
    pip install openai
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... python examples/evaluations/openai_eval.py
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... OPENAI_MODE=reason python examples/evaluations/openai_eval.py
"""

import json
import os

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase

# ---------------------------------------------------------------------------
# Mode: chat completions (gpt-4o-mini) with optional tool use
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_policy",
            "description": "Look up a company policy by topic (refunds, billing, support, etc.).",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    }
]

_POLICY_DB = {
    "refund": "Full refunds are available within 30 days of purchase. Contact support@example.com.",
    "billing": "We accept Visa, Mastercard, PayPal, and bank transfer. Invoices are issued monthly.",
    "support": "Technical support is available 24/7 via live chat and email (support@example.com).",
    "upgrade": "Upgrade via Account → Billing → Change Plan. Changes take effect immediately.",
    "export": "Export your data from Settings → Data → Export. CSV and JSON formats are supported.",
}


def _run_policy_tool(args: dict) -> str:
    topic = args.get("topic", "").lower()
    for key, value in _POLICY_DB.items():
        if key in topic:
            return value
    return "Policy not found for that topic."


def make_chat_fn(oai_client):
    def eval_subject(case: EvaluationCase) -> dict:
        if oai_client is None:
            return {
                "output": f"[stub] Response to: {case.query}",
                "metadata": {"framework": "openai", "mode": "chat"},
            }

        messages = [
            {
                "role": "system",
                "content": "You are a helpful customer support agent. Use the lookup_policy tool when you need policy details.",
            },
            {"role": "user", "content": case.query},
        ]
        trace_events = []

        # First call — may invoke tool
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        msg = resp.choices[0].message

        # Handle tool call
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = _run_policy_tool(args)
                trace_events.append(
                    {
                        "type": "tool_call",
                        "name": tc.function.name,
                        "summary": f"topic={args.get('topic')} → {result[:80]}",
                    }
                )
                messages.append(msg)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

            # Second call with tool result
            resp = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
            )
            msg = resp.choices[0].message

        output = msg.content or ""
        usage = resp.usage

        return {
            "output": output,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "trace": {"events": trace_events} if trace_events else None,
            "metadata": {"framework": "openai", "mode": "chat", "model": resp.model},
        }

    return eval_subject


# ---------------------------------------------------------------------------
# Mode: reasoning (o4-mini) via Responses API — returns reasoning summary
# ---------------------------------------------------------------------------


def make_reasoning_fn(oai_client):
    def eval_subject(case: EvaluationCase) -> dict:
        if oai_client is None:
            return {
                "output": f"[stub] Reasoning response to: {case.query}",
                "metadata": {"framework": "openai", "mode": "reasoning"},
            }

        response = oai_client.responses.create(
            model="o4-mini",
            input=[
                {
                    "role": "system",
                    "content": "You are a helpful customer support agent. Answer accurately and concisely.",
                },
                {"role": "user", "content": case.query},
            ],
            reasoning={"effort": "medium"},
            include=["reasoning.summary"],
        )

        # Extract reasoning summary and final answer from output items
        reasoning_summary = None
        output_text = ""
        for item in response.output:
            if item.type == "reasoning" and getattr(item, "summary", None):
                reasoning_summary = " ".join(
                    s.text for s in item.summary if hasattr(s, "text")
                )
            elif item.type == "message":
                for block in item.content:
                    if hasattr(block, "text"):
                        output_text += block.text

        trace_events = []
        if reasoning_summary:
            trace_events.append(
                {"type": "reasoning", "name": "o4-mini", "summary": reasoning_summary}
            )

        return {
            "output": output_text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "trace": {"events": trace_events} if trace_events else None,
            "metadata": {
                "framework": "openai",
                "mode": "reasoning",
                "model": "o4-mini",
            },
        }

    return eval_subject


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    client = AgentX.from_env()
    mode = os.getenv("OPENAI_MODE", "chat").lower()

    try:
        from openai import OpenAI

        oai_client = OpenAI()
    except ImportError:
        oai_client = None

    eval_fn = (
        make_reasoning_fn(oai_client) if mode == "reason" else make_chat_fn(oai_client)
    )
    display = (
        "OpenAI o4-mini (reasoning)"
        if mode == "reason"
        else "OpenAI GPT-4o-mini (chat + tools)"
    )

    dataset = (
        client.evaluations.datasets.builder(
            name=f"OpenAI Agent Dataset ({mode})",
            description="Evaluates an OpenAI-based customer support agent on billing and account queries.",
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
        .add_case(
            query="What is your refund policy?",
            expected_results="Clearly state the refund eligibility window and how to request one.",
        )
        .publish()
    )

    report = (
        client.evaluations.run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": display,
                "framework": "openai",
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

# ---------------------------------------------------------------------------
# Subject with agentInstructions — the system prompt is sent to the API so the
# AI evaluator can check whether responses actually follow the stated instructions.
#
# subject={
#     "kind": "custom_agent",
#     "displayName": "OpenAI GPT-4o-mini Agent",
#     "framework": "openai",
#     "runtime": "local",
#     "agentInstructions": (
#         "You are a helpful customer support agent for AcmeCorp. "
#         "Always be concise. Never mention competitors. "
#         "If you don't know the answer, say 'I'll escalate this to our team.'"
#     ),
# }
# ---------------------------------------------------------------------------
