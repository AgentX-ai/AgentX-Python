"""
Google Gemini SDK Agent Evaluation
-----------------------------------
Covers two modes selectable via GOOGLE_MODE env var:

  GOOGLE_MODE=standard  (default) — gemini-2.5-flash, fast and cheap
  GOOGLE_MODE=thinking             — gemini-2.5-pro with extended thinking

NOTE: uses google-genai, NOT the deprecated google-generativeai.

Run:
    pip install google-genai
    AGENTX_API_KEY=key GOOGLE_API_KEY=... python examples/evaluations/google_eval.py
    AGENTX_API_KEY=key GOOGLE_API_KEY=... GOOGLE_MODE=thinking python examples/evaluations/google_eval.py

Discovering available models:
    # The AgentX portability/sovereignty set (model ids you can compare against):
    for m in client.evaluations.list_models(provider="Google"):
        print(m.name, m.display, m.context_window)
    # Whatever Google currently serves (source of truth for valid Gemini ids):
    from google import genai
    for m in genai.Client().models.list():
        if "generateContent" in (m.supported_actions or []):
            print(m.name)
"""

import os

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase

_SYSTEM_INSTRUCTION = (
    "You are a helpful customer support agent. Answer concisely and accurately."
)

# ---------------------------------------------------------------------------
# Mode: standard (gemini-2.5-flash)
# ---------------------------------------------------------------------------


def make_standard_fn(genai_client, gtypes):
    def eval_subject(case: EvaluationCase) -> dict:
        if genai_client is None:
            return {
                "output": f"[stub] Response to: {case.query}",
                "metadata": {"framework": "google", "mode": "standard"},
            }

        model = "gemini-2.5-flash"
        response = genai_client.models.generate_content(
            model=model,
            contents=case.query,
            config=gtypes.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
            ),
        )

        return {
            "output": response.text or "",
            "input_tokens": getattr(
                response.usage_metadata, "prompt_token_count", None
            ),
            "output_tokens": getattr(
                response.usage_metadata, "candidates_token_count", None
            ),
            "metadata": {"framework": "google", "mode": "standard", "model": model},
        }

    return eval_subject


# ---------------------------------------------------------------------------
# Mode: extended thinking (gemini-2.5-pro)
# ---------------------------------------------------------------------------


def make_thinking_fn(genai_client, gtypes):
    def eval_subject(case: EvaluationCase) -> dict:
        if genai_client is None:
            return {
                "output": f"[stub] Thinking response to: {case.query}",
                "metadata": {"framework": "google", "mode": "thinking"},
            }

        model = "gemini-2.5-pro"
        response = genai_client.models.generate_content(
            model=model,
            contents=case.query,
            config=gtypes.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                thinking_config=gtypes.ThinkingConfig(thinking_budget=1024),
            ),
        )

        # Separate thinking parts from answer parts
        thinking_text = ""
        answer_text = ""
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False):
                thinking_text = part.text
            else:
                answer_text += part.text

        trace_events = []
        if thinking_text:
            trace_events.append(
                {
                    "type": "thinking",
                    "name": model,
                    "summary": thinking_text[:500]
                    + ("…" if len(thinking_text) > 500 else ""),
                }
            )

        return {
            "output": answer_text,
            "input_tokens": getattr(
                response.usage_metadata, "prompt_token_count", None
            ),
            "output_tokens": getattr(
                response.usage_metadata, "candidates_token_count", None
            ),
            "trace": {"events": trace_events} if trace_events else None,
            "metadata": {"framework": "google", "mode": "thinking", "model": model},
        }

    return eval_subject


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    client = AgentX.from_env()
    mode = os.getenv("GOOGLE_MODE", "standard").lower()

    try:
        from google import genai
        from google.genai import types as gtypes

        genai_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    except ImportError:
        genai_client = None
        gtypes = None

    eval_fn = (
        make_thinking_fn(genai_client, gtypes)
        if mode == "thinking"
        else make_standard_fn(genai_client, gtypes)
    )
    model_label = (
        "gemini-2.5-pro (extended thinking)"
        if mode == "thinking"
        else "gemini-2.5-flash (standard)"
    )

    dataset = (
        client.evaluations.datasets.builder(
            name=f"Google Gemini Agent Dataset ({mode})",
            description="Evaluates a Gemini customer support agent on refund, support, and account queries.",
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
                "displayName": f"Gemini Support Agent ({model_label})",
                "framework": "google",
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
#     "displayName": "Gemini Support Agent",
#     "framework": "google",
#     "runtime": "local",
#     "agentInstructions": (
#         "You are a helpful customer support agent for AcmeCorp. "
#         "Always be concise. Never mention competitors. "
#         "If you don't know the answer, say 'I'll escalate this to our team.'"
#     ),
# }
# ---------------------------------------------------------------------------
