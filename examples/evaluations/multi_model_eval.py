"""
Multi-Model Evaluation
-----------------------
Runs the same dataset through multiple models across OpenAI, Anthropic, and
Google (Gemini) so you can compare quality, latency, and token costs side by
side on the AgentX dashboard.

Each model gets its own evaluation run — all share the same dataset so the
comparison is apples-to-apples.

Setup:
    cp .env.example .env          # fill in your API keys
    pip install openai anthropic google-genai python-dotenv

Run all:
    python examples/evaluations/multi_model_eval.py

Run a subset:
    PROVIDERS=openai,anthropic python examples/evaluations/multi_model_eval.py
    PROVIDERS=google             python examples/evaluations/multi_model_eval.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

# Load .env if present (optional — keys can also be set in the shell)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase

# ---------------------------------------------------------------------------
# Shared agent instructions — passed to the evaluator so it can score
# instruction adherence for every model on the same criteria.
# ---------------------------------------------------------------------------

AGENT_INSTRUCTIONS = (
    "You are a helpful customer support agent. "
    "Always answer concisely (max 3 sentences). "
    "If you don't know the answer say: 'I'll escalate this to our team.' "
    "Never mention competitor products."
)

SYSTEM_PROMPT = f"System: {AGENT_INSTRUCTIONS}"


# ---------------------------------------------------------------------------
# Model descriptor
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    provider: str          # "openai" | "anthropic" | "google"
    model_id: str          # API model name
    display_name: str      # shown in AgentX dashboard
    framework: str         # agentx framework tag
    reasoning: bool = False  # uses extended thinking / reasoning


# ---------------------------------------------------------------------------
# OpenAI runners
# ---------------------------------------------------------------------------

def _openai_chat_fn(model_id: str) -> Callable:
    """Responses API — works for gpt-4.1, gpt-5.x, etc.
    gpt-5.x does not support temperature so we omit it for those models."""
    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError:
        client = None

    def run(case: EvaluationCase) -> dict:
        if client is None:
            return {"output": f"[stub] {model_id}: {case.query}"}

        kwargs = dict(
            model=model_id,
            input=[
                {"role": "system", "content": AGENT_INSTRUCTIONS},
                {"role": "user", "content": case.query},
            ],
        )
        if not model_id.startswith("gpt-5"):
            kwargs["temperature"] = 0

        response = client.responses.create(**kwargs)
        return {
            "output": response.output_text or "",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "metadata": {"model": model_id},
        }

    return run


def _openai_reasoning_fn(model_id: str) -> Callable:
    """Responses API — o4-mini with reasoning summary in trace."""
    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError:
        client = None

    def run(case: EvaluationCase) -> dict:
        if client is None:
            return {"output": f"[stub] {model_id} reasoning: {case.query}"}

        response = client.responses.create(
            model=model_id,
            input=[
                {"role": "system", "content": AGENT_INSTRUCTIONS},
                {"role": "user", "content": case.query},
            ],
            reasoning={"effort": "medium", "summary": "auto"},
        )

        reasoning_summary = None
        output_text = response.output_text or ""
        for item in response.output:
            if item.type == "reasoning":
                summary = getattr(item, "summary", None)
                if summary:
                    reasoning_summary = " ".join(
                        s.text for s in summary if hasattr(s, "text")
                    )

        trace_events = []
        if reasoning_summary:
            trace_events.append({"type": "reasoning", "name": model_id, "summary": reasoning_summary})

        return {
            "output": output_text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "trace": {"events": trace_events} if trace_events else None,
            "metadata": {"model": model_id},
        }

    return run


# ---------------------------------------------------------------------------
# Anthropic runners
# ---------------------------------------------------------------------------

def _anthropic_fn(model_id: str) -> Callable:
    """Standard messages API — haiku, sonnet, etc."""
    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        client = None

    def run(case: EvaluationCase) -> dict:
        if client is None:
            return {"output": f"[stub] {model_id}: {case.query}"}
        msg = client.messages.create(
            model=model_id,
            max_tokens=1024,
            system=AGENT_INSTRUCTIONS,
            messages=[{"role": "user", "content": case.query}],
        )
        return {
            "output": msg.content[0].text,
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
            "metadata": {"model": msg.model},
        }

    return run


def _anthropic_thinking_fn(model_id: str) -> Callable:
    """Extended thinking — claude-opus-4-7."""
    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        client = None

    def run(case: EvaluationCase) -> dict:
        if client is None:
            return {"output": f"[stub] {model_id} thinking: {case.query}"}
        msg = client.messages.create(
            model=model_id,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=AGENT_INSTRUCTIONS,
            messages=[{"role": "user", "content": case.query}],
        )
        thinking_text = ""
        answer_text = ""
        for block in msg.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                answer_text = block.text

        trace_events = []
        if thinking_text:
            trace_events.append({
                "type": "thinking",
                "name": model_id,
                "summary": thinking_text[:500] + ("…" if len(thinking_text) > 500 else ""),
            })

        return {
            "output": answer_text,
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
            "trace": {"events": trace_events} if trace_events else None,
            "metadata": {"model": msg.model},
        }

    return run


# ---------------------------------------------------------------------------
# Google (Gemini) runners
# ---------------------------------------------------------------------------

_GOOGLE_RETRY_DELAYS = [5, 15, 30]


def _google_call_with_retry(fn, *args, **kwargs):
    """Retry Google API calls on 429 rate-limit errors with backoff."""
    last_exc = None
    for delay in [0] + _GOOGLE_RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                last_exc = exc
                continue
            raise
    raise last_exc


def _google_fn(model_id: str) -> Callable:
    """google-genai — gemini-2.5-flash, gemini-2.5-flash-lite, etc."""
    try:
        from google import genai as ggenai
        from google.genai import types as gtypes
        client = ggenai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    except ImportError:
        client = None

    def run(case: EvaluationCase) -> dict:
        if client is None:
            return {"output": f"[stub] {model_id}: {case.query}"}

        response = _google_call_with_retry(
            client.models.generate_content,
            model=model_id,
            contents=case.query,
            config=gtypes.GenerateContentConfig(
                system_instruction=AGENT_INSTRUCTIONS,
            ),
        )
        output = response.text or ""
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

        return {
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "metadata": {"model": model_id},
        }

    return run


def _google_thinking_fn(model_id: str) -> Callable:
    """Gemini thinking models — gemini-2.5-pro."""
    try:
        from google import genai as ggenai
        from google.genai import types as gtypes
        client = ggenai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    except ImportError:
        client = None

    def run(case: EvaluationCase) -> dict:
        if client is None:
            return {"output": f"[stub] {model_id} thinking: {case.query}"}

        response = _google_call_with_retry(
            client.models.generate_content,
            model=model_id,
            contents=case.query,
            config=gtypes.GenerateContentConfig(
                system_instruction=AGENT_INSTRUCTIONS,
                thinking_config=gtypes.ThinkingConfig(thinking_budget=1024),
            ),
        )

        thinking_text = ""
        answer_text = ""
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False):
                thinking_text = part.text
            else:
                answer_text += part.text

        trace_events = []
        if thinking_text:
            trace_events.append({
                "type": "reasoning",
                "name": model_id,
                "summary": thinking_text[:500] + ("…" if len(thinking_text) > 500 else ""),
            })

        input_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

        return {
            "output": answer_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "trace": {"events": trace_events} if trace_events else None,
            "metadata": {"model": model_id},
        }

    return run


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ALL_MODELS: list[ModelConfig] = [
    # OpenAI — chat
    ModelConfig("openai", "gpt-5.4",      "OpenAI GPT-5.4",       "openai"),
    ModelConfig("openai", "gpt-5.5",      "OpenAI GPT-5.5",       "openai"),
    # OpenAI — reasoning
    ModelConfig("openai", "o4-mini",      "OpenAI o4-mini",       "openai", reasoning=True),
    # Anthropic — standard
    ModelConfig("anthropic", "claude-haiku-4-5",   "Claude Haiku 4.5",   "anthropic"),
    ModelConfig("anthropic", "claude-sonnet-4-6",  "Claude Sonnet 4.6",  "anthropic"),
    # Anthropic — thinking
    ModelConfig("anthropic", "claude-opus-4-7",    "Claude Opus 4.7 (thinking)", "anthropic", reasoning=True),
    # Google — standard
    ModelConfig("google", "gemini-3-flash-preview",  "Gemini 3 Flash",          "google"),
    ModelConfig("google", "gemini-3.1-pro-preview",  "Gemini 3.1 Pro",          "google"),
    # Google — thinking
    ModelConfig("google", "gemini-3-pro-preview",    "Gemini 3 Pro (thinking)", "google", reasoning=True),
]


def _make_runner(cfg: ModelConfig) -> Callable:
    if cfg.provider == "openai":
        return _openai_reasoning_fn(cfg.model_id) if cfg.reasoning else _openai_chat_fn(cfg.model_id)
    if cfg.provider == "anthropic":
        return _anthropic_thinking_fn(cfg.model_id) if cfg.reasoning else _anthropic_fn(cfg.model_id)
    if cfg.provider == "google":
        return _google_thinking_fn(cfg.model_id) if cfg.reasoning else _google_fn(cfg.model_id)
    raise ValueError(f"Unknown provider: {cfg.provider}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Filter by PROVIDERS env var, e.g. PROVIDERS=openai,google
    providers_filter = {p.strip().lower() for p in os.getenv("PROVIDERS", "").split(",") if p.strip()}
    # Filter by MODEL env var (partial match on model_id), e.g. MODEL=gpt-5.5
    model_filter = os.getenv("MODEL", "").strip().lower()

    models = [
        m for m in ALL_MODELS
        if (not providers_filter or m.provider in providers_filter)
        and (not model_filter or model_filter in m.model_id.lower())
    ]

    if not models:
        print("No models selected.")
        print("  PROVIDERS=openai,anthropic,google  — filter by provider")
        print("  MODEL=gpt-5.5                      — run a single model")
        sys.exit(1)

    client = AgentX.from_env()

    # One shared dataset — all models evaluated on identical questions
    dataset = (
        client.evaluations.datasets
        .builder(
            name="Multi-Model Benchmark",
            description=(
                "Benchmark dataset for comparing multiple LLM providers and models "
                "on customer support quality, latency, and token efficiency."
            ),
            number_of_requests=2,
            acceptance_criteria="Accurate, concise (max 3 sentences), professional tone.",
            rejection_criteria="No hallucinations, no mention of competitor products, no empty responses.",
        )
        .add_case(
            query="How do I upgrade my plan?",
            expected_results="Describe the upgrade steps clearly, including where to find billing settings.",
        )
        .add_case(
            query="What is your refund policy?",
            expected_results="State the refund eligibility window and how to request one.",
        )
        .add_case(
            query="Can I export my data and in what formats?",
            expected_results="Confirm export availability, list supported formats, and explain how to access it.",
        )
        .add_case(
            query="Is there a free trial? Do I need a credit card to start?",
            expected_results="State trial length and credit card requirements.",
        )
        .publish()
    )

    print(f"\nDataset: {dataset.id}  ({len(dataset.questions)} questions)\n")
    print(f"Running {len(models)} model(s):\n")
    for m in models:
        flag = " 🧠" if m.reasoning else ""
        print(f"  {m.provider:12s}  {m.display_name}{flag}")
    print()

    reports = []
    for cfg in models:
        try:
            runner = _make_runner(cfg)
            report = (
                client.evaluations
                .run(
                    dataset_id=dataset.id,
                    subject={
                        "kind": "custom_agent",
                        "displayName": cfg.display_name,
                        "framework": cfg.framework,
                        "runtime": "local",
                        "agentInstructions": AGENT_INSTRUCTIONS,
                    },
                )
                .execute(runner)
                .finalize()
                .analyze()
            )
            reports.append((cfg, report))
        except Exception as exc:
            print(f"\n  [ERROR] {cfg.display_name}: {exc}\n")

    # Summary table
    if reports:
        print("\n" + "─" * 72)
        print(f"  {'Model':<35} {'Avg rating':>10}  {'Dashboard'}")
        print("─" * 72)
        for cfg, r in reports:
            avg = f"{r.statistics.average_rating:.1f}/10" if r.statistics else "N/A"
            url = r.dashboard_url or ""
            print(f"  {cfg.display_name:<35} {avg:>10}  {url}")
        print("─" * 72)


if __name__ == "__main__":
    main()
