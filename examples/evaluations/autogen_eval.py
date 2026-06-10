"""
AutoGen / AG2 Agent Evaluation
--------------------------------
Works with both AutoGen v0.2 (pyautogen) and AG2 (ag2 package).

Covers two modes selectable via AUTOGEN_MODE env var:

  AUTOGEN_MODE=single  (default) — single ConversableAgent
  AUTOGEN_MODE=group              — GroupChat with specialist agents

Run:
    pip install pyautogen   # or: pip install ag2
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... python examples/evaluations/autogen_eval.py
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... AUTOGEN_MODE=group python examples/evaluations/autogen_eval.py
"""

import os

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase


def _import_conversable_agent():
    try:
        from autogen import ConversableAgent, GroupChat, GroupChatManager

        return ConversableAgent, GroupChat, GroupChatManager
    except ImportError:
        pass
    try:
        from ag2 import ConversableAgent, GroupChat, GroupChatManager

        return ConversableAgent, GroupChat, GroupChatManager
    except ImportError:
        return None, None, None


def _llm_config():
    return {
        "config_list": [
            {"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY", "")}
        ],
        "temperature": 0,
    }


def _extract_tokens(chat_result) -> tuple:
    """Best-effort token extraction from AutoGen chat result cost summary."""
    cost = getattr(chat_result, "cost", None) or {}
    usage = (
        cost.get("usage_including_cached_inference", {})
        if isinstance(cost, dict)
        else {}
    )
    total = usage.get("total", {}) if isinstance(usage, dict) else {}
    return total.get("prompt_tokens"), total.get("completion_tokens")


# ---------------------------------------------------------------------------
# Mode: single ConversableAgent
# ---------------------------------------------------------------------------


def build_single_agent():
    ConversableAgent, _, _ = _import_conversable_agent()
    if ConversableAgent is None:
        return None

    assistant = ConversableAgent(
        name="SupportAgent",
        system_message="You are a helpful customer support agent. Answer concisely and accurately.",
        llm_config=_llm_config(),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )
    user_proxy = ConversableAgent(
        name="UserProxy",
        llm_config=False,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
    )

    def run(query: str):
        chat_result = user_proxy.initiate_chat(
            assistant, message=query, max_turns=1, silent=True
        )
        messages = assistant.chat_messages.get(user_proxy, [])
        output = next(
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "assistant"
            ),
            "",
        )
        input_tokens, output_tokens = _extract_tokens(chat_result)
        return output, input_tokens, output_tokens, []

    return run


# ---------------------------------------------------------------------------
# Mode: GroupChat with researcher + writer
# ---------------------------------------------------------------------------


def build_group_chat():
    ConversableAgent, GroupChat, GroupChatManager = _import_conversable_agent()
    if ConversableAgent is None:
        return None

    researcher = ConversableAgent(
        name="Researcher",
        system_message="You are a policy researcher. Identify the relevant policy information for the customer question.",
        llm_config=_llm_config(),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )
    writer = ConversableAgent(
        name="Writer",
        system_message="You are a support writer. Based on the researcher's findings, write a concise customer-facing response.",
        llm_config=_llm_config(),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )
    user_proxy = ConversableAgent(
        name="UserProxy",
        llm_config=False,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda m: m.get("content", "").strip().endswith("DONE"),
    )

    def run(query: str):
        group_chat = GroupChat(
            agents=[user_proxy, researcher, writer], messages=[], max_round=4
        )
        manager = GroupChatManager(groupchat=group_chat, llm_config=_llm_config())
        chat_result = user_proxy.initiate_chat(manager, message=query, silent=True)

        # Build trace from group chat messages
        trace_events = []
        for msg in group_chat.messages:
            if msg.get("role") == "assistant" and msg.get("name") in (
                "Researcher",
                "Writer",
            ):
                trace_events.append(
                    {
                        "type": "agent_step",
                        "name": msg["name"].lower(),
                        "summary": str(msg.get("content", ""))[:200],
                    }
                )

        # Final answer is the last Writer message
        output = next(
            (
                m.get("content", "")
                for m in reversed(group_chat.messages)
                if m.get("name") == "Writer"
            ),
            "",
        )
        input_tokens, output_tokens = _extract_tokens(chat_result)
        return output, input_tokens, output_tokens, trace_events

    return run


# ---------------------------------------------------------------------------
# Shared eval function factory
# ---------------------------------------------------------------------------


def make_eval_fn(runner, mode: str):
    def eval_subject(case: EvaluationCase) -> dict:
        if runner is None:
            return {
                "output": f"[stub] AutoGen ({mode}) response to: {case.query}",
                "metadata": {"framework": "autogen", "mode": mode},
            }

        output, input_tokens, output_tokens, trace_events = runner(case.query)
        return {
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "trace": {"events": trace_events} if trace_events else None,
            "metadata": {"framework": "autogen", "mode": mode, "model": "gpt-4o-mini"},
        }

    return eval_subject


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    client = AgentX.from_env()
    mode = os.getenv("AUTOGEN_MODE", "single").lower()

    runner = build_group_chat() if mode == "group" else build_single_agent()
    eval_fn = make_eval_fn(runner, mode)
    display = (
        "AutoGen GroupChat (researcher + writer)"
        if mode == "group"
        else "AutoGen Single Agent"
    )

    dataset = (
        client.evaluations.datasets.builder(
            name=f"AutoGen Support Agent Dataset ({mode})",
            description="Evaluates an AutoGen conversational agent on 2FA and API availability queries.",
            number_of_requests=2,
            acceptance_criteria="Accurate, concise answers that fully address the customer's question.",
            rejection_criteria="No off-topic responses, no hallucinated product details.",
        )
        .add_case(
            query="How do I reset my 2FA settings?",
            expected_results="Provide clear steps for resetting two-factor authentication.",
        )
        .add_case(
            query="Is there an API available for your service?",
            expected_results="Confirm API availability and explain how to get started.",
        )
        .add_case(
            query="How do I add a new user to my team?",
            expected_results="Explain the user invitation and onboarding process.",
        )
        .publish()
    )

    report = (
        client.evaluations.run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": display,
                "framework": "autogen",
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
