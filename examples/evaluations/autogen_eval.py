"""
AutoGen / AG2 Agent Evaluation
--------------------------------
Evaluates an AutoGen (or AG2) agent without adding it as a hard SDK dependency.
Works with both AutoGen v0.2 (autogen-agentchat) and AG2 (ag2 package).

Run:
    pip install pyautogen   # or: pip install ag2
    AGENTX_API_KEY=your_key OPENAI_API_KEY=sk-... python examples/evaluations/autogen_eval.py
"""

import os
from agentx import AgentX
from agentx.evaluations.models import EvaluationCase


def build_autogen_agent():
    """Returns a callable that runs an AutoGen agent for a given query."""
    try:
        # Works with both pyautogen and ag2
        try:
            from autogen import ConversableAgent
        except ImportError:
            from ag2 import ConversableAgent

        llm_config = {
            "config_list": [
                {
                    "model": "gpt-4o-mini",
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                }
            ],
            "temperature": 0,
        }

        assistant = ConversableAgent(
            name="SupportAgent",
            system_message="You are a helpful customer support agent. Answer concisely.",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )

        user_proxy = ConversableAgent(
            name="UserProxy",
            llm_config=False,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
        )

        def run_agent(query: str):
            chat_result = user_proxy.initiate_chat(assistant, message=query, max_turns=1, silent=True)
            messages = assistant.chat_messages.get(user_proxy, [])
            output = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    output = msg.get("content", "")
                    break
            # best-effort token extraction from chat cost summary
            cost = getattr(chat_result, "cost", None) or {}
            usage = cost.get("usage_including_cached_inference", {}) if isinstance(cost, dict) else {}
            total = usage.get("total", {}) if isinstance(usage, dict) else {}
            input_tokens = total.get("prompt_tokens")
            output_tokens = total.get("completion_tokens")
            return output, input_tokens, output_tokens

        return run_agent

    except ImportError:
        return None


def make_eval_fn(autogen_runner):
    def eval_subject(case: EvaluationCase) -> dict:
        if autogen_runner is None:
            return {
                "output": f"[stub] AutoGen response to: {case.query}",
                "metadata": {"framework": "autogen"},
            }

        output, input_tokens, output_tokens = autogen_runner(case.query)
        return {
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "metadata": {"framework": "autogen", "model": "gpt-4o-mini"},
        }
    return eval_subject


def main():
    client = AgentX.from_env()
    autogen_runner = build_autogen_agent()

    dataset = (
        client.evaluations.datasets
        .builder(
            name="AutoGen Support Agent Dataset",
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
        .publish()
    )

    print(f"Dataset: {dataset.id}")

    report = (
        client.evaluations
        .run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": "AutoGen Support Agent",
                "framework": "autogen",
                "runtime": "local",
            },
        )
        .execute(make_eval_fn(autogen_runner))
        .finalize()
        .analyze()
    )

    print(f"\nDashboard: {report.dashboard_url}")


if __name__ == "__main__":
    main()
