"""
OpenAI Agents SDK Production Tracing
--------------------------------------
Registers AgentXTracingProcessor as a global tracing processor.
Every agent run in the process then emits one trace to AgentX automatically.

Run:
    pip install agentx[openai-agents] openai-agents
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... python examples/tracing/openai_agents_tracing.py
"""

from agentx import AgentX
from agentx.integrations.openai_agents import AgentXTracingProcessor

agentx = AgentX.from_env()


def main():
    try:
        from agents import Agent, Runner, add_trace_processor
        from agents.tools import function_tool
    except ImportError:
        print("Missing dependency. Run: pip install agentx[openai-agents] openai-agents")
        return

    # Register once — affects every agent run in this process
    processor = AgentXTracingProcessor(
        tracer=agentx.tracer,
        metadata={"env": "production"},
        session_id="session-openai-001",
    )
    add_trace_processor(processor)

    @function_tool
    def get_policy(topic: str) -> str:
        """Return the company policy for a given topic."""
        db = {
            "cancel": "Go to Account → Subscription → Cancel.",
            "trial": "14-day free trial, no credit card required.",
            "refund": "Full refund within 30 days.",
        }
        return db.get(topic.lower(), "No policy found.")

    agent = Agent(
        name="support-agent",
        instructions="You are a helpful support agent. Use get_policy to look up policies.",
        tools=[get_policy],
    )

    print("Running OpenAI Agent with AgentX tracing...")
    result = Runner.run_sync(agent, "How do I cancel my subscription?")
    print(f"Result: {result.final_output}\n")

    agentx.tracer.flush(timeout=10)
    print("Done — traces are live in the AgentX dashboard.")


if __name__ == "__main__":
    main()
