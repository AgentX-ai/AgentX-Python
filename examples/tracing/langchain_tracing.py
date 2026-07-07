"""
LangChain Production Tracing
-----------------------------
Auto-traces every LangChain chain / agent run using AgentXCallbackHandler.
Each top-level chain invocation sends one trace with nested tool calls attached.

Run:
    pip install agentx[langchain] langchain langchain-openai
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... python examples/tracing/langchain_tracing.py
"""

from agentx import AgentX

agentx = AgentX.from_env()


def build_support_agent():
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.prompts import PromptTemplate
    from agentx.integrations.langchain import AgentXCallbackHandler

    @tool
    def policy_lookup(topic: str) -> str:
        """Look up a company policy by topic (cancel, trial, refund)."""
        db = {
            "cancel": "Go to Account → Subscription → Cancel.",
            "trial": "14-day free trial, no credit card required.",
            "refund": "Full refund within 30 days. Email support@example.com.",
        }
        for key, val in db.items():
            if key in topic.lower():
                return val
        return "No policy found. Please contact support."

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [policy_lookup]
    prompt = PromptTemplate.from_template(
        "You are a helpful support agent.\n\n"
        "Tools available:\n{tools}\n\nTool names: {tool_names}\n\n"
        "Question: {input}\nThought: {agent_scratchpad}"
    )
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # Attach the AgentX callback — traces are sent automatically
    handler = AgentXCallbackHandler(
        tracer=agentx.tracer,
        session_id="prod-session-001",
    )
    return executor, handler


def main():
    print("Running LangChain agent with AgentX tracing...")
    try:
        executor, handler = build_support_agent()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install agentx[langchain] langchain langchain-openai")
        return

    queries = [
        "How do I cancel my subscription?",
        "Is there a free trial?",
        "What is the refund policy?",
    ]

    for query in queries:
        result = executor.invoke({"input": query}, config={"callbacks": [handler]})
        print(f"Q: {query}")
        print(f"A: {result['output']}\n")

    agentx.tracer.flush(timeout=10)
    print("Done — traces are live in the AgentX dashboard.")


if __name__ == "__main__":
    main()
