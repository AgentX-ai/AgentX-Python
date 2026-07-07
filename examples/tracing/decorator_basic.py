"""
Basic Production Tracing — Decorator & Context Manager
-------------------------------------------------------
Demonstrates the two core usage patterns for AgentX production tracing:

  1. @agentx.tracer.trace  — decorate any function to auto-capture input/output
  2. tracer.trace(...) as span  — context manager for fine-grained control

Run:
    AGENTX_API_KEY=key python examples/tracing/decorator_basic.py
"""

import os
from agentx import AgentX

agentx = AgentX.from_env()


# ---------------------------------------------------------------------------
# Pattern 1: decorator — input/output captured automatically
# ---------------------------------------------------------------------------

@agentx.tracer.trace("customer-support-agent")
def answer_question(query: str, user_id: str) -> str:
    """Stub: replace with your real agent logic."""
    return f"Hello! You asked: '{query}'. Here is your answer."


# ---------------------------------------------------------------------------
# Pattern 2: context manager — attach tool calls and custom metadata
# ---------------------------------------------------------------------------

def run_with_tools(query: str) -> str:
    tool_results = []

    with agentx.tracer.trace(
        "customer-support-with-tools",
        input={"query": query},
        metadata={"version": "v2"},
        session_id="session-abc-123",
    ) as span:
        # Simulate a tool call
        kb_result = f"KB result for: {query}"
        tool_results.append({
            "name": "knowledge_base_lookup",
            "input": {"query": query},
            "output": kb_result,
        })

        answer = f"Based on our KB: {kb_result}"
        span.output = answer
        span.tool_calls = tool_results

    return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Sending traces to AgentX...")

    result1 = answer_question("How do I cancel my subscription?", user_id="u-001")
    print(f"[decorator]        {result1!r}")

    result2 = run_with_tools("What is the refund policy?")
    print(f"[context manager]  {result2!r}")

    # Flush ensures background thread has delivered all traces before exit
    agentx.tracer.flush(timeout=10)
    print("Done — check the AgentX dashboard for live traces.")


if __name__ == "__main__":
    main()
