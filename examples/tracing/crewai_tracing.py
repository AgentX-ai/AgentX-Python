"""
CrewAI Production Tracing
--------------------------
Two patterns:

  1. observer.kickoff()  — wraps crew.kickoff(), auto-captures task outputs as tool_calls
  2. observer.observe()  — manual context manager for full control

Run:
    pip install agentx[crewai]
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... python examples/tracing/crewai_tracing.py
"""

from agentx import AgentX
from agentx.integrations.crewai import AgentXCrewObserver

agentx = AgentX.from_env()
observer = AgentXCrewObserver(
    tracer=agentx.tracer,
    name="support-crew",
    metadata={"env": "production"},
)


def build_crew():
    from crewai import Agent, Task, Crew

    researcher = Agent(
        role="Policy Researcher",
        goal="Find relevant company policies",
        backstory="You look up policies from the knowledge base.",
        verbose=False,
    )
    writer = Agent(
        role="Support Writer",
        goal="Write clear, friendly support responses",
        backstory="You turn policy details into helpful customer replies.",
        verbose=False,
    )

    research_task = Task(
        description="Look up the cancellation policy.",
        expected_output="A concise policy statement.",
        agent=researcher,
    )
    reply_task = Task(
        description="Write a support reply about cancelling a subscription.",
        expected_output="A friendly, accurate response to send to the customer.",
        agent=writer,
    )

    return Crew(agents=[researcher, writer], tasks=[research_task, reply_task], verbose=False)


def main():
    try:
        crew = build_crew()
    except ImportError:
        print("Missing dependency. Run: pip install agentx[crewai]")
        return

    print("Running CrewAI crew with AgentX tracing (pattern 1: observer.kickoff)...")
    result = observer.kickoff(crew, inputs={"query": "cancel subscription"})
    print(f"Result: {getattr(result, 'raw', result)}\n")

    print("Running again with pattern 2: observer.observe() context manager...")
    with observer.observe(name="support-crew-manual", input={"query": "refund policy"}) as span:
        raw = crew.kickoff(inputs={"query": "refund policy"})
        span.output = getattr(raw, "raw", str(raw))
    print(f"Result: {span.output}\n")

    agentx.tracer.flush(timeout=10)
    print("Done — traces are live in the AgentX dashboard.")


if __name__ == "__main__":
    main()
