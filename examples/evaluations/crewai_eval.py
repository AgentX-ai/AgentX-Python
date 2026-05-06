"""
CrewAI Agent Evaluation
------------------------
Covers two modes selectable via CREWAI_MODE env var:

  CREWAI_MODE=single  (default) — single specialist agent
  CREWAI_MODE=multi              — researcher + writer multi-agent crew

Run:
    pip install crewai crewai-tools openai
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... python examples/evaluations/crewai_eval.py
    AGENTX_API_KEY=key OPENAI_API_KEY=sk-... CREWAI_MODE=multi python examples/evaluations/crewai_eval.py
"""

import os

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase


# ---------------------------------------------------------------------------
# Mode: single specialist agent
# ---------------------------------------------------------------------------

def build_single_crew():
    try:
        from crewai import Agent, Task, Crew, Process

        support_agent = Agent(
            role="Customer Support Specialist",
            goal="Provide accurate, helpful answers to customer questions.",
            backstory="Expert in product knowledge and customer communication.",
            verbose=False,
            allow_delegation=False,
        )

        def run(query: str):
            task = Task(
                description=query,
                agent=support_agent,
                expected_output="A clear, concise customer support response.",
            )
            crew = Crew(agents=[support_agent], tasks=[task], process=Process.sequential, verbose=False)
            result = crew.kickoff()
            return result, str(result), []

        return run
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Mode: multi-agent crew (researcher + writer)
# ---------------------------------------------------------------------------

def build_multi_crew():
    try:
        from crewai import Agent, Task, Crew, Process

        researcher = Agent(
            role="Policy Researcher",
            goal="Find accurate policy information to answer customer questions.",
            backstory="Specialises in locating and summarising company policy details.",
            verbose=False,
            allow_delegation=False,
        )
        writer = Agent(
            role="Support Writer",
            goal="Craft clear, professional customer-facing responses from policy research.",
            backstory="Expert at translating policy details into friendly, actionable answers.",
            verbose=False,
            allow_delegation=False,
        )

        def run(query: str):
            research_task = Task(
                description=f"Research the relevant policy for this customer question: {query}",
                agent=researcher,
                expected_output="Key policy facts relevant to the question.",
            )
            write_task = Task(
                description="Write a concise, friendly customer support response based on the research.",
                agent=writer,
                expected_output="A professional customer support response.",
                context=[research_task],
            )
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, write_task],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff()

            # Build trace events from task outputs
            trace_events = [
                {"type": "agent_step", "name": "policy_researcher", "summary": str(research_task.output)[:200] if research_task.output else ""},
                {"type": "agent_step", "name": "support_writer", "summary": str(result)[:200]},
            ]

            return result, str(result), trace_events

        return run
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Shared eval function factory
# ---------------------------------------------------------------------------

def make_eval_fn(crew_runner, mode: str):
    def eval_subject(case: EvaluationCase) -> dict:
        if crew_runner is None:
            return {
                "output": f"[stub] CrewAI ({mode}) response to: {case.query}",
                "metadata": {"framework": "crewai", "mode": mode},
            }

        result_obj, output, trace_events = crew_runner(case.query)  # type: ignore[misc]
        usage = getattr(result_obj, "usage_metrics", None)

        return {
            "output": output,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "trace": {"events": trace_events} if trace_events else None,
            "metadata": {"framework": "crewai", "mode": mode},
        }

    return eval_subject


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    client = AgentX.from_env()
    mode = os.getenv("CREWAI_MODE", "single").lower()

    crew_runner = build_multi_crew() if mode == "multi" else build_single_crew()
    eval_fn = make_eval_fn(crew_runner, mode)
    display = "CrewAI Multi-Agent Crew" if mode == "multi" else "CrewAI Single Specialist"

    dataset = (
        client.evaluations.datasets
        .builder(
            name=f"CrewAI Support Agent Dataset ({mode})",
            description="Evaluates a CrewAI agent on team account and integration queries.",
            number_of_requests=2,
            acceptance_criteria="Responses must be accurate and address the customer's question directly.",
            rejection_criteria="No hallucinated product features or policies.",
        )
        .add_case(
            query="How do I add a team member to my account?",
            expected_results="Explain the steps to invite and add a new team member.",
        )
        .add_case(
            query="What integrations do you support?",
            expected_results="List the available integrations accurately.",
        )
        .add_case(
            query="Can I set different permission levels for team members?",
            expected_results="Describe available permission levels and how to assign them.",
        )
        .publish()
    )

    report = (
        client.evaluations
        .run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": display,
                "framework": "crewai",
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
