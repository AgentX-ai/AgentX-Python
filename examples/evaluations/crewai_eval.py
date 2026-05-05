"""
CrewAI Agent Evaluation
------------------------
Evaluates a CrewAI agent (crew or individual agent) without adding CrewAI
as a hard dependency of the AgentX SDK itself.

Run:
    pip install crewai crewai-tools openai
    AGENTX_API_KEY=your_key OPENAI_API_KEY=sk-... python examples/evaluations/crewai_eval.py
"""

from agentx import AgentX
from agentx.evaluations.models import EvaluationCase


def build_crew():
    """Build and return a CrewAI crew. Returns None if crewai is not installed."""
    try:
        from crewai import Agent, Task, Crew, Process

        support_agent = Agent(
            role="Customer Support Specialist",
            goal="Provide accurate, helpful answers to customer questions.",
            backstory="Expert in product knowledge and customer communication.",
            verbose=False,
            allow_delegation=False,
        )

        def run_crew(query: str):
            task = Task(
                description=query,
                agent=support_agent,
                expected_output="A clear, concise customer support response.",
            )
            crew = Crew(
                agents=[support_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff()
            return result, str(result)

        return run_crew
    except ImportError:
        return None


def make_eval_fn(crew_runner):
    def eval_subject(case: EvaluationCase) -> dict:
        if crew_runner is None:
            return {
                "output": f"[stub] CrewAI response to: {case.query}",
                "metadata": {"framework": "crewai"},
            }

        result_obj, output = crew_runner(case.query)  # type: ignore[misc]
        usage = getattr(result_obj, "usage_metrics", None)
        return {
            "output": output,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "metadata": {"framework": "crewai"},
        }
    return eval_subject


def main():
    client = AgentX.from_env()
    crew_runner = build_crew()

    dataset = (
        client.evaluations.datasets
        .builder(
            name="CrewAI Support Agent Dataset",
            description="Evaluates a CrewAI specialist agent on team account and integration queries.",
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
        .publish()
    )

    print(f"Dataset: {dataset.id}")

    report = (
        client.evaluations
        .run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": "CrewAI Support Specialist",
                "framework": "crewai",
                "runtime": "local",
            },
        )
        .execute(make_eval_fn(crew_runner))
        .finalize()
        .analyze()
    )

    print(f"\nDashboard: {report.dashboard_url}")


if __name__ == "__main__":
    main()
