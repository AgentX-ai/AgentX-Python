"""
HTTP Endpoint Agent Evaluation
--------------------------------
Uses the HttpEndpointAdapter to evaluate any agent exposed as an HTTP endpoint
— your own FastAPI service, a LangServe deployment, a local n8n webhook, etc.

The SDK calls your endpoint once per evaluation case, passing the query as JSON,
and reads the response text as the agent output.

Expected endpoint contract:
    POST /your-endpoint
    Content-Type: application/json
    Body: { "query": "..." }
    Response: { "output": "..." }   # or any JSON — SDK reads .output or full text

Run:
    AGENTX_API_KEY=your_key python examples/evaluations/http_endpoint_eval.py
"""

from agentx import AgentX
from agentx.evaluations.adapters.http_endpoint import HttpEndpointAdapter


def main():
    client = AgentX.from_env()

    # Replace with your real endpoint URL
    endpoint_url = "http://localhost:8000/agent/invoke"

    adapter = HttpEndpointAdapter(
        url=endpoint_url,
        headers={"Authorization": "Bearer your-service-token"},
        timeout=30,
    )

    dataset = (
        client.evaluations.datasets.builder(
            name="HTTP Endpoint Agent Dataset",
            description="Evaluates an agent deployed as an HTTP service, called once per case by the SDK.",
            number_of_requests=2,
            acceptance_criteria="Responses must be accurate and address the query directly.",
            rejection_criteria="No empty responses, no server errors passed through as answers.",
        )
        .add_case(
            query="Summarize our Q3 performance highlights.",
            expected_results="A concise summary covering key metrics and notable achievements from Q3.",
        )
        .add_case(
            query="What are the action items from last week's meeting?",
            expected_results="A list of specific, actionable items assigned to team members.",
        )
        .publish()
    )

    print(f"Dataset: {dataset.id}")

    report = (
        client.evaluations.run(
            dataset_id=dataset.id,
            subject={
                "kind": "custom_agent",
                "displayName": "My HTTP Agent Service",
                "framework": "other",
                "runtime": "customer_hosted",
                "endpoint": endpoint_url,
            },
        )
        .execute(adapter)
        .finalize()
        .analyze()
    )

    print(f"\nDashboard: {report.dashboard_url}")


if __name__ == "__main__":
    main()
