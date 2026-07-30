from typing import List, Optional
import requests
import os
import logging

from agentx.util import get_headers, api_base
from agentx.resources.agent import Agent
from agentx.resources.workforce import Workforce


class AgentX:

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("AGENTX_API_KEY")
        if self.api_key and not os.getenv("AGENTX_API_KEY"):
            os.environ["AGENTX_API_KEY"] = self.api_key

        # base_url overrides AGENTX_API_BASE_URL env var (and the SDK default)
        self.base_url = base_url or os.getenv("AGENTX_API_BASE_URL")
        if self.base_url:
            os.environ["AGENTX_API_BASE_URL"] = self.base_url

        self.workspace_id = workspace_id or os.getenv("AGENTX_WORKSPACE_ID")

        from agentx.evaluations.client import EvaluationsClient
        from agentx.evaluations.runner import EvaluationsRunner
        from agentx.monitor.client import MonitorClient
        from agentx.tracing.ingest_client import IngestClient
        from agentx.tracing.tracer import Tracer
        from agentx.version import VERSION

        _eval_client = EvaluationsClient(
            api_key=self.api_key,
            sdk_version=VERSION,
            base_url=self.base_url,
            workspace_id=self.workspace_id,
        )
        self.evaluations = EvaluationsRunner(_eval_client)

        # Monitor: create/reuse patterns (client.monitor.patterns) that a trace can be checked
        # against at send time via tracer.trace(..., monitor=True, pattern_ids=[...]), then read
        # back the resulting alerts/findings with client.monitor.signals.
        self.monitor = MonitorClient(
            api_key=self.api_key,
            sdk_version=VERSION,
            base_url=self.base_url,
            workspace_id=self.workspace_id,
        )

        _ingest_client = IngestClient(
            api_key=self.api_key,
            sdk_version=VERSION,
            base_url=self.base_url,
            workspace_id=self.workspace_id,
        )
        self.tracer = Tracer(_ingest_client)

    @classmethod
    def from_env(cls) -> "AgentX":
        """Create an AgentX client using AGENTX_API_KEY (and optionally AGENTX_API_BASE_URL) from the environment."""
        return cls()

    def get_agent(self, id: str) -> Agent:
        url = f"{api_base()}/access/agents/{id}"
        # Make a GET request to the AgentX API
        response = requests.get(url, headers=get_headers())
        # Check if response was successful
        if response.status_code == 200:
            return Agent(**response.json())
        else:
            raise Exception(f"Failed to retrieve agent: {response.reason}")

    def list_agents(self) -> List[Agent]:
        url = f"{api_base()}/access/agents"
        # Make a GET request to the AgentX API
        response = requests.get(url, headers=get_headers())
        # Check if response was successful
        if response.status_code == 200:
            return [Agent(**agent) for agent in response.json()]
        else:
            raise Exception(f"Failed to list agents: {response.reason}")

    @staticmethod
    def list_workforces() -> List["Workforce"]:
        """List all workforces/teams."""
        url = f"{api_base()}/access/teams"
        response = requests.get(url, headers=get_headers())
        if response.status_code == 200:
            return [Workforce(**workforce) for workforce in response.json()]
        else:
            raise Exception(
                f"Failed to list workforces: {response.status_code} - {response.reason}"
            )

    def get_profile(self):
        """Get the current user's profile information."""
        url = f"{api_base()}/access/getProfile"
        response = requests.get(url, headers=get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Failed to get profile: {response.status_code} - {response.reason}"
            )
