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
        # back the resulting alerts/findings with client.monitor.signals. Per-agent coverage and
        # detection settings (sample rate, retention, threshold overrides like the built-in
        # "Latency regression" pattern's threshold) are client.monitor.profile.
        self.monitor = MonitorClient(
            api_key=self.api_key,
            sdk_version=VERSION,
            base_url=self.base_url,
            workspace_id=self.workspace_id,
        )

        from agentx.outcomes import OutcomesClient

        # Report real, after-the-fact outcomes ("the ticket got reopened") against traces - the
        # ground truth behind the dashboard's Judge Calibration card. Self-host only.
        self.outcomes = OutcomesClient(api_key=self.api_key)

        from agentx.feedback import FeedbackClient

        # Forward end-user votes ("up"/"down") on traced responses - a "down" raises a signal
        # directly, and every vote feeds Judge Calibration alongside outcomes. Self-host only.
        self.feedback = FeedbackClient(api_key=self.api_key)

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

    def ping(self) -> dict:
        """Verify the client can actually reach AgentX and that the API key is accepted.

        The constructor is deliberately lazy (no network call - standard SDK behavior, so
        offline construction and tests work), and trace delivery is fire-and-forget, so a
        wrong ``base_url`` or ``api_key`` otherwise surfaces only as a one-time warning in
        logs while traces silently go nowhere. Call this once at startup of a long-running
        service to fail fast instead::

            client = AgentX.from_env()
            client.ping()  # raises immediately on a bad URL or key

        Raises :class:`agentx.exceptions.AgentXConnectionError` when the URL is unreachable,
        :class:`agentx.exceptions.AgentXAuthError` when the key is rejected, and
        :class:`agentx.exceptions.AgentXAPIError` on any other non-OK response. Returns
        ``{"ok": True, "base_url": ...}`` on success.
        """
        from agentx.exceptions import AgentXAPIError, AgentXAuthError, AgentXConnectionError

        base = api_base()
        # /monitor/patterns: the cheapest key-authenticated endpoint that exists on both the
        # hosted API and the self-host engine's SDK-facing router.
        url = f"{base}/monitor/patterns"
        try:
            response = requests.get(url, headers=get_headers(self.api_key), timeout=10)
        except requests.RequestException as exc:
            raise AgentXConnectionError(
                f"Cannot reach AgentX at {base} ({exc.__class__.__name__}: {exc}). "
                "Check base_url / AGENTX_API_BASE_URL - for self-host it should look like "
                "http://localhost:4700/api/v1."
            ) from exc
        if response.status_code in (401, 403):
            raise AgentXAuthError(
                f"AgentX at {base} rejected the API key (HTTP {response.status_code}). "
                "Check api_key / AGENTX_API_KEY - for self-host, copy the 'Default project "
                "API key' from the engine's startup log."
            )
        if not response.ok:
            raise AgentXAPIError(
                f"AgentX at {base} responded HTTP {response.status_code} to the health probe.",
                status_code=response.status_code,
            )
        return {"ok": True, "base_url": base}

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
