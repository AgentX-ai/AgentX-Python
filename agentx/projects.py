from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from agentx.util import api_base, get_headers

logger = logging.getLogger(__name__)


class AgentXProjectsError(Exception):
    pass


class ProjectsClient:
    """Surfaced as ``client.projects``: create, list, and delete the engine's projects
    (self-host). Each project is a fully isolated tenant - own API key, own traces, scorers,
    datasets, and settings. ``create()`` returns the new project's ``apiKey``; construct a new
    ``AgentX(api_key=...)`` with it to work inside that project (the pattern integration tests
    use for per-run isolation).

    In ``AGENTX_AUTH=enabled`` mode project management is session-scoped to signed-in dashboard
    users; this client covers the default self-host (auth-disabled) mode.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key

    def _request(self, method: str, path: str, json: Any = None) -> Any:
        resp = requests.request(
            method,
            f"{api_base()}{path}",
            headers={**get_headers(self._api_key), "Content-Type": "application/json"},
            json=json,
            timeout=15,
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.reason)
            except ValueError:
                detail = resp.reason
            raise AgentXProjectsError(f"Projects request failed ({resp.status_code}): {detail}")
        return resp.json() if resp.text else {}

    def create(self, name: str) -> Dict[str, Any]:
        """Create a project; the returned dict includes ``_id``, ``name``, and ``apiKey``."""
        return self._request("POST", "/projects", json={"name": name})["project"]

    def list(self) -> List[Dict[str, Any]]:
        """All projects on the instance, each with its ``apiKey`` and ``isDefault`` flag."""
        return self._request("GET", "/projects").get("projects", [])

    def delete(self, project_id: str) -> None:
        """Delete a project and every row it owns (traces, scorers, datasets, runs). The
        default project cannot be deleted. Irreversible."""
        self._request("DELETE", f"/projects/{project_id}")
