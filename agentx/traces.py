from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from agentx.util import api_base, get_headers

logger = logging.getLogger(__name__)


class AgentXTracesError(Exception):
    pass


class TracesClient:
    """Surfaced as ``client.traces``: the READ side of tracing (``client.tracer`` writes).

    ``get(trace_id)`` returns one trace's full detail (input/output/error, model, latency,
    token counts incl. cache, session/span linkage, metadata, estimated cost) - the same wire
    the dashboard's trace dialog reads. ``list()`` pages through the project's traces newest
    first. For a whole conversation, ``client.monitor.sessions.spans(session_id)`` remains the
    span-tree read.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._api_key = api_key
        # Captured once at construction (deep-dive round 3, bug #1).
        self._base_url = (base_url or api_base()).rstrip("/")

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        resp = requests.get(
            f"{self._base_url}{path}",
            headers=get_headers(self._api_key),
            params=params or {},
            timeout=15,
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.reason)
            except ValueError:
                detail = resp.reason
            raise AgentXTracesError(f"Trace request failed ({resp.status_code}): {detail}")
        return resp.json()

    def get(self, trace_id: str) -> Dict[str, Any]:
        """One trace's detail row. Raises on 404."""
        return self._request(f"/ingest/traces/{trace_id}")

    def list(
        self,
        limit: int = 50,
        cursor: Optional[str] = None,
        framework: Optional[str] = None,
    ) -> Dict[str, Any]:
        """A page of traces, newest first: ``{"traces": [...], "nextCursor": str | None}``.
        Pass ``cursor`` from the previous page to continue."""
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if framework:
            params["framework"] = framework
        return self._request("/ingest/traces", params)
