import os
from typing import Optional

_DEFAULT_API_BASE = "https://api.agentx.so/api/v1"

_EVALUATIONS_SUFFIX = "/custom-agent-evaluations"


def normalize_base(base: str) -> str:
    """Normalize a user-supplied API base URL so it works for ALL routes: strip a trailing
    slash, and strip the evaluations-specific ``/custom-agent-evaluations`` suffix (users
    copying the eval endpoint out of a dashboard/env file otherwise 404 every non-eval
    sub-client - monitor, outcomes, traces, ingest, ...)."""
    base = base.rstrip("/")
    if base.endswith(_EVALUATIONS_SUFFIX):
        base = base[: -len(_EVALUATIONS_SUFFIX)]
    return base


def api_base() -> str:
    """Return the base URL for all AgentX API calls, respecting AGENTX_API_BASE_URL if set."""
    override = normalize_base(os.getenv("AGENTX_API_BASE_URL", ""))
    if override:
        return override
    return _DEFAULT_API_BASE


def get_headers(api_key: Optional[str] = None):
    return {"accept": "*/*", "x-api-key": api_key or os.getenv("AGENTX_API_KEY")}
