import os
from typing import Optional

_DEFAULT_API_BASE = "https://api.agentx.so/api/v1"


def api_base() -> str:
    """Return the base URL for all AgentX API calls, respecting AGENTX_API_BASE_URL if set."""
    override = os.getenv("AGENTX_API_BASE_URL", "").rstrip("/")
    if override:
        # Strip the evaluations-specific suffix if present so the override works for all routes
        suffix = "/custom-agent-evaluations"
        if override.endswith(suffix):
            override = override[: -len(suffix)]
        return override
    return _DEFAULT_API_BASE


def get_headers(api_key: Optional[str] = None):
    return {"accept": "*/*", "x-api-key": api_key or os.getenv("AGENTX_API_KEY")}


def endpoint_missing(resp) -> bool:
    """True when a 404 means "this deployment has no such route", not "your object is gone".

    The two are indistinguishable by status code alone, and telling a user their dataset does not
    exist when it does - because the route behind the call was never implemented here - sends them
    debugging the wrong thing entirely.

    A handler that looked for something and did not find it says so in the body: the self-host
    engine answers ``{"error": "Dataset not found"}``, the hosted API ``{"message": "Dataset not
    found"}``. An unmatched route does not: the engine's catch-all answers ``{"statusCode": 404,
    "message": "Not found"}`` (see its index.ts) and a bare Express 404 is HTML. So: a 404 whose
    body carries no specific message is a missing route; anything else is a missing object, and
    keeps whatever meaning the caller already had for it.
    """
    if resp.status_code != 404:
        return False
    try:
        body = resp.json()
    except Exception:
        return True  # HTML or empty - no handler wrote this
    if not isinstance(body, dict):
        return True
    if body.get("error"):
        return False  # a handler's own "<thing> not found"
    message = body.get("message")
    return not message or str(message).strip().lower() == "not found"
