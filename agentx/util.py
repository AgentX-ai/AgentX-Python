import os

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


def get_headers(api_key: str = None):
    return {"accept": "*/*", "x-api-key": api_key or os.getenv("AGENTX_API_KEY")}
