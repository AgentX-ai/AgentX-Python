"""AgentX SDK exceptions."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentx.tracing.ci_types import CIRunResult


class AgentXError(Exception):
    """Base class for all AgentX SDK errors."""


class AgentXAuthError(AgentXError):
    """Invalid or missing API key."""


class AgentXAPIError(AgentXError):
    """Unexpected API error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class EndpointNotAvailable(AgentXError):
    """The deployment this SDK is pointed at does not serve the endpoint the call needs.

    Raised instead of a bare HTTP 404 when the route itself is missing, rather than the object
    it addresses. The usual cause is pointing AGENTX_API_BASE_URL at a self-host engine
    (AgentX-ai/AgentX-trace-eval) and calling a part of the SDK that only the hosted API
    implements - list_models, evaluate_trace, or the CI-run surface behind run_eval. The
    message names the call and the base URL, because "404" on its own is indistinguishable
    from a typo in an id.
    """

    def __init__(self, call: str | None, method: str, url: str) -> None:
        subject = f"{call} is not available here" if call else "This deployment has no such endpoint"
        super().__init__(
            f"{subject}: {method} {url} returned 404 with no handler behind it. Pointing "
            f"AGENTX_API_BASE_URL at a self-host engine is the usual cause - see its README "
            f"for which parts of the SDK it serves."
        )
        self.call = call
        self.method = method
        self.url = url


class DatasetNotFound(AgentXError):
    """Dataset ID does not exist or is not accessible from this API key."""


class CINotEnabled(AgentXError):
    """Dataset exists but ci.enabled is false. Enable CI in the dataset settings."""


class CIRunExpired(AgentXError):
    """CI run was not finalized within the 2-hour window."""


class CIGateFailure(AgentXError):
    """Gate result is 'fail'. Raised when fail_on_gate=True."""

    def __init__(self, result: "CIRunResult") -> None:
        super().__init__(
            f"CI gate failed: {result.pass_rate:.0%} passed "
            f"({result.passed_questions}/{result.total_questions} questions)"
        )
        self.result = result
