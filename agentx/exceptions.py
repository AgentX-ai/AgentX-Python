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
