import logging

from agentx.agentx import AgentX
from agentx.version import VERSION
from agentx.exceptions import (
    AgentXError,
    AgentXAuthError,
    AgentXValidationError,
    AgentXConnectionError,
    AgentXAPIError,
    DatasetNotFound,
    CINotEnabled,
    CIRunExpired,
    CIGateFailure,
)

# Library logging hygiene: a library must never call logging.basicConfig - it hijacks the
# host application's root logger (format AND level) and turns the app's own later basicConfig
# into a no-op. Consumers opt into our logs with logging.getLogger("agentx").setLevel(...).
logging.getLogger("agentx").addHandler(logging.NullHandler())

__all__ = [
    "AgentX",
    "AgentXError",
    "AgentXAuthError",
    "AgentXValidationError",
    "AgentXConnectionError",
    "AgentXAPIError",
    "DatasetNotFound",
    "CINotEnabled",
    "CIRunExpired",
    "CIGateFailure",
]
__version__ = VERSION
