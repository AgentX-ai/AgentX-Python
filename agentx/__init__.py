import logging

from agentx.agentx import AgentX
from agentx.version import VERSION
from agentx.exceptions import (
    AgentXError,
    AgentXAuthError,
    AgentXAPIError,
    DatasetNotFound,
    CINotEnabled,
    CIRunExpired,
    CIGateFailure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
)

__all__ = [
    "AgentX",
    "AgentXError",
    "AgentXAuthError",
    "AgentXAPIError",
    "DatasetNotFound",
    "CINotEnabled",
    "CIRunExpired",
    "CIGateFailure",
]
__version__ = VERSION
