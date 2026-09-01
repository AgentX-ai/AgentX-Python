"""Best-effort agent-framework auto-detection (the "platform agnostic" story).

A span whose framework was neither passed explicitly (``tracer.trace(...,
framework="...")``) nor adopted from a framework integration (callback handler,
observer, patched client - see ``_TraceSpan._captured_framework``) gets labeled
by looking at which known ORCHESTRATION framework is actually imported in this
process. ``sys.modules`` is the signal - imported, not merely installed - so a
machine with ten frameworks pip-installed but one in use still resolves.

Only unambiguous answers are given: zero or more than one known framework
loaded means ``None``, and the span goes out unlabeled rather than mislabeled.
The user's explicit ``framework=`` always wins, including totally custom names
for platforms this table has never heard of.

Raw provider SDKs (openai, anthropic, google-genai, ...) are deliberately NOT
in this table: they are transitive dependencies of nearly every framework, so
their presence says nothing about what orchestrates the agent - and their
patched-client integrations already stamp the provider literal on the spans
they create.
"""

from __future__ import annotations

import sys
from typing import Optional

# Top-level module name -> the wire literal the matching integration emits.
# Multiple modules may map to one literal (langgraph is the LangChain family).
_ORCHESTRATOR_MODULES = {
    "langchain": "langchain",
    "langchain_core": "langchain",
    "langgraph": "langchain",
    "crewai": "crewai",
    "llama_index": "llamaindex",
    "autogen": "autogen",
    "autogen_agentchat": "autogen",
    "agents": "openai-agents",  # the OpenAI Agents SDK's import name
    "google.adk": "google-adk",
    "semantic_kernel": "semantic-kernel",
    "haystack": "haystack",
    "pydantic_ai": "pydantic-ai",
    "smolagents": "smolagents",
    "dspy": "dspy",
}


def _looks_like_openai_agents_sdk() -> bool:
    # "agents" is a name any user package could claim - only trust it when the
    # OpenAI Agents SDK's own submodules are loaded alongside it.
    return "agents.run" in sys.modules or "agents.tracing" in sys.modules


def detect_framework() -> Optional[str]:
    """The single unambiguous orchestration framework imported right now, else None."""
    found: set = set()
    for module, literal in _ORCHESTRATOR_MODULES.items():
        if module not in sys.modules:
            continue
        if module == "agents" and not _looks_like_openai_agents_sdk():
            continue
        found.add(literal)
        if len(found) > 1:
            return None
    return found.pop() if len(found) == 1 else None
