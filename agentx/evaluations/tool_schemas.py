"""Tool schema registry client, surfaced as ``client.evaluations.tool_schemas``.

The tool-definition analog of the prompt registry: register the JSON definition your agent
actually passes to its LLM, let production failures accumulate as evidence against it, then
propose -> validate -> publish improved versions. See the dashboard's Improve > Tools & MCPs.
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agentx.evaluations.client import EvaluationsClient


class ToolSchemaClient:
    def __init__(self, client: "EvaluationsClient"):
        self._client = client

    def list(self) -> List[dict]:
        return self._client.list_tool_schemas()

    def create(self, *, name: str, definition: str, description: Optional[str] = None) -> dict:
        return self._client.create_tool_schema(name=name, definition=definition, description=description)

    def get_or_create(self, *, name: str, definition: str, description: Optional[str] = None) -> dict:
        """Idempotent register: returns the existing schema of this name if present."""
        existing = next((t for t in self.list() if t.get("name") == name), None)
        if existing:
            return existing
        return self.create(name=name, definition=definition, description=description)

    def examples(self, tool_schema_id: str, window: Optional[str] = None) -> dict:
        """Failure evidence recorded against this tool (agent-tool-failure signals, low-rated
        eval results that called it) - what propose() rewrites from."""
        return self._client.get_tool_schema_examples(tool_schema_id, window=window)

    def propose(self, tool_schema_id: str, window: Optional[str] = None) -> dict:
        """Judge-written definition rewrite grounded in examples(). Nothing is published."""
        return self._client.propose_tool_schema(tool_schema_id, window=window)

    def publish_version(
        self, tool_schema_id: str, *, definition: str, source: str = "proposed",
        reasoning: Optional[str] = None, based_on_version: Optional[int] = None,
    ) -> dict:
        return self._client.publish_tool_schema_version(
            tool_schema_id, definition=definition, source=source,
            reasoning=reasoning, based_on_version=based_on_version,
        )
