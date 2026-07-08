"""
Google ADK integration for AgentX production tracing.

Usage::

    from agentx.integrations.google_adk import AgentXADKPlugin
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    plugin = AgentXADKPlugin(agentx.tracer, name="my-agent")

    runner = Runner(
        agent=agent,
        app_name="my-app",
        session_service=InMemorySessionService(),
        plugins=[plugin],
    )

Requires: ``pip install "agentx-python[google-adk]"``
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agentx.tracing.tracer import Tracer, _safe_serialize

try:
    from google.adk.plugins.base_plugin import BasePlugin
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "google-adk is required for AgentXADKPlugin. "
        "Install it with: pip install \"agentx-python[google-adk]\""
    ) from exc


def _content_to_text(content: Any) -> Optional[str]:
    """Extract plain text from a google.genai types.Content object."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None)
    if not parts:
        return None
    texts = []
    for part in parts:
        text = getattr(part, "text", None)
        if text and isinstance(text, str):
            texts.append(text)
    return " ".join(texts) if texts else None


class AgentXADKPlugin(BasePlugin):
    """
    Google ADK plugin that sends one AgentX trace per runner invocation.

    Captures input (user message), output (final model reply), model name,
    tool calls, and latency via the ADK plugin callback hooks.

    Register via the ``plugins`` list when constructing the ADK ``Runner``.
    """

    def __init__(
        self,
        tracer: Tracer,
        name: str = "google-adk-agent",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        super().__init__(name="agentx")
        self._tracer = tracer
        self._agent_name = name
        self._metadata = metadata
        self._session_id = session_id
        # invocation_id → accumulated run state
        self._runs: Dict[str, Dict[str, Any]] = {}
        # id(tool_context) → start time float
        self._tool_starts: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    async def before_run_callback(self, *, invocation_context: Any) -> None:
        inv_id = invocation_context.invocation_id
        agent_name = getattr(invocation_context.agent, "name", None) or self._agent_name
        self._runs[inv_id] = {
            "start": time.time(),
            "name": agent_name,
            "input": None,
            "output": None,
            "model": None,
            "tool_calls": [],
            "error": None,
        }

    async def on_user_message_callback(
        self, *, invocation_context: Any, user_message: Any
    ) -> None:
        inv_id = invocation_context.invocation_id
        state = self._runs.get(inv_id)
        if state is None:
            return
        text = _content_to_text(user_message)
        if text and state["input"] is None:
            state["input"] = text

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        inv_id = invocation_context.invocation_id
        state = self._runs.pop(inv_id, None)
        if state is None:
            return
        latency_ms = int((time.time() - state["start"]) * 1000)
        self._tracer._send(
            name=state["name"],
            input=state["input"],
            output=state["output"],
            latency_ms=latency_ms,
            framework="google-adk",
            model=state["model"],
            tool_calls=state["tool_calls"] or None,
            metadata=self._metadata,
            session_id=self._session_id,
        )

    # ------------------------------------------------------------------
    # Model callbacks — capture model name and output
    # ------------------------------------------------------------------

    async def before_model_callback(
        self, *, callback_context: Any, llm_request: Any
    ) -> None:
        inv_id = callback_context.get_invocation_context().invocation_id
        state = self._runs.get(inv_id)
        if state and not state["model"] and llm_request.model:
            state["model"] = str(llm_request.model)

    async def after_model_callback(
        self, *, callback_context: Any, llm_response: Any
    ) -> None:
        inv_id = callback_context.get_invocation_context().invocation_id
        state = self._runs.get(inv_id)
        if state is None:
            return
        content = getattr(llm_response, "content", None)
        text = _content_to_text(content)
        # Keep updating — the last non-empty model reply is the final answer
        if text:
            state["output"] = text

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    async def before_tool_callback(
        self, *, tool: Any, tool_args: Dict[str, Any], tool_context: Any
    ) -> None:
        self._tool_starts[id(tool_context)] = time.time()

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: Dict[str, Any],
        tool_context: Any,
        result: Dict[str, Any],
    ) -> None:
        inv_id = tool_context.get_invocation_context().invocation_id
        state = self._runs.get(inv_id)
        if state is None:
            return
        start = self._tool_starts.pop(id(tool_context), None)
        tool_call: Dict[str, Any] = {
            "name": getattr(tool, "name", "unknown"),
            "input": _safe_serialize(tool_args),
            "output": str(result)[:500] if result is not None else None,
        }
        if start is not None:
            tool_call["latency_ms"] = max(0, int((time.time() - start) * 1000))
        state["tool_calls"].append(tool_call)

    async def on_tool_error_callback(
        self,
        *,
        tool: Any,
        tool_args: Dict[str, Any],
        tool_context: Any,
        error: Exception,
    ) -> None:
        inv_id = tool_context.get_invocation_context().invocation_id
        state = self._runs.get(inv_id)
        if state is None:
            return
        start = self._tool_starts.pop(id(tool_context), None)
        tool_call: Dict[str, Any] = {
            "name": getattr(tool, "name", "unknown"),
            "input": _safe_serialize(tool_args),
            "output": f"ERROR: {error}",
        }
        if start is not None:
            tool_call["latency_ms"] = max(0, int((time.time() - start) * 1000))
        state["tool_calls"].append(tool_call)
