"""
LangChain integration for AgentX production tracing.

Usage::

    from agentx.integrations.langchain import AgentXCallbackHandler

    handler = AgentXCallbackHandler(agentx.tracer, name="my-chain")

    # LCEL chain
    chain.invoke({"query": q}, config={"callbacks": [handler]})

    # LangGraph agent (create_agent / create_react_agent)
    agent.invoke({"messages": [...]}, config={"callbacks": [handler]})

    # AgentExecutor
    agent.invoke({"input": q}, config={"callbacks": [handler]})

Requires: ``pip install "agentx-python[langchain]"``
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from agentx.tracing.tracer import Tracer, _safe_serialize

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "langchain-core is required for AgentXCallbackHandler. "
        "Install it with: pip install \"agentx-python[langchain]\""
    ) from exc


# ---------------------------------------------------------------------------
# Output extraction helpers
# ---------------------------------------------------------------------------

def _extract_output(outputs: Any) -> Any:
    """
    Extract a clean, human-readable output from a chain's return value.

    - LangGraph agents return {"messages": [HumanMessage, ..., AIMessage(final)]}
      → extract the last AI message's text content
    - AgentExecutor returns {"output": "..."}
    - LCEL chains return {"text": "..."} or a plain string
    """
    if not isinstance(outputs, dict):
        return _safe_serialize(outputs)

    # LangGraph: {"messages": [...]}
    messages = outputs.get("messages")
    if isinstance(messages, list) and messages:
        for msg in reversed(messages):
            msg_type = getattr(msg, "type", None) or getattr(msg, "role", None)
            if msg_type in ("ai", "assistant"):
                content = getattr(msg, "content", None)
                if content and isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    # Multi-part content blocks
                    texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    joined = " ".join(t for t in texts if t).strip()
                    if joined:
                        return joined

    # Standard chain / AgentExecutor
    for key in ("output", "text", "answer", "result", "response"):
        val = outputs.get(key)
        if val and isinstance(val, str):
            return val

    return _safe_serialize(outputs)


def _extract_input(inputs: Any) -> Any:
    """
    Extract a clean input from a chain's input dict.

    - LangGraph: {"messages": [HumanMessage(...)]} → first human message text
    - AgentExecutor: {"input": "..."} → the string
    - LCEL: {"query": "...", "question": "..."} → the string value
    """
    if not isinstance(inputs, dict):
        return _safe_serialize(inputs)

    # LangGraph: {"messages": [...]}
    messages = inputs.get("messages")
    if isinstance(messages, list) and messages:
        for msg in messages:
            msg_type = getattr(msg, "type", None) or getattr(msg, "role", None)
            if msg_type in ("human", "user"):
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    return content
        # Fallback: first message regardless of type
        first = messages[0]
        content = getattr(first, "content", None)
        if isinstance(content, str):
            return content

    # Standard
    for key in ("input", "query", "question", "human_input"):
        val = inputs.get(key)
        if val and isinstance(val, str):
            return val

    return _safe_serialize(inputs)


def _extract_tool_calls_from_messages(outputs: Any) -> List[Dict[str, Any]]:
    """
    Fallback: extract tool calls directly from the message history when
    on_tool_end callbacks were not linked to the top-level run.
    """
    if not isinstance(outputs, dict):
        return []

    messages = outputs.get("messages")
    if not isinstance(messages, list):
        return []

    # Build map of tool_call_id → result from ToolMessages
    results: Dict[str, str] = {}
    for msg in messages:
        if getattr(msg, "type", None) == "tool":
            call_id = getattr(msg, "tool_call_id", None)
            content = getattr(msg, "content", "")
            if call_id:
                results[call_id] = str(content)

    # Extract tool calls from AIMessages
    tool_calls: List[Dict[str, Any]] = []
    for msg in messages:
        if getattr(msg, "type", None) != "ai":
            continue
        for tc in getattr(msg, "tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            call_id = tc.get("id", "")
            tool_calls.append({
                "name": tc.get("name", "unknown"),
                "input": str(tc.get("args", "")),
                "output": results.get(call_id, ""),
            })

    return tool_calls


class AgentXCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that captures the top-level chain run and all
    nested tool calls, then sends one trace per top-level chain invocation.

    Compatible with AgentExecutor, LCEL chains, and LangGraph agents
    (``create_agent``, ``create_react_agent``).
    """

    def __init__(
        self,
        tracer: Tracer,
        name: str = "langchain-agent",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._tracer = tracer
        self._name = name
        self._metadata = metadata
        self._session_id = session_id

        self._runs: Dict[UUID, Dict[str, Any]] = {}
        self._top_level: Dict[UUID, bool] = {}
        # Full parent-chain map so _find_top_ancestor can walk arbitrary depth
        self._parents: Dict[UUID, Optional[UUID]] = {}

    # ------------------------------------------------------------------
    # Chain lifecycle
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        self._parents[run_id] = parent_run_id
        is_top = parent_run_id is None
        self._top_level[run_id] = is_top
        if is_top:
            self._runs[run_id] = {
                "start": time.time(),
                "input": _extract_input(inputs),
                "tool_calls": [],
                "model": None,
            }

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        if not self._top_level.get(run_id):
            return
        state = self._runs.pop(run_id, None)
        if state is None:
            return
        latency_ms = int((time.time() - state["start"]) * 1000)
        output = _extract_output(outputs)
        # If callbacks missed tool calls (deep nesting), extract from message history
        tool_calls = state["tool_calls"] or _extract_tool_calls_from_messages(outputs)
        self._tracer._send(
            name=self._name,
            input=state["input"],
            output=output,
            latency_ms=latency_ms,
            framework="langchain",
            model=state.get("model"),
            tool_calls=tool_calls or None,
            metadata=self._metadata,
            session_id=self._session_id,
        )
        self._top_level.pop(run_id, None)
        self._parents.pop(run_id, None)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        if not self._top_level.get(run_id):
            return
        state = self._runs.pop(run_id, None)
        if state is None:
            return
        latency_ms = int((time.time() - state["start"]) * 1000)
        self._tracer._send(
            name=self._name,
            input=state["input"],
            error=str(error),
            latency_ms=latency_ms,
            framework="langchain",
            model=state.get("model"),
            tool_calls=state["tool_calls"] or None,
            metadata=self._metadata,
            session_id=self._session_id,
        )
        self._top_level.pop(run_id, None)
        self._parents.pop(run_id, None)

    # ------------------------------------------------------------------
    # LLM lifecycle
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        self._parents[run_id] = parent_run_id
        self._runs.setdefault(run_id, {})["llm_start"] = time.time()

        top = self._find_top_ancestor(parent_run_id)
        if top and not self._runs[top].get("model"):
            # Check all common field locations across langchain versions
            kw = serialized.get("kwargs", {})
            model = (
                kw.get("model_name")
                or kw.get("model")
                or kwargs.get("invocation_params", {}).get("model")
                or kwargs.get("invocation_params", {}).get("model_name")
                or serialized.get("name")
            )
            if model and model not in ("None", "none"):
                self._runs[top]["model"] = str(model)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        self._runs.pop(run_id, None)
        self._parents.pop(run_id, None)

    # ------------------------------------------------------------------
    # Tool lifecycle
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        self._parents[run_id] = parent_run_id
        self._runs[run_id] = {
            "tool_name": serialized.get("name", "unknown"),
            "tool_input": input_str,
            "start": time.time(),
        }

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        state = self._runs.pop(run_id, None)
        self._parents.pop(run_id, None)
        if state is None:
            return
        latency_ms = int((time.time() - state["start"]) * 1000)
        tool_call = {
            "name": state["tool_name"],
            "input": state["tool_input"],
            "output": str(output)[:500],
            "latency_ms": latency_ms,
        }
        top = self._find_top_ancestor(parent_run_id)
        if top and top in self._runs:
            self._runs[top]["tool_calls"].append(tool_call)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        state = self._runs.pop(run_id, None)
        self._parents.pop(run_id, None)
        if state is None:
            return
        tool_call = {
            "name": state.get("tool_name", "unknown"),
            "input": state.get("tool_input"),
            "output": f"ERROR: {error}",
            "latency_ms": int((time.time() - state["start"]) * 1000),
        }
        top = self._find_top_ancestor(parent_run_id)
        if top and top in self._runs:
            self._runs[top]["tool_calls"].append(tool_call)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_top_ancestor(self, start: Optional[UUID]) -> Optional[UUID]:
        """Walk the full parent chain to find the top-level run_id."""
        current = start
        seen: set = set()
        while current is not None and current not in seen:
            seen.add(current)
            if self._top_level.get(current):
                return current
            current = self._parents.get(current)
        return None
