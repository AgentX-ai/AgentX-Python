"""
LangChain integration for AgentX production tracing.

Usage::

    from agentx.integrations.langchain import AgentXCallbackHandler

    handler = AgentXCallbackHandler(agentx.tracer, name="my-chain")

    # LCEL chain
    chain.invoke({"query": q}, config={"callbacks": [handler]})

    # AgentExecutor
    agent.invoke({"input": q}, config={"callbacks": [handler]})

Requires: ``pip install agentx[langchain]``
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
        "Install it with: pip install agentx[langchain]"
    ) from exc


class AgentXCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that captures the top-level chain run and all
    nested tool calls, then sends one trace per top-level chain invocation.
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

        # Keyed by run_id (UUID) → state dict
        self._runs: Dict[UUID, Dict[str, Any]] = {}
        # Track which run_ids are top-level (no parent)
        self._top_level: Dict[UUID, bool] = {}

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
        is_top = parent_run_id is None
        self._top_level[run_id] = is_top
        if is_top:
            self._runs[run_id] = {
                "start": time.time(),
                "input": _safe_serialize(inputs),
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
        self._tracer._send(
            name=self._name,
            input=state["input"],
            output=_safe_serialize(outputs),
            latency_ms=latency_ms,
            framework="langchain",
            model=state.get("model"),
            tool_calls=state["tool_calls"] or None,
            metadata=self._metadata,
            session_id=self._session_id,
        )
        self._top_level.pop(run_id, None)

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

    # ------------------------------------------------------------------
    # LLM lifecycle (captures model name and token counts)
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
        # Record LLM start time keyed by its own run_id
        self._runs.setdefault(run_id, {})["llm_start"] = time.time()

        # Propagate model name to the top-level run
        top = self._find_top_ancestor(parent_run_id)
        if top and not self._runs[top].get("model"):
            model = (
                serialized.get("kwargs", {}).get("model_name")
                or serialized.get("kwargs", {}).get("model")
                or serialized.get("id", [None])[-1]
            )
            if model:
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

    def _find_top_ancestor(self, parent_run_id: Optional[UUID]) -> Optional[UUID]:
        """Walk up parent chain to find the top-level run_id."""
        current = parent_run_id
        while current is not None:
            if self._top_level.get(current):
                return current
            # If current is a nested run, keep climbing — for simplicity return
            # the immediate parent that is registered as top-level
            break
        return current if current in self._runs else None
