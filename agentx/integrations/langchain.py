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
from agentx.integrations._perf import build_performance_summary

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
        # Retrieval steps that fire before on_chain_start (pre-run RAG pattern).
        # Consumed and attached when the next top-level chain starts.
        self._pending_retrieval_steps: List[Dict[str, Any]] = []
        # run_id → {"start": float, "query": str}
        self._retrieval_starts: Dict[UUID, Dict[str, Any]] = {}

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
            # Consume any retrieval steps that ran before this chain started
            # (pre-run RAG: retriever.invoke() called before agent.invoke())
            pending = self._pending_retrieval_steps[:]
            self._pending_retrieval_steps.clear()
            self._runs[run_id] = {
                "start": time.time(),
                "input": _extract_input(inputs),
                "tool_calls": [],
                "model": None,
                "execution_steps": [],
                "perf_tool_calls": [],
                "retrieval_steps": pending,
                "input_tokens": 0,
                "output_tokens": 0,
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
        tool_calls = state["tool_calls"] or _extract_tool_calls_from_messages(outputs)
        perf = build_performance_summary(
            total_duration_ms=latency_ms,
            execution_steps=state["execution_steps"],
            tool_call_steps=state["perf_tool_calls"],
            retrieval_steps=state["retrieval_steps"],
        )
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
            performance_summary=perf,
            input_tokens=state["input_tokens"] or None,
            output_tokens=state["output_tokens"] or None,
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
        perf = build_performance_summary(
            total_duration_ms=latency_ms,
            execution_steps=state["execution_steps"],
            tool_call_steps=state["perf_tool_calls"],
            retrieval_steps=state["retrieval_steps"],
            has_errors=True,
        )
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
            performance_summary=perf,
            input_tokens=state["input_tokens"] or None,
            output_tokens=state["output_tokens"] or None,
        )
        self._top_level.pop(run_id, None)
        self._parents.pop(run_id, None)

    # ------------------------------------------------------------------
    # LLM lifecycle
    # ------------------------------------------------------------------

    def _record_llm_start(
        self,
        serialized: Dict[str, Any],
        run_id: UUID,
        parent_run_id: Optional[UUID],
        kwargs: Dict[str, Any],
    ) -> None:
        """Shared logic for on_llm_start and on_chat_model_start."""
        self._parents[run_id] = parent_run_id
        kw = serialized.get("kwargs", {})
        model = (
            kw.get("model_name")
            or kw.get("model")
            or kwargs.get("invocation_params", {}).get("model")
            or kwargs.get("invocation_params", {}).get("model_name")
            or serialized.get("name")
        )
        model = str(model) if model and model not in ("None", "none") else None
        self._runs[run_id] = {"llm_start": time.time(), "model": model}
        top = self._find_top_ancestor(parent_run_id)
        if top and not self._runs[top].get("model") and model:
            self._runs[top]["model"] = model

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        self._record_llm_start(serialized, run_id, parent_run_id, kwargs)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        self._record_llm_start(serialized, run_id, parent_run_id, kwargs)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        llm_state = self._runs.pop(run_id, None)
        self._parents.pop(run_id, None)
        if llm_state:
            start_t = llm_state.get("llm_start")
            end_t = time.time()
            top = self._find_top_ancestor(parent_run_id)
            if start_t is not None and top and top in self._runs:
                steps = self._runs[top]["execution_steps"]
                steps.append({
                    "name": f"LLM Call {len(steps) + 1}",
                    "duration_ms": (end_t - start_t) * 1000,
                    "start_time": start_t,
                    "end_time": end_t,
                })
            # Extract token usage from LLMResult
            if top and top in self._runs:
                usage = {}
                if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
                    usage = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
                # Also check generations for token counts (some providers put it there)
                if not usage and hasattr(response, "generations"):
                    for gen_list in (response.generations or []):
                        for gen in (gen_list or []):
                            gen_info = getattr(gen, "generation_info", None) or {}
                            if gen_info.get("prompt_tokens") or gen_info.get("completion_tokens"):
                                usage = gen_info
                                break
                if usage:
                    self._runs[top]["input_tokens"] += int(
                        usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("prompt_token_count") or 0
                    )
                    self._runs[top]["output_tokens"] += int(
                        usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_token_count") or 0
                    )

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
        end_t = time.time()
        start_t = state["start"]
        latency_ms = int((end_t - start_t) * 1000)
        tool_call = {
            "name": state["tool_name"],
            "input": state["tool_input"],
            "output": str(output)[:500],
            "latency_ms": latency_ms,
        }
        top = self._find_top_ancestor(parent_run_id)
        if top and top in self._runs:
            self._runs[top]["tool_calls"].append(tool_call)
            self._runs[top]["perf_tool_calls"].append({
                "name": state["tool_name"],
                "duration_ms": (end_t - start_t) * 1000,
                "start_time": start_t,
                "end_time": end_t,
            })

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
        end_t = time.time()
        start_t = state.get("start", end_t)
        tool_call = {
            "name": state.get("tool_name", "unknown"),
            "input": state.get("tool_input"),
            "output": f"ERROR: {error}",
            "latency_ms": int((end_t - start_t) * 1000),
        }
        top = self._find_top_ancestor(parent_run_id)
        if top and top in self._runs:
            self._runs[top]["tool_calls"].append(tool_call)
            self._runs[top]["perf_tool_calls"].append({
                "name": state.get("tool_name", "unknown"),
                "duration_ms": (end_t - start_t) * 1000,
                "start_time": start_t,
                "end_time": end_t,
            })

    # ------------------------------------------------------------------
    # Retriever lifecycle
    # ------------------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        self._parents[run_id] = parent_run_id
        self._retrieval_starts[run_id] = {"start": time.time(), "query": query}

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        state = self._retrieval_starts.pop(run_id, None)
        self._parents.pop(run_id, None)
        if state is None:
            return
        end_t = time.time()
        start_t = state["start"]
        query: Optional[str] = state["query"] or None
        doc_count = len(documents) if hasattr(documents, "__len__") else None
        step: Dict[str, Any] = {
            "name": "Retrieval 1",  # renumbered below
            "duration_ms": (end_t - start_t) * 1000,
            "start_time": start_t,
            "end_time": end_t,
        }
        if query:
            step["query"] = query
        if doc_count is not None:
            step["doc_count"] = doc_count

        top = self._find_top_ancestor(parent_run_id)
        if top and top in self._runs:
            # Retriever ran inside an active chain — attach directly
            retrievals = self._runs[top]["retrieval_steps"]
            step["name"] = f"Retrieval {len(retrievals) + 1}"
            retrievals.append(step)
        else:
            # Retriever ran before the chain started (pre-run RAG pattern)
            step["name"] = f"Retrieval {len(self._pending_retrieval_steps) + 1}"
            self._pending_retrieval_steps.append(step)

    def on_retriever_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        self._retrieval_starts.pop(run_id, None)
        self._parents.pop(run_id, None)

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
