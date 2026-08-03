"""
LlamaIndex integration for AgentX production tracing.

Usage::

    from agentx.integrations.llamaindex import AgentXLlamaIndexHandler
    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager

    handler = AgentXLlamaIndexHandler(agentx.tracer, name="my-rag-agent")
    Settings.callback_manager = CallbackManager([handler])

    # Or scoped to one query engine / agent:
    #   query_engine = index.as_query_engine(callback_manager=CallbackManager([handler]))

    # Every top-level query()/chat()/retrieve() call, or bare llm.complete()/
    # chat() call, is now traced automatically — including nested retrieval
    # and LLM steps within it.

Requires: ``pip install "agentx-python[llamaindex]"``
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agentx.tracing.tracer import Tracer, _safe_serialize
from agentx.integrations._perf import build_performance_summary

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "llama-index-core is required for AgentXLlamaIndexHandler. "
        "Install it with: pip install \"agentx-python[llamaindex]\""
    ) from exc


# Event types that can anchor a top-level AgentX trace. QUERY/AGENT_STEP are
# the usual top-level boundary for a query engine or agent; LLM/RETRIEVE are
# included too since a bare `llm.complete()`/`retriever.retrieve()` call (no
# enclosing query engine) fires directly at the root with no QUERY wrapper —
# confirmed via a live run against llama-index-core 0.14.x, whose
# `start_trace(trace_id)` uses a fixed operation-name string ("query",
# "completion", ...) rather than a unique id per call, so it isn't safe to
# key state on under concurrent calls — root-run tracking here is driven
# entirely by `on_event_start`/`on_event_end`'s unique `event_id`/`parent_id`
# instead, the same "walk the parent chain to find the top ancestor" pattern
# `langchain.py`'s `AgentXCallbackHandler` already uses for the same reason.
_ROOT_EVENT_TYPES = {CBEventType.QUERY, CBEventType.AGENT_STEP, CBEventType.LLM, CBEventType.RETRIEVE}


def _extract_llm_text(completion: Any) -> Optional[str]:
    if completion is None:
        return None
    text = getattr(completion, "text", None)
    if text:
        return text
    message = getattr(completion, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content:
        return content
    return str(completion) or None


def _extract_usage_tokens(completion: Any) -> tuple:
    """
    Best-effort: LlamaIndex's CallbackManager payload has no dedicated token
    fields, so this digs into the completion/response object's provider-raw
    data (shape varies per LLM integration, hence the broad try/except).
    """
    raw = getattr(completion, "raw", None)
    if not isinstance(raw, dict):
        return None, None
    usage = raw.get("usage")
    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        return input_tokens, output_tokens
    return None, None


def _extract_retrieval_output(nodes: Any) -> tuple:
    if not nodes:
        return None, None
    try:
        doc_count = len(nodes)
        texts = []
        for node in nodes:
            get_content = getattr(node, "get_content", None)
            texts.append(get_content() if callable(get_content) else str(node))
        return doc_count, "\n\n---\n\n".join(t for t in texts if t)
    except Exception:
        return None, None


class AgentXLlamaIndexHandler(BaseCallbackHandler):
    """
    LlamaIndex ``BaseCallbackHandler`` that captures the top-level query/
    chat/retrieve/agent-step call and its nested LLM/retrieval/tool steps,
    then sends one trace per top-level call.
    """

    def __init__(
        self,
        tracer: Tracer,
        name: str = "llamaindex-agent",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._tracer = tracer
        self._name = name
        self._metadata = metadata
        self._session_id = session_id

        self._parents: Dict[str, Optional[str]] = {}
        self._roots: Dict[str, bool] = {}
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._starts: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # BaseCallbackHandler protocol
    # ------------------------------------------------------------------

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass  # see _ROOT_EVENT_TYPES' comment — not used for state tracking

    def end_trace(self, trace_id: Optional[str] = None, trace_map: Optional[Dict[str, List[str]]] = None) -> None:
        pass

    def on_event_start(
        self,
        event_type: "CBEventType",
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        payload = payload or {}
        self._parents[event_id] = parent_id

        root_id = self._find_root(parent_id)
        if root_id is None and event_type in _ROOT_EVENT_TYPES:
            self._roots[event_id] = True
            root_id = event_id
            self._runs[event_id] = {
                "start": time.time(),
                "input": None,
                "output": None,
                "model": None,
                "error": None,
                "execution_steps": [],
                "tool_call_steps": [],
                "retrieval_steps": [],
                "input_tokens": 0,
                "output_tokens": 0,
            }

        self._starts[event_id] = {"start": time.time(), "type": event_type, "payload": payload}

        state = self._runs.get(root_id) if root_id else None
        if state is not None and state["input"] is None:
            if event_type == CBEventType.QUERY:
                state["input"] = payload.get(EventPayload.QUERY_STR)
            elif event_type == CBEventType.LLM:
                prompt = payload.get(EventPayload.PROMPT)
                if prompt is None:
                    messages = payload.get(EventPayload.MESSAGES)
                    prompt = _safe_serialize(messages) if messages else None
                state["input"] = prompt
            elif event_type == CBEventType.RETRIEVE:
                state["input"] = payload.get(EventPayload.QUERY_STR)

        return event_id

    def on_event_end(
        self,
        event_type: "CBEventType",
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        payload = payload or {}
        start_info = self._starts.pop(event_id, None)
        parent_id = self._parents.pop(event_id, None)
        is_root = self._roots.pop(event_id, False)
        root_id = event_id if is_root else self._find_root(parent_id)
        state = self._runs.get(root_id) if root_id else None

        if state is None:
            return

        exception = payload.get(EventPayload.EXCEPTION)
        if exception is not None:
            state["error"] = str(exception)

        start_t = start_info["start"] if start_info else time.time()
        end_t = time.time()

        if event_type == CBEventType.QUERY:
            response = payload.get(EventPayload.RESPONSE)
            if response is not None:
                state["output"] = str(response)
        elif event_type == CBEventType.LLM:
            completion = payload.get(EventPayload.COMPLETION) or payload.get(EventPayload.RESPONSE)
            output_text = _extract_llm_text(completion)
            model = payload.get(EventPayload.MODEL_NAME)
            if model and not state["model"]:
                state["model"] = model
            input_tokens, output_tokens = _extract_usage_tokens(completion)
            if input_tokens is not None:
                state["input_tokens"] += int(input_tokens)
            if output_tokens is not None:
                state["output_tokens"] += int(output_tokens)
            if output_text:
                state["output"] = output_text
            input_text = start_info["payload"].get(EventPayload.PROMPT) if start_info else None
            state["execution_steps"].append({
                "name": f"LLM Call {len(state['execution_steps']) + 1}",
                "duration_ms": (end_t - start_t) * 1000,
                "start_time": start_t,
                "end_time": end_t,
                "model": model,
                "input": input_text,
                "output": output_text or (f"ERROR: {exception}" if exception else None),
                "inputTokenSize": input_tokens,
                "outputTokenSize": output_tokens,
            })
        elif event_type == CBEventType.RETRIEVE:
            nodes = payload.get(EventPayload.NODES)
            doc_count, retrieved_text = _extract_retrieval_output(nodes)
            query = start_info["payload"].get(EventPayload.QUERY_STR) if start_info else None
            step: Dict[str, Any] = {
                "name": f"Retrieval {len(state['retrieval_steps']) + 1}",
                "duration_ms": (end_t - start_t) * 1000,
                "start_time": start_t,
                "end_time": end_t,
            }
            if query:
                step["query"] = query
            if doc_count is not None:
                step["doc_count"] = doc_count
            step["output"] = f"ERROR: {exception}" if exception else retrieved_text
            state["retrieval_steps"].append(step)
        elif event_type == CBEventType.FUNCTION_CALL:
            start_payload = start_info["payload"] if start_info else {}
            tool = start_payload.get(EventPayload.TOOL)
            tool_name = getattr(tool, "name", None) or str(tool) if tool is not None else "unknown"
            tool_input = start_payload.get(EventPayload.FUNCTION_CALL)
            tool_output = payload.get(EventPayload.FUNCTION_OUTPUT)
            state["tool_call_steps"].append({
                "name": tool_name,
                "duration_ms": (end_t - start_t) * 1000,
                "start_time": start_t,
                "end_time": end_t,
                "input": _safe_serialize(tool_input) if tool_input is not None else None,
                "output": f"ERROR: {exception}" if exception else (str(tool_output) if tool_output is not None else None),
            })

        if is_root:
            self._runs.pop(root_id, None)
            self._send_trace(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_root(self, parent_id: Optional[str]) -> Optional[str]:
        current = parent_id
        seen: set = set()
        while current and current != "root" and current not in seen:
            seen.add(current)
            if self._roots.get(current):
                return current
            current = self._parents.get(current)
        return None

    def _send_trace(self, state: Dict[str, Any]) -> None:
        latency_ms = int((time.time() - state["start"]) * 1000)
        perf = build_performance_summary(
            total_duration_ms=latency_ms,
            execution_steps=state["execution_steps"],
            tool_call_steps=state["tool_call_steps"],
            retrieval_steps=state["retrieval_steps"],
            has_errors=state["error"] is not None,
        )
        self._tracer._send(
            name=self._name,
            input=state["input"],
            output=state["output"],
            latency_ms=latency_ms,
            error=state["error"],
            framework="llamaindex",
            model=state["model"],
            metadata=self._metadata,
            session_id=self._session_id,
            performance_summary=perf,
            input_tokens=state["input_tokens"] or None,
            output_tokens=state["output_tokens"] or None,
        )
