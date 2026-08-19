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

import threading
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from agentx.tracing.tracer import Tracer, _safe_serialize
from agentx.integrations._traced_call import capture_tool_definitions

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


def _extract_llm_input(
    prompts: Optional[List[str]] = None,
    messages: Optional[List[List[Any]]] = None,
) -> Optional[str]:
    """Flatten on_llm_start's ``prompts`` or on_chat_model_start's ``messages`` into one string."""
    if prompts:
        return "\n\n".join(prompts) if len(prompts) > 1 else prompts[0]

    if messages:
        lines: List[str] = []
        for message_list in messages:
            for msg in message_list:
                role = getattr(msg, "type", None) or getattr(msg, "role", None) or "user"
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content:
                    lines.append(f"{role}: {content}")
                elif isinstance(content, list):
                    parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    joined = " ".join(t for t in parts if t)
                    if joined:
                        lines.append(f"{role}: {joined}")
        return "\n".join(lines) if lines else None

    return None


def _describe_tool_calls(message: Any) -> Optional[str]:
    """
    Format an AIMessage's ``tool_calls`` as a readable fallback for ``output``
    when the model responded with a pure tool call and no text content.
    """
    tool_calls = getattr(message, "tool_calls", None) if message is not None else None
    if not tool_calls:
        return None
    parts = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            name = tc.get("name") or "unknown"
            args = tc.get("args")
        else:
            name = getattr(tc, "name", None) or "unknown"
            args = getattr(tc, "args", None)
        parts.append(f"{name}({args})" if args is not None else f"{name}()")
    return "[tool call] " + ", ".join(parts)


def _extract_llm_output(response: "LLMResult") -> Optional[str]:
    """Flatten on_llm_end's ``LLMResult`` (chat or completion generations) into one string."""
    texts: List[str] = []
    tool_call_fallbacks: List[str] = []
    for gen_list in getattr(response, "generations", None) or []:
        for gen in gen_list or []:
            message = getattr(gen, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if isinstance(content, str) and content:
                texts.append(content)
                continue
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                joined = " ".join(t for t in parts if t)
                if joined:
                    texts.append(joined)
                    continue
            text = getattr(gen, "text", None)
            if text:
                texts.append(text)
                continue
            # No text content - the model likely responded with a pure tool
            # call instead of commentary. Fall back to describing it so the
            # step's output isn't silently omitted.
            described = _describe_tool_calls(message)
            if described:
                tool_call_fallbacks.append(described)
    if texts:
        return "\n".join(texts)
    if tool_call_fallbacks:
        return "\n".join(tool_call_fallbacks)
    return None


# Nested chain runs that are plumbing, not agent structure - LCEL composition wrappers,
# prompt/parse steps, and LangGraph's internal channel machinery. Skipped when deciding which
# chain runs become "node" child spans; their LLM/tool descendants re-parent to the nearest
# non-noise ancestor (see _emit_span_tree's resolve_parent walk).
_NOISE_CHAIN_NAMES = frozenset({
    "RunnableSequence",
    "RunnableParallel",
    "RunnableLambda",
    "RunnableAssign",
    "RunnablePick",
    "RunnableBinding",
    "RunnableWithFallbacks",
    "RunnableWithMessageHistory",
    "RunnableBranch",
    "ChatPromptTemplate",
    "PromptTemplate",
    "StrOutputParser",
    "JsonOutputParser",
    "ToolsAgentOutputParser",
    "OpenAIFunctionsAgentOutputParser",
    "LangGraph",
    "CompiledStateGraph",
    "Prompt",
    "_Exception",
})

_NOISE_CHAIN_PREFIXES = ("ChannelWrite", "ChannelRead", "Branch<", "RunnableParallel<", "_")


def _is_noise_chain(name: str) -> bool:
    return name in _NOISE_CHAIN_NAMES or name.startswith(_NOISE_CHAIN_PREFIXES)


class AgentXCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that captures the top-level chain run and all
    nested tool calls, then sends one trace per top-level chain invocation.

    Compatible with AgentExecutor, LCEL chains, and LangGraph agents
    (``create_agent``, ``create_react_agent``).

    The trace is a real span tree: LangGraph graph nodes (and named non-plumbing sub-chains)
    become child spans, and each LLM call / tool call / retrieval becomes a span parented under
    the node that ran it - so the engine's Execution Timeline shows the actual graph trajectory
    (which nodes ran, in what order, and what each did), not a flat step list.
    """

    def __init__(
        self,
        tracer: Tracer,
        name: str = "langchain-agent",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        max_run_age_seconds: float = 3600.0,
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
        # Guards appends to a top-level run's shared aggregate lists (tool_calls,
        # perf_tool_calls, execution_steps, retrieval_steps). LangGraph's ToolNode
        # runs multiple tool calls from one AIMessage concurrently via a thread
        # pool (see langgraph.prebuilt.tool_node.ToolNode._func), so on_tool_end /
        # on_tool_error can fire from several threads at once for the same
        # top-level run.
        self._state_lock = threading.Lock()
        # Safety net for a run_id whose start callback never gets a matching
        # end/error callback at all (e.g. a hard crash, or a custom Runnable
        # that swallows exceptions before they reach LangChain's own callback
        # dispatch) - normal on_*_end/on_*_error pairing already frees
        # everything else. This handler is typically a long-lived singleton
        # across many requests, so entries older than this are swept out the
        # next time a new top-level chain starts.
        self._max_run_age_seconds = max_run_age_seconds

    def _prune_stale_entries(self) -> None:
        """Sweep out run_id entries older than max_run_age_seconds - see __init__'s comment."""
        cutoff = time.time() - self._max_run_age_seconds

        stale_run_ids = [
            run_id
            for run_id, state in self._runs.items()
            if (state.get("start") or state.get("llm_start") or 0) < cutoff
        ]
        for run_id in stale_run_ids:
            self._runs.pop(run_id, None)
            self._top_level.pop(run_id, None)
            self._parents.pop(run_id, None)

        stale_retrieval_ids = [
            run_id for run_id, state in self._retrieval_starts.items() if state.get("start", 0) < cutoff
        ]
        for run_id in stale_retrieval_ids:
            self._retrieval_starts.pop(run_id, None)
            self._parents.pop(run_id, None)

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
            self._prune_stale_entries()
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
                "retrieval_steps": pending,
                "input_tokens": 0,
                "output_tokens": 0,
                # Graph structure captured under this top-level run: named nested chain runs
                # (LangGraph nodes, sub-agents) keyed by run_id, and the FULL nested-chain
                # parent map (noise chains included) so _emit_span_tree can walk through
                # skipped plumbing runs to the nearest emitted ancestor.
                "node_runs": {},
                "chain_parents": {},
                # The request's tools=[...] as seen on the first LLM call's invocation params -
                # attached to the root trace's metadata so the engine's unregistered-tool
                # listing can surface the REAL definition (not one inferred from arguments).
                "tool_definitions": None,
            }
        else:
            top = self._find_top_ancestor(parent_run_id)
            if top is None:
                return
            state = self._runs.get(top)
            if state is None:
                return
            name = kwargs.get("name") or (serialized or {}).get("name")
            # LangGraph stamps its node runs with metadata.langgraph_node - trust that over the
            # noise heuristic when present (a user's node could legitimately be named
            # "RunnableLambda"-style by a wrapper).
            run_meta = kwargs.get("metadata") or {}
            is_node = bool(name) and (run_meta.get("langgraph_node") == name or not _is_noise_chain(str(name)))
            with self._state_lock:
                state["chain_parents"][run_id] = parent_run_id
                if is_node:
                    state["node_runs"][run_id] = {
                        "name": str(name),
                        "start": time.time(),
                        "end": None,
                        "input": _extract_input(inputs),
                        "output": None,
                        "error": None,
                        "parent": parent_run_id,
                    }

    def _finalize_node(self, run_id: UUID, *, output: Any = None, error: Optional[str] = None) -> None:
        """Close a nested chain run's node record (if it became one) with end time + result."""
        top = self._find_top_ancestor(run_id)
        if top is None:
            return
        state = self._runs.get(top)
        if state is None:
            return
        with self._state_lock:
            node = state.get("node_runs", {}).get(run_id)
            if node is None:
                return
            node["end"] = time.time()
            if output is not None:
                node["output"] = output
            if error is not None:
                node["error"] = error

    def _emit_span_tree(self, span, state: Dict[str, Any], tool_calls: List[Dict[str, Any]]) -> None:
        """
        Emit the recorded run as a hierarchical span tree under ``span``: node runs first (in
        start order, parented to their nearest emitted ancestor), then every LLM call, tool
        call, and retrieval parented under the node that ran it. Records whose parent chain is
        entirely noise (or missing, e.g. pre-run retrievals) land directly under the root span.
        """
        emitted: Dict[Any, Any] = {}
        chain_parents: Dict[Any, Any] = state.get("chain_parents", {})
        node_runs: Dict[Any, Dict[str, Any]] = state.get("node_runs", {})

        def resolve_parent(parent_id: Any):
            seen: set = set()
            current = parent_id
            while current is not None and current not in seen:
                seen.add(current)
                if current in emitted:
                    return emitted[current]
                current = chain_parents.get(current)
            return span

        for run_id, node in sorted(node_runs.items(), key=lambda kv: kv[1]["start"]):
            parent_span = resolve_parent(node.get("parent"))
            emitted[run_id] = parent_span.child_span(
                node["name"],
                start_time=node["start"],
                end_time=node.get("end") or node["start"],
                input=node.get("input"),
                output=node.get("output"),
                error=node.get("error"),
            )

        llm_count = 0
        for step in state.get("execution_steps", []):
            llm_count += 1
            resolve_parent(step.get("parent_run_id")).child_span(
                step.get("name") or f"LLM Call {llm_count}",
                start_time=step.get("start_time"),
                end_time=step.get("end_time"),
                duration_ms=step.get("duration_ms"),
                input=step.get("input"),
                output=step.get("output"),
                model=step.get("model"),
                input_tokens=step.get("inputTokenSize"),
                output_tokens=step.get("outputTokenSize"),
            )
        for tc in tool_calls:
            resolve_parent(tc.get("parent_run_id")).child_span(
                tc.get("name") or "Tool call",
                start_time=tc.get("start_time"),
                end_time=tc.get("end_time"),
                duration_ms=tc.get("latency_ms"),
                input=tc.get("input"),
                output=tc.get("output"),
                error=None if tc.get("success", True) else str(tc.get("output") or "Tool call failed"),
            )
        for step in state.get("retrieval_steps", []):
            resolve_parent(step.get("parent_run_id")).child_span(
                step.get("name") or "Retrieval",
                start_time=step.get("start_time"),
                end_time=step.get("end_time"),
                duration_ms=step.get("duration_ms"),
                input=step.get("query"),
                output=step.get("output"),
                metadata={"kind": "retrieval"},
            )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        # Pop _top_level/_parents for every chain run, not just top-level ones.
        # Nested chain steps (LCEL sub-chains, LangGraph nodes, AgentExecutor
        # internals) fire on_chain_start/on_chain_end too - leaving their
        # entries behind here would leak forever in a long-lived singleton
        # handler, since nothing else ever cleans up a non-top-level run_id.
        is_top = self._top_level.pop(run_id, None)
        if not is_top:
            # Close the node record before dropping this run's _parents entry - the top-ancestor
            # walk inside _finalize_node still needs it.
            self._finalize_node(run_id, output=_extract_output(outputs))
            self._parents.pop(run_id, None)
            return
        self._parents.pop(run_id, None)
        state = self._runs.pop(run_id, None)
        if state is None:
            return
        output = _extract_output(outputs)
        # Each tool_call dict already carries its own start_time/end_time (set in
        # on_tool_end/on_tool_error), so no re-pairing against perf_tool_calls by
        # index is needed here. That used to be done via zip(), which silently
        # mispaired timestamps when LangGraph's ToolNode ran several tool calls
        # from one AIMessage concurrently (see _state_lock's docstring): two
        # lists appended to from different threads don't necessarily end up in
        # the same relative order.
        tool_calls = state["tool_calls"] or _extract_tool_calls_from_messages(outputs)

        active_span = self._tracer.current_span
        if active_span is not None:
            if state.get("tool_definitions") and not (active_span._metadata or {}).get("tools"):
                active_span._metadata = {**(active_span._metadata or {}), "tools": state["tool_definitions"]}
            # Part of a `with tracer.trace(...)` block (e.g. an orchestrator
            # spanning several chain/agent/retriever calls) - fold this
            # top-level run into it instead of sending an independent trace.
            active_span._merge_child_run(
                tool_calls=tool_calls,
                input=state["input"],
                output=output,
                model=state.get("model"),
                framework="langchain",
                input_tokens=state["input_tokens"] or None,
                output_tokens=state["output_tokens"] or None,
                emit_steps=False,
            )
            self._emit_span_tree(active_span, state, tool_calls)
        else:
            # Standalone usage (no enclosing `with tracer.trace()`): open a real root span for
            # this chain invocation and let _merge_child_run explode its accumulated
            # execution_steps/tool_calls/retrieval_steps into real child-span rows.
            with self._tracer.trace(
                self._name,
                metadata=(
                    {**(self._metadata or {}), "tools": state["tool_definitions"]}
                    if state.get("tool_definitions")
                    else self._metadata
                ),
                session_id=self._session_id,
            ) as span:
                # __enter__ just set _start to "now" - overridden to the chain's real start time,
                # see llamaindex.py's _send_trace for the identical fix and full rationale.
                span._start = state["start"]
                span._merge_child_run(
                    tool_calls=tool_calls,
                    input=state["input"],
                    output=output,
                    model=state.get("model"),
                    framework="langchain",
                    input_tokens=state["input_tokens"] or None,
                    output_tokens=state["output_tokens"] or None,
                    emit_steps=False,
                )
                self._emit_span_tree(span, state, tool_calls)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        # See on_chain_end's comment - pop for every chain run, not just top-level.
        is_top = self._top_level.pop(run_id, None)
        if not is_top:
            self._finalize_node(run_id, error=str(error))
            self._parents.pop(run_id, None)
            return
        self._parents.pop(run_id, None)
        state = self._runs.pop(run_id, None)
        if state is None:
            return

        active_span = self._tracer.current_span
        if active_span is not None:
            if state.get("tool_definitions") and not (active_span._metadata or {}).get("tools"):
                active_span._metadata = {**(active_span._metadata or {}), "tools": state["tool_definitions"]}
            active_span.set_error(str(error))
            active_span._merge_child_run(
                tool_calls=state["tool_calls"],
                input=state["input"],
                model=state.get("model"),
                framework="langchain",
                input_tokens=state["input_tokens"] or None,
                output_tokens=state["output_tokens"] or None,
                emit_steps=False,
            )
            self._emit_span_tree(active_span, state, state["tool_calls"])
        else:
            # See on_chain_end's matching branch - same standalone-usage handling.
            with self._tracer.trace(
                self._name,
                metadata=(
                    {**(self._metadata or {}), "tools": state["tool_definitions"]}
                    if state.get("tool_definitions")
                    else self._metadata
                ),
                session_id=self._session_id,
            ) as span:
                span._start = state["start"]
                span.set_error(str(error))
                span._merge_child_run(
                    tool_calls=state["tool_calls"],
                    input=state["input"],
                    model=state.get("model"),
                    framework="langchain",
                    input_tokens=state["input_tokens"] or None,
                    output_tokens=state["output_tokens"] or None,
                    emit_steps=False,
                )
                self._emit_span_tree(span, state, state["tool_calls"])

    # ------------------------------------------------------------------
    # LLM lifecycle
    # ------------------------------------------------------------------

    def _record_llm_start(
        self,
        serialized: Dict[str, Any],
        run_id: UUID,
        parent_run_id: Optional[UUID],
        kwargs: Dict[str, Any],
        *,
        prompts: Optional[List[str]] = None,
        messages: Optional[List[Any]] = None,
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
        top_for_tools = self._find_top_ancestor(parent_run_id)
        if top_for_tools and top_for_tools in self._runs and not self._runs[top_for_tools].get("tool_definitions"):
            captured = capture_tool_definitions(kwargs.get("invocation_params", {}).get("tools"))
            if captured:
                self._runs[top_for_tools]["tool_definitions"] = captured
        self._runs[run_id] = {
            "llm_start": time.time(),
            "model": model,
            "input": _extract_llm_input(prompts=prompts, messages=messages),
        }
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
        self._record_llm_start(serialized, run_id, parent_run_id, kwargs, prompts=prompts)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        self._record_llm_start(serialized, run_id, parent_run_id, kwargs, messages=messages)

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

            # Extract token usage from LLMResult for this call
            call_input_tokens: Optional[int] = None
            call_output_tokens: Optional[int] = None
            with self._state_lock:
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
                        call_input_tokens = int(
                            usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("prompt_token_count") or 0
                        )
                        call_output_tokens = int(
                            usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_token_count") or 0
                        )
                        self._runs[top]["input_tokens"] += call_input_tokens
                        self._runs[top]["output_tokens"] += call_output_tokens

                if start_t is not None and top and top in self._runs:
                    steps = self._runs[top]["execution_steps"]
                    steps.append({
                        "parent_run_id": parent_run_id,
                        "name": f"LLM Call {len(steps) + 1}",
                        "duration_ms": (end_t - start_t) * 1000,
                        "start_time": start_t,
                        "end_time": end_t,
                        "model": llm_state.get("model"),
                        "input": llm_state.get("input"),
                        "output": _extract_llm_output(response),
                        "inputTokenSize": call_input_tokens,
                        "outputTokenSize": call_output_tokens,
                    })

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
            "parent_run_id": parent_run_id,
            "name": state["tool_name"],
            "input": state["tool_input"],
            "output": str(output),
            "latency_ms": latency_ms,
            "success": True,
            # Set directly on the tool_call dict (not just perf_tool_calls below) so
            # on_chain_end's merged-span path doesn't need to re-pair the two lists by
            # index later. See _state_lock's docstring for why that used to be unsafe.
            "start_time": start_t,
            "end_time": end_t,
        }
        top = self._find_top_ancestor(parent_run_id)
        with self._state_lock:
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
        end_t = time.time()
        start_t = state.get("start", end_t)
        tool_call = {
            "parent_run_id": parent_run_id,
            "name": state.get("tool_name", "unknown"),
            "input": state.get("tool_input"),
            "output": f"ERROR: {error}",
            "latency_ms": int((end_t - start_t) * 1000),
            "success": False,
            "start_time": start_t,
            "end_time": end_t,
        }
        top = self._find_top_ancestor(parent_run_id)
        with self._state_lock:
            if top and top in self._runs:
                self._runs[top]["tool_calls"].append(tool_call)

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
            "parent_run_id": parent_run_id,
            "name": "Retrieval 1",  # renumbered below
            "duration_ms": (end_t - start_t) * 1000,
            "start_time": start_t,
            "end_time": end_t,
        }
        if query:
            step["query"] = query
        if doc_count is not None:
            step["doc_count"] = doc_count
        if hasattr(documents, "__iter__"):
            contents = [getattr(d, "page_content", None) or str(d) for d in documents]
            if contents:
                step["output"] = "\n\n---\n\n".join(contents)

        top = self._find_top_ancestor(parent_run_id)
        if top and top in self._runs:
            # Retriever ran inside an active chain - attach directly
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
        state = self._retrieval_starts.pop(run_id, None)
        self._parents.pop(run_id, None)
        if state is None:
            return
        end_t = time.time()
        start_t = state["start"]
        query: Optional[str] = state["query"] or None
        step: Dict[str, Any] = {
            "name": "Retrieval 1",  # renumbered below
            "duration_ms": (end_t - start_t) * 1000,
            "start_time": start_t,
            "end_time": end_t,
            "output": f"ERROR: {error}",
        }
        if query:
            step["query"] = query

        top = self._find_top_ancestor(parent_run_id)
        if top and top in self._runs:
            # Retriever ran inside an active chain - attach directly
            retrievals = self._runs[top]["retrieval_steps"]
            step["name"] = f"Retrieval {len(retrievals) + 1}"
            retrievals.append(step)
        else:
            # Retriever ran before the chain started (pre-run RAG pattern)
            step["name"] = f"Retrieval {len(self._pending_retrieval_steps) + 1}"
            self._pending_retrieval_steps.append(step)

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
