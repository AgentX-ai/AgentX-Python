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
from agentx.integrations._perf import build_performance_summary

try:
    from google.adk.plugins.base_plugin import BasePlugin
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "google-adk is required for AgentXADKPlugin. "
        "Install it with: pip install \"agentx-python[google-adk]\""
    ) from exc


def _content_to_text(content: Any) -> Optional[str]:
    """
    Extract plain text from a google.genai types.Content object, falling back
    to a description of any function_call parts when there's no text (Gemini
    function calling — the model responded with a pure tool call).
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None)
    if not parts:
        return None
    texts = []
    function_calls = []
    for part in parts:
        text = getattr(part, "text", None)
        if text and isinstance(text, str):
            texts.append(text)
            continue
        fc = getattr(part, "function_call", None)
        if fc is not None:
            name = getattr(fc, "name", "unknown")
            args = getattr(fc, "args", None)
            function_calls.append(f"{name}({args})")
    if texts:
        return " ".join(texts)
    if function_calls:
        return "[tool call] " + ", ".join(function_calls)
    return None


def _contents_to_text(contents: Any) -> Optional[str]:
    """Extract plain text from an LlmRequest's ``contents`` (a list of Content)."""
    if contents is None:
        return None
    if not isinstance(contents, (list, tuple)):
        contents = [contents]
    texts = [t for t in (_content_to_text(c) for c in contents) if t]
    return "\n".join(texts) if texts else None


class AgentXADKPlugin(BasePlugin):
    """
    Google ADK plugin that sends one AgentX trace per runner invocation.

    Captures input (user message), output (final model reply), model name,
    tool calls, latency, and a performance_summary via the ADK plugin callbacks.

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
        # invocation_id → pre-buffered user input text
        # (on_user_message_callback fires *before* before_run_callback)
        self._pending_inputs: Dict[str, str] = {}
        # id(tool_context) → start time float
        self._tool_starts: Dict[int, float] = {}
        # invocation_id → stack of model call start times (FIFO)
        # ADK creates new CallbackContext objects for before/after model callbacks,
        # so we cannot use id(callback_context) as a key — use invocation_id instead.
        self._model_starts: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    async def on_user_message_callback(
        self, *, invocation_context: Any, user_message: Any
    ) -> None:
        # Called *before* before_run_callback, so self._runs doesn't exist yet.
        # Buffer the input and consume it in before_run_callback.
        inv_id = invocation_context.invocation_id
        text = _content_to_text(user_message)
        if text:
            self._pending_inputs[inv_id] = text

    async def before_run_callback(self, *, invocation_context: Any) -> None:
        inv_id = invocation_context.invocation_id
        agent_name = getattr(invocation_context.agent, "name", None) or self._agent_name
        self._runs[inv_id] = {
            "start": time.time(),
            "name": agent_name,
            "input": self._pending_inputs.pop(inv_id, None),
            "output": None,
            "model": None,
            "tool_calls": [],
            "error": None,
            "execution_steps": [],
            "perf_tool_calls": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        inv_id = invocation_context.invocation_id
        state = self._runs.pop(inv_id, None)
        if state is None:
            return
        latency_ms = int((time.time() - state["start"]) * 1000)
        perf = build_performance_summary(
            total_duration_ms=latency_ms,
            execution_steps=state["execution_steps"],
            tool_call_steps=state["perf_tool_calls"],
        )
        self._tracer._send(
            name=state["name"],
            input=state["input"],
            output=state["output"],
            latency_ms=latency_ms,
            error=state["error"],
            framework="google-adk",
            model=state["model"],
            tool_calls=state["tool_calls"] or None,
            metadata=self._metadata,
            session_id=self._session_id,
            performance_summary=perf,
            input_tokens=state["input_tokens"] or None,
            output_tokens=state["output_tokens"] or None,
        )

    # ------------------------------------------------------------------
    # Model callbacks — capture model name, output, and LLM step timing
    # ------------------------------------------------------------------

    async def before_model_callback(
        self, *, callback_context: Any, llm_request: Any
    ) -> None:
        inv_id = callback_context.get_invocation_context().invocation_id
        state = self._runs.get(inv_id)
        model = getattr(llm_request, "model", None)
        model_str = str(model) if model else None
        if state and not state["model"] and model_str:
            state["model"] = model_str
        # Push start time + this call's model/input onto the per-invocation
        # stack, so after_model_callback can pair them back up.
        # ADK creates different CallbackContext objects for before vs after, so
        # id(callback_context) cannot be used as a key across the two calls.
        if inv_id not in self._model_starts:
            self._model_starts[inv_id] = []
        self._model_starts[inv_id].append({
            "start": time.time(),
            "model": model_str,
            "input": _contents_to_text(getattr(llm_request, "contents", None)),
        })

    async def after_model_callback(
        self, *, callback_context: Any, llm_response: Any
    ) -> None:
        inv_id = callback_context.get_invocation_context().invocation_id
        state = self._runs.get(inv_id)
        # Pop the earliest queued call (FIFO — model calls are sequential)
        starts = self._model_starts.get(inv_id, [])
        call_start = starts.pop(0) if starts else None
        start_t = call_start.get("start") if call_start else None
        end_t = time.time()

        if state is None:
            return

        content = getattr(llm_response, "content", None)
        text = _content_to_text(content)
        # Keep updating — the last non-empty model reply is the final answer
        if text:
            state["output"] = text

        # Token counts for this call
        usage = getattr(llm_response, "usage_metadata", None)
        call_input_tokens = getattr(usage, "prompt_token_count", None) if usage is not None else None
        call_output_tokens = getattr(usage, "candidates_token_count", None) if usage is not None else None
        if call_input_tokens is not None:
            state["input_tokens"] += int(call_input_tokens)
        if call_output_tokens is not None:
            state["output_tokens"] += int(call_output_tokens)

        # Execution step
        if start_t is not None:
            steps = state["execution_steps"]
            steps.append({
                "name": f"LLM Call {len(steps) + 1}",
                "duration_ms": (end_t - start_t) * 1000,
                "start_time": start_t,
                "end_time": end_t,
                "model": call_start.get("model") if call_start else None,
                "input": call_start.get("input") if call_start else None,
                "output": text,
                "inputTokenSize": call_input_tokens,
                "outputTokenSize": call_output_tokens,
            })

    async def on_model_error_callback(
        self, *, callback_context: Any, llm_request: Any, error: Exception
    ) -> None:
        inv_id = callback_context.get_invocation_context().invocation_id
        state = self._runs.get(inv_id)
        # Pop the earliest queued call (FIFO — model calls are sequential),
        # same pairing after_model_callback uses, so a failed call's
        # timing/model/input isn't lost even though there's no llm_response.
        starts = self._model_starts.get(inv_id, [])
        call_start = starts.pop(0) if starts else None
        start_t = call_start.get("start") if call_start else None
        end_t = time.time()

        if state is None:
            return

        state["error"] = str(error)

        if start_t is not None:
            steps = state["execution_steps"]
            steps.append({
                "name": f"LLM Call {len(steps) + 1}",
                "duration_ms": (end_t - start_t) * 1000,
                "start_time": start_t,
                "end_time": end_t,
                "model": call_start.get("model") if call_start else None,
                "input": call_start.get("input") if call_start else None,
                "output": f"ERROR: {error}",
            })

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
        start_t = self._tool_starts.pop(id(tool_context), None)
        end_t = time.time()
        tool_name = getattr(tool, "name", "unknown")
        tool_input = _safe_serialize(tool_args)
        tool_output = str(result) if result is not None else None
        tool_call: Dict[str, Any] = {
            "name": tool_name,
            "input": tool_input,
            "output": tool_output,
        }
        if start_t is not None:
            tool_call["latency_ms"] = max(0, int((end_t - start_t) * 1000))
        state["tool_calls"].append(tool_call)
        if start_t is not None:
            state["perf_tool_calls"].append({
                "name": tool_name,
                "duration_ms": (end_t - start_t) * 1000,
                "start_time": start_t,
                "end_time": end_t,
                "input": tool_input,
                "output": tool_output,
            })

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
        start_t = self._tool_starts.pop(id(tool_context), None)
        end_t = time.time()
        tool_name = getattr(tool, "name", "unknown")
        tool_input = _safe_serialize(tool_args)
        tool_output = f"ERROR: {error}"
        tool_call: Dict[str, Any] = {
            "name": tool_name,
            "input": tool_input,
            "output": tool_output,
        }
        if start_t is not None:
            tool_call["latency_ms"] = max(0, int((end_t - start_t) * 1000))
        state["tool_calls"].append(tool_call)
        if start_t is not None:
            state["perf_tool_calls"].append({
                "name": tool_name,
                "duration_ms": (end_t - start_t) * 1000,
                "start_time": start_t,
                "end_time": end_t,
                "input": tool_input,
                "output": tool_output,
            })
