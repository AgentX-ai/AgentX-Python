"""
Microsoft AutoGen integration for AgentX production tracing.

Targets the modern ``autogen-agentchat``/``autogen-core`` architecture (the
actively maintained v0.4+ rewrite), not the older ``pyautogen``/``ag2`` fork.

Usage::

    from agentx.integrations.autogen import AgentXAutoGenObserver
    from autogen_agentchat.agents import AssistantAgent

    agent = AssistantAgent("assistant", model_client=model_client)
    observer = AgentXAutoGenObserver(agentx.tracer, name="my-agent")

    result = await observer.run(agent, task="What's the weather in NYC?")
    # Works the same for a Team: await observer.run(team, task=...)

Requires: ``pip install "agentx-python[autogen]"``
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agentx.tracing.tracer import Tracer, _safe_serialize


def _message_text(message: Any) -> Optional[str]:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if content is not None:
        return _safe_serialize(content)
    return None


def _message_tokens(message: Any) -> tuple:
    usage = getattr(message, "models_usage", None)
    if usage is None:
        return None, None
    return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)


def _function_calls(message: Any) -> List[Any]:
    content = getattr(message, "content", None)
    return list(content) if isinstance(content, list) else []


class AgentXAutoGenObserver:
    """
    Wraps an ``AssistantAgent``/``Team`` ``.run(task=...)`` call and sends one
    AgentX trace per run.

    Scope: covers ``.run()`` only (returns the full ``TaskResult`` up front),
    not ``.run_stream()`` - the same "trace the request/response call, leave
    the streaming variant as a deliberate follow-up" boundary
    ``patch_openai_client``/``google_genai``'s async-streaming gap already
    draw elsewhere in this package, rather than silently half-supporting it.

    Per-step timing is derived from each ``BaseChatMessage``'s real
    ``created_at`` timestamp (chained against the previous message's
    timestamp, or the run's own start for the first one) - AutoGen's message
    schema has no explicit start/end pair per LLM call, so this is real but
    approximate, the same caveat CrewAI's per-task timing has for its
    positional tasks_output correlation.
    """

    def __init__(
        self,
        tracer: Tracer,
        name: str = "autogen-agent",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._metadata = metadata
        self._session_id = session_id

    async def run(self, agent_or_team: Any, task: Any = None, **kwargs: Any) -> Any:
        """
        Call ``agent_or_team.run(task=task, **kwargs)``, capture the result,
        and send a trace. Returns the raw ``TaskResult`` unchanged.
        """
        start_t = time.time()
        error: Optional[str] = None
        result = None
        try:
            result = await agent_or_team.run(task=task, **kwargs)
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            messages = getattr(result, "messages", None) or []
            execution_steps, tool_call_steps, output_text, input_tokens, output_tokens = self._summarize_messages(
                messages, start_t
            )
            # Read the input back off messages[0] rather than the `task` arg
            # directly - `task` can be a plain str, a single BaseChatMessage,
            # or a sequence of them (per AssistantAgent.run's signature), but
            # TaskResult.messages[0] is always a real message object with a
            # normalized .content, regardless of which form was passed in.
            input_text = _message_text(messages[0]) if messages else (task if isinstance(task, str) else None)

            # No `return` here - this whole block runs inside the try's `finally`, and an
            # explicit return/break/continue there would silently swallow any exception
            # propagating from agent_or_team.run() above (see crewai.py's kickoff() for the same
            # hazard spelled out in full).
            with self._tracer.trace(self._name, metadata=self._metadata, session_id=self._session_id) as span:
                span._start = start_t
                if error:
                    span.set_error(error)
                span._merge_child_run(
                    execution_steps=execution_steps,
                    tool_calls=tool_call_steps,
                    input=input_text,
                    output=output_text,
                    framework="autogen",
                    input_tokens=input_tokens or None,
                    output_tokens=output_tokens or None,
                )

    def _summarize_messages(self, messages: List[Any], run_start: float) -> tuple:
        execution_steps: List[Dict[str, Any]] = []
        tool_call_steps: List[Dict[str, Any]] = []
        output_text: Optional[str] = None
        total_input_tokens = 0
        total_output_tokens = 0

        prev_t = run_start
        # call_id -> {"name", "input", "start_time"}, filled in by a
        # ToolCallRequestEvent and consumed by the matching
        # ToolCallExecutionEvent so each tool call gets one combined step.
        pending_tool_calls: Dict[str, Dict[str, Any]] = {}

        for message in messages:
            msg_type = getattr(message, "type", None)
            created_at = getattr(message, "created_at", None)
            end_t = created_at.timestamp() if created_at is not None else prev_t
            start_t = prev_t
            prev_t = end_t

            if msg_type == "ToolCallRequestEvent":
                for call in _function_calls(message):
                    call_id = getattr(call, "id", None)
                    if call_id is None:
                        continue
                    pending_tool_calls[call_id] = {
                        "name": getattr(call, "name", "unknown"),
                        "input": getattr(call, "arguments", None),
                        "start_time": start_t,
                    }
                continue

            if msg_type == "ToolCallExecutionEvent":
                for result_item in _function_calls(message):
                    call_id = getattr(result_item, "call_id", None)
                    pending = pending_tool_calls.pop(call_id, None) if call_id is not None else None
                    tool_start = pending["start_time"] if pending else start_t
                    is_error = getattr(result_item, "is_error", False)
                    output = getattr(result_item, "content", None)
                    tool_call_steps.append({
                        "name": pending["name"] if pending else getattr(result_item, "name", "unknown"),
                        "duration_ms": (end_t - tool_start) * 1000,
                        "start_time": tool_start,
                        "end_time": end_t,
                        "input": pending["input"] if pending else None,
                        "output": f"ERROR: {output}" if is_error else (str(output) if output is not None else None),
                    })
                continue

            # models_usage is only set on messages a real LLM call produced
            # (TextMessage / ToolCallRequestEvent) - user-authored task
            # messages and ToolCallSummaryMessage don't have it.
            input_tokens, out_tokens = _message_tokens(message)
            text = _message_text(message)
            if input_tokens is not None or out_tokens is not None:
                total_input_tokens += input_tokens or 0
                total_output_tokens += out_tokens or 0
                execution_steps.append({
                    "name": f"LLM Call {len(execution_steps) + 1}",
                    "duration_ms": (end_t - start_t) * 1000,
                    "start_time": start_t,
                    "end_time": end_t,
                    "input": None,
                    "output": text,
                    "inputTokenSize": input_tokens,
                    "outputTokenSize": out_tokens,
                })

            if text:
                output_text = text

        return execution_steps, tool_call_steps, output_text, total_input_tokens, total_output_tokens
