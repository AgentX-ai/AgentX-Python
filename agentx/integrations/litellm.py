"""
LiteLLM integration for AgentX production tracing.

Usage::

    from agentx.integrations.litellm import AgentXLiteLLMLogger
    import litellm

    litellm.callbacks = [AgentXLiteLLMLogger(agentx.tracer, name="my-agent")]

    # All subsequent litellm.completion()/acompletion() calls are now traced
    # automatically — sync, async, and streaming.

Requires: ``pip install "agentx-python[litellm]"``
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from agentx.tracing.tracer import Tracer, _safe_serialize
from agentx.integrations._traced_call import finish_llm_call

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "litellm is required for AgentXLiteLLMLogger. "
        "Install it with: pip install \"agentx-python[litellm]\""
    ) from exc


def _extract_output_text(response: Any) -> Optional[str]:
    """
    Extract the assistant's text reply from a LiteLLM ``ModelResponse``,
    falling back to a description of any tool calls when the response is a
    pure tool call with no accompanying text.
    """
    choices = getattr(response, "choices", None) or []
    texts = []
    tool_call_descriptions = []
    for choice in choices:
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content:
            texts.append(content)
        tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        for tc in tool_calls or []:
            fn = getattr(tc, "function", None)
            fn_name = getattr(fn, "name", "unknown") if fn is not None else "unknown"
            fn_args = getattr(fn, "arguments", None) if fn is not None else None
            tool_call_descriptions.append(f"{fn_name}({fn_args})")
    if texts:
        return "\n".join(texts)
    if tool_call_descriptions:
        return "[tool call] " + ", ".join(tool_call_descriptions)
    return None


def _extract_usage_tokens(response: Any) -> Tuple[Optional[int], Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)


class AgentXLiteLLMLogger(CustomLogger):
    """
    LiteLLM ``CustomLogger`` that sends one AgentX trace per completion call.

    Covers ``litellm.completion``/``acompletion``, streaming or not — LiteLLM
    reassembles a streamed response into one final ``ModelResponse`` before
    invoking these callbacks, so no separate streaming handling is needed
    here, and no ``call_and_trace``-style async-coroutine detection either:
    LiteLLM's own sync/async callback split (``log_*_event`` vs
    ``async_log_*_event``) already tells you which one fired.

    Register via ``litellm.callbacks`` — affects every call in the process
    from that point on, the same scope ``litellm.callbacks`` itself has.
    """

    def __init__(
        self,
        tracer: Tracer,
        name: str = "litellm-agent",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._metadata = metadata
        self._session_id = session_id

    def _finish(self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any, error: Optional[str]) -> None:
        model = kwargs.get("model")
        input_repr = _safe_serialize(kwargs.get("messages"))
        output = None
        input_tokens = None
        output_tokens = None
        if error is None and response_obj is not None:
            output = _extract_output_text(response_obj)
            input_tokens, output_tokens = _extract_usage_tokens(response_obj)

        finish_llm_call(
            self._tracer,
            name=self._name,
            framework="litellm",
            metadata=self._metadata,
            session_id=self._session_id,
            start_t=start_time.timestamp(),
            end_t=end_time.timestamp(),
            input_repr=input_repr,
            output=output,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )

    @staticmethod
    def _extract_error(kwargs: Dict[str, Any]) -> str:
        exc = kwargs.get("exception")
        return str(exc) if exc is not None else "litellm call failed"

    # ------------------------------------------------------------------
    # CustomLogger protocol
    # ------------------------------------------------------------------

    def log_success_event(self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        self._finish(kwargs, response_obj, start_time, end_time, error=None)

    async def async_log_success_event(
        self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._finish(kwargs, response_obj, start_time, end_time, error=None)

    def log_failure_event(self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        self._finish(kwargs, response_obj, start_time, end_time, error=self._extract_error(kwargs))

    async def async_log_failure_event(
        self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._finish(kwargs, response_obj, start_time, end_time, error=self._extract_error(kwargs))
