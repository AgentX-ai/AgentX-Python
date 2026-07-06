"""
CrewAI integration for AgentX production tracing.

Usage::

    from agentx.integrations.crewai import AgentXCrewObserver

    observer = AgentXCrewObserver(agentx.tracer, name="my-crew")
    result = observer.kickoff(crew, inputs={"topic": "AI"})

Or as a context manager around your own kickoff::

    with observer.observe(name="my-crew", input={"topic": "AI"}) as span:
        result = crew.kickoff(inputs={"topic": "AI"})
        span.output = result.raw

Requires: ``pip install agentx[crewai]``
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from agentx.tracing.tracer import Tracer, _safe_serialize


class AgentXCrewObserver:
    """
    Wraps a CrewAI ``Crew.kickoff()`` call and sends one trace per execution.
    Does not require any CrewAI version-specific event system.
    """

    def __init__(
        self,
        tracer: Tracer,
        name: str = "crewai-crew",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._metadata = metadata
        self._session_id = session_id

    def kickoff(self, crew: Any, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Call ``crew.kickoff(inputs=inputs)``, capture result, and send a trace.
        Returns the raw CrewAI ``CrewOutput`` object unchanged.
        """
        start = time.time()
        error: Optional[str] = None
        result = None
        try:
            result = crew.kickoff(inputs=inputs or {})
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            latency_ms = int((time.time() - start) * 1000)
            output = None
            if result is not None:
                output = getattr(result, "raw", None) or _safe_serialize(result)

            # Collect task outputs as tool_calls for observability
            tool_calls = []
            if result is not None:
                task_outputs = getattr(result, "tasks_output", []) or []
                for task_out in task_outputs:
                    tool_calls.append(
                        {
                            "name": getattr(task_out, "description", "task")[:100],
                            "output": str(getattr(task_out, "raw", ""))[:500],
                        }
                    )

            self._tracer._send(
                name=self._name,
                input=_safe_serialize(inputs) if inputs else None,
                output=output,
                latency_ms=latency_ms,
                error=error,
                framework="crewai",
                tool_calls=tool_calls or None,
                metadata=self._metadata,
                session_id=self._session_id,
            )

    def observe(
        self,
        name: Optional[str] = None,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ):
        """Return a context-manager span for manual kickoff wrapping."""
        from agentx.tracing.tracer import _TraceSpan

        return _TraceSpan(
            tracer=self._tracer,
            name=name or self._name,
            input=_safe_serialize(input) if input is not None else None,
            metadata=metadata or self._metadata,
            framework="crewai",
            session_id=session_id or self._session_id,
        )
