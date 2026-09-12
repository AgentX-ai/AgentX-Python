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

Requires: ``pip install "agentx-python[crewai]"``
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from agentx.tracing.tracer import Tracer, _safe_serialize

logger = logging.getLogger(__name__)

# Warn once per process when CrewAI's event bus can't be imported and task timings fall back
# to the evenly-divided approximation - fabricated timings shouldn't be silent.
_warned_no_event_bus = False


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

        Real per-task timing is captured via CrewAI's event bus
        (``TaskStartedEvent``/``TaskCompletedEvent``/``TaskFailedEvent``,
        available since the ``crewai.events`` module was introduced) when
        present; on older CrewAI versions that predate it, this falls back to
        evenly dividing the total latency across tasks.
        """
        task_timings, unregister = self._start_task_timing_capture(crew)

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
            unregister()
            latency_ms = int((time.time() - start) * 1000)
            output = None
            if result is not None:
                output = getattr(result, "raw", None) or _safe_serialize(result)

            task_outputs = list(getattr(result, "tasks_output", []) or []) if result is not None else []
            execution_steps: List[Dict[str, Any]] = []

            if task_timings:
                execution_steps, _ = self._build_steps_from_timings(task_timings, task_outputs)
            elif task_outputs:
                execution_steps, _ = self._build_steps_evenly_divided(task_outputs, latency_ms, start)

            # Each task becomes its own real child span. tool_calls isn't passed to
            # _merge_child_run here: _build_steps_from_timings/_build_steps_evenly_divided both
            # build it from the exact same per-task loop as execution_steps (no separate timing of
            # its own), so passing both would double-emit each task as two child spans. No
            # `return` here (this whole method body runs inside the try's `finally`) - an
            # explicit return/break/continue in a finally block silently swallows any exception
            # propagating from crew.kickoff() above.
            # span_kind="agent": the root of a standalone crew kickoff is the agent run itself.
            with self._tracer.trace(
                self._name, metadata=self._metadata, session_id=self._session_id, span_kind="agent"
            ) as span:
                span._start = start
                if error:
                    span.set_error(error)
                span._merge_child_run(
                    execution_steps=execution_steps,
                    input=_safe_serialize(inputs) if inputs else None,
                    output=output,
                    framework="crewai",
                )

    def _start_task_timing_capture(self, crew: Any = None):
        """
        Register temporary, additive listeners on CrewAI's event bus to
        capture each task's real start/end wall-clock time, keyed by
        ``task_id`` (correct even when CrewAI runs tasks concurrently via
        ``async_execution=True`` - unlike attributing the most-recently-
        started task, which would misattribute end times under overlap).

        The event bus is a global singleton, so events from a DIFFERENT crew's
        overlapping kickoff arrive here too - listeners are scoped to ``crew``
        (event/source crew identity when the event carries it, this kickoff's
        task ids otherwise) so each trace only records its own kickoff's tasks.

        Returns ``(task_timings, unregister)``. ``task_timings`` stays empty
        (and ``unregister`` is a no-op) on CrewAI versions that predate the
        events module - callers should fall back to the evenly-divided
        approximation in that case (warned once per process, since those
        timings are fabricated).

        Uses ``crewai_event_bus.on()``/``.off()`` directly rather than
        ``scoped_handlers()`` - the latter temporarily disables *every*
        other registered listener (including the user's own and CrewAI's
        built-in ones) for the duration of the `with` block, which isn't
        what we want for a handler meant to run alongside them.
        """
        global _warned_no_event_bus
        task_timings: Dict[str, Dict[str, Any]] = {}
        try:
            # Modern shape first (the crewai.events module), then the older
            # crewai.utilities.events layout that shipped the same bus/events.
            try:
                from crewai.events.event_bus import crewai_event_bus
                from crewai.events.types.task_events import (
                    TaskCompletedEvent,
                    TaskFailedEvent,
                    TaskStartedEvent,
                )
            except ImportError:
                from crewai.utilities.events import crewai_event_bus
                from crewai.utilities.events.task_events import (
                    TaskCompletedEvent,
                    TaskFailedEvent,
                    TaskStartedEvent,
                )
        except ImportError:
            if not _warned_no_event_bus:
                _warned_no_event_bus = True
                logger.warning(
                    "CrewAI's event bus is not importable (tried crewai.events and "
                    "crewai.utilities.events) - per-task timings will be approximated by "
                    "evenly dividing the kickoff's total latency across tasks"
                )
            return task_timings, lambda: None

        # This kickoff's own task ids - the fallback scope filter when an event carries no
        # crew reference to compare against.
        own_task_ids = {
            str(getattr(task, "id", None))
            for task in (getattr(crew, "tasks", None) or [])
            if getattr(task, "id", None) is not None
        }

        def is_ours(source: Any, event: Any) -> bool:
            """Only record events that belong to THIS kickoff's crew - the bus is global,
            so a concurrent kickoff's task events land on every registered listener."""
            if crew is None:
                return True
            if source is crew:
                return True
            event_crew = getattr(event, "crew", None) or getattr(source, "crew", None)
            if event_crew is not None:
                return event_crew is crew
            if own_task_ids:
                return str(getattr(event, "task_id", None)) in own_task_ids
            return True

        def on_task_started(source: Any, event: Any) -> None:
            task_id = getattr(event, "task_id", None)
            if task_id is None or not is_ours(source, event):
                return
            task_timings[task_id] = {
                "name": getattr(event, "task_name", None),
                "start": event.timestamp.timestamp(),
                "end": None,
                "error": None,
            }

        def on_task_completed(source: Any, event: Any) -> None:
            entry = task_timings.get(getattr(event, "task_id", None))
            if entry is not None:
                entry["end"] = event.timestamp.timestamp()

        def on_task_failed(source: Any, event: Any) -> None:
            entry = task_timings.get(getattr(event, "task_id", None))
            if entry is not None:
                entry["end"] = event.timestamp.timestamp()
                entry["error"] = getattr(event, "error", None)

        crewai_event_bus.on(TaskStartedEvent)(on_task_started)
        crewai_event_bus.on(TaskCompletedEvent)(on_task_completed)
        crewai_event_bus.on(TaskFailedEvent)(on_task_failed)

        def unregister() -> None:
            crewai_event_bus.off(TaskStartedEvent, on_task_started)
            crewai_event_bus.off(TaskCompletedEvent, on_task_completed)
            crewai_event_bus.off(TaskFailedEvent, on_task_failed)

        return task_timings, unregister

    def _build_steps_from_timings(
        self,
        task_timings: Dict[str, Dict[str, Any]],
        task_outputs: List[Any],
    ) -> tuple:
        tool_calls: List[Dict[str, Any]] = []
        execution_steps: List[Dict[str, Any]] = []

        # Real execution order (correct even under concurrent tasks, unlike
        # insertion order into task_outputs for hierarchical/async crews).
        ordered = sorted(task_timings.values(), key=lambda t: t["start"])

        for i, timing in enumerate(ordered):
            # Best-effort positional correlation to tasks_output for the raw
            # output text - CrewOutput.tasks_output carries no task id to
            # match on directly. Accurate for the default sequential
            # process, where output order matches start order; for
            # hierarchical/concurrent crews a task could pair with the wrong
            # output text (timing itself stays correct either way, since
            # it's keyed by task_id, not position).
            task_out = task_outputs[i] if i < len(task_outputs) else None
            description = getattr(task_out, "description", None) if task_out is not None else None
            name = (timing.get("name") or description or f"Task {i + 1}")[:100]
            task_output_text = str(getattr(task_out, "raw", "")) if task_out is not None else None

            end = timing.get("end") or timing["start"]
            duration_ms = max(0.0, (end - timing["start"]) * 1000)
            error = timing.get("error")
            output_text = f"ERROR: {error}" if error else task_output_text

            execution_steps.append({
                "name": name,
                "duration_ms": duration_ms,
                "start_time": timing["start"],
                "end_time": end,
                "input": description,
                "output": output_text,
                # A CrewAI task is an agent turn, not a model call - without this,
                # _merge_child_run's default stamped every task span "llm".
                "kind": "agent",
            })
            if description is not None or task_output_text is not None:
                tool_calls.append({"name": name, "input": description, "output": task_output_text})

        return execution_steps, tool_calls

    def _build_steps_evenly_divided(self, task_outputs: List[Any], latency_ms: float, start: float) -> tuple:
        """
        Fallback for CrewAI versions predating the events module: no
        per-task timing is available, so attribute the total latency evenly
        across tasks so the timeline still sums to the measured wall-clock
        duration. Each synthesized step also gets a real start_time/end_time
        (consecutive per_step_s slices from the kickoff's start) so the
        dashboard timeline can position it, not just size it.
        """
        tool_calls: List[Dict[str, Any]] = []
        execution_steps: List[Dict[str, Any]] = []
        per_step_ms = latency_ms / len(task_outputs)
        per_step_s = per_step_ms / 1000.0
        for i, task_out in enumerate(task_outputs):
            description = getattr(task_out, "description", "task")
            name = description[:100]
            task_output = str(getattr(task_out, "raw", ""))
            tool_calls.append({"name": name, "input": description, "output": task_output})
            step_start = start + i * per_step_s
            execution_steps.append({
                "name": name,
                "duration_ms": per_step_ms,
                "start_time": step_start,
                "end_time": step_start + per_step_s,
                "input": description,
                "output": task_output,
            })

        return execution_steps, tool_calls

    def observe(
        self,
        name: Optional[str] = None,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        sync: bool = False,
    ):
        """Return a context-manager span for manual kickoff wrapping.

        Pass ``sync=True`` to send synchronously so ``span.trace_id`` is populated once the
        block exits - e.g. to attach the trace to an evaluation result. See ``Tracer.trace()``.
        """
        from agentx.tracing.tracer import _TraceSpan

        return _TraceSpan(
            tracer=self._tracer,
            name=name or self._name,
            input=_safe_serialize(input) if input is not None else None,
            metadata=metadata or self._metadata,
            framework="crewai",
            session_id=session_id or self._session_id,
            sync=sync,
            # Same statement kickoff() makes: the root of a crew run is the agent run itself.
            span_kind="agent",
        )
