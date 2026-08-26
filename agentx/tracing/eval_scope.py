"""The eval-run scope: how traces created inside an evaluation stop passing as production.

An offline run executes the user's own agent function, and an instrumented agent traces itself -
which is exactly what makes trajectory matching and retrieval-context extraction work. But those
traces are not production traffic, and before this scope existed the burden of saying so sat on
every caller: remember ``monitor=False`` on every ``tracer.trace(...)`` inside an eval, or the
engine would double-judge each case, raise signals on synthetic questions, and count the run's
latencies into production KPIs. Nobody remembered - including our own samples.

``EvaluationRunContext.execute()`` enters this scope around the whole run. While it is active,
every trace the tracer sends is stamped:

  - ``source="eval-run"``  - the engine files it as eval traffic (excluded from monitor KPIs,
    metrics, sessions and the Live Traces default view; cost keeps it, split out)
  - ``monitor=False``      - unless the caller explicitly passed ``monitor=True``, which is
    respected as a deliberate choice
  - ``metadata.evalRunId`` - so a trace can always be walked back to the run that produced it

A ``contextvars.ContextVar`` rather than tracer state: it nests correctly, cannot leak across
concurrent runs in async code, and costs nothing when no run is active. The one known limit is
threads the agent function spawns itself - a context var does not cross a bare ``Thread()`` -
which matches the tracer's existing documented posture for user-managed threads.
"""

from contextvars import ContextVar
from typing import Optional

EVAL_RUN_SOURCE = "eval-run"

_current_eval_run_id: ContextVar[Optional[str]] = ContextVar("agentx_eval_run_id", default=None)


def enter_eval_run(run_id: str):
    """Mark the current context as inside an eval run. Returns the token for ``exit_eval_run``."""
    return _current_eval_run_id.set(run_id)


def exit_eval_run(token) -> None:
    _current_eval_run_id.reset(token)


def current_eval_run_id() -> Optional[str]:
    """The run id when inside ``execute()``, else None."""
    return _current_eval_run_id.get()
