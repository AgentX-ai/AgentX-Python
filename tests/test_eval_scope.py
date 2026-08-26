"""The eval-run scope: traces created inside execute() stop passing as production.

Before this, the burden sat on every caller: remember monitor=False on every trace inside an
eval or the engine double-judges, raises signals on synthetic questions, and counts eval
latencies into production KPIs. These pin that the stamping is automatic, respects an explicit
monitor=True, and vanishes completely outside the scope.
"""

from agentx.tracing.eval_scope import current_eval_run_id, enter_eval_run, exit_eval_run
from agentx.tracing.tracer import Tracer


class _CaptureTracer(Tracer):
    """A tracer whose network is a list."""

    def __init__(self):  # noqa: D401 - bypass real client setup
        self.sent = []
        self._active_spans = []
        self._pending_tool_calls = []
        self._pending_retrievals = []

    def _send(self, sync=False, **kwargs):
        self.sent.append({k: v for k, v in kwargs.items() if v is not None})
        return "trace-1"

    def _dispatch(self, wire, *, sync=False):
        self.sent.append(wire)
        return "trace-child"

    # The bits of Tracer the span touches.
    def _push_active_span(self, span):
        self._active_spans.append(span)

    def _pop_active_span(self, span):
        if span in self._active_spans:
            self._active_spans.remove(span)

    @property
    def current_span(self):
        return self._active_spans[-1] if self._active_spans else None

    def flush(self, timeout=5.0):
        return True


def test_outside_the_scope_nothing_changes():
    tracer = _CaptureTracer()
    with tracer.trace("prod-agent", input={"q": "hi"}) as span:
        span.output = "hello"
    wire = tracer.sent[-1]
    assert "source" not in wire
    assert "monitor" not in wire  # None is filtered out, same as before
    assert current_eval_run_id() is None


def test_inside_the_scope_traces_are_stamped():
    tracer = _CaptureTracer()
    token = enter_eval_run("run-42")
    try:
        with tracer.trace("agent-under-eval", input={"q": "case 1"}) as span:
            span.output = "answer"
    finally:
        exit_eval_run(token)
    wire = tracer.sent[-1]
    assert wire["source"] == "eval-run"
    assert wire["monitor"] is False
    assert wire["metadata"]["evalRunId"] == "run-42"


def test_explicit_monitor_true_is_respected():
    # Someone deliberately pointing checks at eval traffic is a choice, not a mistake.
    tracer = _CaptureTracer()
    token = enter_eval_run("run-42")
    try:
        with tracer.trace("agent-under-eval", monitor=True) as span:
            span.output = "x"
    finally:
        exit_eval_run(token)
    wire = tracer.sent[-1]
    assert wire["source"] == "eval-run"
    assert wire["monitor"] is True


def test_child_spans_are_eval_traffic_too():
    tracer = _CaptureTracer()
    token = enter_eval_run("run-42")
    try:
        with tracer.trace("agent-under-eval") as span:
            span.child_span("kb_search", output=["chunk"])
            span.output = "x"
    finally:
        exit_eval_run(token)
    child = next(w for w in tracer.sent if w.get("name") == "kb_search")
    assert child["source"] == "eval-run"


def test_the_scope_does_not_leak():
    tracer = _CaptureTracer()
    token = enter_eval_run("run-42")
    exit_eval_run(token)
    with tracer.trace("prod-again") as span:
        span.output = "y"
    assert "source" not in tracer.sent[-1]


def test_caller_metadata_survives_the_stamp():
    tracer = _CaptureTracer()
    token = enter_eval_run("run-42")
    try:
        with tracer.trace("agent", metadata={"promptName": "support-v3"}) as span:
            span.output = "x"
    finally:
        exit_eval_run(token)
    md = tracer.sent[-1]["metadata"]
    assert md["promptName"] == "support-v3"
    assert md["evalRunId"] == "run-42"
