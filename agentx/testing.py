"""pytest-friendly assertions over evaluation runs.

The DeepEval-style dev loop (``assert_test`` inside a pytest suite) on top of AgentX's existing
run + CI-gate primitives: run the evaluation however you like, then make the test fail with a
readable verdict when quality drops. No plugin registration needed - it's a plain function that
raises ``AssertionError``, so it works in any test runner and any CI.

Usage::

    from agentx import AgentX
    from agentx.testing import assert_evaluation

    def test_support_agent_quality():
        client = AgentX.from_env()
        report = (
            client.evaluations
            .run(dataset_id=DATASET_ID, scorer_id=SCORER_ID, subject=SUBJECT)
            .execute(my_agent)
            .finalize()
        )
        assert_evaluation(report, min_rating=7.0, no_regression=True)

The check rides the engine's CI gate, so every pytest verdict is also recorded in the
dashboard's gate history (CI Gates tab) with ``caller="pytest"`` - a red test and the
dashboard's gate row are the same event, not two systems drifting apart.
"""

from typing import Any, List, Optional


class EvaluationAssertionError(AssertionError):
    """Raised when an evaluation run fails its quality checks.

    Subclasses ``AssertionError`` so pytest renders it as a plain test failure; carries the
    ``gate`` result for programmatic inspection in test hooks.
    """

    def __init__(self, message: str, gate: Any = None):
        super().__init__(message)
        self.gate = gate


def _format_failures(gate: Any) -> str:
    lines: List[str] = []
    checks = getattr(gate, "checks", None) or []
    for check in checks:
        get = check.get if isinstance(check, dict) else lambda k, d=None: getattr(check, k, d)
        status = "PASS" if get("passed") else "FAIL"
        lines.append(f"  [{status}] {get('name', 'check')}: {get('detail', '')}")
    average = getattr(gate, "average_rating", None)
    if average is not None:
        lines.append(f"  average rating: {average}")
    return "\n".join(lines) if lines else f"  gate: {gate!r}"


def assert_evaluation(
    report: Any,
    *,
    min_rating: Optional[float] = None,
    no_regression: bool = False,
    tolerance: Optional[float] = None,
    caller: str = "pytest",
) -> Any:
    """Assert a finalized evaluation run meets its quality floor.

    ``report`` is the finalized :class:`~agentx.evaluations.runner.EvaluationRunContext`
    returned by ``.execute(...).finalize()`` (or any object exposing the same ``.gate()``).

    - ``min_rating`` - fail when the run's average judge rating is below this floor (0-10).
    - ``no_regression`` - fail when the average dropped more than ``tolerance`` (default 0.5;
      judge scores are noisy) below the dataset's previous completed run.

    At least one check is required. Returns the ``GateResult`` on success; raises
    :class:`EvaluationAssertionError` with a per-check verdict on failure.
    """
    if min_rating is None and not no_regression:
        raise ValueError("assert_evaluation needs at least one check: min_rating and/or no_regression=True")
    gate = report.gate(
        fail_under=min_rating,
        no_regression=no_regression,
        tolerance=tolerance,
        caller=caller,
    )
    if getattr(gate, "passed", False):
        return gate
    run_id = getattr(report, "run_id", None) or getattr(getattr(report, "_run", None), "run_id", "?")
    raise EvaluationAssertionError(
        f"Evaluation run {run_id} failed its quality gate:\n{_format_failures(gate)}",
        gate=gate,
    )
