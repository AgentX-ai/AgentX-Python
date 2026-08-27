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
        # The engine names each check under the key "check" ("fail-under" / "no-regression");
        # "name" is kept as a fallback for any older payload shape.
        lines.append(f"  [{status}] {get('check') or get('name', 'check')}: {get('detail', '')}")
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


def _format_pairwise(comparison: Any) -> str:
    summary = getattr(comparison, "summary", None)
    lines: List[str] = []
    if summary is not None:
        flip = getattr(summary, "flip_rate", None)
        lines.append(
            f"  A won {getattr(summary, 'a_wins', 0)}, B won {getattr(summary, 'b_wins', 0)}, "
            f"{getattr(summary, 'ties', 0)} tied, out of {getattr(summary, 'total', 0)}"
            + (f" (flip rate {flip})" if flip is not None else "")
        )
    for case in getattr(comparison, "cases", None) or []:
        if getattr(case, "winner", None) == "b":
            query = (getattr(case, "query", None) or "").strip()
            lines.append(f"  lost: {query[:80]} - {(getattr(case, 'justification', None) or '')[:120]}")
    return "\n".join(lines) if lines else f"  comparison: {comparison!r}"


def assert_pairwise(
    comparison: Any,
    *,
    must_win: bool = False,
    max_losses: Optional[int] = None,
    max_flip_rate: Optional[float] = None,
) -> Any:
    """Assert a head-to-head comparison went the candidate's way.

    ``comparison`` is what ``client.evaluations.compare_pairwise(a, b)`` returns; run A is the
    candidate and run B is the baseline it has to beat.

    - ``must_win`` - fail unless A won more cases than B. A tie fails: "no worse than before" is
      not the same claim as "better", and a change that cannot win its own comparison has not
      earned a green test.
    - ``max_losses`` - fail when A lost more than this many individual cases, even if it won
      overall. This is the check that catches a change that lifts the average by improving easy
      cases while breaking hard ones.
    - ``max_flip_rate`` - fail when too many verdicts reversed with the presentation order. That
      is position bias rather than quality, and it means the comparison itself is inconclusive,
      so treating it as a pass would be worse than a red test. Only meaningful for a comparison
      run with ``both_orders=True``; a comparison without it has no flip rate and this check is
      skipped rather than quietly passing.

    At least one check is required. Returns the comparison on success; raises
    :class:`EvaluationAssertionError` naming the cases that lost.
    """
    if not must_win and max_losses is None and max_flip_rate is None:
        raise ValueError(
            "assert_pairwise needs at least one check: must_win, max_losses, and/or max_flip_rate"
        )
    summary = getattr(comparison, "summary", None)
    if summary is None:
        raise ValueError("assert_pairwise expects the result of compare_pairwise()")

    failures: List[str] = []
    a_wins = getattr(summary, "a_wins", 0)
    b_wins = getattr(summary, "b_wins", 0)
    if must_win and a_wins <= b_wins:
        failures.append(f"run A did not win ({a_wins} vs {b_wins})")
    if max_losses is not None and b_wins > max_losses:
        failures.append(f"run A lost {b_wins} cases, more than the {max_losses} allowed")
    flip_rate = getattr(summary, "flip_rate", None)
    if max_flip_rate is not None and flip_rate is not None and flip_rate > max_flip_rate:
        failures.append(
            f"verdicts flipped on {flip_rate:.0%} of cases with the presentation order, above the "
            f"{max_flip_rate:.0%} allowed - this comparison is inconclusive, not a pass"
        )

    if not failures:
        return comparison
    batch_id = getattr(comparison, "batch_id", "?")
    raise EvaluationAssertionError(
        f"Head-to-head {batch_id} failed: {'; '.join(failures)}\n{_format_pairwise(comparison)}",
        gate=comparison,
    )
