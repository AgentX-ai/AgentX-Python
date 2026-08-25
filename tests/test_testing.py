"""agentx.testing.assert_evaluation - the pytest-native quality gate wrapper."""

import pytest

from agentx.evaluations.runner import GateResult
from agentx.testing import EvaluationAssertionError, assert_evaluation


class FakeReport:
    def __init__(self, gate_payload):
        self.run_id = "run-123"
        self.gate_kwargs = None
        self._payload = gate_payload

    def gate(self, **kwargs):
        self.gate_kwargs = kwargs
        return GateResult(self._payload)


def test_passing_gate_returns_result_and_forwards_checks():
    report = FakeReport({"passed": True, "averageRating": 8.2, "checks": [{"name": "floor", "passed": True}]})
    gate = assert_evaluation(report, min_rating=7.0, no_regression=True, tolerance=0.3)
    assert gate.passed is True
    assert report.gate_kwargs == {"fail_under": 7.0, "no_regression": True, "tolerance": 0.3, "caller": "pytest"}


def test_failing_gate_raises_assertion_error_with_verdict():
    report = FakeReport({
        "passed": False,
        "averageRating": 5.1,
        "checks": [
            {"name": "floor", "passed": False, "detail": "average 5.1 below fail_under 7"},
            {"name": "regression", "passed": True, "detail": "no baseline"},
        ],
    })
    with pytest.raises(EvaluationAssertionError) as excinfo:
        assert_evaluation(report, min_rating=7.0)
    message = str(excinfo.value)
    assert "run-123" in message
    assert "[FAIL] floor" in message
    assert "average 5.1 below fail_under 7" in message
    # AssertionError subclass, so pytest treats it as a normal test failure.
    assert isinstance(excinfo.value, AssertionError)
    assert excinfo.value.gate.average_rating == 5.1


def test_requires_at_least_one_check():
    with pytest.raises(ValueError):
        assert_evaluation(FakeReport({"passed": True}))
