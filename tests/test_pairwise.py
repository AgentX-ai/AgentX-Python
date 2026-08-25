"""Head-to-head judging: the client's request shape and the pytest assertion over the result."""

import pytest

from agentx.evaluations.models import PairwiseComparison
from agentx.testing import EvaluationAssertionError, assert_pairwise


def comparison(**over) -> PairwiseComparison:
    payload = {
        "batchId": "batch-1",
        "runAId": "run-candidate",
        "runBId": "run-baseline",
        "bothOrders": False,
        "judgeModel": "gpt-5.6-luna",
        "summary": {"total": 3, "aWins": 2, "bWins": 1, "ties": 0, "winner": "a", "flipRate": None},
        "cases": [
            {
                "questionIndex": 2,
                "query": "Who pays return shipping?",
                "winner": "b",
                "presentedFirst": "a",
                "justification": "Answer 2 names both cases explicitly.",
            }
        ],
        "skipped": [],
    }
    payload.update(over)
    return PairwiseComparison(**payload)


class FakeClient:
    """Captures the request instead of sending it - the wire shape is the contract with the
    engine, and camelCase is the convention it has to keep."""

    # The pairwise routes live on the /evaluate dialect, reached through this property.
    _api_root = "https://engine.example/api/v1"

    def __init__(self, response=None):
        self.calls = []
        self._response = response or {"comparison": comparison().model_dump(by_alias=True)}

    def _request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return self._response


def test_compare_pairwise_sends_camelcase_and_omits_unset_options():
    from agentx.evaluations.client import EvaluationsClient

    client = FakeClient()
    result = EvaluationsClient.compare_pairwise(client, "run-candidate", "run-baseline")

    method, path, kwargs = client.calls[0]
    assert (method, path) == ("POST", "/evaluate/runs/pairwise")
    # Defaults are the server's to choose; the SDK does not invent a criteria string or a
    # judge model, and does not send bothOrders unless the caller asked for it.
    assert kwargs["json"] == {"runAId": "run-candidate", "runBId": "run-baseline"}
    assert result.summary.a_wins == 2
    assert result.cases[0].presented_first == "a"


def test_compare_pairwise_forwards_the_options_it_is_given():
    from agentx.evaluations.client import EvaluationsClient

    client = FakeClient()
    EvaluationsClient.compare_pairwise(
        client, "a", "b", criteria="Which is more concise?", judge_model="gpt-5.6-luna", both_orders=True
    )
    assert client.calls[0][2]["json"] == {
        "runAId": "a",
        "runBId": "b",
        "criteria": "Which is more concise?",
        "judgeModel": "gpt-5.6-luna",
        "bothOrders": True,
    }


def test_assert_pairwise_passes_a_clear_win():
    result = assert_pairwise(comparison(), must_win=True, max_losses=1)
    assert result.summary.winner == "a"


def test_a_tie_is_not_a_win():
    tied = comparison(summary={"total": 2, "aWins": 1, "bWins": 1, "ties": 0, "winner": "tie", "flipRate": None})
    with pytest.raises(EvaluationAssertionError) as excinfo:
        assert_pairwise(tied, must_win=True)
    assert "did not win" in str(excinfo.value)


def test_max_losses_catches_a_win_that_broke_hard_cases():
    # Wins overall, but lost more individual cases than the caller tolerates.
    lossy = comparison(summary={"total": 10, "aWins": 5, "bWins": 4, "ties": 1, "winner": "a", "flipRate": None})
    with pytest.raises(EvaluationAssertionError) as excinfo:
        assert_pairwise(lossy, max_losses=2)
    message = str(excinfo.value)
    assert "lost 4 cases" in message
    # The failure names the case that lost, so the test output is actionable on its own.
    assert "Who pays return shipping?" in message


def test_a_high_flip_rate_fails_instead_of_passing_on_position_bias():
    biased = comparison(
        bothOrders=True,
        summary={"total": 4, "aWins": 3, "bWins": 1, "ties": 0, "winner": "a", "flipRate": 0.5},
    )
    with pytest.raises(EvaluationAssertionError) as excinfo:
        assert_pairwise(biased, max_flip_rate=0.2)
    assert "inconclusive" in str(excinfo.value)


def test_flip_rate_check_is_skipped_when_both_orders_was_not_run():
    # No flip rate exists to check, so this must not silently fail or silently pass a made-up 0.
    assert_pairwise(comparison(), max_flip_rate=0.0)


def test_requires_at_least_one_check():
    with pytest.raises(ValueError):
        assert_pairwise(comparison())
