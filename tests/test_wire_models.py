"""Wire-model regressions: Dataset round-tripping its grading config (import_dataset used to
silently drop it) and RunResultRow.response populating from the engine's `output` object."""

from typing import Any, Dict

from agentx.evaluations.models import Dataset, RunResultRow


DATASET_WIRE: Dict[str, Any] = {
    "_id": "ds-1",
    "name": "support quality",
    "description": "d",
    "numberOfRequests": 2,
    "acceptanceCriteria": "acc",
    "questions": [{"main_question": {"query": "q0"}}],
    "vectorSimilarity": {"enabled": True, "model": "text-embedding-3-small"},
    "jaccardSimilarity": {"enabled": True},
    "bleuScore": {"enabled": True},
    "rougeScore": {"enabled": True},
    "judgePrompt": "Grade strictly.",
    "judgeModel": "gpt-5.6-luna",
    "sovereigntyIndex": {"enabled": True, "models": ["m1", "m2"]},
    "codeScorers": [{"id": "cs1", "name": "n", "code": "c", "enabled": True}],
}


def test_dataset_round_trips_metrics_judge_and_sovereignty_config():
    """P2 regression: extra="ignore" dropped these fields on read, so a typed Dataset fed to
    import_dataset produced a copy with no metrics/judge/sovereignty config."""
    ds = Dataset(**DATASET_WIRE)
    dumped = ds.model_dump(by_alias=True)
    for key in (
        "vectorSimilarity",
        "jaccardSimilarity",
        "bleuScore",
        "rougeScore",
        "judgePrompt",
        "judgeModel",
        "sovereigntyIndex",
        "codeScorers",
    ):
        assert dumped[key] == DATASET_WIRE[key], key
    # The hoisted convenience list still works alongside the raw object.
    assert ds.sovereignty_models == ["m1", "m2"]


def test_import_dataset_copies_the_full_grading_config():
    from agentx.evaluations.datasets import DatasetClient

    class FakeEvalClient:
        def create_dataset(self, payload):
            self.payload = payload
            return payload

    fake = FakeEvalClient()
    DatasetClient(fake).import_dataset(Dataset(**DATASET_WIRE), name="copy")
    payload = fake.payload
    assert payload["name"] == "copy"
    assert payload["vectorSimilarity"] == DATASET_WIRE["vectorSimilarity"]
    assert payload["jaccardSimilarity"] == DATASET_WIRE["jaccardSimilarity"]
    assert payload["bleuScore"] == DATASET_WIRE["bleuScore"]
    assert payload["rougeScore"] == DATASET_WIRE["rougeScore"]
    assert payload["judgePrompt"] == DATASET_WIRE["judgePrompt"]
    assert payload["judgeModel"] == DATASET_WIRE["judgeModel"]
    assert payload["sovereigntyIndex"] == DATASET_WIRE["sovereigntyIndex"]
    assert payload["codeScorers"] == DATASET_WIRE["codeScorers"]


def test_run_result_row_response_populates_from_output_object():
    """P2 regression: the engine sends the agent's answer as an `output` object, so
    row.response was permanently None."""
    row = RunResultRow.from_wire({
        "rating": 8,
        "output": {"text": "the answer", "tokens": 12},
        "questionIndex": 0,
    })
    assert row.response == "the answer"
    assert row.rating == 8


def test_run_result_row_response_still_accepts_plain_string():
    row = RunResultRow.from_wire({"response": "plain answer"})
    assert row.response == "plain answer"
