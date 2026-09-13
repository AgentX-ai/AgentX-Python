"""patterns.update against the engine's REAL wire shape: the GET body's flat condition fields
are display-only placeholders (toWire hardcodes includeTerms/excludeTerms to [] and matchMode to
"any"; regex/semanticPrompt are omitted entirely - conditions is the only truth), and the PUT
rebuilds the WHOLE conditions array exactly when the body carries includeTerms/regex/
semanticPrompt (the engine's sentConditionFields set, minus conditions).

So update() must infer detectorKind from the trigger field actually sent (regex -> "regex",
semantic_prompt -> "semantic", else "contains" - back-filling the STORED kind 400s a kind
change and corrupts the inverse), back-fill ONLY matchTarget from the stored row, never copy
the placeholder includeTerms/excludeTerms/matchMode into the payload (the old merge destroyed
real conditions that way), and refuse exclude_terms=/match_mode=/match_target= sent alone
(alone the engine silently ignores them)."""

from unittest.mock import MagicMock

import pytest

from agentx.monitor.patterns import MonitorPattern, MonitorPatternClient


def _wire_pattern(**overrides):
    """A GET /patterns/:id body exactly as the self-host engine's toWire emits it: flat
    condition fields are placeholders, the real rule lives in `conditions`."""
    body = {
        "_id": "p1",
        "workspaceId": "local",
        "key": "ssn-in-response",
        "name": "SSN in response",
        "source": "custom",
        "detectorKind": "regex",
        "matchTarget": ["response"],
        "matchMode": "any",  # toWire hardcodes this
        "includeTerms": [],  # toWire hardcodes this
        "excludeTerms": [],  # toWire hardcodes this
        # No "regex" / "semanticPrompt" keys at all - the wire omits them.
        "conditions": [
            {
                "connector": "and",
                "negate": False,
                "sources": ["response"],
                "detector": "regex",
                "value": r"\d{3}-\d{2}-\d{4}",
                "caseSensitive": False,
            }
        ],
        "severity": "high",
        "polarity": "failure",
        "enabled": True,
        "sampleRate": 1.0,
        "scopeMode": "all",
        "agentIds": [],
        "readOnly": False,
        "createdAt": "2026-01-01T00:00:00.000Z",
        "updatedAt": "2026-01-01T00:00:00.000Z",
    }
    body.update(overrides)
    return body


CONTAINS_WIRE = _wire_pattern(
    key="refund-promise",
    name="Refund promise",
    detectorKind="contains",
    conditions=[
        {
            "connector": "and",
            "negate": False,
            "sources": ["response"],
            "detector": "phrase",
            "value": "guaranteed refund",
            "caseSensitive": False,
        },
        {
            "connector": "or",
            "negate": False,
            "sources": ["response"],
            "detector": "phrase",
            "value": "money back",
            "caseSensitive": False,
        },
    ],
)


def make_client(wire_body=None):
    inner = MagicMock()
    inner._api_root.return_value = "http://x/api/v1"
    client = MonitorPatternClient(inner)
    body = wire_body or _wire_pattern()
    client.get = MagicMock(return_value=MonitorPattern(**body))  # type: ignore[method-assign]
    inner._request.return_value = {"pattern": body}
    return client, inner


def test_regex_only_update_sends_regex_detector_kind_and_match_target():
    client, inner = make_client()
    client.update("p1", regex=r"\b\d{9}\b")
    payload = inner._request.call_args.kwargs["json"]
    assert payload == {
        "regex": r"\b\d{9}\b",
        "detectorKind": "regex",
        "matchTarget": ["response"],
    }
    client.get.assert_called_once_with("p1")  # type: ignore[union-attr]


def test_include_terms_update_never_sends_placeholder_siblings():
    client, inner = make_client(CONTAINS_WIRE)
    client.update("p1", include_terms=["guaranteed refund", "money back", "full refund"])
    payload = inner._request.call_args.kwargs["json"]
    assert payload["includeTerms"] == ["guaranteed refund", "money back", "full refund"]
    assert payload["detectorKind"] == "contains"
    assert payload["matchTarget"] == ["response"]
    # The wire GET's excludeTerms/matchMode are display-only placeholders ([] / "any") -
    # copying them in is exactly the merge that destroyed conditions.
    assert "excludeTerms" not in payload
    assert "matchMode" not in payload


def test_cross_kind_update_sends_the_sent_fields_kind_not_the_stored_one():
    """update(pid, regex=...) on a stored "contains" pattern must send detectorKind "regex" -
    back-filling the stored kind 400s the change (and the inverse, include_terms= on a regex
    pattern, would silently write phrase conditions under detectorKind "regex")."""
    client, inner = make_client(CONTAINS_WIRE)
    client.update("p1", regex=r"\b\d{9}\b")
    payload = inner._request.call_args.kwargs["json"]
    assert payload == {
        "regex": r"\b\d{9}\b",
        "detectorKind": "regex",
        "matchTarget": ["response"],
    }


def test_semantic_prompt_update_sends_semantic_kind():
    client, inner = make_client(CONTAINS_WIRE)
    client.update("p1", semantic_prompt="The response promises a refund.")
    payload = inner._request.call_args.kwargs["json"]
    assert payload["detectorKind"] == "semantic"


def test_match_target_only_raises_instead_of_silently_doing_nothing():
    """matchTarget sent alone never lands: the engine only reads it during a conditions
    rebuild, and a body without a trigger field never rebuilds."""
    client, inner = make_client()
    with pytest.raises(ValueError, match="matchTarget"):
        client.update("p1", match_target=["input", "response"])
    inner._request.assert_not_called()
    client.get.assert_not_called()  # type: ignore[union-attr]


def test_exclude_terms_only_raises_instead_of_silently_doing_nothing():
    client, inner = make_client()
    with pytest.raises(ValueError, match="conditions"):
        client.update("p1", exclude_terms=["as an AI"])
    inner._request.assert_not_called()
    client.get.assert_not_called()  # type: ignore[union-attr]


def test_match_mode_only_raises_too():
    client, inner = make_client()
    with pytest.raises(ValueError, match="matchMode"):
        client.update("p1", match_mode="all")
    inner._request.assert_not_called()


def test_exclude_terms_alongside_a_trigger_field_is_allowed():
    """exclude_terms rides along fine when the rebuild actually happens (a trigger field is
    present) - the engine folds them into the rebuilt conditions."""
    client, inner = make_client(CONTAINS_WIRE)
    client.update("p1", include_terms=["guaranteed refund"], exclude_terms=["hypothetically"])
    payload = inner._request.call_args.kwargs["json"]
    assert payload["includeTerms"] == ["guaranteed refund"]
    assert payload["excludeTerms"] == ["hypothetically"]
    assert "matchMode" not in payload


def test_explicit_conditions_win_outright_no_read_back():
    stored_conditions = [
        {
            "connector": "and",
            "negate": False,
            "sources": ["response"],
            "detector": "phrase",
            "value": "wire transfer",
            "caseSensitive": False,
        }
    ]
    client, inner = make_client()
    client.update("p1", conditions=stored_conditions, exclude_terms=["test"])
    payload = inner._request.call_args.kwargs["json"]
    assert payload["conditions"] == stored_conditions
    client.get.assert_not_called()  # type: ignore[union-attr]


def test_non_condition_edit_stays_sparse():
    client, inner = make_client()
    client.update("p1", sample_rate=0.05)
    payload = inner._request.call_args.kwargs["json"]
    assert payload == {"sampleRate": 0.05}
    client.get.assert_not_called()  # type: ignore[union-attr]
