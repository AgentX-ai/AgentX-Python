"""Unit tests for the tool schema registry client - wire-level, no engine required: the
per-call price rides the camelCase wire on create, PATCH edits are metadata-only and can clear
the price with None, and a bad price is refused locally before any request."""

from typing import Any, Dict, List

import pytest

from agentx.evaluations.client import EvaluationsClient
from agentx.evaluations.tool_schemas import ToolSchemaClient


class RecordingClient(EvaluationsClient):
    def __init__(self, responses: List[Any]):
        super().__init__(api_key="agtx_test", base_url="http://engine:4700/api/v1")
        self.calls: List[Dict[str, Any]] = []
        self._responses = responses

    def _request(self, method: str, path: str, timeout: int = 30, base=None, retry: bool = True, **kwargs: Any) -> Any:  # type: ignore[override]
        self.calls.append({"method": method, "path": path, "retry": retry, **kwargs})
        return self._responses.pop(0) if self._responses else {}


def test_create_sends_the_price_on_the_camelcase_wire_without_retry():
    client = RecordingClient([{"_id": "t1", "name": "geocode", "pricePerCallUsd": 0.25}])
    created = ToolSchemaClient(client).create(name="geocode", definition="{}", price_per_call_usd=0.25)
    call = client.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/evaluate/tool-schemas"
    assert call["retry"] is False
    assert call["json"] == {"name": "geocode", "definition": "{}", "pricePerCallUsd": 0.25}
    assert created["pricePerCallUsd"] == 0.25


def test_create_omits_the_price_when_not_given():
    client = RecordingClient([{}])
    ToolSchemaClient(client).create(name="clock", definition="{}")
    assert "pricePerCallUsd" not in client.calls[0]["json"]


def test_set_price_patches_metadata_and_none_clears_it():
    client = RecordingClient([{"ok": True}, {"ok": True}])
    tools = ToolSchemaClient(client)
    tools.set_price("t1", 0.01)
    tools.set_price("t1", None)
    assert [c["method"] for c in client.calls] == ["PATCH", "PATCH"]
    assert client.calls[0]["path"] == "/evaluate/tool-schemas/t1"
    assert client.calls[0]["json"] == {"pricePerCallUsd": 0.01}
    assert client.calls[1]["json"] == {"pricePerCallUsd": None}
    assert all(c["retry"] is False for c in client.calls)


def test_update_meta_sends_only_the_fields_given():
    client = RecordingClient([{"ok": True}])
    client.update_tool_schema_meta("t1", description="Order lookup")
    assert client.calls[0]["json"] == {"description": "Order lookup"}


def test_a_bad_price_is_refused_before_any_request():
    client = RecordingClient([])
    tools = ToolSchemaClient(client)
    with pytest.raises(ValueError, match="non-negative"):
        tools.create(name="x", definition="{}", price_per_call_usd=-1)
    with pytest.raises(ValueError, match="number"):
        tools.set_price("t1", "cheap")  # type: ignore[arg-type]
    assert client.calls == []


def test_delete_never_retries():
    client = RecordingClient([{}])
    ToolSchemaClient(client).delete("t1")
    assert client.calls[0]["method"] == "DELETE"
    assert client.calls[0]["path"] == "/evaluate/tool-schemas/t1"
    assert client.calls[0]["retry"] is False


def test_get_or_create_keeps_the_existing_row_and_its_price():
    client = RecordingClient([{"toolSchemas": [{"_id": "t1", "name": "geocode", "pricePerCallUsd": 0.25}]}])
    existing = ToolSchemaClient(client).get_or_create(name="geocode", definition="{}", price_per_call_usd=9)
    assert existing["pricePerCallUsd"] == 0.25
    assert [c["method"] for c in client.calls] == ["GET"]
