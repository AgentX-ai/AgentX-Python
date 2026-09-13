"""Regressions for the workforce credential plumbing:

- AgentX.list_workforces was a @staticmethod whose body referenced ``self`` - any non-empty
  response raised NameError before a single Workforce was constructed. It is an instance
  method now, binding the client's own credentials into each result.
- Workforce._bind used to bind only the workforce itself: its nested manager/agents issued
  their calls with env-fallback credentials against the default host.
"""

from unittest.mock import MagicMock, patch

from agentx import AgentX

_USER_WIRE = {
    "_id": "u1",
    "name": "Robin",
    "email": "robin@example.com",
    "deleted": False,
    "createdAt": "2026-01-01T00:00:00.000Z",
    "updatedAt": "2026-01-01T00:00:00.000Z",
    "avatar": "",
    "status": 1,
    "customer": "c1",
}

_WORKFORCE_WIRE = {
    "_id": "wf1",
    "name": "Support crew",
    "image": "",
    "description": "handles tickets",
    "agents": [
        {"_id": "a1", "name": "Triage"},
        {"_id": "a2", "name": "Resolver"},
    ],
    "manager": {"_id": "m1", "name": "Manager"},
    "creator": _USER_WIRE,
    "context": 5,
    "references": True,
    "createdAt": "2026-01-01T00:00:00.000Z",
    "updatedAt": "2026-01-01T00:00:00.000Z",
}


def _ok_response(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


def test_list_workforces_no_nameerror_and_binds_workforce_and_children():
    with AgentX(api_key="k-wf", base_url="http://engine:9999/api/v1") as client:
        with patch(
            "agentx.agentx.requests.get", return_value=_ok_response([_WORKFORCE_WIRE])
        ) as mock_get:
            workforces = client.list_workforces()  # used to NameError on any non-empty body

    assert mock_get.call_args.args[0] == "http://engine:9999/api/v1/access/teams"
    assert mock_get.call_args.kwargs["headers"]["x-api-key"] == "k-wf"

    assert len(workforces) == 1
    wf = workforces[0]
    assert wf._api_key == "k-wf"
    assert wf._base_url == "http://engine:9999/api/v1"

    # Children are bound too - they issue their own authenticated calls.
    bound_children = [wf.manager, *wf.agents]
    assert len(bound_children) == 3
    for child in bound_children:
        assert child._api_key == "k-wf"
        assert child._base_url == "http://engine:9999/api/v1"
