from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

from agentx.util import api_base, get_headers

logger = logging.getLogger(__name__)


class AgentXFeedbackError(Exception):
    pass


class FeedbackClient:
    """Surfaced as ``client.feedback``: forward an END USER's reaction to a traced response
    (self-host only, ``POST /feedback``) - the vote button in your own app's UI, relayed to
    AgentX. The cheapest ground truth there is: a "down" raises a "Negative user feedback"
    signal directly (the user is the detector, no sampling or judge call), and every report
    also feeds Judge Calibration, so AgentX's automated verdicts get measured against real
    human reactions.

    Distinct from ``client.outcomes``: an outcome is an after-the-fact SYSTEM result ("ticket
    reopened", reported by a workflow); feedback is a human vote with up/down semantics.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key

    def report(
        self,
        trace_id: str,
        rating: str,
        *,
        comment: Optional[str] = None,
        end_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Report one end-user vote on a trace.

        ``rating`` is ``"up"`` or ``"down"``. ``comment`` is the user's own words, if your UI
        collects them (a "down" comment becomes the signal's summary and improvement evidence).
        ``end_user_id`` is your app's identifier for the voter, kept opaque by AgentX.

        Example, from your app's vote handler::

            client.feedback.report(
                trace_id=trace_id,
                rating="down",
                comment="It never answered my question",
                end_user_id=current_user.id,
            )
        """
        if rating not in ("up", "down"):
            raise AgentXFeedbackError('rating must be "up" or "down"')
        payload: Dict[str, Any] = {"traceId": trace_id, "rating": rating}
        if comment:
            payload["comment"] = comment
        if end_user_id:
            payload["endUserId"] = end_user_id

        resp = requests.post(
            f"{api_base()}/feedback",
            headers={**get_headers(self._api_key), "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.reason)
            except ValueError:
                detail = resp.reason
            raise AgentXFeedbackError(f"Failed to report feedback ({resp.status_code}): {detail}")
        logger.info("Reported %s feedback on trace %s", rating, trace_id)
        return resp.json().get("feedback", {})
