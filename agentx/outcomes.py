from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

from agentx.util import api_base, get_headers

logger = logging.getLogger(__name__)


class AgentXOutcomesError(Exception):
    pass


class OutcomesClient:
    """Surfaced as ``client.outcomes``: report a REAL, after-the-fact result for something an
    agent did (self-host only, ``POST /outcomes``), e.g. a ticket the agent "resolved" getting
    reopened three days later, or a human confirming an answer was correct.

    This is the ground truth behind the dashboard's Judge Calibration card: each report is
    compared against whatever verdict AgentX itself recorded for the same trace at the time
    (pattern hits, online-evaluator scores, eval-run ratings), turning "trust the LLM judge"
    into a measured agreement rate. The intended caller is usually another system (an incident
    tracker's webhook, a CRM workflow), and this method is the SDK's way to be that caller.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key

    def report(
        self,
        *,
        outcome: str,
        is_negative: bool,
        trace_id: Optional[str] = None,
        evaluation_run_result_id: Optional[str] = None,
        reason: Optional[str] = None,
        reported_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Report one real-world outcome against a trace (or an offline eval result).

        ``outcome`` is a free label in your own taxonomy ("reopened", "confirmed_good", ...);
        ``is_negative`` is the explicit polarity calibration actually compares against, since
        AgentX can't guess which of your labels are bad. One of ``trace_id`` /
        ``evaluation_run_result_id`` is required.

        Example, from an incident tracker's "reopened" webhook::

            client.outcomes.report(
                trace_id=trace_id,
                outcome="reopened",
                is_negative=True,
                reason="Customer reopened the ticket within 3 days",
                reported_by="servicenow-webhook",
            )
        """
        if not trace_id and not evaluation_run_result_id:
            raise AgentXOutcomesError("trace_id or evaluation_run_result_id is required")
        payload: Dict[str, Any] = {"outcome": outcome, "isNegative": is_negative}
        if trace_id:
            payload["traceId"] = trace_id
        if evaluation_run_result_id:
            payload["evaluationRunResultId"] = evaluation_run_result_id
        if reason:
            payload["reason"] = reason
        if reported_by:
            payload["reportedBy"] = reported_by

        resp = requests.post(
            f"{api_base()}/outcomes",
            headers={**get_headers(self._api_key), "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.reason)
            except ValueError:
                detail = resp.reason
            raise AgentXOutcomesError(f"Failed to report outcome ({resp.status_code}): {detail}")
        logger.info("Reported outcome %r (negative=%s)", outcome, is_negative)
        return resp.json().get("report", {})
