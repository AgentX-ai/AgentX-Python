"""Shared HTTP transport for the monitor sub-clients that own their ``_request`` (scorers,
judge_scorers, scorer_groups, improvement_groups): one retry schedule mirroring
``MonitorClient._request``, so ``retry=False`` means the same thing everywhere the client.py
comment promises it ("retry=False for ANY non-idempotent write")."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# Same schedule as MonitorClient._request (agentx/monitor/client.py).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRY_BACKOFF = [1.0, 2.0, 4.0]


def request_with_retries(
    method: str, url: str, *, retry: bool = True, **kwargs: Any
) -> requests.Response:
    """``requests.request`` with MonitorClient's transport posture: when ``retry`` is true,
    connection errors and retryable statuses (429/5xx) walk the backoff schedule; the last
    response (whatever its status) is returned for the caller's own error taxonomy.

    ``retry=False`` is single-shot - for non-idempotent writes (creates, deletes) and
    judge-billing POSTs, where a client-side timeout must not fire the same work twice.
    Transport errors keep their ``requests`` exception type (callers guard on
    ``requests.Timeout`` for judge-billing endpoints)."""
    schedule = [0.0] + _RETRY_BACKOFF if retry else [0.0]
    last_exc: Optional[Exception] = None
    for attempt, wait in enumerate(schedule):
        if wait:
            time.sleep(wait)
        try:
            resp = requests.request(method, url, **kwargs)
        except requests.RequestException as e:
            last_exc = e
            logger.debug("Request error (attempt %d): %s", attempt + 1, e)
            continue
        if retry and resp.status_code in _RETRYABLE_STATUS and attempt < len(schedule) - 1:
            logger.debug("Retryable status %d (attempt %d)", resp.status_code, attempt + 1)
            continue
        return resp
    assert last_exc is not None  # every non-raising path returned above
    raise last_exc
