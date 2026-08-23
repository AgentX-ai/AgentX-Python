from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import requests

from agentx.util import api_base, get_headers

logger = logging.getLogger(__name__)


class AgentXExportError(Exception):
    pass


class ExportClient:
    """Surfaced as ``client.export``: bulk NDJSON egress for backup and migration (self-host).

    The engine's ``GET /export`` manifest lists every exportable entity (traces, signals,
    events, runs, feedback, outcomes, scorer config, ...) with live row counts;
    ``GET /export/<entity>`` streams the rows as NDJSON. Everything is scoped to the API key's
    project, so an export can never cross a tenant boundary.

    Typical uses::

        client.export.dump("./backup")                     # full backup, one .ndjson per entity
        client.export.dump("./nightly", since=yesterday)   # incremental
        for row in client.export.iter("traces"):           # stream without touching disk
            ...

    Restore paths are documented in the self-host backup runbook: replay traces through
    ``client.tracer`` / ``POST /ingest/traces``, or restore at the database level
    (``pg_dump`` / SQLite file copy). There is deliberately no blind row-import endpoint.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._api_key = api_key
        # Captured once at construction (deep-dive round 3, bug #1).
        self._base_url = (base_url or api_base()).rstrip("/")

    def manifest(self) -> List[Dict[str, Any]]:
        """The exportable entities with live row counts: ``[{entity, rows, path}, ...]``."""
        resp = requests.get(
            f"{self._base_url}/export", headers=get_headers(self._api_key), timeout=30
        )
        if resp.status_code >= 400:
            raise AgentXExportError(f"Export manifest failed ({resp.status_code}): {resp.text[:200]}")
        return resp.json().get("entities", [])

    def iter(self, entity: str, since: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Stream one entity's rows as dicts without buffering the whole table in memory.
        ``since`` is an ISO-8601 date for incremental pulls (filters on the entity's own
        timestamp column, e.g. ``createdAt`` for traces, ``lastSeenAt`` for signals)."""
        params = {"since": since} if since else None
        resp = requests.get(
            f"{self._base_url}/export/{entity}",
            headers=get_headers(self._api_key),
            params=params,
            stream=True,
            timeout=120,
        )
        if resp.status_code >= 400:
            raise AgentXExportError(f"Export of {entity!r} failed ({resp.status_code}): {resp.text[:200]}")
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.strip():
                yield json.loads(line)

    def dump(
        self,
        directory: str,
        entities: Optional[List[str]] = None,
        since: Optional[str] = None,
    ) -> Dict[str, int]:
        """Write ``<entity>.ndjson`` files (plus a ``manifest.json``) into ``directory`` and
        return ``{entity: rows_written}``. Defaults to every entity the engine advertises;
        pass ``entities`` to restrict, ``since`` for an incremental snapshot."""
        os.makedirs(directory, exist_ok=True)
        manifest = self.manifest()
        wanted = [e["entity"] for e in manifest] if entities is None else entities
        written: Dict[str, int] = {}
        for entity in wanted:
            path = os.path.join(directory, f"{entity}.ndjson")
            count = 0
            with open(path, "w", encoding="utf-8") as fh:
                for row in self.iter(entity, since=since):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
            written[entity] = count
            logger.debug("Exported %d %s rows to %s", count, entity, path)
        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {"entities": manifest, "written": written, **({"since": since} if since else {})},
                fh,
                indent=2,
            )
        return written
