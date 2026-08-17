"""
Moveworks integration for AgentX production tracing.

Moveworks agents run inside the Moveworks cloud (Agent Studio + Reasoning Engine), so unlike the
LangChain/CrewAI integrations there is no in-process callback to hook: Moveworks Python Script
Actions have outbound network access disabled at the infrastructure level. The sanctioned export
surface is the read-only Moveworks **Data API** (OData, ``https://api.moveworks.ai/export/v1``),
which exposes ``conversations``, ``interactions``, ``plugin-calls``, ``plugin-resources`` and
``users`` tables. This module pulls those records and replays them into AgentX as traces:

- conversation      -> AgentX session (``session_id = mw_<conversation_id>``)
- interaction       -> one AgentX trace (input/output, timestamps, latency where derivable)
- plugin call       -> ``tool_calls`` entries on its interaction's trace

Usage::

    from agentx.integrations.moveworks import MoveworksDataAPIClient, MoveworksImporter

    client = MoveworksDataAPIClient(api_key="mw-data-api-key")
    importer = MoveworksImporter(
        client,
        agentx_api_key="agtx_local_...",
        agentx_base_url="http://localhost:4700/api/v1",   # self-host engine
    )
    report = importer.sync(since=datetime.now(timezone.utc) - timedelta(days=1))
    print(report)

Or from a shell / cron job (keeps an incremental cursor between runs)::

    export MOVEWORKS_API_KEY=... AGENTX_API_KEY=... AGENTX_API_BASE_URL=http://localhost:4700/api/v1
    agentx-moveworks sync --since 7d

Notes and limits:

- The Data API retains 30 days of history and follows a ~24-hour freshness SLA - this is
  near-real-time observability, not live tracing.
- Re-syncing a window is safe: every trace carries a deterministic ``span_id``
  (``mw:<interaction_id>``) and the engine skips spans it has already ingested.
- Moveworks does not export token counts, so these traces do not contribute to the LLM cost
  chart; plugin calls carry served/used flags but no failure semantics, so no tool failures are
  fabricated.
- Data API credentials are minted by a Moveworks superadmin (Setup -> Credentials). Field names
  in Data API records vary between deployments/versions; the importer reads each field from a
  list of known candidates and tolerates absences.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests

from agentx.tracing.ingest_client import IngestClient
from agentx.version import VERSION

_DEFAULT_BASE_URL = "https://api.moveworks.ai/export/v1"
_DEFAULT_TIMESTAMP_FIELD = "created_time"
_DEFAULT_CURSOR_FILE = Path.home() / ".agentx" / "moveworks_cursor.json"


# ----------------------------------------------------------------------------------------------
# Defensive record readers - Data API field names vary between deployments/versions.
# ----------------------------------------------------------------------------------------------

def _first(record: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_time(value: Any) -> Optional[datetime]:
    """ISO-8601 strings and epoch seconds/millis, to aware UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Millisecond epochs are > year 33658 when read as seconds; use magnitude to decide.
        seconds = value / 1000.0 if value > 1e11 else float(value)
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _record_time(record: Dict[str, Any]) -> Optional[datetime]:
    return _parse_time(
        _first(record, "created_time", "created_at", "timestamp", "event_time", "time", "start_time")
    )


class MoveworksDataAPIClient:
    """
    Minimal client for the Moveworks Data API (OData, GET-only).

    ``api_key`` is sent as ``Authorization: Bearer <key>`` by default; override ``auth_header`` /
    ``auth_scheme`` if your deployment differs. Pagination follows ``odata.nextlink``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        auth_header: str = "Authorization",
        auth_scheme: str = "Bearer",
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_key:
            raise ValueError("A Moveworks Data API key is required (minted by a Moveworks superadmin)")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session or requests.Session()
        value = f"{auth_scheme} {api_key}".strip() if auth_scheme else api_key
        self._session.headers.update({auth_header: value, "Accept": "application/json"})

    def records(
        self,
        table: str,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        timestamp_field: str = _DEFAULT_TIMESTAMP_FIELD,
        extra_params: Optional[Dict[str, str]] = None,
        max_pages: int = 1000,
    ) -> Iterator[Dict[str, Any]]:
        """Yield records from ``/records/<table>``, following OData pagination."""
        filters: List[str] = []
        if since is not None:
            filters.append(f"{timestamp_field} ge {since.astimezone(timezone.utc).isoformat()}")
        if until is not None:
            filters.append(f"{timestamp_field} lt {until.astimezone(timezone.utc).isoformat()}")
        params: Dict[str, str] = dict(extra_params or {})
        if filters:
            params["$filter"] = " and ".join(filters)

        url: Optional[str] = f"{self._base_url}/records/{table}"
        for _ in range(max_pages):
            if not url:
                return
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
            body = response.json()
            for record in body.get("value", []):
                if isinstance(record, dict):
                    yield record
            # nextlink is absolute and already carries its own query string.
            url = body.get("odata.nextlink") or body.get("@odata.nextLink")
            params = {}


class MoveworksSyncReport:
    def __init__(self) -> None:
        self.conversations = 0
        self.interactions = 0
        self.ingested = 0
        self.failed = 0
        self.plugin_calls_attached = 0
        self.skipped_no_time = 0
        # Distinct AgentX session ids touched by this run - what judge_sessions() operates on.
        self.session_ids: List[str] = []
        self.sessions_judged = 0
        self.sessions_judge_skipped = 0
        self.sessions_judge_failed = 0

    def __repr__(self) -> str:  # also what the CLI prints
        base = (
            f"MoveworksSyncReport(conversations={self.conversations}, interactions={self.interactions}, "
            f"ingested={self.ingested}, failed={self.failed}, "
            f"plugin_calls_attached={self.plugin_calls_attached}, skipped_no_time={self.skipped_no_time}"
        )
        if self.sessions_judged or self.sessions_judge_skipped or self.sessions_judge_failed:
            base += (
                f", sessions_judged={self.sessions_judged}, "
                f"sessions_judge_skipped={self.sessions_judge_skipped}, "
                f"sessions_judge_failed={self.sessions_judge_failed}"
            )
        return base + ")"


class MoveworksImporter:
    """
    Pulls Moveworks Data API records and replays them into AgentX as traces.

    ``agent_name`` overrides the agent every trace is attributed to; by default traces group per
    Moveworks domain as ``moveworks-<domain>`` (falling back to ``moveworks-assistant``).
    """

    def __init__(
        self,
        client: MoveworksDataAPIClient,
        *,
        agentx_api_key: str,
        agentx_base_url: Optional[str] = None,
        agent_name: Optional[str] = None,
        timestamp_field: str = _DEFAULT_TIMESTAMP_FIELD,
    ) -> None:
        self._client = client
        self._agent_name = agent_name
        self._timestamp_field = timestamp_field
        self._ingest = IngestClient(agentx_api_key, sdk_version=VERSION, base_url=agentx_base_url)

    # -- record -> wire mapping ----------------------------------------------------------------

    def _conversation_index(self, since: datetime, until: Optional[datetime]) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        try:
            for conversation in self._client.records(
                "conversations", since=since, until=until, timestamp_field=self._timestamp_field
            ):
                conversation_id = _first(conversation, "conversation_id", "id")
                if conversation_id:
                    index[str(conversation_id)] = conversation
        except requests.RequestException:
            # Conversations only enrich metadata (domain/route); interactions alone still import.
            pass
        return index

    def _plugin_calls_index(self, since: datetime, until: Optional[datetime]) -> Dict[str, List[Dict[str, Any]]]:
        index: Dict[str, List[Dict[str, Any]]] = {}
        try:
            for call in self._client.records(
                "plugin-calls", since=since, until=until, timestamp_field=self._timestamp_field
            ):
                key = _first(call, "interaction_id", "conversation_id")
                if key:
                    index.setdefault(str(key), []).append(call)
        except requests.RequestException:
            pass
        return index

    @staticmethod
    def _tool_call_wire(call: Dict[str, Any]) -> Dict[str, Any]:
        wire: Dict[str, Any] = {"name": str(_first(call, "plugin_name", "name") or "moveworks-plugin")}
        served = _first(call, "served", "is_served")
        used = _first(call, "used", "is_used")
        # served/used are surfacing metrics, not success/failure - never fabricate a failure.
        if served is not None:
            wire["served"] = bool(served)
        if used is not None:
            wire["used"] = bool(used)
        return wire

    def _interaction_wire(
        self,
        interaction: Dict[str, Any],
        conversations: Dict[str, Dict[str, Any]],
        plugin_calls: Dict[str, List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        started = _record_time(interaction)
        if started is None:
            return None

        interaction_id = str(_first(interaction, "interaction_id", "id") or "")
        conversation_id = _first(interaction, "conversation_id")
        conversation = conversations.get(str(conversation_id)) if conversation_id else None

        domain = _first(interaction, "domain") or (conversation and _first(conversation, "domain"))
        route = _first(interaction, "route") or (conversation and _first(conversation, "route"))
        interaction_type = _first(interaction, "interaction_type", "type")

        user_text = _first(interaction, "utterance", "user_input", "text", "message", "query", "content")
        response_text = _first(interaction, "response", "bot_response", "reply", "answer")

        calls: List[Dict[str, Any]] = []
        if interaction_id and interaction_id in plugin_calls:
            calls = plugin_calls[interaction_id]
        elif conversation_id and str(conversation_id) in plugin_calls:
            calls = plugin_calls[str(conversation_id)]

        metadata: Dict[str, Any] = {"source": "moveworks"}
        for key, value in (
            ("conversationId", conversation_id),
            ("interactionId", interaction_id or None),
            ("interactionType", interaction_type),
            ("domain", domain),
            ("route", route),
            ("userId", _first(interaction, "user_id")),
        ):
            if value:
                metadata[key] = value

        latency_ms: Optional[int] = None
        ended = _parse_time(_first(interaction, "resolved_time", "completed_time", "end_time", "updated_time"))
        if ended is not None and ended >= started:
            latency_ms = int((ended - started).total_seconds() * 1000)

        name = self._agent_name or (f"moveworks-{domain}" if domain else "moveworks-assistant")

        wire: Dict[str, Any] = {
            "name": name,
            "framework": "moveworks",
            "metadata": metadata,
            "started_at_unix_nano": str(int(started.timestamp() * 1_000_000_000)),
        }
        if interaction_id:
            wire["span_id"] = f"mw:{interaction_id}"
        if conversation_id:
            wire["session_id"] = f"mw_{conversation_id}"
        if user_text is not None:
            wire["input"] = user_text
        else:
            wire["input"] = json.dumps({"interaction_type": interaction_type or "unknown"})
        if response_text is not None:
            wire["output"] = response_text
        if latency_ms is not None:
            wire["latency_ms"] = latency_ms
        if calls:
            wire["tool_calls"] = [self._tool_call_wire(call) for call in calls]
        return wire

    # -- public --------------------------------------------------------------------------------

    def sync(
        self,
        since: datetime,
        until: Optional[datetime] = None,
        *,
        monitor: bool = False,
        judge_sessions: bool = False,
        dry_run: bool = False,
        on_payload: Optional[Any] = None,
    ) -> MoveworksSyncReport:
        """
        Import every interaction in ``[since, until)``. Safe to re-run over the same window - the
        engine deduplicates on the deterministic ``span_id`` (and skips re-judging deduped spans).

        ``monitor=True`` sets ``monitor: true`` on every trace - the engine's explicit opt-in that
        runs pattern/built-in checks (PII, empty response, tool failure, active patterns) on each
        imported trace. ``judge_sessions=True`` additionally asks the engine to judge every
        imported session with each enabled session-scoped evaluator after the sync; the request
        carries ``ifStale=true`` so a session already scored (e.g. by the engine's own 24h sweep)
        is never judged twice. Returns a summary report.
        """
        report = MoveworksSyncReport()
        conversations = self._conversation_index(since, until)
        report.conversations = len(conversations)
        plugin_calls = self._plugin_calls_index(since, until)
        session_ids: List[str] = []

        for interaction in self._client.records(
            "interactions", since=since, until=until, timestamp_field=self._timestamp_field
        ):
            report.interactions += 1
            wire = self._interaction_wire(interaction, conversations, plugin_calls)
            if wire is None:
                report.skipped_no_time += 1
                continue
            if monitor:
                wire["monitor"] = True
            report.plugin_calls_attached += len(wire.get("tool_calls", []))
            session_id = wire.get("session_id")
            if session_id and session_id not in session_ids:
                session_ids.append(session_id)
            if on_payload is not None:
                on_payload(wire)
            if dry_run:
                continue
            if self._ingest.send_trace_sync(wire) is not None:
                report.ingested += 1
            else:
                report.failed += 1

        report.session_ids = session_ids
        if judge_sessions and not dry_run:
            self.judge_sessions(session_ids, report=report)
        return report

    def judge_sessions(
        self, session_ids: List[str], *, report: Optional[MoveworksSyncReport] = None
    ) -> MoveworksSyncReport:
        """
        Judge each session with every enabled session-scoped evaluator via the engine's on-demand
        judge route, with ``ifStale=true`` so an up-to-date verdict (from the sweep or a previous
        run) is left alone instead of judged again.
        """
        report = report or MoveworksSyncReport()
        # Same base/key the ingest client already resolved - the judge routes live under
        # /agent-monitoring on the same engine.
        base = self._ingest._base_url
        http = self._ingest._session
        try:
            response = http.get(f"{base}/agent-monitoring/online-evaluators", timeout=30)
            response.raise_for_status()
            evaluators = [
                e for e in response.json().get("evaluators", [])
                if e.get("enabled") and e.get("scope") == "session"
            ]
        except requests.RequestException:
            report.sessions_judge_failed += len(session_ids)
            return report
        for session_id in session_ids:
            for evaluator in evaluators:
                try:
                    result = http.post(
                        f"{base}/agent-monitoring/sessions/{session_id}/judge/{evaluator['_id']}",
                        params={"ifStale": "true"},
                        timeout=120,
                    )
                    if result.status_code == 200 and result.json().get("skipped"):
                        report.sessions_judge_skipped += 1
                    elif result.ok:
                        report.sessions_judged += 1
                    else:
                        report.sessions_judge_failed += 1
                except requests.RequestException:
                    report.sessions_judge_failed += 1
        return report


# ----------------------------------------------------------------------------------------------
# CLI: agentx-moveworks sync
# ----------------------------------------------------------------------------------------------

def _parse_since(value: str) -> datetime:
    """Either an ISO timestamp or a relative shorthand like ``7d`` / ``24h`` / ``30m``."""
    text = value.strip().lower()
    if text and text[-1] in ("d", "h", "m") and text[:-1].isdigit():
        amount = int(text[:-1])
        delta = {"d": timedelta(days=amount), "h": timedelta(hours=amount), "m": timedelta(minutes=amount)}[text[-1]]
        return datetime.now(timezone.utc) - delta
    parsed = _parse_time(value)
    if parsed is None:
        raise SystemExit(f"agentx-moveworks: could not parse --since {value!r} (use ISO-8601 or e.g. 7d)")
    return parsed


def _read_cursor(path: Path) -> Optional[datetime]:
    try:
        return _parse_time(json.loads(path.read_text()).get("last_synced"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _write_cursor(path: Path, value: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_synced": value.isoformat()}))


def cli_main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agentx-moveworks",
        description="Sync Moveworks Data API activity into AgentX as traces/sessions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync", help="Import a time window of Moveworks activity")
    sync.add_argument("--since", help="ISO timestamp or relative (7d, 24h). Default: cursor file, else 24h")
    sync.add_argument("--until", help="ISO timestamp (default: now)")
    sync.add_argument("--cursor-file", default=str(_DEFAULT_CURSOR_FILE), help="Incremental cursor path")
    sync.add_argument("--no-cursor", action="store_true", help="Ignore and don't update the cursor file")
    sync.add_argument("--agent-name", help="Attribute every trace to this agent (default: per Moveworks domain)")
    sync.add_argument("--timestamp-field", default=_DEFAULT_TIMESTAMP_FIELD, help="Data API record timestamp field")
    sync.add_argument("--moveworks-base-url", default=os.getenv("MOVEWORKS_BASE_URL", _DEFAULT_BASE_URL))
    sync.add_argument(
        "--monitor",
        action="store_true",
        help="Run pattern/built-in checks (PII, empty response, tool failure, active patterns) on each imported trace",
    )
    sync.add_argument(
        "--judge-sessions",
        action="store_true",
        help="After the sync, judge each imported session with every enabled session-scoped evaluator "
        "(skips sessions that already have an up-to-date verdict, so the engine's own sweep is never duplicated)",
    )
    sync.add_argument("--dry-run", action="store_true", help="Map and print payloads without ingesting")

    args = parser.parse_args(argv)

    moveworks_key = os.getenv("MOVEWORKS_API_KEY", "")
    agentx_key = os.getenv("AGENTX_API_KEY", "")
    if not moveworks_key:
        raise SystemExit("agentx-moveworks: set MOVEWORKS_API_KEY (Data API credential from Moveworks Setup)")
    if not agentx_key and not args.dry_run:
        raise SystemExit("agentx-moveworks: set AGENTX_API_KEY (your AgentX project API key)")

    cursor_path = Path(args.cursor_file)
    since = (
        _parse_since(args.since)
        if args.since
        else ((None if args.no_cursor else _read_cursor(cursor_path)) or (datetime.now(timezone.utc) - timedelta(hours=24)))
    )
    until = _parse_time(args.until) if args.until else datetime.now(timezone.utc)

    client = MoveworksDataAPIClient(moveworks_key, base_url=args.moveworks_base_url)
    importer = MoveworksImporter(
        client,
        agentx_api_key=agentx_key or "dry-run",
        agentx_base_url=os.getenv("AGENTX_API_BASE_URL"),
        agent_name=args.agent_name,
        timestamp_field=args.timestamp_field,
    )

    print(f"agentx-moveworks: syncing {since.isoformat()} -> {until.isoformat()}")
    on_payload = (lambda wire: print(json.dumps(wire))) if args.dry_run else None
    report = importer.sync(
        since,
        until,
        monitor=args.monitor,
        judge_sessions=args.judge_sessions,
        dry_run=args.dry_run,
        on_payload=on_payload,
    )
    print(report)

    if not args.dry_run and not args.no_cursor and report.failed == 0:
        _write_cursor(cursor_path, until)
        print(f"agentx-moveworks: cursor -> {cursor_path}")
    elif report.failed:
        print("agentx-moveworks: failures occurred - cursor NOT advanced, re-run to retry", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
