"""
Databricks / MLflow integration for AgentX.

Two complementary paths for agents built on Databricks (Agent Bricks or the Mosaic AI Agent
Framework - both are auto-instrumented by MLflow 3 Tracing):

1. **Push (live)** - ``enable_mlflow_export()``: point MLflow Tracing's native OTLP exporter at
   the AgentX engine's OTel endpoint. Works anywhere MLflow traces run (notebooks, jobs, Model
   Serving endpoints via environment variables). Dual export keeps Databricks' own MLflow UI and
   inference tables working alongside AgentX::

       from agentx.integrations.databricks import enable_mlflow_export

       enable_mlflow_export(
           api_key=os.environ["AGENTX_API_KEY"],
           base_url="http://localhost:4700/api/v1",   # your AgentX engine
           service_name="my-databricks-agent",
       )
       # ... then trace as usual (@mlflow.trace, autolog, Agent Framework, ...)

   On a Model Serving endpoint, set the equivalent environment variables instead (this helper
   prints them with ``dry_run=True``).

2. **Pull (batch)** - ``agentx-databricks sync``: import finished MLflow traces from a Databricks
   (or any MLflow 3) tracking server into AgentX - full span trees, tool calls, sessions -
   deduplicated on deterministic span ids so re-running a window never duplicates, with
   ``--monitor`` / ``--judge-sessions`` mirroring ``agentx-moveworks``::

       export AGENTX_API_KEY=...            # AgentX project key
       export DATABRICKS_HOST=... DATABRICKS_TOKEN=...   # or MLFLOW_TRACKING_URI
       agentx-databricks sync --experiment-id 123456 --since 24h

Requires (pull path): ``pip install "agentx-python[databricks]"`` (mlflow>=3).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agentx.tracing.ingest_client import IngestClient
from agentx.version import VERSION
from agentx.integrations.moveworks import (
    _parse_since,
    _parse_time,
    _read_cursor,
    _write_cursor,
    judge_sessions_via_engine,
)

_DEFAULT_CURSOR_FILE = Path.home() / ".agentx" / "databricks_sync_cursor.json"

# MLflow's session grouping metadata key (mlflow.update_current_trace(metadata={...})) - traces
# sharing it become one AgentX session, judged as a conversation by session-scoped evaluators.
_SESSION_METADATA_KEYS = ("mlflow.trace.session", "mlflow.trace.session_id", "session_id")


# ----------------------------------------------------------------------------------------------
# Path 1: push - MLflow OTLP export pointed at the AgentX engine
# ----------------------------------------------------------------------------------------------

def enable_mlflow_export(
    *,
    api_key: str,
    base_url: str,
    service_name: str = "databricks-agent",
    dual: bool = True,
    genai_semconv: bool = False,
    dry_run: bool = False,
) -> Dict[str, str]:
    """
    Configure MLflow Tracing's built-in OTLP exporter to send every trace to the AgentX engine.

    Must run BEFORE the first trace starts (MLflow reads these once, at tracer setup). ``dual``
    keeps MLflow's own tracking export too (Databricks MLflow UI / inference tables keep
    working); ``genai_semconv`` switches the wire format to OTel GenAI semantic conventions -
    AgentX ingests both, and the default (MLflow-native attributes) is the higher-fidelity
    mapping for plain ``@mlflow.trace`` functions. Returns the environment variables set - with
    ``dry_run=True`` nothing is set, so the dict can be copied onto a Databricks Model Serving
    endpoint's environment variables instead.
    """
    env = {
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": f"{base_url.rstrip('/')}/otel/v1/traces",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS": f"x-api-key={api_key}",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_SERVICE_NAME": service_name,
    }
    if dual:
        env["MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT"] = "true"
    if genai_semconv:
        env["MLFLOW_ENABLE_OTEL_GENAI_SEMCONV"] = "true"
    if not dry_run:
        os.environ.update(env)
    return env


# ----------------------------------------------------------------------------------------------
# Path 2: pull - MLflow trace search -> AgentX span trees
# ----------------------------------------------------------------------------------------------

class DatabricksSyncReport:
    def __init__(self) -> None:
        self.traces = 0
        self.spans = 0
        self.tool_calls = 0
        self.ingested = 0
        self.failed = 0
        self.skipped_in_progress = 0
        self.session_ids: List[str] = []
        self.sessions_judged = 0
        self.sessions_judge_skipped = 0
        self.sessions_judge_failed = 0

    def __repr__(self) -> str:  # also what the CLI prints
        parts = [
            f"traces={self.traces}",
            f"spans={self.spans}",
            f"tool_calls={self.tool_calls}",
            f"ingested={self.ingested}",
            f"failed={self.failed}",
            f"sessions={len(self.session_ids)}",
        ]
        if self.skipped_in_progress:
            parts.append(f"skipped_in_progress={self.skipped_in_progress}")
        if self.sessions_judged or self.sessions_judge_skipped or self.sessions_judge_failed:
            parts.append(
                f"judged={self.sessions_judged} judge_skipped={self.sessions_judge_skipped} "
                f"judge_failed={self.sessions_judge_failed}"
            )
        return f"DatabricksSyncReport({' '.join(parts)})"


def _span_time_ns(span: Any, attr: str) -> Optional[int]:
    value = getattr(span, attr, None)
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


def _span_error(span: Any) -> Optional[str]:
    status = getattr(span, "status", None)
    code = str(getattr(status, "status_code", "") or "")
    if "ERROR" in code.upper():
        return str(getattr(status, "description", None) or "error")
    return None


def _serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


class DatabricksTraceImporter:
    """
    Replays finished MLflow traces (Databricks-hosted or any MLflow 3 tracking server) into
    AgentX as full span trees. ``span_id`` is deterministic (``dbx:<trace_id>[:<span_id>]``), so
    re-syncing a window is idempotent - the engine dedupes on replay and skips re-judging.
    """

    def __init__(
        self,
        mlflow_client: Any,
        *,
        agentx_api_key: str,
        agentx_base_url: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> None:
        self._client = mlflow_client
        self._agent_name = agent_name
        self._ingest = IngestClient(agentx_api_key, sdk_version=VERSION, base_url=agentx_base_url)

    # -- trace -> wires ------------------------------------------------------------------------

    def _trace_wires(self, trace: Any) -> "tuple[List[Dict[str, Any]], Optional[str], int]":
        """One MLflow Trace -> [root wire, *child wires], its session id, and its tool-call count."""
        info = trace.info
        trace_id = str(getattr(info, "trace_id", None) or getattr(info, "request_id", ""))
        spans = list(getattr(trace.data, "spans", None) or [])
        if not trace_id or not spans:
            return [], None, 0

        metadata_bag: Dict[str, Any] = {}
        for source in (getattr(info, "trace_metadata", None), getattr(info, "tags", None)):
            if isinstance(source, dict):
                metadata_bag.update(source)
        session_raw = next((metadata_bag[k] for k in _SESSION_METADATA_KEYS if metadata_bag.get(k)), None)
        session_id = f"dbx_{session_raw}" if session_raw else None

        root = next((s for s in spans if not getattr(s, "parent_id", None)), spans[0])
        root_span_id = str(getattr(root, "span_id", "") or "root")
        wire_span_id = {root_span_id: f"dbx:{trace_id}"}
        for span in spans:
            sid = str(getattr(span, "span_id", "") or "")
            if sid and sid not in wire_span_id:
                wire_span_id[sid] = f"dbx:{trace_id}:{sid}"

        tool_calls: List[Dict[str, Any]] = []
        wires: List[Dict[str, Any]] = []
        for span in spans:
            sid = str(getattr(span, "span_id", "") or "")
            is_root = span is root
            start_ns = _span_time_ns(span, "start_time_ns")
            end_ns = _span_time_ns(span, "end_time_ns")
            error = _span_error(span)
            span_type = str(getattr(span, "span_type", "") or "").upper()
            wire: Dict[str, Any] = {
                "name": (self._agent_name if is_root and self._agent_name else str(getattr(span, "name", "span"))),
                "framework": "databricks",
                "span_id": wire_span_id.get(sid, f"dbx:{trace_id}:{sid or 'span'}"),
            }
            if not is_root:
                parent_sid = str(getattr(span, "parent_id", "") or "")
                wire["parent_span_id"] = wire_span_id.get(parent_sid, f"dbx:{trace_id}")
            if session_id:
                wire["session_id"] = session_id
            if start_ns:
                wire["started_at_unix_nano"] = str(start_ns)
            if start_ns and end_ns and end_ns > start_ns:
                wire["latency_ms"] = int((end_ns - start_ns) / 1_000_000)
            inputs = _serialize(getattr(span, "inputs", None))
            outputs = _serialize(getattr(span, "outputs", None))
            if inputs is not None:
                wire["input"] = inputs
            if outputs is not None:
                wire["output"] = outputs
            if error:
                wire["error"] = error
            if is_root:
                wire["metadata"] = {
                    "source": "databricks",
                    "mlflowTraceId": trace_id,
                    **({"experimentId": str(getattr(info, "experiment_id", ""))} if getattr(info, "experiment_id", None) else {}),
                }
            if span_type == "TOOL":
                tool_calls.append(
                    {
                        "name": str(getattr(span, "name", "tool")),
                        "input": inputs,
                        "output": outputs,
                        "latency_ms": wire.get("latency_ms"),
                        "success": not error,
                    }
                )
            wires.append(wire)

        # Root carries the flat tool_calls mirror - what the engine's Tool-failure check and
        # trajectory matching read (same posture as tracer._merge_child_run).
        if tool_calls:
            wires[spans.index(root)]["tool_calls"] = tool_calls
        # Root first so the engine resolves the agent before children arrive.
        wires.sort(key=lambda w: 0 if "parent_span_id" not in w else 1)
        return wires, session_id, len(tool_calls)

    def _search(self, experiment_ids: List[str], since: datetime, until: datetime) -> Iterable[Any]:
        """Newest-first paginated search, stopping once a page is entirely older than ``since``."""
        page_token: Optional[str] = None
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)
        while True:
            page = self._client.search_traces(
                experiment_ids=experiment_ids,
                max_results=100,
                page_token=page_token,
                order_by=["timestamp_ms DESC"],
            )
            oldest_seen = None
            for trace in page:
                ts = getattr(trace.info, "request_time", None) or getattr(trace.info, "timestamp_ms", None)
                ts_ms = int(ts.timestamp() * 1000) if isinstance(ts, datetime) else (int(ts) if ts else None)
                oldest_seen = ts_ms if ts_ms is not None else oldest_seen
                if ts_ms is not None and (ts_ms < since_ms or ts_ms >= until_ms):
                    if ts_ms < since_ms:
                        continue
                    continue
                yield trace
            page_token = getattr(page, "token", None)
            if not page_token or (oldest_seen is not None and oldest_seen < since_ms):
                return

    # -- public --------------------------------------------------------------------------------

    def sync(
        self,
        experiment_ids: List[str],
        since: datetime,
        until: Optional[datetime] = None,
        *,
        monitor: bool = False,
        judge_sessions: bool = False,
        dry_run: bool = False,
        on_payload: Optional[Any] = None,
    ) -> DatabricksSyncReport:
        """
        Import every finished MLflow trace in ``[since, until)`` from the given experiments.
        Safe to re-run over the same window (span_id dedupe). ``monitor=True`` opts every
        imported root trace into the engine's ingest-time checks; ``judge_sessions=True`` judges
        each imported session afterwards (``ifStale`` - never duplicates the engine's own sweep).
        """
        report = DatabricksSyncReport()
        until = until or datetime.now(timezone.utc)
        for trace in self._search(experiment_ids, since, until):
            state = str(getattr(trace.info, "state", "") or "")
            if state and "IN_PROGRESS" in state.upper():
                report.skipped_in_progress += 1
                continue
            wires, session_id, tool_count = self._trace_wires(trace)
            if not wires:
                continue
            report.traces += 1
            report.spans += len(wires)
            report.tool_calls += tool_count
            if session_id and session_id not in report.session_ids:
                report.session_ids.append(session_id)
            for wire in wires:
                if monitor and "parent_span_id" not in wire:
                    wire["monitor"] = True
                if on_payload is not None:
                    on_payload(wire)
                if dry_run:
                    continue
                if self._ingest.send_trace_sync(wire) is not None:
                    report.ingested += 1
                else:
                    report.failed += 1
        if judge_sessions and not dry_run:
            judged, skipped, failed = judge_sessions_via_engine(self._ingest, report.session_ids)
            report.sessions_judged += judged
            report.sessions_judge_skipped += skipped
            report.sessions_judge_failed += failed
        return report


# ----------------------------------------------------------------------------------------------
# CLI: agentx-databricks sync
# ----------------------------------------------------------------------------------------------

def cli_main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agentx-databricks",
        description="Sync MLflow traces (Databricks agents) into AgentX as span trees/sessions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync", help="Import a time window of MLflow traces")
    sync.add_argument("--experiment-id", action="append", required=True, help="MLflow experiment id (repeatable)")
    sync.add_argument("--since", help="ISO timestamp or relative (7d, 24h). Default: cursor file, else 24h")
    sync.add_argument("--until", help="ISO timestamp (default: now)")
    sync.add_argument("--cursor-file", default=str(_DEFAULT_CURSOR_FILE), help="Incremental cursor path")
    sync.add_argument("--no-cursor", action="store_true", help="Ignore and don't update the cursor file")
    sync.add_argument("--agent-name", help="Attribute every trace to this agent (default: the root span's name)")
    sync.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "databricks"),
        help='MLflow tracking URI (default: MLFLOW_TRACKING_URI, else "databricks" - uses DATABRICKS_HOST/TOKEN)',
    )
    sync.add_argument("--monitor", action="store_true", help="Run ingest-time checks on each imported trace")
    sync.add_argument(
        "--judge-sessions",
        action="store_true",
        help="After the sync, judge each imported session with every enabled session-scoped evaluator "
        "(skips sessions that already have an up-to-date verdict)",
    )
    sync.add_argument("--dry-run", action="store_true", help="Map and print payloads without ingesting")

    args = parser.parse_args(argv)

    agentx_key = os.getenv("AGENTX_API_KEY", "")
    if not agentx_key and not args.dry_run:
        raise SystemExit("agentx-databricks: set AGENTX_API_KEY (your AgentX project API key)")

    try:
        from mlflow.client import MlflowClient
    except ImportError as exc:  # pragma: no cover
        raise SystemExit('agentx-databricks: pip install "agentx-python[databricks]" (needs mlflow>=3)') from exc

    cursor_path = Path(args.cursor_file)
    since = (
        _parse_since(args.since)
        if args.since
        else ((None if args.no_cursor else _read_cursor(cursor_path)) or (datetime.now(timezone.utc) - timedelta(hours=24)))
    )
    until = _parse_time(args.until) if args.until else datetime.now(timezone.utc)

    importer = DatabricksTraceImporter(
        MlflowClient(tracking_uri=args.tracking_uri),
        agentx_api_key=agentx_key,
        agentx_base_url=os.getenv("AGENTX_API_BASE_URL") or None,
        agent_name=args.agent_name,
    )
    report = importer.sync(
        args.experiment_id,
        since,
        until,
        monitor=args.monitor,
        judge_sessions=args.judge_sessions,
        dry_run=args.dry_run,
        on_payload=(lambda wire: print(json.dumps(wire, default=str))) if args.dry_run else None,
    )
    print(report)
    if not args.no_cursor and not args.dry_run and until is not None:
        _write_cursor(cursor_path, until)
