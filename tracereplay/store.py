"""SQLite span store + OpenTelemetry SpanExporter that writes into it."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
  run_id TEXT, trace_id TEXT, span_id TEXT PRIMARY KEY, parent_id TEXT,
  name TEXT, step INTEGER, start_ns INTEGER, end_ns INTEGER, status TEXT, attributes TEXT
);
CREATE INDEX IF NOT EXISTS ix_run ON spans(run_id, step);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def insert(self, rows: list[tuple]):
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?)", rows
            )
            self.conn.commit()

    def runs(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT run_id, attributes, start_ns, end_ns FROM spans WHERE name='agent.run' ORDER BY start_ns"
        )
        out = []
        for run_id, attrs, s, e in cur:
            a = json.loads(attrs)
            n = self.conn.execute(
                "SELECT COUNT(*) FROM spans WHERE run_id=? AND step IS NOT NULL",
                (run_id,),
            ).fetchone()[0]
            out.append(
                {
                    "run_id": run_id,
                    "task": a.get("agent.task"),
                    "final": a.get("agent.final_answer"),
                    "parent_run": a.get("replay.parent_run"),
                    "from_step": a.get("replay.from_step"),
                    "steps": n,
                    "ms": round((e - s) / 1e6, 2),
                }
            )
        return out

    def steps(self, run_id: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT name, step, start_ns, end_ns, attributes FROM spans WHERE run_id=? AND step IS NOT NULL "
            "ORDER BY step",
            (run_id,),
        )
        out = []
        for name, step, s, e, attrs in cur:
            a = json.loads(attrs)
            out.append(
                {
                    "step": step,
                    "kind": name,
                    "ms": round((e - s) / 1e6, 3),
                    "attributes": a,
                    "request": json.loads(a.get("replay.request", "null")),
                    "response": json.loads(a.get("replay.response", "null")),
                }
            )
        return out


class SQLiteSpanExporter(SpanExporter):
    def __init__(self, store: Store):
        self.store = store

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        rows = []
        for sp in spans:
            a = dict(sp.attributes or {})
            rows.append(
                (
                    a.get("agent.run_id"),
                    format(sp.context.trace_id, "032x"),
                    format(sp.context.span_id, "016x"),
                    format(sp.parent.span_id, "016x") if sp.parent else None,
                    sp.name,
                    a.get("replay.step"),
                    sp.start_time,
                    sp.end_time,
                    sp.status.status_code.name,
                    json.dumps(a),
                )
            )
        self.store.insert(rows)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass
