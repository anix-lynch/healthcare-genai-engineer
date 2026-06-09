"""Durable action/task/state store (SQLite).

Why SQLite: cheap, inspectable, restart-durable, and enough to prove the loop.
Every metric in the Phase-1 gate is backed by a row here, so an independent
auditor can open the `.db` file and reconstruct the number without trusting any
in-memory eval boolean.

Idempotency is a DB-level guarantee, not an in-memory set: the world-state
table `committed_dispositions` has the idempotency key as PRIMARY KEY, so a
replay can never apply a second side effect even across process restarts.

Tables
------
cases                  one row per evidence record that reached intake
decisions              the Bed Ops disposition computed for a case
tasks                  durable work items (kind = action | escalation), with a
                       separate `acknowledged_at` set by the worker (receiver ACK
                       is a distinct second transition, not a creation-time flag)
actions                one row per action execution attempt-set: tool, status,
                       attempts_used, before/after state, receipt, tool_reported_ok
committed_dispositions THE durable world state the tool mutates; idempotency_key
                       is the PRIMARY KEY -> duplicate side effects impossible
outcomes               intended vs observed (re-read) state + verified / false_success
escalations            durable human work created when recovery is exhausted
events                 append-only stage log -> trace reconstruction
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    correlation_id   TEXT PRIMARY KEY,
    contract_version TEXT NOT NULL,
    source           TEXT NOT NULL,
    source_id        TEXT NOT NULL,
    evidence_json    TEXT NOT NULL,
    status           TEXT NOT NULL,          -- accepted | blocked
    received_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    correlation_id TEXT PRIMARY KEY,
    disposition    TEXT NOT NULL,
    reason         TEXT NOT NULL,
    decided_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id         TEXT PRIMARY KEY,
    correlation_id  TEXT NOT NULL,
    kind            TEXT NOT NULL,           -- action | escalation
    owner           TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    status          TEXT NOT NULL,           -- created | acknowledged
    created_at      TEXT NOT NULL,
    acknowledged_at TEXT
);

CREATE TABLE IF NOT EXISTS actions (
    action_id         TEXT PRIMARY KEY,
    correlation_id    TEXT NOT NULL,
    idempotency_key   TEXT NOT NULL,
    tool              TEXT NOT NULL,
    status            TEXT NOT NULL,         -- committed | idempotent_skip | exhausted
    attempts_used     INTEGER NOT NULL,
    max_attempts      INTEGER NOT NULL,
    tool_reported_ok  INTEGER NOT NULL,      -- what the tool CLAIMED (may lie)
    before_state_json TEXT NOT NULL,
    after_state_json  TEXT NOT NULL,
    receipt_json      TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

-- Durable world state. idempotency_key is PK => the tool can apply a given
-- disposition for a case at most once, even across restarts.
CREATE TABLE IF NOT EXISTS committed_dispositions (
    idempotency_key TEXT PRIMARY KEY,
    correlation_id  TEXT NOT NULL,
    disposition     TEXT NOT NULL,
    action_seq      INTEGER NOT NULL,
    applied_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outcomes (
    correlation_id        TEXT PRIMARY KEY,
    intended_json         TEXT NOT NULL,
    observed_json         TEXT NOT NULL,
    verified              INTEGER NOT NULL,
    false_success_detected INTEGER NOT NULL,
    verified_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS escalations (
    escalation_id  TEXT PRIMARY KEY,
    correlation_id TEXT NOT NULL,
    reason         TEXT NOT NULL,
    owner          TEXT NOT NULL,
    status         TEXT NOT NULL,            -- open
    attempts_used  INTEGER NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id TEXT NOT NULL,
    stage          TEXT NOT NULL,
    detail_json    TEXT NOT NULL,
    ts             TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActionStore:
    """Thin durable store. One connection; explicit commits keep state durable."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ── event log (trace reconstruction) ────────────────────────────────────
    def log_event(self, correlation_id: str, stage: str, detail: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO events (correlation_id, stage, detail_json, ts) VALUES (?,?,?,?)",
            (correlation_id, stage, json.dumps(detail), _now()),
        )
        self.conn.commit()

    def events_for(self, correlation_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT stage, detail_json, ts FROM events WHERE correlation_id=? ORDER BY event_id",
            (correlation_id,),
        ).fetchall()
        return [{"stage": r["stage"], "detail": json.loads(r["detail_json"]), "ts": r["ts"]} for r in rows]

    # ── cases / decisions ───────────────────────────────────────────────────
    def record_case(self, correlation_id: str, contract_version: str, source: str,
                    source_id: str, evidence_json: dict[str, Any], status: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cases "
            "(correlation_id, contract_version, source, source_id, evidence_json, status, received_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (correlation_id, contract_version, source, source_id, json.dumps(evidence_json), status, _now()),
        )
        self.conn.commit()

    def record_decision(self, correlation_id: str, disposition: str, reason: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO decisions (correlation_id, disposition, reason, decided_at) VALUES (?,?,?,?)",
            (correlation_id, disposition, reason, _now()),
        )
        self.conn.commit()

    def get_decision(self, correlation_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT disposition, reason FROM decisions WHERE correlation_id=?", (correlation_id,)
        ).fetchone()
        return dict(row) if row else None

    # ── tasks (durable work + receiver ACK) ─────────────────────────────────
    def create_task(self, task_id: str, correlation_id: str, kind: str, owner: str,
                    idempotency_key: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO tasks "
            "(task_id, correlation_id, kind, owner, idempotency_key, status, created_at) "
            "VALUES (?,?,?,?,?, 'created', ?)",
            (task_id, correlation_id, kind, owner, idempotency_key, _now()),
        )
        self.conn.commit()

    def acknowledge_task(self, task_id: str) -> bool:
        """Worker ACKs ownership — a SEPARATE second state transition.

        Only flips a task that is still `created`; returns whether it moved.
        """
        cur = self.conn.execute(
            "UPDATE tasks SET status='acknowledged', acknowledged_at=? "
            "WHERE task_id=? AND status='created'",
            (_now(), task_id),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    def action_tasks_for(self, correlation_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM tasks WHERE correlation_id=? AND kind='action' ORDER BY created_at",
            (correlation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── world state (idempotent) ────────────────────────────────────────────
    def get_committed(self, idempotency_key: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM committed_dispositions WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
        return dict(row) if row else None

    def try_commit_disposition(self, idempotency_key: str, correlation_id: str,
                               disposition: str) -> bool:
        """Attempt the durable state change. Returns True iff this call applied it.

        Relies on the PRIMARY KEY: `INSERT OR IGNORE` is a no-op when the key
        already exists, so a replay applies nothing. `action_seq` is a global
        monotonic counter so before/after state is provably different.
        """
        next_seq = (self.conn.execute(
            "SELECT COALESCE(MAX(action_seq), 0) + 1 FROM committed_dispositions"
        ).fetchone()[0])
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO committed_dispositions "
            "(idempotency_key, correlation_id, disposition, action_seq, applied_at) "
            "VALUES (?,?,?,?,?)",
            (idempotency_key, correlation_id, disposition, next_seq, _now()),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def count_committed(self, idempotency_key: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM committed_dispositions WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()[0]

    # ── action receipts ─────────────────────────────────────────────────────
    def record_action(self, action_id: str, correlation_id: str, idempotency_key: str,
                       tool: str, status: str, attempts_used: int, max_attempts: int,
                       tool_reported_ok: bool, before_state: dict[str, Any],
                       after_state: dict[str, Any], receipt: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO actions "
            "(action_id, correlation_id, idempotency_key, tool, status, attempts_used, "
            " max_attempts, tool_reported_ok, before_state_json, after_state_json, receipt_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (action_id, correlation_id, idempotency_key, tool, status, attempts_used,
             max_attempts, int(tool_reported_ok), json.dumps(before_state),
             json.dumps(after_state), json.dumps(receipt), _now()),
        )
        self.conn.commit()

    def get_action(self, correlation_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM actions WHERE correlation_id=? ORDER BY created_at DESC LIMIT 1",
            (correlation_id,),
        ).fetchone()
        return dict(row) if row else None

    # ── outcomes ─────────────────────────────────────────────────────────────
    def record_outcome(self, correlation_id: str, intended: dict[str, Any],
                       observed: dict[str, Any], verified: bool,
                       false_success_detected: bool) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO outcomes "
            "(correlation_id, intended_json, observed_json, verified, false_success_detected, verified_at) "
            "VALUES (?,?,?,?,?,?)",
            (correlation_id, json.dumps(intended), json.dumps(observed),
             int(verified), int(false_success_detected), _now()),
        )
        self.conn.commit()

    def get_outcome(self, correlation_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM outcomes WHERE correlation_id=?", (correlation_id,)
        ).fetchone()
        return dict(row) if row else None

    # ── escalations ──────────────────────────────────────────────────────────
    def create_escalation(self, escalation_id: str, correlation_id: str, reason: str,
                          owner: str, attempts_used: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO escalations "
            "(escalation_id, correlation_id, reason, owner, status, attempts_used, created_at) "
            "VALUES (?,?,?,?, 'open', ?, ?)",
            (escalation_id, correlation_id, reason, owner, attempts_used, _now()),
        )
        self.conn.commit()

    def get_escalation(self, correlation_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM escalations WHERE correlation_id=?", (correlation_id,)
        ).fetchone()
        return dict(row) if row else None
