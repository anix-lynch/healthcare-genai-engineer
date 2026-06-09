"""Focused tests for the Phase-1 durable Bed Ops action loop.

Covers the required slice: happy path, replay/idempotency, transient retry,
exhausted escalation, invalid contract (fail closed), and false-success
detection. Each test asserts against DURABLE state (SQLite rows), not just the
in-memory return value, because durable state is what the gate and an auditor
read.
"""
from __future__ import annotations

import pytest

from action_engine.adapters import RawPassthroughAdapter, SyntheticFixtureAdapter
from action_engine.contract import CONTRACT_VERSION, validate_evidence
from action_engine.loop import run_action_loop
from action_engine.store import ActionStore
from action_engine.tools import Injection, TransientToolError, commit_disposition_tool
from action_engine.worker import ActionRetryPolicy


def _record(cid: str, *, triage="NOW", los=6, bp="low", beds=3, occ=70.0, queue=2, inject=None):
    rec = {
        "correlation_id": cid,
        "evidence": {
            "ingested_at": "2026-06-08T00:00:00Z",
            "triage_level": triage,
            "predicted_los_hours": los,
            "bed_pressure_risk": bp,
            "er_state": {"available_beds": beds, "occupancy_pct": occ, "queue_length": queue},
        },
    }
    if inject:
        rec["inject"] = inject
    return rec


@pytest.fixture()
def store(tmp_path):
    s = ActionStore(tmp_path / "loop.db")
    yield s
    s.close()


# ── happy path ───────────────────────────────────────────────────────────────
def test_happy_path_creates_durable_task_acks_executes_and_verifies(store):
    r = run_action_loop(store=store, record=_record("happy-1", beds=3))
    assert r.accepted and r.disposition == "assign_bed"

    # durable task created and acknowledged as a SEPARATE transition
    tasks = store.action_tasks_for("happy-1")
    assert len(tasks) == 1
    assert tasks[0]["status"] == "acknowledged"
    assert tasks[0]["acknowledged_at"] is not None
    assert tasks[0]["created_at"] is not None

    # durable state changed: before != after, committed row exists
    action = store.get_action("happy-1")
    assert action["status"] == "committed"
    assert action["attempts_used"] == 1
    assert store.get_committed("happy-1:assign_bed") is not None

    # outcome verified by re-reading state
    outcome = store.get_outcome("happy-1")
    assert outcome["verified"] == 1 and outcome["false_success_detected"] == 0


# ── replay / idempotency ─────────────────────────────────────────────────────
def test_replay_is_idempotent_no_duplicate_side_effect(store):
    rec = _record("replay-1", beds=2)
    first = run_action_loop(store=store, record=rec)
    second = run_action_loop(store=store, record=rec)

    assert first.action.status == "committed"
    assert second.action.status == "idempotent_skip"
    # the durable world state was applied exactly once
    assert store.count_committed("replay-1:assign_bed") == 1
    # the committed receipt is preserved (not overwritten by the replay skip)
    assert store.get_action("replay-1")["status"] == "committed"


def test_idempotency_is_db_level_unique_constraint(store):
    """The guarantee survives independent tool calls, not an in-memory set."""
    key = "x:assign_bed"
    applied_first = store.try_commit_disposition(key, "x", "assign_bed")
    applied_second = store.try_commit_disposition(key, "x", "assign_bed")
    assert applied_first is True and applied_second is False
    assert store.count_committed(key) == 1


# ── transient retry within budget ────────────────────────────────────────────
def test_transient_failure_retries_then_succeeds_within_budget(store):
    r = run_action_loop(
        store=store,
        record=_record("transient-1", inject={"mode": "transient", "fail_attempts": 1}),
    )
    assert r.action.status == "committed"
    assert r.action.attempts_used == 2  # retry actually executed
    assert r.action.attempts_used <= r.action.max_attempts
    assert store.get_outcome("transient-1")["verified"] == 1
    # the failed first attempt is durably logged
    stages = [e["stage"] for e in store.events_for("transient-1")]
    assert "action_attempt_failed" in stages and "action_retry_scheduled" in stages


def test_retry_never_exceeds_budget(store):
    r = run_action_loop(
        store=store,
        record=_record("budget-1", inject={"mode": "permanent"}),
        policy=ActionRetryPolicy(max_attempts=2),
    )
    assert r.action.attempts_used == 2  # exactly the budget, never more


# ── exhausted failure -> durable human escalation ─────────────────────────────
def test_exhausted_failure_creates_durable_escalation(store):
    r = run_action_loop(
        store=store,
        record=_record("exhaust-1", inject={"mode": "permanent"}),
    )
    assert r.action.status == "exhausted"
    esc = store.get_escalation("exhaust-1")
    assert esc is not None
    assert esc["status"] == "open"
    assert esc["owner"] == "charge_nurse_review_queue"
    assert esc["attempts_used"] == 2
    # no false success: exhausted is honestly unverified, not reported done
    assert store.get_outcome("exhaust-1")["verified"] == 0
    assert store.get_committed("exhaust-1:assign_bed") is None


# ── invalid contract fails closed BEFORE the decision ─────────────────────────
def test_invalid_contract_blocks_before_decision(store):
    r = run_action_loop(
        store=store,
        record=_record("bad-1", triage="EMERGENT", los=-3, bp="extreme"),
    )
    assert r.accepted is False
    assert r.disposition is None  # decision engine never ran
    assert store.get_decision("bad-1") is None
    assert store.action_tasks_for("bad-1") == []
    stages = [e["stage"] for e in store.events_for("bad-1")]
    assert stages == ["intake_blocked"]


def test_contract_version_mismatch_is_rejected():
    raw = {
        "contract_version": "evidence.v0",
        "correlation_id": "v0", "source": "x", "source_id": "x",
        "ingested_at": "t", "triage_level": "NOW",
        "predicted_los_hours": 5, "bed_pressure_risk": "low", "er_state": {},
    }
    result = validate_evidence(raw)
    assert result.ok is False
    assert any("contract_version" in e for e in result.errors)


def test_raw_passthrough_cannot_bypass_validation(store):
    # An adapter that trusts the caller still goes through fail-closed intake.
    bad = {"correlation_id": "rp-1", "triage_level": "NOPE",
           "predicted_los_hours": 5, "bed_pressure_risk": "low"}
    r = run_action_loop(store=store, record=bad, adapter=RawPassthroughAdapter())
    assert r.accepted is False


# ── false-success detection (verifier re-reads state) ─────────────────────────
def test_false_success_is_caught_by_state_reread(store):
    r = run_action_loop(
        store=store,
        record=_record("false-1", inject={"mode": "false_success"}),
    )
    # tool claimed success...
    assert r.action.tool_reported_ok is True
    # ...but durable state was never written, so the verifier catches it
    assert store.get_committed("false-1:assign_bed") is None
    outcome = store.get_outcome("false-1")
    assert outcome["verified"] == 0
    assert outcome["false_success_detected"] == 1


def test_false_success_would_pass_if_verifier_trusted_tool_ok(store):
    """Discriminator: the catch depends on re-reading state, not the ok flag.

    The tool genuinely returns ok=True with no durable write — so any verifier
    that trusted ok would (wrongly) pass it. Proving ok=True here means the
    state re-read in verify_outcome is what does the real work.
    """
    receipt = commit_disposition_tool(
        store=store, correlation_id="d", idempotency_key="d:assign_bed",
        disposition="assign_bed", attempt_no=1, injection=Injection("false_success"),
    )
    assert receipt.ok is True and receipt.applied is False
    assert store.get_committed("d:assign_bed") is None


# ── trace reconstruction ──────────────────────────────────────────────────────
def test_full_trace_reconstructable_from_evidence_to_outcome(store):
    run_action_loop(store=store, record=_record("trace-1", beds=3))
    stages = [e["stage"] for e in store.events_for("trace-1")]
    for expected in ("intake_accepted", "decided", "task_created",
                     "task_acknowledged", "action_succeeded", "outcome_verified"):
        assert expected in stages


# ── decision is the EXISTING engine (no reimplementation) ─────────────────────
def test_decision_reuses_existing_bed_ops_and_varies_with_inputs(store):
    free = run_action_loop(store=store, record=_record("dec-free", beds=3, triage="NOW"))
    saturated = run_action_loop(
        store=store, record=_record("dec-sat", beds=0, occ=98.0, queue=12, triage="NOW", bp="high"))
    assert free.disposition == "assign_bed"
    assert saturated.disposition == "divert"


def test_synthetic_adapter_stamps_contract_version(store):
    raw = SyntheticFixtureAdapter().to_contract(_record("a-1"))
    assert raw["contract_version"] == CONTRACT_VERSION
    assert raw["source"] == "synthetic_fixture"
