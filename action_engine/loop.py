"""The durable Bed Ops action loop — the Phase-1 vertical spine.

Stages (each one a durable write the audit can replay):

    intake (validate contract, fail closed)
      -> decision (REUSES app.bed_ops_agent.decide_bed_disposition)
      -> create durable action task
      -> worker acknowledges ownership (separate transition)
      -> idempotent state-changing tool + before/after receipt
      -> outcome verifier re-reads durable state
      -> exhausted recovery -> durable human escalation task

The decision is intentionally the existing, already-tested Bed Ops function:
that call IS the "evidence caused the decision" link, and it keeps existing
behavior intact. The action engine adds the durable execution spine around it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from app.bed_ops_agent import decide_bed_disposition
from app.schemas import ERState

from action_engine.adapters import SourceAdapter, SyntheticFixtureAdapter
from action_engine.contract import CanonicalEvidence, validate_evidence
from action_engine.store import ActionStore
from action_engine.tools import Injection
from action_engine.worker import ActionResult, ActionRetryPolicy, acknowledge, execute_action, verify_outcome


@dataclass
class LoopResult:
    correlation_id: str
    accepted: bool
    blocked_reason: Optional[list[str]]
    disposition: Optional[str]
    task_id: Optional[str]
    acknowledged: bool
    action: Optional[ActionResult]
    outcome: Optional[dict[str, Any]]
    escalation_id: Optional[str]


def _idempotency_key(correlation_id: str, disposition: str) -> str:
    """Stable key shared by task, action, world-state, and outcome for a case."""
    return f"{correlation_id}:{disposition}"


def _decide(evidence: CanonicalEvidence) -> dict[str, Any]:
    """Run the EXISTING Bed Ops decision over canonical evidence."""
    er = ERState(
        available_beds=evidence.er_state.available_beds,
        occupancy_pct=evidence.er_state.occupancy_pct,
        queue_length=evidence.er_state.queue_length,
    )
    return decide_bed_disposition(
        er_state=er,
        triage_level=evidence.triage_level,
        predicted_los_hours=evidence.predicted_los_hours,
        bed_pressure_risk=evidence.bed_pressure_risk,
    )


def run_action_loop(
    *,
    store: ActionStore,
    record: dict[str, Any],
    adapter: SourceAdapter | None = None,
    injection: Injection | None = None,
    policy: ActionRetryPolicy | None = None,
    sleep: Callable[[float], None] | None = None,
) -> LoopResult:
    """Process one source record through the full durable action loop.

    `record` is a source-specific row; `adapter` maps it to the canonical
    contract. `injection` is an optional failure-injection directive (defaults
    to none / production behavior).
    """
    adapter = adapter or SyntheticFixtureAdapter()
    injection = injection or Injection.from_raw(record.get("inject"))

    raw = adapter.to_contract(record)
    result = validate_evidence(raw)
    correlation_id = (raw.get("correlation_id") or record.get("correlation_id") or "unknown")

    # ── Eyes: fail closed before the decision engine ────────────────────────
    if not result.ok:
        store.record_case(
            correlation_id=correlation_id,
            contract_version=str(raw.get("contract_version")),
            source=str(raw.get("source", adapter.name)),
            source_id=str(raw.get("source_id", "")),
            evidence_json=raw, status="blocked",
        )
        store.log_event(correlation_id, "intake_blocked", {"errors": result.errors})
        return LoopResult(
            correlation_id=correlation_id, accepted=False, blocked_reason=result.errors,
            disposition=None, task_id=None, acknowledged=False, action=None,
            outcome=None, escalation_id=None,
        )

    evidence = result.evidence
    assert evidence is not None
    store.record_case(
        correlation_id=correlation_id, contract_version=evidence.contract_version,
        source=evidence.source, source_id=evidence.source_id,
        evidence_json=raw, status="accepted",
    )
    store.log_event(correlation_id, "intake_accepted", {"lineage": evidence.lineage()})

    # ── Brain: existing Bed Ops decision over canonical evidence ────────────
    decision = _decide(evidence)
    disposition = decision["disposition"]
    store.record_decision(correlation_id, disposition, decision["reason"])
    store.log_event(correlation_id, "decided", {
        "disposition": disposition, "inputs_used": decision["inputs_used"],
    })

    idem = _idempotency_key(correlation_id, disposition)
    task_id = f"task:{idem}"

    # ── Coordination: durable task + separate receiver ACK ──────────────────
    store.create_task(task_id, correlation_id, kind="action", owner="bed_ops_worker", idempotency_key=idem)
    store.log_event(correlation_id, "task_created", {"task_id": task_id, "idempotency_key": idem})
    acked = acknowledge(store, task_id, correlation_id)
    # Replay: task already acknowledged on the first pass; durable state is the
    # source of truth, so treat an already-acknowledged task as owned.
    if not acked:
        task = store.get_task(task_id)
        acked = bool(task and task["status"] == "acknowledged")

    # ── Hands + Brakes: idempotent execution with bounded retry/escalation ──
    action = execute_action(
        store=store, correlation_id=correlation_id, idempotency_key=idem,
        disposition=disposition, injection=injection, policy=policy, sleep=sleep,
    )

    # ── Nerves: verify outcome by re-reading durable state ──────────────────
    outcome = verify_outcome(
        store=store, correlation_id=correlation_id, idempotency_key=idem,
        intended_disposition=disposition, action=action,
    )

    return LoopResult(
        correlation_id=correlation_id, accepted=True, blocked_reason=None,
        disposition=disposition, task_id=task_id, acknowledged=acked,
        action=action, outcome=outcome, escalation_id=action.escalation_id,
    )
