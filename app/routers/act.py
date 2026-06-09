"""POST /v1/act — the one workflow where the service ACTS, not just answers.

/v1/ask retrieves evidence and returns a grounded answer (it advises).
/v1/act drives the durable Bed Ops action loop end to end and returns a
receipt proving a real state change:

    evidence → intake (fail closed) → decision (reuses bed_ops_agent)
            → durable task + receiver ACK → idempotent state-changing tool
            → outcome verified by re-reading state → bounded retry / escalate

The decision logic is identical to the offline action eval — the same
canonical contract, the same `run_action_loop`. The receipt shows
`before_committed != after_committed`, which a recommendation-only RAG
endpoint structurally cannot produce. Replaying the same correlation_id is
idempotent: the durable write applies once, so a second call reports the
existing state without a duplicate side effect.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter

from action_engine.adapters import SyntheticFixtureAdapter
from action_engine.loop import run_action_loop
from action_engine.store import ActionStore
from app.schemas import ActReceipt, ActRequest

router = APIRouter()

# Durable action-state path. Configurable so a deploy can point it at a
# mounted volume; on stateless Cloud Run it is per-instance durable, which is
# enough to prove a real before/after state change and idempotent replay.
_DB_PATH = Path(os.getenv("ACTION_DB_PATH", "outputs/live_action_loop.db"))


@router.post("/act", response_model=ActReceipt)
def act(req: ActRequest) -> ActReceipt:
    record = {
        "correlation_id": req.correlation_id,
        "id": req.correlation_id,
        "evidence": {
            "ingested_at": req.ingested_at or "1970-01-01T00:00:00Z",
            "triage_level": req.triage_level,
            "predicted_los_hours": req.predicted_los_hours,
            "bed_pressure_risk": req.bed_pressure_risk,
            "er_state": req.er_state.model_dump(exclude_none=True),
        },
    }

    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = ActionStore(_DB_PATH)
    try:
        result = run_action_loop(store=store, record=record, adapter=SyntheticFixtureAdapter())
        action = store.get_action(result.correlation_id)
        outcome = store.get_outcome(result.correlation_id)
    finally:
        store.close()

    before_committed = after_committed = None
    if action:
        before_committed = json.loads(action["before_state_json"])["committed"]
        after_committed = json.loads(action["after_state_json"])["committed"]

    return ActReceipt(
        correlation_id=result.correlation_id,
        accepted=result.accepted,
        blocked_reason=result.blocked_reason,
        disposition=result.disposition,
        task_id=result.task_id,
        acknowledged=result.acknowledged,
        before_committed=before_committed,
        after_committed=after_committed,
        outcome_verified=bool(outcome["verified"]) if outcome else None,
        false_success_detected=bool(outcome["false_success_detected"]) if outcome else False,
        attempts_used=action["attempts_used"] if action else None,
        max_attempts=action["max_attempts"] if action else None,
        escalation_id=result.escalation_id,
    )
