"""Bed Ops worker — owns acknowledgement, execution, recovery, escalation.

Conceptually a different actor from whoever created the task: the loop creates a
durable `created` task, then the worker performs a SEPARATE `acknowledged`
transition before it touches the tool. That second write is what makes
"receiver acknowledged ownership" a real, queryable event rather than a flag
set at creation time.

Recovery is real machinery: `execute_action` runs the tool up to
`policy.max_attempts`, retrying only on `TransientToolError`, recording
`attempts_used`, and creating a durable human escalation task when the budget
is exhausted. The outcome verifier (`verify_outcome`) RE-READS durable state and
compares it to the intended disposition — it never trusts the tool's `ok` flag,
so an injected false success is caught.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from action_engine.store import ActionStore
from action_engine.tools import Injection, ToolReceipt, TransientToolError, commit_disposition_tool

# Bounded by design. Mirrors the planner's Bed Ops retry budget (2 attempts).
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_BACKOFF_SECONDS = [30, 120]
ESCALATION_OWNER = "charge_nurse_review_queue"
WORKER_OWNER = "bed_ops_worker"


@dataclass(frozen=True)
class ActionRetryPolicy:
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff_seconds: list[int] = field(default_factory=lambda: list(DEFAULT_BACKOFF_SECONDS))


@dataclass
class ActionResult:
    status: str                 # committed | idempotent_skip | exhausted
    attempts_used: int
    max_attempts: int
    tool_reported_ok: bool
    receipt: Optional[ToolReceipt]
    escalation_id: Optional[str]
    errors: list[str]


def acknowledge(store: ActionStore, task_id: str, correlation_id: str) -> bool:
    """Separate receiver-ACK transition: created -> acknowledged."""
    acked = store.acknowledge_task(task_id)
    store.log_event(correlation_id, "task_acknowledged", {"task_id": task_id, "acked": acked})
    return acked


def execute_action(
    *,
    store: ActionStore,
    correlation_id: str,
    idempotency_key: str,
    disposition: str,
    injection: Injection,
    policy: ActionRetryPolicy | None = None,
    sleep: Callable[[float], None] | None = None,
) -> ActionResult:
    """Run the tool with bounded retry; escalate on exhaustion.

    `sleep` is injectable so backoff is deterministic in tests/eval (default
    no-op). Backoff durations are still recorded in the event log.
    """
    policy = policy or ActionRetryPolicy()
    sleep = sleep or (lambda _seconds: None)
    action_id = f"act:{idempotency_key}"
    errors: list[str] = []
    attempts_used = 0
    last_receipt: Optional[ToolReceipt] = None

    for attempt_no in range(1, policy.max_attempts + 1):
        attempts_used = attempt_no
        try:
            receipt = commit_disposition_tool(
                store=store,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                disposition=disposition,
                attempt_no=attempt_no,
                injection=injection,
            )
            last_receipt = receipt
            committed_row = store.get_committed(idempotency_key)
            if receipt.applied:
                # Real durable state change: before.committed False, after True.
                status = "committed"
            elif committed_row is not None:
                # Genuine replay: state already holds this disposition.
                status = "idempotent_skip"
            else:
                # Tool claimed ok but wrote nothing and no prior commit exists
                # => a false success. Record it so the verifier's catch is auditable.
                status = "reported_ok_no_effect"

            # Don't overwrite a prior committed receipt on a genuine replay skip;
            # the committed row is the durable proof of the real transition.
            if status != "idempotent_skip" or store.get_action(correlation_id) is None:
                store.record_action(
                    action_id=action_id, correlation_id=correlation_id,
                    idempotency_key=idempotency_key, tool="commit_disposition",
                    status=status, attempts_used=attempts_used, max_attempts=policy.max_attempts,
                    tool_reported_ok=receipt.ok, before_state=receipt.before_state,
                    after_state=receipt.after_state,
                    receipt={"detail": receipt.detail, "applied": receipt.applied},
                )
            store.log_event(correlation_id, "action_succeeded", {
                "attempt": attempt_no, "status": status, "applied": receipt.applied,
                "before": receipt.before_state, "after": receipt.after_state,
            })
            return ActionResult(
                status=status, attempts_used=attempts_used, max_attempts=policy.max_attempts,
                tool_reported_ok=receipt.ok, receipt=receipt, escalation_id=None, errors=errors,
            )
        except TransientToolError as exc:
            errors.append(str(exc))
            store.log_event(correlation_id, "action_attempt_failed", {
                "attempt": attempt_no, "max_attempts": policy.max_attempts, "error": str(exc),
            })
            if attempt_no < policy.max_attempts:
                backoff = policy.backoff_seconds[min(attempt_no - 1, len(policy.backoff_seconds) - 1)] \
                    if policy.backoff_seconds else 0
                store.log_event(correlation_id, "action_retry_scheduled", {
                    "next_attempt": attempt_no + 1, "backoff_seconds": backoff,
                })
                sleep(backoff)

    # Budget exhausted -> durable, actionable human escalation.
    escalation_id = f"esc:{correlation_id}"
    store.record_action(
        action_id=action_id, correlation_id=correlation_id, idempotency_key=idempotency_key,
        tool="commit_disposition", status="exhausted", attempts_used=attempts_used,
        max_attempts=policy.max_attempts, tool_reported_ok=False,
        before_state={"committed": False}, after_state={"committed": False},
        receipt={"detail": "all attempts failed", "errors": errors},
    )
    store.create_escalation(
        escalation_id=escalation_id, correlation_id=correlation_id,
        reason=f"Bed Ops automation exhausted {attempts_used}/{policy.max_attempts} attempts for "
               f"disposition '{disposition}': {errors[-1] if errors else 'unknown error'}",
        owner=ESCALATION_OWNER, attempts_used=attempts_used,
    )
    store.log_event(correlation_id, "escalation_created", {
        "escalation_id": escalation_id, "owner": ESCALATION_OWNER, "attempts_used": attempts_used,
    })
    return ActionResult(
        status="exhausted", attempts_used=attempts_used, max_attempts=policy.max_attempts,
        tool_reported_ok=False, receipt=last_receipt, escalation_id=escalation_id, errors=errors,
    )


def verify_outcome(
    *,
    store: ActionStore,
    correlation_id: str,
    idempotency_key: str,
    intended_disposition: str,
    action: ActionResult,
) -> dict[str, Any]:
    """Re-read durable state and compare to intent. Never trusts tool.ok.

    Returns the outcome record. `verified` is True only when durable state
    actually holds the intended disposition. `false_success_detected` is True
    when the tool claimed success but durable state does not back it up.
    """
    observed = store.get_committed(idempotency_key)
    observed_disposition = observed["disposition"] if observed else None
    verified = observed_disposition == intended_disposition

    # The tool claimed ok, action wasn't an exhausted failure, yet durable
    # state does not reflect the intended disposition => the tool lied.
    false_success_detected = (
        action.tool_reported_ok and action.status != "exhausted" and not verified
    )

    intended = {"disposition": intended_disposition, "committed": True}
    observed_view = {
        "disposition": observed_disposition,
        "committed": observed is not None,
    }
    store.record_outcome(
        correlation_id=correlation_id, intended=intended, observed=observed_view,
        verified=verified, false_success_detected=false_success_detected,
    )
    stage = (
        "outcome_false_success" if false_success_detected
        else "outcome_verified" if verified
        else "outcome_unverified"
    )
    store.log_event(correlation_id, stage, {
        "intended": intended, "observed": observed_view,
        "verified": verified, "false_success_detected": false_success_detected,
    })
    return {
        "verified": verified,
        "false_success_detected": false_success_detected,
        "intended": intended,
        "observed": observed_view,
    }
