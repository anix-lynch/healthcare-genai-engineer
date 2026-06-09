"""The state-changing tool: commit a Bed Ops disposition to durable world state.

This is the "Hands" stage. It is the only thing that mutates durable external
state (`committed_dispositions`). Properties that matter for the L2 benchmark:

- Idempotent: the durable write is `INSERT OR IGNORE` on the idempotency key
  (PK), so a replay applies nothing and reports `applied=False`.
- Receipted: every attempt captures before-state and after-state, so a
  successful commit provably shows `before != after`.
- Honestly fallible: failure injection drives the REAL path — `transient`
  makes the tool genuinely raise so the worker's real retry runs; `false_success`
  makes the tool LIE (returns ok=True without writing) so the outcome verifier
  has something real to catch. Nothing here is simulated after the fact.

`Injection` is a test/eval affordance, not production wiring. In production the
directive is simply absent (`mode="none"`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from action_engine.store import ActionStore


class TransientToolError(RuntimeError):
    """A dependency-style failure (timeout / stale state) the worker may retry."""


@dataclass(frozen=True)
class Injection:
    """Deterministic failure-injection directive carried by a fixture."""

    mode: str = "none"            # none | transient | permanent | false_success
    fail_attempts: int = 0        # for transient: fail this many leading attempts

    @classmethod
    def from_raw(cls, raw: Optional[dict[str, Any]]) -> "Injection":
        if not raw:
            return cls()
        return cls(
            mode=raw.get("mode", "none"),
            fail_attempts=int(raw.get("fail_attempts", 0)),
        )


@dataclass(frozen=True)
class ToolReceipt:
    """What actually happened on one tool invocation."""

    ok: bool                      # what the tool CLAIMS (false_success makes this lie)
    applied: bool                 # whether durable state actually changed on this call
    before_state: dict[str, Any]
    after_state: dict[str, Any]
    detail: str


def _state_snapshot(store: ActionStore, idempotency_key: str) -> dict[str, Any]:
    committed = store.get_committed(idempotency_key)
    return {
        "committed": committed is not None,
        "disposition": committed["disposition"] if committed else None,
        "action_seq": committed["action_seq"] if committed else None,
    }


def commit_disposition_tool(
    *,
    store: ActionStore,
    correlation_id: str,
    idempotency_key: str,
    disposition: str,
    attempt_no: int,
    injection: Injection,
) -> ToolReceipt:
    """Execute one attempt of the durable bed-disposition commit.

    Raises `TransientToolError` when injection says this attempt should fail,
    so the worker's bounded retry is exercised for real.
    """
    before = _state_snapshot(store, idempotency_key)

    if injection.mode == "permanent":
        raise TransientToolError(
            f"permanent dependency failure committing {disposition} (attempt {attempt_no})"
        )
    if injection.mode == "transient" and attempt_no <= injection.fail_attempts:
        raise TransientToolError(
            f"transient dependency failure committing {disposition} "
            f"(attempt {attempt_no} of injected {injection.fail_attempts})"
        )

    if injection.mode == "false_success":
        # The tool LIES: claims success but does not write durable state.
        # The outcome verifier must catch this by re-reading state.
        after = _state_snapshot(store, idempotency_key)
        return ToolReceipt(
            ok=True,
            applied=False,
            before_state=before,
            after_state=after,
            detail="tool reported success but performed no durable write (injected false success)",
        )

    applied = store.try_commit_disposition(idempotency_key, correlation_id, disposition)
    after = _state_snapshot(store, idempotency_key)
    detail = (
        f"committed disposition {disposition} (seq {after['action_seq']})"
        if applied
        else f"idempotent skip: {disposition} already committed (seq {after['action_seq']})"
    )
    return ToolReceipt(ok=True, applied=applied, before_state=before, after_state=after, detail=detail)
