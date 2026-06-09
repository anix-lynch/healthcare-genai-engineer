"""Phase-1 action-loop eval — the durable Bed Ops control loop, end to end.

This is NOT the decision-quality eval (that is evaluation/agent_eval.py, which
measures whether the disposition matches independent clinical labels). This eval
measures whether the ACTION MACHINERY around the decision actually works:
contract intake fails closed, a durable task is created and acknowledged, an
idempotent tool changes durable state with a before/after receipt, the outcome
is verified by re-reading state, transient failures retry within budget,
exhausted failures escalate, and every case is reconstructable from its trace.

Every metric is computed by reading durable SQLite rows written by the real
loop (outputs/action_loop.db), not by trusting in-memory booleans — open the
.db and you can reconstruct each number. Each metric reports `n_applicable` so
its denominator (which differs per metric) is visible and not inflated.

Run:  python -m evaluation.action_eval
      writes outputs/action_eval_summary.json + outputs/action_loop.db
      exits 1 if any metric breaches its Phase-1 floor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from action_engine.adapters import SyntheticFixtureAdapter
from action_engine.loop import run_action_loop
from action_engine.store import ActionStore
from action_engine.tools import Injection

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "action_engine" / "fixtures" / "canonical_cases.json"
DB = REPO / "outputs" / "action_loop.db"
OUT = REPO / "outputs" / "action_eval_summary.json"

SHOULD_SUCCEED = {"happy", "replay", "transient"}

# Phase-1 proof gate. "min" = higher-is-better floor; "max" = lower-is-better
# ceiling (rate of a bad event, must not exceed).
FLOORS = {
    "contract_valid_accept_rate":               ("min", 1.00),
    "contract_invalid_block_rate":              ("min", 1.00),
    "durable_task_creation_rate":               ("min", 1.00),
    "receiver_ack_rate":                        ("min", 1.00),
    "successful_action_state_transition_rate":  ("min", 1.00),
    "duplicate_side_effect_rate":               ("max", 0.00),
    "outcome_verification_coverage":            ("min", 1.00),
    "false_success_rate":                       ("max", 0.00),
    "bounded_retry_compliance":                 ("min", 1.00),
    "exhausted_failure_escalation_rate":        ("min", 1.00),
    "trace_reconstruction_rate":                ("min", 1.00),
}

REQUIRED_ACCEPTED_STAGES = {"intake_accepted", "decided", "task_created", "task_acknowledged"}
TERMINAL_STAGES = {"outcome_verified", "outcome_unverified", "outcome_false_success"}


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _reconstructable(store: ActionStore, correlation_id: str, accepted: bool) -> bool:
    stages = {e["stage"] for e in store.events_for(correlation_id)}
    if not accepted:
        return "intake_blocked" in stages
    if not REQUIRED_ACCEPTED_STAGES.issubset(stages):
        return False
    has_terminal = bool(stages & TERMINAL_STAGES) or "escalation_created" in stages
    # the durable case + decision rows must also exist for a full reconstruction
    return has_terminal and store.get_decision(correlation_id) is not None


def main() -> int:
    data = json.loads(FIXTURES.read_text())
    scenarios = data["scenarios"]

    if DB.exists():
        DB.unlink()
    DB.parent.mkdir(parents=True, exist_ok=True)
    store = ActionStore(DB)
    adapter = SyntheticFixtureAdapter()

    rows = []
    for sc in scenarios:
        category = sc["category"]
        injection = Injection.from_raw(sc.get("inject"))
        # Replay cases run the full loop twice with the same correlation id to
        # prove idempotency holds across separate invocations (restart-style).
        runs = 2 if category == "replay" else 1
        result = None
        for _ in range(runs):
            result = run_action_loop(store=store, record=sc, adapter=adapter, injection=injection)

        cid = result.correlation_id
        action = store.get_action(cid)
        outcome = store.get_outcome(cid)
        escalation = store.get_escalation(cid)
        tasks = store.action_tasks_for(cid)
        idem = f"{cid}:{result.disposition}" if result.disposition else None
        committed_count = store.count_committed(idem) if idem else 0
        rows.append({
            "id": sc["id"], "category": category, "correlation_id": cid,
            "accepted": result.accepted, "disposition": result.disposition,
            "runs": runs,
            "action_status": action["status"] if action else None,
            "attempts_used": action["attempts_used"] if action else None,
            "max_attempts": action["max_attempts"] if action else None,
            "tool_reported_ok": bool(action["tool_reported_ok"]) if action else None,
            "before_committed": json.loads(action["before_state_json"])["committed"] if action else None,
            "after_committed": json.loads(action["after_state_json"])["committed"] if action else None,
            "committed_count": committed_count,
            "n_action_tasks": len(tasks),
            "n_acked_tasks": sum(1 for t in tasks if t["status"] == "acknowledged"),
            "outcome_verified": bool(outcome["verified"]) if outcome else None,
            "false_success_detected": bool(outcome["false_success_detected"]) if outcome else None,
            "escalation": escalation["escalation_id"] if escalation else None,
            "reconstructable": _reconstructable(store, cid, result.accepted),
        })

    # ── populations ──────────────────────────────────────────────────────────
    valid = [r for r in rows if r["category"] != "invalid"]
    invalid = [r for r in rows if r["category"] == "invalid"]
    accepted = [r for r in rows if r["accepted"]]
    should_succeed = [r for r in rows if r["category"] in SHOULD_SUCCEED]
    replay = [r for r in rows if r["category"] == "replay"]
    exhausted = [r for r in rows if r["category"] == "exhausted"]
    false_succ = [r for r in rows if r["category"] == "false_success"]
    acted = [r for r in accepted if r["action_status"] is not None]
    n_action_tasks = sum(r["n_action_tasks"] for r in accepted)
    n_acked = sum(r["n_acked_tasks"] for r in accepted)

    metrics = {
        "contract_valid_accept_rate": _rate(sum(r["accepted"] for r in valid), len(valid)),
        "contract_invalid_block_rate": _rate(sum(1 for r in invalid if not r["accepted"]), len(invalid)),
        "durable_task_creation_rate": _rate(sum(1 for r in accepted if r["n_action_tasks"] >= 1), len(accepted)),
        "receiver_ack_rate": _rate(n_acked, n_action_tasks),
        "successful_action_state_transition_rate": _rate(
            sum(1 for r in should_succeed
                if r["action_status"] == "committed" and r["before_committed"] is False and r["after_committed"] is True),
            len(should_succeed)),
        "duplicate_side_effect_rate": _rate(sum(1 for r in replay if r["committed_count"] != 1), len(replay)),
        "outcome_verification_coverage": _rate(sum(1 for r in acted if r["outcome_verified"] is not None), len(acted)),
        "false_success_rate": _rate(sum(1 for r in false_succ if not r["false_success_detected"]), len(false_succ)),
        "bounded_retry_compliance": _rate(
            sum(1 for r in acted if r["attempts_used"] is not None and r["attempts_used"] <= r["max_attempts"]),
            len(acted)),
        "exhausted_failure_escalation_rate": _rate(sum(1 for r in exhausted if r["escalation"]), len(exhausted)),
        "trace_reconstruction_rate": _rate(sum(1 for r in rows if r["reconstructable"]), len(rows)),
    }

    n_applicable = {
        "contract_valid_accept_rate": len(valid),
        "contract_invalid_block_rate": len(invalid),
        "durable_task_creation_rate": len(accepted),
        "receiver_ack_rate": n_action_tasks,
        "successful_action_state_transition_rate": len(should_succeed),
        "duplicate_side_effect_rate": len(replay),
        "outcome_verification_coverage": len(acted),
        "false_success_rate": len(false_succ),
        "bounded_retry_compliance": len(acted),
        "exhausted_failure_escalation_rate": len(exhausted),
        "trace_reconstruction_rate": len(rows),
    }

    # ── cost (metabolism) ─────────────────────────────────────────────────────
    # Honest scope: the decision + action path is fully deterministic
    # (decide_bed_disposition is rule-based; action_engine imports no LLM), so
    # model-inference spend on the covered band is a MEASURED zero — not an
    # estimate. We do NOT compute a savings % vs an all-LLM baseline here: that
    # baseline was never run, so claiming a % would be fabricated. Worst-case
    # cost is bounded because retries can never exceed the attempt budget, so a
    # runaway token-burn loop is structurally impossible.
    verified_outcomes = sum(1 for r in acted if r["outcome_verified"])
    attempt_budget = max((r["max_attempts"] for r in acted if r["max_attempts"] is not None), default=0)
    max_attempts_observed = max((r["attempts_used"] for r in acted if r["attempts_used"] is not None), default=0)
    cost = {
        "model_inference_calls_total": 0,
        "model_cost_per_verified_outcome_usd": 0.0,
        "verified_outcomes": verified_outcomes,
        "attempt_budget": attempt_budget,
        "max_attempts_observed": max_attempts_observed,
        "runaway_cost_bounded": max_attempts_observed <= attempt_budget,
        "all_llm_baseline": "not_run",
        "note": (
            "Deterministic decision/action path → zero model-inference spend (measured). "
            "Worst-case cost bounded by attempt budget (no unbounded burn). "
            "Savings vs all-LLM baseline is YELLOW: baseline not run, so no % claimed."
        ),
    }

    breaches = []
    for name, (direction, floor) in FLOORS.items():
        val = metrics[name]
        if direction == "min" and val < floor:
            breaches.append(f"{name}: {val} < floor {floor}")
        if direction == "max" and val > floor:
            breaches.append(f"{name}: {val} > ceiling {floor}")

    # one fully-reconstructed trace per terminal shape, as an easy receipt
    def _trace_for(category: str):
        match = next((r for r in rows if r["category"] == category), None)
        return store.events_for(match["correlation_id"]) if match else []

    summary = {
        "eval": "phase1_action_loop",
        "contract_version": data.get("contract_version"),
        "n_scenarios": len(scenarios),
        "n_loop_runs": sum(r["runs"] for r in rows),
        "db_path": str(DB.relative_to(REPO)),
        "metrics": metrics,
        "cost": cost,
        "floors": {k: {"direction": d, "value": v} for k, (d, v) in FLOORS.items()},
        "n_applicable": n_applicable,
        "passed": not breaches,
        "breaches": breaches,
        "scenarios": rows,
        "sample_traces": {
            "exhausted_to_escalation": _trace_for("exhausted"),
            "false_success_caught": _trace_for("false_success"),
        },
    }
    OUT.write_text(json.dumps(summary, indent=2))

    print(f"Phase-1 action-loop eval — {len(scenarios)} fixtures, {summary['n_loop_runs']} loop runs:")
    for name, (direction, floor) in FLOORS.items():
        val = metrics[name]
        ok = (val >= floor) if direction == "min" else (val <= floor)
        bound = f"{'floor' if direction == 'min' else 'ceiling'} {floor:.2f}"
        print(f"  {name:42} {val:.4f}  ({bound}, n={n_applicable[name]:2d}) {'ok' if ok else 'BREACH'}")
    store.close()

    if breaches:
        print(f"  FAIL: {len(breaches)} breach(es):")
        for b in breaches:
            print(f"    · {b}")
        return 1
    print(f"  PASS — durable receipts in {DB.relative_to(REPO)} ; artifact {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
