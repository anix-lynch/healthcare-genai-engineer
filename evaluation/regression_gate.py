"""Regression gate — fails the build if eval metrics regress vs baseline.

Use in CI:
    python -m evaluation.regression_gate
        compares outputs/eval_summary.json     vs evaluation/baseline.json        (retrieval/grounding)
        compares outputs/agent_eval_summary.json vs evaluation/agent_baseline.json (agent/handoff)
        exits 1 if any tracked metric drops by more than its tolerance

Tracked metrics + tolerances:
    RETRIEVAL / GROUNDING (eval_summary.json):
        hit_at_5_rate        tolerance 0.05  (5pp drop blocks)  [real hit@k, not vacuous presence]
        faithfulness_avg     tolerance 0.05
        relevance_avg        tolerance 0.10
        p95_latency_ms       tolerance +200ms (latency creeping UP blocks)
    AGENT / HANDOFF (agent_eval_summary.json):
        task_completion_rate tolerance 0.05
        decision_correctness tolerance 0.05
        tool_call_success    tolerance 0.05
        handoff_correctness  tolerance 0.05

This is the gate "your repo will not merge if Recall@K dropped 10% OR the agent
handoff stopped routing to the right owner." Wired into .github/workflows/eval.yml.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT = REPO_ROOT / "outputs" / "eval_summary.json"
BASELINE = REPO_ROOT / "evaluation" / "baseline.json"
AGENT_CURRENT = REPO_ROOT / "outputs" / "agent_eval_summary.json"
AGENT_BASELINE = REPO_ROOT / "evaluation" / "agent_baseline.json"

TOLERANCES = {
    "hit_at_5_rate":     0.05,
    "faithfulness_avg":  0.05,
    "relevance_avg":     0.10,
    "p95_latency_ms":    -200,   # NEGATIVE because higher latency = worse
}

# Agent-execution metrics live in agent_eval_summary.json under "metrics",
# not "aggregates" — higher is better, 5pp drop blocks.
AGENT_TOLERANCES = {
    "task_completion_rate": 0.05,
    "decision_correctness": 0.05,
    "tool_call_success":    0.05,
    "handoff_correctness":  0.05,
}


class RegressionViolation(ValueError):
    """Raised when an eval metric drops past its tolerance."""


def _check(cur_agg: dict, base_agg: dict, tolerances: dict) -> list[str]:
    """Compare one metrics block against its baseline. Empty = pass."""
    violations: list[str] = []
    for metric, tol in tolerances.items():
        cur = cur_agg.get(metric)
        base = base_agg.get(metric)
        if cur is None or base is None:
            continue
        if tol >= 0:
            # higher = better; alert if cur drops by more than tol
            if (base - cur) > tol:
                violations.append(
                    f"{metric}: regressed {base:.3f} → {cur:.3f} (drop > tolerance {tol})"
                )
        else:
            # negative tol = lower-is-better metric (latency); alert if cur rises by more than |tol|
            if (cur - base) > abs(tol):
                violations.append(
                    f"{metric}: regressed {base} → {cur} (rise > tolerance {abs(tol)})"
                )
    return violations


def check_regression(current: dict, baseline: dict) -> list[str]:
    """Retrieval/grounding metrics live under 'aggregates'."""
    return _check(current.get("aggregates", {}), baseline.get("aggregates", {}), TOLERANCES)


def check_agent_regression(current: dict, baseline: dict) -> list[str]:
    """Agent-execution metrics live under 'metrics'."""
    return _check(current.get("metrics", {}), baseline.get("metrics", {}), AGENT_TOLERANCES)


def main():
    if not CURRENT.exists():
        sys.exit(f"current eval missing: {CURRENT}. Run `python -m evaluation.ragas_runner` first.")
    if not BASELINE.exists():
        print(f"no baseline at {BASELINE} — copying current eval as baseline (first run)")
        BASELINE.write_text(CURRENT.read_text())
        sys.exit(0)

    current = json.loads(CURRENT.read_text())
    baseline = json.loads(BASELINE.read_text())
    violations = check_regression(current, baseline)

    # Agent/handoff gate — only runs if both the current run and a baseline exist.
    if AGENT_CURRENT.exists():
        if not AGENT_BASELINE.exists():
            print(f"no agent baseline at {AGENT_BASELINE} — seeding from current agent eval (first run)")
            AGENT_BASELINE.write_text(AGENT_CURRENT.read_text())
        else:
            violations += check_agent_regression(
                json.loads(AGENT_CURRENT.read_text()),
                json.loads(AGENT_BASELINE.read_text()),
            )

    print("=== regression gate ===")
    if not violations:
        print("✅ PASS — no metric regressed past tolerance (retrieval + grounding + agent/handoff)")
        sys.exit(0)
    print(f"❌ FAIL — {len(violations)} violations:")
    for v in violations:
        print(f"  · {v}")
    sys.exit(1)


if __name__ == "__main__":
    main()
