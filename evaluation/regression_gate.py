"""Regression gate — fails the build if eval metrics regress vs baseline.

Use in CI:
    python -m evaluation.regression_gate
        compares outputs/eval_summary.json vs evaluation/baseline.json
        exits 1 if any tracked metric drops by more than its tolerance

Tracked metrics + tolerances:
    any_hit_rate         tolerance 0.05  (5pp drop blocks)
    faithfulness_avg      tolerance 0.05
    relevance_avg          tolerance 0.10
    p95_latency_ms         tolerance +200ms (latency creeping UP blocks)

This is the gate "your repo will not merge if Recall@K dropped 10%."
Mainly here as a signal — the actual CI wiring is queued.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT = REPO_ROOT / "outputs" / "eval_summary.json"
BASELINE = REPO_ROOT / "evaluation" / "baseline.json"

TOLERANCES = {
    "any_hit_rate":      0.05,
    "faithfulness_avg":  0.05,
    "relevance_avg":     0.10,
    "p95_latency_ms":    -200,   # NEGATIVE because higher latency = worse
}


class RegressionViolation(ValueError):
    """Raised when an eval metric drops past its tolerance."""


def check_regression(current: dict, baseline: dict) -> list[str]:
    """Return list of violation messages. Empty = pass."""
    violations: list[str] = []
    cur_agg = current.get("aggregates", {})
    base_agg = baseline.get("aggregates", {})

    for metric, tol in TOLERANCES.items():
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

    print("=== regression gate ===")
    if not violations:
        print("✅ PASS — no metric regressed past tolerance")
        sys.exit(0)
    print(f"❌ FAIL — {len(violations)} violations:")
    for v in violations:
        print(f"  · {v}")
    sys.exit(1)


if __name__ == "__main__":
    main()
