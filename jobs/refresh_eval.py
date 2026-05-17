"""Refresh-eval job — run the golden set + check regression gate.

Composes evaluation.ragas_runner + evaluation.regression_gate so a single
cron entry runs both and exits 1 on degradation.

Run:
    python -m jobs.refresh_eval
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("=== refresh_eval ===\n")

    # 1) Run the golden set
    print("step 1: ragas_runner")
    r1 = subprocess.run(
        [sys.executable, "-m", "evaluation.ragas_runner"],
        cwd=str(REPO_ROOT),
    )
    if r1.returncode != 0:
        print("\n❌ ragas_runner failed; aborting before regression gate", file=sys.stderr)
        return r1.returncode

    # 2) Compare vs baseline
    print("\nstep 2: regression_gate")
    r2 = subprocess.run(
        [sys.executable, "-m", "evaluation.regression_gate"],
        cwd=str(REPO_ROOT),
    )
    return r2.returncode


if __name__ == "__main__":
    sys.exit(main())
