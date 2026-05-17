"""Ingest job — pull source docs into data/processed/.

Today: copies data/raw/healthcare_dataset.csv into data/processed/ with a
basic schema check. In production, this is where you'd pull from Cloud
Storage / S3 / customer EHR feed and normalize before indexing.

Run:
    python -m jobs.ingest_documents
"""
from __future__ import annotations
import csv
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "healthcare_dataset.csv"
PROCESSED = REPO_ROOT / "data" / "processed" / "healthcare_dataset.csv"

REQUIRED_COLS = {"Name", "Age", "Gender", "Medical Condition", "Admission Type"}


def main() -> int:
    if not RAW.exists():
        print(f"ERROR: raw CSV missing at {RAW}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(RAW.open(newline="")))
    if not rows:
        print("ERROR: raw CSV is empty", file=sys.stderr)
        return 1

    headers = set(rows[0].keys())
    missing = REQUIRED_COLS - headers
    if missing:
        print(f"ERROR: schema check failed — missing cols: {sorted(missing)}", file=sys.stderr)
        return 1

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ ingested {len(rows)} rows → {PROCESSED}")
    print(f"   schema: {len(headers)} columns")
    print(f"   ts: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    return 0


if __name__ == "__main__":
    sys.exit(main())
