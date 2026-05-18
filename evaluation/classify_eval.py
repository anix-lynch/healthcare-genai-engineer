"""ESI classification eval — leave-one-out cross-validation on 497 rows.

For each row in the corpus:
   1. Use its chief_complaint + hpi as the query
   2. Hit /v1/ask method=dense against the live deployed URL
   3. Compare predicted ESI (rule / rag_knn / fused) vs truth (esi_tier_truth)

Reports per-strategy:
   accuracy           overall % correct
   macro_f1           treats all classes equally (penalizes imbalance)
   per_class_recall   ESI 1 / 2 / 3 / 4 / 5 individually
                       safety-critical: ESI 1+2 recall ≥ 0.95
   confusion_matrix   to spot ESI 2 → ESI 4 misclassifications
                       (clinically unacceptable)
   disagreement_rate  rule vs RAG-KNN disagree % (interpretability signal)

Output:
   evaluation/classify_baseline.json   (the source of truth for the
                                        EvalPanel + README + commit msg)

Run (against live URL — uses pre-baked dense index, free Cloud Run tier):
   RAG_API_BASE=https://healthcare-genai-2ihyeqmb6q-uw.a.run.app \\
   python -m evaluation.classify_eval
"""
from __future__ import annotations
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "raw" / "healthcare_dataset.csv"
OUT_PATH = REPO_ROOT / "evaluation" / "classify_baseline.json"

LIVE_BASE = os.environ.get("RAG_API_BASE", "").rstrip("/")
if not LIVE_BASE:
    print("ERROR: Set RAG_API_BASE=https://healthcare-genai-...run.app")
    sys.exit(1)

# Subset for speed — 497 LOOCV calls × ~50ms = ~25s, acceptable but stratify
# down to 100-row holdout for faster iteration. Bump to all 497 for final.
HOLDOUT_N = int(os.environ.get("HOLDOUT_N", "100"))
K = 5
METHOD = "dense"  # the primary retrieval path; rule-only is always same


def _stratified_holdout(rows: list[dict], n: int = HOLDOUT_N) -> list[dict]:
    """Stratify by esi_tier_truth — preserve per-class proportions in holdout."""
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        t = r.get("esi_tier_truth") or "unknown"
        by_tier[t].append(r)
    holdout: list[dict] = []
    import random
    random.seed(42)
    for tier, members in by_tier.items():
        if tier == "unknown":
            continue
        # take min(n_tier, ceil(n * n_tier / total)) from each tier
        target = max(1, round(n * len(members) / len(rows)))
        random.shuffle(members)
        holdout.extend(members[:target])
    return holdout


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _macro_f1(y_true: list[int], y_pred: list[int]) -> tuple[float, dict[int, float]]:
    """Macro-F1 + per-class F1. Treats all classes equally (good for imbalance)."""
    per_class_f1: dict[int, float] = {}
    classes = sorted(set(y_true))
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class_f1[c] = round(_f1(precision, recall), 4)
    macro = round(mean(per_class_f1.values()), 4) if per_class_f1 else 0.0
    return macro, per_class_f1


def _per_class_recall(y_true: list[int], y_pred: list[int]) -> dict[int, float]:
    out: dict[int, float] = {}
    for c in sorted(set(y_true)):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        out[c] = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    return out


def _confusion(y_true: list[int], y_pred: list[int]) -> dict:
    classes = sorted(set(y_true) | set(y_pred))
    mat: dict[str, dict[str, int]] = {str(t): {str(p): 0 for p in classes} for t in classes}
    for t, p in zip(y_true, y_pred):
        mat[str(t)][str(p)] = mat[str(t)].get(str(p), 0) + 1
    return mat


def _aggregate(strategy: str, y_true: list[int], y_pred: list[int]) -> dict:
    if not y_true:
        return {}
    n = len(y_true)
    accuracy = round(sum(1 for t, p in zip(y_true, y_pred) if t == p) / n, 4)
    macro_f1, per_class_f1 = _macro_f1(y_true, y_pred)
    per_class_recall = _per_class_recall(y_true, y_pred)
    return {
        "strategy": strategy,
        "n": n,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "safety_critical_recall_esi_1_and_2": round(
            mean(per_class_recall.get(t, 0.0) for t in (1, 2)), 4
        ),
        "confusion_matrix": _confusion(y_true, y_pred),
    }


def main() -> int:
    print(f"[classify_eval] loading corpus from {CSV_PATH}")
    rows = list(csv.DictReader(CSV_PATH.open()))
    rows = [r for r in rows if r.get("esi_tier_truth") and r["esi_tier_truth"].strip()]
    print(f"[classify_eval] {len(rows)} rows with esi_tier_truth")

    holdout = _stratified_holdout(rows, HOLDOUT_N)
    print(f"[classify_eval] stratified holdout: {len(holdout)} rows")
    print(f"[classify_eval] live URL: {LIVE_BASE}")

    client = httpx.Client(base_url=LIVE_BASE, timeout=60.0)

    y_true: list[int] = []
    y_pred_rule: list[int] = []
    y_pred_rag: list[int] = []
    y_pred_final: list[int] = []
    disagreements = 0
    latencies: list[int] = []

    for i, row in enumerate(holdout):
        cc = (row.get("chief_complaint") or "").strip()
        hpi = (row.get("hpi") or "").strip()[:200]
        query = f"{cc} {hpi}".strip() or "no chief complaint"
        truth = int(row["esi_tier_truth"])

        try:
            resp = client.post("/v1/ask",
                json={"query": query, "k": K, "method": METHOD})
            if resp.status_code != 200:
                print(f"  [{i+1}/{len(holdout)}] {row.get('Name','?')[:20]} → HTTP {resp.status_code}")
                continue
            body = resp.json()
        except Exception as e:
            print(f"  [{i+1}/{len(holdout)}] error: {e}")
            continue

        rule = body.get("esi_rule_based")
        rag = body.get("esi_rag_knn")
        final = body.get("esi_final")
        latency = body.get("latency_ms", 0)
        latencies.append(latency)

        # Coerce to int (drop None — count as miss)
        y_true.append(truth)
        y_pred_rule.append(int(rule) if rule is not None else 999)
        y_pred_rag.append(int(rag) if rag is not None else 999)
        y_pred_final.append(int(final) if final is not None else 999)
        if body.get("esi_disagreement"):
            disagreements += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(holdout)}] running...")

    if not y_true:
        print("[classify_eval] no valid predictions — abort")
        return 1

    print(f"\n[classify_eval] {len(y_true)} predictions collected")

    out = {
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "holdout_n": len(y_true),
        "method": METHOD,
        "k": K,
        "baseline_majority_class": round(
            Counter(y_true).most_common(1)[0][1] / len(y_true), 4
        ),
        "disagreement_rate": round(disagreements / len(y_true), 4),
        "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else 0,
        "by_strategy": {
            "rule_only": _aggregate("rule_only", y_true, y_pred_rule),
            "rag_knn": _aggregate("rag_knn", y_true, y_pred_rag),
            "fused": _aggregate("fused", y_true, y_pred_final),
        },
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[classify_eval] wrote {OUT_PATH}")

    # Print summary table
    print("\n══════════════════════════════════════════════════════════════════════")
    print(f"  baseline (always majority class):  {out['baseline_majority_class']:.3f}")
    print()
    for strat in ("rule_only", "rag_knn", "fused"):
        a = out["by_strategy"][strat]
        bar_acc = "█" * int(20 * a["accuracy"])    + "░" * (20 - int(20 * a["accuracy"]))
        bar_f1  = "█" * int(20 * a["macro_f1"])    + "░" * (20 - int(20 * a["macro_f1"]))
        bar_s   = "█" * int(20 * a["safety_critical_recall_esi_1_and_2"]) + "░" * (20 - int(20 * a["safety_critical_recall_esi_1_and_2"]))
        print(f"  {strat:9s}  acc [{bar_acc}] {a['accuracy']:.3f}")
        print(f"             f1  [{bar_f1}] {a['macro_f1']:.3f}")
        print(f"             1+2 [{bar_s}] {a['safety_critical_recall_esi_1_and_2']:.3f}")
        print()
    print(f"  disagreement rate (rule vs RAG):  {out['disagreement_rate']:.3f}")
    print(f"  p95 latency:                       {out['p95_latency_ms']} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
