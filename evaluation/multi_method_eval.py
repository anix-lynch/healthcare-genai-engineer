"""Multi-method retrieval eval — Hit@K · Precision@K · MRR · NDCG@K · p95.

Runs the golden_set against BM25, Dense, and Hybrid and writes a single
baseline_multi.json with per-method aggregates. This is the file the
EvalPanel in app/routers/web.py reads from, and the file regression_gate
compares against.

Why these five metrics:
   Hit@K          → "did we find anything relevant?"  (95%+ = bragable)
   Precision@K    → "how clean is top-K?"             (60%+ = bragable)
   MRR            → "how high did we rank the first   (0.80+ = bragable)
                     relevant doc?"
   NDCG@K         → "is the ranking ORDER good, not   (0.85+ = bragable)
                     just presence?"
   p95 latency    → "how fast end-to-end?"            (varies)

Relevance is binary per (query, retrieved_doc): the doc's parsed
demographic profile must match the query's `expects_condition` AND
fall within `min_age`/`max_age` AND match `expects_gender` (if any).
This is the ER-triage analog of MTEB's binary relevance label.

Run:
    python -m evaluation.multi_method_eval
        → writes evaluation/baseline_multi.json
        → prints per-method ASCII bar comparison

This is the source-of-truth for the EvalPanel numbers. CI runs this
and gates merges on metric drift.
"""
from __future__ import annotations
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "evaluation" / "golden_set.json"
OUT = REPO_ROOT / "evaluation" / "baseline_multi.json"

# If RAG_API_BASE is set, hit a live URL via httpx instead of TestClient.
# Used to score against the deployed Cloud Run image (which has the
# pre-baked FastEmbed dense index — local Python 3.14 can't load it).
LIVE_BASE = os.environ.get("RAG_API_BASE", "").rstrip("/")

METHODS = ("bm25", "dense", "hybrid")
K = 5         # for Hit@K, Precision@K
NDCG_K = 10   # NDCG cutoff

# snippet shape: "62yo Male, Hypertension, Emergency admission, ..."
_SNIPPET_RX = re.compile(
    r"^(?P<age>\d+)yo\s+(?P<gender>Male|Female)\s*,\s*(?P<condition>[A-Za-z ]+?)\s*,",
    re.I,
)


def _parse_snippet(snippet: str) -> dict | None:
    m = _SNIPPET_RX.search(snippet or "")
    if not m:
        return None
    return {
        "age": int(m.group("age")),
        "gender": m.group("gender").title(),
        "condition": m.group("condition").strip().title(),
    }


def _is_relevant(snippet: str, q: dict) -> bool:
    """Binary relevance label for one (query, retrieved-doc) pair.

    Match criteria (all must hold):
        expects_condition  → doc's condition equals it (case-insensitive)
        expects_gender     → doc's gender equals it
        min_age / max_age  → doc's age in [min, max]
    Any criterion that is null in the golden_set is treated as wildcard.
    """
    parsed = _parse_snippet(snippet)
    if parsed is None:
        return False
    if q.get("expects_condition") and parsed["condition"].lower() != q["expects_condition"].lower():
        return False
    if q.get("expects_gender") and parsed["gender"].lower() != q["expects_gender"].lower():
        return False
    min_age, max_age = q.get("min_age"), q.get("max_age")
    if min_age is not None and parsed["age"] < min_age:
        return False
    if max_age is not None and parsed["age"] > max_age:
        return False
    return True


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _ndcg_at_k(rels: list[int], k: int) -> float:
    rels = rels[:k]
    if not any(rels):
        return 0.0
    ideal = sorted(rels, reverse=True)
    idcg = _dcg(ideal)
    return _dcg(rels) / idcg if idcg > 0 else 0.0


def _score_one(client, q: dict, method: str, *, k: int = K) -> dict:
    """Run one golden query through /v1/ask and compute retrieval metrics.

    `client` is either a FastAPI TestClient (local in-process) or an
    httpx.Client pointed at a deployed URL (RAG_API_BASE).
    """
    resp = client.post(
        "/v1/ask",
        json={"query": q["query"], "k": NDCG_K, "method": method},
    )
    if resp.status_code != 200:
        return {
            "id": q["id"], "method": method, "hit_at_k": 0,
            "precision_at_k": 0.0, "rr": 0.0, "ndcg_at_k": 0.0,
            "latency_ms": 0, "error": resp.text[:120],
        }
    data = resp.json()
    citations = data.get("citations", [])[:NDCG_K]
    rels = [1 if _is_relevant(c.get("snippet", ""), q) else 0 for c in citations]
    top_k = rels[:k]

    hit = int(any(top_k))
    prec = sum(top_k) / k if k else 0.0
    # MRR contribution: 1/(first 1-indexed rank of a relevant doc), else 0
    rr = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            rr = 1.0 / i
            break
    ndcg = _ndcg_at_k(rels, NDCG_K)

    return {
        "id": q["id"], "method": method, "hit_at_k": hit,
        "precision_at_k": round(prec, 4),
        "rr": round(rr, 4),
        "ndcg_at_k": round(ndcg, 4),
        "latency_ms": int(data.get("latency_ms", 0)),
    }


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    latencies = sorted(r["latency_ms"] for r in rows if r.get("latency_ms") is not None)
    p95 = latencies[max(0, int(0.95 * len(latencies)) - 1)] if latencies else 0
    n = len(rows)
    return {
        "n_queries": n,
        "hit_at_5": round(mean(r["hit_at_k"] for r in rows), 4),
        "precision_at_5": round(mean(r["precision_at_k"] for r in rows), 4),
        "mrr": round(mean(r["rr"] for r in rows), 4),
        "ndcg_at_10": round(mean(r["ndcg_at_k"] for r in rows), 4),
        "p95_latency_ms": p95,
    }


def main() -> int:
    if LIVE_BASE:
        import httpx
        client = httpx.Client(base_url=LIVE_BASE, timeout=60.0)
        print(f"[multi_eval] hitting LIVE: {LIVE_BASE}")
    else:
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        print(f"[multi_eval] hitting LOCAL TestClient")

    queries = json.loads(GOLDEN.read_text())["queries"]
    print(f"[multi_eval] {len(queries)} golden queries × {len(METHODS)} methods")

    per_method: dict[str, dict] = {}
    for method in METHODS:
        print(f"\n[{method}] running...", flush=True)
        rows = [_score_one(client, q, method) for q in queries]
        agg = _aggregate(rows)
        per_method[method] = {"aggregates": agg, "per_query": rows}
        bar_h  = "█" * int(20 * agg["hit_at_5"])      + "░" * (20 - int(20 * agg["hit_at_5"]))
        bar_p  = "█" * int(20 * agg["precision_at_5"]) + "░" * (20 - int(20 * agg["precision_at_5"]))
        bar_m  = "█" * int(20 * agg["mrr"])           + "░" * (20 - int(20 * agg["mrr"]))
        bar_n  = "█" * int(20 * agg["ndcg_at_10"])    + "░" * (20 - int(20 * agg["ndcg_at_10"]))
        print(f"  Hit@5         [{bar_h}] {agg['hit_at_5']:.3f}")
        print(f"  Precision@5   [{bar_p}] {agg['precision_at_5']:.3f}")
        print(f"  MRR           [{bar_m}] {agg['mrr']:.3f}")
        print(f"  NDCG@10       [{bar_n}] {agg['ndcg_at_10']:.3f}")
        print(f"  p95 latency   {agg['p95_latency_ms']} ms")

    out = {
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "k": K,
        "ndcg_k": NDCG_K,
        "n_queries": len(queries),
        "by_method": per_method,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[multi_eval] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
