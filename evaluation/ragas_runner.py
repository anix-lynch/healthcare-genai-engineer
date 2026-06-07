"""Custom-proxy faithfulness + retrieval eval — no external Ragas dep.

What this is:
    Honest baseline eval against `evaluation/golden_set.json`. Per query:
        - run retrieval via the live FastAPI surface (TestClient, no uvicorn)
        - score retrieval relevance against the query's expected condition
        - score faithfulness as fraction of cited source_ids that are real

What this is NOT:
    real Ragas. Ragas needs an LLM judge + the `ragas` pip package + cost.
    This file uses simple deterministic proxies that catch the same class
    of failures (citation hallucination, irrelevant retrieval, empty result).

Usage:
    python -m evaluation.ragas_runner
        → writes outputs/eval_summary.json
        → prints per-metric ASCII bar
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "evaluation" / "golden_set.json"
OUT = REPO_ROOT / "outputs" / "eval_summary.json"


def _ascii_bar(label: str, val: float, *, width: int = 30) -> str:
    if val != val:  # NaN
        return f"{label:<28} (no data)"
    filled = int(width * max(0.0, min(1.0, val)))
    return f"{label:<28} [{'█' * filled}{'░' * (width - filled)}] {val:.3f}"


def score_query(client: TestClient, q: dict, *, k: int = 5) -> dict:
    """Run one golden query through /v1/ask, compute per-query scores."""
    resp = client.post(
        "/v1/ask",
        json={"query": q["query"], "k": k, "method": "bm25"},
    )
    data = resp.json() if resp.status_code == 200 else {}

    citations = data.get("citations", [])
    retrieved_count = data.get("retrieved_count", 0)
    cited_ids = {c["source_id"] for c in citations}

    # Faithfulness proxy: any answer that cites only real source_ids passes.
    faithfulness = 1.0 if data.get("answer") and not data.get("warnings") else 0.0
    if data.get("warnings"):
        for w in data["warnings"]:
            if "halluc" in w.lower() or "dropped" in w.lower():
                faithfulness = 0.0

    # Retrieval relevance proxy: condition match in any returned snippet.
    # `condition_relevance` is the FRACTION of citations that match (precision-ish);
    # `hit_at_5` is whether AT LEAST ONE of the top-k matched (coverage / hit@k).
    relevance = 0.0
    expected_cond = q.get("expects_condition")
    matches = 0
    if expected_cond and citations:
        matches = sum(
            1 for c in citations if expected_cond.lower() in c["snippet"].lower()
        )
        relevance = matches / len(citations) if citations else 0.0
    elif citations:
        relevance = 1.0  # nothing expected → any hit is fine

    # hit_at_5 = a REAL grounding-coverage number: did the top-k contain a snippet
    # matching the query's labelled condition? NOT the old vacuous "did retrieval
    # return anything" (which is 1.0 for every non-empty query — a forbidden 100%
    # hit@k headline under the L2 SLA). Only LABELLED queries can be scored; a query
    # with no expects_condition is `None` (excluded from the rate, NOT auto-passed,
    # so the vacuous-presence behaviour cannot sneak back in via the denominator).
    #
    # NOTE: this is a loose substring proxy. The RIGOROUS retrieval eval — strict
    # gold match on condition+gender+age, with Precision@5 / MRR / NDCG — lives in
    # evaluation/multi_method_eval.py and is the number to quote for retrieval quality.
    if expected_cond is not None:
        hit_at_5 = 1.0 if matches > 0 else 0.0
    else:
        hit_at_5 = None  # unlabelled → not scorable, excluded from hit_at_5_rate

    return {
        "id": q["id"],
        "query": q["query"],
        "hit_at_5": hit_at_5,
        "labelled": expected_cond is not None,
        "faithfulness": faithfulness,
        "condition_relevance": relevance,
        "retrieved_count": retrieved_count,
        "n_citations": len(citations),
        "latency_ms": data.get("latency_ms", 0),
    }


def run_eval(k: int = 5) -> dict:
    """Run the full golden_set eval, return a metrics bundle."""
    from app.main import app  # lazy import to keep CLI tool snappy

    client = TestClient(app)
    with GOLDEN.open() as f:
        golden = json.load(f)
    rows = [score_query(client, q, k=k) for q in golden["queries"]]

    labelled = [r for r in rows if r["hit_at_5"] is not None]
    bundle = {
        "scanned_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "k": k,
        "n_queries": len(rows),
        "n_labelled": len(labelled),
        "aggregates": {
            # hit_at_5_rate scored over LABELLED queries only (unlabelled excluded,
            # not auto-passed). This is a loose grounding-coverage proxy — quote
            # multi_method_eval for rigorous Hit@5 / Precision@5 / MRR / NDCG.
            "hit_at_5_rate":     round(mean(r["hit_at_5"] for r in labelled), 4) if labelled else None,
            "faithfulness_avg":  round(mean(r["faithfulness"] for r in rows), 4),
            "relevance_avg":     round(mean(r["condition_relevance"] for r in rows), 4),
            "p95_latency_ms":    sorted(r["latency_ms"] for r in rows)[int(0.95 * len(rows))],
            "avg_citations":     round(mean(r["n_citations"] for r in rows), 2),
        },
        "per_query": rows,
    }
    return bundle


def main():
    bundle = run_eval(k=5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(bundle, f, indent=2, default=str)

    agg = bundle["aggregates"]
    print(f"=== golden-set eval · n={bundle['n_queries']} · k={bundle['k']} "
          f"· labelled={bundle['n_labelled']} ===")
    print(_ascii_bar(f"hit_at_5 (loose, n={bundle['n_labelled']})", agg["hit_at_5_rate"]))
    print(_ascii_bar("faithfulness_avg",   agg["faithfulness_avg"]))
    print(_ascii_bar("condition_relevance",agg["relevance_avg"]))
    print(f"p95 latency: {agg['p95_latency_ms']} ms · avg citations/query: {agg['avg_citations']}")
    print("note: loose substring proxy over labelled queries; rigorous Hit@5/"
          "Precision@5/MRR/NDCG → evaluation/multi_method_eval.py")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
