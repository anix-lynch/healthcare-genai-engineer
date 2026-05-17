"""Build-index job — warm the BM25 (and optional dense) retrieval index.

Today: touches the retrieval pipeline so the BM25 index loads + is cached.
In production: this would also push embeddings to pgvector / Vertex Vector
Search and tag the index version for the regression gate.

Run:
    python -m jobs.build_index
"""
from __future__ import annotations
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from retrieval.query_pipeline import QueryPipeline

    t0 = time.time()
    pipeline = QueryPipeline()
    # Warm-up query to force lazy-load of the BM25 index.
    sample = pipeline.retrieve("warm-up query", k=1, method="bm25")
    elapsed = time.time() - t0

    print(f"✅ BM25 index warm")
    print(f"   sample hit: {sample[0]['case_id'] if sample else '(empty)'}")
    print(f"   warm-up time: {elapsed:.2f}s")
    print(f"   ts: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    print(f"   prod swap path: pgvector | qdrant | vertex_vector_search")
    return 0


if __name__ == "__main__":
    sys.exit(main())
