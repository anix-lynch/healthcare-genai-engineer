"""
Pattern 1 — Rachel · Hybrid retriever (BM25 + dense via RRF).

Reciprocal Rank Fusion combines a BM25 candidate list and a dense candidate
list into a single ranked output. RRF is parameter-light, robust, and
sidesteps the "score normalization across different scorers" headache.

RRF formula (per candidate):
    score_rrf(doc) = Σ over retrievers r:   1 / (k + rank_r(doc))
        k = 60 default (Cormack & Buettcher 2009)
        rank_r(doc) = 1-indexed position of doc in retriever r's output;
                     omit doc if not present

Why hybrid beats either alone:
    BM25 wins on exact-phrase + rare-token queries ("Lipitor 80mg").
    Dense wins on paraphrase + synonym ("elephant on chest" → cardiac).
    Hybrid covers both failure modes with one retriever.

Wraps:
    shared.retrieval.retriever.search()         (existing BM25)
    shared.retrieval.dense.search()             (existing MiniLM dense)
"""
from __future__ import annotations

from .retriever import search as _bm25_search
from . import dense as _dense

RRF_K = 60


def reciprocal_rank_fusion(
    rank_lists: list[list[str]],
    *,
    k: int = RRF_K,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked id lists into a single RRF-scored top-K.

    Args:
        rank_lists: each inner list is doc_ids in rank order (best first).
        k:          RRF smoothing constant. 60 is the canonical default.
        top_k:      cut at top-K.

    Returns:
        list of (doc_id, rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:top_k]


def hybrid_search(
    query: str,
    *,
    k: int = 10,
    bm25_k: int = 50,
    dense_k: int = 50,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """
    Run BM25 + dense, fuse with RRF, return top-K as {case_id, snippet, score}.

    Falls back gracefully:
        - if dense fails (encoder missing), returns BM25-only with a flag.
        - if BM25 returns empty, returns dense-only with a flag.
        - if both empty, returns [].

    The output shape matches retriever.search() so callers can swap one for
    the other without changing downstream code.
    """
    # BM25 side
    bm25_hits = _bm25_search(query, k=bm25_k)
    bm25_rank = [h["case_id"] for h in bm25_hits]
    bm25_lookup = {h["case_id"]: h for h in bm25_hits}

    # Dense side
    dense_failed = False
    try:
        dense_hits = _dense.search(query, k=dense_k)
        dense_rank = [h["case_id"] for h in dense_hits]
        dense_lookup = {h["case_id"]: h for h in dense_hits}
    except Exception:
        dense_failed = True
        dense_rank = []
        dense_lookup = {}

    if not bm25_rank and not dense_rank:
        return []

    rank_lists = [r for r in (bm25_rank, dense_rank) if r]
    fused = reciprocal_rank_fusion(rank_lists, k=rrf_k, top_k=k)

    # Re-attach snippet from whichever retriever knew the doc; prefer BM25's.
    out: list[dict] = []
    for doc_id, rrf_score in fused:
        src = bm25_lookup.get(doc_id) or dense_lookup.get(doc_id)
        if not src:
            continue
        out.append({
            "case_id": doc_id,
            "snippet": src.get("snippet", ""),
            "score": round(rrf_score, 6),     # RRF score (not BM25 / cosine)
            "rrf_source": (
                "bm25+dense" if (dense_rank and not dense_failed) else "bm25_only"
            ),
        })
    return out
