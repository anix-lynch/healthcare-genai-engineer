"""Single-call retrieval orchestrator.

Hides the BM25 / dense / hybrid choice behind one interface so app/routers
doesn't have to care which retriever ran. Falls back to BM25 if dense
encoder is missing.
"""
from __future__ import annotations
from typing import Literal

from .retriever import search as _bm25_search
from . import dense as _dense
from .hybrid_retriever import hybrid_search as _hybrid_search

Method = Literal["bm25", "dense", "hybrid"]


class QueryPipeline:
    """Lightweight retrieval facade for the /ask endpoint."""

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        method: Method = "bm25",
    ) -> list[dict]:
        """Return list[{case_id, snippet, score}]. Method picks the engine.

        Fallback discipline: if dense/hybrid raises (encoder missing or
        OOM under Cloud Run cold start), drop silently to BM25 — the
        caller still gets a usable response with method_used reflecting
        what actually ran.
        """
        if method == "dense":
            try:
                hits = _dense.search(query, k=k)
                if hits:
                    return hits
            except Exception:
                pass  # fall through to BM25
        if method == "hybrid":
            try:
                hits = _hybrid_search(query, k=k)
                if hits:
                    return hits
            except Exception:
                pass  # fall through to BM25
        return _bm25_search(query, k=k)
