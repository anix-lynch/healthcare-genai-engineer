"""FastAPI dependency injection — keeps router code thin.

The pipeline is created once (lazy module-level singleton) so the BM25
index doesn't reload on every request. For multi-process deployments
the index would be loaded per worker.
"""
from __future__ import annotations
from functools import lru_cache

from retrieval.query_pipeline import QueryPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> QueryPipeline:
    """Singleton retrieval pipeline. FastAPI calls this once per worker."""
    return QueryPipeline()
