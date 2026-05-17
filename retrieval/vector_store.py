"""
Pattern 1 — Rachel · Vector store facade.

The "where vectors live + how queries hit them" layer. Today it's an
in-memory numpy matrix. The interface is intentionally swap-shaped so
pgvector / Qdrant / Vertex Vector Search can plug in without touching
hybrid_retriever.py or services/rag-api/.

Production swap path:
    pgvector       — drop in psycopg2 + cosine_distance index
    Qdrant         — qdrant-client; same put / query shape
    Vertex Vector  — vertexai matching engine; uses GCP $900 credit
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class StoredHit:
    case_id: str
    score: float          # cosine similarity (already in [-1, 1], usually [0, 1] for normalized embeddings)
    metadata: dict


class InMemoryVectorStore:
    """
    Single-process numpy matrix. Good for ≤ 1M rows, ≤ 1k dim.
    For Layer 1's 55K rows × 384d = ~21 MB. Fits in RAM trivially.

    NOT for prod. NOT for concurrent writes. NOT for HA. Swap to pgvector
    or Vertex Vector Search via the same put() / query() interface.
    """

    def __init__(self):
        self._matrix: np.ndarray | None = None  # (N, D) float32
        self._ids: list[str] = []
        self._meta: list[dict] = []
        self._dim: int | None = None

    # ── write side ─────────────────────────────────────────────────────────
    def put_many(self, case_ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None:
        if len(case_ids) != vectors.shape[0] or len(case_ids) != len(metadata):
            raise ValueError("len(case_ids) must match vectors.shape[0] and len(metadata)")
        if self._dim is None:
            self._dim = vectors.shape[1]
        if vectors.shape[1] != self._dim:
            raise ValueError(f"dim mismatch: got {vectors.shape[1]}, store is {self._dim}")
        # append
        self._matrix = vectors.astype(np.float32) if self._matrix is None \
            else np.vstack([self._matrix, vectors.astype(np.float32)])
        self._ids.extend(case_ids)
        self._meta.extend(metadata)

    # ── read side ──────────────────────────────────────────────────────────
    def query(self, query_vector: np.ndarray, k: int = 10) -> list[StoredHit]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        q = query_vector.astype(np.float32).reshape(-1)
        if q.shape[0] != self._dim:
            raise ValueError(f"query dim {q.shape[0]} != store dim {self._dim}")
        # cosine via dot product (matrix is pre-normalized by embed_corpus)
        scores = self._matrix @ q
        topk = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        topk = topk[np.argsort(-scores[topk])]
        return [
            StoredHit(case_id=self._ids[i], score=float(scores[i]), metadata=self._meta[i])
            for i in topk
        ]

    # ── inspection ─────────────────────────────────────────────────────────
    def size(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[0])

    def dim(self) -> int | None:
        return self._dim


# ── Singleton helper for the default dense path ─────────────────────────────
_DEFAULT_STORE: InMemoryVectorStore | None = None


def default_store() -> InMemoryVectorStore:
    """Return process-wide default store. Lazy-build empty if first call."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = InMemoryVectorStore()
    return _DEFAULT_STORE
