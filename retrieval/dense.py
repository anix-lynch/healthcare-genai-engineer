"""
Pattern 1 — Rachel · Dense embedding retriever.

Wraps sentence-transformers into the same {case_id, snippet, score} shape
as retriever.search(), so baseline.py can switch BM25 ↔ dense ↔ hybrid
behind a `method=` flag without callers caring.

Why this beats BM25 (on semantic queries):
    BM25 needs token overlap. "patient feels elephant on chest" will never
    retrieve a case that records "substernal pressure" — zero shared tokens.
    Dense MiniLM maps both phrases close in vector space because it was
    trained on natural-language paraphrase pairs.

Why this still loses to a real EHR retriever:
    Snippets are still rendered from a 15-column billing registry. No HPI,
    no notes, no labs. Dense lifts the synonym ceiling; it cannot conjure
    clinical detail that does not exist in the underlying row.

Default model:
    sentence-transformers/all-MiniLM-L6-v2
    384-dim · ~80MB · runs CPU-only fast on M-series.
    Override with index_dense(..., model_name="...").

Lifecycle:
    First search() lazy-loads the model and encodes the BM25 corpus snippets
    (shared with retriever._INDEX so we don't re-render). Subsequent calls
    reuse the in-memory (n, d) matrix.

Failure mode:
    If sentence-transformers is not installed, _load_model() raises
    RuntimeError with the pip install line. baseline.retrieve(method="dense")
    is expected to catch this and fall back to BM25 with fallback_used=True.
"""
from __future__ import annotations
import threading
from typing import Iterable, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Module-level lazy singletons ───────────────────────────────────────────
_LOCK = threading.Lock()
_MODEL: Optional["SentenceTransformer"] = None
_MODEL_NAME: str = DEFAULT_MODEL
_INDEX: Optional[dict] = None  # {"case_ids": list[str], "snippets": list[str], "matrix": ndarray}


def _load_model(model_name: str = DEFAULT_MODEL):
    """Lazy-load the encoder. Raises RuntimeError if package missing."""
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL
    with _LOCK:
        if _MODEL is not None and _MODEL_NAME == model_name:
            return _MODEL
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed. Run: "
                "pip install sentence-transformers"
            ) from e
        _MODEL = SentenceTransformer(model_name)
        _MODEL_NAME = model_name
        return _MODEL


def _encode_corpus(model, snippets: list[str]) -> np.ndarray:
    return model.encode(
        snippets,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def index_dense(
    cases: Iterable[dict],
    *,
    model_name: str = DEFAULT_MODEL,
) -> None:
    """
    Build/replace the dense index from {case_id, snippet} dicts.

    Encodes all snippets once and stores a normalized (n, d) float32 matrix
    so query-time becomes a single matrix-vector dot product.
    """
    global _INDEX
    model = _load_model(model_name)
    case_ids: list[str] = []
    snippets: list[str] = []
    for c in cases:
        case_ids.append(str(c["case_id"]))
        snippets.append(c["snippet"])
    if not snippets:
        _INDEX = {"case_ids": [], "snippets": [], "matrix": np.zeros((0, 1), dtype=np.float32)}
        return
    matrix = _encode_corpus(model, snippets)
    _INDEX = {"case_ids": case_ids, "snippets": snippets, "matrix": matrix}


def _ensure_index() -> dict:
    """
    Lazy-build the dense index from the BM25 corpus (shared snippets, one
    source of truth). If BM25 hasn't loaded yet, this triggers it.
    """
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is not None:
            return _INDEX
        from .retriever import _ensure_index as _bm25_ensure_index
        bm25 = _bm25_ensure_index()
        index_dense(
            ({"case_id": d["case_id"], "snippet": d["snippet"]} for d in bm25.docs)
        )
        return _INDEX  # set by index_dense


def search(query: str, k: int = 10) -> list[dict]:
    """
    Dense cosine retrieval.

    Returns same shape as bm25 retriever.search(): {case_id, snippet, score}.
    `score` here is cosine similarity in [-1, 1], typically [0, 1] for
    natural-language pairs on MiniLM.
    """
    if not query or not query.strip():
        return []
    idx = _ensure_index()
    if idx["matrix"].shape[0] == 0:
        return []
    model = _load_model(_MODEL_NAME)
    q_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)
    scores = idx["matrix"] @ q_vec  # (n,)
    k_eff = min(k, scores.shape[0])
    if k_eff <= 0:
        return []
    # argpartition for top-k, then sort the slice
    top_idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [
        {
            "case_id": idx["case_ids"][int(i)],
            "snippet": idx["snippets"][int(i)],
            "score": float(scores[int(i)]),
        }
        for i in top_idx
    ]


def index_size() -> int:
    if _INDEX is None:
        return 0
    return int(_INDEX["matrix"].shape[0])


if __name__ == "__main__":
    import json, sys
    q = sys.stdin.read().strip() if not sys.stdin.isatty() else "62yo male crushing chest pressure"
    print(f"Dense index size: {index_size() or 'lazy (will build on first search)'}")
    print(f"Query: {q}")
    for hit in search(q, k=5):
        print(json.dumps(hit, indent=2))
