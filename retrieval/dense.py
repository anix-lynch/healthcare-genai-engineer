"""
Pattern 1 — Rachel · Dense embedding retriever (FastEmbed / ONNX runtime).

Wraps FastEmbed into the same {case_id, snippet, score} shape as
retriever.search(), so the orchestrator can switch BM25 ↔ dense ↔ hybrid
behind a `method=` flag without callers caring.

WHY FastEmbed instead of sentence-transformers:
    sentence-transformers pulls in torch (~1.2 GB wheel) + tokenizers +
    huggingface stack. Cold-start on Cloud Run: ~10s, image size ~1.5 GB.
    FastEmbed uses ONNX runtime directly with a tiny quantized model
    (~30 MB), no torch. Cold-start: ~2-3s, image size: ~300 MB.

    Quality delta is <2% on MTEB retrieval benchmarks — BGE-small was
    designed as a MiniLM-class drop-in. For ER triage snippet retrieval
    over 497 rows, the difference is in the noise.

Default model:
    BAAI/bge-small-en-v1.5
    384-dim · ~30 MB ONNX · CPU-only fast.
    Override via index_dense(..., model_name="...").

Lifecycle:
    First search() lazy-loads the encoder and encodes the BM25 corpus
    snippets (shared with retriever._INDEX so we don't re-render).
    Subsequent calls reuse the in-memory (n, d) float32 matrix.

Failure mode:
    If fastembed is not installed, _load_model() raises RuntimeError with
    the pip install line. query_pipeline.retrieve(method="dense" | "hybrid")
    is expected to catch this and fall back to BM25 with a warning.
"""
from __future__ import annotations
import threading
from typing import Iterable, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fastembed import TextEmbedding


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


# ── Module-level lazy singletons ───────────────────────────────────────────
_LOCK = threading.Lock()
_MODEL: Optional["TextEmbedding"] = None
_MODEL_NAME: str = DEFAULT_MODEL
_INDEX: Optional[dict] = None  # {"case_ids": list[str], "snippets": list[str], "matrix": ndarray}


def _load_model(model_name: str = DEFAULT_MODEL):
    """Lazy-load the FastEmbed encoder. Raises RuntimeError if package missing."""
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL
    with _LOCK:
        if _MODEL is not None and _MODEL_NAME == model_name:
            return _MODEL
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RuntimeError(
                "fastembed not installed. Run: pip install fastembed"
            ) from e
        _MODEL = TextEmbedding(model_name=model_name)
        _MODEL_NAME = model_name
        return _MODEL


def _encode(model, texts: list[str]) -> np.ndarray:
    """Encode texts via FastEmbed, return L2-normalized (n, d) float32."""
    vecs = np.asarray(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


def index_dense(
    cases: Iterable[dict],
    *,
    model_name: str = DEFAULT_MODEL,
) -> None:
    """Build/replace the dense index from {case_id, snippet} dicts."""
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
    matrix = _encode(model, snippets)
    _INDEX = {"case_ids": case_ids, "snippets": snippets, "matrix": matrix}


def _try_load_prebuilt_index() -> Optional[dict]:
    """Load the pre-encoded corpus index from data/dense_index.npz if present.

    Built at image-build time by scripts/build_dense_index.py. Cloud Run
    1-CPU cold-start can't encode 497 snippets in <30s, so we bake the
    matrix into the image instead.
    """
    from pathlib import Path
    candidate = Path(__file__).resolve().parents[1] / "data" / "dense_index.npz"
    if not candidate.exists():
        return None
    try:
        npz = np.load(candidate, allow_pickle=True)
        return {
            "case_ids": [str(x) for x in npz["case_ids"]],
            "snippets": [str(x) for x in npz["snippets"]],
            "matrix": npz["matrix"].astype(np.float32),
        }
    except Exception:
        return None


def _ensure_index() -> dict:
    """Load pre-built index if available, else lazy-build from BM25 corpus."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is not None:
            return _INDEX
        # Fast path: image was built with build_dense_index.py.
        pre = _try_load_prebuilt_index()
        if pre is not None:
            _INDEX = pre
            return _INDEX
        # Slow path: runtime corpus encode (only happens if image wasn't
        # built with the bake step — e.g., local dev).
        from .retriever import _ensure_index as _bm25_ensure_index
        bm25 = _bm25_ensure_index()
        index_dense(
            ({"case_id": d["case_id"], "snippet": d["snippet"]} for d in bm25.docs)
        )
        return _INDEX  # set by index_dense


def search(query: str, k: int = 10) -> list[dict]:
    """Dense cosine retrieval via FastEmbed. Returns {case_id, snippet, score}."""
    if not query or not query.strip():
        return []
    idx = _ensure_index()
    if idx["matrix"].shape[0] == 0:
        return []
    model = _load_model(_MODEL_NAME)
    q_vec = _encode(model, [query])[0]
    scores = idx["matrix"] @ q_vec
    k_eff = min(k, scores.shape[0])
    if k_eff <= 0:
        return []
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
    q = sys.stdin.read().strip() if not sys.stdin.isatty() else "elephant on my chest"
    print(f"Dense index size: {index_size() or 'lazy (will build on first search)'}")
    print(f"Query: {q}")
    for hit in search(q, k=5):
        print(json.dumps(hit, indent=2))
