"""
Pattern 1 — Rachel · Embedding facade.

Thin, interview-grade API over the existing dense.py engine. Keeps
naming aligned with the GenAI Engineer mental model:

    embed(text) → vector
    embed_corpus(snippets) → matrix

What this is:
    a public surface for "how documents become vectors."

What this is NOT:
    a training script. Use the pretrained model. Fine-tuning is out of
    scope for this portfolio.

Chunking strategy:
    Layer 1 snippets are already one-row-per-encounter (no chunking
    needed). For free-text guidelines/protocols, use chunk_text() with
    window+overlap before embedding.
"""
from __future__ import annotations
from typing import Iterable

import numpy as np

from . import dense as _dense


def embed(text: str) -> np.ndarray:
    """Single-text → unit-normalised vector. Lazy-loads the encoder."""
    model = _dense._load_model()
    vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vec[0], dtype=np.float32)


def embed_corpus(texts: Iterable[str], *, batch_size: int = 64) -> np.ndarray:
    """Many-texts → (N, D) unit-normalised matrix. Batched for speed."""
    model = _dense._load_model()
    arr = model.encode(
        list(texts),
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return np.asarray(arr, dtype=np.float32)


def chunk_text(
    text: str,
    *,
    chunk_chars: int = 800,
    overlap_chars: int = 100,
) -> list[str]:
    """
    Window+overlap chunking for long free-text (guidelines, protocols).
    Char-based instead of token-based on purpose: fewer deps, deterministic,
    fast. Swap to tiktoken if exact token-counting becomes critical.
    """
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def model_name() -> str:
    """Public read of the active embedding model (for audit logs)."""
    return _dense._MODEL_NAME or _dense.DEFAULT_MODEL


def embedding_dim() -> int:
    """384 for MiniLM-L6. Used by vector_store to validate shape on insert."""
    _dense._load_model()  # ensure loaded
    # MiniLM hard-codes 384; this stays accurate unless model swapped.
    return embed("dim probe").shape[0]
