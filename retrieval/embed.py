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
import os
from typing import Iterable

import numpy as np

# EMBEDDING_BACKEND=vertex → Vertex AI text-embedding-005 (768-dim, managed)
# EMBEDDING_BACKEND=local  → FastEmbed ONNX (384-dim, local, default)
_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")

if _BACKEND == "vertex":
    from . import vertex_embed as _backend
else:
    from . import dense as _backend  # type: ignore[no-redef]

from . import dense as _dense  # always available for model_name fallback


def embed(text: str) -> np.ndarray:
    """Single-text → unit-normalised vector. Routes to Vertex or local."""
    return _backend.embed(text)


def embed_corpus(texts: Iterable[str], *, batch_size: int = 64) -> np.ndarray:
    """Many-texts → (N, D) unit-normalised matrix."""
    return _backend.embed_corpus(texts, batch_size=batch_size)


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
    """Active embedding model name — local or vertex."""
    return _backend.model_name()


def embedding_dim() -> int:
    """Dimension of active model. 384 (local) or 768 (vertex)."""
    return _backend.embedding_dim()


def active_backend() -> str:
    """'vertex' or 'local' — for audit logs and eval comparison."""
    return _BACKEND
