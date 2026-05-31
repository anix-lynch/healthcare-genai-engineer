"""Vertex AI Embeddings — text-embedding-005.

Replaces local FastEmbed/ONNX with managed Vertex AI embeddings API.
768-dim vs 384-dim (local) — higher quality, state-of-art MTEB scores.

Toggle:
    EMBEDDING_BACKEND=vertex  → this module (Vertex API call)
    EMBEDDING_BACKEND=local   → retrieval/dense.py (FastEmbed ONNX, default)

Auth: uses GOOGLE_APPLICATION_CREDENTIALS SA key (bchan-genai-deploy).
Cost: ~$0.00002 per 1k characters — negligible on 497-record corpus.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

import numpy as np

_log = logging.getLogger(__name__)

VERTEX_MODEL = "text-embedding-005"
VERTEX_PROJECT = os.getenv("GCP_PROJECT", "bchan-genai-lab")
VERTEX_LOCATION = os.getenv("GCP_REGION", "us-central1")
EMBEDDING_DIM = 768  # text-embedding-005 output dimension


def _get_client():
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel
        vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        return TextEmbeddingModel.from_pretrained(VERTEX_MODEL)
    except ImportError:
        raise RuntimeError(
            "google-cloud-aiplatform not installed. "
            "pip install google-cloud-aiplatform"
        )


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _get_client()
    return _CLIENT


def embed(text: str) -> np.ndarray:
    """Single text → 768-dim unit-normalised vector via Vertex AI."""
    model = _client()
    result = model.get_embeddings([text])
    vec = np.array(result[0].values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_corpus(texts: Iterable[str], *, batch_size: int = 5) -> np.ndarray:
    """Many texts → (N, 768) matrix. Batched — Vertex API max 5 per call."""
    model = _client()
    text_list = list(texts)
    vecs = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        results = model.get_embeddings(batch)
        for r in results:
            vec = np.array(r.values, dtype=np.float32)
            norm = np.linalg.norm(vec)
            vecs.append(vec / norm if norm > 0 else vec)
        _log.debug("vertex_embed batch %d/%d done", i + batch_size, len(text_list))
    return np.vstack(vecs)


def model_name() -> str:
    return f"vertex/{VERTEX_MODEL}"


def embedding_dim() -> int:
    return EMBEDDING_DIM


def is_available() -> bool:
    """Check if Vertex credentials + SDK are in place."""
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds or not os.path.exists(creds):
        return False
    try:
        import vertexai  # noqa: F401
        return True
    except ImportError:
        return False
