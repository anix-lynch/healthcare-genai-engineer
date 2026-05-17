"""Window+overlap chunking for free-text docs (guidelines, protocols).

Char-based on purpose: fewer deps, deterministic, fast. Swap to tiktoken
if exact token counting becomes critical.
"""
from __future__ import annotations


def chunk_text(text: str, *, chunk_chars: int = 800, overlap_chars: int = 100) -> list[str]:
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
