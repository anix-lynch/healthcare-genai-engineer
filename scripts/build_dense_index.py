"""Pre-encode the corpus into a dense index at IMAGE BUILD time.

Runs during `docker build` (see Dockerfile). Encodes all 497 BM25 corpus
snippets into FastEmbed BGE-small-en-v1.5 (384-dim) and saves the
case_ids + snippets + matrix to a single .npz file.

Why: Cloud Run cold-start corpus encode on 1 CPU takes >180s for 497
snippets. Baking the index at build time moves that cost into the
gcloud builds submit step (which runs on beefier hardware) and the
runtime container just np.load()s the matrix.

At runtime, retrieval/dense.py checks for the pre-built index at
/app/data/dense_index.npz before falling back to runtime encode.

Output: data/dense_index.npz  (kept inside the deployed image)
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from retrieval.dense import _load_model, _encode, DEFAULT_MODEL
from retrieval.retriever import _ensure_index as _bm25_ensure_index


def main() -> int:
    print(f"[build_dense_index] loading BM25 corpus...")
    bm25 = _bm25_ensure_index()
    cases = [{"case_id": d["case_id"], "snippet": d["snippet"]} for d in bm25.docs]
    print(f"[build_dense_index] {len(cases)} cases in corpus")

    print(f"[build_dense_index] loading FastEmbed model: {DEFAULT_MODEL}")
    model = _load_model(DEFAULT_MODEL)

    case_ids = [c["case_id"] for c in cases]
    snippets = [c["snippet"] for c in cases]
    print(f"[build_dense_index] encoding {len(snippets)} snippets...")
    matrix = _encode(model, snippets)
    print(f"[build_dense_index] matrix shape: {matrix.shape}")

    out_path = REPO / "data" / "dense_index.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        case_ids=np.array(case_ids, dtype=object),
        snippets=np.array(snippets, dtype=object),
        matrix=matrix,
        model_name=np.array([DEFAULT_MODEL], dtype=object),
    )
    print(f"[build_dense_index] wrote {out_path} ({out_path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
