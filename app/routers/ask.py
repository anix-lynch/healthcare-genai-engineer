"""POST /v1/ask — the one GenAI workflow.

Flow:
    request → retrieve (BM25 | hybrid)
            → generate grounded answer with citations
            → validate citations against retrieved set
            → return AskResponse
"""
from __future__ import annotations
import time
from fastapi import APIRouter, Depends

from app.dependencies import get_pipeline
from app.schemas import AskRequest, AskResponse, Citation
from generation.generate import generate_answer
from retrieval.query_pipeline import QueryPipeline

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: QueryPipeline = Depends(get_pipeline)) -> AskResponse:
    t0 = time.time()
    hits = pipeline.retrieve(req.query, k=req.k, method=req.method)
    gen = generate_answer(req.query, hits)

    citations = [
        Citation(
            source_id=h["case_id"],
            snippet=(h.get("snippet") or "")[:200],
            similarity=min(1.0, max(0.0, float(h.get("score", 0)) / 12.0)),
        )
        for h in hits
        if h.get("case_id") in set(gen["citations"]) or not gen["citations"]
    ]

    return AskResponse(
        query=req.query,
        answer=gen["answer"],
        citations=citations[: req.k],
        method_used=req.method,
        retrieved_count=len(hits),
        latency_ms=int((time.time() - t0) * 1000),
        warnings=gen.get("warnings", []),
    )
