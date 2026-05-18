"""POST /v1/ask — the one GenAI workflow.

Flow:
    request → input guardrails (sanitize · injection · PII redact)
            → retrieve (BM25 | hybrid)
            → generate grounded answer with citations
            → output guardrails (citations valid · forbidden actions · min length)
            → return AskResponse with all warnings surfaced
"""
from __future__ import annotations
import time
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_pipeline
from app.schemas import AskRequest, AskResponse, Citation
from generation.generate import generate_answer
from guardrails import (
    validate_input, InputGuardError,
    validate_output,
    mask_pii,
)
from retrieval.query_pipeline import QueryPipeline
from workflows.classify_esi import rule_based_esi, rag_knn_esi, fuse_esi

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: QueryPipeline = Depends(get_pipeline)) -> AskResponse:
    t0 = time.time()
    warnings: list[str] = []

    # 1) Input guardrails
    try:
        clean_query = validate_input(req.query)
    except InputGuardError as e:
        raise HTTPException(status_code=400, detail={"error": "input_guard", "message": str(e)})

    # 2) PII detection on query (redact for logs, retrieve on cleaned text)
    cleaned, pii_counts = mask_pii(clean_query)
    if pii_counts:
        warnings.append(f"pii redacted from query: {dict(pii_counts)}")
        clean_query = cleaned

    # 3) Retrieve
    hits = pipeline.retrieve(clean_query, k=req.k, method=req.method)

    # 4) Generate
    gen = generate_answer(clean_query, hits)
    warnings.extend(gen.get("warnings", []))

    # 5) Output guardrails (soft — surface as warnings, only block on HARD failures)
    valid_ids = {h.get("case_id") for h in hits if h.get("case_id")}
    verdict = validate_output(gen["answer"], valid_source_ids=valid_ids)
    warnings.extend(verdict["warnings"])
    if verdict["hard_failures"]:
        raise HTTPException(
            status_code=422,
            detail={"error": "output_guard", "hard_failures": verdict["hard_failures"]},
        )

    citations = [
        Citation(
            source_id=h["case_id"],
            snippet=(h.get("snippet") or "")[:200],
            similarity=min(1.0, max(0.0, float(h.get("score", 0)) / 12.0)),
        )
        for h in hits
        if h.get("case_id") in set(gen["citations"]) or not gen["citations"]
    ]

    # 6) ESI classification — rule-based floor + RAG-KNN refine + fuse.
    #    Rule-only uses query text (keywords + safety floors); RAG-KNN
    #    weighted-votes the esi_tier_truth labels of the top-K retrieved
    #    cases. Production pattern: rules-as-floor + ML-as-lift.
    rule_tier, red_flags = rule_based_esi(clean_query)
    rag_tier, rag_conf, rag_votes = rag_knn_esi(hits)
    esi_final, esi_conf, disagreement = fuse_esi(rule_tier, red_flags, rag_tier, rag_conf)
    if disagreement:
        warnings.append(
            f"esi_disagreement: rule predicted {rule_tier}, RAG-KNN predicted {rag_tier}. "
            "Rule wins by safety policy; flag for human review."
        )

    return AskResponse(
        query=req.query,
        answer=gen["answer"],
        citations=citations[: req.k],
        method_used=req.method,
        retrieved_count=len(hits),
        latency_ms=int((time.time() - t0) * 1000),
        warnings=warnings,
        esi_rule_based=rule_tier,
        esi_rag_knn=rag_tier,
        esi_final=esi_final,
        esi_confidence=esi_conf,
        esi_disagreement=disagreement,
        esi_red_flags=red_flags,
        esi_votes=rag_votes,
    )
