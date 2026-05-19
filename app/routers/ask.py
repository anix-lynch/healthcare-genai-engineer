"""POST /v1/ask — the one GenAI workflow.

Flow:
    request → input guardrails (sanitize · injection · PII redact)
            → retrieve (BM25 | hybrid)
            → generate grounded answer with citations
            → output guardrails (citations valid · forbidden actions · min length)
            → return AskResponse with all warnings surfaced

Telemetry note:
    The pipeline body is wrapped as a Weave op so the whole request becomes
    a single trace tree rooted at one call_id. We capture that call_id from
    inside the op via weave.get_current_call() — works on both the success
    (200) and failure (HTTPException → 4xx) paths.

    The 4xx case enriches the HTTPException detail dict with trace_call_id
    so observability is consistent across success and rejection responses.
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

try:
    import weave
except Exception:
    weave = None  # type: ignore[assignment]

router = APIRouter()


def _current_trace_id() -> str | None:
    """Return the Weave call_id for the currently-executing op, or None."""
    if weave is None:
        return None
    try:
        call = weave.get_current_call()
    except Exception:
        return None
    if call is None:
        return None
    return getattr(call, "id", None)


def _build_response(req: AskRequest, pipeline: QueryPipeline) -> AskResponse:
    """Core pipeline body. Wrapped as a Weave op below so the trace tree
    has a single root call.

    Captures the active Weave call_id at the top of the body. On the 200
    path it's attached to AskResponse. On the 4xx HTTPException path it's
    attached to the exception detail dict (we re-wrap before re-raising)."""
    trace_id = _current_trace_id()
    t0 = time.time()
    warnings: list[str] = []

    # 1) Input guardrails
    t_guard = time.perf_counter()
    try:
        clean_query = validate_input(req.query)
    except InputGuardError as e:
        msg = str(e)
        if "injection" in msg.lower():
            guard_type = "injection"
        elif "exceeds max" in msg.lower():
            guard_type = "length"
        elif "empty" in msg.lower():
            guard_type = "empty"
        else:
            guard_type = "unknown"
        detail = {"error": "input_guard", "guard_type": guard_type, "message": msg}
        if trace_id:
            detail["trace_call_id"] = trace_id
        raise HTTPException(status_code=400, detail=detail)

    # 2) PII detection on query (redact for logs, retrieve on cleaned text)
    guard_triggered = False
    guard_type_200 = "none"
    cleaned, pii_counts = mask_pii(clean_query)
    if pii_counts:
        warnings.append(f"pii redacted from query: {dict(pii_counts)}")
        clean_query = cleaned
        guard_triggered = True
        guard_type_200 = "pii"
    guard_ms = int((time.perf_counter() - t_guard) * 1000)

    # 3) Retrieve
    t_retrieve = time.perf_counter()
    hits = pipeline.retrieve(clean_query, k=req.k, method=req.method)
    retrieve_ms = int((time.perf_counter() - t_retrieve) * 1000)

    # 4) Generate
    t_generate = time.perf_counter()
    gen = generate_answer(clean_query, hits)
    generate_ms = int((time.perf_counter() - t_generate) * 1000)
    warnings.extend(gen.get("warnings", []))

    # 5) Output guardrails
    valid_ids = {h.get("case_id") for h in hits if h.get("case_id")}
    verdict = validate_output(gen["answer"], valid_source_ids=valid_ids)
    warnings.extend(verdict["warnings"])
    if verdict["hard_failures"]:
        detail = {"error": "output_guard", "hard_failures": verdict["hard_failures"]}
        if trace_id:
            detail["trace_call_id"] = trace_id
        raise HTTPException(status_code=422, detail=detail)

    citations = [
        Citation(
            source_id=h["case_id"],
            snippet=(h.get("snippet") or "")[:200],
            similarity=min(1.0, max(0.0, float(h.get("score", 0)) / 12.0)),
        )
        for h in hits
        if h.get("case_id") in set(gen["citations"]) or not gen["citations"]
    ]

    # 6) ESI classification — rule + RAG-KNN + fuse
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
        guard_ms=guard_ms,
        retrieve_ms=retrieve_ms,
        generate_ms=generate_ms,
        guard_triggered=guard_triggered,
        guard_type=guard_type_200,
        trace_call_id=trace_id,
    )


# Wrap the pipeline body as a Weave op when available so request → retrieve
# → generate becomes ONE trace tree, and weave.get_current_call() inside
# _build_response returns *this* request's root call.
if weave is not None:
    try:
        _build_response = weave.op(_build_response)  # type: ignore[assignment]
    except Exception:
        pass


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: QueryPipeline = Depends(get_pipeline)) -> AskResponse:
    return _build_response(req, pipeline)
