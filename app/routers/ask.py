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
from app.grounding import build_grouped_evidence, enrich_citations
from app.prediction import get_prediction_signal
from app.schemas import AskRequest, AskResponse, Citation, PredictionSignal, TriageLevel
from generation.citations import extract_citations
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


def _identity_op(fn):
    return fn


_stage_op = getattr(weave, "op", _identity_op) if weave is not None else _identity_op


@_stage_op
def input_guard(query: str) -> str:
    """Stage 1 — sanitize and reject clearly malicious input."""
    return validate_input(query)


@_stage_op
def pii_redact(query: str) -> tuple[str, bool, str, list[str]]:
    """Stage 2 — redact soft PII hits while preserving retrievable text."""
    warnings: list[str] = []
    guard_triggered = False
    guard_type = "none"
    cleaned, pii_counts = mask_pii(query)
    if pii_counts:
        warnings.append(f"pii redacted from query: {dict(pii_counts)}")
        guard_triggered = True
        guard_type = "pii"
    return cleaned, guard_triggered, guard_type, warnings


@_stage_op
def classify_rule(query: str) -> tuple[int, list[str]]:
    """Stage 4 — deterministic safety-floor triage vote."""
    return rule_based_esi(query)


@_stage_op
def classify_rag(hits: list[dict]) -> tuple[int | None, float, dict[int, int]]:
    """Stage 5 — retrieval-driven triage vote."""
    return rag_knn_esi(hits)


@_stage_op
def generate_grounded_answer(query: str, hits: list[dict]) -> dict:
    """Stage 6 — grounded answer generation."""
    return generate_answer(query, hits)


@_stage_op
def output_guard_and_fuse(
    *,
    answer: str,
    hits: list[dict],
    rule_tier: int,
    red_flags: list[str],
    rag_tier: int | None,
    rag_conf: float,
    rag_votes: dict[int, int],
) -> tuple[list[str], list[Citation], int | None, float | None, bool, list[str], dict[int, int]]:
    """Stage 7 — validate the answer, build citations, and fuse triage votes."""
    warnings: list[str] = []
    valid_ids = {h.get("case_id") for h in hits if h.get("case_id")}
    verdict = validate_output(answer, valid_source_ids=valid_ids)
    warnings.extend(verdict["warnings"])
    cited_ids, _ = extract_citations(answer, valid_ids)
    citations = [
        Citation(
            source_id=h["case_id"],
            snippet=(h.get("snippet") or "")[:200],
            similarity=min(1.0, max(0.0, float(h.get("score", 0)) / 12.0)),
        )
        for h in hits
        if h.get("case_id") in set(cited_ids) or not cited_ids
    ]
    esi_final, esi_conf, disagreement = fuse_esi(rule_tier, red_flags, rag_tier, rag_conf)
    if disagreement:
        warnings.append(
            f"esi_disagreement: rule predicted {rule_tier}, RAG-KNN predicted {rag_tier}. "
            "Rule wins by safety policy; flag for human review."
        )
    return (
        warnings,
        citations,
        esi_final,
        esi_conf,
        disagreement,
        verdict["hard_failures"],
        rag_votes,
    )


def _triage_level_from_esi(esi: int | None) -> TriageLevel | None:
    if esi is None:
        return None
    if esi <= 2:
        return "NOW"
    if esi == 3:
        return "SOON"
    return "WAIT"


@_stage_op
def prediction_signal_node(
    query: str,
    hits: list[dict],
    *,
    er_state,
    triage_level: TriageLevel | None,
) -> PredictionSignal:
    """Stage 6 — future-risk signal for planning and monitoring."""
    return get_prediction_signal(
        {"query": query, "hits": hits, "triage_level": triage_level},
        er_state=er_state,
    )


@_stage_op
def orchestration_override_rules(
    *,
    triage_level: TriageLevel | None,
    prediction_signal: PredictionSignal,
    red_flags: list[str],
) -> tuple[list[str], bool, str | None, list[str], str]:
    """Decide how prediction can influence operations without breaking safety."""
    decision_basis: list[str] = []
    operational_recommendations = list(prediction_signal.recommended_operational_actions)
    override_applied = False
    override_reason: str | None = None

    if red_flags:
        decision_basis.append(f"Safety rule triggers: {', '.join(red_flags[:3])}")
    if triage_level:
        decision_basis.append(f"Acute triage level: {triage_level}")
    decision_basis.append(
        f"Prediction signal: risk={prediction_signal.risk_level}, "
        f"deterioration={prediction_signal.deterioration_risk}, "
        f"bed_pressure={prediction_signal.bed_pressure_risk}"
    )

    if triage_level == "NOW":
        operational_recommendations = [
            "Immediate clinician review",
            "Notify charge nurse",
            *operational_recommendations,
        ]
        if prediction_signal.deterioration_risk != "high":
            override_applied = True
            override_reason = "Acute safety rule: triage NOW cannot be downgraded by prediction."
            decision_basis.append("Conflict surfaced: prediction appears lower-risk than acute triage, but safety preserves NOW.")
        explanation = (
            "The patient requires immediate attention based on current acute signals. "
            "The prediction signal is used only for operational planning and does not reduce urgency."
        )
    elif triage_level == "SOON":
        explanation = (
            "The patient should be seen soon based on current acute signals. "
            "Prediction is used to decide whether closer monitoring or operational escalation is needed."
        )
        if prediction_signal.deterioration_risk == "high":
            override_applied = True
            override_reason = "Prediction signal increased monitoring priority within the SOON pathway."
            operational_recommendations = [
                "Urgent monitoring / more frequent recheck",
                "Flag nurse review for possible escalation",
                *operational_recommendations,
            ]
            decision_basis.append("SOON triage kept, but high future-risk signal increased monitoring intensity.")
    else:  # WAIT or unknown
        explanation = (
            "The current acute triage signal does not indicate immediate danger. "
            "Prediction is used to decide whether the patient should be monitored or rechecked more closely."
        )
        if prediction_signal.risk_level == "high" or prediction_signal.deterioration_risk == "high":
            override_applied = True
            override_reason = "Prediction signal flagged future risk; maintain WAIT triage but add recheck and nurse review."
            operational_recommendations = [
                "Flag nurse review",
                "Schedule recheck / repeat vitals",
                *operational_recommendations,
            ]
            decision_basis.append("Prediction conflicts with a low-acuity acute signal, so monitoring is added without forcing NOW.")

    if prediction_signal.predicted_los_hours and prediction_signal.predicted_los_hours >= 36:
        decision_basis.append(f"Predicted LOS ~{prediction_signal.predicted_los_hours:g}h adds bed-planning context.")
        operational_recommendations.append("Flag bed management because predicted LOS is high")

    return (
        list(dict.fromkeys(decision_basis)),
        override_applied,
        override_reason,
        list(dict.fromkeys(operational_recommendations)),
        explanation,
    )


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
        clean_query = input_guard(req.query)
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
    clean_query, guard_triggered, guard_type_200, pii_warnings = pii_redact(clean_query)
    warnings.extend(pii_warnings)
    guard_ms = int((time.perf_counter() - t_guard) * 1000)

    # 3) Retrieve
    t_retrieve = time.perf_counter()
    hits, method_used = pipeline.retrieve_with_method(clean_query, k=req.k, method=req.method)
    retrieve_ms = int((time.perf_counter() - t_retrieve) * 1000)

    # 4) Rule-based classification
    rule_tier, red_flags = classify_rule(clean_query)

    # 5) Retrieval-based classification
    rag_tier, rag_conf, rag_votes = classify_rag(hits)

    triage_level = _triage_level_from_esi(rule_tier if red_flags else rag_tier if rag_tier is not None else rule_tier)

    # 6) Prediction signal
    prediction_signal = prediction_signal_node(
        clean_query,
        hits,
        er_state=req.er_state,
        triage_level=triage_level,
    )

    # 7) Generate
    t_generate = time.perf_counter()
    gen = generate_grounded_answer(clean_query, hits)
    generate_ms = int((time.perf_counter() - t_generate) * 1000)
    warnings.extend(gen.get("warnings", []))

    # 8) Output guardrails + fuse
    final_warnings, citations, esi_final, esi_conf, disagreement, hard_failures, rag_votes = output_guard_and_fuse(
        answer=gen["answer"],
        hits=hits,
        rule_tier=rule_tier,
        red_flags=red_flags,
        rag_tier=rag_tier,
        rag_conf=rag_conf,
        rag_votes=rag_votes,
    )
    warnings.extend(final_warnings)
    if hard_failures:
        detail = {"error": "output_guard", "hard_failures": hard_failures}
        if trace_id:
            detail["trace_call_id"] = trace_id
        raise HTTPException(status_code=422, detail=detail)

    triage_level = _triage_level_from_esi(esi_final)
    decision_basis, override_applied, override_reason, operational_recommendations, explanation_for_human = orchestration_override_rules(
        triage_level=triage_level,
        prediction_signal=prediction_signal,
        red_flags=red_flags,
    )

    _orchestration_state = {
        "case": {"query": req.query, "er_state": req.er_state.model_dump() if req.er_state else None},
        "facts": {"method_used": method_used, "retrieved_count": len(hits)},
        "triage_classification": {
            "rule": rule_tier,
            "rag": rag_tier,
            "final": esi_final,
            "triage_level": triage_level,
        },
        "prediction_signal": prediction_signal.model_dump(),
        "final_decision": {
            "override_applied": override_applied,
            "override_reason": override_reason,
            "operational_recommendations": operational_recommendations,
        },
        "explanation": explanation_for_human,
    }
    if disagreement:
        warnings.append(
            "prediction/triage review note: future-risk signal is visible in the explanation and did not hide the acute triage priority."
        )
    if override_applied and override_reason:
        warnings.append(f"override_applied: {override_reason}")

    enriched_citations = enrich_citations(citations[: req.k], hits)
    grouped = build_grouped_evidence(hits)

    return AskResponse(
        query=req.query,
        answer=gen["answer"],
        citations=enriched_citations,
        method_used=method_used,
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
        triage_level=triage_level,
        prediction_signal=prediction_signal,
        decision_basis=decision_basis,
        override_applied=override_applied,
        override_reason=override_reason,
        operational_recommendations=operational_recommendations,
        explanation_for_human=explanation_for_human,
        guard_ms=guard_ms,
        retrieve_ms=retrieve_ms,
        generate_ms=generate_ms,
        guard_triggered=guard_triggered,
        guard_type=guard_type_200,
        trace_call_id=trace_id,
        grouped_evidence=grouped,
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
