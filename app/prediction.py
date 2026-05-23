"""Lightweight prediction signal layer for operational planning.

Prediction is intentionally heuristic for now.
It is a forward-looking signal, not an acute-triage replacement.
"""
from __future__ import annotations

import re
from typing import Any

from app.schemas import ERState, PredictionSignal, RiskLevel

try:
    import weave
    _weave_op = weave.op
except Exception:
    def _weave_op(fn):  # type: ignore[misc]
        return fn


_HIGH_DETERIORATION = (
    "chest pain", "shortness of breath", "sob", "altered mental", "ams",
    "stroke", "stemi", "nstemi", "anaphylax", "respiratory failure",
    "sepsis", "septic", "dka", "overdose", "suicid", "code blue",
)
_MEDIUM_DETERIORATION = (
    "vomiting", "fever", "abdominal pain", "headache", "back pain",
    "generalized weakness", "dehydration",
)
_LONG_LOS_HINTS = (
    "cancer", "chemo", "metastatic", "immunosuppressed", "abnormal",
    "inconclusive", "admit", "admission", "broad-spectrum antibiotics",
)


def _risk_rank(level: RiskLevel) -> int:
    return {"low": 0, "medium": 1, "high": 2}[level]


def _max_risk(*levels: RiskLevel) -> RiskLevel:
    return max(levels, key=_risk_rank)


def _bucket(value: int | float | None, medium_at: float, high_at: float) -> RiskLevel:
    if value is None:
        return "low"
    if value >= high_at:
        return "high"
    if value >= medium_at:
        return "medium"
    return "low"


def _to_text(case: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("query", "chief_complaint", "hpi", "physician_note"):
        val = case.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.lower())
    for hit in case.get("hits", []) or []:
        snippet = hit.get("snippet")
        if isinstance(snippet, str) and snippet.strip():
            parts.append(snippet.lower())
    return " | ".join(parts)


def _estimate_los(text: str, deterioration: RiskLevel, er_state: ERState | None) -> float | None:
    los = 6.0
    if deterioration == "medium":
        los = 12.0
    if deterioration == "high":
        los = 24.0
    if any(term in text for term in _LONG_LOS_HINTS):
        los += 24.0
    if re.search(r"\b(7[5-9]|8\d|9\d)yo\b", text):
        los += 12.0
    if er_state and er_state.available_beds is not None and er_state.available_beds <= 3:
        los += 6.0
    return round(los, 1)


@_weave_op
def get_prediction_signal(case: dict[str, Any], er_state: ERState | None = None) -> PredictionSignal:
    """Return a lightweight future-risk signal for orchestration use."""
    text = _to_text(case)
    reasons: list[str] = []
    actions: list[str] = []

    deterioration: RiskLevel = "low"
    if any(term in text for term in _HIGH_DETERIORATION):
        deterioration = "high"
        reasons.append("acute-sounding symptoms or precedent patterns suggest elevated deterioration risk")
    elif any(term in text for term in _MEDIUM_DETERIORATION):
        deterioration = "medium"
        reasons.append("moderate symptom burden suggests recheck/monitoring risk")

    los_hours = _estimate_los(text, deterioration, er_state)
    if los_hours is not None and los_hours >= 36:
        reasons.append("predicted length of stay is high, which increases operational load")
    elif los_hours is not None and los_hours >= 12:
        reasons.append("predicted length of stay is moderate, which may affect flow planning")

    bed_pressure = "low"
    if er_state:
        occupancy_risk = _bucket(er_state.occupancy_pct, 85, 95)
        queue_risk = _bucket(er_state.queue_length, 10, 20)
        wait_risk = _bucket(er_state.avg_wait_minutes, 60, 120)
        bed_count_risk = "high" if (er_state.available_beds is not None and er_state.available_beds <= 3) else \
            "medium" if (er_state.available_beds is not None and er_state.available_beds <= 8) else "low"
        bed_pressure = _max_risk(occupancy_risk, queue_risk, wait_risk, bed_count_risk)
        if bed_pressure == "high":
            reasons.append("current ER state indicates high bed/queue pressure")
        elif bed_pressure == "medium":
            reasons.append("current ER state indicates moderate bed/queue pressure")
    elif los_hours is not None and los_hours >= 36:
        bed_pressure = "medium"
        reasons.append("long predicted stay creates likely downstream bed pressure even without live ER state")

    risk_level = _max_risk(deterioration, bed_pressure, "high" if (los_hours or 0) >= 48 else "medium" if (los_hours or 0) >= 24 else "low")

    if deterioration == "high":
        actions.append("Increase monitoring frequency")
        actions.append("Flag nurse review for potential escalation")
    elif deterioration == "medium":
        actions.append("Schedule reassessment / vital recheck")

    if los_hours is not None and los_hours >= 36:
        actions.append("Flag bed management because predicted LOS is high")
    elif los_hours is not None and los_hours >= 12:
        actions.append("Prepare for moderate care coordination needs")

    if bed_pressure == "high":
        actions.append("Notify charge nurse about bed pressure risk")
    elif bed_pressure == "medium":
        actions.append("Monitor queue and bed availability")

    unique_actions = list(dict.fromkeys(actions))
    confidence = min(0.9, round(0.55 + 0.08 * len(reasons), 2))
    if not reasons:
        reasons.append("no strong future-risk indicators detected from current text and retrieved precedents")

    return PredictionSignal(
        risk_level=risk_level,
        predicted_los_hours=los_hours,
        deterioration_risk=deterioration,
        bed_pressure_risk=bed_pressure,
        confidence=confidence,
        reasons=reasons,
        recommended_operational_actions=unique_actions,
    )
