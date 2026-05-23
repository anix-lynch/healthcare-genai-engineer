from __future__ import annotations

from app.routers.ask import orchestration_override_rules
from app.schemas import PredictionSignal


def _signal(
    *,
    risk: str,
    los: float | None,
    deterioration: str,
    bed_pressure: str,
):
    return PredictionSignal(
        risk_level=risk,
        predicted_los_hours=los,
        deterioration_risk=deterioration,
        bed_pressure_risk=bed_pressure,
        confidence=0.78,
        reasons=["test signal"],
        recommended_operational_actions=["Baseline operational action"],
    )


def test_case_a_now_triage_low_prediction_still_now():
    decision_basis, override_applied, override_reason, ops, explanation = orchestration_override_rules(
        triage_level="NOW",
        prediction_signal=_signal(risk="low", los=4, deterioration="low", bed_pressure="low"),
        red_flags=["resuscitation_keyword:stroke"],
    )
    assert override_applied is True
    assert override_reason is not None
    assert "Immediate clinician review" in ops
    assert "does not reduce urgency" in explanation
    assert any("Acute triage level: NOW" in item for item in decision_basis)


def test_case_b_now_triage_high_los_adds_bed_planning():
    decision_basis, _, _, ops, _ = orchestration_override_rules(
        triage_level="NOW",
        prediction_signal=_signal(risk="high", los=72, deterioration="high", bed_pressure="medium"),
        red_flags=["resuscitation_keyword:stroke"],
    )
    assert any("Predicted LOS ~72h adds bed-planning context." == item for item in decision_basis)
    assert any("bed" in item.lower() for item in ops)


def test_case_c_soon_triage_high_deterioration_adds_monitoring():
    _, override_applied, override_reason, ops, explanation = orchestration_override_rules(
        triage_level="SOON",
        prediction_signal=_signal(risk="high", los=24, deterioration="high", bed_pressure="medium"),
        red_flags=[],
    )
    assert override_applied is True
    assert "SOON pathway" in (override_reason or "")
    assert any("monitor" in item.lower() or "nurse review" in item.lower() for item in ops)
    assert "seen soon" in explanation


def test_case_d_wait_triage_high_future_risk_flags_recheck():
    _, override_applied, override_reason, ops, explanation = orchestration_override_rules(
        triage_level="WAIT",
        prediction_signal=_signal(risk="high", los=48, deterioration="high", bed_pressure="high"),
        red_flags=[],
    )
    assert override_applied is True
    assert "maintain WAIT triage" in (override_reason or "")
    assert any("recheck" in item.lower() for item in ops)
    assert "does not indicate immediate danger" in explanation


def test_case_e_conflict_is_surfaced():
    decision_basis, override_applied, override_reason, _, _ = orchestration_override_rules(
        triage_level="NOW",
        prediction_signal=_signal(risk="low", los=6, deterioration="low", bed_pressure="low"),
        red_flags=["high_risk_keyword:chest pain"],
    )
    assert override_applied is True
    assert "cannot be downgraded by prediction" in (override_reason or "")
    assert any("Conflict surfaced" in item for item in decision_basis)
