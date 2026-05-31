from __future__ import annotations

from app.agents import plan_agent_collaboration
from app.bed_ops_agent import decide_bed_disposition
from app.schemas import ERState, PredictionSignal


def _signal(
    *,
    risk: str = "low",
    los: float | None = 12,
    deterioration: str = "low",
    bed_pressure: str = "low",
) -> PredictionSignal:
    return PredictionSignal(
        risk_level=risk,
        predicted_los_hours=los,
        deterioration_risk=deterioration,
        bed_pressure_risk=bed_pressure,
        confidence=0.74,
        reasons=["test"],
        recommended_operational_actions=["test action"],
    )


def test_now_routes_to_triage_and_bed_ops():
    plan = plan_agent_collaboration(
        triage_level="NOW",
        prediction_signal=_signal(risk="high", los=24),
        red_flags=["resuscitation_keyword:stroke"],
        operational_recommendations=["Immediate clinician review"],
    )
    assert plan.primary_agent == "er_triage"
    assert [h.agent_id for h in plan.handoffs][:2] == ["er_triage", "bed_ops"]
    assert "Bed Ops Agent" in plan.summary


def test_wait_high_future_risk_routes_to_followup():
    plan = plan_agent_collaboration(
        triage_level="WAIT",
        prediction_signal=_signal(risk="high", los=48, deterioration="high", bed_pressure="high"),
        red_flags=[],
        operational_recommendations=["Schedule recheck / repeat vitals"],
    )
    ids = {h.agent_id for h in plan.handoffs}
    assert {"er_triage", "bed_ops", "care_followup"}.issubset(ids)
    followup = next(h for h in plan.handoffs if h.agent_id == "care_followup")
    assert any("recheck" in action.lower() for action in followup.actions)


def test_every_plan_has_at_least_two_nodes():
    plan = plan_agent_collaboration(
        triage_level="SOON",
        prediction_signal=_signal(),
        red_flags=[],
        operational_recommendations=[],
    )
    assert len(plan.handoffs) >= 2


# ── Bed Ops EXECUTES — output changes with inputs (real agent, not a label) ──

def test_bed_ops_output_varies_with_capacity():
    free = decide_bed_disposition(
        er_state=ERState(available_beds=3, occupancy_pct=70, queue_length=2),
        triage_level="NOW", predicted_los_hours=6, bed_pressure_risk="low")
    saturated = decide_bed_disposition(
        er_state=ERState(available_beds=0, occupancy_pct=98, queue_length=12),
        triage_level="NOW", predicted_los_hours=6, bed_pressure_risk="high")
    assert free["disposition"] == "assign_bed"
    assert saturated["disposition"] == "divert"
    assert free["disposition"] != saturated["disposition"]


def test_bed_ops_long_stay_scarce_holds_for_observation():
    out = decide_bed_disposition(
        er_state=ERState(available_beds=1, occupancy_pct=80, queue_length=3),
        triage_level="WAIT", predicted_los_hours=48, bed_pressure_risk="high")
    assert out["disposition"] == "hold_observation"


def test_bed_ops_acute_full_calm_queue_boards_not_diverts():
    """Safety: full ED with a calm queue boards the acute patient, never diverts."""
    out = decide_bed_disposition(
        er_state=ERState(available_beds=0, occupancy_pct=96, queue_length=2),
        triage_level="NOW", predicted_los_hours=6, bed_pressure_risk="medium")
    assert out["disposition"] == "board_ed"


def test_bed_ops_decision_reads_er_state():
    out = decide_bed_disposition(
        er_state=ERState(available_beds=2, occupancy_pct=60, queue_length=1),
        triage_level="SOON", predicted_los_hours=12, bed_pressure_risk="low")
    assert out["inputs_used"]["available_beds"] == 2
    assert out["reason"]


def test_planner_attaches_executed_bed_ops_output():
    plan = plan_agent_collaboration(
        triage_level="NOW", prediction_signal=_signal(risk="high", los=10, bed_pressure="high"),
        red_flags=[], operational_recommendations=[],
        er_state=ERState(available_beds=1, occupancy_pct=92, queue_length=6))
    bed_ops = [h for h in plan.handoffs if h.agent_id == "bed_ops"]
    assert bed_ops and bed_ops[0].executed
    assert bed_ops[0].output and bed_ops[0].output["disposition"] == "assign_bed"
