"""Small multi-agent handoff planner for the ER workflow.

This is deliberately not a free-running agent swarm. The API returns a
deterministic collaboration graph that downstream agent runtimes can execute:
triage owns acute safety, then Bed Ops or Care Follow-up receives the case
when the decision needs action outside the first clinical answer.
"""
from __future__ import annotations

from app.bed_ops_agent import decide_bed_disposition
from app.schemas import (
    AgentCollaboration,
    AgentHandoff,
    ERState,
    PredictionSignal,
    TriageLevel,
)


def plan_agent_collaboration(
    *,
    triage_level: TriageLevel | None,
    prediction_signal: PredictionSignal,
    red_flags: list[str],
    operational_recommendations: list[str],
    er_state: ERState | None = None,
) -> AgentCollaboration:
    """Return the visible two-plus-node action-agent plan for this case."""
    level = triage_level or "WAIT"
    receives_common = ["query", "retrieved_evidence", "citations", "prediction_signal"]

    handoffs: list[AgentHandoff] = [
        AgentHandoff(
            agent_id="er_triage",
            label="ER Triage Agent",
            role="Owns acute urgency and safety-floor routing.",
            trigger=f"Fused ESI produced {level}.",
            receives=["query", "retrieved_evidence", "ESI votes", "red_flags"],
            actions=[
                f"Route case as {level}",
                "Keep safety-floor override if red flags fire",
            ],
        )
    ]

    needs_bed_ops = (
        prediction_signal.bed_pressure_risk in {"medium", "high"}
        or (prediction_signal.predicted_los_hours or 0) >= 36
        or level == "NOW"
    )
    if needs_bed_ops:
        # Bed Ops actually EXECUTES: it computes a disposition from live ER state
        # + predicted LOS, so its output changes with the inputs (not a label).
        bed_decision = decide_bed_disposition(
            er_state=er_state,
            triage_level=triage_level,
            predicted_los_hours=prediction_signal.predicted_los_hours,
            bed_pressure_risk=prediction_signal.bed_pressure_risk,
        )
        handoffs.append(
            AgentHandoff(
                agent_id="bed_ops",
                label="Bed Ops Agent",
                role="Turns triage and LOS risk into capacity actions.",
                trigger=(
                    "NOW route, high bed pressure, or long predicted LOS."
                    if level == "NOW"
                    else "Bed pressure or predicted LOS needs operational planning."
                ),
                receives=[*receives_common, "er_state", "triage_level", "predicted_los_hours", "bed_pressure_risk"],
                actions=[
                    f"Disposition: {bed_decision['disposition']}",
                    bed_decision["reason"],
                ],
                output=bed_decision,
                executed=True,
            )
        )

    needs_followup = level in {"SOON", "WAIT"} or prediction_signal.deterioration_risk == "high"
    if needs_followup:
        handoffs.append(
            AgentHandoff(
                agent_id="care_followup",
                label="Care Follow-up Agent",
                role="Keeps non-NOW patients from disappearing after the answer.",
                trigger=(
                    "Patient is not NOW, or prediction says future risk is high."
                ),
                receives=[*receives_common, "triage_level", "recommended_operational_actions"],
                actions=[
                    "Schedule recheck or repeat vitals",
                    "Give patient-facing follow-up instructions for staff review",
                    "Escalate back to ER Triage if risk worsens",
                ],
            )
        )

    if len(handoffs) == 1:
        handoffs.append(
            AgentHandoff(
                agent_id="care_followup",
                label="Care Follow-up Agent",
                role="Default second node so every case has an action handoff.",
                trigger="No downstream risk trigger fired, but the user still needs a next-step owner.",
                receives=[*receives_common, "triage_level"],
                actions=operational_recommendations[:2] or ["Document answer and monitor for changed symptoms"],
            )
        )

    summary = " -> ".join(h.label for h in handoffs)
    return AgentCollaboration(
        primary_agent="er_triage",
        handoffs=handoffs,
        summary=summary,
    )
