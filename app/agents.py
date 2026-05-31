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
    RetryPolicy,
    TriageLevel,
)


def _handoff_key(agent_id: str, level: str, signal: PredictionSignal) -> str:
    return (
        f"{agent_id}:{level}:risk-{signal.risk_level}:"
        f"det-{signal.deterioration_risk}:bed-{signal.bed_pressure_risk}"
    )


def _triage_retry_policy(level: str, red_flags: list[str]) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=1,
        backoff_seconds=[],
        retry_on=[],
        stop_conditions=[
            "triage_level_confirmed",
            "safety_floor_applied" if red_flags or level == "NOW" else "no_red_flags_found",
        ],
        escalation="clinician_review_if_NOW_or_red_flags_fire",
    )


def _bed_ops_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=2,
        backoff_seconds=[30, 120],
        retry_on=["capacity_api_timeout", "stale_er_state"],
        stop_conditions=["bed_status_confirmed", "retry_budget_exhausted"],
        escalation="charge_nurse_review",
    )


def _followup_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_attempts=2,
        backoff_seconds=[60, 300],
        retry_on=["handoff_queue_timeout", "patient_message_pending"],
        stop_conditions=["recheck_scheduled", "nurse_review_requested", "retry_budget_exhausted"],
        escalation="nurse_review_queue",
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
            handoff_key=_handoff_key("er_triage", level, prediction_signal),
            label="ER Triage Agent",
            role="Owns acute urgency and safety-floor routing.",
            trigger=f"Fused ESI produced {level}.",
            receives=["query", "retrieved_evidence", "ESI votes", "red_flags"],
            actions=[
                f"Route case as {level}",
                "Keep safety-floor override if red flags fire",
            ],
            retry_policy=_triage_retry_policy(level, red_flags),
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
                handoff_key=_handoff_key("bed_ops", level, prediction_signal),
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
                retry_policy=_bed_ops_retry_policy(),
                output=bed_decision,
                executed=True,
            )
        )

    needs_followup = level in {"SOON", "WAIT"} or prediction_signal.deterioration_risk == "high"
    if needs_followup:
        handoffs.append(
            AgentHandoff(
                agent_id="care_followup",
                handoff_key=_handoff_key("care_followup", level, prediction_signal),
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
                retry_policy=_followup_retry_policy(),
            )
        )

    if len(handoffs) == 1:
        handoffs.append(
            AgentHandoff(
                agent_id="care_followup",
                handoff_key=_handoff_key("care_followup", level, prediction_signal),
                label="Care Follow-up Agent",
                role="Default second node so every case has an action handoff.",
                trigger="No downstream risk trigger fired, but the user still needs a next-step owner.",
                receives=[*receives_common, "triage_level"],
                actions=operational_recommendations[:2] or ["Document answer and monitor for changed symptoms"],
                retry_policy=_followup_retry_policy(),
            )
        )

    # Contract guard: a collaboration plan is a bounded graph, not a free-running
    # swarm. If a future edit creates duplicate nodes, fail before serving a loop.
    keys = [h.handoff_key for h in handoffs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate handoff_key in agent collaboration plan")
    if len(handoffs) > 3:
        raise ValueError("agent collaboration plan exceeds max_graph_steps=3")

    summary = " -> ".join(h.label for h in handoffs)
    return AgentCollaboration(
        primary_agent="er_triage",
        handoffs=handoffs,
        summary=summary,
    )
