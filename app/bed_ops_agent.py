"""Bed Ops Agent — a downstream action node that actually executes.

Unlike a routing label, this computes a capacity disposition FROM its inputs
(live ER state + predicted length-of-stay + triage level). The output changes
when the inputs change — that is what makes it a real agent action rather than
a hardcoded string. Deterministic on purpose (a clinical capacity protocol is
not a place for LLM nondeterminism); the "tool" it uses is the live ER state.

Decision space (disposition):
  assign_bed        — capacity exists, give the patient a bed
  board_ed          — acute patient, no bed yet → hold in ED, escalate placement
  hold_observation  — long predicted stay + scarce beds → manage as a boarder
  divert            — saturated + queue backed up → ambulance divert
  discharge_plan    — no inpatient bed indicated, route to follow-up
"""
from __future__ import annotations

from typing import Any

from app.schemas import ERState, RiskLevel, TriageLevel

# occupancy at/above this with no free beds = saturated
SATURATED_PCT = 95.0
# a queue this long while saturated is what tips an acute case to divert
DIVERT_QUEUE = 8
# predicted stay (hours) that makes a patient a capacity problem, not a quick visit
LONG_STAY_HOURS = 36.0


def decide_bed_disposition(
    *,
    er_state: ERState | None,
    triage_level: TriageLevel | None,
    predicted_los_hours: float | None,
    bed_pressure_risk: RiskLevel,
) -> dict[str, Any]:
    """Compute a bed disposition from live capacity + clinical inputs.

    Returns a structured result whose `disposition` and `reason` vary with the
    inputs. `inputs_used` records exactly what the decision read (audit trail).
    """
    beds = er_state.available_beds if er_state and er_state.available_beds is not None else None
    occ = er_state.occupancy_pct if er_state and er_state.occupancy_pct is not None else None
    queue = er_state.queue_length if er_state and er_state.queue_length is not None else None
    los = predicted_los_hours or 0.0
    level = triage_level or "WAIT"

    saturated = (beds is not None and beds <= 0) or (occ is not None and occ >= SATURATED_PCT)
    has_bed = beds is not None and beds >= 1

    if level == "NOW":
        if has_bed:
            disposition = "assign_bed"
            reason = f"Acute (NOW) patient and {beds} bed(s) free — assign immediately."
        elif saturated and (queue or 0) >= DIVERT_QUEUE:
            disposition = "divert"
            reason = (
                f"Acute patient but ED saturated (occupancy {occ}%, {beds} beds, "
                f"queue {queue}) — request ambulance divert and escalate placement."
            )
        else:
            disposition = "board_ed"
            reason = (
                "Acute patient with no free bed — board in ED and escalate bed "
                "placement; do not divert while queue is manageable."
            )
    elif los >= LONG_STAY_HOURS and (beds is None or beds <= 2):
        disposition = "hold_observation"
        reason = (
            f"Predicted stay {los:.0f}h with scarce capacity ({beds} beds) — "
            "manage as an observation/boarding case and pre-plan inpatient flow."
        )
    elif has_bed:
        disposition = "assign_bed"
        reason = f"Non-acute but inpatient flow indicated; {beds} bed(s) available — plan admit."
    else:
        disposition = "discharge_plan"
        reason = "No inpatient bed indicated — route to Care Follow-up with a discharge plan."

    # high bed-pressure risk is advisory: never downgrades an acute decision,
    # but flags capacity follow-up on the softer dispositions.
    capacity_flag = bed_pressure_risk == "high" and disposition in {"hold_observation", "discharge_plan"}

    return {
        "disposition": disposition,
        "reason": reason,
        "bed_assigned": disposition == "assign_bed",
        "capacity_followup_flagged": capacity_flag,
        "inputs_used": {
            "available_beds": beds,
            "occupancy_pct": occ,
            "queue_length": queue,
            "predicted_los_hours": los,
            "triage_level": level,
            "bed_pressure_risk": bed_pressure_risk,
        },
    }
