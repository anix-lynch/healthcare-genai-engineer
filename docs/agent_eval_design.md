# Bed Ops Agent — Eval Design Note

> One page on **how the action-agent is scored, why the floors are where they are, and
> what the misses mean.** Pairs with `evaluation/agent_scenarios.json` (the labelled set)
> and `evaluation/agent_eval.py` (the runner). Read this before trusting the green number.

## What this eval is — and is not

The Bed Ops agent (`app/bed_ops_agent.py` + `app/agents.py`) is **deterministic by design**.
A clinical capacity protocol is not a place for LLM nondeterminism — the same ER state must
always produce the same disposition, and the output must change when the inputs change.

So this is a **clinical-protocol-conformance eval, not an LLM-reasoning eval.** We assert the
coded capacity protocol against **independent clinical labels** across 50 ER states. A green
number means *the agent actually executed and its computed disposition matched an independently
labelled clinical expectation* — not that a router matched its own labels.

## Labelling methodology (why this is not circular)

The trap: if `expected_disposition` is copied from the agent's output, the eval scores 100% by
construction and proves nothing. We avoid it explicitly:

- Each scenario's `expected_disposition` and `expect_bed_ops_triggered` are labelled from the
  **clinical rubric below**, written from protocol intent — *not* by running the agent.
- Where the coded thresholds disagree with clinical intent at a boundary, the scenario is left
  as a **real miss** and classified — it is **not** rigged to agree, and the production logic is
  **not** patched to chase a green number.
- **A perfect score would be the failure signal, not the success signal.** The committed
  baseline is `decision 0.94 / handoff 0.94` — it carries 6 documented boundary misses. If a
  future edit makes it `1.00 / 1.00`, suspect circular labelling and re-check the boundaries.

### Clinical labelling rubric (independent ground truth)

**Disposition**
- **Acute (`NOW`)**: a free bed → `assign_bed` (acute gets the bed, even at high occupancy).
  No free bed → `divert` when the ED is in extreme saturation with a backed-up queue, else
  `board_ed` (hold in ED, escalate placement — do not divert while the queue is manageable).
- **Long predicted stay (≥36h)** with **genuinely scarce** capacity → `hold_observation`
  (manage as a boarder). "Scarce" = few beds **and** high occupancy — two free beds on a
  half-empty floor is *not* scarce.
- **Otherwise**: a bed is free and inpatient flow is indicated → `assign_bed`; no bed
  indicated → `discharge_plan` (route to follow-up).

**Handoff (should Bed Ops engage?)** — engage only when a **capacity action** is actually
needed: acute (`NOW`), long stay (≥36h), genuine high bed-pressure, or any admit/hold that
needs a bed confirmed. **Do not** engage when the patient is simply discharged — even under
medium bed-pressure, a discharge needs no bed operation.

## Floors — and why they are not 1.0

| Metric | Floor | Rationale |
|--------|-------|-----------|
| `task_completion_rate` | 1.00 | Mechanical: a valid disposition must always be emitted. A drop is a real bug. |
| `tool_call_success` | 1.00 | Mechanical: the agent must read live ER state into `inputs_used`. A drop is a real bug. |
| `decision_correctness` | 0.90 | The labelled set intentionally includes boundary cases where independent clinical judgement disagrees with the coded constants. Those misses are signal — accepted at the floor, not patched. |
| `handoff_correctness` | 0.85 | The planner over-triggers Bed Ops on medium bed-pressure even for a discharge. Documented, accepted, not hidden behind a green number. |

**Regression vs capability split** — two different gates:
- **Capability floors** (above, in `agent_eval.py`): absolute bar the protocol must always clear.
- **Regression gate** (`evaluation/regression_gate.py`, tolerance 0.05): blocks a PR that drops
  any metric >5pp vs `evaluation/agent_baseline.json`. This catches *silent erosion* — a change
  that quietly turns a passing case into a miss — even while the absolute floor still holds.

## Failure taxonomy (the 6 classified misses)

Run-truth: `outputs/agent_eval_summary.json → failure_taxonomy`. Three patterns:

### 1. Divert under-fires in extreme saturation — *decision*, 2 cases
`MISS-divert-01`, `MISS-divert-02`. At 99–100% occupancy with **zero** free beds and queue 7,
clinical protocol diverts; the coded rule gates divert **solely on `queue ≥ 8`**, so it returns
`board_ed`. The occupancy extremity is ignored once the queue is one short of the threshold.
- **Status:** accepted at floor. **Candidate fix (not applied):** add an
  `occupancy ≥ 98 and beds == 0` divert path independent of the queue constant.

### 2. Hold over-fires when capacity is actually fine — *decision*, 1 case
`MISS-hold-01`. A 40h stay with **2 free beds at 55% occupancy** is labelled `assign_bed`
(admit — the floor is half empty), but the code flags `beds ≤ 2` as scarce **regardless of
occupancy** and returns `hold_observation`. Over-conservative boarding.
- **Status:** accepted at floor. **Candidate fix (not applied):** make the scarcity test
  occupancy-aware (`beds ≤ 2 and occupancy ≥ ~80`) instead of bed-count alone.

### 3. Bed Ops handoff over-triggers on medium pressure for discharges — *handoff*, 3 cases
`MISS-handoff-01/02/03`. A short-stay patient routed to `discharge_plan` needs no capacity
action, but `needs_bed_ops` fires on **any** medium/high bed-pressure regardless of disposition,
producing a spurious handoff. Three classified instances confirm it is a systematic rule
weakness, not noise.
- **Status:** accepted at floor (0.85). **Candidate fix (not applied):** gate the medium-pressure
  trigger on the disposition — skip Bed Ops when the computed disposition is `discharge_plan`.

> **Why accept rather than fix now:** the SLA requires failures *published and classified*, not
> eliminated. Patching production logic mid-eval to hit 100% is both scope creep and a subtle rig.
> These three candidate fixes are real backlog items, tracked here, to be made as deliberate
> protocol changes with their own review — not as a number-chasing edit inside the eval harness.
