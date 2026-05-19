"""ESI tier classification — rule-based floor + RAG-KNN refinement + fuse.

Three predictors, all returning ESI tier ∈ {1..5}:

    rule_based_esi(query)
        Port of healthcare-fde's keyword-driven ESI logic. Text-only
        version (no vitals — the GenAI endpoint only takes free-text
        query). Returns the safety-conservative tier — if a keyword
        fires, the rule takes precedence.

    rag_knn_esi(hits)
        Majority vote over the esi_tier_truth labels of the top-K
        retrieved cases, weighted by similarity score. Catches
        paraphrases the rules don't have a keyword for.

    fuse_esi(rule_tier, rule_flags, rag_tier, rag_conf)
        Production pattern: rules are the SAFETY FLOOR (audit-clean,
        deterministic). RAG-KNN refines INSIDE the floor:
          - if rule fired (red flag present)  → trust rule, RAG is
                                                  the second opinion
          - if rule didn't fire               → trust RAG-KNN
        Returns (esi_final, confidence, disagreement_flag).

Why this design:
    Pure ML classifier on a 497-row corpus with 1.8% ESI 1 and 1.0%
    ESI 5 would either over-predict the majority class (ESI 2, 59%
    of corpus) or under-predict the safety-critical tiers.

    Real ER triage systems (Epic SmartList, Cerner FirstNet) layer
    deterministic safety rules (sepsis, pediatric, suicidal, AMS)
    on top of any ML signal. Floor-then-refine matches that.

Source-of-truth for rules: healthcare-fde/workflows/triage_assistant
    (this file ports the text-only branches; SIRS-shape sepsis is
    omitted since the GenAI endpoint has no vitals).
"""
from __future__ import annotations
from collections import Counter
from typing import Iterable

try:
    import weave
    _weave_op = weave.op
except Exception:
    def _weave_op(fn):  # type: ignore[misc]
        return fn


# Default ESI when no signal fires. ESI 4 = less-urgent, not 3, because
# the rule layer is the SAFETY FLOOR — false-negative escalation costs
# less than false-positive de-escalation in clinical practice.
DEFAULT_TIER = 4


@_weave_op
def rule_based_esi(query: str) -> tuple[int, list[str]]:
    """Text-only ESI rule classifier. Returns (tier, red_flags).

    Ports the keyword-driven branches from healthcare-fde:
      ESI 1 resuscitation     cardiac arrest · stroke · STEMI · anaphylax · respiratory failure
      ESI 2 high-risk         chest pain · sepsis · diaphoresis · altered mental · AMS · DKA
      ESI 2 safety floor      suicidal ideation patterns
      ESI 2 pediatric floor   "8-month-old", "infant" patterns (text proxy
                              since we don't have age field here)

    NOT included (out of text-only scope):
      SIRS-shape sepsis       needs vitals (BP/HR/RR/Temp) — see FDE
      Pediatric under-1y      needs structured age field
    """
    text = (query or "").lower()
    red_flags: list[str] = []
    tier = DEFAULT_TIER

    # ESI 1: resuscitation triggers (highest urgency)
    for kw in ("cardiac arrest", "stroke", "stemi", "nstemi",
               "anaphylax", "respiratory failure", "code blue"):
        if kw in text:
            red_flags.append(f"resuscitation_keyword:{kw}")
            tier = 1
            break

    # ESI 2: high-risk patterns
    if tier > 2:
        for kw in ("chest pain", "sepsis", "diaphoresis",
                   "altered mental", "ams", "dka", "diabetic ketoacidosis",
                   "septic shock", "pulmonary embolism", "pe ",
                   "stemi", "nstemi"):
            if kw in text:
                red_flags.append(f"high_risk_keyword:{kw}")
                tier = min(tier, 2)

    # ESI 2 safety floor: suicidal ideation
    for kw in ("suicid", "self harm", "self-harm", "harm self",
               "wants to die", "kill myself", "overdose intent"):
        if kw in text:
            red_flags.append("safety_floor:suicidal_ideation")
            tier = min(tier, 2)
            break

    # ESI 2 pediatric floor (text proxy — no age field available)
    for kw in ("infant", "month-old", "newborn", "neonate"):
        if kw in text:
            red_flags.append("safety_floor:pediatric_under_1y")
            tier = min(tier, 2)
            break

    # ESI 3: moderate-acuity
    if tier > 3:
        for kw in ("vomiting", "headache", "abdominal pain",
                   "fever", "shortness of breath", "sob ",
                   "joint pain", "back pain"):
            if kw in text:
                red_flags.append(f"moderate_keyword:{kw}")
                tier = min(tier, 3)

    return tier, red_flags


def _esi_from_snippet(snippet: str) -> int | None:
    """Extract the ground-truth ESI label from a snippet's "ESI N" tag.

    Snippet shape (per retrieval/retriever._row_to_snippet):
        "... · ESI 3 · flags: none"
    Returns the int tier, or None if not present.
    """
    if not snippet or "ESI " not in snippet:
        return None
    try:
        # find "ESI <n>" — look between "ESI " and the next non-digit
        tail = snippet.split("ESI ", 1)[1]
        digits = ""
        for c in tail:
            if c.isdigit():
                digits += c
            else:
                break
        if digits:
            return int(digits)
    except (ValueError, IndexError):
        pass
    return None


@_weave_op
def rag_knn_esi(hits: Iterable[dict]) -> tuple[int | None, float, dict[int, int]]:
    """KNN-vote ESI from retrieved cases.

    Weighted vote: each top-K hit contributes its esi_tier_truth label
    weighted by its similarity score. Returns (esi_predicted, confidence,
    per-tier vote counts).

    Returns:
        (winning_tier, confidence, votes_count)
        - winning_tier: int 1-5 or None
        - confidence: winning vote share ∈ [0, 1]
        - votes_count: {tier: raw count} — unweighted, for UI display
                       e.g. {2: 4, 3: 1} → "4× ESI 2 · 1× ESI 3"

    Returns (None, 0.0, {}) if no hits or no labels recoverable.
    """
    votes_weight: Counter = Counter()
    votes_count: Counter = Counter()
    weight_total = 0.0
    for h in hits:
        tier = _esi_from_snippet(h.get("snippet", ""))
        if tier is None:
            continue
        w = max(0.01, float(h.get("score", 0.01)))
        votes_weight[tier] += w
        votes_count[tier] += 1
        weight_total += w
    if not votes_weight or weight_total == 0:
        return None, 0.0, {}
    winning_tier, winning_weight = votes_weight.most_common(1)[0]
    confidence = winning_weight / weight_total
    return int(winning_tier), round(confidence, 3), dict(votes_count)


# SAFETY-FLOOR rule prefixes — these are non-negotiable per FDE
# customer-brief.md: "zero AI-initiated down-triage of suicidal ideation,
# pediatric <1y, cardiac arrest, STEMI, etc." Only THESE override RAG.
# Generic high_risk_keyword:chest_pain (etc) is a VOTE, not a floor —
# it might disagree with RAG correctly (e.g., chest pain in an arthritic
# 29yo is ESI 3, not ESI 2 just because the keyword fired).
_SAFETY_FLOOR_PREFIXES = (
    "resuscitation_keyword",   # cardiac arrest, stroke, STEMI, anaphylax, etc.
    "safety_floor",             # suicidal_ideation, pediatric_under_1y
)


@_weave_op
def fuse_esi(
    rule_tier: int,
    rule_flags: list[str],
    rag_tier: int | None,
    rag_conf: float,
) -> tuple[int, float, bool]:
    """Production fusion: SAFETY-FLOOR rules override; otherwise trust RAG.

    Decision tree (revised after eval showed naive fuse hurt accuracy):
      1. If a SAFETY FLOOR rule fired (cardiac arrest / suicidal /
         pediatric) → rule wins regardless of RAG. Audit-clean.
         confidence = 1.0 if RAG agrees, else 0.7 + disagreement flag.
      2. Else, if RAG-KNN has a prediction → trust RAG-KNN.
         confidence = rag_conf
      3. Else → DEFAULT_TIER (4), confidence = 0.3.

    The earlier "any rule_flag fires → trust rule" version overrode
    RAG too eagerly: text-only "chest pain" keyword on every 62yo
    arthritis case tagged ESI 2 even when ground-truth was ESI 3.
    Safety-floors-only fixes this without sacrificing the audit
    contract on the cases that actually matter (suicidal, pediatric,
    resuscitation).

    Returns (esi_final, confidence, disagreement):
       disagreement = True iff a safety-floor rule fired AND rag
       predicted a different tier (review-flag this case).
    """
    safety_floor_fired = any(
        f.startswith(_SAFETY_FLOOR_PREFIXES) for f in rule_flags
    )
    if safety_floor_fired:
        if rag_tier is None or rag_tier == rule_tier:
            return rule_tier, 1.0, False
        return rule_tier, 0.7, True
    # No safety floor — defer to RAG (high-confidence ML lift)
    if rag_tier is not None:
        return rag_tier, rag_conf, False
    return DEFAULT_TIER, 0.3, False
