"""Output validator — runs after generation, before returning to caller.

Three checks:
    1. citations_valid:    every cited source_id exists in the retrieved set
    2. min_length:         empty / 1-word answers are blocked
    3. forbidden_actions:  no "call <phone>" / "email <addr>" / "go to <url>"
                            patterns that suggest the LLM is taking action
                            on behalf of the user

Outcome: returns a Verdict dict. Caller decides whether to surface as
warnings (soft fail) or HTTP 422 (hard fail).
"""
from __future__ import annotations
import re
from typing import TypedDict

from generation.citations import extract_citations

try:
    import weave
    _weave_op = weave.op
except Exception:
    def _weave_op(fn):  # type: ignore[misc]
        return fn


class Verdict(TypedDict):
    passed: bool
    warnings: list[str]
    hard_failures: list[str]


class OutputGuardError(ValueError):
    """Raised when output fails a HARD check (caller maps to 422)."""


# Pattern is intentionally loose — false-positive friendly, since a hospital
# AI must NOT instruct callers to phone, email, or visit URLs autonomously.
FORBIDDEN_ACTION = re.compile(
    r"\b(call|phone|email|visit|go\s+to|click)\s+(\d|http|www|@)", re.I
)


@_weave_op
def validate_output(
    answer: str,
    *,
    valid_source_ids: set[str],
    min_length: int = 10,
) -> Verdict:
    """
    Soft-check the generated answer; never raises. Caller inspects Verdict.

    Returns:
        Verdict with passed=False if anything fired; warnings + hard_failures
        list the specific issues.
    """
    warnings: list[str] = []
    hard_failures: list[str] = []

    if not answer or len(answer.strip()) < min_length:
        hard_failures.append(f"answer too short (< {min_length} chars)")

    _, dropped = extract_citations(answer, valid_source_ids)
    if dropped:
        warnings.append(
            f"dropped {len(dropped)} unresolved citations: {dropped[:3]}"
        )

    if FORBIDDEN_ACTION.search(answer):
        hard_failures.append(
            "answer contains forbidden action verb + target "
            "(call/email/visit + number/URL)"
        )

    return {
        "passed": not (warnings or hard_failures),
        "warnings": warnings,
        "hard_failures": hard_failures,
    }
