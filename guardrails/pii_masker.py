"""PII detection + masking — regex baseline.

Detection only on inbound queries (never block — just redact). For real
EHR data we'd swap this for spaCy NER + Comprehend Medical / Healthcare NLP.
This file is the floor: deterministic, no model deps, transparent.

Patterns covered:
    SSN, phone, email, credit-card-like, MRN-shaped, DOB-shaped.

NOT covered (intentional honest scope):
    - patient names without context (would over-redact)
    - addresses (postal pattern is too variable)
    - free-form date+name combos that imply identity
"""
from __future__ import annotations
import re

try:
    import weave
    _weave_op = weave.op
except Exception:
    def _weave_op(fn):  # type: ignore[misc]
        return fn

PII_PATTERNS = {
    "ssn":          re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone":        re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "email":        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "credit_card":  re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "mrn_like":     re.compile(r"\bMRN[#:\s]?\d{6,}\b", re.I),
    "dob_like":     re.compile(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(19|20)\d{2}\b"),
}


@_weave_op
def find_pii(text: str) -> dict[str, list[str]]:
    """Return {pattern_name: [matched_strings]} for everything found.

    Caller can use this for logging without modifying the text.
    """
    if not text:
        return {}
    hits: dict[str, list[str]] = {}
    for label, rx in PII_PATTERNS.items():
        matches = rx.findall(text)
        if matches:
            # findall returns tuples when the regex has groups; coerce to str
            hits[label] = [m if isinstance(m, str) else "/".join(m) for m in matches]
    return hits


@_weave_op
def mask_pii(text: str, *, mask: str = "[REDACTED]") -> tuple[str, dict[str, int]]:
    """Replace matched PII spans with the mask token.

    Returns:
        (masked_text, {pattern_name: count_redacted})
    """
    counts: dict[str, int] = {}
    out = text or ""
    for label, rx in PII_PATTERNS.items():
        n = 0
        def _sub(_m, _l=label):
            nonlocal n
            n += 1
            return mask
        out = rx.sub(_sub, out)
        if n:
            counts[label] = n
    return out, counts
