"""Citation extraction + validation.

Pattern: every source_id in the generated answer must resolve to a
retrieved hit. Hallucinated cites get dropped + flagged.

Source-id grammar today:
    L1-NNNNNN     a Layer 1 encounter id
    P-XXXXXXXXXX  a patient id (short SHA256)
    GUIDE-...     a guideline doc id
"""
from __future__ import annotations
import re

SOURCE_ID_RE = re.compile(r"\b(L1-\d{6}|P-[a-f0-9]{10}|GUIDE-[A-Z0-9-]+)\b")


def extract_citations(text: str, valid_ids: set[str]) -> tuple[list[str], list[str]]:
    """Find source_ids in text, split into (resolved, dropped) per valid_ids.

    Returns:
        (kept, dropped) — both sorted, deduplicated lists of source_id strings.
    """
    found = set(SOURCE_ID_RE.findall(text))
    kept = sorted(found & valid_ids)
    dropped = sorted(found - valid_ids)
    return kept, dropped


def validate_citations(text: str, valid_ids: set[str]) -> bool:
    """True iff every citation in text resolves to a valid_id (or no citations)."""
    _, dropped = extract_citations(text, valid_ids)
    return not dropped
