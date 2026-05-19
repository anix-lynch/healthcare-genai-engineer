"""Input validator — runs before the LLM/retrieval sees the query.

Three layers, all fast + deterministic:
    1. Sanitize: strip control chars + normalize whitespace
    2. Prompt-injection scan: regex for known attack patterns
    3. Token-limit: cap input length (chars-based, no tokenizer dep)

If any layer fails → raise InputGuardError. Caller maps to HTTP 400.
"""
from __future__ import annotations
import re

try:
    import weave
    _weave_op = weave.op
except Exception:
    def _weave_op(fn):  # type: ignore[misc]
        return fn

MAX_INPUT_CHARS = 4000

# Conservative prompt-injection patterns. Not exhaustive — would expand with
# Llama Guard or a small classifier in production. Honest baseline.
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(your|the)\s+(system\s+)?prompt", re.I),
    re.compile(r"forget\s+(everything|your\s+role|your\s+instructions)", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(different|new)\s+", re.I),
    re.compile(r"(reveal|print|show|leak)\s+(your\s+)?(system\s+)?(prompt|instructions)", re.I),
]

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


class InputGuardError(ValueError):
    """Raised when input fails any guardrail check."""


def _sanitize(text: str) -> str:
    """Strip control chars, normalize whitespace, trim."""
    text = CONTROL_CHARS.sub("", text)
    return " ".join(text.split())


def _scan_injection(text: str) -> list[str]:
    """Return list of injection patterns that matched."""
    hits: list[str] = []
    for rx in INJECTION_PATTERNS:
        if rx.search(text):
            hits.append(rx.pattern[:60])
    return hits


@_weave_op
def validate_input(text: str, *, max_chars: int = MAX_INPUT_CHARS) -> str:
    """
    Run all input checks. Return sanitized text or raise InputGuardError.

    Args:
        text: raw user input.
        max_chars: input length cap.

    Returns:
        sanitized text (safe to feed to retrieval/LLM).

    Raises:
        InputGuardError on any guard failure.
    """
    if not text or not text.strip():
        raise InputGuardError("empty input")
    sanitized = _sanitize(text)
    if len(sanitized) > max_chars:
        raise InputGuardError(
            f"input exceeds max {max_chars} chars (got {len(sanitized)})"
        )
    injection_hits = _scan_injection(sanitized)
    if injection_hits:
        raise InputGuardError(
            f"prompt-injection patterns detected: {injection_hits[:3]}"
        )
    return sanitized
