"""Grounded answer generation.

Two paths:
    1. TEMPLATE (default, feature-flag off) — deterministic, auditable,
       zero LLM cost. Always cites a real source_id.
    2. LLM-ENHANCED (USE_LLM=true) — sends grounded prompt to an LLM
       provider (Anthropic or OpenAI). Falls back to TEMPLATE if the
       provider errors or the SDK isn't installed.

Provider selection (when USE_LLM=true):
    LLM_PROVIDER=anthropic (default) → uses ANTHROPIC_API_KEY + claude-haiku
    LLM_PROVIDER=openai              → uses OPENAI_API_KEY + gpt-4o-mini

The LLM path is INTENTIONALLY SMALL — one provider call, one prompt,
strict grounding rule. No agent loops, no tool calls, no multi-turn.
Scope is "answer the query grounded in retrieved snippets" and nothing
else. Citations validated against the hit set after generation.
"""
from __future__ import annotations
import os

try:
    import weave
    _weave_op = weave.op
except Exception:
    def _weave_op(fn):  # type: ignore[misc]
        return fn

from .citations import extract_citations

USE_LLM = os.environ.get("USE_LLM", "false").lower() in ("1", "true", "yes")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Hardcoded output cap for both providers. 400 tokens ≈ 4-5 sentence answer,
# enough for grounded clinical reasoning without bloat. Tune here, not inline.
MAX_GEN_TOKENS = int(os.environ.get("MAX_GEN_TOKENS", "400"))

GROUNDED_PROMPT_TEMPLATE = """You are answering a clinical retrieval query.
You MUST ground every claim in the retrieved snippets below. Each snippet
has a source_id like "L1-NNNNNN". Cite the source_id inline when you reference it.

If the retrieved snippets do not support the answer, say so. Do NOT invent
clinical detail. Do NOT recommend autonomous actions (no "call this number"
or "go to this URL").

Query: {query}

Retrieved snippets:
{snippets}

Write a 3-5 sentence grounded answer. Include source_id citations inline."""


def _render_snippets(hits: list[dict]) -> str:
    """Format the retrieved hits as numbered context for the LLM prompt."""
    return "\n".join(
        f"[{i+1}] source_id={h['case_id']}  similarity={h.get('score', 0):.3f}\n    {h.get('snippet', '')}"
        for i, h in enumerate(hits)
    )


# ── Provider abstractions (lazy imports so default path needs zero deps) ────
def _call_anthropic(prompt: str) -> str | None:
    """Call Anthropic. Return None on any failure (caller falls back to template)."""
    try:
        import anthropic  # type: ignore
    except ImportError:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_GEN_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return None


def _call_openai(prompt: str) -> str | None:
    """Call OpenAI. Return None on any failure."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=MAX_GEN_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def _llm_call(prompt: str) -> str | None:
    if LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    return _call_anthropic(prompt)


# ── Template path (default — deterministic, zero-cost, auditable) ───────────
def _template_answer(query: str, hits: list[dict]) -> str:
    """Deterministic grounded answer. Always cites the top source_id."""
    top = hits[0]
    extras = hits[1:3]
    parts: list[str] = []
    parts.append(
        f"Based on similar past records, the most relevant precedent is "
        f"{top['case_id']}: \"{top.get('snippet', '')[:150]}\"."
    )
    if extras:
        parts.append(
            "Additional supporting precedents: "
            + ", ".join(f"{h['case_id']}" for h in extras)
            + "."
        )
    parts.append(
        f"Total candidate cases returned: {len(hits)}. "
        f"This answer is grounded — every claim cites a retrieved source_id."
    )
    return " ".join(parts)


# ── Public API ──────────────────────────────────────────────────────────────
@_weave_op
def generate_answer(query: str, hits: list[dict]) -> dict:
    """Produce a grounded answer + citation list from retrieved hits.

    Args:
        query:  free-text user question.
        hits:   list of {case_id, snippet, score, ...} from retrieval.

    Returns:
        {"answer": str, "citations": list[str], "warnings": list[str],
         "generation_method": "template" | "llm_anthropic" | "llm_openai"}
    """
    if not hits:
        return {
            "answer": (
                f"No relevant past cases were retrieved for the query: \"{query}\". "
                "Returning empty rather than guessing."
            ),
            "citations": [],
            "warnings": ["empty_retrieval_set"],
            "generation_method": "template",
        }

    warnings: list[str] = []
    method = "template"
    answer: str | None = None

    # 1) LLM path (only when feature-flagged AND provider is reachable)
    if USE_LLM:
        prompt = GROUNDED_PROMPT_TEMPLATE.format(
            query=query, snippets=_render_snippets(hits)
        )
        llm_out = _llm_call(prompt)
        if llm_out:
            answer = llm_out
            method = f"llm_{LLM_PROVIDER}"
        else:
            warnings.append(
                f"LLM ({LLM_PROVIDER}) unavailable or errored — template fallback"
            )

    # 2) Template fallback (also the default path)
    if answer is None:
        answer = _template_answer(query, hits)

    # 3) Validate citations against retrieved set
    valid_ids = {h.get("case_id") for h in hits if h.get("case_id")}
    cited, dropped = extract_citations(answer, valid_ids)
    if dropped:
        warnings.append(f"dropped {len(dropped)} hallucinated citation(s): {dropped[:3]}")

    return {
        "answer": answer,
        "citations": cited,
        "warnings": warnings,
        "generation_method": method,
    }
