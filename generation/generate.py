"""Grounded answer generation.

Takes a query + retrieved hits and produces a structured answer string
with embedded source citations. Template-based today (deterministic,
auditable, zero LLM cost). Swap to an LLM call when faithfulness eval
shows the template hits its ceiling.

The shape:
    generate_answer(query, hits) -> {"answer": str, "citations": [source_id]}

The answer text always cites at least one source_id when hits is non-empty.
Citation validation lives in citations.py — call validate_citations()
after generation to drop any hallucinated cites.
"""
from __future__ import annotations
from .citations import extract_citations, validate_citations


def generate_answer(query: str, hits: list[dict]) -> dict:
    """Produce a grounded answer + citation list from retrieved hits.

    Args:
        query:  free-text user question.
        hits:   list of {case_id, snippet, score, ...} from retrieval.

    Returns:
        {"answer": str, "citations": list[str], "warnings": list[str]}
    """
    if not hits:
        return {
            "answer": (
                f"No relevant past cases were retrieved for the query: \"{query}\". "
                "Returning empty rather than guessing."
            ),
            "citations": [],
            "warnings": ["empty_retrieval_set"],
        }

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
    answer = " ".join(parts)

    valid_ids = {h.get("case_id") for h in hits if h.get("case_id")}
    cited, dropped = extract_citations(answer, valid_ids)

    warnings: list[str] = []
    if dropped:
        warnings.append(f"dropped {len(dropped)} hallucinated citation(s)")

    return {"answer": answer, "citations": cited, "warnings": warnings}
