"""Grounding evidence contract for the /vertex ER Insight Console.

4 source lanes — honest about what is real vs placeholder:

  doc:    chunked text retrieval (physician notes, HPI, chief complaint)
          REAL — powered by BM25 / dense / hybrid over healthcare CSV
  struct: structured row data (vitals, labs, admission type, medication)
          REAL — same CSV hits, raw field extracted for key-value display
  web:    web search grounding
          PLACEHOLDER — no adapter implemented yet
  vid:    video / media semantic search
          PLACEHOLDER — no adapter implemented yet

Invariant: is_real=False evidence must never be displayed as proven grounding.
The /vertex UI must surface is_real status so doctors see honest provenance.
"""
from __future__ import annotations
import json

from app.schemas import GroundingEvidence, SourceType


# ── Struct lane: extract vitals + labs from raw CSV row ────────────────────

def _struct_snippet(raw: dict) -> str:
    parts: list[str] = []

    bp_s = raw.get("bp_systolic")
    bp_d = raw.get("bp_diastolic")
    if bp_s and bp_d:
        parts.append(f"BP {bp_s}/{bp_d}")
    if raw.get("heart_rate"):
        parts.append(f"HR {raw['heart_rate']}")
    if raw.get("respiratory_rate"):
        parts.append(f"RR {raw['respiratory_rate']}")
    if raw.get("spo2_pct"):
        parts.append(f"O2 {raw['spo2_pct']}%")
    if raw.get("temperature_f"):
        parts.append(f"Temp {raw['temperature_f']}°F")
    vitals = " · ".join(parts) if parts else "vitals: unavailable"

    labs_str = ""
    try:
        labs = json.loads(raw.get("lab_panel_json") or "{}")
        lab_parts = [
            f"{k.split('_')[0]}={v}"
            for k, v in labs.items()
            if v is not None
        ]
        if lab_parts:
            labs_str = " | Labs: " + " · ".join(lab_parts[:4])
    except Exception:
        pass

    condition = raw.get("Medical Condition") or raw.get("medical_condition") or ""
    admission = raw.get("Admission Type") or raw.get("admission_type") or ""
    medication = raw.get("Medication") or raw.get("medication") or ""

    header = " · ".join(filter(None, [condition, admission]))
    rx = f" · Rx: {medication}" if medication else ""
    return f"{header} — {vitals}{labs_str}{rx}"


# ── Lane builders ──────────────────────────────────────────────────────────

def _doc_evidence(hits: list[dict]) -> list[GroundingEvidence]:
    """Text-retrieval lane: physician notes, HPI, chief complaint."""
    out: list[GroundingEvidence] = []
    for h in hits:
        case_id = h.get("case_id", "")
        snippet = (h.get("snippet") or "")[:200]
        similarity = min(1.0, max(0.0, float(h.get("score", 0)) / 12.0))
        out.append(GroundingEvidence(
            source_type="doc",
            source_id=f"doc:{case_id}",
            snippet=snippet,
            similarity=similarity,
            is_real=True,
            provenance="BM25 / dense retrieval over physician notes + HPI",
        ))
    return out


def _struct_evidence(hits: list[dict]) -> list[GroundingEvidence]:
    """Structured data lane: vitals, labs, admission type from CSV rows."""
    out: list[GroundingEvidence] = []
    for h in hits:
        raw = h.get("raw") or {}
        if not raw:
            continue
        case_id = h.get("case_id", "")
        snippet = _struct_snippet(raw)[:200]
        similarity = min(1.0, max(0.0, float(h.get("score", 0)) / 12.0))
        out.append(GroundingEvidence(
            source_type="struct",
            source_id=f"struct:{case_id}",
            snippet=snippet,
            similarity=similarity,
            is_real=True,
            provenance="Structured fields: vitals, labs, admission type, medication",
        ))
    return out


def _web_placeholder() -> list[GroundingEvidence]:
    """Web search lane — honest placeholder, no adapter yet."""
    return [GroundingEvidence(
        source_type="web",
        source_id="web:placeholder",
        snippet="Web grounding adapter not yet implemented. No external search results were used for this answer.",
        similarity=0.0,
        is_real=False,
        provenance="Planned: Vertex AI Search (website corpus). Not yet wired.",
    )]


def _vid_placeholder() -> list[GroundingEvidence]:
    """Video / media lane — honest placeholder, no adapter yet."""
    return [GroundingEvidence(
        source_type="vid",
        source_id="vid:placeholder",
        snippet="Video / media semantic search not yet implemented. No media evidence was used for this answer.",
        similarity=0.0,
        is_real=False,
        provenance="Planned: Vertex AI Media Search (procedure / anatomy video). Not yet wired.",
    )]


# ── Public API ─────────────────────────────────────────────────────────────

def build_grouped_evidence(hits: list[dict]) -> dict[str, list[GroundingEvidence]]:
    """Return evidence grouped by source_type lane.

    doc + struct are real (sourced from CSV retrieval hits).
    web + vid are honest placeholders until adapters are implemented.
    """
    return {
        "doc":    _doc_evidence(hits),
        "struct": _struct_evidence(hits),
        "web":    _web_placeholder(),
        "vid":    _vid_placeholder(),
    }


def enrich_citations(
    citations: list,
    hits: list[dict],
) -> list:
    """Attach source_type='doc' to existing Citation objects.

    Citations come from the doc retrieval lane only. This ensures the
    /vertex UI source tag is driven by real backend data, not JS inference.
    """
    hit_map: dict[str, str] = {}
    for h in hits:
        cid = h.get("case_id", "")
        hit_map[cid] = "doc"

    enriched = []
    for c in citations:
        raw_id = c.source_id
        enriched.append(c.model_copy(update={"source_type": hit_map.get(raw_id, "doc")}))
    return enriched
