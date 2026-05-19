"""
Pattern 1 — Rachel · Retrieval over Layer 1 patient corpus.

"Find me past ER cases that smell like this chief complaint."

Implementation: BM25 in pure Python. No external vector DB, no embedding API.
Operates over the Layer 1 healthcare_dataset.csv (55,500 encounters). Each
row is rendered to a searchable text snippet at index time:

    "{Age}yo {Gender}, {Medical Condition}, {Admission Type} admission,
     treated with {Medication}, test results {Test Results}"

Why BM25 (not embeddings) for v1:
    - zero API keys, zero cost, zero network
    - reproducible / deterministic / explainable for compliance
    - 55K docs index in <5s on M-series Mac
    - LLM uplift swappable later via shared.cloud.adapter without changing
      the search() contract

Lifecycle:
    search(query) lazy-loads & indexes the CSV on first call. Subsequent
    calls reuse the in-memory index.
"""

from __future__ import annotations
import math
import re
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable

try:
    import weave
    _weave_op = weave.op
except Exception:
    def _weave_op(fn):  # type: ignore[misc]
        return fn

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "healthcare_dataset.csv"
)

K1 = 1.5
B = 0.75
TOKEN_RE = re.compile(r"[a-z0-9]+")


# ── Tokenizer ──────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


# ── Snippet builder for Layer 1 row ────────────────────────────────────────
def _row_to_snippet(row: dict) -> str:
    """Render one enriched corpus row into a BM25/dense-friendly snippet.

    Snippet shape (kept under ~400 chars to keep BM25 + FastEmbed fast):

       "29yo Female · Arthritis · Emergency · Lipitor · Normal labs
        · CC: Severe joint pain and swelling in hands and knees
        · HPI: 29F acute polyarticular pain, no fever, no trauma...
        · vitals: bp 128/82 hr 86 rr 16 spo2 99 temp 98.7
        · ESI 3 · flags: none"

    Why so much:
        Earlier we used the 6-field demographic-only snippet, which left
        ~80% of the enriched row's signal on the floor. Dense retrieval
        hit a ceiling at Hit@5=0.75 because paraphrases like "elephant
        on chest" had nowhere to land in token-space. Surfacing CC + HPI
        + vitals + ESI tier into the snippet text lets BOTH retrievers
        (keyword + semantic) match real clinical narrative tokens.

    Source-of-truth: the canonical enriched corpus lives in the
    `healthcare-api` repo (L1 substrate) at
    `data/raw/healthcare_dataset_enriched.csv`. The copy in this repo
    at `data/raw/healthcare_dataset.csv` is the same 497 rows; rebuild
    via `cp ~/dev/healthcare-api/data/raw/healthcare_dataset_enriched.csv
    ./data/raw/healthcare_dataset.csv` then redeploy (Dockerfile
    re-runs scripts/build_dense_index.py at build time).
    """
    age = row.get("Age", "")
    gender = row.get("Gender", "")
    condition = row.get("Medical Condition", "")
    admission = row.get("Admission Type", "")
    medication = row.get("Medication", "")
    test_results = row.get("Test Results", "")
    cc = (row.get("chief_complaint") or "").strip()
    hpi = (row.get("hpi") or "").strip()
    esi = row.get("esi_tier_truth")
    flags = (row.get("acuity_red_flags") or "").strip() or "none"

    # Vitals block — only include populated readings; skip blanks.
    vitals_parts = []
    bp_s, bp_d = row.get("bp_systolic"), row.get("bp_diastolic")
    if bp_s and bp_d:
        vitals_parts.append(f"bp {bp_s}/{bp_d}")
    for label, key in (("hr", "heart_rate"), ("rr", "respiratory_rate"),
                       ("spo2", "spo2_pct"), ("temp", "temperature_f")):
        val = row.get(key)
        if val:
            vitals_parts.append(f"{label} {val}")
    vitals = " ".join(vitals_parts) or "n/a"

    # Trim HPI to keep snippet under budget. Sentence-bounded if possible.
    hpi_trimmed = hpi[:180]
    if len(hpi) > 180:
        cut = hpi_trimmed.rfind(". ")
        if cut > 80:
            hpi_trimmed = hpi[:cut + 1]
        hpi_trimmed += "…"

    parts = [
        f"{age}yo {gender}",
        f"{condition}",
        f"{admission}",
        f"{medication}",
        f"{test_results} labs",
    ]
    if cc:
        parts.append(f"CC: {cc}")
    if hpi_trimmed:
        parts.append(f"HPI: {hpi_trimmed}")
    parts.append(f"vitals: {vitals}")
    if esi not in (None, "", "nan"):
        parts.append(f"ESI {esi}")
    parts.append(f"flags: {flags}")

    return " · ".join(parts)


# ── Index ──────────────────────────────────────────────────────────────────
class BM25Index:
    """In-memory BM25 index over an iterable of {case_id, snippet, raw} dicts."""

    def __init__(self, k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_lens: list[int] = []
        self.avg_dl: float = 0.0
        self.term_df: dict[str, int] = defaultdict(int)
        self.term_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.N: int = 0

    def add(self, case_id: str, snippet: str, raw: dict | None = None) -> None:
        tokens = _tokenize(snippet)
        doc_idx = len(self.docs)
        self.docs.append({"case_id": case_id, "snippet": snippet, "raw": raw or {}})
        self.doc_tokens.append(tokens)
        self.doc_lens.append(len(tokens))
        tf = Counter(tokens)
        for term, freq in tf.items():
            self.term_postings[term].append((doc_idx, freq))
            self.term_df[term] += 1

    def freeze(self) -> None:
        self.N = len(self.docs)
        self.avg_dl = sum(self.doc_lens) / self.N if self.N else 0.0

    def _idf(self, term: str) -> float:
        df = self.term_df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(((self.N - df + 0.5) / (df + 0.5)) + 1.0)

    def search(self, query: str, k: int = 10) -> list[dict]:
        q_tokens = _tokenize(query)
        if not q_tokens or self.N == 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(q_tokens):
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for doc_idx, freq in self.term_postings.get(term, []):
                dl = self.doc_lens[doc_idx]
                norm = 1 - self.b + self.b * (dl / self.avg_dl) if self.avg_dl else 1
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * norm
                scores[doc_idx] += idf * (num / den)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            {
                "case_id": self.docs[idx]["case_id"],
                "snippet": self.docs[idx]["snippet"],
                "score":   round(score, 4),
                "raw":     self.docs[idx]["raw"],
            }
            for idx, score in ranked
        ]


# ── Module-level lazy singleton ────────────────────────────────────────────
_INDEX: BM25Index | None = None


def _build_default_index() -> BM25Index:
    if not DEFAULT_CORPUS.exists():
        raise FileNotFoundError(
            f"Layer 1 corpus not found at {DEFAULT_CORPUS}. "
            f"Pass an explicit corpus to index_cases() instead."
        )
    idx = BM25Index()
    with DEFAULT_CORPUS.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            snippet = _row_to_snippet(row)
            idx.add(case_id=f"L1-{i:06d}", snippet=snippet, raw=row)
    idx.freeze()
    return idx


def index_cases(cases: Iterable[dict]) -> None:
    """
    Build/replace the in-memory index from any iterable of dicts.

    Each dict needs: 'case_id' (str) and either 'snippet' (str) or the
    Layer 1 columns (Age, Gender, Medical Condition, etc.).
    """
    global _INDEX
    idx = BM25Index()
    for c in cases:
        case_id = str(c.get("case_id") or c.get("id") or len(idx.docs))
        snippet = c.get("snippet") or _row_to_snippet(c)
        idx.add(case_id=case_id, snippet=snippet, raw=c)
    idx.freeze()
    _INDEX = idx


def _ensure_index() -> BM25Index:
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_default_index()
    return _INDEX


# ── Public API ─────────────────────────────────────────────────────────────
@_weave_op
def search(query: str, k: int = 10) -> list[dict]:
    """
    Return top-k past cases ranked by BM25 against the query.

    Returns: list of {case_id, snippet, score, raw}, length <= k.
    """
    return _ensure_index().search(query, k=k)


def search_for_case(case: dict, k: int = 5) -> list[dict]:
    """Convenience: build a retrieval query from an ER triage case."""
    cc = case.get("cc", "") or ""
    hpi = case.get("hpi", "") or ""
    age_match = re.search(r"\b(\d{1,3})\s*(yo|y/o|year)", hpi, re.I)
    age = age_match.group(1) if age_match else ""
    gender = "Male" if re.search(r"\b(M|male|man|boy)\b", hpi) else (
        "Female" if re.search(r"\b(F|female|woman|girl)\b", hpi) else ""
    )
    query = f"{age} {gender} {cc} {hpi}".strip()
    return search(query, k=k)


def index_size() -> int:
    return _ensure_index().N


if __name__ == "__main__":
    import json, sys
    q = sys.stdin.read().strip() if not sys.stdin.isatty() else "62yo male chest pain hypertension"
    print(f"Index size: {index_size()}")
    print(f"Query: {q}")
    for hit in search(q, k=5):
        print(json.dumps({k: v for k, v in hit.items() if k != "raw"}, indent=2))
