# Roadmap — Incremental Population from Monorepo

Source of truth: [`healthcare-genai-fullstack`](https://github.com/anix-lynch/healthcare-genai-fullstack).
This file tracks what lands in this presentation cut, in small commits.

**Sequencing principle:** vertical slice FIRST, internals after. The repo
should feel alive (one working `POST /ask`) before the folders look pretty.

---

## Target scaffold (GenAI Engineer lens)

```
healthcare-genai-engineer/
├── app/
│   ├── main.py                    # FastAPI entrypoint
│   ├── schemas.py
│   ├── dependencies.py
│   └── routers/
│       ├── ask.py                 # POST /v1/ask — the one workflow
│       └── health.py
├── retrieval/
│   ├── chunking.py
│   ├── embed.py
│   ├── vector_store.py
│   ├── hybrid_retriever.py
│   ├── retriever.py               # BM25 engine (supporting)
│   ├── dense.py                   # MiniLM engine (supporting)
│   └── query_pipeline.py          # facade
├── generation/
│   ├── generate.py                # grounded answer
│   └── citations.py               # validate every cite
├── guardrails/
│   ├── input_validator.py
│   ├── output_validator.py
│   └── pii_masker.py
├── evaluation/
│   ├── ragas_runner.py            # custom-proxy faithfulness + recall
│   ├── golden_set.json            # 20 hand-curated queries
│   ├── regression_gate.py         # exit 1 on metric drop
│   └── baseline.json              # seeded on first eval run
├── jobs/                          # ingest · build_index · refresh_eval
├── tests/                         # pytest + FastAPI TestClient
├── demo/                          # sample_queries.md
├── data/raw/                      # 497-row enriched corpus
├── outputs/                       # eval_summary.json
├── docs/                          # eval-results.md
├── Makefile
├── requirements.txt
└── Dockerfile
```

---

## Phase status (audit-friendly — phase = dependency order, NOT calendar)

```
☑️ Phase 1 — scaffold                                        commits cc60189 → 8be6dfb
   ☑ repo + README + ROADMAP + .gitignore + .env.example + folder tree

☑️ Phase 2 — THIN VERTICAL SLICE (recruiter signal)          commit 3c51c72
   ☑ app/main.py            FastAPI mount
   ☑ app/schemas.py         AskRequest / AskResponse / Citation / HealthResponse
   ☑ app/dependencies.py    lazy retrieval-pipeline singleton
   ☑ app/routers/ask.py     POST /v1/ask — the one workflow
   ☑ app/routers/health.py  GET /health
   ☑ retrieval/query_pipeline.py    facade over BM25/hybrid
   ☑ generation/generate.py         grounded answer
   ☑ generation/citations.py         extract + validate

☑️ Phase 3 — RETRIEVAL INTERNALS                              commit 3c51c72 (same)
   ☑ retrieval/embed.py             MiniLM facade
   ☑ retrieval/vector_store.py      in-memory + swap path
   ☑ retrieval/hybrid_retriever.py  BM25 + dense + RRF
   ☑ retrieval/chunking.py          window + overlap
   ☑ retrieval/retriever.py         BM25 engine (supporting)
   ☑ retrieval/dense.py             MiniLM engine (supporting)

☑️ Phase 4 — GUARDRAILS + REGRESSION GATE                    commit 6c87465
   ☑ guardrails/input_validator.py  sanitize + injection scan + token cap
   ☑ guardrails/output_validator.py citation valid + min length + forbidden actions
   ☑ guardrails/pii_masker.py        regex baseline (SSN/phone/email/CC/MRN/DOB)
   ☑ evaluation/regression_gate.py  exit 1 on tolerance violation
   ☑ wired into app/routers/ask.py  input → retrieve → generate → output → return

☑️ Phase 5 — EVAL HARNESS                                     commit 6c87465 (same)
   ☑ evaluation/ragas_runner.py     custom-proxy faithfulness + relevance
   ☑ evaluation/golden_set.json     20 hand-curated queries
   ☑ evaluation/baseline.json       seeded on first run
   ☑ outputs/eval_summary.json      first run results
   ☑ docs/eval-results.md           human-readable summary

☑️ Phase 6 — OPERATIONAL POLISH                              commit pending (this turn)
   ☑ Makefile        install · serve · demo · test · eval · gate · clean
   ☑ tests/          test_ask.py — 3 tests pass
   ☑ jobs/           ingest_documents.py · build_index.py · refresh_eval.py
   ☑ demo/           sample_queries.md (5 curl commands + expected outputs)
   ☑ Dockerfile      python:3.11-slim + HEALTHCHECK
   ☑ docker-compose.yml
```

---

## Why this order (Comet's prescription, internalized)

```
WRONG ORDER                          RIGHT ORDER
─────────────────────────────────────────────────────────────────────
folder population first              vertical slice first
   ↓                                    ↓
"look, retrieval/ exists"            "POST /ask works end-to-end"
   ↓                                    ↓
recruiter: "OK but does it run?"     recruiter: "show me the JSON"
                                        (5-second skim → unlock)
```

Comet's remaining-gap critique = runtime proof, not folder completeness.
Phase 2 = the vertical slice (`/ask` → retrieve → generate → cite → return).
Phase 3+ = harden the internals AFTER the slice runs.

---

## Anti-overbuild reminders

- Recycle from monorepo. Do not rewrite.
- Honest tone in every README. No enterprise buzzword inflation.
- One use case (ER triage). Not all 7 patterns at once.
- Run before pretty.
- Phase order = dependency order. No calendar implication. Ship in one
  sitting if energy allows.
