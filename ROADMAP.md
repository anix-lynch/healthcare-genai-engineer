# Roadmap — Incremental Population from Monorepo

Source of truth: [`healthcare-genai-fullstack`](https://github.com/anix-lynch/healthcare-genai-fullstack).  
This file tracks what lands in this presentation cut, in small steps.

---

## Target scaffold (GenAI Engineer lens)

```
healthcare-genai-engineer/
├── app/
│   ├── main.py                    # FastAPI entrypoint
│   ├── schemas.py
│   ├── dependencies.py
│   └── routers/
│       ├── ask.py
│       └── health.py
├── core/                          # config · logging · prompts
├── retrieval/
│   ├── chunking.py
│   ├── embed.py
│   ├── vector_store.py
│   ├── hybrid_retriever.py
│   └── query_pipeline.py
├── generation/
│   ├── generate.py
│   └── citations.py
├── guardrails/
│   ├── input_validator.py
│   ├── output_validator.py
│   └── pii_masker.py
├── evaluation/
│   ├── ragas_runner.py
│   ├── golden_set.json
│   └── regression_gate.py
├── jobs/                          # ingest · build_index · refresh_eval
├── tests/
├── demo/
└── docs/
```

---

## Phase status

- [x] **Phase 1:** repo + README + ROADMAP + minimal folder tree
- [ ] **Phase 2:** retrieval/ — copy embed / vector_store / hybrid_retriever / chunking from monorepo
- [ ] **Phase 3:** generation/ + guardrails/ — copy generate + output validator + pii masker
- [ ] **Phase 4:** app/main.py + evaluation/ — copy FastAPI surface + golden_set + ragas_runner
- [ ] **Phase 5:** jobs/ + demo/ — split orchestration DAG + sample queries
- [ ] **Phase 6:** tests + Makefile + Dockerfile

Each phase = one commit. No phase invents new architecture.

---

## Anti-overbuild reminders

- Recycle from monorepo. Do not rewrite.
- Honest tone in every README. No enterprise buzzword inflation.
- One use case (ER triage). Not all 7 patterns at once.
- Run before pretty.
