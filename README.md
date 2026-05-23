# healthcare-genai-engineer

> **Healthcare RAG service** — FastAPI + BM25/dense hybrid retrieval + custom-proxy eval + PII guardrails + regression gate. One ER-triage workflow, end-to-end.

![Demo](demo.gif)

🔗 **Live:** https://healthcare-genai-2ihyeqmb6q-uw.a.run.app

🔎 **Architecture:** [docs/architecture.md](docs/architecture.md)

[![eval-gate](https://github.com/anix-lynch/healthcare-genai-engineer/actions/workflows/eval.yml/badge.svg)](https://github.com/anix-lynch/healthcare-genai-engineer/actions/workflows/eval.yml)

```
POST /v1/ask
  → input guard (sanitize · injection scan · PII mask)
  → BM25 retrieval over 497-row enriched healthcare corpus
  → prediction signal (future-risk / LOS / bed pressure)
  → grounded answer (template baseline, LLM path behind USE_LLM flag)
  → orchestration / override rules
  → output guard (citation valid · forbidden actions · length)
  → JSON response with cited source_ids + warnings + prediction signal
```

---

## Quick demo (no server needed)

```bash
git clone https://github.com/anix-lynch/healthcare-genai-engineer
cd healthcare-genai-engineer
make install
make demo
```

Output (live, just run it):

```json
{
  "query": "62yo male chest pain hypertension",
  "answer": "Based on similar past records, the most relevant precedent is L1-000085: \"62yo Male, Hypertension, Emergency admission, treated with Aspirin, test results Abnormal\". Additional supporting precedents: L1-000149, L1-000121. Total candidate cases returned: 5. This answer is grounded — every claim cites a retrieved source_id.",
  "citations": [
    {
      "source_id": "L1-000085",
      "snippet": "62yo Male, Hypertension, Emergency admission, treated with Aspirin, test results Abnormal",
      "similarity": 0.617
    },
    {
      "source_id": "L1-000149",
      "snippet": "62yo Female, Hypertension, Emergency admission, treated with Aspirin, test results Abnormal",
      "similarity": 0.559
    },
    {
      "source_id": "L1-000121",
      "snippet": "62yo Male, Arthritis, Emergency admission, treated with Ibuprofen, test results Abnormal",
      "similarity": 0.471
    }
  ],
  "method_used": "bm25",
  "retrieved_count": 5,
  "latency_ms": 6,
  "warnings": []
}
```

---

## What's inside

```
app/                  FastAPI: main.py + schemas.py + dependencies.py
                       + routers/ask.py · routers/health.py
                       + prediction.py
retrieval/            BM25 (from-scratch, Okapi k1=1.5/b=0.75)
                       + dense (sentence-transformers MiniLM)
                       + RRF hybrid fusion (k=60, Cormack & Buettcher)
                       + chunking · vector_store · query_pipeline
generation/           grounded answer + citation validation
                       template baseline (default), LLM path via USE_LLM flag
                       supports Anthropic OR OpenAI providers
guardrails/           input_validator: sanitize + injection regex + token cap
                       output_validator: citation + length + forbidden-action
                       pii_masker: SSN · phone · email · CC · MRN · DOB
evaluation/           golden_set.json (20 hand-curated queries)
                       ragas_runner.py (custom-proxy faithfulness + relevance)
                       regression_gate.py (exit 1 on metric drop past tolerance)
                       baseline.json (committed snapshot)
jobs/                 ingest_documents · build_index · refresh_eval
data/raw/             497-row enriched healthcare corpus
docs/eval-results.md  human-readable summary with ASCII bars
demo/                 sample_queries.md (5 curl recipes)
tests/                3 pytest cases over FastAPI TestClient
Dockerfile · docker-compose.yml · Makefile
```

---

## Prediction Signal Node

This repo now includes a lightweight prediction layer inside the existing
workflow.

- **APIs / facts** answer: what is true right now?
- **Prediction** answers: what is likely to happen next?
- **The agent / orchestrator** answers: what should we do?
- **Safety rules** decide when prediction can and cannot influence the final
  recommendation.

```text
Facts/API
    ↓
Triage Classification
    ↓
Prediction Signal
    ↓
Orchestration / Override Rules
    ↓
Human-facing Recommendation
```

Important rule:

- **Prediction does not override acute safety.**
- A `NOW` case stays `NOW`.
- Prediction is used for monitoring, LOS planning, bed planning, and staffing
  context.
- If prediction conflicts with current facts or acute triage, the conflict is
  surfaced explicitly in the explanation.

---

## Eval numbers (committed at `outputs/eval_summary.json`)

```
any_hit_rate              0.650    BM25 baseline — hybrid targets 0.85+
faithfulness_avg          0.650    (no hallucinated citations on hits)
condition_relevance       0.567    BM25-only — dense path targets 0.75+
p95 latency               5 ms     (BM25 over 497 rows, in-memory)
avg citations / query     1.95
n_queries                 20
```

These are **intentionally pre-hybrid BM25 baseline numbers**. The dense path
is wired and swap-ready (`pip install sentence-transformers` then change
default method in `retrieval/query_pipeline.py`). The baseline exists so the
regression gate has a defensible floor to protect before adding model deps.
Full breakdown + per-query scores in [`docs/eval-results.md`](docs/eval-results.md).

The regression gate (`make gate`) blocks merges if any metric drops past
tolerance. CI runs it on every PR via [`.github/workflows/eval.yml`](.github/workflows/eval.yml).

---

## Record your own demo (asciinema)

The JSON above is a faithful sample of `make demo` output. To capture a
moving terminal recording (useful for LinkedIn / portfolio):

```bash
# install asciinema (one-time)
brew install asciinema      # macOS
# or: pip install asciinema

asciinema rec demo.cast
# inside the recording:
make demo
# Ctrl-D to stop

# upload + share (free public link)
asciinema upload demo.cast
```

`demo.cast` is plain text (no binary GIF). Embeds cleanly in any
markdown reader, plays back at recorded speed.

---

## Common commands

```bash
make install     # pip install requirements
make serve       # uvicorn on :8000
make demo        # fire one /v1/ask via TestClient, print JSON
make test        # pytest tests/
make eval        # run golden-set, write outputs/eval_summary.json
make gate        # compare vs baseline, exit 1 on regression
make clean       # remove __pycache__ + .pytest_cache + outputs/
```

Or with Docker:

```bash
docker compose up --build       # service on :8000 with HEALTHCHECK
curl localhost:8000/health
```

---

## LLM-enhanced mode (optional)

The default `generation/generate.py` uses a deterministic template (zero LLM
cost, deterministic output, useful as a faithfulness baseline). To swap in a
real LLM call:

```bash
export USE_LLM=true
export LLM_PROVIDER=anthropic         # or openai
export ANTHROPIC_API_KEY=...          # or OPENAI_API_KEY
make serve
```

The LLM path:
- sends a grounded prompt with the retrieved snippets
- enforces inline `source_id` citations
- falls back to the template if the provider errors or the SDK isn't installed
- never raises — the caller always gets a structured response

---

## Healthcare context

Synthetic 497-row corpus with chief complaint, HPI, vitals, lab flags, and
ESI ground-truth tags. Patient identity resolver bridges 55K encounters → 40K
unique patients so the cross-patient leak guard has real surface area to defend.

Out of scope (intentionally): real EHR / FHIR feeds, HIPAA BAA, multi-cloud
deployment, autonomous decisioning. The repo is one focused healthcare-RAG
vertical, not a hospital system.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the phase-ordered audit trail (Phase 1 → 6,
each with the commit hash that shipped it).

---

## Related repos

- [healthcare-ai-data-engineer](https://github.com/anix-lynch/healthcare-ai-data-engineer) — Layer 1 data backbone (dbt medallion + FastAPI + enrichment + quality gate)
- [healthcare-forward-deployed-engineer](https://github.com/anix-lynch/healthcare-forward-deployed-engineer) — customer-deployment package (integrations + runbook + acceptance tests + Docker)
- [healthcare-genai-fullstack](https://github.com/anix-lynch/healthcare-genai-fullstack) — full 3-layer monorepo (this repo is the GenAI Engineer slice)

---

## License

MIT.
