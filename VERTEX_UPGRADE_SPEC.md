# Vertex-Native Upgrade Spec
**Goal:** DIY RAG → managed Vertex AI platform. Same healthcare questions, same eval gate.  
**Credit:** `01BF27-7A90D2-EDD523` (gozeroshot.dev GCP org, $900)  
**Branch:** `vertex-native`

---

## Architecture: DIY baseline vs Vertex-native lane

```
CURRENT (DIY)                     VERTEX-NATIVE
─────────────────────────────     ─────────────────────────────
BM25 / FastEmbed (local ONNX)  →  text-embedding-005 (Vertex AI)
Custom vector store (numpy)    →  Vertex Vector Search
CSV flat file corpus           →  Vertex AI Search data stores
  doc: BM25 over snippets      →    document data store
  struct: field extraction     →    structured data store
  web: ❌ placeholder           →    Google Search grounding
  vid: ❌ placeholder           →    (future: media search)
Custom citation validation     →  Check Grounding API
Custom Python orchestration    →  Agent Builder / ADK playbook
Custom eval harness            →  Gen AI Eval Service
```

---

## Implementation slices (ordered by signal:effort)

### Slice 1 — Vertex AI Embeddings ✅ IN PROGRESS
**File:** `retrieval/vertex_embed.py`  
**Model:** `text-embedding-005` (768-dim, state-of-art MTEB)  
**Toggle:** `EMBEDDING_BACKEND=vertex` env var (fallback: local FastEmbed)  
**Cost:** ~$0.00002/1k tokens — negligible on 497-record corpus  

### Slice 2 — Vertex AI Search data store
**File:** `retrieval/vertex_search.py`  
**Setup:** Discovery Engine API → create data store → import CSV  
**Toggle:** `method=vertex` in `/v1/ask`  
**Benchmark:** compare hit@5 vs DIY hybrid on golden eval set  

### Slice 3 — Google Search grounding (web lane)
**File:** `app/grounding.py` → `web` lane from `is_real=False` → `is_real=True`  
**API:** Grounding with Google Search via `google-genai` SDK  
**Signal:** "public medical guidelines grounded via Google Search"  

### Slice 4 — Check Grounding API
**File:** `generation/vertex_grounding_check.py`  
**Purpose:** validate citations against Vertex grounding corpus  
**Signal:** "grounding validity verified by Vertex Check Grounding API"  

### Slice 5 — Gen AI Eval Service
**File:** `evaluation/vertex_eval.py`  
**Replaces:** custom Ragas runner with managed Vertex evaluation  

---

## Benchmark target (bragable bar)

| Metric | DIY baseline | Vertex target |
|---|---|---|
| hit@5 | 100% (hybrid) | ≥ 100% |
| embedding latency | ~2ms (local) | ~50ms (API) |
| citation validity | custom check | Check Grounding API |
| web grounding | ❌ | Google Search ✅ |

---

## Status

- [x] Branch created: `vertex-native`
- [x] Spec written
- [ ] Slice 1: Vertex embeddings
- [ ] Slice 2: Vertex AI Search data store
- [ ] Slice 3: Google Search grounding
- [ ] Slice 4: Check Grounding API
- [ ] Slice 5: Gen AI Eval
