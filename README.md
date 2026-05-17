# healthcare-genai-engineer

> **Focused presentation cut of [`healthcare-genai-fullstack`](https://github.com/anix-lynch/healthcare-genai-fullstack) — GenAI Engineer lens.**

This repo presents the **retrieval + generation + evaluation + guardrails** slice of the master monorepo, scoped to the GenAI Engineer workflow:

- one working RAG vertical
- evaluation harness with visible numbers
- guardrails and leakage protection
- FastAPI service surface
- runtime scaffolding (not a production deployment)

It does **not** duplicate the full monorepo. Layer 1 data backbone, Layer 3 governance scripts, and full 7-pattern coverage live in the master repo.

---

## Status

🚧 Work in progress — incrementally extracted from the master monorepo.  
See [`ROADMAP.md`](ROADMAP.md) for what is landing next, in small commits.

---

## Master monorepo

Full architecture context (3 layers · 7 patterns · multi-cloud adapter):

→ https://github.com/anix-lynch/healthcare-genai-fullstack

---

## Source of truth

This repo is a **presentation lens**, not an independent codebase.  
When in doubt, the monorepo is authoritative.

The goal here is **focused presentation**, not infrastructure invention.
