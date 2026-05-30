"""FastAPI entrypoint for healthcare-genai-engineer.

Mounts /ask (the one GenAI workflow) and /health (sanity check).
Run:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations
import os
from fastapi import FastAPI

from app.routers import ask, health, web, vertex

# Weave tracing — no-op if WANDB_API_KEY is absent (local dev without key)
if os.environ.get("WANDB_API_KEY"):
    try:
        import weave
        weave.init("alynch-zeroshot/healthcare-genai")
    except Exception:
        pass

app = FastAPI(
    title="healthcare-genai-engineer",
    version="0.1.0",
    description=(
        "Focused GenAI runtime over the healthcare corpus: "
        "retrieve → generate → cite → return."
    ),
)

app.include_router(web.router, tags=["web"])           # GET / → single-page RAG visualizer
app.include_router(vertex.router, tags=["vertex"])     # GET /vertex → ER Insight Console
app.include_router(health.router, tags=["meta"])
app.include_router(ask.router, prefix="/v1", tags=["rag"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
