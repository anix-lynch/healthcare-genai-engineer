"""GET /health — tiny sanity check.

Confirms the FastAPI process is alive AND the retrieval pipeline imports
cleanly. Used by docker-compose healthcheck + manual curl probes.
"""
from __future__ import annotations
from datetime import datetime
from fastapi import APIRouter

from app.schemas import HealthResponse
from app.dependencies import get_pipeline

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Touch the pipeline so dep-import problems surface at /health, not /ask.
    _ = get_pipeline()
    return HealthResponse(
        status="ok",
        service="healthcare-genai-engineer",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
