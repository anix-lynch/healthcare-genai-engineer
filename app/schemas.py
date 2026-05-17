"""Request / response contracts for the FastAPI surface.

Pydantic models so OpenAPI docs are auto-generated and validation is free.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


Method = Literal["bm25", "hybrid"]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., min_length=1, description="free-text question to RAG over")
    k: int = Field(5, ge=1, le=50, description="number of hits to retrieve")
    method: Method = Field("bm25", description="retrieval strategy")


class Citation(BaseModel):
    source_id: str
    snippet: str = Field(..., description="up to 200-char preview of the cited record")
    similarity: float = Field(..., ge=0, le=1)


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    answer: str = Field(..., description="grounded answer; every claim cites a source_id")
    citations: list[Citation] = Field(default_factory=list)
    method_used: Method
    retrieved_count: int = Field(..., ge=0)
    latency_ms: int = Field(..., ge=0)
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
