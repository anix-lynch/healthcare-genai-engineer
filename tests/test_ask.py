"""Smoke test the one workflow — /v1/ask end-to-end through the FastAPI app."""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "healthcare-genai-engineer"


def test_ask_returns_grounded_answer(client):
    r = client.post(
        "/v1/ask",
        json={"query": "62yo male chest pain hypertension", "k": 5, "method": "bm25"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["query"] == "62yo male chest pain hypertension"
    assert isinstance(data["answer"], str) and len(data["answer"]) > 10
    assert data["retrieved_count"] >= 0
    assert isinstance(data["citations"], list)
    assert data["method_used"] == "bm25"
    assert data["latency_ms"] >= 0


def test_ask_empty_query_rejected(client):
    r = client.post("/v1/ask", json={"query": "", "k": 5, "method": "bm25"})
    assert r.status_code == 422  # pydantic validation
