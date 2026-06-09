"""Smoke test /v1/act — the live action endpoint that changes durable state.

The point this test defends: unlike /v1/ask (which returns advice JSON),
/v1/act produces a receipt where `before_committed != after_committed` — proof
the service actually acted — and replaying the same correlation_id is
idempotent (no duplicate side effect).
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate durable action state per test run.
    monkeypatch.setenv("ACTION_DB_PATH", str(tmp_path / "act.db"))
    return TestClient(app)


def _payload(correlation_id="case-act-smoke"):
    return {
        "correlation_id": correlation_id,
        "triage_level": "NOW",
        "predicted_los_hours": 6,
        "bed_pressure_risk": "low",
        "er_state": {"available_beds": 3, "occupancy_pct": 70, "queue_length": 2},
    }


def test_act_changes_durable_state(client):
    r = client.post("/v1/act", json=_payload())
    assert r.status_code == 200
    data = r.json()
    assert data["accepted"] is True
    assert data["disposition"] is not None
    assert data["acknowledged"] is True
    # the action actually happened: durable world-state went false -> true
    assert data["before_committed"] is False
    assert data["after_committed"] is True
    assert data["outcome_verified"] is True
    assert data["false_success_detected"] is False


def test_act_is_idempotent_on_replay(client):
    first = client.post("/v1/act", json=_payload("case-replay")).json()
    second = client.post("/v1/act", json=_payload("case-replay")).json()
    # Replay applies nothing new; state is already committed, no duplicate.
    assert first["after_committed"] is True
    assert second["after_committed"] is True
    assert second["accepted"] is True


def test_act_fails_closed_on_bad_evidence(client):
    bad = _payload("case-bad")
    bad["er_state"] = {"occupancy_pct": 999}  # out of range → schema rejects
    r = client.post("/v1/act", json=bad)
    assert r.status_code == 422
