"""Tests for the 4-lane grounding evidence contract.

Proves:
- source prefixes survive through the API (doc:, struct:, web:, vid:)
- all 4 source lanes are represented in grouped_evidence
- web + vid lanes are honest (is_real=False)
- doc + struct lanes are real (is_real=True)
- /vertex route still renders (200)
- empty hits are handled gracefully
- no existing ask behavior regresses
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.grounding import build_grouped_evidence, enrich_citations
from app.schemas import Citation


@pytest.fixture
def client():
    return TestClient(app)


# ── Unit tests: grounding module ───────────────────────────────────────────

class TestGroundingContract:
    FAKE_HIT = {
        "case_id": "L1-000001",
        "snippet": "29yo Female, Arthritis, Emergency admission",
        "score": 8.5,
        "raw": {
            "Medical Condition": "Arthritis",
            "Admission Type": "Emergency",
            "Medication": "Lipitor",
            "bp_systolic": 128,
            "bp_diastolic": 82,
            "heart_rate": 86,
            "respiratory_rate": 16,
            "temperature_f": 98.7,
            "spo2_pct": 99,
            "lab_panel_json": '{"troponin_ng_ml": null, "wbc_k_ul": 7.2, "glucose_mg_dl": 98}',
        },
    }

    def test_all_4_lanes_present(self):
        grouped = build_grouped_evidence([self.FAKE_HIT])
        assert set(grouped.keys()) == {"doc", "struct", "web", "vid"}

    def test_doc_lane_is_real(self):
        grouped = build_grouped_evidence([self.FAKE_HIT])
        assert all(e.is_real for e in grouped["doc"])

    def test_struct_lane_is_real(self):
        grouped = build_grouped_evidence([self.FAKE_HIT])
        assert all(e.is_real for e in grouped["struct"])

    def test_web_lane_is_placeholder(self):
        grouped = build_grouped_evidence([self.FAKE_HIT])
        assert all(not e.is_real for e in grouped["web"])

    def test_vid_lane_is_placeholder(self):
        grouped = build_grouped_evidence([self.FAKE_HIT])
        assert all(not e.is_real for e in grouped["vid"])

    def test_doc_source_id_prefix(self):
        grouped = build_grouped_evidence([self.FAKE_HIT])
        for e in grouped["doc"]:
            assert e.source_id.startswith("doc:")

    def test_struct_source_id_prefix(self):
        grouped = build_grouped_evidence([self.FAKE_HIT])
        for e in grouped["struct"]:
            assert e.source_id.startswith("struct:")

    def test_web_source_id_prefix(self):
        grouped = build_grouped_evidence([self.FAKE_HIT])
        for e in grouped["web"]:
            assert e.source_id.startswith("web:")

    def test_vid_source_id_prefix(self):
        grouped = build_grouped_evidence([self.FAKE_HIT])
        for e in grouped["vid"]:
            assert e.source_id.startswith("vid:")

    def test_struct_snippet_includes_vitals(self):
        grouped = build_grouped_evidence([self.FAKE_HIT])
        snippet = grouped["struct"][0].snippet
        assert "BP 128/82" in snippet
        assert "HR 86" in snippet

    def test_struct_snippet_includes_labs(self):
        grouped = build_grouped_evidence([self.FAKE_HIT])
        snippet = grouped["struct"][0].snippet
        assert "Labs:" in snippet

    def test_empty_hits_returns_4_lanes(self):
        grouped = build_grouped_evidence([])
        assert set(grouped.keys()) == {"doc", "struct", "web", "vid"}
        assert grouped["doc"] == []
        assert grouped["struct"] == []
        assert len(grouped["web"]) == 1   # honest placeholder
        assert len(grouped["vid"]) == 1   # honest placeholder

    def test_empty_hits_placeholders_honest(self):
        grouped = build_grouped_evidence([])
        assert not grouped["web"][0].is_real
        assert not grouped["vid"][0].is_real

    def test_hit_without_raw_skipped_in_struct(self):
        hit_no_raw = {"case_id": "L1-000002", "snippet": "test", "score": 5.0}
        grouped = build_grouped_evidence([hit_no_raw])
        assert grouped["struct"] == []

    def test_similarity_clamped_0_to_1(self):
        hit_high_score = dict(self.FAKE_HIT, score=999.0)
        grouped = build_grouped_evidence([hit_high_score])
        for lane in ("doc", "struct"):
            for e in grouped[lane]:
                assert 0.0 <= e.similarity <= 1.0

    def test_enrich_citations_sets_source_type(self):
        citations = [
            Citation(source_id="L1-000001", snippet="test", similarity=0.8),
        ]
        hits = [{"case_id": "L1-000001", "snippet": "test", "score": 5.0}]
        enriched = enrich_citations(citations, hits)
        assert enriched[0].source_type == "doc"

    def test_enrich_citations_unknown_id_defaults_doc(self):
        citations = [
            Citation(source_id="unknown-id", snippet="test", similarity=0.5),
        ]
        enriched = enrich_citations(citations, [])
        assert enriched[0].source_type == "doc"


# ── Integration tests: /v1/ask response shape ──────────────────────────────

class TestAskGroundingIntegration:
    def test_ask_returns_grouped_evidence(self, client):
        r = client.post(
            "/v1/ask",
            json={"query": "62yo male chest pain hypertension", "k": 3, "method": "bm25"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert "grouped_evidence" in data

    def test_ask_grouped_evidence_has_4_lanes(self, client):
        r = client.post(
            "/v1/ask",
            json={"query": "sepsis workup fever", "k": 3, "method": "bm25"},
        )
        data = r.json()
        ge = data["grouped_evidence"]
        assert set(ge.keys()) >= {"doc", "struct", "web", "vid"}

    def test_ask_doc_lane_real(self, client):
        r = client.post(
            "/v1/ask",
            json={"query": "DKA presentation", "k": 3, "method": "bm25"},
        )
        data = r.json()
        doc_items = data["grouped_evidence"]["doc"]
        assert all(item["is_real"] for item in doc_items)

    def test_ask_web_lane_placeholder(self, client):
        r = client.post(
            "/v1/ask",
            json={"query": "stroke vs migraine", "k": 3, "method": "bm25"},
        )
        data = r.json()
        web_items = data["grouped_evidence"]["web"]
        assert all(not item["is_real"] for item in web_items)

    def test_ask_source_prefixes_in_grouped(self, client):
        r = client.post(
            "/v1/ask",
            json={"query": "chest pain MI rule out", "k": 3, "method": "bm25"},
        )
        data = r.json()
        ge = data["grouped_evidence"]
        for item in ge["doc"]:
            assert item["source_id"].startswith("doc:")
        for item in ge["struct"]:
            assert item["source_id"].startswith("struct:")
        for item in ge["web"]:
            assert item["source_id"].startswith("web:")
        for item in ge["vid"]:
            assert item["source_id"].startswith("vid:")

    def test_ask_citations_have_source_type(self, client):
        r = client.post(
            "/v1/ask",
            json={"query": "pediatric asthma", "k": 3, "method": "bm25"},
        )
        data = r.json()
        for c in data["citations"]:
            assert "source_type" in c
            assert c["source_type"] in ("doc", "web", "vid", "struct")

    def test_existing_ask_behavior_unchanged(self, client):
        r = client.post(
            "/v1/ask",
            json={"query": "62yo male chest pain hypertension", "k": 5, "method": "bm25"},
        )
        assert r.status_code == 200
        data = r.json()
        # All pre-existing fields still present
        assert isinstance(data["answer"], str) and len(data["answer"]) > 10
        assert isinstance(data["citations"], list)
        assert data["method_used"] == "bm25"
        assert data["triage_level"] in {"NOW", "SOON", "WAIT"}
        assert "prediction_signal" in data
        assert "decision_basis" in data
        assert "esi_final" in data


# ── /vertex route ──────────────────────────────────────────────────────────

class TestVertexRoute:
    def test_vertex_renders_200(self, client):
        r = client.get("/vertex")
        assert r.status_code == 200

    def test_vertex_is_html(self, client):
        r = client.get("/vertex")
        assert "text/html" in r.headers["content-type"]

    def test_vertex_layout_intact(self, client):
        r = client.get("/vertex")
        html = r.text
        assert "ER Insight Console" in html
        assert 'id="query"' in html
        assert 'id="esiBlock"' in html
        assert 'id="citations"' in html
        assert 'id="nextActions"' in html

    def test_vertex_source_tag_uses_source_type(self, client):
        r = client.get("/vertex")
        assert "c.source_type" in r.text
