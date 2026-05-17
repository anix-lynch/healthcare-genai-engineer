"""Tests for the LLM-enhanced generation path (mocked — no real API call).

The default path (USE_LLM=false) is exercised by test_ask.py. This file
patches the provider call to verify the branching logic works without
needing ANTHROPIC_API_KEY or network access.
"""
from __future__ import annotations
from unittest.mock import patch

import pytest

from generation import generate as gen_module


def test_llm_branch_wires_when_provider_returns_text(monkeypatch):
    """USE_LLM=true + provider returns text → answer comes from LLM, method tagged."""
    monkeypatch.setattr(gen_module, "USE_LLM", True)
    monkeypatch.setattr(gen_module, "LLM_PROVIDER", "anthropic")

    fake_llm_output = "Per L1-000001, 62yo M chest pain matches the cardiac workup pattern."
    with patch.object(gen_module, "_call_anthropic", return_value=fake_llm_output):
        result = gen_module.generate_answer(
            "62yo male chest pain",
            [{"case_id": "L1-000001", "snippet": "62yo M HTN", "score": 5.2}],
        )

    assert result["answer"] == fake_llm_output
    assert result["generation_method"] == "llm_anthropic"
    assert "L1-000001" in result["citations"]


def test_llm_failure_falls_back_to_template(monkeypatch):
    """USE_LLM=true + provider returns None → template path used + warning surfaced."""
    monkeypatch.setattr(gen_module, "USE_LLM", True)
    monkeypatch.setattr(gen_module, "LLM_PROVIDER", "anthropic")

    with patch.object(gen_module, "_call_anthropic", return_value=None):
        result = gen_module.generate_answer(
            "test query",
            [{"case_id": "L1-000001", "snippet": "62yo M HTN", "score": 5.2}],
        )

    assert result["generation_method"] == "template"
    assert any("LLM (anthropic) unavailable" in w for w in result["warnings"])
    assert "L1-000001" in result["citations"]  # template still cites real source


def test_openai_provider_selection(monkeypatch):
    """LLM_PROVIDER=openai routes through the OpenAI call path."""
    monkeypatch.setattr(gen_module, "USE_LLM", True)
    monkeypatch.setattr(gen_module, "LLM_PROVIDER", "openai")

    with patch.object(gen_module, "_call_openai", return_value="openai response citing L1-000001") as mock_oa, \
         patch.object(gen_module, "_call_anthropic", return_value="should not be called") as mock_an:
        result = gen_module.generate_answer(
            "test query",
            [{"case_id": "L1-000001", "snippet": "test", "score": 1.0}],
        )

    assert mock_oa.called
    assert not mock_an.called
    assert result["generation_method"] == "llm_openai"


def test_default_path_skips_llm_entirely(monkeypatch):
    """USE_LLM=false (default) → template path, no provider import attempted."""
    monkeypatch.setattr(gen_module, "USE_LLM", False)

    with patch.object(gen_module, "_call_anthropic", return_value="should not be called") as mock_an:
        result = gen_module.generate_answer(
            "test query",
            [{"case_id": "L1-000001", "snippet": "test", "score": 1.0}],
        )

    assert not mock_an.called
    assert result["generation_method"] == "template"


def test_empty_hits_short_circuits():
    """No hits → empty-set message + warning, never calls LLM."""
    result = gen_module.generate_answer("anything", [])
    assert "empty_retrieval_set" in result["warnings"]
    assert result["citations"] == []
    assert result["generation_method"] == "template"
