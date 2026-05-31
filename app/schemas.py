"""Request / response contracts for the FastAPI surface.

Pydantic models so OpenAPI docs are auto-generated and validation is free.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Literal


Method = Literal["bm25", "dense", "hybrid"]
RiskLevel = Literal["low", "medium", "high"]
TriageLevel = Literal["NOW", "SOON", "WAIT"]
SourceType = Literal["doc", "web", "vid", "struct"]
AgentId = Literal["er_triage", "bed_ops", "care_followup"]
RuntimeMode = Literal["cloud_run_24_7_stateless"]


class RetryPolicy(BaseModel):
    """Bounded retry contract for an action-agent handoff."""
    model_config = ConfigDict(extra="forbid")
    max_attempts: int = Field(..., ge=1, le=3)
    backoff_seconds: list[int] = Field(default_factory=list)
    retry_on: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)
    escalation: str


class ERState(BaseModel):
    """Optional live ER operational context."""
    model_config = ConfigDict(extra="forbid")
    queue_length: int | None = Field(None, ge=0)
    available_beds: int | None = Field(None, ge=0)
    occupancy_pct: float | None = Field(None, ge=0.0, le=100.0)
    avg_wait_minutes: int | None = Field(None, ge=0)


class PredictionSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_level: RiskLevel
    predicted_los_hours: float | None = Field(None, ge=0.0)
    deterioration_risk: RiskLevel
    bed_pressure_risk: RiskLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    recommended_operational_actions: list[str] = Field(default_factory=list)


class AgentHandoff(BaseModel):
    """One action-agent node in the post-triage collaboration graph."""
    model_config = ConfigDict(extra="forbid")
    agent_id: AgentId
    handoff_key: str = Field(..., description="deterministic idempotency key; prevents duplicate self-requeue loops")
    label: str
    role: str
    trigger: str
    receives: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    retry_policy: RetryPolicy
    # Populated only for nodes that actually EXECUTE (e.g. Bed Ops computes a
    # disposition from live ER state). None = a routing/descriptive node.
    output: dict[str, Any] | None = None
    executed: bool = False


class AgentCollaboration(BaseModel):
    """Human-readable multi-agent routing plan for the current case."""
    model_config = ConfigDict(extra="forbid")
    primary_agent: AgentId
    handoffs: list[AgentHandoff] = Field(default_factory=list)
    summary: str
    runtime_mode: RuntimeMode = "cloud_run_24_7_stateless"
    loop_guard: str = Field("no_self_requeue_duplicate_keys_or_unbounded_retries", description="how the graph avoids wall loops")
    max_graph_steps: int = Field(3, ge=1, le=5)
    dead_end_policy: str = "escalate_to_human_owner_after_retry_budget"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., min_length=1, description="free-text question to RAG over")
    k: int = Field(5, ge=1, le=50, description="number of hits to retrieve")
    method: Method = Field("bm25", description="retrieval strategy")
    er_state: ERState | None = Field(None, description="optional live ER operational context")


class GroundingEvidence(BaseModel):
    """Normalized evidence item with honest provenance for the /vertex console."""
    model_config = ConfigDict(extra="forbid")
    source_type: SourceType
    source_id: str = Field(..., description="prefixed: doc:L1-000001, struct:L1-000001, web:..., vid:...")
    snippet: str = Field(..., description="human-readable evidence preview")
    similarity: float = Field(..., ge=0.0, le=1.0)
    is_real: bool = Field(..., description="False = honest placeholder, adapter not yet implemented")
    provenance: str | None = Field(None, description="human-readable source description shown in UI")


class Citation(BaseModel):
    source_id: str
    snippet: str = Field(..., description="up to 200-char preview of the cited record")
    similarity: float = Field(..., ge=0, le=1)
    source_type: SourceType = Field("doc", description="evidence lane; drives source tag in /vertex UI")


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    answer: str = Field(..., description="grounded answer; every claim cites a source_id")
    citations: list[Citation] = Field(default_factory=list)
    method_used: Method
    retrieved_count: int = Field(..., ge=0)
    latency_ms: int = Field(..., ge=0)
    warnings: list[str] = Field(default_factory=list)
    # ESI tier classification — rule-based floor + RAG-KNN refinement + fuse.
    # See workflows/classify_esi.py for the design rationale.
    esi_rule_based: int | None = Field(None, ge=1, le=5, description="rule-only ESI tier (text keyword + safety floors)")
    esi_rag_knn: int | None = Field(None, ge=1, le=5, description="RAG-KNN ESI tier (top-K esi_tier_truth weighted vote)")
    esi_final: int | None = Field(None, ge=1, le=5, description="fused ESI tier (rule floor wins if fires, else RAG)")
    esi_confidence: float | None = Field(None, ge=0.0, le=1.0, description="fused confidence ∈ [0,1] (1.0 = rule+RAG agree)")
    esi_disagreement: bool = Field(False, description="True iff rule fired AND RAG predicted a different tier")
    esi_red_flags: list[str] = Field(default_factory=list, description="rule-based safety triggers that fired")
    esi_votes: dict[int, int] = Field(default_factory=dict, description="per-tier raw vote count from top-K retrieved cases (e.g. {2:4, 3:1} → '4× ESI 2 · 1× ESI 3')")
    triage_level: TriageLevel | None = Field(None, description="human-readable urgency bucket derived from fused ESI")
    prediction_signal: PredictionSignal | None = Field(None, description="future-risk signal used for operational planning")
    decision_basis: list[str] = Field(default_factory=list, description="ordered list of facts/rules/signals that drove the decision")
    override_applied: bool = Field(False, description="True iff safety/override logic had to resolve a conflict or add explicit monitoring behavior")
    override_reason: str | None = Field(None, description="why the override logic fired")
    operational_recommendations: list[str] = Field(default_factory=list, description="human-readable operational next actions")
    explanation_for_human: str | None = Field(None, description="plain-English explanation of how triage and prediction were combined")
    agent_collaboration: AgentCollaboration | None = Field(None, description="multi-agent handoff plan: triage owner plus downstream action agents")
    # Per-node trace timings (ms). Populated on every request for the live trace panel.
    guard_ms: int = Field(0, ge=0, description="input-guardrail latency ms")
    retrieve_ms: int = Field(0, ge=0, description="retrieval latency ms")
    generate_ms: int = Field(0, ge=0, description="generation latency ms")
    # Guard telemetry — PII-only on the 200 path. Injection / length / empty
    # patterns short-circuit with HTTP 400 (detail.guard_type names which
    # rule fired) and never reach this response model. So on a 200 response,
    # guard_triggered == (guard_type == "pii"). Kept as two fields so a
    # future non-PII soft trigger can populate guard_type without breaking
    # the boolean contract.
    guard_triggered: bool = Field(False, description="True iff a soft guard fired on the 200 path (currently only PII redaction).")
    guard_type: Literal["none", "pii"] = Field("none", description="Which soft guard fired on the 200 path. Injection/length/empty are HTTP 400 — see error detail.guard_type.")
    # Weave call ID for this request — enables deep-link to the trace tree.
    # None when Weave is not initialized (no WANDB_API_KEY) or capture failed.
    trace_call_id: str | None = Field(None, description="Weave call ID for this request's root trace; pair with WEAVE_PROJECT to build a deep link.")
    # Grouped grounding evidence for the /vertex ER Insight Console.
    # Keys: "doc", "struct", "web", "vid". Empty list = lane not yet supported.
    # is_real=False entries are honest placeholders — the UI must not fake grounding.
    grouped_evidence: dict[str, list[GroundingEvidence]] = Field(
        default_factory=dict,
        description="evidence grouped by source_type lane; drives /vertex source provenance panel",
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
