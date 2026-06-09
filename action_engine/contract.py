"""Canonical Evidence Contract — the stable, versioned boundary into L2.

The action engine never reads a source-specific shape (openFDA fields, raw
FHIR, a synthetic fixture row). It only reads a `CanonicalEvidence`. A source
adapter (see `adapters.py`) is the *only* place allowed to know the source
shape; it must emit this contract or fail.

`validate_evidence` is the Eyes/intake stage. It FAILS CLOSED: a malformed or
incompatible record is rejected here, before the decision engine is ever
called. That is what makes "invalid contract blocks before decision" a real
property and not a comment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Versioned on purpose: a future governed-L1 adapter must declare the same
# version, and the version is carried into every durable row + the audit
# artifact so evidence lineage stays visible.
CONTRACT_VERSION = "evidence.v1"

_TRIAGE_LEVELS = {"NOW", "SOON", "WAIT"}
_RISK_LEVELS = {"low", "medium", "high"}


@dataclass(frozen=True)
class ERStateView:
    """Live ER operational context, contract-normalized (not source-shaped)."""

    available_beds: Optional[int] = None
    occupancy_pct: Optional[float] = None
    queue_length: Optional[int] = None


@dataclass(frozen=True)
class CanonicalEvidence:
    """The stable contract the decision/action engine consumes.

    `correlation_id` threads every downstream task/action/handoff/outcome.
    `source` + `source_id` + `contract_version` are the lineage the audit
    artifact must be able to show.
    """

    contract_version: str
    correlation_id: str
    source: str
    source_id: str
    ingested_at: str
    triage_level: str
    predicted_los_hours: float
    bed_pressure_risk: str
    er_state: ERStateView = field(default_factory=ERStateView)

    def lineage(self) -> dict[str, str]:
        return {
            "contract_version": self.contract_version,
            "correlation_id": self.correlation_id,
            "source": self.source,
            "source_id": self.source_id,
            "ingested_at": self.ingested_at,
        }


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    evidence: Optional[CanonicalEvidence]
    errors: list[str]


def _as_optional_int(value: Any, label: str, errors: list[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an int or null")
        return None
    if value < 0:
        errors.append(f"{label} must be >= 0")
        return None
    return value


def _as_optional_float(value: Any, label: str, errors: list[str], lo: float, hi: float) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} must be a number or null")
        return None
    if not (lo <= float(value) <= hi):
        errors.append(f"{label} must be within [{lo}, {hi}]")
        return None
    return float(value)


def validate_evidence(raw: dict[str, Any]) -> ValidationResult:
    """Validate a contract record and FAIL CLOSED on any problem.

    Returns a `ValidationResult`; `.ok is False` means the decision engine must
    never run for this record. The errors list is durable evidence of *why* a
    record was blocked.
    """
    errors: list[str] = []

    if not isinstance(raw, dict):
        return ValidationResult(False, None, ["evidence must be a JSON object"])

    version = raw.get("contract_version")
    if version != CONTRACT_VERSION:
        errors.append(
            f"contract_version must be {CONTRACT_VERSION!r}, got {version!r}"
        )

    correlation_id = raw.get("correlation_id")
    if not isinstance(correlation_id, str) or not correlation_id.strip():
        errors.append("correlation_id must be a non-empty string")

    source = raw.get("source")
    if not isinstance(source, str) or not source.strip():
        errors.append("source must be a non-empty string")

    source_id = raw.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        errors.append("source_id must be a non-empty string")

    ingested_at = raw.get("ingested_at")
    if not isinstance(ingested_at, str) or not ingested_at.strip():
        errors.append("ingested_at must be a non-empty string")

    triage_level = raw.get("triage_level")
    if triage_level not in _TRIAGE_LEVELS:
        errors.append(f"triage_level must be one of {sorted(_TRIAGE_LEVELS)}")

    los = raw.get("predicted_los_hours")
    if isinstance(los, bool) or not isinstance(los, (int, float)):
        errors.append("predicted_los_hours must be a number")
        los = None
    elif los < 0:
        errors.append("predicted_los_hours must be >= 0")
        los = None

    bed_pressure_risk = raw.get("bed_pressure_risk")
    if bed_pressure_risk not in _RISK_LEVELS:
        errors.append(f"bed_pressure_risk must be one of {sorted(_RISK_LEVELS)}")

    raw_er = raw.get("er_state") or {}
    if not isinstance(raw_er, dict):
        errors.append("er_state must be a JSON object or null")
        raw_er = {}
    er = ERStateView(
        available_beds=_as_optional_int(raw_er.get("available_beds"), "er_state.available_beds", errors),
        occupancy_pct=_as_optional_float(raw_er.get("occupancy_pct"), "er_state.occupancy_pct", errors, 0.0, 100.0),
        queue_length=_as_optional_int(raw_er.get("queue_length"), "er_state.queue_length", errors),
    )

    if errors:
        return ValidationResult(False, None, errors)

    evidence = CanonicalEvidence(
        contract_version=version,
        correlation_id=correlation_id,
        source=source,
        source_id=source_id,
        ingested_at=ingested_at,
        triage_level=triage_level,
        predicted_los_hours=float(los),
        bed_pressure_risk=bed_pressure_risk,
        er_state=er,
    )
    return ValidationResult(True, evidence, [])
