"""Source adapters — the ONLY place allowed to know a source's raw shape.

The action engine consumes `CanonicalEvidence`. An adapter takes a
source-specific record and produces a contract record (a plain dict that
`validate_evidence` then checks). The synthetic fixture adapter is the first
adapter; a governed-L1 adapter can be added later behind the same interface
WITHOUT touching the decision/action/retry/handoff/outcome engine.

Boundary rule (from the L2 benchmark): no openFDA / FHIR / source-specific
field names may appear outside an adapter. Keep that true here.
"""
from __future__ import annotations

from typing import Any, Protocol

from action_engine.contract import CONTRACT_VERSION


class SourceAdapter(Protocol):
    """Maps one source-specific record to a canonical-contract dict.

    Returning a dict (not a `CanonicalEvidence`) is intentional: the adapter
    proposes, `validate_evidence` disposes. A buggy adapter that emits a bad
    shape is then caught by the same fail-closed intake as any other input.
    """

    name: str

    def to_contract(self, record: dict[str, Any]) -> dict[str, Any]:
        ...


class SyntheticFixtureAdapter:
    """First adapter: labelled synthetic ER/FHIR-style scenarios.

    The fixture rows are authored already in canonical field names, so this
    adapter is a thin pass-through that stamps source provenance + contract
    version. A real L1 adapter would do the actual field remapping here; the
    engine downstream cannot tell the difference, which is the whole point.
    """

    name = "synthetic_fixture"

    def to_contract(self, record: dict[str, Any]) -> dict[str, Any]:
        evidence = record.get("evidence", record)
        return {
            "contract_version": CONTRACT_VERSION,
            "correlation_id": record.get("correlation_id") or evidence.get("correlation_id"),
            "source": self.name,
            "source_id": record.get("id") or evidence.get("source_id") or record.get("correlation_id"),
            "ingested_at": evidence.get("ingested_at", "1970-01-01T00:00:00Z"),
            "triage_level": evidence.get("triage_level"),
            "predicted_los_hours": evidence.get("predicted_los_hours"),
            "bed_pressure_risk": evidence.get("bed_pressure_risk"),
            "er_state": evidence.get("er_state", {}),
        }


class RawPassthroughAdapter:
    """Escape hatch for already-canonical dicts (e.g. an API caller).

    Stamps the contract version + source but otherwise trusts the caller's
    field names. Still goes through `validate_evidence`, so it cannot bypass
    fail-closed intake.
    """

    name = "raw_passthrough"

    def to_contract(self, record: dict[str, Any]) -> dict[str, Any]:
        out = dict(record)
        out.setdefault("contract_version", CONTRACT_VERSION)
        out.setdefault("source", self.name)
        out.setdefault("source_id", record.get("correlation_id", ""))
        out.setdefault("ingested_at", "1970-01-01T00:00:00Z")
        return out
