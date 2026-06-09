"""L2 Action Engine — one complete durable Bed Ops action loop.

This package is the Phase-1 vertical action spine. It is deliberately separate
from the retrieval/RAG/UI code in `app/` and `retrieval/`: the only thing it
borrows from the existing system is the *decision* (`app.bed_ops_agent.
decide_bed_disposition`) so that "evidence caused the decision" is the real,
already-tested behavior rather than a reimplementation.

The loop proves the L2 control loop end to end:

    canonical evidence (validated contract)
      -> existing Bed Ops decision
      -> durable action task created            (SQLite row)
      -> worker acknowledges ownership          (separate state transition)
      -> idempotent state-changing tool         (DB UNIQUE key)
      -> before/after state receipt             (durable receipt row)
      -> outcome verifier re-reads state        (never trusts tool ok)
      -> transient failure retries within budget
      -> exhausted failure -> durable human escalation task
      -> every stage logged for trace reconstruction

Nothing here uses an LLM: a clinical capacity protocol is deterministic by
design, which is exactly what makes the action loop testable.
"""
from __future__ import annotations

from action_engine.contract import CONTRACT_VERSION, CanonicalEvidence, validate_evidence
from action_engine.loop import LoopResult, run_action_loop
from action_engine.store import ActionStore

__all__ = [
    "CONTRACT_VERSION",
    "CanonicalEvidence",
    "validate_evidence",
    "ActionStore",
    "LoopResult",
    "run_action_loop",
]
