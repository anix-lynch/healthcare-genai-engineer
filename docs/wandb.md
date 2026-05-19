# W&B / Weave Trace Model

## Overview

This project treats tracing as operational CCTV for the request pipeline.

The goal is not to collect pretty dashboards. The goal is to make it easy to
see:

- which stage ran
- which stage was slow
- which stage failed
- which stage fell back
- how upstream behavior affected downstream answer quality

## CCTV map for the healthcare workflow

- Front door camera: `input_guard`
- Triage room camera: `retrieve`
- Clinical review room camera: `classify_rule` + `classify_rag`
- Report-writing room camera: `generate`
- Exit security camera: `output_guard`
- Shift supervisor: `fuse`

## What a useful trace should answer

For a single request:

- Which stages executed?
- How long did each stage take?
- Which stage failed, degraded, or fell back?
- What inputs and outputs mattered?
- Where did confidence drop?

For the system as a whole:

- Which stage has the highest latency?
- Which stage fails most often?
- Which stage falls back most often?
- Which stage drives disagreement or quality loss?

## Surface-level telemetry vs operational telemetry

Surface-level telemetry usually means:

- bars move
- a trace link exists
- a dashboard looks active

Operational telemetry means you can explain:

- why retrieval latency spiked
- whether dense search improved quality enough to justify its cost
- how often rule-based and retrieval-based triage disagree
- when a request should escalate to human review
- whether a weak hit set caused a weak final answer

## One-line takeaway

Useful telemetry does not just prove that the system ran. It proves which node
helped, which node hurt, and why the final outcome looked the way it did.
