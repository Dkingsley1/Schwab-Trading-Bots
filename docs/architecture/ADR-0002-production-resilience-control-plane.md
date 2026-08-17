# ADR-0002: Production Resilience Control Plane

## Status

Accepted on 2026-08-10.

## Context

The paper soak must keep collecting through production-only evidence debt, while failures in runtime truth, storage, auth, order state, or monitoring must fail closed. Independent controls previously exposed strong detail but did not provide one enforceable contract for framework implementation, unattended paper operation, and live-promotion evidence.

## Decision

`config/production_resilience_v1.json` declares ten required hardening sections. `scripts/ops/production_resilience_control.py` evaluates each section from fresh owner-produced evidence and publishes three separate verdicts:

1. Framework implementation: all ten controls and owners exist.
2. Paper-soak readiness: every paper-critical control has fresh healthy evidence.
3. Live-promotion readiness: all ten sections have fresh production evidence.

The control never grants live execution authority. Explicit operator release remains required after every live-promotion section passes. Profitability remains probabilistic and raw economic grades cannot be rewritten by control grades.

## Consequences

- A dirty or unsynchronized release blocks live promotion without stopping healthy paper collection.
- Missing off-host heartbeat delivery remains visible as live-monitoring debt while the local deadman can still protect the soak.
- RPO, RTO, chaos, order-ledger, ownership, and profitability evidence become independently inspectable blockers.
- Duplicate writers, stale artifacts, repair loops without circuits, and cosmetic profitability grades fail their owning section.
- Generated dashboard artifacts remain evidence, not configuration or execution authority.
