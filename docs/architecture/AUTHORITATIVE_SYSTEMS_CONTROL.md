# Authoritative Systems Control

## Purpose

This control translates public primary-source patterns from 39 authoritative systems, firms, and standards into 18 locally owned, executable contracts. The references are design inputs, not dependencies, alpha claims, endorsements, or readiness points.

The canonical registry is `config/authoritative_systems_v1.json`. Run:

```bash
./scripts/ops/opsctl.sh authoritative-systems --json
```

## Eighteen Enforced Controls

1. Broker capability conformance separates market data, simulated paper, native live, asset classes, order types, time in force, precision, rate-limit pools, and production eligibility.
2. The durable order state machine enforces idempotent reservation, legal transitions, immutable broker identity, monotonic fills, terminal-state immutability, and reconciliation after ambiguous submission.
3. Point-in-time validity combines high-confidence AST rules, future-suffix invariance, and recursive warmup stability. Failure blocks candidate promotion and live execution.
4. Event-time watermarks accept bounded out-of-order events and quarantine late, future-skewed, duplicate-conflicting, or invalid events. HTTP success and event usability remain separate facts.
5. Causal attribution records observed signal, sizing, risk, execution, cost, and outcome fields. Unavailable values remain null rather than becoming invented zeros.
6. Paper/live equivalence compares mode-invariant order intent while allowing fills, fees, slippage, latency, broker IDs, and venue status to differ.
7. Deterministic fault scenarios cover normal fill, latency stress, interrupted submission, duplicate intent, progressive partial fill, global halt, price gap, and cancel/fill race.
8. End-to-end traceability carries a deterministic trace ID and eight hash-linked stage receipts from source through outcome in existing execution artifacts.
9. Exchange sequence integrity applies explicit channel, session, contiguous-sequence, duplicate-conflict, gap-recovery, and reset semantics without claiming native ITCH or OUCH connectivity.
10. Atomic archive snapshots hash manifests, compare-and-swap the catalog head, preserve readable lineage, and require replacement verification before source retirement. The structural probe does not migrate production storage.
11. Formal safety specification mirrors order invariants in a bounded executable Python state model and `formal/OrderSafety.tla`. A passing local probe is not a TLC model-check receipt or independent formal review.
12. Build provenance emits and verifies SLSA-shaped subject and material digests. Unsigned local provenance remains untrusted and cannot count as CI or promotion evidence.
13. Canonical trade lifecycle enforces immutable product economics and hash-linked execution, confirmation, allocation, settlement, cancellation, exercise, and expiration transitions using FINOS CDM event-model principles.
14. Independent pricing and risk reconciliation compares a request-bound local result with a distinct provider and per-measure tolerances. A synthetic oracle probe is permanently ineligible as external evidence.
15. Constrained portfolio advice produces multi-period candidate-bound weights only after at least four mature, low-correlation sleeves clear post-cost evidence. Cash, concentration, and turnover bounds are mandatory, and the result cannot create orders.
16. Declarative data-quality checkpoints enforce schema, type, range, uniqueness, monotonicity, future-time, and freshness expectations with quarantine or block actions. The Python 3.14 implementation adopts Great Expectations semantics without requiring an unsupported runtime dependency.
17. The research data platform unifies canonical products, permitted use, point-in-time queries, bitemporal revisions, alpha lifecycle evidence, source value, portfolio advice, simulation semantics, feed SLOs, and reproducibility. Its ten internal contracts are metadata and research controls with no trading authority.
18. The institutional research extension contract joins factor benchmarks, pipeline incident ownership, material-change governance, candidate risk schedules, execution speed-cost frontiers, checkpointable research DAGs, immutable dataset versions, and cross-engine valuation. Its eight internal controls are advisory-only and report structural readiness separately from candidate and external evidence.

## Readiness Semantics

An `A+` from this control means only that all 18 local structural implementations pass. It does not prove profitability, satisfy candidate-bound forward runtime, validate paper/live equivalence without observed pairs, authorize live orders, or replace independent promotion controls.

The artifact publishes an additional external-evidence section. Native exchange-protocol observations, production archive snapshots, TLC and independent-review receipts, signed trusted-builder attestations, external CDM interoperability, signed independent oracle observations, candidate portfolio outcomes, optional Great Expectations runtime validation, candidate-bound research-data-platform evidence, and institutional-extension evidence are all false until independently observed. That evidence count does not reduce the structural grade or disrupt paper collection, but it cannot be relabeled as complete.

The change is classified as additive production hardening. Existing soak segments remain documented and the soak clock is not reset, but post-change observation is still required for the new behavior.

`scripts/paper_live_equivalence_report.py` continuously compares observed paper intents with promoted live-shadow intents. No live-shadow samples produce `awaiting_live_shadow_samples`: this is live-evidence debt with `paper_impact=none`, not a paper-trading failure or permission to submit live orders.

## Public Primary References

The registry includes LEAN, NautilusTrader, Qlib, Hummingbot, Freqtrade, vn.py, Zipline Reloaded, ABIDES, FinRL, QuantRocket, FIX Trading Community, Apache Flink, Apache Kafka, Temporal, MLflow, Feast, OpenLineage, OpenTelemetry, SEC Rule 15c3-5, NIST CSF 2.0, Nasdaq ITCH/OUCH specifications, Apache Iceberg, TLA+, SLSA, FINOS Common Domain Model, OpenGamma Strata, CVXPortfolio, Great Expectations, Trexquant, and ten official references from Point72/Cubist, AQR, Man Group/AHL, Two Sigma, D. E. Shaw Research, and Goldman Sachs. Each record contains its official URL, adopted local controls, and exact adopted principles. Public firm material is influence-only and is never represented as proprietary replication.
