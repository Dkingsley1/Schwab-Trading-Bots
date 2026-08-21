# Authoritative Systems Control

## Purpose

This control translates public primary-source patterns from 20 authoritative systems and standards into eight locally owned, executable contracts. The references are design inputs, not dependencies, alpha claims, endorsements, or readiness points.

The canonical registry is `config/authoritative_systems_v1.json`. Run:

```bash
./scripts/ops/opsctl.sh authoritative-systems --json
```

## Eight Enforced Controls

1. Broker capability conformance separates market data, simulated paper, native live, asset classes, order types, time in force, precision, rate-limit pools, and production eligibility.
2. The durable order state machine enforces idempotent reservation, legal transitions, immutable broker identity, monotonic fills, terminal-state immutability, and reconciliation after ambiguous submission.
3. Point-in-time validity combines high-confidence AST rules, future-suffix invariance, and recursive warmup stability. Failure blocks candidate promotion and live execution.
4. Event-time watermarks accept bounded out-of-order events and quarantine late, future-skewed, duplicate-conflicting, or invalid events. HTTP success and event usability remain separate facts.
5. Causal attribution records observed signal, sizing, risk, execution, cost, and outcome fields. Unavailable values remain null rather than becoming invented zeros.
6. Paper/live equivalence compares mode-invariant order intent while allowing fills, fees, slippage, latency, broker IDs, and venue status to differ.
7. Deterministic fault scenarios cover normal fill, latency stress, interrupted submission, duplicate intent, progressive partial fill, global halt, price gap, and cancel/fill race.
8. End-to-end traceability carries a deterministic trace ID and eight hash-linked stage receipts from source through outcome in existing execution artifacts.

## Readiness Semantics

An `A+` from this control means only that all eight local structural implementations pass. It does not prove profitability, satisfy candidate-bound forward runtime, validate paper/live equivalence without observed pairs, authorize live orders, or replace independent promotion controls.

The change is classified as additive production hardening. Existing soak segments remain documented and the soak clock is not reset, but post-change observation is still required for the new behavior.

`scripts/paper_live_equivalence_report.py` continuously compares observed paper intents with promoted live-shadow intents. No live-shadow samples produce `awaiting_live_shadow_samples`: this is live-evidence debt with `paper_impact=none`, not a paper-trading failure or permission to submit live orders.

## Public Primary References

The registry includes LEAN, NautilusTrader, Qlib, Hummingbot, Freqtrade, vn.py, Zipline Reloaded, ABIDES, FinRL, QuantRocket, FIX Trading Community, Apache Flink, Apache Kafka, Temporal, MLflow, Feast, OpenLineage, OpenTelemetry, SEC Rule 15c3-5, and NIST CSF 2.0. Each record contains its official URL, adopted local controls, and the exact principles used.
