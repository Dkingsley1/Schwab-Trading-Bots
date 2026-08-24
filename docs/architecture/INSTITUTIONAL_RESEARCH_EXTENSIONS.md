# Institutional Research Extensions

## Purpose

This layer turns useful public patterns from Point72/Cubist, AQR, Man AHL, Two Sigma, D. E. Shaw Research, and Goldman Sachs into eight locally owned controls. It does not claim access to proprietary firm systems, install their software, fetch their datasets, store credentials, alter trading decisions, or authorize orders.

The owning files are:

- `config/institutional_research_extensions_v1.json`
- `core/institutional_research_extensions.py`
- `scripts/ops/institutional_research_extensions_control.py`
- `tests/test_institutional_research_extensions.py`

Run the evaluator with:

```bash
./scripts/ops/opsctl.sh institutional-research-extensions --json
```

## Eight Controls

1. **Independent factor benchmarks** fit candidate-bound ridge factor diagnostics against cash, equity-market, value, momentum, quality, low-beta, and cross-asset-trend identities. An intercept is never labeled proven alpha, and factor exposure cannot admit a strategy.
2. **Pipeline incident ownership** requires an incident ID, pipeline, named owner, severity, detection time, acknowledgement, severity-specific SLOs, and root-cause and remediation receipts at closeout. Recurrence fingerprints expose repeated causes.
3. **Material strategy-change governance** classifies observability, data/research, strategy, and risk/execution changes. It emits required reviews, regressions, and affected-scope forward evidence windows. Cumulative soak history is preserved, but the changed scope must accrue honestly and live self-approval is forbidden.
4. **Candidate risk schedules** check fresh broker truth, candidate identity, gross/net exposure, symbol and sleeve concentration, drawdown, daily loss, buying-power use, asset types, and instructions. Existing pre-trade controls retain all actual risk and execution authority.
5. **Execution speed-cost frontiers** compare expected alpha decay, spread, impact, adverse selection, fees, participation, fill probability, and independent fill support. Thin or non-positive alternatives produce `abstain`, not a forced trade.
6. **Research DAG checkpoint/resume** topologically orders source, point-in-time feature, label, train, validation, cost-replay, and candidate-packet stages. A checkpoint is reusable only when candidate, code, dataset, and dependency receipts match; a stale ancestor invalidates every descendant. The planner cannot launch a process.
7. **Versioned research-dataset storage** creates immutable content-addressed manifests, verifies parent chains, and resolves historical versions by knowledge time. Source retirement remains a separate proof-gated action.
8. **Cross-engine valuation reconciliation** reuses the independent risk-oracle contract to compare request-bound product measures from distinct providers and models under per-measure tolerances. Synthetic probes and unsigned observations never become external promotion evidence.

Every decision-family route receives only a hash-bound metadata summary of these controls. The metadata is intentionally excluded from signal, quantity, risk-limit, paper-order, live-order, and promotion authority.

## Public Design Provenance

The policy records ten official references:

- Point72/Cubist: quantitative trade lifecycle, data reliability ownership, and public risk-governance disclosure.
- AQR: published and revision-aware factor research datasets.
- Man Group/AHL: versioned time-series design and the speed-versus-trading-cost tradeoff. ArcticDB remains `influence_only`; production adoption requires a separate license review.
- Two Sigma: reproducible engineering boundaries and explicit workflow call graphs.
- D. E. Shaw Research: immutable, historically addressable research-dataset versions.
- Goldman Sachs: explicit product, pricing, and risk-measure contracts. GS Quant connectivity and institutional credentials are not assumed.

These references are provenance, not endorsements, dependencies, readiness points, or evidence that the local implementation matches a firm's private platform.

## Evidence Semantics

The runtime artifact reports structural implementation and earned evidence separately. A structural `8/8 A+` proves that the local contracts and tests exist. It does not manufacture:

- candidate-forward factor observations;
- an exercised incident-response record;
- mature independent fills for a cost frontier;
- current candidate risk snapshots;
- completed DAG checkpoints;
- durable dataset versions;
- signed independent valuation observations; or
- profitability and live-release approval.

The evaluator is paper-safe and reports `paper_impact=none`, `reset_soak_clock=false`, and `live_execution_authority=false`. Accepted cumulative soak segments remain documented, while affected candidate scopes continue from their accepted change boundary.
