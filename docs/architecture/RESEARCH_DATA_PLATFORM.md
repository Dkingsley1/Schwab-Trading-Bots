# Research Data Platform

## Purpose

The research data platform gives collection, research, simulation, and sleeve decisions one bounded metadata contract. It does not fetch data, change signals, size positions, allocate capital, promote candidates, purchase sources, or submit orders.

The canonical policy is `config/research_data_platform_v1.json`. The implementation is `core/research_data_platform.py`; the runtime evaluator is `scripts/ops/research_data_platform_control.py`.

Run:

```bash
./scripts/ops/opsctl.sh research-data-platform --json
```

## Ten Contracts

1. **Canonical catalog:** every data product has a stable ID, owner, schema, source lineage, natural key, temporal fields, consumers, freshness SLO, and storage class.
2. **License and entitlement registry:** declared uses are allowlisted, prohibited uses fail closed, restricted products require explicit entitlements and terms-review receipts, and credentials are never stored in the catalog.
3. **Point-in-time research API:** a query binds datasets, consumer, purpose, decision time, valid time, catalog versions, and a deterministic receipt.
4. **Bitemporal revisions:** effective time and knowledge time are separate; only revisions knowable at the requested decision time may be selected.
5. **Alpha lifecycle:** hash-linked evidence gates every transition from raw feature through candidate, validated alpha, sleeve strategy, and portfolio candidate. Skipped states fail closed.
6. **Source-value accounting:** source quality, freshness, availability, incremental information, post-cost contribution, and nonredundancy are measured separately. Source count is never alpha.
7. **Portfolio alpha combination:** candidate-bound qualified sleeves delegate to the existing constrained multi-period advisory under cash, concentration, turnover, correlation, capacity, and fill-evidence floors.
8. **Unified simulation semantics:** research, backtest, replay, shadow, paper, live-shadow, and live representations share event identity while retaining explicit mode-specific fill and latency fields. A live representation has no submission authority.
9. **Per-feed service levels:** freshness, completeness, validity, availability, and correction ratios are evaluated per product. Missing observations are unknown, never green.
10. **Research reproducibility:** candidate, code, dataset, parameter, label, cost, and result materials are content hashed into a local receipt. A local receipt is not external attestation.

## Runtime Binding

`core/collector_capability_routing.py` maps each sleeve route's existing producers to relevant data products. The point-in-time feature store and candidate-outcome evidence products are mandatory for every one of the 15 decision families. The bounded product IDs and catalog receipt flow through `core/institutional_decision_flow.py` into decision summaries and traces as metadata only.

Existing paper routes retain their current authority. New consumers must call the entitlement-aware query contract before reading a restricted product. This staged adoption avoids silently interrupting the active soak while preventing future undeclared integrations.

## Evidence Semantics

The control reports two separate scores:

- **Implementation:** whether all ten local contracts and regression probes pass.
- **Earned evidence:** whether current runtime products, terms reviews, point-in-time coverage, revision history, candidate lifecycle, source-value outcomes, portfolio advice, observed mode pairs, feed SLOs, and reproducibility receipts exist.

An implementation `A+` is not profitability evidence or live-money readiness. Evidence debt remains visible and accrues forward without rewriting history. The change is additive research governance, preserves cumulative soak history, does not reset the main soak clock, and does not change signal or order semantics.

## Public Influence

Trexquant's public overview and careers material describe a directional sequence from data variables to alpha signals, proprietary simulation, strategies, and market-neutral portfolios, plus feed validity, timeliness, lineage, event orchestration, and observability. Those public signals influenced this local design; they do not expose Trexquant's proprietary architecture or code.

- `https://trexquant.com/`
- `https://trexquant.com/careers/13A32AB65C`
- `https://trexquant.com/careers/6D8F95262F`
