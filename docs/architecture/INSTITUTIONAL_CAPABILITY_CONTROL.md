# Institutional Capability Control

## Purpose

`institutional_capability_control.py` is the single evidence-facing contract for six capabilities commonly visible in public descriptions of mature quantitative platforms. It does not claim that this local system is equivalent to an institutional firm. It records which local analogs are implemented, which are safe for guarded paper operation, which have current-candidate economic evidence, which depend on an external entitlement, and which are ready for live promotion.

The canonical policy is `config/institutional_capability_control_v1.json`. The runtime artifact is `governance/health/institutional_capability_control_latest.json`.

## Source-count policy

Two Sigma publicly describes a proprietary library containing more than 10,000 sources. That figure reflects the scale of a large organization, not a minimum viable source count for this repository. See [Two Sigma Investment Management](https://www.twosigma.com/businesses/investment-management/).

This system instead targets a compact set of roughly 15 to 30 authoritative provider families, admitted only when a strategy declares a real capability need. Shared snapshots may produce hundreds or thousands of point-in-time features, but a derived feature is not relabeled as a new source. Provider count never earns readiness or profitability credit by itself.

The admission order is:

1. Reuse an existing fresh, authoritative shared snapshot.
2. Derive a point-in-time feature with source lineage and no future leakage.
3. Add an independent corroborating provider when the decision risk justifies it.
4. Buy a direct or licensed feed only when an activated strategy family requires it and the expected value supports the cost.

## Six pillars

| Pillar | Local implementation | Candidate/live evidence that remains separate |
| --- | --- | --- |
| Scientific research platform | complete strategy contracts, point-in-time store, signed append-only experiment ledger, quantitative challengers, multiple-testing guard | validated candidate-forward strategies, exact replay bundles, independent attestation |
| Market visibility and lineage | verified source bundles, 25 logical planes, shared producer routing, capability receipts | live-ready routes, direct depth where required, independent failover, zero required capability debt |
| Independent execution evidence | realistic execution simulator, content-addressed independent-fill intake, exact candidate binding, calibration report | enough broker-paper or licensed replay fills, live broker reconciliation |
| Selection-bias and overfit control | complete registered-family floor, compressed immutable-ledger ingestion, FDR, deflated Sharpe, PBO | passing candidate-forward p-values and aligned periods, exact replay, independent review |
| Resource routing and role separation | autonomic budgets, performance-core-primary bounded work, declared system roles, exclusive mutable-state ownership | sustained runtime proof under target load; no cosmetic core pinning claim |
| Market-access risk controls | isolated pre-trade service, execution budgets, kill switch, transactional live-order ledger, reconciliation boundary | evaluated order evidence, healthy operational inputs, canary clearance, explicit operator release |

Public references establish the capability categories, not local readiness:

- [Jane Street quantitative research](https://www.janestreet.com/quantitative-research/) describes large-data analysis, machine learning, model testing, strategies, and implementation.
- [Nasdaq TotalView](https://www.nasdaq.com/solutions/data/equities/nasdaq-totalview) documents full depth-of-book and auction imbalance data.
- [CME Market Data Platform](https://www.cmegroup.com/market-data/distributor/market-data-platform.html) documents direct futures and options dissemination through MDP 3.0.
- [AQR Trading Costs](https://www.aqr.com/Insights/Research/Working-Paper/Trading-Costs) discusses calibration from realized institutional execution data.
- [The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb) provides the research basis for explicit PBO controls.
- [SEC Rule 15c3-5](https://www.sec.gov/rules-regulations/2011/06/risk-management-controls-brokers-or-dealers-market-access) documents pre-set financial and regulatory market-access controls. It applies to covered broker-dealers; this project uses it as a design reference, not a claim of broker-dealer compliance.

## State semantics

Every pillar publishes four independent booleans:

- `implementation_ready`: the owner code and structural contract exist and are fresh.
- `paper_soak_ready`: guarded collection and paper operation may continue without the optional live entitlement.
- `candidate_evidence_ready`: current-candidate observations satisfy that pillar's evidence floor.
- `live_promotion_ready`: the live-specific evidence and human release requirements are satisfied.

`ready_with_evidence_debt` is a healthy paper state, not a live-ready state. `paper_attention` means at least one pillar cannot currently support guarded paper operation. `blocked` means the structural implementation is incomplete or stale.

Resource heat is evaluated by workload impact. External macOS or user-app heat remains visible and may pause widening or training, but it does not fail the paper pillar while required collection and read-only observation lanes remain available, paper execution is not the dominant pressure source, and memory pressure is normal. Paper-dominant pressure, paused required lanes, or elevated memory pressure fail the pillar.

## Entitlements

Nasdaq depth, CME direct depth, and licensed news, estimates, borrow, or corporate-action bundles are conditional. They do not block the general paper soak. They become live blockers only when an activated family declares the associated capability as required. Broker-paper receipts or licensed venue replay are different: they are evidence inputs and cannot be synthesized by the expected-fill model.

The controller may recommend bounded local refresh commands for stale artifacts. It cannot purchase data, enter credentials, manufacture fills, issue an attestation, admit a strategy, alter the candidate, or unlock live execution.

## Operator commands

```bash
./scripts/ops/opsctl.sh institutional-capability-control --json
./scripts/ops/opsctl.sh multiple-testing --json
.venv314/bin/python scripts/ops/independent_fill_evidence_acquisition.py --apply --json
.venv314/bin/python scripts/paper_execution_calibration_report.py --json
```

The livefeed `[institutional-capabilities]` row reports the four six-pillar counts, verified source-bundle count, provider-family target, conditional entitlements, candidate binding, bounded refresh count, and external actions. It has no order authority and does not turn organic evidence debt into a runtime outage.
