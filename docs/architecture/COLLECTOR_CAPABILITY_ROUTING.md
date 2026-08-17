# Collector Capability Routing

## Purpose

The collector capability layer gives every organized bot a deterministic, inspectable data subscription without creating one collector process per bot. It separates three things that must not be conflated:

1. A logical capability describes data or evidence a bot may consume.
2. A physical producer publishes one shared, bounded snapshot that may satisfy many capabilities.
3. A bot binding points to a content-addressed shared profile instead of duplicating fetch work.

The layer is observation and metadata only. It cannot fetch external data, launch collectors, change a decision, mutate the bot registry, rewrite history, place a paper or live order, promote a bot, or guarantee profitability.

## Canonical Sources

- Catalog and routing policy: `config/collector_capability_catalog_v1.json`
- Materialization policy: `config/capability_materialization_v1.json`
- Derivative contract source: `config/derivatives_contract_master_v1.json`
- Versioned stress sources: `config/stress_scenarios/*.json`
- Source-backed materializer: `core/capability_materialization.py`
- Materialization writer: `scripts/ops/capability_materialization_control.py`
- Validator and deterministic router: `core/collector_capability_routing.py`
- Health and artifact writer: `scripts/ops/collector_capability_control.py`
- Physical producer contracts: `scripts/collector_contracts.py`
- Bot assignment input: `governance/bot_organization/bot_hierarchy_latest.json`
- Health evidence: `governance/health/collector_capability_control_latest.json`
- Direct proof evidence: `governance/collector_capabilities/materialized_capabilities_latest.json`
- Shared subscription artifact: `governance/collector_capabilities/bot_subscriptions_latest.json`

Generated artifacts are evidence, not source. Change the catalog or router first and regenerate them with:

```bash
./scripts/ops/opsctl.sh bot-organization --json
./scripts/ops/opsctl.sh collector-contracts --include-data-plane --json
./scripts/ops/opsctl.sh capability-materialization --json
./scripts/ops/opsctl.sh collector-capability-control --json
./scripts/ops/opsctl.sh provider-mesh --json
```

## Data Planes

The v1 catalog contains 25 planes and at least 250 stable capability IDs:

1. Instrument identity.
2. Price and tape.
3. Liquidity and microstructure.
4. Options and volatility.
5. Futures and forwards.
6. Company fundamentals.
7. Earnings and expectations.
8. Corporate actions.
9. Ownership and positioning.
10. Growth and inflation.
11. Monetary and fiscal policy.
12. Rates, credit, and funding.
13. FX and commodities.
14. Cross-asset state.
15. News and events.
16. Broker and account truth.
17. Order and fill truth.
18. Execution quality.
19. Portfolio risk.
20. Accounting and tax.
21. Point-in-time training.
22. Research evidence.
23. Profitability evidence.
24. Data governance.
25. Operational health.

## Routing Contract

Each plane declares target scopes, roles, and relevance tokens. Scope and role requirements establish the minimum profile. Relevant planes add bounded required and optional capabilities. Unknown regime axes receive a separate runtime-context profile so observation access is available without inventing a bot preference.

Profiles are canonicalized and content addressed. Identical subscriptions share one profile. Physical producers remain independently scheduled by their existing owners, publish shared snapshots, and use their own freshness, fallback, cache, and failure-isolation contracts.

Producer health alone is not sufficient for capabilities that declare a field-level proof. The router verifies exact payload paths or a capability-specific direct receipt, publishes the selected producer and every usable failover, and reports required-capability redundancy separately from availability. This prevents a healthy producer from implicitly claiming fields it did not publish.

## Failure Semantics

The control fails closed when the catalog is invalid, a current collector is unmapped, hierarchy assignments are missing, any bot lacks a binding, or an authority flag is enabled. A current required collector failure blocks guarded paper-soak readiness.

Unsupported and temporarily unavailable logical capabilities are reported as explicit coverage debt. Capabilities required by the candidate block live promotion; optional catalog gaps are advisory and cannot veto an otherwise complete candidate. Full-catalog coverage remains visible as a separate research target. This distinction prevents both false implementation claims and false live blockers.

The four formerly hard gaps are materialized from concrete sources: `exchange-calendars` schedules and point-in-time session state, a versioned ten-root derivative contract master, and two versioned stress scenarios with source and content receipts. Each publishes direct proof semantics, a point-in-time timestamp, a content receipt, and zero execution or promotion authority.

## Soak Integration

The bounded readiness accrual order is:

`collector contracts -> source verification -> capability materialization -> capability routing -> provider mesh -> evidence and profitability refresh`

Unattended-soak readiness requires fresh `4/4` direct materialization receipts, a fresh routing artifact, complete current-collector mapping, complete bot binding coverage, a safe authority contract, and no current required collector failures. The runtime dashboard, live feed, self-awareness needs engine, artifact freshness SLO, daily refresh, runtime artifact refresh, source-mutation guard, and exclusive ownership registry all carry the same control.
