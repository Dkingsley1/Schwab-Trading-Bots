# Collector Capability Routing

## Purpose

The collector capability layer gives every organized bot a deterministic, inspectable data subscription without creating one collector process per bot. It separates three things that must not be conflated:

1. A logical capability describes data or evidence a bot may consume.
2. A physical producer publishes one shared, bounded snapshot that may satisfy many capabilities.
3. A bot binding points to a content-addressed shared profile instead of duplicating fetch work.

The layer is observation and metadata only. It cannot fetch external data, launch collectors, change a decision, mutate the bot registry, rewrite history, place a paper or live order, promote a bot, or guarantee profitability.

## Canonical Sources

- Logical catalog: `config/collector_capability_catalog_v1.json`
- Decision-aligned routing policy: `config/sleeve_ingestion_routing_v2.json`
- Sleeve decision families: `config/institutional_decision_flow_v1.json`
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
./scripts/ops/opsctl.sh sleeve-ingestion-production-control --apply --json
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

The v2 router resolves each bot and runtime sleeve to one of the same 15 families used by the institutional decision flow. Each family owns an explicit core, deferred, and cold data plan, cadence, degradation policy, paper-required capability set, live-required enrichment set, and bounded optional set. Scope alone cannot subscribe a bot to an entire plane.

Paper and live consume the same route definition but have different evidence floors. Paper needs the capabilities required to form and qualify a bounded decision. Live additionally needs the full family enrichment set, the higher route-quality floor, and independent failover evidence. Missing live enrichment remains promotion debt and does not stop unrelated collection or qualified paper activity.

Profiles are canonicalized and content addressed. Identical subscriptions share one profile. Physical producers remain independently scheduled by their existing owners, publish shared snapshots, and use their own freshness, fallback, cache, and failure-isolation contracts.

Producer health alone is not sufficient for capabilities that declare a field-level proof. The router verifies exact payload paths or a capability-specific direct receipt, scores eligible producers using authority, collector quality, freshness, proof, source coverage, error budget, and payload integrity, then publishes the selected producer and independent failure-domain failovers. Every bot binding, delivery route, runtime binding, and bounded decision-route summary is receipt-bound.

The shared transport contract applies to synchronous and bounded asynchronous collection. It enforces response-size limits, transient-only retries, `Retry-After`, URL query redaction, payload digests, watermarks, dead letters, request IDs, route IDs, capability IDs, and signed transport receipts. The async facade reuses this canonical implementation under a concurrency semaphore so the two paths cannot drift.

## Failure Semantics

The control fails closed when either policy is invalid, decision families do not align, a current collector is unmapped, hierarchy assignments are missing, any bot or runtime sleeve lacks a signed binding, transport safeguards are incomplete, or an authority flag is enabled. A current required collector failure blocks guarded paper-soak readiness.

Unsupported and temporarily unavailable logical capabilities are reported as explicit coverage debt. Capabilities required by the candidate block live promotion; optional catalog gaps are advisory and cannot veto an otherwise complete candidate. Full-catalog coverage remains visible as a separate research target. This distinction prevents both false implementation claims and false live blockers.

The four formerly hard gaps are materialized from concrete sources: `exchange-calendars` schedules and point-in-time session state, a versioned ten-root derivative contract master, and two versioned stress scenarios with source and content receipts. Each publishes direct proof semantics, a point-in-time timestamp, a content receipt, and zero execution or promotion authority.

## Soak Integration

The bounded readiness accrual order is:

`collector contracts -> source verification -> capability materialization -> capability routing -> provider mesh -> evidence and profitability refresh`

Unattended-soak readiness requires fresh `4/4` direct materialization receipts, a fresh routing artifact, complete current-collector mapping, complete bot and runtime binding coverage, valid route receipts, a complete transport contract, a safe authority contract, and no current required collector failures. Runtime decisions carry a bounded route summary into the institutional trace and operator summary. The live feed reports fleet route coverage and per-decision route quality separately.

The runtime artifact graph first attempts bounded refreshes for route-critical optional source context, then refreshes collector contracts and capability routes before reapplying `sleeve_ingestion_production_control_v2`. Collector contracts publish their exact owner commands, so a stale route names a concrete repair instead of only reporting degradation. A failed optional provider remains isolated to the affected route; it cannot stop unrelated collection, and it cannot be treated as live-ready evidence.

This order keeps route enforcement and the signed policy receipt current without resetting cumulative soak history. A changed accepted candidate scope still starts its own honest clean evidence window.
