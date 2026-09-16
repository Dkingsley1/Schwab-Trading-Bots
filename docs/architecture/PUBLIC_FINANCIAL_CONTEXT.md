# Public Financial Context

## Purpose

`public_financial_context_v1` adds bounded official public evidence to paper decisions, training, replay, and research. It classifies evidence before routing it so bots receive context relevant to their decision family instead of a system-wide broadcast. It has no action, sizing, allocation, paper-order, live-order, promotion, candidate-mutation, history-rewrite, legal-opinion, or profitability-guarantee authority.

## Official Sources

| Source | Evidence | Scope | Cadence |
| --- | --- | --- | --- |
| SEC XBRL Company Facts | financial statements, quality, profitability, cash flow, leverage, and distress inputs | issuer symbol | filing driven |
| Office of Financial Research Financial Stress Index | aggregate, credit, funding, safe-asset, volatility, and equity-valuation stress | financial system | daily with publication lag |
| FDIC BankFind failures API | bank-failure count and failed-asset severity | U.S. banking system | event driven |
| Federal Register API | financial-agency document activity and higher-impact event context | U.S. policy and regulation | business daily |
| ECB Data Portal API | euro short-term rate level, five-day change, and funding pressure | euro-area money market | business daily |
| New York Fed Primary Dealer Statistics | dealer Treasury/corporate inventory, repo balance, and Treasury financing fails | U.S. dealer and collateral system | weekly with reporting lag |
| FDIC BankFind financials API | aggregate assets, deposits, loans, noncurrent loans, lending intensity, and annual balance-sheet growth | U.S. insured banking system | quarterly |

The Federal Register stream is informational policy context, not legal truth. Public availability also does not establish commercial redistribution or live-trading entitlement.

## Classification And Routing

The canonical taxonomy is `config/public_financial_context_routing_v1.json`. It classifies every feature across ten dimensions:

1. Evidence class
2. Entity scope
3. Geographic scope
4. Market domain
5. Cadence
6. Latency class
7. Semantic direction
8. Decision family
9. Decision plane
10. Authority class

The taxonomy maps seventeen logical capabilities and thirty-seven normalized features. Runtime paper decisions and training gap fill require the current sleeve's institutional decision-family identifier to appear in the feature route. A missing or unrecognized family fails closed for these features. Cross-family broadcast is forbidden, and any emitted feature missing from the taxonomy is quarantined from bot context.

## Data Integrity

- Economic observations after the collection as-of time are rejected.
- Missing or unavailable values are omitted, never neutralized through zero filling.
- SEC facts retain units and point-in-time filing metadata.
- Symbol features require symbol-level Company Facts coverage.
- Capability readiness uses field-level Boolean proofs rather than source-name presence.
- Source, taxonomy, and snapshot receipts make changes detectable.
- The canonical economic registry verifies that every adapter has a real producer, declared capabilities, bounded planes and families, and official HTTPS provenance.
- Every adapter is exception-isolated, so one provider or parser failure cannot abort the remaining source collection.
- Weekly dealer and quarterly bank evidence is supplemental; its absence cannot downgrade an otherwise ready five-source baseline.
- Network and schema failures remain explicit in health evidence.

The collector is optional and safe to degrade. Its absence removes its features but does not lower required-source grades, block otherwise healthy paper collection, or reset cumulative soak history. Affected evidence scopes simply accrue forward after recovery.

## Integration

The collector registers with the collector-contract mesh and materializes seventeen capabilities spanning fundamentals, dealer balance sheets, repo and settlement conditions, aggregate bank credit, funding and credit conditions, risk regime, volatility, and regulatory policy. The decision-context mesh uses the classified features as optional corroboration in fiscal liquidity, funding stress, cross-border capital, positioning, securities lending, credit curves, volatility surfaces, and market calendars. Capability routing selects the producer only when its freshness, quality, payload, and field-level proof contracts pass.

The decision-context artifact publishes compact feature-route receipts. Both the hot paper loop and runtime training enrichment enforce the same family membership. This preserves paper/live decision-flow parity without granting the context source execution authority.

## Operations

Refresh the source:

```bash
./scripts/ops/opsctl.sh public-financial-sync --json
./scripts/ops/opsctl.sh economic-source-inventory --list
```

Refresh the consuming mesh without extra network collection:

```bash
./scripts/ops/opsctl.sh decision-context-sync --no-network --no-refresh-capacity --json
```

Inspect collector and capability routing:

```bash
./scripts/ops/opsctl.sh collector-contracts --include-data-plane --json
./scripts/ops/opsctl.sh capability-materialization --json
./scripts/ops/opsctl.sh collector-capability-control --json
```

Canonical artifacts:

- Collector: `scripts/collect_public_financial_context.py`
- Taxonomy: `config/public_financial_context_routing_v1.json`
- Latest payload: `exports/external_context/public_financial_context_latest.json`
- Latest health: `governance/health/public_financial_context_sync_latest.json`
- Decision mesh: `exports/external_context/decision_context_mesh_latest.json`
