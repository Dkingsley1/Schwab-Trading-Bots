# Decision Context Mesh

## Purpose

`decision_context_mesh_v1` synchronizes twelve point-in-time context planes into one governed artifact for paper decisions, training, replay, and research. It does not submit orders, alter risk limits, promote a model, unlock live execution, or guarantee profitability.

The six macro planes are fiscal liquidity, funding stress, cross-border capital, credit and yield curves, market-calendar structure, and supply-chain/inventory state. The six micro planes are positioning/crowding, securities lending, volatility surfaces, passive/mechanical flows, estimate revisions/dispersion, and capacity/market impact.

## Sources

The collector reuses fresh system artifacts for Fed and New York Fed liquidity, Treasury fiscal data, World Bank indicators, CFTC positioning, Cboe options statistics, FINRA short volume, Nasdaq halts and threshold lists, SEC filings and failures-to-deliver, options chains, Schwab public news context, yield curves, paper execution calibration, and portfolio capacity.

Three missing dimensions are collected directly from official public endpoints:

- U.S. Treasury TIC monthly cross-border portfolio flows.
- EIA Weekly Petroleum Status Report inventory tables.
- Bureau of Transportation Statistics Freight Transportation Services Index.

Every source has an expected publication or refresh cadence and a separate hard staleness limit. A source earns full freshness while it is current for its real cadence, then decays toward zero before the hard limit. Direct sources may use the prior successful value only inside that bounded window. A missing or expired value is omitted; it is never converted into a zero.

The estimate plane consumes Nasdaq analyst forecasts containing EPS consensus, high/low dispersion, analyst counts, and four-week up/down revisions. Direct readiness requires the exact governed 16-symbol membership, `16/16` fresh symbol coverage, and `16/16` revision-history coverage. The collector uses a persistent, process-safe UTC-day request budget, counts failed calls, checkpoints after every symbol, applies a run deadline, and keeps availability-time snapshots. Alpha Vantage `EARNINGS_ESTIMATES` remains an optional credentialed fallback.

The public Nasdaq route is authorized here only for internal personal research, paper decisions, and training context. The collector does not infer commercial redistribution or live-trading data rights. A separate verified entitlement is required before this source can be admitted to a commercial or live execution context.

## Point-In-Time Contract

Every routed feature carries publisher, source URL, artifact time, economic observation time, field path, confidence, fallback state, and point-in-time validity. Future observations are rejected before feature construction. The mesh records rejected observations and fails its consumer contract if any future observation is selected.

The runtime consumes a mesh only when all twelve planes are present, the complete normalized feature schema is valid, the artifact is under 24 hours old, every plane clears its minimum score, macro and micro averages clear their contracts, point-in-time and missing-value policies are declared, and all execution and promotion authorities remain false.

## Grades

Each plane earns a percentage from six evidence components:

| Component | Weight |
| --- | ---: |
| Source health | 30% |
| Feature completeness | 25% |
| Freshness | 15% |
| Point-in-time lineage | 15% |
| Routing completeness | 10% |
| Cross-verification | 5% |

Macro and micro percentages are the arithmetic means of their six plane scores. `A+` requires at least 97%, `A` at least 93%, and lower letter grades follow the scale embedded in the artifact. Grades are calculated from current evidence and are never operator overrides.

The estimate-revision plane is capped at `B+` while it relies on SEC and broker-news proxies. The cap lifts only when the direct consensus contract has the exact governed symbol membership, all `16/16` symbols, all `16/16` revision histories, freshness, and point-in-time lineage. A configured name, an empty artifact, partial coverage, or a missing revision field cannot lift it.

## Artifacts And Operations

- Configuration: `config/decision_context_mesh_v1.json`
- Consumer contract: `core/decision_context_mesh.py`
- Collector: `scripts/collect_decision_context_mesh.py`
- Optional consensus collector: `scripts/collect_analyst_consensus_context.py`
- Optional consensus configuration: `config/analyst_consensus_context_v1.json`
- Latest context: `exports/external_context/decision_context_mesh_latest.json`
- Latest health and grades: `governance/health/decision_context_mesh_latest.json`
- Append-only history: `data/external_context/decision_context_mesh_history/*.jsonl.gz`
- Refresh command: `./scripts/ops/opsctl.sh decision-context-sync --json`
- Consensus refresh command: `./scripts/ops/opsctl.sh analyst-consensus-sync --json`

The daily refresh runs the mesh after its upstream collectors and before collector, capability, provider, training, and source-verification controls. Source-verification autorefresh can invoke the same bounded command when the artifact becomes stale or semantically invalid.
