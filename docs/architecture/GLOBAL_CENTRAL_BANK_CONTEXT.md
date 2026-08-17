# Global Central Bank Point-In-Time Context

## Purpose

This layer gives paper decisions and training a governed view of important monetary authorities without turning central-bank data into an order instruction. It separates three truths:

1. What a central bank reported.
2. What independent market and macro sources showed at that time.
3. Whether the joined evidence is fresh, consistent, and usable for a particular bot or symbol.

The raw collector and cross-source router have no live-execution, capital-allocation, registry-mutation, or automatic-promotion authority.

## Governed Coverage

The registry contains 32 institutions. The collector also preserves the wider unregistered BIS area universe in coverage diagnostics so expansion does not require pretending unsupported institutions are governed.

| Tier | Institutions |
| --- | --- |
| 1 | Federal Reserve, European Central Bank, People's Bank of China, Bank of Japan, Bank of England |
| 2 | Bank of Canada, Swiss National Bank, Reserve Bank of Australia, Reserve Bank of New Zealand, Reserve Bank of India, Central Bank of Brazil, Bank of Mexico, Bank of Korea, Norges Bank, Sveriges Riksbank, Monetary Authority of Singapore, Hong Kong Monetary Authority, Bank Indonesia, South African Reserve Bank, Central Bank of the Republic of Turkiye, Saudi Central Bank, Bank of Russia |
| 3 | Bank of Israel, Bank Negara Malaysia, Bank of Thailand, Bangko Sentral ng Pilipinas, National Bank of Poland, Danmarks Nationalbank, Czech National Bank, Central Bank of Chile, Central Bank of the UAE, Central Bank of Argentina |

Tier controls governance priority and coverage weighting, not trading authority. Exchange-rate, peg, currency-board, and multi-instrument frameworks are not forced into a fictional single policy rate. A dimension is required only when the registry says it is meaningful for that institution.

## Source Contracts

| Layer | Primary input | Function |
| --- | --- | --- |
| Policy rates | BIS `WS_CBPOL` | Daily member-central-bank reported policy-rate history |
| Central-bank assets | BIS `WS_CBTA` | Quarterly USD total assets compiled from national central-bank and official sources |
| Detailed U.S. liquidity | Fed, FRED, and New York Fed contract | Balance sheet, reserves, TGA, repo/RRP, swaps, funding rates, corridor, and stress |
| FX transmission | ECB reference-rate history plus canonical pair reconciliation | Currency reaction and provider-conflict detection |
| Sovereign macro | World Bank indicators | Inflation, real-rate, current-account, debt, and GDP context when reported |
| Official events | Existing official macro calendar/news rows | Matched central-bank communications when a verified row is available |
| Cross-asset confirmation | Official macro cross-asset snapshot | Rates, credit, volatility, dollar, commodity, and risk-state confirmation |

The BIS URLs are the numerical lineage origin. An institution's official policy URL is retained separately as publisher/reference metadata and is never mislabeled as the transport used for the BIS observation.

## Synchronization Contract

Every bank row is joined on explicit jurisdiction, currency, and observation-time keys. A raw central-bank row cannot mark itself synchronized. Readiness requires at least one fresh point-in-time link from a distinct source.

For every usable dimension the router records:

- source identifier and source URL
- publisher/reference URL when different
- source artifact timestamp
- economic observation time
- field path
- source confidence and freshness
- future-observation and point-in-time verdicts

Future observations are excluded before routing. Stale sources do not contribute. Missing raw values remain absent. Neutral aggregate values are permitted only alongside explicit coverage features. A high or critical canonical FX-provider divergence blocks the affected bank route and fails the synchronized consumer contract. Lower-severity divergence remains visible as an advisory conflict.

## Bot Routing

The router publishes both global features and symbol-scoped features. Bank-specific context overrides the global aggregate only for mapped currencies, country ETFs, rates, credit, and macro proxies. Each symbol row carries bank-level FX, macro, liquidity, lineage, conflict, and freshness coverage so a neutral value cannot be mistaken for complete evidence.

The same contract is consumed by:

- shadow and guarded paper decisions
- runtime training gap fill
- behavior-dataset version `trade_behavior_features_v5`
- training-label source routing
- source verification and collector-capability routing

The behavior dataset gives symbol-scoped synchronized evidence precedence over the global average and stores the contract/routing metadata in dataset lineage.

## Artifacts And Commands

```bash
./scripts/ops/opsctl.sh global-central-bank-sync --json
./scripts/ops/opsctl.sh central-bank-context-sync --json
./scripts/ops/opsctl.sh macro-context-sync --json
./scripts/ops/opsctl.sh source-verification --json
./scripts/ops/opsctl.sh collector-contracts --json
```

Primary artifacts:

- `config/global_central_bank_registry_v1.json`
- `exports/external_context/global_central_bank_context_latest.json`
- `exports/external_context/central_bank_cross_source_latest.json`
- `governance/health/global_central_bank_context_sync_latest.json`
- `governance/health/central_bank_cross_source_sync_latest.json`
- append-only gzip snapshots under `data/external_context/global_central_bank_history/`

The global collector can serve a bounded last-good snapshot for a transient BIS transport failure, but only while the consumer freshness contract remains valid. No fallback can erase source failure, extend freshness indefinitely, manufacture a missing dimension, or authorize execution.

## Interpretation Boundary

Policy and balance-sheet data can improve regime awareness, risk context, and training labels. Publication cadence, revisions, policy-framework differences, market anticipation, and transmission lags remain real limitations. This layer does not guarantee that a trade will be profitable and does not replace paper evidence, out-of-sample validation, broker truth, risk controls, or explicit live release.
