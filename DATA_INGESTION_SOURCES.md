# Data Ingestion Sources (Free / Low-Cost APIs)

This file tracks external data sources you can ingest for macro, fundamentals, and market context.

## Key Requirement Matrix

| Source | Domain Fit | API Key Required | Notes |
|---|---|---|---|
| FRED / ALFRED | US macro, Fed balance sheet, funding rates, and revisions | Optional / preferred | The API key supports the JSON path; official public FRED CSV is the bounded fallback. ALFRED revision vintages remain a future walk-forward extension. |
| Federal Reserve Board | Policy events, releases, H.4.1 context | No | Official calendar and release feeds provide central-bank activity context. |
| Federal Reserve Bank of New York | SOFR, EFFR, OBFR, repo, and reverse repo | No | Official money-market and open-market-operation context; FRED series are normalized into the central-bank liquidity contract. |
| BIS central-bank policy rates (`WS_CBPOL`) | Comparable policy rates across more than 40 economies | No | Member-central-bank reported daily history; used by the governed 32-bank point-in-time registry. |
| BIS central-bank total assets (`WS_CBTA`) | Comparable balance-sheet context across more than 50 economies | No | Quarterly USD series compiled from national central-bank and official sources. |
| BLS Public Data API | Labor + inflation | No (default) / Optional (higher limits) | Public access works without registration; key is optional for higher volume. |
| BEA API | GDP/NIPA/IO tables | Yes | Official US national accounts source. |
| US Census API | Demographic + business data | No (default) / Optional (higher limits) | Keyless use has tighter daily limits. |
| US Treasury FiscalData API | Debt/cash/fiscal operations | No | Good policy/liquidity backdrop inputs. |
| US Treasury TIC monthly history | Cross-border portfolio capital flows | No | Direct official monthly inflow series used by the point-in-time cross-border plane. |
| EIA Weekly Petroleum Status Report CSV | Crude, SPR, gasoline, and distillate inventories | No | Direct weekly inventory context; separate from the keyed EIA Open Data API. |
| Bureau of Transportation Statistics Freight TSI | Freight, truck, rail, and intermodal activity | No | Official supply-chain and transportation activity context. |
| EIA Open Data API | Energy prices/supply | Yes | Strong for inflation and commodity-linked sleeves. |
| World Bank Indicators API | Global macro indicators | No | Broad country coverage. |
| OECD API (SDMX) | International macro/structure | No | Free; subject to rate limiting. |
| IMF API (SDMX) | Global macro/BoP/IFS style data | No | Use for global regime context. |
| SEC EDGAR APIs | Filings/fundamentals/events | No | No auth key required; set a compliant User-Agent in requests. |
| Nasdaq analyst forecasts | EPS consensus, dispersion, analyst counts, and four-week revisions | No | Bounded internal personal research/paper context. Commercial or live use requires separately verified entitlements. |
| Alpha Vantage | Market, TA, and analyst consensus/revision features | Yes | Free key; strict free-tier throughput. `EARNINGS_ESTIMATES` is opt-in and quota-bounded. |
| Twelve Data | Market + technicals | Yes | Free key; tighter historical/rate limits on free tier. |
| Nasdaq Data Link | Alternative/econ/market datasets | Yes (recommended) | Some data is free, but key-backed access is more stable/trackable. |

## Recommended Ingestion Order

1. FRED, BLS, BEA, Treasury FiscalData, EIA (official US macro baseline).
2. Treasury TIC, EIA weekly inventories, and BTS freight TSI for the governed cross-border and supply-chain planes.
3. BIS policy rates/assets plus World Bank, OECD, and IMF (global monetary and macro regime extension).
4. SEC EDGAR (fundamental/event risk features).
5. Nasdaq analyst forecasts, Alpha Vantage, Twelve Data, and Nasdaq Data Link (supplemental context with explicit entitlement boundaries).

## Practical Notes

- Treat official statistical agencies as primary truth and commercial feeds as secondary overlays.
- Store `source_name`, `dataset_id`, `as_of_utc`, and `revision_tag` for every pull.
- Keep per-provider retry/backoff and rate-limit guards separate, then normalize into one canonical schema before model feature extraction.
- Exclude observations dated after the collection as-of date, apply cadence-aware freshness limits, and keep net-liquidity formulas labeled as heuristics rather than official accounting identities.
- See [docs/architecture/CENTRAL_BANK_LIQUIDITY_CONTEXT.md](docs/architecture/CENTRAL_BANK_LIQUIDITY_CONTEXT.md) for the required Fed-liquidity series and consumer contract.
- See [docs/architecture/GLOBAL_CENTRAL_BANK_CONTEXT.md](docs/architecture/GLOBAL_CENTRAL_BANK_CONTEXT.md) for the 32-bank registry, BIS source contracts, point-in-time joins, conflict policy, and bot routing.

## No-Key Starter Set (1-4)

Use these as your immediate no-key ingestion baseline:

1. US Treasury FiscalData API
2. World Bank Indicators API
3. OECD API (SDMX)
4. IMF API (SDMX)

These four can be added first with minimal setup and are suitable for regime/context features before adding keyed providers.

## BLS Public API (Keyless Mode)

- You can ingest BLS in public mode without a key.
- For higher request limits, add `BLS_API_KEY` later.
