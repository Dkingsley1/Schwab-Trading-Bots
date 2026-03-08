# Data Ingestion Sources (Free / Low-Cost APIs)

This file tracks external data sources you can ingest for macro, fundamentals, and market context.

## Key Requirement Matrix

| Source | Domain Fit | API Key Required | Notes |
|---|---|---|---|
| FRED / ALFRED | US macro + revisions | Yes | Strong replacement for many macro series; revision history is valuable for walk-forward realism. |
| BLS Public Data API | Labor + inflation | No (default) / Optional (higher limits) | Public access works without registration; key is optional for higher volume. |
| BEA API | GDP/NIPA/IO tables | Yes | Official US national accounts source. |
| US Census API | Demographic + business data | No (default) / Optional (higher limits) | Keyless use has tighter daily limits. |
| US Treasury FiscalData API | Debt/cash/fiscal operations | No | Good policy/liquidity backdrop inputs. |
| EIA Open Data API | Energy prices/supply | Yes | Strong for inflation and commodity-linked sleeves. |
| World Bank Indicators API | Global macro indicators | No | Broad country coverage. |
| OECD API (SDMX) | International macro/structure | No | Free; subject to rate limiting. |
| IMF API (SDMX) | Global macro/BoP/IFS style data | No | Use for global regime context. |
| SEC EDGAR APIs | Filings/fundamentals/events | No | No auth key required; set a compliant User-Agent in requests. |
| Alpha Vantage | Market + TA features | Yes | Free key; strict free-tier throughput. |
| Twelve Data | Market + technicals | Yes | Free key; tighter historical/rate limits on free tier. |
| Nasdaq Data Link | Alternative/econ/market datasets | Yes (recommended) | Some data is free, but key-backed access is more stable/trackable. |

## Recommended Ingestion Order

1. FRED, BLS, BEA, Treasury FiscalData, EIA (official US macro baseline).
2. World Bank, OECD, IMF (global regime extension).
3. SEC EDGAR (fundamental/event risk features).
4. Alpha Vantage / Twelve Data / Nasdaq Data Link (supplemental market features only).

## Practical Notes

- Treat official statistical agencies as primary truth and commercial feeds as secondary overlays.
- Store `source_name`, `dataset_id`, `as_of_utc`, and `revision_tag` for every pull.
- Keep per-provider retry/backoff and rate-limit guards separate, then normalize into one canonical schema before model feature extraction.

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
