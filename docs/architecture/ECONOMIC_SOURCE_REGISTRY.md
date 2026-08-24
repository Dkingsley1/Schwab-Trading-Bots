# Public Economic Source Registry

`config/economic_source_registry_v1.json` is the canonical inventory for public macro, microeconomic, policy, funding, market-structure, and issuer evidence. Run `./scripts/ops/opsctl.sh economic-source-inventory --list` for the generated, machine-validated list and `--json` for complete routes.

## Current inventory

| Producer | Active public sources |
| --- | --- |
| `bls_census` | BLS Public Data API; Census ACS; FRED Economic Data; BEA economic accounts |
| `official_macro_context` | Federal Reserve policy events; Treasury policy announcements; BLS release calendar; BEA release calendar |
| `global_central_bank_context` | 32 official central-bank policy sources from `config/global_central_bank_registry_v1.json` |
| `public_policy_context` | Treasury Debt to the Penny; Treasury Average Interest Rates; World Bank Indicators |
| `extended_quant_context` | CFTC Commitments of Traders; New York Fed reference rates; Cboe options statistics; Nasdaq Reg SHO threshold list; SEC fails-to-deliver data |
| `decision_context_mesh` | Treasury International Capital; EIA Weekly Petroleum; BTS Freight TSI |
| `sec_edgar_context` | SEC EDGAR ticker map; submissions; filing archive |
| `public_financial_context` | SEC CompanyFacts; OFR Financial Stress Index; FDIC bank failures; Federal Register financial rules; ECB euro short-term rate; New York Fed primary-dealer statistics; FDIC quarterly bank financials |
| `market_micro_context` | Treasury auction results; FINRA short-sale volume; Nasdaq trade halts |
| `tradingeconomics_guest` | Trading Economics guest feed, secondary cross-check only |

The registry currently contains 33 direct source contracts plus 32 expanded central-bank members. The count is inventory only: it does not raise readiness, alpha, profitability, or promotion grades.

## Added direct evidence

- `nyfed_primary_dealer_statistics` adds weekly aggregate Treasury and corporate dealer inventory, Treasury repo/reverse-repo balance, and financing fails. It routes only to funding, positioning, securities-lending, and credit consumers.
- `fdic_bank_financials` uses FDIC server-side quarterly aggregation for assets, deposits, loans, and noncurrent loans. It avoids institution-level bulk downloads and routes only to bank-credit and funding consumers.
- Existing `treasury_auctions` parsing now rejects future auction dates and derives bounded bid-to-cover demand, indirect demand, dealer absorption, and issuance pressure instead of treating every auction window as generic credit stress.

## Routing and failure contract

1. A source must map to an existing physical producer and capabilities declared by that producer.
2. Every source has explicit macro/micro scope, evidence domains, decision planes, and decision families.
3. Network work occurs once per shared collector snapshot; per-bot network fanout is forbidden.
4. Future observations are rejected. Missing values are omitted and never replaced by neutral or favorable zeros.
5. New weekly and quarterly sources are supplemental. Their failure is visible but does not stop healthy collection or paper trading and does not lower the baseline public-financial status.
6. Transport receipts, observation times, source confidence, schema confidence, and point-in-time lineage travel with classified features.
7. Unclassified features are quarantined before runtime enrichment.
8. Economic context cannot authorize an action, alter quantity, submit an order, promote a candidate, or claim profitability.

## Operator commands

```bash
./scripts/ops/opsctl.sh economic-source-inventory
./scripts/ops/opsctl.sh economic-source-inventory --list
./scripts/ops/opsctl.sh economic-source-inventory --json
./scripts/ops/opsctl.sh public-financial-sync --json
./scripts/ops/opsctl.sh decision-context-sync --json
```
