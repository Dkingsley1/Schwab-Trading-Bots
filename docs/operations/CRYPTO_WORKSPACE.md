# Crypto Workspace

The existing phone-feed dashboard now has a `/crypto` section. It is a local extension of the Schwab trading platform, not a separate hosted service.

## Open

Start the existing `phone-feed` command with `--host 127.0.0.1` and an unused port, then open `/crypto` at that address. The existing home screen also links to Crypto. If a feed token is configured, the Crypto page uses that token, never Coinbase credentials. A running pre-patch server must be restarted by its owner to load the extension; the installer does not restart services.

## Views

- Portfolio reads the already-saved Coinbase snapshot through the documented importer's owner-only file checks. The HTTP response includes only currency and decimal balance fields, with producer timestamp and freshness. No credentials, account IDs, permission payloads or raw files reach the browser. Verification older than five minutes is explicitly cached/stale; no provider request or credential write is made here. Refreshing the dashboard rereads local evidence only.
- Practice provides separate BTC-USD day and swing rule-based historical baselines. They are not trained bots or forward-paper cohorts. Virtual capital defaults to $1,000 per profile and is independent of actual holdings. The profiles are long-only spot, with no borrowing or leverage.
- Performance shows modeled entries/exits, post-cost P&L, bar-close equity drawdown, round trips, fees and execution friction. Buy-and-hold uses the same evaluation window, entry/exit fees and shared cost model. An unavailable run is not shown as zero performance. In-memory results are replaced per profile and cleared on server restart; the download button exports a research-only JSON record with data hash and assumptions.
- Readiness shows the existing breaker, launcher, crypto collection and training evidence. Stale, future, missing or malformed producer timestamps cannot indicate current readiness. Historical replay cannot admit the BTC profiles to a forward-paper cohort even when the general launcher becomes ready.

## Schwab Market Context

The existing `crypto-market-sync` collector command now requests one batch from the fixed Schwab `/marketdata/v1/quotes` endpoint before optional external sources, inside its existing runtime budget. No new daemon or schedule is added. The native context switch is `SCHWAB_CRYPTO_CONTEXT_ENABLED` (default `1`; `0` disables it). The retired speculative spot bridge's `SCHWAB_CRYPTO_DATA_ENABLED`, URL template, symbol map, and bearer-token settings do not enable a spot feed or redirect credentials. No extra key is required.

Supported context instruments are `/BTC`, `/MBT`, `IBIT`, `FBTC`, `/ETH`, `/MET`, `ETHA`, and `ETHE`, filtered to tracked BTC/ETH assets. Root futures requests resolve only to the provider's explicitly identified active contract with matching product and future expiry. Fund quotes must be typed EQUITY/ETF or EQUITY/CEF. Prices are USD per underlying unit for futures, USD per share for funds; multiplier and expiry remain separate contract fields. Never splice rolling contract marks into spot history or use fund share prices as BTC prices.

Every normalized row retains provider, asset, requested symbol, source contract/symbol, instrument type, currency provenance, observed time, provider quote time, real-time flag, quality, and context-only role. Fresh quotes include mark, bid/ask, spread bps, session volume, and futures open interest. Real-time must be a true boolean; quote and both book-side timestamps must be no more than 120 seconds old and not future-dated. Missing, stale, delayed, crossed, implausible, inactive, or wrong-type quotes contribute no features. Market closure is not replaced with fabricated freshness. The UI independently ages producer and quote timestamps on each local refresh.

Six features enter the collector's existing `derived.symbol_features`: `crypto_schwab_{future,etp}_{available,return,spread}_norm`. Each type is aggregated separately by asset across usable instruments. Availability is 1; return is the mean percentage change from previous settlement/close mapped by `clip(0.5 + pct / 20, 0, 1)`; spread is mean bps divided by 100 and clipped to 1. Missing types remain zero in the collector's feature map. Full instrument rows stay in `sources.schwab_instruments` and the existing sync-health source receipt. The two legacy Schwab spot-availability/agreement features remain zero. These are contextual predictors, not training outcomes; downstream strategy feature selection, chronological validation, qualification and admission remain independent requirements.

The adapter reads the existing root `token.json` atomically by file descriptor without changing it. Expired/missing credentials produce a visible non-secret status and defer to the existing Schwab auth owner. There are no refresh requests, retries, redirects, proxies or token-bearing configurable URLs; responses are limited to 512 KiB with an eight-second maximum network budget. Existing non-owner-only token permissions are reported as auth-owner debt, not silently changed. Tokens, HTTP error bodies and raw account data never enter collector output or browser responses.

The shadow-loop feature consumer recomputes Schwab features from independently aged quote/book receipts on every consumption, not cached aggregate values. Unknown, future, or expired timestamps yield no Schwab features. This does not expand model input schemas or train existing bots automatically. The existing daily-refresh owner's cache cadence remains unchanged, so this is sampled context, not a new continuous quote stream; evidence expires between refreshes instead of pretending to stay current. Existing long-running bot processes need their next controlled restart to adopt the new consumer.

On September 10, 2026, a read-only native API check verified all eight instruments as usable real-time context. This is a point-in-time feed check, not a soak or live-release result. Direct Schwab BTC/ETH spot API symbols/schema are still unverified; Coinbase remains the spot practice source. New collector invocations load the changes; an already-running dashboard needs its owner's restart to load the backend additions.

## Research Contract

Day: 288 five-minute bars (24 hours), first 50 bars warm up EMA state. Enter on a closed-bar 12-bar breakout above EMA 20 above EMA 50. Exit below EMA 20 or at 72 bars (six hours).

Swing: 280 six-hour bars (70 days), first 50 bars warm up EMA state. Enter on close above EMA 20 above EMA 50. Exit below EMA 20 or at 20 bars (five days). These are maximum holding periods, not promises to hold every position that long.

Signals see completed bar i-1; fills reference bar i open. Any final position is liquidated at the evaluation window's last close and labeled explicitly. No same-bar high/low stop ordering is inferred. Both profiles assume full fills because historical candles have no measured order-book depth. The shared `core.execution_simulator` prices spread, latency, volatility and other modeled friction; its embedded fee is removed before the selected fee is charged once. Drawdown is sampled at bar closes and is not intrabar drawdown. The benchmark has the same fees and spread assumptions.

Only the fixed public `api.exchange.coinbase.com/products/BTC-USD/candles` GET endpoint is used. TLS validation stays enabled; there are no redirects, proxies or credentials. The response is capped at 512 KiB and 300 rows. Nonfinite/malformed prices, gaps, duplicate or misaligned timestamps, incomplete windows and insufficient closed bars fail the replay. It never fills gaps with synthetic prices.

One replay child per server is allowed at a time. It receives a minimal noncredential environment, single-thread numeric-library settings and nice +10. It has a 25-second parent-enforced deadline and is killed/reaped by `subprocess.run` on timeout. It does not create grandchildren. Repeated identical successful requests use a five-minute cache; changed or failed requests have a 60-second per-profile cooldown. The dashboard rereads small local status files every 30 seconds only while visible. Nothing schedules collectors or retraining.

## Money Lock

There is no order placement, cancellation, transfer, account refresh, model training, candidate mutation, ledger write, launcher or unlock endpoint. Unsupported POST routes return 405. Existing execution controls remain unchanged. This section's lack of execution capabilities is not a new certification of overall live-money readiness.

Crypto HTTP routes require a loopback peer and exact loopback Host with server port. Cross-site requests and foreign origins are rejected. Research POST also requires same-origin JSON and a body of at most 2 KiB. Existing feed authentication is applied to every data endpoint. Responses are no-store, have a restrictive CSP, cannot be framed, and make no third-party browser requests. Crypto request logs omit URLs and query tokens. Account data is never deployed to Sites or other cloud hosting.

## Sources

- [Schwab crypto futures](https://www.schwab.com/learn/story/crypto-futures-solana-ripple-now-on-thinkorswim): official futures product roots. Native quote response types and active contract fields were separately verified against the connected account's read-only feed.
- [Schwab market-data API](https://developer.schwab.com/products/trader-api--individual/details/specifications/Market%20Data%20Production): production API specification portal. Product/UI availability alone is not proof of a spot-crypto API contract.
- [Coinbase candle API](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles): bucket schema, closed-window limits and incomplete-history caveat.
- [Coinbase Advanced fees](https://help.coinbase.com/en/coinbase/trading-and-funding/advanced-trade/advanced-trade-fees): fees vary by order type and account tier. The UI's editable 60 bps default is an assumption, not verified account pricing.
- Icons use a small vendored Lucide subset under the ISC license embedded in `crypto_assets/icons.svg`.
