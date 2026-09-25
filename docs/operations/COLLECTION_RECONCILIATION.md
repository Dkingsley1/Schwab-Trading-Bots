# Collection Reconciliation

## Inventory Is Not Coverage

The reviewed 2026-09-25 raw inventory contains 748 JSONL paths, including 342
zero-byte slots. Its 76.9 GiB is predominantly decision/control logs, not price
history. A prefix fingerprint or `training_eligible` flag does not establish
full-file integrity, unique events, labels, point-in-time suitability or lifetime
collection coverage.

`raw-inventory-cleanup` accepts only the exact reviewed inventory digest. Preview
first with `--audit PATH --verify-only --kind empty --limit 342`. Explicit
`--apply` retires only closed historical placeholders with stable identity,
idle-handle checks, native owner locks and durable restoration receipts. Active
inboxes, financial evidence and archive custody records remain protected.

For `--kind duplicates`, full-stream SHA-256 equality is mandatory. Consolidation
retains a content-addressed `.payload` under the BOT_LOGS deep-cold root and
preserves both original lookup paths. This avoids a dangling alias when an
ordinary JSONL compactor later retires a raw file. Custody and native storage
receipts precede source replacement; interrupted operations remain explicit.
No broad wildcard deletion or automatic payload expiry is authorized.

## Targeted Data

`collection-gap-census --manifest PATH` reads an explicit, bounded selection of
existing sources. It reconciles held equities and research symbols, distinguishes
physical copies from distinct candles, and plans only missing closed intervals.
It is not a lifetime archive scan. Census mode never downloads anything.

After reconciling the selected sources, an explicitly selected request can use
`--request PATH --fetch-public-coinbase --output governance/collection_backfills/NAME.json`
or `--fetch-schwab` for one exact missing candle interval. These are fixed-owner,
bounded read-only GETs with zero automatic retries, power/provider guards and
new-receipt-only output. The Schwab call is isolated in a 90-second child with
execution disabled. These receipts do not grant catalog research entitlement,
training readiness or historical live evidence. FRED still requires an authentic
vintage response through the existing import path and a configured API key.

Use original Schwab captures or validated owner-response receipts. Import a
bounded response with `--request PATH --response PATH --observed-at-utc TIME
--source-endpoint ENDPOINT`. Preserve provider event time separately from the
actual backfill observation time. A retrieved historical bar cannot become
evidence that the bot knew it historically.

Coinbase Exchange candle jobs use the existing BTC-USD adapter contract, with
at most 299 requested buckets per job and a 300-row response ceiling. FRED jobs
pin `realtime_start` and `realtime_end` to an explicit historical vintage. Never
substitute current revised CSV values for missing vintage evidence. See the
[Coinbase candle contract](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)
and [FRED observations contract](https://fred.stlouisfed.org/docs/api/fred/series_observations.html).

Account dividend postings establish only observed account activity, not a
complete issuer ex-date, split or adjustment calendar. Missing credentials,
entitlements, unexplained gaps and adjustment uncertainty remain visible.
Reconcile existing catalog coverage before dispatch; require current resource
admission and provider access, bounded GETs, and no automatic retries.

## Freshness And Evidence

Training publication preserves the atomic rows/manifest contract. Incremental
readers tolerate out-of-order rows without stopping prematurely, reject future
records, and report partial source scans. Research context accepts eligible
compressed decision sources but rejects archives exceeding its read budget.
Fresh report timestamps do not certify fresh source content.

The remaining outcome and point-in-time gaps require genuine timestamped events
and attributable outcomes. Cleanup or backfills cannot manufacture candidate
fills, establish profitability, or grant training/trading authority.

## Decision Charts

Every action through the shared DecisionLogger records a small provider/symbol
capture pointer, or an explicit unavailable reason. BUY, SELL, HOLD and blocked
decisions are reportable. `decision-candle-capture --symbol O` supports an
explicit Schwab equity symbol; it does not scrape charts or request orders.
The native Bitcoin observer retains shared Coinbase BTC-USD 5-minute and 6-hour
candles on its existing 15-minute cadence, without a second API fetch per bot.
Its three observation decisions are logged with the indicators and rules that
actually produced their observations. `decision-chart-report --bitcoin-bot
btc_intraday_breakout_watch` reports the latest recorded observation.

Original SCHD captures remain supported. Shared provider/hash-bound captures
support other Schwab equities and Coinbase USD spot products when their collector
has published matching data. Unsupported instruments and missing historical
captures remain explicit: an ETF or futures chart is not spot crypto evidence.
Coinbase uses 24/7 intervals rather than an equity exchange calendar. No data for
additional crypto products is fetched automatically by the BTC-only observer.

Charts render on demand, not once per decision in the trading loop. The report
includes original reasons, indicator features, thresholds and gates, separately
from chart-derived SMA, RSI, ATR, range and candle diagnostics. Feature presence
alone does not prove causal use. Chart images are not new model inputs, and no
performance improvement is certified by creating these reports.
Actual executions require broker execution legs with time, price, quantity and
reconciliation provenance. Proposed actions are visually separate. Events
outside a displayed interval are listed rather than placed on an invented bar.

Future successful supervised reconciliation retains bounded original execution
responses in the existing hashed ledger event when the original native decision
is bound to the intent. Reports read those events through a bounded read-only
SQLite connection and verify their event hashes. No extra broker request, order,
schema migration or ledger writer is opened by reporting. Old aggregate-only
fills cannot be retroactively assigned invented timestamps or decision IDs.
Native rehearsal reports link the original decision chart sidecar; explicit
`decision-chart-report` selects an original log entry by exact decision ID.

A later capture can produce a separately labeled retrospective execution review;
it is never an input to the original decision. Missing captures and executions
remain unavailable. The legacy retrospective capture selector remains SCHD-only.
Coinbase executions require a future exchange reconciliation owner; Schwab fill
receipts cannot be plotted as Coinbase trades. HOLD has no expected fill and is
labeled not applicable rather than degraded for missing execution evidence.
The shared capture store is capped at 512 files/64 MiB (1 MiB per capture), and
report assets at 128 files/32 MiB. Capacity exhaustion is explicit; neither owner
silently deletes decision-bound evidence. These caches are not an unlimited audit
archive. Native SCHD decision reads also support complete bounded gzip sources;
oversized or malformed reads still fail closed without selecting a convenient
older decision.
