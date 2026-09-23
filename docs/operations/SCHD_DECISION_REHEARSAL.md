# SCHD Decision Evidence And Rehearsal

This is an isolated conditional simulation, not an order command or a new trading
strategy. Status, demo, evaluate and cached native observation are offline. The
explicit `charts`, `maintain` and native market-refresh commands use the existing
authenticated Schwab client for price-history/quote GETs only, never
account or order APIs. Nothing activates trading, alters the O holding or writes
canonical paper fill evidence. Maintenance uses the existing adaptive-ops cadence;
there is no new scheduler or automatic live transition.

## Commands

```bash
./scripts/ops/opsctl.sh schd-decision-rehearsal status --json
./scripts/ops/opsctl.sh schd-decision-rehearsal demo --json
./scripts/ops/opsctl.sh schd-decision-rehearsal charts --json
./scripts/ops/opsctl.sh schd-decision-rehearsal maintain --json
./scripts/ops/opsctl.sh schd-decision-rehearsal native --json
./scripts/ops/opsctl.sh schd-decision-rehearsal native --refresh-market-data --json
./scripts/ops/opsctl.sh schd-decision-rehearsal evaluate --input /absolute/path/packet.json --json
```

`status` is read-only. With no imported evidence it says `waiting_entry`, not
ready to trade. `demo` tests fictional BUY and SELL decisions, quotes and candles;
its prices are not current SCHD prices and its strategy is not a trained bot.
`evaluate` accepts explicitly imported recorded packets. Provider names in an
import are declarations, not independently authenticated provenance. `native`
observes the existing dividend sleeve's SCHD grand-master log and cached candles.
It does not select among strategies to find a BUY or run a background observer.
A green fixture demonstration is not an actual native round trip.

Artifacts are bounded latest files under `governance/rehearsals/schd/`:

- `synthetic_latest.json` and `.md`: synthetic demonstration only.
- `recorded_state.json` and `.md`: at most one conditional entry and exit, their
  exact evidence packets, and the most recent assessment.
- `writer.lock`: shared singleton ownership for capture, retention and rehearsal.
- `market_latest.json`: one bounded latest Schwab capture shared by charts and
  native observation. A failed refresh retains its original capture time.
- `schwab_charts_latest.json` and `.md`: actual Schwab OHLC context, no invented
  bot decision or executable quote. Price adjustment remains explicitly unverified.
- Fixed-prefix PNG diagrams, overwritten in place, for each report/timeframe.
- `captures/<sha256>.json`: bounded immutable online context, not permanent audit
  history. Original lookup uses the recorded hash, never a newer substitute.
- `evidence_maintenance_latest.json`: native refresh/retention and actual decision
  freshness; a healthy process heartbeat cannot certify a fresh SCHD decision.

`charts` uses three bounded price-history requests (daily from January 1 two years
ago, five-minute from nine days ago, one-minute from two days ago), excludes open
candles, and derives the longer views from complete daily sessions. A 90-second
child timeout, native provider cooldown/slot and system-power checks remain active.
Authentication is noninteractive; failure does not present old candles as fresh.
The provider payload digest, endpoint, capture time and excluded-row counts are
included. Candle timestamps, not file mtimes, determine coverage.

`maintain` runs through adaptive-ops every 60 seconds during the regular session
and hourly outside it for retention only. A 25-second history child and
45-second outer budget, shared nonblocking writer lock, provider cooldown and
power-OFF checks bound work. It never renders PNGs, restarts a worker or modifies
a decision. Failed fetches preserve the previous bytes and source timestamp.
Fresh wrappers with missing closed candles are not marked current. The online
cache is bounded at 128 entries/64 MiB. Each pass verifies at most 8 MiB and prunes
at most eight owned captures older than 30 minutes, excluding latest, recent,
future, corrupt, unknown and hardlinked files. Complete abandoned capture-build
files additionally need an old filesystem modification time. Decision logs,
recorded simulation/order evidence, manifests and external volumes are untouched.
An expired historical capture cannot be reconstructed as original evidence.

Maintenance also reads the fixed dividend/equities/Schwab ingress owner. It
reports an observed event/session/provider/resource pause or interval wait
separately from the native decision age. Missing, corrupt, linked, wrong-scope,
future or more-than-five-minute-old observations cannot explain current producer
state. This is last-observed diagnostic evidence, not proof of current process
liveness, a fresh decision, or permission to restart a worker.

Collector publication keeps heartbeat and ingress pause states consistent.
Interval-wait metadata shows the adaptive interval, current external pressure
floor and duty-cycle sleep; updating this metadata does not append another
ingress-count summary. A temporary external floor is not stored in adaptive
state, so it releases on the next evaluated cycle when its owner clears it.
Independent memory/overload throttles and all existing gates remain in force.
The existing launcher watches collector source and can recycle read-only workers
after its configured settle window. Maintenance cannot request that restart;
runtime code adoption does not accept a release candidate or clear trading.

`charts` also includes a separately attributed sample of the latest recorded
decision with original action, reasons, score/threshold, age and source receipt.
Those newly retrieved charts are context, not proof the bot used them. A sample
cannot approve an order, claim a good entry, or change HOLD to BUY.

## Native Decision Connection

`native` reads only `decisions/shadow_dividend_equities/` for the current and
previous UTC day. Selection is fixed to SCHD `grand_master_bot`, dividend
profile/grand-master layer in shadow or paper mode. The scan has a 64 MiB total
tail bound, 1 MiB row bound and eight-second budget. It is not a full historical
inventory. Missing records, malformed or unfinished rows, conflicting identities,
future timestamps and unstable source files cannot authorize simulation.
Protected/external routes and symlinks are rejected before traversal.

The report records exact source path, byte offset, raw-row and canonical hashes,
candidate/receipt binding and scan limits. Original action, reasons, gates and
features are preserved. A logged `decision: EXECUTE` means the gates passed;
`action: HOLD` still means no trade. A newer HOLD or veto cancels a pending
simulated submission without retry. Candidate or implementation changes require
review and cannot silently reset the isolated state.

New source records also carry original Schwab quote time, realtime flag and raw
last price separately from collector/history-derived values. The grand-master
metadata binds the available bounded candle capture by hash before logging.
Missing or overwritten context fails closed; retrieving new candles cannot
retroactively prove an old decision's context. The separately guarded live
handoff is described in `SUPERVISED_BROKER_TEST.md`; this rehearsal still never
submits orders or grants authority to that command.

`--refresh-market-data` adds one read-only SCHD quote GET to the three bounded
history requests. Provider quote time is required; collector observation time is
never relabeled as provider time. A fresh quote cannot repair missing provenance
of the quote used by an older decision. Candles after the decision are excluded;
history fetched afterward is labeled retrospective context, not proof of what
the bot knew then. Native price adjustment must be verified independently before
admission. Missing quote time, unverified price basis or retrospective context
remain blockers, even when the diagrams are useful. Recorded feature reasons
and computed chart diagnostics remain separate.

Diagrams show OHLC bodies/wicks, volume, prior-range levels and the latest SMA
levels where available. SMA lines are latest-value references, not historical
moving-average curves. All diagrams label UTC times, source, completion and
adjustment caveats. Synthetic/imported charts are visibly distinct from API
capture. At most 60 closed candles per diagram are rendered. The trailing-180d
diagram is one aggregate window, not a misleading invented series of 180d bars.
Yearly and monthly charts omit the unfinished current period.

JSON is the authority. It is atomically replaced before rendering Markdown.
A Markdown failure cannot repeat a committed conditional fill. Candidate, source
hash and numeric execution-simulator environment bind the state across restarts;
changes require review, not automatic reset. Terminal states cannot re-enter.

## Evidence Contract

A decision packet contains:

- `schema_version: 1`, `symbol: SCHD`, `candidate_id`, `evidence_kind: recorded`.
- `price_basis: split_adjusted_dividends_unadjusted`, consistent across candles
  and quotes. The importer must verify this; the rehearsal cannot infer it.
- `decision`: the existing DecisionLogger shape: aware `timestamp_utc`,
  `symbol`, unique `decision_id`, `strategy`, `action`, `decision`, nonempty
  boolean `gates`, nonempty recorded `reasons`, optional model score/threshold and
  `features`, plus `metadata.snapshot_id` matching the decision quote. Optional
  `metadata.invalidation_conditions` and `decision_horizon` are not invented.
- `quote`: SCHD, `provider: schwab`, `source_quality_label: broker_native`,
  distinct `snapshot_id`, aware source `timestamp_utc`, numeric `bid`, `ask`,
  `last`, `bid_size`, `ask_size`. Last must be positive; ask strictly above bid.
- `candles`: `5m` and `1d` arrays, optional real `1m`. Each row contains explicit
  aware `start_utc`, `end_utc`, and numeric `open/high/low/close/volume`.
  Ambiguous broker date stamps must be normalized by the upstream importer,
  never guessed. At most 6,000 bars per source array and 6 MiB per input file.
- Optional `corporate_actions`: source-declared events, always marked
  unverified; these do not establish dividend eligibility or total return.

Only closed XNYS regular-session candles are admitted. The exchange calendar
handles weekends, holidays, early closes and daylight-saving changes. Complete
15-minute and hourly bars are built from contiguous 5-minute bars anchored to the
open. The last 30 minutes of a normal session are not mislabeled a full hour.
Required intraday/daily views need at least 21 closed bars, no gaps, and the
latest complete interval. SMA50 remains missing without 50 bars.

Monthly and yearly candles require every daily session in each completed period;
the in-progress month/year is excluded. Missing older periods remain explicit.
At least the latest completed month/year must be available. The 180d view is
exactly 180 calendar days ending on the latest closed daily session, not 180
trading sessions, a six-month bar, or independent repeated samples. Missing daily
coverage blocks its completeness. One-minute data is optional and never invented.

Every report includes:

- Bot identity, reasons, score/threshold, gates and retained features, verbatim.
- Source/candidate/input hashes, actual quote times, ages, bid/ask/spread/depth.
- Recent OHLCV, direction, body, upper/lower wick, close-in-candle position,
  quote position relative to the most recent candle, and missing history.
- SMA20/50 distances, simple RSI14 and ATR14, three-bar returns, prior-20-bar
  high/low and range position, relative volume and closed-bar VWAP proxy.
- Supporting and opposing directional context, invalidation and explicit gaps.
- Corporate-action/price-adjustment warnings; raw price gains are not total
  return. An ex-dividend gap is not automatically evidence of selling pressure.
- Full execution model inputs/outputs, embedded friction, conditional fill
  assumptions, cash/share reconciliation and any modeled loss.

Diagnostics are not a claim that the bot used these facts. Model score is not a
calibrated win probability. Prior extremes are not independently validated
support/resistance. RSI/ATR use simple averages, not Wilder smoothing. Relative
volume is not time-of-day adjusted and typical-price VWAP is not tick VWAP.

## Conditional Fill Contract

An affirmative, fresh BUY starts one simulated market submission using $100
virtual cash and a $1 reserve. No user cash balance is inferred. A second packet
must carry the same candidate/evidence kind and `fill_quote` with an independent
snapshot timestamp at least 120 ms after submission, before a 60-second deadline.
The quote must be fresh within 30 seconds, in the regular session, and at most
25 bps wide. A later, separately recorded SELL can close exactly the one virtual
share. There is no forced profitable exit, no sale of actual holdings and no
automatic re-entry. Expiry, no fill and ambiguity never retry automatically.

The existing execution simulator estimates prices. Its fill includes modeled
fees/friction once. A whole-share fill is explicitly a conditional assumption,
not an observation or an inference from a fractional fill-quality score. Model
rejection or partial-fill-watch cannot become a full fill. Missing measured
one-minute volatility uses a disclosed zero baseline, not a worst-case estimate.
The synthetic scenarios also exercise rejection, no-fill and partial ambiguity.

No dividend is assumed. News/macro risk, valuation, account/settlement/tax status,
calibration and out-of-sample economics remain unverified. A completed conditional
round trip proves only the tested wiring and virtual arithmetic, not profitability,
broker execution or readiness for unattended trading.
