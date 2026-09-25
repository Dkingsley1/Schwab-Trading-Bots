# Supervised Broker Function Test

This operator-only workflow tests broker execution plumbing and buy-and-hold
accounting. It is separate from autonomous strategy promotion. Production soak,
profitability, independent-fill, and stage-graduation policies are unchanged;
test results never certify those policies or grant autonomous order authority.

## Reviewed Scope

- Designated Roth account only; existing settled cash, no borrowing.
- O only, buy and hold, $300 total budget including a $1 cost reserve.
- Maximum five whole shares; maximum buy limit $57.09 per share.
- Proposed entry price: the lower of the fresh Schwab bid and $57.09, rounded
  down to a cent. Size is the affordable whole-share quantity, capped at five.
  Five shares at the ceiling cost $285.45 before costs.
- This is a passive execution-price proposal, not a valuation assessment or
  assurance that O is cheap, profitable, or likely to fill.
- LIMIT, NORMAL, DAY, single equity leg only. No market fallback, replacement,
  automatic price chasing, follow-on purchase, dividend reinvestment, or sell.
- One entry attempt per test ID, retained in the native durable order ledger
  across restarts, new attestations, and release candidates. A rejection,
  cancellation, or ambiguous attempt does not authorize an automatic retry.
- A sell test is optional and separately operator-confirmed. It can sell only
  verified, unencumbered shares acquired by this test, never unrelated holdings.
- The budget caps purchase funding, not subsequent investment losses. There is
  no enforced dollar stop-loss, automatic liquidation, or guaranteed exit price.

## Commands

Run from the project root using the normal managed Schwab environment:

```bash
./scripts/ops/opsctl.sh supervised-broker-test status --json
./scripts/ops/opsctl.sh supervised-broker-test preview --json
./scripts/ops/opsctl.sh supervised-broker-test observe --json
```

`status` is offline and does not connect or create runtime authority. `preview`
refreshes account, release, tax, ledger, open-order, and quote evidence, proposes
the bounded limit, and reports exact technical and personal-review blockers. It
cannot issue attestation or submit/cancel/replace orders. `observe` reads broker
order/position/transaction truth and reconciles the already-existing test. It
does not require production promotion or create an order.

An additional purchase outside the test no longer has to look like unexplained
position drift. The observer reports the broker total, original test remainder,
and additional purchase quantity separately. It requires fresh account-bound,
complete transaction coverage, verified original test postings and individually
identified valid equity purchases to explain the total. Identical transaction
IDs are deduplicated; conflicting IDs, outside reductions, transfers, corporate
actions and incomplete evidence still require review. This is observation only:
the original sell lifecycle remains unchanged, outside shares cannot be sold by
the test, and cash/settlement proof is not implied by position reconciliation.

The separate native [purchase-proposal observer](PURCHASE_PROPOSALS.md) schedules
this read-only observation under the existing evidence cadence. It never invokes
submit. Observation now includes exact transaction/net-cash matching, settlement,
and comparable cash-baseline checks; missing evidence stays explicitly pending.

Only the operator runs `supervised-broker-test submit` from an interactive
terminal after reviewing the preview. The command displays the exact request,
requires current personal confirmations, current settled cash, and a typed
confirmation containing side, symbol, quantity, price, and Roth account intent.
It refreshes evidence after that confirmation and does not alter the approved
price or size. If the quote, source, identity, or risk checks change, it stops.
There is no unattended submit option, saved standing order permission, order scheduler,
or Codex automation. Neither implementing nor publishing this command executes it.

An optional SELL requires `--action SELL --quantity N --limit-price PRICE` and a
new interactive confirmation. Buying and holding is the default; a buy does not
prove the sell path. Exit orders remain explicit limit orders. The independent
Schwab UI is the operator's fallback for emergency control.

## Separate SCHD Session Test

The fixed `roth_schd_round_trip_001` policy is separate from the consumed O entry.
It permits one whole SCHD share, a $100 funding cap including $1 reserve, and
an explicitly supplied limit. The $99 policy ceiling is a safety cap, not a
suggested price. Each side requires its own interactive confirmation; a BUY
never schedules a SELL. Unrelated shares cannot be used for the exit test.

```bash
./scripts/ops/opsctl.sh supervised-broker-test status --symbol SCHD --session AM --json
./scripts/ops/opsctl.sh supervised-broker-test status --symbol SCHD --session PM --json
./scripts/ops/opsctl.sh supervised-broker-test observe --symbol SCHD --json
```

A read-only preview additionally requires `--quantity 1 --limit-price PRICE`,
where `PRICE` is the operator's current reviewed per-share limit, not a literal.
Only the operator can invoke interactive `submit`; this document does not
authorize an order or provide an attestation. Missing current cash, release,
risk or storage evidence still blocks submission.

The scoped lane supports LIMIT/DAY in NORMAL, AM or PM. AM is 07:00-09:25
Eastern and PM is 16:05-20:00 Eastern on regular full exchange sessions.
Holidays, early-close days, overnight, closed sessions and the final 75 seconds
are rejected. These windows follow the
[Schwab extended-hours specification](https://www.schwab.com/stocks/extended-hours-trading);
early-close support is deliberately not inferred. The production executor
remains NORMAL-only; O's existing policy remains unchanged.

Extended hours require fresh bid **and** ask timestamps for the requested
symbol (15 seconds maximum), positive displayed sizes, spread and price-distance
bounds, and a fresh dispatch-time session/quote check. A recent last trade is
not a substitute for current two-sided quotes. Session and DAY duration are
bound into exact confirmation, attestation, request hash, sealed preflight and
broker reconciliation. Reduced liquidity, wider spreads, partial fills and
failure to cancel require explicit review and independent broker access.

## Native SCHD Market Handoff

The explicit `--bot-market` mode is restricted to the same one-share SCHD test,
NORMAL/DAY, and the existing lifetime BUY/SELL intent IDs. It cannot retry or
re-enter by changing order type, candidate or process. The default/manual path
still requires a LIMIT; O and AM/PM never acquire a market-order fallback.

```bash
./scripts/ops/opsctl.sh supervised-broker-test readiness --symbol SCHD --json
./scripts/ops/opsctl.sh supervised-broker-test preview --symbol SCHD --bot-market --action BUY --json
./scripts/ops/opsctl.sh supervised-broker-test attestation-checklist --symbol SCHD --bot-market --json
```

Readiness refreshes risk, tax, release observation and order-ledger owners without
source acceptance, freeze activation, attestation or orders. Preview reads only
the pinned dividend-sleeve SCHD grand-master source, never an imported BUY. HOLD,
veto, stale/changed evidence, missing provider timestamps, unverified price basis
and incomplete candles remain blockers. The collector preserves provider quote
time, raw last price and realtime status; a quote replaced by a history fallback
does not become original quote evidence. A bounded, hash-bound candle-context
receipt identifies what capture existed when a new decision was logged. Context
is not claimed to be model input. Existing records cannot be repaired retroactively.

`preflight.technical_gate_diagnostics.risk_boundary` exposes the risk owner's
dependency blockers, observed ages and age limits. A freshly published degraded
risk report counts as refreshed observation but still blocks admission. A failed,
unchanged, stale or future publication cannot count as a successful refresh.

Risk refresh calls the execution-budget and sleeve-health owners in order. The
SLO owner reads the active process watchdog, not the retired daily event stream;
an explicit legacy `--event-log` remains a historical diagnostic and cannot
provide the current-source receipt required by the budget. The normal adaptive-ops
loop refreshes risk evidence after 2.25 minutes with a 30-second command bound.
It adds no scheduler, process restart or trading permission. Missing or stale
inputs produce a blocked zero-action budget, not a timestamp-only renewal.

The operator alone can use the corresponding interactive `submit` mode after a
reviewed clean release and all gates pass. Each side requires fresh account/risk
confirmations, current settled cash, an explicit market-price-risk answer and the
exact phrase naming BUY or SELL, one SCHD share, MARKET, Roth and unguaranteed
price. A fresh SELL decision and verified unencumbered test entry are required
for exit. Neither a successful BUY nor blanket chat approval schedules a SELL.

The handoff rechecks the same native decision after human review and again
validates evidence at dispatch. Two-sided current quotes must be at most 15
seconds old, positive-size, at most 25 bps wide and within 35 bps of the original
decision quote. The entry estimate includes a 35 bps cushion and $1 cost reserve
inside $100; these are admission estimates, not a broker-enforced market-price
cap. SELL preview requires the current bid above the verified entry fill, but
neither a higher fill nor profit is guaranteed. Actual execution legs, not
estimates, determine reconciliation. No automatic waiting, cancellation retry,
market replacement or autonomous execution is added.

The price-history response currently has no independently verified adjustment
receipt. It remains `provider_as_returned_not_independently_verified`; missing
proof is not cleared by a successful fetch, smooth chart or user attestation.
New quote/context fields take effect only when the reviewed collector source is
deployed; the command does not restart the platform or accept a release.

The existing adaptive-ops owner now maintains the shared candle cache on a
60-second regular-session cadence, with hourly out-of-session retention.
Immutable bounded captures let recent decisions retain their exact original
context across a refresh. Only verified expired online-cache files are pruned;
decision and order evidence is not garbage-collected by this owner. Its receipt
reports native decision age separately from worker health and never grants
clearance. `schd-decision-rehearsal charts --json` includes a recorded explanation
sample beside separately retrieved Schwab charts, not reconstructed model inputs.

## Shared Technical Checks

The scoped test does not wait for production's 720-hour strategy validation,
but it still requires a clean, accepted, immutable source release; designated
account binding; current account and tax evidence; explicit Roth review; cash
funding; no calls, borrowing or collateral violations; the normal exchange
session outside auction buffers (or the explicitly selected SCHD extended
session); current risk-service and durable-ledger proof;
storage above the existing pressure floor; no halt or broker quarantine; fresh
Schwab quotes; correct broker capabilities; and an empty current order inventory.

The API inventory covers the latest 60 days and fails closed if the response
could be truncated. The operator must additionally review **all** open orders in
Schwab and avoid concurrent manual account activity. Unknown is not empty.

The test uses the existing live-writer role and lease, exact-request hashing,
sealed execution evidence, and native hash-chained SQLite order ledger. The
production executor and its entry gates are not modified or armed. Test
attestation is owner-only, purpose/policy/candidate/account bound, and cannot be
reused as production attestation. Both test buys and sells retain technical
preflight, unlike an independent emergency risk-reduction procedure.

## Submission And Recovery

An intent is durably reserved before exactly one broker dispatch. A lost reply
remains uncertain and is not retried. Only scoped broker reads with matching
order identity and actual execution legs can confirm a fill; the limit price
is never substituted for an execution price.

While the operator's command is running, it watches the order and requests
cancellation at the 60-second deadline if still unfilled. A cancellation request
is not a cancellation confirmation. Network failure, process interruption,
missing order ID, partial fill, or changed broker identity requires reconciliation
and may require direct action in Schwab. The deadline is not guaranteed if the
host or network is unavailable. Phone supervision must therefore include access
to the correct Roth account and direct cancellation capability.

The entry is not sold automatically after filling. `observe` compares current
broker quantity to the recorded test fills after restart and reports mismatches
explicitly. Position consistency does not by itself certify cash/fee accounting.
Dividends require symbol-bound broker transactions, are deduplicated, and remain
unattributed rather than being labeled strategy alpha. Missing symbol, partial
history, or failed transaction reads remain incomplete. A current observation is
not a claim of uninterrupted monitoring; normal account/tax collectors keep their
existing cadence, and this workflow adds no background submit process.

An optional explicit broker cashBalance read before an operator-confirmed
dispatch is sealed into the ledger for later cash accounting. It is not settled
cash certification and does not replace personal attestations or funding checks.
Older test entries without that baseline cannot be retroactively certified by
matching a funding proxy to the current balance.

## Evidence

- Policy: `config/supervised_broker_test_v1.json`.
- Separate SCHD scope: `config/supervised_schd_broker_test_v1.json`.
- Session/calendar owner: `core/equity_order_sessions.py`.
- Core request, price, lifecycle, and reconciliation: `core/supervised_broker_test.py`.
- Operator command: `scripts/ops/supervised_broker_test.py`.
- Private attestation: `governance/runtime/supervised_broker_test_attestation.json`.
- Native order state: `governance/runtime/live_order_ledger.sqlite3`.
- Latest redacted report: `governance/health/supervised_broker_test_latest.json`.

Keep execution success, holding observation, dividend evidence, cash
reconciliation, and economic readiness separate. No report is a trade permission
or evidence of profitability merely because a broker operation succeeded.
