# Supervised Broker Function Test

This operator-only workflow tests broker execution plumbing and buy-and-hold
accounting. It is separate from autonomous strategy promotion. Production soak,
profitability, independent-fill, and stage-graduation policies are unchanged;
test results never certify those policies or grant autonomous order authority.

## Reviewed Scope

- Designated Roth account only; existing settled cash, no borrowing.
- O only, buy and hold, $300 total budget including a $1 cost reserve.
- Maximum five whole shares; maximum buy limit $58.08 per share.
- Proposed entry price: the lower of the fresh Schwab bid and $58.08, rounded
  down to a cent. Size is the affordable whole-share quantity, capped at five.
  Five shares at the ceiling cost $290.40 before costs.
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

Only the operator runs `supervised-broker-test submit` from an interactive
terminal after reviewing the preview. The command displays the exact request,
requires current personal confirmations, current settled cash, and a typed
confirmation containing side, symbol, quantity, price, and Roth account intent.
It refreshes evidence after that confirmation and does not alter the approved
price or size. If the quote, source, identity, or risk checks change, it stops.
There is no unattended submit option, saved standing order permission, scheduler,
or Codex automation. Neither implementing nor publishing this command executes it.

An optional SELL requires `--action SELL --quantity N --limit-price PRICE` and a
new interactive confirmation. Buying and holding is the default; a buy does not
prove the sell path. Exit orders remain explicit limit orders. The independent
Schwab UI is the operator's fallback for emergency control.

## Non-Negotiable Technical Checks

The scoped test does not wait for production's 720-hour strategy validation,
but it still requires a clean, accepted, immutable source release; designated
account binding; current account and tax evidence; explicit Roth review; cash
funding; no calls, borrowing or collateral violations; the normal exchange
session outside auction buffers; current risk-service and durable-ledger proof;
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

## Evidence

- Policy: `config/supervised_broker_test_v1.json`.
- Core request, price, lifecycle, and reconciliation: `core/supervised_broker_test.py`.
- Operator command: `scripts/ops/supervised_broker_test.py`.
- Private attestation: `governance/runtime/supervised_broker_test_attestation.json`.
- Native order state: `governance/runtime/live_order_ledger.sqlite3`.
- Latest redacted report: `governance/health/supervised_broker_test_latest.json`.

Keep execution success, holding observation, dividend evidence, cash
reconciliation, and economic readiness separate. No report is a trade permission
or evidence of profitability merely because a broker operation succeeded.
