# Supervised Live Canary Runbook

This runbook governs the first microscopic live-money order. It does not grant live execution authority. The default remains `MARKET_DATA_ONLY=1`, `ALLOW_ORDER_EXECUTION=0`, and the paper-trade lock present.

## Scope

- Designated account: `schwab_roth_ira_primary`, the operator-verified Roth IRA ending in `5625`.
- Capital: settled cash at or above the configured $200 canary cap.
- Route: `dividend_liquid_etf_candidate_v1` only.
- Stage 1 symbol: `SCHD` only.
- Order: one whole share, explicit limit, `NORMAL` session, `DAY` duration.
- Open-order limit: one.
- Loss limits: $2 daily and $10 cumulative.
- Unfilled-order deadline: 60 seconds, followed by cancel and reconciliation. No replace.
- Supervision: Schwab web or mobile must remain available to the operator for independent cancel or emergency control.
- Retirement wrapper: existing Roth cash only. The plan assumes no new IRA contribution and cannot infer contribution eligibility or replaceable contribution capacity.

## Hard Prerequisites

1. Complete the required organic soak and candidate validation. Runtime, profitability, independent-fill, and promotion evidence must remain earned evidence; grades cannot substitute for observations.
2. Refresh Schwab account truth and confirm in Schwab that the designated account has at least $200 of settled cash, no pending deposit, no account restriction or call, no borrowing authority in use, and no unexpected open order.
3. Confirm the negative provider balance fields, if present, against the Schwab balances page. A provider `MARGIN` label or negative balance field is not accepted as settled-cash evidence.
4. Refresh tax history and review any same-symbol sale in the prior 31 days across visible taxable accounts. A taxable-account loss followed by a substantially identical Roth purchase may create a permanently disallowed loss under [IRS Revenue Ruling 2008-5](https://www.irs.gov/irb/2008-03_IRB).
5. Explicitly review Roth loss capacity. A canary loss reduces tax-advantaged account value and must not be treated as automatically replaceable contribution room. Any later contribution remains subject to the current [IRS IRA contribution rules](https://www.irs.gov/retirement-plans/plan-participant-employee/retirement-topics-ira-contribution-limits).
6. Finish source changes, tests, secret scanning, commit, and upstream synchronization. Create the immutable release manifest only from that clean commit.
7. Keep the global halt, operator stop, broker-boundary quarantine, and live-order ambiguity surfaces clear.

## Evidence Refresh

Run these from the project root while live execution is still disabled:

```bash
./scripts/ops/opsctl.sh schwab-account-hash-sync --json
./scripts/ops/opsctl.sh schwab-account-snapshot-refresh --json
./scripts/ops/opsctl.sh account-position-study --json
./scripts/ops/opsctl.sh schwab-tax-ledger-refresh --json
./scripts/ops/opsctl.sh portfolio-risk-ledger --json
./scripts/ops/opsctl.sh portfolio-allocator --json
./scripts/ops/opsctl.sh risk-service --json
./scripts/ops/opsctl.sh live-order-ledger --json
./scripts/ops/opsctl.sh live-canary-preflight --json
./scripts/ops/opsctl.sh live-canary-dress-rehearsal --symbol SCHD --json
```

The hash sync discovers connected accounts, maps them only through the operator-verified last-four aliases, and stores the opaque routing references in the macOS Keychain. It never writes raw hashes to the repository or grants live authority. The live runtime then pins `SCHWAB_ACCOUNT_HASH` to the Roth account policy selected by the canary plan.

The preflight and dress rehearsal are expected to remain `ready_locked` until funding, the release manifest, the short-lived operator attestation, the normal market session, and all earned-evidence gates are ready.

## Connected Read-Only Dress Rehearsal

The connected dress rehearsal is the last safe step before a supervised canary window. It refreshes all connected Schwab account truth, fetches a real `SCHD` provider quote, selects the designated account only through its opaque Keychain binding, and constructs the exact one-share `LIMIT` / `NORMAL` / `DAY` request in memory. The persisted artifact contains only the account-reference digest and compact quote evidence, never the raw account number, routing hash, or provider payload.

Review `governance/health/live_canary_dress_rehearsal_latest.json` for:

- broker-visible settled cash and any funding or settlement shortfall;
- the projected post-fill `SCHD` quantity and remaining settled cash;
- confirmation that buying power and provider margin fields were not used as cash;
- preservation of existing covered-position collateral;
- quote age, spread, venue, and provider-payload receipt hash;
- the exact redacted order preview and submit, acknowledgement, fill, cancel, and reconciliation expectations.

An `ok=true` result means the connected read-only control completed without a broker mutation. It does not mean `canary_ready=true`. Run with `--require-canary-ready` only during the final supervised window when a non-ready result should return a failing exit code.

## Immutable Release

The candidate must be committed, pushed, clean, and synchronized before creating the canary release receipt:

```bash
./scripts/ops/opsctl.sh release-freeze --activate-days 1 --reason supervised_live_canary --write-release-manifest --json
```

If any tracked source changes afterward, stop. Re-run tests, publish a new commit, and create a new manifest. Do not reuse the prior release receipt.

## Operator Attestation And Allowlist

After verifying the current Schwab balances and restrictions, issue a private, short-lived attestation and then a candidate/account-bound stage-1 allowlist:

```bash
./scripts/ops/opsctl.sh live-canary-preflight --issue-attestation --issue-allowlist --stage 1 --settled-cash-usd 200 --duration-minutes 60 --confirmation "I CONFIRM SUPERVISED LIVE CANARY" --confirm-all --confirm-retirement-account-risk --json
```

This command refuses to issue the attestation unless the broker snapshot also shows the configured cash floor. A Roth account additionally requires the explicit retirement-risk flag, which records loss-capacity, contribution-capacity, and cross-account wash-sale confirmations. It writes owner-only runtime files, binds them to the selected account hash and current candidate, and still does not arm execution.

Run the check again immediately before submit:

```bash
./scripts/ops/opsctl.sh live-canary-dress-rehearsal --symbol SCHD --require-canary-ready --json
./scripts/ops/opsctl.sh live-canary-preflight --symbol SCHD --action BUY --json
./scripts/ops/opsctl.sh live-canary-control --json
./scripts/ops/opsctl.sh live-canary-readiness --json
```

All four must report ready. The exchange-calendar check must show an open `XNYS` session outside the five-minute opening and closing buffers.

## Supervised Submit

1. Keep Schwab web or mobile open and confirm the intended account and current quote independently.
2. Arm only the dedicated live execution lane using the established operator-controlled runtime environment. Do not remove `PAPER_TRADE_LOCK.flag`; it remains a required safety marker.
3. Submit one stage-1 `SCHD` share with an explicit cent-valid limit price no greater than the $100 order cap.
4. Verify the durable intent exists before broker dispatch, then verify the Schwab order identifier, ledger state, account position, and fill price.
5. If the order is not filled within 60 seconds, cancel it. Do not replace it. Reconcile the order and position before any later attempt.
6. Stop after one reconciled order. Do not progress stages in the same session.

## Immediate Stop Conditions

Stop and set the operator or global halt when any of these occurs:

- Account, candidate, route, symbol, or quote provenance mismatch.
- Stale risk, account, tax, release, or order-ledger evidence.
- Broker response ambiguity, unknown order state, duplicate intent, or position mismatch.
- Spread above policy, partial fill outside the expected state machine, or price outside the explicit limit.
- Schwab UI becomes unavailable to the operator.
- Any loss boundary, account call, restriction, borrowing indication, or uncovered option condition appears.

## Rollback And Closeout

1. Use Schwab web, mobile, or phone support for independent cancellation if the software path is uncertain.
2. Keep live execution disarmed after the first order.
3. Reconcile broker orders and positions, refresh account truth, refresh the tax ledger, and record the canary result.
4. Preserve the live-order ledger and release manifest. Never delete or rewrite uncertain-order evidence.
5. Resume paper-only collection after the incident or canary closeout is documented.

The first canary validates execution plumbing and controls. It does not prove future profitability, justify scaling, or replace the required post-cost and out-of-sample evidence.

## Post-Canary Reconciliation

The first successful fill spends the issued allowlist's one-new-entry budget. That prevents a second entry from following automatically. The firewall still preserves an explicitly verified reduce-only exit so the canary position can be closed without weakening the safety boundary.

After every terminal entry or exit fill, leave live execution disarmed and run:

```bash
./scripts/ops/opsctl.sh live-order-ledger --json
./scripts/ops/opsctl.sh schwab-account-snapshot-refresh --json
./scripts/ops/opsctl.sh account-position-study --json
./scripts/ops/opsctl.sh live-canary-closeout --intent-id INTENT_ID --json
./scripts/ops/opsctl.sh live-canary-closeout --intent-id INTENT_ID --capture --json
./scripts/ops/opsctl.sh live-canary-graduation --json
```

The live execution lane must reconcile the broker order before closeout. `live-order-ledger` verifies the local durable chain; it does not independently fetch or mutate a Schwab order. Preview the closeout before using `--capture`. The recorder appends only when the terminal hash-chained fill, exact candidate/account/route, fresh post-fill account study, equity position delta, isolated cash delta including a conservative cost floor, and account safety state all agree. It stores hashes instead of raw Schwab account and broker order references. A stale or pre-fill account snapshot stays `reconciliation_pending`; missing organic evidence stays `ready_idle` and is not reported as system degradation.

An entry and its later reduce-only exit each need their own closeout receipt. Never edit or delete a receipt. A duplicate, tampered receipt, unresolved broker operation, identity mismatch, account call, pending deposit, borrowing indication, uncovered short option, or non-reducing exit fails closed.

## Earned Stage Progression

A single successful fill is execution-plumbing evidence only. Stage 1 completion and stage 2 review require all of the following:

1. At least three fully reconciled entry/exit round trips.
2. At least three independent trading days.
3. At least two source-backed regime buckets.
4. Positive total post-cost P&L and positive mean post-cost return.
5. At least half of observed trading days positive.
6. Maximum fill deviation at or below 20 bps.
7. Daily loss at or below $2, cumulative drawdown at or below $10, and no more than two consecutive losing round trips.
8. No safety violation, broker ambiguity, identity mismatch, or manual evidence override.

Run `./scripts/ops/opsctl.sh live-canary-graduation --json` to review progress. A review-eligible result is still advisory. It cannot issue an allowlist, progress a stage, change limits, or submit an order. Stage 2 requires a newly reviewed policy boundary, immutable release, operator attestation, and candidate-bound allowlist. Stage 3 requires at least six total reconciled round trips; its own allowlist remains explicit and short-lived.

## Capital Ladder

Capital scaling is separate from symbol-stage progression and is never automatic:

| Review tier | Proposed capital | Minimum evidence |
| --- | ---: | --- |
| Micro validation | $200 | Current canary boundary only; no automatic follow-on order |
| Micro two | $400 | 10 round trips, 5 days, 2 regimes, positive 95% normal-approximation lower confidence bound, positive benchmark excess, and actual fee evidence |
| Micro four | $800 | 25 round trips, 10 days, 3 regimes, positive 95% normal-approximation lower confidence bound, positive benchmark excess, and actual fee evidence |
| Foundation and personal | $1,600-$100,000 | Escalating live-history, regime, confidence-bound, benchmark, fee, sleeve-breadth, and capacity-headroom evidence |
| Advanced personal | $250,000-$1,000,000 | Personal evidence plus independent model validation, financial-controls review, and isolated multi-account reconciliation |
| Professional multi-account | $2,500,000-$10,000,000 | Advanced controls plus broker failover, formal compliance review, and institutional transaction-cost analysis |
| Institutional | $25,000,000-$100,000,000 | Professional controls plus custody architecture, independent risk oversight, and business-continuity evidence |
| Large institutional | $250,000,000-$1,000,000,000 | Institutional controls plus market-impact governance, staff-duty separation, and an external audit program |

No tier permits martingale sizing, loss chasing, automatic doubling, or averaging down merely because a trade lost. A profitable first trade is not a scaling signal. Each accepted tier requires explicit operator review, a new candidate-scope validation, a new immutable release, a new attestation, and a new bounded allowlist. Failure at any gate means hold the current cap, return to paper collection, or roll back.

The funded capital ladder and the organic growth ladder answer different questions. A deposit can support a later funded tier after its evidence review, but it does not prove that the system grew the seed. Organic progress begins at `$200` and adds only post-cost P&L from fully reconciled canary round trips. Account deposits, total account equity, unrelated positions, unrealized P&L, unattributed dividend income, and another account's outcomes do not count. Twenty-two tracked targets end at a `$1,000,000,000` long-range research ceiling. That endpoint is an ambition and stress-test boundary, not a forecast, certified capacity, or promise of profit.

For each next target, the selector produces a target-capital sleeve plan independently from the current plan. This allows a one-sleeve `$200` canary to grow toward a broader plan when measured correlation, route compatibility, capacity headroom, and earned evidence support diversification. Positive profit can produce a bounded reinvestment proposal with a retained reserve; losses or excessive growth drawdown force seed defense. The proposal remains inactive until reviewed. Only `personal_brokerage` is currently enabled; advanced, professional, institutional, and large-institutional tiers stay fail-closed until their named controls are independently evidenced.

The policy applies to every classified account policy key, but evidence never pools between accounts. Each Roth, traditional IRA, taxable, or cash account keeps its own candidate binding, post-cost P&L, drawdown, graduation receipts, preflight, and operator review. The current `$200` Roth canary remains the separately selected active account; cataloging another account does not activate it.

The architecture may be transferred to another computer, but execution authority may not. Code, policy, account-policy keys, and non-secret evidence can move after integrity verification. Raw account identifiers, credentials, OAuth tokens, and Keychain bindings cannot. A new host must rebuild dependencies, verify policy hashes, rediscover account aliases, rebind the Keychain, obtain fresh OAuth, reconcile broker capabilities, reissue preflight, recalibrate execution, and pass the dress rehearsal. New-host live reactivation is always manual.

## Sleeve And Portfolio Selection

Before proposing a later canary tier or a broader application route, run:

```bash
./scripts/ops/opsctl.sh sleeve-scalability-selector --json
```

The selector answers a narrow question: which sleeve or low-correlation set is best-supported for the current candidate, designated account, configured route, active capital tier, and regime. A sleeve must earn candidate-bound sample and day depth, positive conservative post-cost evidence, persistence, current-regime compatibility, independent contribution, drawdown control, and enough independently calibrated capacity. Multi-sleeve plans additionally require measured common-day correlation below the policy ceiling; unknown correlation fails closed.

The twelve system goals progress from one independently qualified sleeve at micro scale through an eight-sleeve, five-times-headroom large-institutional capacity objective. The organic ladder separately measures whether each account's original seed has earned its way toward each larger target. Goal progress is evidence, not permission. The selector publishes advisory weights and notionals only, never writes the runtime allocator, pools account evidence, changes capital or stage limits, activates an account or host, issues an allowlist, or creates an order. Any selected plan still requires the normal immutable release, operator review, account preflight, and bounded canary controls.
