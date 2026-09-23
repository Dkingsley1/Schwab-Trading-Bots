# Native Purchase Proposals

This workflow automates observation and draft purchase evaluation inside the
platform. It cannot submit, cancel, replace, sell, reinvest, issue attestation,
accept source, or enable trading. It does not turn the supervised test into an
unattended executor.

## Persistent Policy

`config/purchase_proposal_policy_v1.json` binds the existing Roth/O test to its
original $300 lifetime budget, five-share maximum, $57.09 ceiling, $1 cost
reserve, and one lifetime entry attempt. The filled entry counts against that
scope across process restarts, new dates and release candidates. The remaining
dollars are not permission for a second purchase. SCHD and other accounts are
not added. The policy expires October 16, 2026 UTC; expiry never renews itself.

The policy is a proposal constraint, not standing order authorization. Edits
require source review. Revocation is a persistent local stop, checked before
broker reads; a scheduled refresh cannot remove it. There is deliberately no
resume or execution command.

```bash
./scripts/ops/opsctl.sh purchase-proposals status --json
./scripts/ops/opsctl.sh purchase-proposals evaluate --json
./scripts/ops/opsctl.sh purchase-proposals revoke --json
```

Status is offline and displays the last evaluation with its freshness. Evaluate
performs bounded read-only observation. Revoke disables this proposal observer,
not other platform components and not any order already at the broker.

## Native Cadence

The existing production-hardening watch's `accrual` profile owns the nominal
15-minute cadence. No new LaunchAgent, background daemon, or Codex automation
is created. Existing launchd lifecycle/resource admission still applies; actual
spacing can be longer during resource deferral, sleep, or other work. Each
connected child is restricted to `supervised_broker_test.py observe --json`,
with all execution environment flags off and a 90-second process-group bound.
The outer owner has 120 seconds. A singleton lock prevents concurrent readers.

Failed reads, timeouts, malformed data, invalid routes and unpublished reports
cannot reuse an old observation as fresh evidence. Routine passes write one
compact report; this workflow does not send recurring notifications.

## Price And Purchase Evaluation

An unused test scope can produce a review-only LIMIT/NORMAL/DAY draft after a
fresh, account/test-bound observation, accepted source, empty position and
normal exchange session check. An unused scope must also pass the existing
native technical preflight, including risk, halt, storage, ledger, account and
open-order checks; personal attestations are not granted by observation.
Price uses the lower of the fresh Schwab bid
and the configured ceiling; whole-share size respects observed funding and the
cost reserve. Stale/wrong-symbol/wide-spread quotes, unknown attempts, consumed
scope and unresolved accounting cause abstention. Drafts expire after 15
seconds and are not consumed by an executor.

Observed funding is a broker cash proxy, not certified settled cash. A passive
quote-based limit does not establish fair value or investment attractiveness.
Economic assessment and the operator's current personal review remain separate.
For the existing filled O test, the expected state is `holding_only`, with no
follow-on purchase, automatic sell, price chasing, or dividend reinvestment.

## Accounting

The supervised observer now reads an unfiltered account transaction window and
matches postings to the exact scoped broker order. It verifies actual equity
quantity, execution gross, cash direction, posted status, and transaction
identity. Identical postings are deduplicated; conflicting IDs and incomplete
windows remain debt. Effective charges are derived from broker net cash versus
verified execution gross, not an assumed zero fee or the approved limit.
Individual fee breakdown and tax characterization are not thereby certified.

Cash, positions, posted trade cash, and settlement have independent results.
Full account cash-delta proof requires comparable explicit broker cashBalance
snapshots before and after the trade plus complete intervening account activity.
New operator-controlled test attempts capture a redacted, account-bound cash
baseline in the existing ledger; the optional read does not change execution
authority or funding rules. Missing baseline evidence blocks cash certification.
An old funding proxy cannot be retroactively relabeled as a cashBalance snapshot,
even when the arithmetic matches. Broker settlement remains pending until its
explicit settlement confirmation is observed. An elapsed settlement date alone
is reported separately and never grants settled-cash certification; providers
without an explicit confirmation leave settlement pending. The bounded 59-day transaction lookback must not be
reported as complete lifetime history for an older holding.

Dividends remain symbol-bound, deduplicated observations, not automatically
attributed test profit. An unknown-symbol dividend is unresolved; unrelated
symbols are excluded. No reinvestment is authorized.

## Staged Validation

`governance/health/purchase_proposals_latest.json` reports the policy digest,
candidate, evidence ages, exact abstention reasons, accounting, and validation.
The owner-only runtime validation file counts at most one fully reconciled
observation per 15-minute slot, rejects repeated evidence and future timestamps,
and retains at most 256 slots. A policy or candidate change starts its own count.
Three independent UTC days and twenty observations are an observation threshold,
not execution readiness. This initial threshold is a conservative engineering
default, not evidence of strategy profitability.

Candidate-bound regression evidence, objective-appropriate economic evidence,
and explicit execution-policy review remain required independently. Meeting the
observation threshold cannot change stage, grow capital, approve another entry,
or activate an executor. Existing production promotion gates are unchanged.
