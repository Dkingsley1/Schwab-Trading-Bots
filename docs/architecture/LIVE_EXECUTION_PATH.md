# Live Execution Path

## Purpose

This is the future microscopic-canary execution contract. It reduces avoidable order-path failures without claiming that software can make live trading risk-free or profitable. Live orders remain disabled until every independent promotion, economic, broker, risk, soak, and operator gate passes.

## Authority Boundary

The rehearsal, envelope, ledger, recovery, and reconciliation layers cannot create a signal, promote a candidate, select an account, unlock live execution, or place an order by themselves. The runtime must still satisfy `MARKET_DATA_ONLY=0`, `ALLOW_ORDER_EXECUTION=1`, an explicit candidate-bound canary allowlist, a pinned broker account, broker readiness, risk controls, and operator release.

## Ordered Flow

1. A promoted strategy publishes a candidate-bound intent with action, quantity, symbol, strategy receipt, evaluation digest, and event time.
2. The exclusive live lane rejects missing, expired, future-dated, duplicate, or stale intents before broker work begins.
3. Startup and periodic recovery reconcile unresolved durable intents against broker truth before new live intent consumption.
4. Account reconciliation supplies current positions and a timestamped account-snapshot digest.
5. The order builder normalizes symbol, asset type, order type, limit semantics, quantity, session, and duration through the broker capability contract.
6. A short-lived envelope seals the exact intent and broker request to the accepted candidate, pinned account hash, account snapshot, risk-policy hash, quote timestamp, bid, ask, and deterministic client-order ID.
7. The final production firewall verifies the envelope, quote age, future-clock skew, spread, request parity, symbol lifecycle, canary stage, account identity, position transition, and every existing global risk and promotion gate.
8. The durable ledger records `reserved` and then `submitting` before the broker mutation can leave the process.
9. Place, cancel, and replace are one-shot mutations. No mutation is blindly retried after possible dispatch.
10. A confirmed response advances legal broker-order state. An ambiguous response advances to `submit_unknown` or `cancel_unknown`, raises the halt boundary where required, and permits reconciliation only.
11. Broker reads reconstruct accepted, rejected, open, partial, filled, canceled, or unresolved state with monotonic fill accounting and immutable terminal states.
12. Position truth, fills, costs, slippage, markouts, paper/live divergence, and canary budgets feed evidence and rollback decisions. They cannot retroactively alter the submitted intent.

## Retry Contract

| Operation | Maximum dispatch attempts | Recovery rule |
| --- | ---: | --- |
| Place order | 1 | Query by broker or client identity; reconcile before any new intent |
| Cancel order | 1 | Query current order state; preserve `cancel_unknown` until proven |
| Replace order | Disabled for initial canary | Cancel, reconcile, then require a newly sealed intent |
| Account, position, quote, or order read | Bounded transient retries | Fail closed when truth remains unavailable or stale |

This follows the principle that a timeout after a mutation does not prove failure. Repeating the mutation can duplicate the economic action even when the first response was lost.

## Crash And Restart Rules

| Durable state found at restart | Meaning | Required action |
| --- | --- | --- |
| `reserved` beyond grace | Broker dispatch was not recorded | Close as rejected without broker mutation |
| `submitting` beyond grace | Dispatch may have occurred | Mark `submit_unknown` and reconcile broker truth |
| `submit_unknown` or `cancel_unknown` | Outcome is unresolved | Block new dependent work and reconcile only |
| Open or partially filled | Broker owns current truth | Rebuild materialized state and position awareness |
| Terminal | Final economic state is immutable | Verify chain and retain for evidence |

## Rehearsal

```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh live-execution-rehearsal --json
```

The rehearsal runs without a broker client or network access. It verifies 14 structural controls and ten negative paths covering tampered payloads, stale and future quotes, wide spreads, expired envelopes, candidate drift, account drift, missing or stale account snapshots, and risk-policy drift. An A+ rehearsal result proves only that the local control path passed those checks.

## External Design Influences

- SEC Rule 15c3-5: pre-trade financial and regulatory controls, restricted access, and regular review.
- FINRA Regulatory Notice 15-09: change control, segregated testing, pilot deployment, real-time monitoring, disable mechanisms, reconciliation, and capacity controls.
- FIX order-state transitions and Nasdaq OUCH semantics: explicit accepted, rejected, executed, canceled, and pending states.
- Idempotent API design: mutation retries require a server-recognized idempotency identity and matching intent; absent that proof, reconcile instead of retrying.

These are design influences, not claims of exchange certification, broker-dealer compliance, native FIX/OUCH connectivity, or regulatory approval.

## Promotion Debt That Cannot Be Engineered Away

The live path remains locked while any candidate-specific economic, drawdown, independent-fill, paper/live-divergence, broker entitlement, clean-window, operator-release, or external-attestation requirement is incomplete. Structural controls reduce execution risk; they do not guarantee fills, uptime, returns, or the absence of production incidents.
