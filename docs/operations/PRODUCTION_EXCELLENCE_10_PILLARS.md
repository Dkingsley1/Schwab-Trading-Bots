# Production Excellence: Ten-Pillar Release Contract

This contract separates a healthy paper runtime from proof that the system is ready to be considered for a microscopic live-money canary. Paper collection may remain healthy while production evidence is still pending. Missing evidence never earns an A+ and never grants live execution authority.

## The Ten Pillars

1. Frozen production candidate with per-scope source fingerprints and a hash-chained acceptance log.
2. Clean 720-hour soak against that unchanged candidate, including a seven-day checkpoint.
3. Ten verified recovery drills covering auth, broker network, process, reboot, disk, external storage, memory, database, market data, and the order lifecycle.
4. Durable live execution with pre-trade limits, read-only release boundaries, and a transactional order-intent ledger.
5. Independent fill evidence that excludes model-derived fills and pre-candidate samples.
6. At least four real, evidence-backed promotion candidates with complete promotion packets.
7. Positive post-cost forward profitability with a positive 95% lower confidence bound, bounded drawdown, and sleeve diversity.
8. A controlled canary with at least 400 baseline and 400 candidate samples, at most 1% initial weight, and long-only cash equities.
9. Non-gameable grading with zero credit for missing evidence, explicit raw-versus-controlled labels, provenance checks, and tamper-evident logs.
10. Fresh security, alerting, backup/restore, blackstart, rollback, and institutional operating evidence.

## Candidate Workflow

Freeze the intended candidate only after its source changes are committed:

```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh production-excellence --apply --initialize-candidate --json
```

Inspect without mutating state:

```bash
./scripts/ops/opsctl.sh production-excellence --json
```

When a reviewed source change is necessary, accept it explicitly:

```bash
./scripts/ops/opsctl.sh production-excellence --apply \
  --accept-candidate-change \
  --change-reason "Describe the reviewed production change" \
  --json
```

Acceptance creates a new candidate generation. Only scopes affected by the changed source fingerprint restart their evidence windows. Historical raw profitability remains intact; the system measures a separate post-candidate forward cohort instead of rewriting old losses.

## Live-Order Safety

Every real broker submit requires a stable decision intent ID. The SQLite ledger reserves that ID transactionally before broker submission. A timeout or lost response becomes `submit_unknown`, and an uncertain cancellation becomes `cancel_unknown`; neither may be retried automatically. Broker reconciliation must move the original intent to a known state.

```bash
./scripts/ops/opsctl.sh live-order-ledger --json
```

After independently checking broker order history, reconcile an ambiguous intent with explicit evidence:

```bash
./scripts/ops/opsctl.sh live-order-ledger \
  --resolve-intent DECISION_ID \
  --resolution not_submitted \
  --evidence "Broker order history proves no order was accepted" \
  --json
```

Use `open`, `partially_filled`, `filled`, `canceled`, `rejected`, or `expired` only when broker truth proves that state. Open or filled resolutions require the broker order ID. The reconciliation itself becomes another hash-chained ledger event.

The production firewall also requires all ten pillars, explicit execution arming, market-data-only disabled, no halt flags, a live canary symbol allowlist, exact order-leg symbol matching, cash-equity BUY/SELL instructions, a defensible reference price, a maximum $100 order, and a $25 daily-loss cap. The controller reports readiness but never grants execution authority by itself. A separately marked emergency liquidation path may reduce broker-confirmed exposure even when entry gates or halt flags are active; it cannot create a new position.

## Recovery Evidence

A normal health artifact is not a drill. Record a drill only after an isolated exercise proves containment, recovery time, and no duplicate orders:

```bash
./scripts/ops/opsctl.sh chaos-drills \
  --record-drill broker_network_outage \
  --result pass \
  --recovery-seconds 42 \
  --containment-verified \
  --no-duplicate-orders \
  --evidence governance/evidence/drills/broker_network_outage.json \
  --json
```

Run the same evidence protocol for every required drill listed in `config/production_excellence_v1.json`. Fabricated or merely inferred drill completions must not be recorded.

## Soak Semantics

The initial candidate freeze begins a new 30-day production-excellence evidence window. Operational paper health is reported separately and continues collecting during evidence buildup. A later accepted change resets only the affected scopes, but the full soak uses the newest start among all soak scopes. Unaccepted drift blocks the soak clock until it is reviewed.

The livefeed row `[production-excellence]` is advisory to paper runtime and authoritative for live-money consideration. A blocked pillar therefore means "evidence is not yet sufficient for live money," not "stop healthy paper collection."
