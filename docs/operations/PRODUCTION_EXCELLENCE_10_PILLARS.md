# Production Excellence: Ten-Pillar Release Contract

This contract separates a healthy paper runtime from proof that the system is ready to be considered for a microscopic live-money canary. Paper collection may remain healthy while production evidence is still pending. Missing evidence never earns an A+ and never grants live execution authority.

## The Ten Pillars

1. Frozen production candidate with complete runtime-source coverage, per-file manifests, per-scope fingerprints, and a hash-chained acceptance log.
2. Scope-aware unchanged-candidate validation: operations require 72 hours and three completed XNYS sessions; data and dependencies require 120 hours and five sessions; promotion logic requires 336 hours and ten sessions; strategy, execution, and risk remain at 720 hours and twenty sessions.
3. Ten verified recovery drills covering auth, broker network, process, reboot, disk, external storage, memory, database, market data, and the order lifecycle.
4. Durable live execution with pre-trade limits, read-only release boundaries, and a transactional order-intent ledger.
5. Independent fill evidence that excludes model-derived fills and pre-candidate samples.
6. At least four real, evidence-backed promotion candidates with complete promotion packets.
7. Positive post-cost forward profitability with a positive 95% lower confidence bound, bounded drawdown, and sleeve diversity.
8. A controlled canary with at least 400 baseline and 400 candidate samples, at most 1% initial weight, and long-only cash equities.
9. Non-gameable grading with zero credit for missing evidence, explicit raw-versus-controlled labels, provenance checks, and tamper-evident logs.
10. Fresh security, alerting, backup/restore, blackstart, rollback, and institutional operating evidence.

## Candidate Workflow

Initialize the first intended candidate after its source is reviewed:

```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh production-excellence --apply --initialize-candidate --json
```

Inspect without mutating state:

```bash
./scripts/ops/opsctl.sh production-excellence --json
./scripts/ops/opsctl.sh source-mutation-guard --json
```

When a reviewed source change is necessary, run focused regressions and accept the exact working-tree fingerprint explicitly before committing it. The pre-commit hook blocks candidate-scoped source that does not match the accepted fingerprint:

```bash
./scripts/ops/opsctl.sh production-excellence --apply \
  --accept-candidate-change \
  --change-reason "Describe the reviewed production change" \
  --json
```

Acceptance creates a new candidate generation. The event records the exact added, modified, and removed files, source-coverage receipt, affected scopes, and reason. Only scopes affected by the changed source fingerprint restart their evidence windows. Historical raw profitability remains intact; the system measures a separate post-candidate forward cohort instead of rewriting old losses.

Candidate acceptance is intentionally never a self-healing action. The periodic source guard, readiness refresh, cross-system drift mesh, and pre-commit hook may detect, report, and fail closed on drift, but only an explicit operator-reviewed command may advance the immutable generation. A newly added runtime source must match the declared inventory and belong to at least one scope; otherwise acceptance is refused.

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

`config/candidate_scope_validation_v1.json` is the canonical elapsed-evidence policy. Every blocking scope must satisfy both its credited wall-clock requirement and its completed-session requirement. Partial market sessions do not count. Planned maintenance preserves evidence earned before the event, but offline hours and maintenance-interrupted sessions earn no credit. If the XNYS calendar or policy cannot be verified, session credit is zero. An unknown scope inherits the strict `material_trading` tier.

Candidate acceptance resets only the scopes touched by the reviewed source fingerprint. Operations-only changes therefore stop restarting strategy, execution, and risk evidence, while any material trading change still carries the full 720-hour and twenty-session burden. Exact generated outputs declared by the generated-artifact policy do not mutate the candidate; canonical documentation, tests, policy, and runtime source still require explicit acceptance and at least the operations tier.

The cumulative segmented main-soak clock remains visible for developmental and operating-history review, but it is not promotion credit. The scope-aware receipt in `governance/health/continuous_soak_integrity_control_latest.json` is authoritative for elapsed candidate validation. The separate seven-day sustained-all-gates canary interlock remains in force after scope validation; neither receipt grants order authority.

The livefeed row `[production-excellence]` is advisory to paper runtime and authoritative for live-money consideration. A blocked pillar therefore means "evidence is not yet sufficient for live money," not "stop healthy paper collection."
