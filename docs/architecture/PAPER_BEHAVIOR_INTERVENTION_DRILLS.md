# Paper Behavior Intervention Drills

## Purpose

The paper behavior intervention pack tests whether bounded, candidate-bound controls improve the handling of already-authorized paper entries under known failure modes. It is a champion/challenger diagnostic, not a signal generator. The drill can recommend abstention or reduce entry size; it cannot create a trade, reverse an action, enlarge an entry, interfere with a `HOLD`, or block a `SELL` or reduce-only exit.

The production flow is:

`accepted candidate -> deterministic drill proposal -> paper-profitability single-writer admission -> expiring runtime overlay -> final paper action and quantity -> candidate-bound outcome labels`

The drill owns only `governance/research/paper_behavior_intervention_drill_latest.json`. `scripts/ops/paper_profitability_control.py` is the only writer allowed to admit a fresh, complete proposal into `governance/health/paper_runtime_profitability_controls_latest.json`. `scripts/run_shadow_training_loop.py` validates the admitted contract and current candidate again at the point where paper action and quantity are finalized.

## Scenario Pack

The fourteen scenarios cover:

1. stale quote abstention
2. negative post-cost edge
3. weak evidence abstention
4. illiquid-market abstention
5. thin-liquidity size throttling
6. deep-drawdown abstention
7. loss-streak throttling
8. crowding and concentration caps
9. regime-transition hysteresis
10. unconfirmed recovery re-entry
11. partial-fill inventory protection
12. disciplined additions to winning positions
13. horizon-conflict abstention
14. candidate maturity and action integrity

The evaluated interventions are stale-data abstention, post-cost edge gating, evidence-quality gating, liquidity throttling, drawdown and loss-streak throttling, crowding caps, regime hysteresis, staged recovery, partial-fill inventory guards, winner-add discipline, horizon-conflict abstention, and candidate-maturity scaling.

## Admission And Rollback

Admission requires all configured scenarios and cases, an `A+` drill grade, a fresh proposal inside its 24-hour TTL, exact candidate ID and generation, exact candidate-state file hash, exact policy hash, a valid proposal receipt, and a zero-authority resource receipt. A partial `--scenario` run is useful for diagnosis but cannot replace or qualify the full proposal.

The admitted overlay starts in `paper_probation`. Candidate mismatch, policy mismatch, expiry, malformed receipts, unknown interventions, or any authority violation disables it as a no-op while preserving the existing profitability and risk controls. No invalid overlay may widen risk.

## Evidence Boundary

An `A+` grades deterministic mechanics and invariant coverage only. Synthetic diagnostic improvement is not organic PnL, proof of alpha, promotion evidence, or a profitability guarantee. Advancement requires fresh, candidate-bound, intervention-tagged, post-cost paper fills. The contract never grants live execution, promotion, paper-order submission, broker access, threshold self-tuning, or candidate mutation.

Run the complete pack with:

```bash
./scripts/ops/opsctl.sh paper-behavior-intervention-drill --json
```

Inspect one scenario without changing admission eligibility with:

```bash
./scripts/ops/opsctl.sh paper-behavior-intervention-drill --scenario thin_liquidity_throttle --json
```

The unattended evidence and runtime-artifact refresh graphs run the complete drill after paper performance and before the paper profitability controller, so every admitted contract is fresh and candidate-bound.
