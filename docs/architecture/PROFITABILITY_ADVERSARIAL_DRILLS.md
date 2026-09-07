# Profitability Adversarial Drills

## Purpose

The adversarial profitability pack tests economic failure modes that are separate
from host recovery, broker reconciliation, replay integrity, and the historical
financial-crisis drills. It is deterministic, candidate-bound, paper/research
only, and resource bounded.

Run the complete pack:

```bash
./scripts/ops/opsctl.sh profitability-adversarial-drill --json
```

Run one module by canonical ID or alias:

```bash
./scripts/ops/opsctl.sh profitability-adversarial-drill --scenario capacity --json
./scripts/ops/opsctl.sh profitability-adversarial-drill --scenario whipsaw --json
```

The latest artifact is
`governance/research/profitability_adversarial_drill_latest.json`.

## Fourteen Scenarios

1. `regime_transition_whipsaw` suppresses weak transition signals and preserves confirmed recovery re-entry.
2. `net_alpha_break_even_ladder` finds the modeled cost boundary where gross edge no longer clears the required margin.
3. `capital_capacity_scaling` builds full-fleet capital, cost, impact, participation, and unit-feasibility curves.
4. `liquidity_evaporation_partial_fill` blocks new exposure after depth disappears while preserving a reduce-only exit.
5. `correlation_crowding_collapse` rejects a portfolio whose qualified sleeves become one correlated factor.
6. `gradual_strategy_decay` distinguishes a stable control from a slowly deteriorating edge with bounded Page-CUSUM diagnostics.
7. `point_in_time_contamination` proves that later revisions cannot enter an earlier replay.
8. `recovery_reentry_timing` compares immediate, staged, and delayed recovery paths after costs and drawdown penalties.
9. `benchmark_opportunity_cost` separates prudent abstention from inactivity that trails cash, SGOV, or passive exposure.
10. `signal_horizon_conflict` keeps intraday, swing, and long-term signals inside their owning sleeve horizon.
11. `corporate_action_calendar` checks security identity, splits, dividends, symbol changes, delistings, options adjustments, DST, half-days, and futures rolls.
12. `volatility_dependent_data_loss` detects missing-not-at-random observations and rejects stale last-good substitution.
13. `false_model_consensus` clusters correlated models before counting independent votes.
14. `portfolio_path_dependency` proves that equal endpoint returns can have different drawdown and stop outcomes.

## Capacity Model

The capacity module covers the current `111` runtime sleeves and all `12`
objective classes. The `25` `control_only` sleeves are marked not applicable
and receive no return, cost, or deployable-capital claims. The `86` trading
sleeves receive:

- twenty-two account-capital tiers from `$200` through a `$1,000,000,000`
  long-range research endpoint;
- normal, wide-spread, thin-liquidity, high-volatility/latency, flash-crash
  dislocation, and crowded-exit states;
- 5%, 10%, and 25% allocation-fraction sensitivity;
- objective- and asset-aware gross-edge, spread, fee, slippage, volume,
  volatility, impact, contract-multiplier, margin, and minimum-unit assumptions;
- risk-budget and assumed-loss limits before target notional is tested;
- minimum tradable-unit feasibility, participation, impact, total cost,
  net-alpha, expected net dollars, stress survival, and breakpoints;
- explicit `$200` canary and `$1,000,000,000` target-scale snapshots;
- inheritance counts for the `879` hot strategies and the `12,000`-strategy
  research library.

A curve that remains clear at the highest diagnostic tier means only that the
configured upper bound was not reached. It does not certify that amount as
deployable.

Every trading sleeve retains visible calibration requirements for
candidate-forward strategy edge lower bounds, realized volume, spread, slippage,
fill ratio, market impact, independent days, and independent symbols. A strategy
inherits its sleeve curve only as a planning diagnostic. Live scaling requires a
narrower candidate-bound strategy curve and the existing portfolio, risk,
promotion, broker-truth, and operator-release gates.

## Authority Boundary

The pack:

- starts zero persistent processes;
- performs zero network, market-data, or broker requests;
- submits zero paper or live orders;
- cannot change actions, allocations, risk limits, labels, thresholds, candidate
  state, promotion state, or historical outcomes;
- cannot count synthetic parameters as organic evidence;
- cannot certify future profitability.

An `A+` is therefore a control grade: all requested scenarios executed,
detected their designed failure modes, respected candidate immutability, and
stayed inside the resource and authority contracts. It is not a raw
profitability grade, capacity attestation, or live-release decision.

## Soak Treatment

Running the drill does not reset or rewrite soak history. Accepting code that
changes this surface should preserve cumulative segmented soak time and begin a
new clean affected-scope segment for the accepted candidate. The drill itself
cannot accept a candidate or modify soak accounting.
