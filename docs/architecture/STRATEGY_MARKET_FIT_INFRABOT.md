# Strategy Market-Fit Infrabot

## Purpose

The strategy market-fit layer gives the infrastructure bots one bounded, repeatable way to inspect the complete `12,000`-strategy research catalog against current market conditions. It ranks research attention; it does not manufacture evidence or control trading.

The scanner reads every catalog row in batches of `500`, verifies identity and contract receipts, binds available evidence to the current production candidate, and emits compact top-strategy, family, sleeve, drift, and batch receipts. The full catalog is read transiently rather than copied into another large artifact.

## Five-Strategy Challenger Cohort

The first cohort uses five existing cold contracts. No new strategy identity or offspring is created.

| Role | Strategy ID | Signal family |
| --- | --- | --- |
| Mean reversion | `sleeve::stat_arb_market_neutral::research_residual_reversal_regime_conditioned::v1` | `mean_reversion` |
| Macro event | `sleeve::international_macro::research_policy_surprise_regime_conditioned::v1` | `event` |
| Carry/value | `sleeve::international_macro::research_carry_roll_down_stress_tested::v1` | `carry_value` |
| Liquidity/execution | `sleeve::international_macro::research_liquidity_stress_cost_adjusted::v1` | `liquidity_execution` |
| Volatility | `sleeve::variance_volatility_swaps::research_tail_convexity_cost_adjusted::v1` | `volatility` |

The cohort is capped at five slots and requires all five IDs to exist with distinct signal families. A fresh but `thin` regime source may support provisional ranking, but it queues the cohort. Only a fresh `ready` regime source permits `shadow_observe`, which is still counterfactual research with no order authority.

## Evidence Semantics

`market_fit_score` means research priority under the current regime. It is not expected return, probability of profit, a paper-trading instruction, or live approval.

A strategy may be described as `proven_working_now` only when all of the following are candidate-bound and present:

- a `validated_good` quality verdict;
- positive post-cost evidence;
- a positive clustered 95% lower confidence bound;
- current cost, liquidity, and capacity clearance.

Counterfactual `HOLD` paths and sleeve-level rank-IC diagnostics can improve research routing, but they do not count as trade profit. Missing evidence remains unknown rather than being called bad.

## Infrabot Roles

- `strategy_market_fit_scanner_infrabot` checks all catalog contracts in bounded batches.
- `challenger_cohort_curator_infrabot` keeps the five exact contracts present and shadow-only.
- `strategy_evidence_guard_infrabot` prevents rankings from being promoted into profitability claims.
- `strategy_market_drift_sentinel_infrabot` reports regime and top-ranking turnover without changing runtime behavior.

All authority flags are false. These infrabots cannot change actions, quantities, allocator weights, labels, candidate identity, soak history, promotion state, paper orders, or live orders.

## Operations

Run a full scan:

```bash
./scripts/ops/opsctl.sh strategy-market-fit --force
```

Run the cached path used by unattended operations:

```bash
./scripts/ops/opsctl.sh strategy-market-fit
```

Install the 30-minute low-priority launchd check:

```bash
./scripts/install_strategy_market_fit_infrabot_launchd.sh
```

The unattended runner uses a source-signature cache, background task policy, nice level `15`, a single maintenance slot, and a default `180`-second runtime limit. It may run during market hours but still respects resource-pressure controls.

## Artifacts

- Policy: `config/strategy_market_fit_infrabot_v1.json`
- Scanner health: `governance/health/strategy_market_fit_infrabot_latest.json`
- Five-slot cohort: `governance/research/strategy_shadow_challenger_cohort_latest.json`
- Strategy source: `governance/research/sleeve_strategy_library_latest.json`

The runtime refresh DAG regenerates the scan after strategy specialization, alpha-generation control, and alpha-concept measurement. Livefeed exposes checked coverage, cohort states, candidate binding, regime trust, drift, proof count, and order-authority flags.

## Soak Contract

This layer is observational metadata for the existing candidate. It does not reset the soak clock, relabel historical fills, pool evidence across candidates, activate cold strategies, or alter paper behavior. Any future runtime activation remains subject to the existing promotion, multiple-testing, capacity, risk, and live-release controls.
