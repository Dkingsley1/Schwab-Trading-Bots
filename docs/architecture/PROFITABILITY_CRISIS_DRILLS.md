# Profitability Crisis Drills

## Purpose

The crisis drill asks two separate questions:

1. Does the paper decision and execution path avoid adding risk when liquidity, spreads, volatility, funding, and correlation conditions break?
2. Does it become eligible to participate again after market quality and post-cost edge recover?

A system that passes only the first question may be safe but permanently inactive. A system that passes only the second may chase rebounds without surviving the collapse. The contract requires both behaviors.

## Scenarios

The versioned scenario set covers:

- `gfc_2008_financial_crisis`: dealer funding, Lehman, money-market stress, forced deleveraging, and policy stabilization.
- `covid_2020_pandemic_crash`: March liquidity failure, cross-asset liquidation, policy backstop, commodity aftershock, and early recovery.
- `us_regional_banking_2023`: duration losses, uninsured-deposit runs, regional-bank contagion, First Republic aftershock, and broad-market stabilization.

Official Federal Reserve and FDIC material anchors the event sequence. The phase-level return, spread, volatility, depth, quote-age, and latency inputs are deterministic diagnostic parameters. They are not reconstructed market ticks, forecasts, or executable signals.

## Contract

For every phase, the runner:

- compares BUY, HOLD, and SELL_SHORT after the shared execution simulator's costs;
- evaluates new long exposure through `evaluate_profitability_entry` with strict evidence enabled;
- exercises a separate reduce-only SELL for an existing long position;
- expects every severe phase to block new long exposure;
- expects the named recovery phase to permit a positive post-cost BUY opportunity;
- verifies official-source, scenario, policy, and candidate receipts;
- hashes the production candidate before and after the run.

The runner performs no network or broker access and has no paper-order, live-order, promotion, candidate-mutation, training-label, historical-rewrite, or profitability-claim authority.

## Run

```bash
./scripts/ops/opsctl.sh profitability-crisis-drill --json
./scripts/ops/opsctl.sh profitability-crisis-drill --scenario gfc_2008_financial_crisis --json
```

The latest artifact is `governance/research/profitability_crisis_drill_latest.json`. Its control grade measures whether the drill and safety expectations executed correctly. It does not upgrade raw profitability or count toward live-money promotion evidence.
