# Trading Behavior Drill Program

## Purpose

The trading behavior drill program tests how a production candidate responds to difficult market and execution conditions before a paper behavior overlay can affect runtime decisions. It is a deterministic, candidate-bound, paper-only evaluation layer. It does not contact Schwab or Coinbase, submit orders, retrain models, rewrite outcomes, promote candidates, change allocations, or enable live execution.

Drills do two different jobs:

1. **Diagnostic drills** expose weak behavior. They answer questions such as: Does the candidate abstain when quotes are stale? Does modeled edge survive costs? Does sizing fall when liquidity disappears? Does the system avoid correlated crowding and false model consensus?
2. **Intervention drills** prove that a narrowly defined paper-only response is monotonic. A proposed response may block a new BUY or reduce its existing size. It may not create a trade, enlarge a trade, reverse an action, suppress a SELL, or change live behavior.

An A+ drill grade means the control mechanics and authority boundaries passed. It is not organic profitability evidence and does not guarantee future returns.

## Execution Lifecycle

Every complete run follows the same lifecycle:

1. **Snapshot**: freeze the production-candidate ID, generation, candidate-file hash, program-policy hash, and all suite-policy hashes.
2. **Preflight**: verify the candidate is valid, policy receipts exist, requested suites resolve, and the program has no broker, network, order, promotion, candidate-mutation, allocation, risk-relaxation, or live authority.
3. **Execute**: run the suites serially under one run ID and bounded work/runtime budgets. Serial execution makes attribution clear and avoids competing with the live paper loops.
4. **Contain**: after every suite, compare the candidate-file hash with the frozen receipt. A mutation stops further execution and rejects admission.
5. **Evaluate**: normalize suite scores, scenario coverage, failed checks, resource use, and authority use.
6. **Compare**: compare only with a prior complete run that has the same candidate and input-policy signature. Lower scores, fewer passing scenarios, or more failed checks reject admission.
7. **Publish**: atomically update each detailed suite artifact, publish the program-level latest artifact, and retain a compact immutable run summary. Detailed payloads are not duplicated in history.
8. **Admit or reject**: a complete A+, non-regressed program may pass its behavior proposal to `paper_profitability_control.py`, the only runtime writer. The drill program cannot apply it itself.
9. **Observe**: intervention-tagged candidate-forward paper fills accumulate post-cost evidence. No threshold widening or profitability claim is allowed from synthetic drill results.

The program lock at `governance/locks/trading_behavior_drill_program.lock` prevents overlapping drill runs.

## Drill Suites

### Crisis Suite

The crisis suite exercises the shapes of the 2008 financial crisis, the 2020 pandemic liquidity break, and the 2023 regional-bank failures. The scenario parameters are explicit diagnostics rather than reconstructed historical ticks.

It verifies severe-phase new-long abstention, reduce-only exit availability, spread/latency/depth/partial-fill costs, staged recovery re-entry, and cost-adjusted action ranking across each crisis phase.

### Adversarial Profitability Suite

The adversarial suite challenges behavior that can look profitable in a backtest but fail in paper or live execution. It covers regime whipsaw, break-even edge, capacity, liquidity evaporation, crowding, gradual decay, point-in-time leakage, recovery timing, benchmark opportunity cost, horizon conflict, corporate actions, volatility-linked data loss, false consensus, and portfolio path dependency.

It verifies that profitability assumptions remain bounded after costs and that failure modes are detected without automatic tuning or capital expansion.

### Paper Behavior Intervention Suite

The intervention suite runs champion/challenger cases for stale data, post-cost edge, evidence quality, liquidity, drawdown and loss streaks, crowding, regime transitions, recovery, partial fills, winner additions, horizon conflict, and candidate maturity.

Allowed challenger outcomes are deliberately narrow:

- preserve the existing action and size;
- preserve BUY while reducing its size;
- convert a proposed BUY to HOLD;
- preserve SELL and HOLD exactly.

## Behavior Change Boundary

Drills do not directly make the strategy smarter and they do not manufacture alpha. They improve behavior by preventing known bad expressions of an otherwise valid signal. The expected benefit is lower avoidable loss, better cost discipline, less concentration, and cleaner recovery behavior.

The runtime path is:

```text
candidate decision
  -> admitted paper profitability controls
  -> admitted paper behavior intervention overlay
  -> paper order intent
  -> paper execution truth and post-cost evidence
```

The live path is unchanged. Live execution remains separately locked and requires its own earned evidence and explicit operator authority.

## Admission Rules

A behavior proposal is eligible for the single runtime writer only when all of the following are true:

- all three required suites execute;
- every required suite is A+ and has no failed checks;
- the run is non-regressed or establishes the first valid baseline;
- the candidate ID, generation, and file hash still match;
- all authority and resource counters remain zero;
- the intervention pack is complete and candidate-bound;
- the program and proposal are fresh;
- the proposal is paper-only and cannot enlarge, originate, or reverse a trade.

Any missing, stale, partial, mutated, over-budget, or mismatched evidence rejects the proposal while preserving existing controls.

## Evidence Files

- Program policy: `config/trading_behavior_drill_program_v1.json`
- Program latest: `governance/research/trading_behavior_drill_program_latest.json`
- Compact history: `governance/research/trading_behavior_drill_runs/*.json`
- Crisis details: `governance/research/profitability_crisis_drill_latest.json`
- Adversarial details: `governance/research/profitability_adversarial_drill_latest.json`
- Intervention details: `governance/research/paper_behavior_intervention_drill_latest.json`
- Admitted runtime view: `governance/health/paper_runtime_profitability_controls_latest.json`

## Commands

Run the complete program:

```bash
./scripts/ops/opsctl.sh trading-behavior-drill-program --json
```

Run one suite for diagnosis. A partial run is never admission eligible:

```bash
./scripts/ops/opsctl.sh trading-behavior-drill-program --suite profitability_crisis --json
./scripts/ops/opsctl.sh trading-behavior-drill-program --suite profitability_adversarial --json
./scripts/ops/opsctl.sh trading-behavior-drill-program --suite paper_behavior_intervention --json
```

Refresh the single-writer paper control after a complete passing program:

```bash
./scripts/ops/opsctl.sh paper-profitability-control --apply --json
```

The unattended readiness and runtime-artifact refresh graphs run the complete program after paper-performance refresh and before the paper-profitability controller.
