# Live Multi-Asset Paper Trading Platform

## What This Showcases

- Live multi-sleeve shadow trading across Schwab equities, Schwab futures, Coinbase spot, and Coinbase futures
- Paper trading locked in parallel with decision generation and logging
- Specialist, master, and grand-master decision layers
- Continuous health tracking around live loops

## Architecture

```mermaid
flowchart TD
    Feed["Schwab and Coinbase live market feeds"] --> Loop["run_shadow_training_loop.py"]
    Loop --> Sleeves["Parallel sleeves / runtime lanes"]
    Sleeves --> Specialists["Within each sleeve: signal, options, futures, and infrastructure specialists"]
    Specialists --> Master["Within each sleeve: master and grand-master selection"]
    Master --> Allocator["Cross-sleeve allocator + execution bridge"]
    Master --> DecisionLog["Decision explanations + governance events"]
    Allocator --> Paper["Paper execution bridge"]
    Paper --> PaperLog["Paper trades + performance artifacts"]
    DecisionLog --> Training["Later behavior-model training"]
    PaperLog --> Training
```

## Repo Areas

- `scripts/run_shadow_training_loop.py`
- `scripts/run_all_sleeves.py`
- `scripts/run_parallel_shadows.py`
- `scripts/run_parallel_aggressive_modes.py`
- `core/base_trader.py`
- `core/live_execution_controls.py`
- `governance/health/data_ingress_latest_*.json`

## Talking Points

- The platform is designed to keep execution in paper mode by default while still collecting real decision traces.
- Options and dividend collection lanes are intentionally protected so they keep gathering data unless they materially underperform.
- Multiple live sleeves can run in parallel without sharing one brittle execution path.
- A sleeve is a runtime lane or container. Inside a sleeve, specialist sub-bots influence that sleeve's master and grand-master decision stack.
