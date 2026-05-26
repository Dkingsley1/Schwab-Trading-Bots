# Auto-Refreshed Highlights

_Generated at 2026-05-26 20:53 UTC_

## Platform Snapshot

- Registered bots: `1651`
- Active bots: `1603`
- Live lane artifacts tracked: `89`
- Running lane artifacts: `10`
- Institutional readiness: `96.21/100` (`industry_leaning`)
- Live readiness: `100.00/100` (`ready`)
- Runtime separation: `degraded`
- Crypto source coverage: `14/14`
- Crypto news coverage: `7/7`
- Correlation mode: `exact`
- Last training result: `1 trained / 0 failed`
- PyTorch shadow-assist: `disabled` (MLX-primary live runtime)
- Autonomy control plane: `71.68/100` (`blocked`)
- Architecture upgrades: `9/12` ready proof surfaces

## Key Highlights

- Registry currently tracks 1651 bots with 1603 active across crypto_sub_bot, futures_sub_bot, infrastructure_bot, infrastructure_sub_bot, macro_sub_bot, options_sub_bot, signal_sub_bot lanes.
- Live ingestion is wired across 89 lane artifacts with 10 currently reporting `running`.
- Institutional-readiness score is `96.21/100` with status `industry_leaning` across 12 governance domains.
- Live readiness is `ready` at `100.00/100`, with broker/session ready=`True/True` and watchdog healthy=`True`.
- Runtime separation is `degraded` with contention score `2` and live-read-only=`True`.
- Autonomy control plane is `blocked` at `71.68/100`, with `39` triggered playbooks, `2` open incidents, and promotion state `repairing_readiness`.
- Architecture upgrade scoreboard tracks `9` ready proof surfaces out of `12`, with host profile `max_throughput` and portable proof `ready`.
- Crypto context is aggregating 14/14 healthy sources and 7/7 healthy crypto news feeds.
- Latest divergence check is `ok=True` with worst relative spread 0.00%.
- Market/crypto correlation overlay is running in `exact` mode with 0 aligned pairs and cache hits/misses 0/0.
- Latest training summary is 2.2d old: 1 trained, 0 failed, `confirmed_training_success=False` with reason `trained_ok_but_not_promotable:skipped_by_flag`.
- Process watchdog currently tracks `4` services with `4` healthy targets, `0` restart storms, and `0` alerts. Tripwire event flag remains `active=False`.
- PyTorch sidecar stays observation-only, but it now carries `0` active shadow-assist candidate profiles across `0` tracked runs.
- Latest daily ops quality score is 100.0 and the weakest institutional domain is `immutable_experiment_tracking` (68.00).

## Executive Summary

- These features matter because they describe the platform’s real differentiators: unified-memory-aware runtime tuning on Apple Silicon, one control surface across shadow/paper/live, event-to-trade intelligence, self-healing operational control, and a broker-agnostic portability contract.
- The proof is intended to be operational rather than promotional. If a feature is still blocked, replay-only, or waiting on better parity proof, the document says so directly.
- Read the feature proof notes as both a strength map and a watch list: they show what is already impressive and what still needs to mature to make the feature impossible to hand-wave away.

## Real-World Readiness

- Institutional posture: `industry_leaning` at `96.21/100`.
- Live operating posture: `ready` at `100.00/100` with runtime separation `degraded`.
- Watchdog coverage: `4/4` healthy targets, restart storms `0`, alerts `0`.
- Training lane: `1` trained / `0` failed, artifact `2.2d old`.
- PyTorch research lane: `disabled` during MLX-primary live collection.
- Autonomy posture: `blocked` at `71.68/100`, triggered playbooks `39`, open incidents `2`.
- Architecture posture: `9/12` proof surfaces ready, host profile `max_throughput`, portable proof `ready`.

## Special Features

- Adaptive Apple Silicon Brain: host-aware tuning now recognizes `Apple M1 Max`, sees memory architecture `unified`, and lands on `max_throughput` before the stack starts.
- Three-Mode Switchboard: mission control now tracks shadow/paper/live with `3` active modes and runtime clearance `awaiting_coverage_cycles`.
- Event-to-Trade Intelligence: the macro lane now surfaces live-detection and media ingest proof as `ready` with `relevance=high transcript_quality=official_release media_status=missing idle_ready=0`.
- Self-Healing Ops Plane: autonomy currently sits at `71.68/100` with `39` triggered playbooks.
- Portable Brain Contract: the host contract now recommends `native` mode with proof-node status `ready`, backend `pytorch`, and parity focus `mlx_vs_portable_replay` while keeping the broker/runtime seam portable.

## Special Feature Proof Notes

### Adaptive Apple Silicon Brain
- Adaptive Apple Silicon Brain: host-aware tuning now recognizes `Apple M1 Max`, sees memory architecture `unified`, and lands on `max_throughput` before the stack starts.
- Why it matters: This matters because Apple Silicon unified memory gives the live stack one shared CPU and GPU pool for feature windows, broker-context caches, and MLX inference, so the same code can stay responsive on a MacBook Air and then scale up hard on Max-class machines without copy-heavy rewrites.
- Recognized host `Apple M1 Max` on `Darwin` with profile `max_throughput`.
- Memory architecture is `unified` with shared CPU/GPU pool `True`.
- Apple Silicon unified memory keeps CPU, GPU, and MLX tensors in one pool, which cuts copy overhead for large feature windows, broker context caches, and multi-model inference.
- Recommended runtime posture is `native` with backend `mlx`.
- Host override file is `present` at `/Users/dankingsley/PycharmProjects/schwab_trading_bot/config/.env.host_profile_override`.
- Current watch item: Portable posture is still strongest on native Apple Silicon; non-Mac proof is `ready` and still about replay parity rather than full live parity.

### Three-Mode Switchboard
- Three-Mode Switchboard: mission control now tracks shadow/paper/live with `3` active modes and runtime clearance `awaiting_coverage_cycles`.
- Why it matters: This is the control surface that keeps the same trading brain coherent across shadow, paper, and live instead of forcing three separate systems to drift apart over time.
- Switchboard currently tracks `3` active modes and `3` ready modes.
- Active modes: `shadow, paper, live`; ready modes: `shadow, paper, live`.
- Control surface clearance is `awaiting_coverage_cycles` with live read-only `True`.
- Current watch item: Runtime clearance is still `awaiting_coverage_cycles`, which means the switchboard is operationally honest about when live should stay read-only.

### Event-to-Trade Intelligence
- Event-to-Trade Intelligence: the macro lane now surfaces live-detection and media ingest proof as `ready` with `relevance=high transcript_quality=official_release media_status=missing idle_ready=0`.
- Why it matters: It gives the platform a route from macro hearings, policy streams, and transcripts into market-aware stance, relevance, and bulletin surfaces that the rest of the brain can actually use.
- Latest macro event status is `ready` from `NVIDIA IR` with speaker `NVIDIA management`.
- Transcript quality is `official_release` at `0.0000`, cue match `0.0000`.
- Market read is `mixed` with sentiment `-0.2000` and relevance `high`.
- Current watch item: Current transcript pipeline is `official_release` and should keep moving toward fully clean replay-grade transcripts for every event.

### Self-Healing Ops Plane
- Self-Healing Ops Plane: autonomy currently sits at `71.68/100` with `39` triggered playbooks.
- Why it matters: It is the difference between a platform that merely runs and one that can diagnose pressure, throttle itself, freeze bad lanes, and preserve operator trust while the rest of the stack keeps moving.
- Autonomy score is `71.68/100` with `3` autonomous repair paths.
- Triggered playbooks: `39`; notification ladder `ready`; incident review `blocked`.
- Process watchdog shows `4/4` healthy targets and `0` restart storms; chaos drill score `100.00`.
- Current watch item: Incident review is currently `blocked`, so the self-healing story is strong but still not fully frictionless.

### Portable Brain Contract
- Portable Brain Contract: the host contract now recommends `native` mode with proof-node status `ready`, backend `pytorch`, and parity focus `mlx_vs_portable_replay` while keeping the broker/runtime seam portable.
- Why it matters: This is the selling point that keeps the platform from being a dead-end Mac-only build: Apple Silicon stays first-class for the live brain, but the runtime now has an explicit broker-agnostic contract for replay, research, and proof on Linux and Windows.
- Native contract is `native` on backend `mlx`, portable contract is `portable` on `pytorch`.
- Broker-specific news, options, and calendar context now sit behind adapter seams instead of being hardwired to one brokerage client.
- Apple Silicon keeps a live-trading edge through `unified` memory architecture while the proof node preserves Linux and Windows replay portability.
- Cross-platform proof node is `ready` and nightly parity support is `True`.
- Linux and Windows deployment matrix entries are present, with next step `run replay and parity checks on the non-Mac node before claiming live portability`.
- Current watch item: Next portability milestone is `run replay and parity checks on the non-Mac node before claiming live portability`, which is still the bridge between strong design and undeniable parity proof.

## Next Proof Targets

- Reduce the data-plane drag so autonomy and runtime-separation proofs are not still competing with queue pressure.
- Keep pushing portability from strong design into undeniable parity by running more non-Mac replay and parity checks.
- Tighten transcript quality and event replay quality so Event-to-Trade Intelligence stays convincing on both live and replay paths.
- Turn the current proof surfaces into a stronger portfolio of stable, repeatable reports rather than one-off wins.


## Current Active Lineup

| Bot | Role | Test Accuracy | Quality Score |
| --- | --- | ---: | ---: |
| brain_refinery_v95_rates_regime_bond_bot | signal_sub_bot | 100.0% | 0.230 |
| brain_refinery_v99_defensive_dividend_concentration | options_sub_bot | 100.0% | 0.923 |
| brain_refinery_v10_seasonal | signal_sub_bot | 93.8% | 0.992 |
| brain_refinery_v96_credit_spread_rotation_bot | signal_sub_bot | 89.5% | 0.229 |
| brain_refinery_v35_dmi_state_machine | signal_sub_bot | 85.2% | 0.998 |

## Showcase Links

- [Showcase Index](../README.md)
- [Live Multi-Asset Paper Trading Platform](../projects/01-live-multi-asset-paper-platform.md)
- [Quant Research and Model Training System](../projects/02-quant-research-and-model-training.md)
- [Data Fusion and Verification Pipeline](../projects/03-data-fusion-and-verification-pipeline.md)
- [Reliability, Safety, and Ops Automation](../projects/04-reliability-safety-and-ops-automation.md)
- [Cross-Market Crypto and Macro Intelligence](../projects/05-cross-market-crypto-and-macro-intelligence.md)

