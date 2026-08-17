# Schwab Trading Bot Hardening Delta

**Reporting window:** 2026-07-22 20:53:55 ET through 2026-08-04
**Branch:** `codex/showcase-modularize`
**Runtime boundary:** market data and paper execution only; live order submission remains locked

## Executive Assessment

This window was not a clean no-change soak. It was an engineering and recovery interval inside the broader paper-soak program. The system stayed live-data/paper-only while storage, memory, ingestion, lifecycle, evidence, and observability weaknesses were found and hardened. Those changes improve the next uninterrupted evidence window, but they do not erase the interventions or convert historical paper losses into profitable evidence.

At the end of the work, the guarded paper runtime reports A+ unattended-soak readiness, all 221 tracked collectors observing, 15 healthy sleeve child processes, a healthy Schwab auth lease, bounded storage pressure, and live execution locked. The active fleet contains 1,780 bots, including 1,584 paper-enabled bots. Paper truth is A+ at 97/100. Controlled profitability posture is A+, while raw realized paper profitability remains D with a $17,864.29 recovery gap because historical net P&L evidence is still negative. Live-money promotion remains blocked on evidence, calibration, and sustained clean-runtime milestones.

## Continuity Baseline

- Planned Apple battery service pause began 2026-07-14 14:40 ET.
- Segment 2 acceptance occurred 2026-07-22 20:53:55 ET after restore, storage-route, auth, paper-lane, runtime, and soak-readiness checks passed.
- Documented offline duration was 8 days, 6 hours, 13 minutes, 55 seconds.
- Pre-window acceptance evidence included A+ soak readiness, live-submit count zero, external storage routes ready, paper truth A+ at 97, the paper ramp armed, and a healthy watchdog target set.
- The hardware-service pause remains planned downtime, not a trading-system failure. The later engineering interventions remain visible as soak interventions.

## Major Production Hardening

### Runtime And Lifecycle

- Separated guarded paper readiness from live-money authority throughout health, dashboard, and grade contracts.
- Kept `MARKET_DATA_ONLY=1`, `ALLOW_ORDER_EXECUTION=0`, and live-order paths locked across startup, recovery, and self-healing controls.
- Added bounded process ownership, singleton locks, restart budgets, restart-storm isolation, child-fanout certification, and post-restart settlement checks.
- Fixed a stop/start lifecycle defect that left the operations watchdog and SQL writer LaunchAgents disabled after an explicit stop. Start now restores the watchdog, writer, failover, caffeinate, observability, local livefeed, auth, and reboot-resilience supervisors.
- Added an explicit stack-stop marker so reboot recovery cannot undo a deliberate operator stop.
- Added a recurring `health-fast` refresh inside the livefeed freshness budget, eliminating a 10-minute feed contract versus 30-minute evidence refresh mismatch.
- Kept heavy livefeed viewers bounded by TTL and process-tree cleanup, while preserving one supervised local mirror.

### Storage, Queueing, And Cold Archive

- Added adaptive hot, local-fallback, external, spillover, and cold-archive routing with route verification and failback controls.
- Added quota, reserve, reconnect, eject, retention, compaction, disaster-recovery, and storage-pressure control planes.
- Made cold archive policy adaptive, hash-verifiable, deduplicated, compacted, and readable through manifests and bounded restore paths.
- Added SQLite local failover, shard-level writers, writer coordination, queue integrity checks, and bounded market-hours versus off-hours drain behavior.
- Fixed SQL writer false-idle detection. A completed historical cycle can no longer certify health if fresh queue evidence shows backlog reaccumulation.
- Corrected ingestion scorecard double-penalization when drain estimates are not applicable to a bounded stable queue.

### Data Collection, Decisions, And Livefeed

- Preserved collection during paper-entry quarantine and weak-profile abstention; collectors are not parked merely because execution is withheld.
- Expanded labeling, lineage, route, source-quality, schema, freshness, and feature-context metadata used by training and self-awareness layers.
- Repaired legacy and new decision route metadata. Listed Schwab symbols such as SPYD now resolve to `schwab_equities` instead of `unclassified`.
- The livefeed now reports `lane_source`, asset class, source quality, schema state, record age, and file age for decisions.
- Replaced keyword-only red highlighting with explicit `ok`, `watch`, `alert`, and `flow` levels. Empty `failed=` or `warnings=` fields no longer create false alarms.
- Added freshness-aware, contradiction-resistant status synthesis for system, collection, auth, storage, throttle, provider, and soak rows.
- Added an authoritative headline that separates source visibility from operational health and only publishes `walkaway=true` when guarded paper, current soak evidence, storage, throttle, auth, and the live-money lock all agree.
- Reconciled fresh paper-ramp and throttle policy against otherwise-ready health snapshots, made unattended-soak freshness mandatory, and made operational blocking return a failing command status even when source files are fresh.
- Clarified operator labels as `active_issues`, `managed_watches`, and `live=locked_read_only`, including explicit paper impact and one bounded next action.
- Added Twelve Data authentication-failure classification, a bounded six-hour retry cooldown, automatic FX realtime-child isolation, and context-only fallback reporting. This optional provider watch does not degrade the Schwab paper soak while fallback collection remains healthy.
- Dashboard output now separates active degradation, managed paper-soak debt, and pending promotion evidence.

### Paper Execution And Profitability

- Added broker-truth, account-position awareness, paper reconciliation, fill calibration, attribution, profitability, and regression layers.
- Kept all eligible paper paths observable and eligible while allowing strategies to abstain when no qualifying trade exists.
- Added weak-profile entry quarantine while keeping sells and reduce-only paths available.
- Fixed One Numbers current-day decision counting; the repaired report observed approximately 1.687 million current-day rows rather than a false zero.
- Hardened the UTC-day rollover so One Numbers rebuilds the new source day and then refreshes its full 97-day history without fabricating rows.
- Raw profitability remains evidence-based. The system displays `A+ controlled / D raw`, the exact P&L recovery gap, weak-entry quarantine, and open reduce-only paths rather than rewriting historical losses or promotion evidence.

### Auth, Positions, Tax, And Compliance

- Added supervised Schwab token refresh, post-refresh verification, lease reconciliation, and operator-gated browser renewal paths.
- Added account snapshot refresh, position-opportunity observation, bounded round-trip analysis, and amount-aware account buildout planning.
- Added a tax ledger, estimator, yearly regulation manifest, federal 2026 policy baseline, and annual update workflow. Tax output remains an estimate and does not replace professional advice.
- Added personal-use, commercial-readiness, data-governance, privacy, evidence-retention, and release-boundary guardrails without granting live execution.

### Training And Bot Fleet

- Expanded all-bot labeling quality, label provenance, abstention semantics, route metadata, and training-requalification controls.
- Added targeted strategy maturation, lane-upgrade, weak-sleeve repair, and coverage-gap workflows.
- Improved Python and MLX routing, runtime selection, dependency health, resource-aware training, and cold-lane scheduling.
- Kept collect-only and evidence-thin bots visible as maturity debt rather than falsely promoting them.

## August Storage And Memory Incident

The August 1 incident began with local-disk exhaustion and application-memory pressure. The immediate operator symptom was shell failure to create temporary here-doc files (`no space left on device`) inside the livefeed. Root causes included oversized decision/explanation payloads, write amplification, unbounded incident aggregation, stale SQL progress assumptions, and insufficient local reserve enforcement.

Measured recovery and hardening included:

- Explanation data compacted from 50.747 GB to 15.014 GB, reclaiming 35.733 GB with hash verification.
- Approximately 54.85 GB of current-day decision data was identified as incident-era storage debt and placed under corrected retention/compaction controls.
- The master infrastructure artifact was reduced from roughly 930 KB and 942 embedded sources to a typical 80-140 KB artifact with 16 retained summaries while still evaluating the complete source set in memory.
- One supervised drainer wave reduced backlog from 13,602 to 3,832 lines and merged 17,006 queued records.
- Recovery reclaimed approximately 9.5 GB externally, 1.4 GB internally, 2.6 GB from raw lanes, and roughly 10.5 GB from the hot plane by estimate.
- Post-recovery free capacity was approximately 143 GB on the external primary route and 104 GB locally, with about 299 GB free on the VIDEO cold-archive volume and 234.6 GB of policy-available spillover.
- The 30-day storage projection retained roughly 22-24 GB of primary margin before cold spillover; a later live readiness snapshot showed more than 31 GB of primary external margin.

## Recurrence Guards Added

1. Payload size and retained-source bounds for decisions, explanations, and infrastructure summaries.
2. Atomic writes, bounded diagnostics, and no unbounded stdout embedding in health artifacts.
3. Local reserve and no-space preemption before shell, SQLite, or livefeed temp-file failure.
4. Adaptive compression, deduplication, manifests, hash verification, and restore-readable cold archives.
5. Queue age, integrity, lane, writer-progress, and fresh-state reconciliation.
6. SQL writer false-idle rejection whenever bounded queue limits are exceeded.
7. Market-hours bounded recovery and off-hours heavy-drain routing.
8. Memory, compressor, swap, fanout, and process-pressure governors with automatic downshift and recovery.
9. Start/stop LaunchAgent reconciliation and explicit-stop protection.
10. Freshness-aware livefeed status contracts and periodic `health-fast` regeneration.
11. Decision route/schema repair at write time plus historical in-memory normalization.
12. Provider failure classification, cooldown, failed-child isolation, and zero-error context fallback.
13. Process-success checks that reject a merely alive sleeve when its ingestion request stream is failing persistently.
14. Post-settlement paper/throttle verification and terminal evidence convergence so dashboard, master supervisor, self-model, drift, and operator cockpit consume the same refresh cycle.
15. Incident closeout reconciliation for intentionally deferred cold research lanes so managed separation is not reopened as a false critical incident.
16. Regression tests for storage, auth, lifecycle, profitability, paper truth, route labeling, livefeed semantics, provider fallback, sleeve liveness, evidence convergence, and unattended soak behavior.

## Verification Evidence

- The final full regression run passed 3,610 tests with zero failures in 69.85 seconds. Focused livefeed, accountability, dashboard, storage, watchdog, provider, sleeve-isolation, and lifecycle suites also passed during the hardening pass.
- The final unattended refresh completed 201 ordered steps with all 106 required artifacts present and fresh, zero blocked, degraded, or error steps, and paper-soak readiness preserved before and after convergence.
- An actual state-snapshot drill verified copied-file hashes and byte identity for five bounded files. The approximately 203 GB SQLite database was metadata-only because it exceeded the 2 GB drill cap; this is not claimed as a full database restore.
- An actual backup restore verification copied and restored the selected readiness artifact byte-for-byte with a measured RTO of approximately 0.002 seconds and RPO of approximately 961 seconds.
- Collection evidence reported 221 of 221 tracked collectors observing with zero unmanaged zero-observation bots and 15 healthy sleeve children.
- Paper evidence reported 1,780 active bots, 1,584 paper-enabled bots, more than 6.17 million processed paper rows, zero known pending rows, and live execution disabled.
- Schwab auth evidence reported ready/healthy with successful network, auth, and broker probes.
- Dashboard evidence reported no active degradation; promotion, training maturity, optional FX realtime credentials, and scheduled chaos cadence remained visible as managed or forensic debt.
- Live execution stayed locked throughout this work.

## Soak And Live-Money Interpretation

The system can continue an unattended guarded-paper soak after final regression and runtime checks are green. This report does not certify live-money readiness. The engineering work should be treated as a reset of the clean, unchanged evidence interval for canary decisions because runtime behavior, storage ownership, lifecycle recovery, and observability contracts changed materially.

Before a live-money canary, require a new sustained interval with no unplanned intervention, no stale evidence gaps, no queue-integrity breach, no auth lapse, no restart storm, and no false paper-truth state. Also require independent fill calibration, stable post-cost expectancy, sufficient mature walk-forward evidence, raw profitability recovery, promotion packet completion, and a deliberately approved low-risk canary plan.

## Commit History In Window

The committed portion of this window includes the following grouped themes:

- July 23: unattended-soak controls, bot registry expansion, market-data collectors, training profitability coverage, ticker and rotation guards, production flow guardrails, CI and production-quality controls, infrabot live-canary policy, and the production hardening watch.
- July 27: production soak governance, degradation guards, support-pressure paper readiness, auth and runtime recovery, operator-gated Schwab credentials, managed advisory cleanup, and live-canary confidence milestones.
- July 28: production dependency smoke, upgrade hardeners, infrabot self-awareness, managed ingestion watch, promotion evidence, raw profitability awareness, nervous-system reflex routing, and bounded self-healing playbooks.
- July 29: backpressure, storage, stale-tail, recovery-gate, and bounded-lock hardening.
- July 30: deferred-backlog relief, high-backlog ownership, operator-grade personal autonomy, commercial-readiness expansion, sleeve-ingestion control, and whole-fleet posture.
- August 1-4 working set: storage and memory incident recovery, compaction and archive hardening, tax and position intelligence, bot labeling and training upgrades, decision route repair, livefeed semantics, SQL writer recurrence protection, and stop/start lifecycle recovery.

## Final Boundary

This system is stronger and substantially more autonomous than it was at Segment 2 acceptance. Its current success claim is production-hardened personal paper operation with live market data and locked live execution. It is not a guarantee of profitability, not a substitute for tax or compliance professionals, and not authorization to trade live money.
