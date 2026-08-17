# 2026-08-01 Local Disk and Application Memory Incident

## Summary

The macOS startup data volume reached 100% utilization while the paper stack was writing a rapidly growing execution-intent queue and duplicate governance telemetry. Shell here-documents could no longer create temporary files, and macOS reported that application memory was exhausted because swap and temporary-file growth had no disk headroom.

Live execution remained disabled. The stack was stopped and protected by a runtime maintenance hold during recovery.

## Impact

- Live-feed status snapshots failed with `no space left on device`.
- The desktop application restarted during the storage incident.
- Paper collection and execution were paused during the maintenance window.
- A current-day risk-only governance archive segment is incomplete after two pending source segments resolved to one destination during the first emergency compaction. Canonical decision evidence for the same window remains intact.

## Primary Causes

- A quarantined corrupt queue database occupied 125.875 GiB on the startup disk.
- The active queue grew to 54.72 GiB, including 1,756,223 execution-intent rows and roughly 53.175 GiB of repeated payload JSON.
- Execution transport carried full feature vectors although the execution consumer used only a bounded subset.
- Decision and risk evidence was also duplicated into a large execution-lane JSONL stream.
- The resource guard reserved only 2 GiB of startup-disk capacity, which was insufficient for macOS swap and temporary files.
- An orphaned `strategy_research_lane.py` process (PPID 1) consumed about 7.3 GiB RSS while scanning attribution history.
- Attribution discovery followed duplicate symlink aliases, including migration-backup paths, and accumulated all decoded JSONL rows in one unbounded Python list. The aliases expanded the effective scan toward roughly 110 GiB.
- The support-maintenance freeze helper rewrote frozen health artifacts with a current top-level timestamp. Old memory measurements therefore appeared fresh and kept runtime controls degraded after host memory had recovered.
- Memory intelligence treated macOS's retained swap allocation as active pressure even when VM pressure was green, compressor residency was low, no pages were throttled, and 91% of memory was free.
- The swap governor's optional-heavy-job termination list incorrectly included the paper collection parent and child loops. Repeated stale high-swap evaluations sent the sleeve fanout `SIGTERM`, after which the watchdog restarted it on `child_fanout_below_floor`.
- The all-sleeves circuit breaker independently consumed a stale Friday one-numbers snapshot on Saturday. Its `data_quality_low:40.00` response terminated the entire collection fanout, and the watchdog then restarted the hollow parent on `child_fanout_below_floor`. This repeated at the breaker/cooldown cadence even after swap pressure had cleared.
- After the maintenance window, the replay truth gate treated an otherwise valid low-row collection window with no counterfactual candidate as a broken replay instead of evidence still collecting.
- Drift, architecture, self-model, and master-infrastructure reports formed a circular status dependency during final recovery. Each healthy control waited for another report in the same cycle to refresh, leaving a false blocked/degraded rollup after the underlying surfaces had recovered.
- Source-verification recovery exposed an unbounded retry shape: the strict-ready ticker-news refresh attempted all 500 symbols with the same 240-second deadline as its parent controller, so the parent could terminate it before it wrote a fresh artifact.
- `/Volumes/VIDEO` and `/Volumes/BOT_LOGS` are separate partitions on the same external physical disk. That device stopped completing I/O after the emergency archive work, so SQLite WAL shared-memory growth blocked inside `ftruncate` and ordinary report writes blocked inside the kernel.
- The device failure accumulated 39 uninterruptible processes, including 29 bot processes. Repeated `SIGKILL` and forced unmount requests correctly remained pending because macOS cannot retire a process blocked in device I/O; the accumulation contributed to the application-memory alert even though active heap pressure had already recovered.
- Hot-path health reports retained an external storage-mode label after runtime links had been moved to local fallback. That stale classification prevented the local backlog drainer from accepting an otherwise healthy internal route.
- Canonical decision attribution was mirrored into risk-channel JSONL at full payload size. Two active risk mirrors grew to roughly 1.13 GiB and 0.60 GiB while adding no distinct execution evidence.
- The sampled accountability-write path returned a false negative for deliberately skipped rows. Callers interpreted the sample decision as a failed write and retried, amplifying duplicate telemetry.
- Collector accountability context used a stale Schwab broker label for Coinbase data, routing otherwise valid observations through the wrong attribution namespace.
- The stack starter and process watchdog could both own long-running sleeve and Coinbase workers. Shell-owned `nohup` children did not reliably survive handoff, while the watchdog could later create a second ownership path.
- Idempotent stack startup ran duplicate-process cleanup even when the healthy all-sleeves parent already existed, allowing preflight to terminate one of its valid child launchers.
- The process watchdog treated a completed on-demand SQL writer as missing when older coordinator artifacts were stale, even though the fresh writer-progress artifact proved a clean idle completion.

## Recovery

- Engaged a runtime maintenance hold and stopped the paper stack.
- Resumed a verified deep-cold transfer from an existing 63,006,834,688-byte partial copy.
- SHA-256 verified and released the 135,156,744,192-byte corrupt source to `/Volumes/VIDEO` through a symlink.
- Compacted 103.268 GiB of raw governance telemetry into 13.619 GiB on `/Volumes/BOT_LOGS`.
- Drained 1,238,494 stale paper-only intents without placing orders.
- Deleted only acknowledged queue rows and reduced the queue database from 54.72 GiB to 0.263 GiB.
- Restored startup-disk headroom from about 0.236 GiB to about 249 GiB.
- Verified healthy host state after recovery: 83% memory free and no bot-owned stopped processes.
- Terminated the orphaned attribution process and restored system-wide free memory from 36% to 91% without rebooting.
- Stopped the stack, marked all device-blocked bot processes for termination, physically reconnected the external disk, and verified that every uninterruptible bot process retired.
- Switched 91 active runtime links to the internal `local_fallback_storage` root while preserving disabled and migration-backup links for forensic recovery.
- Restarted one clean all-sleeves parent with all 7 jobs and 15 observed child processes, two Coinbase collection loops, and the paper execution lane. No live execution authority was enabled.
- Disabled the redundant risk mirrors, retained one bounded canonical sampled attribution stream, and verified that both former high-growth risk files stopped changing.
- Corrected broker attribution for Coinbase collectors and made sampled accountability writes acknowledge intentional skips so callers do not retry them as failures.
- Moved ownership of all-sleeves, Coinbase spot, and Coinbase futures workers to the process watchdog. Stack startup now performs a watchdog handoff and certifies the resulting PIDs instead of retaining shell-owned workers.
- Refreshed the profitability, runtime, paper-ramp, and storage controls, then ran the bounded hot-shard drainer. The raw queue fell from about 5,856 pending rows to fewer than 500 without pausing collection or enabling live orders.
- Ran a fresh five-target state snapshot and restore drill. Four bounded control files were copied and hash-verified end to end; the 87 GiB SQLite database used the configured large-file metadata proof.

## Hardening

- Deep-cold transfer now resumes and verifies partial copies instead of restarting them.
- Governance compaction now uses collision-safe deterministic archive names.
- Execution transport now uses a consumer-required feature allowlist with payload lineage hashes.
- Queue retention now shortens acknowledged retention under capacity pressure and supports incremental vacuum.
- Runtime maintenance jobs are exempt from support throttling while an authorized maintenance hold is active.
- Startup-disk headroom is now part of memory-pressure classification: warning below 32 GiB and critical below 8 GiB by default.
- Unattended self-healing now reconciles external routing, prunes acknowledged queue rows, performs critical-only compaction and resumable verified cold offload, and rechecks resource state before restoring fanout.
- Paper P&L schema v2 now persists book state, avoids profile-total double counting, and publishes confidence-bounded post-cost expectancy.
- Independent fill calibration and post-cost expectancy are operational soak advisories while evidence matures, but remain hard blockers for live-money promotion.
- Strategy attribution now resolves and deduplicates source targets, ignores migration-backup aliases, streams rows instead of retaining the corpus, and caps bootstrap and incremental work at 64 MiB per source by default.
- Attribution output publishes `complete_history`, scanned/skipped/pending bytes, and bounded-file counts so an incomplete history scan cannot masquerade as complete evidence.
- Runtime throttle excludes its own short-lived refresh process from sustained CPU attribution.
- Storage clearance and paper admission distinguish a routed, integrity-clean, sub-critical draining queue from a hard storage breach; the soft latency target remains visible, while pressure index 1.0 or any hard limit still fails closed.
- Frozen health artifacts now preserve their source measurement timestamp and publish separate controller, source-age, and freshness metadata, so a maintenance write cannot make stale telemetry look current.
- The swap-pressure governor is exempt from support-maintenance freezing because host-safety telemetry must remain live during recovery.
- Swap-pressure termination is now restricted to optional offline research, report, retrain, and retention jobs. Paper collection, broker loops, and the paper executor are explicitly protected even if a future pattern list accidentally includes them; their in-process training is downshifted through runtime pause controls instead.
- All-sleeves breaker evidence now has explicit artifact and measurement freshness limits and respects closed Schwab sessions. Missing, stale, or closed-session evidence is observation-only and cannot terminate a process.
- Valid data-quality, blocked-rate, or P&L breaker trips now park only execution consumers for the bounded cooldown. Collection sleeves remain live so they can gather the evidence needed for recovery, and breaker-parked execution jobs are reported as policy parked instead of a broken fanout.
- Memory intelligence retains the raw macOS swap allocation for forensics but discounts it from active pressure only when the canonical swap governor and fresh VM signals independently prove a green host.
- Sparse replay windows now remain visible as non-grade-blocking collection advisories when their only failure is insufficient rows; stale-only, corrupt, negative mature, or otherwise failed replay evidence still blocks.
- Guarded-paper self-model normalization breaks circular supervisor debt only when the dashboard, health-fast gate, paper guard, and soak are all green and the remaining master-infrastructure checks are limited to governance freshness or self-audit debt. Any blocked check or failed repair remains degraded.
- Source autorefresh now bounds ticker-news work in both strict and guarded modes, replaces an oversized inherited runtime limit, and keeps the collector deadline at least 30 seconds inside the parent timeout so a valid partial snapshot can be written before control returns.
- A fresh authoritative `no_bot_needs_training_candidates` result is now a healthy training-idle state only when guarded-paper health is clear, collection is flowing, and training quality remains above threshold. Storage, resource, eligibility, or quality blockers still remain visible and block training.
- Hot runtime state is pinned to internal APFS with `BOT_LOGS_PREFER_EXTERNAL=0`, `BOT_CHANNEL_QUEUE_PREFER_LOCAL=1`, and `SQL_LINK_SERVICE_FORCE_LOCAL_FALLBACK=1`. External volumes are archive destinations, not runtime dependencies.
- Route reconciliation is lexical and does not follow an unresponsive external symlink while selecting local fallback. Active nested governance and SQLite routes are reconciled together, while backup and disabled links remain untouched.
- SQL writer and guarded-maintenance children have hard deadlines and process-group termination. A child still blocked in kernel I/O is reported as uninterruptible instead of causing its parent to wait forever.
- The process watchdog has a host-local singleton lock, bounded subprocess collection, and a no-I/O external probe whenever local-hot policy is active. It publishes `local_fallback` as the effective storage mode rather than carrying stale external state forward.
- Backlog draining accepts the explicit local-hot availability contract even when reading a report produced by an older process, and every writer command has a bounded runtime.
- Accountability sampling now has an explicit success contract: a policy-selected sample skip is an acknowledged write outcome, while genuine write failures remain failures.
- Risk-channel mirroring is disabled for the high-volume paper collection path. Canonical attribution remains sampled and preserves nonzero PnL and position-bearing evidence.
- Runtime broker context comes from the active collector broker, preventing Coinbase observations from being mislabeled as Schwab evidence.
- The process watchdog is the single lifecycle owner for persistent sleeve and Coinbase workers. Startup hands off desired targets, verifies stable processes, and reports the watchdog-owned logs.
- Idempotent startup passes `--allow-running` to preflight when the all-sleeves parent is already live, so duplicate cleanup cannot kill valid child fanout during a harmless restart command.
- A fresh `status=ok`, `running=false`, `current_step=complete` SQL progress artifact certifies healthy on-demand idle. Running, failed, or stale progress cannot use that exemption.
- The paper continuity regression guard now verifies that stale hot artifacts trigger refresh rather than silently closing an otherwise eligible paper lane, and that soft resource pressure capacity-limits paper work without stopping it.

## Verification

- 320 integrated storage, queue, execution, profitability, calibration, promotion, memory, and soak regressions passed.
- Modified Python modules passed bytecode compilation.
- 175 focused attribution, storage-clearance, paper-ramp, runtime-throttle, live-separation, and long-runtime regressions passed after the post-incident hardening pass.
- 330 memory, swap, maintenance-freeze, storage, paper-ramp, runtime, attribution, and execution-truth regressions passed after the final stale-state and sparse-replay fixes.
- 77 focused all-sleeves launcher and self-model cycle regressions passed.
- 475 integrated maintenance, memory, swap, storage, training, paper-truth, architecture, drift, and long-runtime regressions passed after final recovery.
- 506 integrated maintenance, memory, swap, storage, source-refresh, training, paper-truth, architecture, drift, and long-runtime regressions passed after the final source and training-idle hardening.
- 181 focused storage-router, watchdog, disaster-recovery, failback, shard-writer, backlog-drain, and maintenance regressions passed after the external-device isolation changes.
- An additional 32 watchdog and backlog-route regressions passed after fixing the stale external-mode classification, including a test that forbids external I/O under local-hot policy.
- A fresh five-target state snapshot and restore drill passed. Small control artifacts were copied and hash-verified; the 86 GiB database used the bounded large-file metadata proof. Storage resilience returned to ready with fresh restore evidence and zero split-brain conflicts.
- Post-restart host checks showed about 230 GiB of internal free space, 92% system memory free, and zero bot processes in uninterruptible I/O.
- Source verification recovered from 8 stale contexts to 16/16 verified sources; the bounded ticker-news pass completed 293/300 symbols with 97.7% coverage.
- The repaired all-sleeves parent remained live for more than five hours beyond the former 23-minute restart cadence with all 7 jobs and 14 child processes healthy.
- System drift finished with 25 surfaces ready, zero blocked/degraded/stale surfaces, and an empty drift-autopilot repair queue.
- Live execution authority remained disabled throughout recovery.
- 84 focused lifecycle, startup, and SQL-writer idle regressions passed after consolidating worker ownership.
- 363 integrated accountability, storage, watchdog, runtime, paper-ramp, paper-truth, and soak regressions passed after the final ownership and write-amplification fixes.
- The final bounded storage cycle completed 17/17 shards with no timeout, reduced the hot queue below 500 rows, and returned ingestion storage to ready with A+ backlog relief.
- The final runtime paper regression guard reported zero failed guards, health-fast reported strict all-clear, the runtime dashboard reported `ok`, and unattended 30-day soak readiness reported A+ with no blockers.
