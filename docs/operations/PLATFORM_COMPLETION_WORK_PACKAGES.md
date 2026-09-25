# Platform Completion Work Packages

Definition version: 1. Baseline source: `fa18085f6990adb35ea23a3d976d5eda6aed1609`.

This expands the 26 areas in [the completion checklist](PLATFORM_COMPLETION_CHECKLIST.md)
into 157 individually identifiable acceptance requirements. They are not 157
confirmed defects, and documenting them does not implement or close them.
The checklist remains the status index; [the hardening ledger](PLATFORM_HARDENING_LEDGER.md)
records reproduced defects, corrections, and verification. The
[source-of-truth map](../architecture/SOURCE_OF_TRUTH.md) owns source selection.
This document specifies required behavior; it does not assert that behavior
already exists or install a new runtime control or completion evaluator.

## Ownership And Evidence

Each package names a lead source owner and collaborating sources. This assigns
code ownership, not a human on-call assignment. A named accountable maintainer,
reviewer, and escalation contact must be recorded in the release evidence before
closure; a script name cannot acknowledge an incident or approve a release.
Existing test files below are starting points, not claims of full coverage.
Existing `latest` reports are discovery inputs, not immutable closure receipts.

For every requirement, retain the following in the release evidence package:

| Field | Required Meaning |
| --- | --- |
| Identity | Requirement ID, definition version, release/source manifest digest, dependency-lock digest, and redacted configuration identity. |
| Scope | Exact data families, processes, accounts by nonsecret policy ID, consumers, and operating modes tested; list exclusions. |
| Responsibility | Named accountable maintainer, reviewer, escalation contact, and any operator approval receipt. Unassigned is not complete. |
| Baseline | Reproduced failure, or an explicit unproven-behavior obligation linked to the ledger. |
| Acceptance | Expected result and the owning policy's thresholds, units, clock, resource budget, and expiry. Freeze these before the run. |
| Test Execution | Exact command, fixture/input digest, environment, start/end times, exit status, assertions, and bounded logs. |
| Runtime Proof | Producer identity/time, input watermarks, process-start/adoption identity, observed outcome, and bounded observation window. |
| Integrity | Immutable artifact reference plus content digest; record input generation, completeness, and producer/consumer agreement. |
| Failure Accounting | Failed, missing, timed-out, deferred, flaky, skipped, and blocked checks remain explicit. Unknown does not mean zero. |
| Resolution | Implemented, test-verified, runtime-verified, or closed; reason, unresolved dependencies, reviewer, and reopening trigger. |

Work progresses from definition to implementation, test verification, deployed
observation, and reviewed closure. An external prerequisite is an explicit
blocker on that stage, not a reason to mark the whole area done. Do not add a
second mutable score or manually overwrite generated evidence to track this.
Closure receipts should be attached to the existing release/evidence owners.

Every configurable limit must have a finite value, units, allowed range,
source/precedence, owner, expiry if temporary, and adoption action. Use existing
policy values. If a necessary limit is absent, add and review it in its owning
policy before acceptance; there is no implicit infinity or newly invented SLO.
Changes to acceptance thresholds require separate review and fresh evidence.

## Data And Storage

### C01 Storage Routes

**Lead:** `core/storage_router.py`.
**Collaborators:** `core/storage_target_override.py`, `scripts/ops/ingestion_data_contract.py`.
**Baseline:** Routing and protected-path controls exist; complete mutable-family and process-adoption coverage is not proven.
**Verify with:** C02, C05, C07, C08, C16.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C01.1 | Inventory every mutable family: primary databases, WAL/SHM sidecars, queue/spool, shards, snapshots, models, reports, caches, archives, and temporary files. Each row names exactly one writer owner and its readers. An unclassified path blocks coverage. |
| C01.2 | Bind configured route, permitted physical root/device identity, resolved route, and loaded process configuration. Test conflicting overrides and verify that configured intent alone cannot certify actual routing. |
| C01.3 | Define unavailable/read-only/full-device behavior per family: fail closed, pause, or explicitly permitted bounded fallback. Test absent mounts and restart; never silently create a replacement directory at an unmounted external path. |
| C01.4 | Reject direct, relative, chained-symlink, and swapped-parent routes to protected storage before target I/O. Use local fixtures only; `/Volumes/VIDEO` remains off-limits, including metadata. |
| C01.5 | Bind database and sidecars to one physical writable home and one writer lease. Attempt concurrent startup and route change; prove no split-brain writer, misplaced sidecar, or stale writer after handoff. |
| C01.6 | Produce a bounded route census tied to the running process identity, including fallback backlog and return-to-primary conditions. Observe recovery without deleting deferred data or overriding a still-active owner. |

**Verification start:** `tests/test_storage_router.py`, `tests/test_ingestion_data_contract.py`, `tests/test_storage_reconnect_recovery.py`.
**Closure evidence:** Per-family configured/physical/adopted route matrix, denial tests, concurrent-writer test, and deployed route receipts. No mount repair or protected-volume inspection is authorized.

### C02 Shared Capacity Budget

**Lead:** `core/local_storage_reserve.py`.
**Collaborators:** `scripts/ops/local_storage_reserve_guard.py`, `scripts/ops/sqlite_reclaim_control.py`, `scripts/daily_state_snapshot_drill.py`, `core/cpu_workload_policy.py`.
**Baseline:** Combined vacuum allocation and snapshot admission are corrected; fleet-wide reservation, forecasting, and resource-interaction proof remain incomplete.
**Verify with:** C01, C03, C07, C09, C10, C11.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C02.1 | Account by physical filesystem for live bytes, WAL growth, pending queues, copies, restore expansion, vacuum scratch, model/training outputs, and foreground reserve. Shared devices sum simultaneous demand; unknown capacity blocks admission. |
| C02.2 | Give every heavy producer an atomic admission/reservation contract with owner, amount, deadline, release, and abandoned-lease recovery. Two simultaneous requests that exceed headroom cannot both succeed; identify nonparticipating writers as uncovered demand. |
| C02.3 | Define measured growth windows, forecast horizon, warning/critical thresholds, hysteresis, and staleness limits in owning policy. Replay bursts and zero/invalid/missing measurements; stale forecasts cannot admit work. |
| C02.4 | Couple disk admission to memory pressure, CPU class, concurrency limits, and foreground use. Inject combined pressure while live collection runs in an isolated load test; preserve policy-defined critical-lane latency and reserve floors. |
| C02.5 | Define defer, bounded retry, next eligibility, and escalation for every rejection. Capacity recovery must not trigger simultaneous vacuum, restore, compaction, and training starts. Test a recovery surge. |
| C02.6 | Measure peak allocated bytes, WAL growth, memory, duration, and reservation cleanup during accepted work and forced termination. Reconcile forecast versus actual demand and retain overruns as failures. |

**Verification start:** `tests/test_local_storage_reserve_guard.py`, `tests/test_sqlite_reclaim_control.py`, `tests/test_daily_state_snapshot_drill.py`.
**Closure evidence:** Shared-device admission ledger, combined-pressure test, unattended deferral/recovery receipts, and actual peak resource measurements. Heavy production-sized tests require an approved window.

### C03 Full Recovery

**Lead:** `scripts/ops/storage_disaster_recovery.py`.
**Collaborators:** `scripts/daily_state_snapshot_drill.py`, `scripts/backup_restore_verify.py`, `scripts/ops/blackstart_recovery.py`.
**Baseline:** Bounded control drills and strict manifest checks exist; the full-platform restore producer/evidence path is not implemented. Its RTO result intentionally remains unverified.
**Verify with:** C01, C02, C05, C06, C08, C12, C24.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C03.1 | Freeze a production-scope inventory covering data, models, configuration, schema versions, queue checkpoints, release manifests, and restart-critical control evidence. Record omissions and authorized secret rebind procedures; never place raw credentials in the inventory. |
| C03.2 | Approve an isolated destination, resource window, copy budget, failure-injection scope, and rollback plan. Prove the destination cannot replace active routes or reach a broker mutation endpoint. |
| C03.3 | Produce a consistent snapshot with complete typed file digests, logical database checks, transactional watermarks, and cross-file generation binding. Include committed WAL data; skipped large files make production coverage incomplete. |
| C03.4 | Restore every required object into isolation; verify digests, database integrity, row/key counts, relationships, checkpoint continuity, and model loading. Explicitly compare source and restored logical state where byte equality is not meaningful. |
| C03.5 | Start the restored application with execution disabled, verify dependency ordering, replay/checkpoint resumption, and no duplicated workers or acknowledged-event loss beyond the frozen target. Observe actual service readiness, not just successful file copying. |
| C03.6 | Measure RPO from the restored durable event watermark and RTO from the declared outage/start point through verified service readiness. Capture immutable receipts, interruption handling, failed attempts, and teardown verification; a small control drill cannot satisfy full restore. |

**Verification start:** `tests/test_storage_disaster_recovery.py`, `tests/test_daily_state_snapshot_drill.py`, `tests/test_production_recovery_drill_harness.py`.
**Closure evidence:** Approved inventory/window, complete isolated restore receipt, actual restart checks, measured RPO/RTO, and reviewer approval. Existing code defaults are 720 minutes RPO and 30 seconds RTO via `BOT_RECOVERY_RPO_TARGET_MINUTES` and `BOT_RECOVERY_RTO_TARGET_SECONDS`; they are not proof of an approved achievable production target. Freeze the effective values before testing and report a miss honestly.

### C04 Archive Integrity

**Lead:** `scripts/ops/cold_archive_compactor.py`.
**Collaborators:** `core/archive_snapshot_control.py`, `scripts/ops/storage_disaster_recovery.py`.
**Baseline:** A damaged legacy archive is preserved; another archive with the same date does not establish equivalence or recoverability.
**Verify with:** C01, C02, C03, C05, C07.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C04.1 | Identify the exact original from the existing maintenance receipt, record ownership, known failure, size, mutability, and provenance without a broad volume scan. Keep it excluded from automated deletion. |
| C04.2 | Create an approved isolated recovery copy under shared admission, recording original/copy digests where readable and all read failures. No in-place repair or destructive experimentation on the original. |
| C04.3 | Classify container, compression, SQLite/page, index, truncation, and logical-record failures as applicable. Run bounded read-only diagnostics on the copy and preserve outputs, tool versions, and error offsets. |
| C04.4 | Salvage into a distinct output, preserving source identity and per-record quarantine. Compare keys, counts, time ranges, schemas, and duplicates against independent known manifests or checkpoints. Unknown loss is not zero loss. |
| C04.5 | Replay recovered records through the normal ingestion/schema validation path in isolation. Prevent duplicate credit from overlapping archives and preserve the distinction between original records and reconstructed values. |
| C04.6 | Publish verified recoverable, unrecoverable, and unknown ranges with downstream research impact. Keep the original until separately approved retention/deletion conditions pass; a successful salvage does not authorize disposal. |

**Verification start:** `tests/test_cold_archive_compactor.py`, `tests/test_archive_snapshot_control.py`.
**Closure evidence:** Original-preservation receipt, approved-copy lineage, diagnostic/salvage results, independent comparisons, and explicit loss disclosure. Source inventory is documented in `docs/operations/SOURCE_AND_STORAGE_MAINTENANCE.md`.

### C05 Ingestion Semantics

**Lead:** `scripts/link_jsonl_to_sql.py`.
**Collaborators:** `scripts/collector_contracts.py`, `core/collector_transport.py`, `scripts/ops/sql_link_writer_service.py`, `scripts/ops/ingestion_data_contract.py`, `scripts/ops/data_plane_recovery_controller.py`.
**Baseline:** Definitions, bounded transport, and checkpoint machinery exist; full interruption/replay semantics and current-write recovery qualification need end-to-end proof.
**Verify with:** C01, C02, C06, C09, C13, C18, C21.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C05.1 | Define each event's provider, source ID, event ID/idempotency key, schema, event time, observation/arrival time, sequence, and payload digest. Specify corrections and revisions without silently rewriting original facts. |
| C05.2 | Define validation, usable-data criteria, null/missing semantics, units, quality status, quarantine reason, retention, and replay authority per family. Successful HTTP fetches cannot certify usable events. |
| C05.3 | Specify the acknowledgment boundary: fetched, spooled, transaction committed, checkpoint advanced, downstream merged. Kill the writer between each pair; no checkpoint may skip an uncommitted record or acknowledge undurable data. |
| C05.4 | Specify delivery semantics honestly, deduplication scope, late/out-of-order windows, gap detection, and correction handling. Replay duplicate and shuffled batches through the actual writer and compare final logical state. |
| C05.5 | Define provider outage, rate-limit, malformed/oversize record, full-disk, SQL outage, and downstream backpressure behavior. Bound retries, spool growth, discovery, and quarantine; preserve critical-family priorities. |
| C05.6 | Separate fresh current write-path evidence from historical failures and backup qualification. Recovery must require newer route, writer, integrity, queue, and durable-progress proof; preserve historical incident counts and independent restore/live gates. |

**Verification start:** `tests/test_link_jsonl_to_sql.py`, `tests/test_ingestion_data_contract.py`, `tests/test_lane_thaw_and_data_plane_recovery.py`.
**Closure evidence:** Per-family ingestion contracts, deterministic replay/interruption comparisons, bounded-loss/duplication measurements, and deployed durable-progress receipts. Fetch count, file presence, and an aggregate green grade are insufficient.

### C06 Schema Compatibility

**Lead:** `scripts/schema_migration_guard.py`.
**Collaborators:** `scripts/collector_contracts.py`, `core/data_quality_checkpoints.py`, producer/consumer owners named in each contract.
**Baseline:** The guard checks field/version presence for six artifacts, not full typed compatibility or all production contracts.
**Verify with:** C03, C05, C13, C17, C18, C21, C24.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C06.1 | Inventory persisted and interprocess contracts with producer, consumers, schema/version, units, nullability, authority, and compatibility class. Unknown versions must be rejected or explicitly handled as nonqualifying legacy data. |
| C06.2 | Validate types, ranges, enums, nested records, uniqueness, references, and cross-field invariants. Test bool-as-number, nonfinite values, malformed timestamps, missing/null values, and inconsistent totals. |
| C06.3 | Publish supported producer/consumer version pairs and additive-versus-breaking rules. Test old-reader/new-writer and new-reader/old-writer combinations; field presence alone cannot pass the matrix. |
| C06.4 | Implement owned, resumable migrations with preflight, transaction/checkpoint boundaries, backup, and failure recovery. Stop midway and verify restart neither double-applies nor silently partially upgrades state. |
| C06.5 | Verify rollback compatibility with the prior supported release or explicitly declare a reviewed restore-based rollback. Preserve the original data and replay fixtures; do not imply reversibility for destructive migrations. |
| C06.6 | Run point-in-time replay and malformed-contract fixtures against the actual release's producers and consumers. Emit per-contract errors and coverage; untested contracts remain uncovered even when all tested rows pass. |

**Verification start:** `tests/test_schema_migration_guard.py`, `tests/test_data_quality_checkpoints.py`, `tests/test_point_in_time_event_store.py`.
**Closure evidence:** Complete contract inventory, compatibility matrix, interrupted migration/rollback receipts, and release-bound replay results.

### C07 Retention And Deletion

**Lead:** `scripts/data_retention_policy.py`.
**Collaborators:** `config/tiered_ingestion_lifecycle_v1.json`, `core/tiered_ingestion_lifecycle.py`, `scripts/ops/storage_retention_unison.py`, `scripts/daily_state_snapshot_drill.py`.
**Baseline:** Successful-snapshot-only retention is hardened; complete family policies and reference-safe deletion are not yet proven.
**Verify with:** C01, C02, C03, C04, C05, C16, C21.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C07.1 | Define each family's owner, age clock, hot/warm/cold periods, capacity cap, transition condition, grace period, and deletion authority in existing policy. Missing policy means preserve, not disposable. |
| C07.2 | Distinguish active/mutable, immutable, checkpointed, merged, referenced, damaged, quarantined, and legal/operational-hold data. Test that failed or unknown state never qualifies as successfully archived. |
| C07.3 | Before retirement, verify destination integrity, restore suitability, lineage/checkpoint references, and required redundancy. File age, duplicate basename, and a copy exit code are not sufficient proof. |
| C07.4 | Make planning read-only; mutation requires a bounded approved action set, owned lock, route revalidation, and audit receipt. Revalidate identity immediately before deletion to resist path replacement. |
| C07.5 | Test concurrent readers/writers, interrupted moves, partial copies, manifest failure, zero retention, stale references, and failed-current snapshots. Preserve the last verified usable generation and all unresolved originals. |
| C07.6 | Verify deletion or retirement with exact object identities, reason, authority, bytes actually reclaimed, and references left intact. Report refused/deferred actions separately; never book planned savings as recovered space. |

**Verification start:** `tests/test_data_retention_policy_reports.py`, `tests/test_storage_retention_unison.py`, `tests/test_tiered_ingestion_lifecycle.py`, `tests/test_daily_state_snapshot_drill.py`.
**Closure evidence:** Versioned family policy matrix, reference/restore checks, interruption tests, and authorized action receipts. No blanket cleanup authority is added.

## Runtime And Operations

### C08 Configuration Provenance

**Lead:** `scripts/ops/load_runtime_env.sh`.
**Collaborators:** `core/runtime_override_precedence.py`, `config/control_surface_ownership_v1.json`, `scripts/ops/runtime_dependency_profiles.py`.
**Baseline:** Runtime loading and pressure precedence exist; full per-key winning-source and process-adoption evidence is not established.
**Verify with:** C01, C02, C09, C12, C16, C19, C24.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C08.1 | Inventory every supported setting with type, units, allowed values, default, owner, safety scope, sensitivity, and reload/restart requirement. Classify unrecognized overrides rather than silently trusting them. |
| C08.2 | Report precedence and winning source across defaults, profile, environment, launch configuration, and runtime overrides. Explain pressure-authoritative exceptions; test competing values in the real loader. |
| C08.3 | Publish allowlisted nonsecret effective values and source digests. For secrets expose only readiness/reference metadata; never secret values or brute-forceable value fingerprints in logs, Git, or health artifacts. |
| C08.4 | Give temporary overrides creator, reason, scope, effective/expiry times, and bounded allowed values. Test expiry, future clocks, missing owner, malformed content, and stale files; safety restrictions cannot disappear silently. |
| C08.5 | Stamp each process with code/config/dependency identity and start/load time. Compare expected versus loaded state; a newly edited file cannot certify that a long-running process adopted it. |
| C08.6 | Verify reload, managed restart, and rollback paths preserve single-writer ownership and live locks. Config drift must remain explicit until the approved adoption action and post-adoption checks finish. |

**Verification start:** `tests/test_runtime_dependency_profiles.py`, `tests/test_ingestion_storage_control.py`; add loader/precedence and process-adoption cases at the owning surfaces.
**Closure evidence:** Redacted per-key contract/provenance inventory, temporary-override tests, and actual process-load/adoption receipts. No secret changes or production restarts are implied.

### C09 Scheduled-Job Lifecycle

**Lead:** `scripts/ops/readiness_evidence_refresh.py`.
**Collaborators:** `scripts/ops/run_production_hardening_watch_launchd.sh`, existing per-job installers/runners, `scripts/ops/long_runtime_common.py`.
**Baseline:** Missing producers have been added to the refresh graph; fleet-wide lifecycle and unattended adoption still need verification.
**Verify with:** C02, C08, C10, C11, C12, C13, C15.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C09.1 | Inventory every scheduled job, canonical runner, installation state, cadence/timezone, dependencies, authority, resource class, deadline, and owner. A checked-in installer is not evidence of an installed job. |
| C09.2 | Record scheduled, eligible, deferred, started, completed, and failed transitions with run ID and timestamps. Publish next eligibility and reason; missing transition evidence cannot look like successful idleness. |
| C09.3 | Define catch-up/coalescing after sleep, missed runs, scheduler restart, and wall-clock changes. Test DST/clock jumps without a restart storm or duplicate logical run; deadlines use a monotonic clock. |
| C09.4 | Make quiet hours, capacity, dependency freshness, single-flight locks, and per-job cooldown explicit admission gates. A deferred job retains its work and gets a bounded future opportunity. |
| C09.5 | Verify dependency-closed ordering and matching publication generations. An allowed child exit code or fresh parent cannot certify an old/missing child artifact; retain per-step failure reasons. |
| C09.6 | Observe deployed success, controlled failure, deferral, and retry across unattended cycles, with next-run evidence. Reconcile installed launch arguments against the source release and identify orphan/duplicate schedulers. |

**Verification start:** `tests/test_readiness_evidence_refresh.py`, `tests/test_long_runtime_common.py`.
**Closure evidence:** Complete job inventory, state-transition/failure fixtures, installed configuration identity, and unattended run receipts. Existing accrual/production cadences remain 15/45 minutes; this definition creates no job.

### C10 Bounded Execution

**Lead:** `scripts/ops/long_runtime_common.py`.
**Collaborators:** `scripts/build_runtime_training_snapshot.py`, `core/runtime_training_common.py`, `scripts/ops/readiness_evidence_refresh.py`.
**Baseline:** Snapshot construction has a total child deadline and atomic digest-bound publication; original timeout root cause, broad worker adoption, and sustained unattended behavior remain unproven.
**Verify with:** C02, C05, C09, C11, C13, C21.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C10.1 | Inventory worker bounds for startup, discovery, lock wait, reads, parse, network, fallback, computation, and publication. Require a total deadline in addition to per-phase budgets. |
| C10.2 | Reproduce the training-snapshot hang/overrun under bounded fixtures and record the responsible phase. Test oversized source lists, blocked SQLite, malformed rows, slow fallback, and publication stalls. |
| C10.3 | Cancel and reap the owned process group within a declared grace period; test descendant processes, ignored signals, early parent exit, and interruption. Never kill unrelated process IDs. |
| C10.4 | Release only owned locks/reservations and remove only owned temporary state. After failure, readers get the last complete valid generation or an explicit unavailable state, never partial success. |
| C10.5 | Bound attempts and backoff independently per surface. Distinguish timeout, cancellation, partial coverage, empty source, and success; none may fake producer freshness or full scan coverage. |
| C10.6 | Observe actual unattended heavy and reuse paths, collecting phase duration, peak resources, timeout frequency, child cleanup, and publication digest agreement. Diagnose recurrence instead of only raising the timeout. |

**Verification start:** `tests/test_long_runtime_common.py`, `tests/test_long_runtime_hardening.py`, `tests/test_build_runtime_training_snapshot.py`.
**Closure evidence:** Reproduced cause/fix, per-worker budget inventory, cancellation/cleanup receipts, and release-bound unattended runs. The snapshot default deadline remains 150 seconds; changing it is not itself a fix.

### C11 Self-Healing Circuits

**Lead:** `scripts/ops/soak_reliability_sentinel.py`.
**Collaborators:** `scripts/ops/soak_self_healing_control.py`, `scripts/ops/grade_regression_guard.py`, `scripts/ops/grade_regression_autopilot.py`.
**Baseline:** Lightweight detection and bounded heavy-repair controls exist; complete cross-controller exhaustion/interruption behavior needs proof.
**Verify with:** C02, C09, C10, C12, C13, C15, C16.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C11.1 | Assign each repairable surface one detector, one repair owner, prerequisites, allowlisted actions, and evidence producer. Multiple observers cannot create competing repair writers. |
| C11.2 | Define closed/open/half-open states, failure classification, per-surface attempt budget, cooldown, and probe eligibility. Test one failing surface cannot consume or reset another's budget. |
| C11.3 | Keep detection lightweight; admit heavy repair only through quiet-window/capacity/ownership gates. Prove a healthy pass is a no-op and does not refresh the full graph or restart healthy collectors. |
| C11.4 | Make repair resumable or explicitly abortable at each mutation boundary. Kill the repair process and prove stale locks/holds are reclaimed by verified ownership without repeating destructive work. |
| C11.5 | Require fresh post-action producer evidence to close a circuit. A successful command, cooldown expiry, or manual report edit cannot erase the root failure or release execution. |
| C11.6 | Exercise terminal exhaustion and escalation, recovery probes, successful repair, and recurrence. Preserve attempts and incident history; no blanket reset of circuits, failures, or candidate clocks. |

**Verification start:** `tests/test_soak_reliability_sentinel.py`, `tests/test_soak_self_healing_control.py`.
**Closure evidence:** Per-surface repair contracts, failure/interruption tests, deployed action/ownership receipts, and evidence-based resolution with live execution still locked.

### C12 Startup And Failover

**Lead:** `scripts/ops/reboot_resilience_guard.py`.
**Collaborators:** `scripts/ops/blackstart_recovery.py`, `scripts/run_all_sleeves.py`, `scripts/ops/process_watchdog.py`, `scripts/ops/production_recovery_drill_harness.py`.
**Baseline:** Existing startup and isolated recovery controls need deployment-level proof, not an assumed rewrite.
**Verify with:** C01, C03, C08, C09, C10, C11, C13, C17, C24.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C12.1 | Define dependency ordering and readiness handshakes for storage, credentials, broker truth, queues, collectors, reports, and execution. Startup of a process is not readiness of its dependencies. |
| C12.2 | Test stale PID/lease/lock files, PID reuse, duplicate launch requests, and interrupted startup. Prove at most one owner per writer and no old process certifying a new release. |
| C12.3 | Exercise missing/read-only mounts, network loss, token expiration, and invalid configuration in isolation. Collection/paper/execution states remain distinct and every fallback follows its explicit contract. |
| C12.4 | Verify interrupted-write and ambiguous-order restart recovery with durable checkpoints and ledger reconciliation. Do not replay a potentially submitted order as a fresh intent. |
| C12.5 | Under a separately approved interruption scope, test actual restart, login/session changes, sleep/wake, and reboot. Record before/after process inventory, downtime, startup order, source adoption, and residual debt. |
| C12.6 | Verify failback and rollback without duplicate consumers/writers/orders; regenerate dependency evidence from the new process/auth epoch. Old green receipts cannot reopen a guarded lane. |

**Verification start:** `tests/test_reboot_resilience_guard.py`, `tests/test_production_recovery_drill_harness.py`, `tests/test_live_order_ledger.py`.
**Closure evidence:** Isolated failure matrix plus approved actual-deployment restart/failover receipts. No reboot, production interruption, or order submission is authorized by this document.

### C13 Evidence Freshness

**Lead:** `scripts/ops/long_runtime_common.py`.
**Collaborators:** `scripts/ops/readiness_evidence_refresh.py`, `scripts/ops/health_fast.py`, `scripts/ops/system_architecture_hardening.py`, all safety-relevant consumers.
**Baseline:** Several producers/consumers now use producer-time and verified publication; coverage and unattended recurrence still need verification.
**Verify with:** C05, C06, C08, C09, C11, C14, C17, C20, C24.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C13.1 | Inventory every safety-relevant artifact and consumer with producer, scope, schema, authority, expected cadence, maximum age, skew tolerance, and dependency identities. Unspecified freshness is not unlimited validity. |
| C13.2 | Reject absent, invalid, naive/ambiguous where unsupported, stale, or excessive-future producer timestamps using the owning contract. Never substitute file mtime, wrapper time, or child exit status. |
| C13.3 | Bind evidence to source/configuration/candidate/account/auth epoch and input generation as applicable. A fresh receipt for another identity or an incomplete input set remains unusable. |
| C13.4 | Make multi-artifact publication atomic or reject mixed generations at readers. Test interrupted publish, stale sibling, digest mismatch, and concurrent publisher; last-good validity cannot outlive its original expiry. |
| C13.5 | Distinguish published, operationally healthy, evidence-complete, qualified, and unavailable. Refreshing an honest blocked result is publication success, not readiness success. |
| C13.6 | Observe unattended propagation through each dependent consumer and dashboard. Expire a producer in a controlled test and prove its consumers become unavailable within their contracted deadline without blanket restarts. |

**Verification start:** `tests/test_platform_evidence_freshness.py`, `tests/test_long_runtime_common.py`, `tests/test_readiness_evidence_refresh.py`.
**Closure evidence:** Consumer coverage matrix, malformed/identity/mixed-generation tests, and deployed expiry/recovery propagation receipts.

### C14 Independent Monitoring

**Lead:** `scripts/observability_exporter.py`.
**Collaborators:** `scripts/install_observability_exporter_launchd.sh`, `scripts/ops/remote_alert_control.py`.
**Baseline:** A separate local stdlib monitor exists; a local heartbeat cannot prove off-host host-disappearance detection or receipt delivery.
**Verify with:** C08, C09, C12, C13, C15, C19.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C14.1 | Identify monitor runtime, process/service manager, dependency isolation, heartbeat source, cadence, and failure domain. Prove the monitor is not a child dependent on the trading worker staying alive. |
| C14.2 | Obtain an approved off-host receiver and responsible recipient with allowed payload, retention, authentication method, timeout, and escalation destination. Never invent an endpoint or expose credentials in evidence. |
| C14.3 | Send an approved test heartbeat and capture receiver-side delivery/observation evidence bound to a unique run ID. Local HTTP intent or channel configuration alone does not count. |
| C14.4 | Stop or isolate the main worker/watchdog in an approved drill; verify the independent monitor detects disappearance within the frozen threshold and does not grant repair or trading authority. |
| C14.5 | Exercise monitor/host disappearance from the receiver side, plus DNS/network/auth/rate-limit failures. The remote deadman must detect a missing host even when no local process can send an alert. |
| C14.6 | Verify recovery notification, bounded retries, deduplication, recipient acknowledgment, and ongoing receiver freshness. Local collection readiness and remote-delivery/live-promotion readiness stay separate. |

**Verification start:** `tests/test_remote_alert_control.py`; extend monitor/receiver failure fixtures at the existing exporter owner.
**Closure evidence:** Approved receiver contract, sender and receiver receipts, host-missing drill, and delivery/recovery acknowledgment. An off-host destination and interruption scope are external prerequisites.

### C15 Alerts And Incidents

**Lead:** `scripts/pager_alert_router.py`.
**Collaborators:** `scripts/ops/remote_alert_control.py`, `scripts/ops/incident_timeline.py`, `scripts/ops/incident_closeout_autopilot.py`.
**Baseline:** Routing, grouping, acknowledgment, and incident reports exist; complete terminal-failure and resolution behavior needs proof.
**Verify with:** C09, C11, C13, C14, C16, C19.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C15.1 | Map each surface/event to severity, operational impact, accountable person/role, delivery route, acknowledgment deadline, escalation deadline, and recovery evidence. Missing ownership is visible debt. |
| C15.2 | Define stable incident identity and deduplication windows independent of volatile message text. Test repeated alerts, changed severity, simultaneous surfaces, and separate accounts/generations without inappropriate suppression. |
| C15.3 | Record queued, attempted, delivered, failed, acknowledged, resolved, and reopened states separately. An attempted send is not delivery; acknowledgment is not technical recovery. |
| C15.4 | Bound transport retries, timeout, backoff, and queue size; preserve delivery failures and escalate exhausted attempts through the approved alternate route. Test terminal errors and lost responses. |
| C15.5 | Resolve only on fresh matching owner evidence or explicit documented operator disposition. Preserve original failure/time and reopen on recurrence; no blanket acknowledgment can clear unrelated incidents. |
| C15.6 | Test the complete incident lifecycle with redacted payloads, receiver receipts, acknowledgments, escalation, recovery, and recurrence. Retain immutable action history under its retention policy. |

**Verification start:** `tests/test_remote_alert_control.py`, `tests/test_incident_timeline.py`, `tests/test_incident_closeout_autopilot.py`.
**Closure evidence:** Severity/ownership matrix, state-machine failure tests, bounded delivery receipts, and approved end-to-end incident exercise.

## Safety And Research

### C16 Authority Enforcement

**Lead:** `core/system_role_contracts.py`.
**Collaborators:** `config/system_role_contracts_v1.json`, `config/control_surface_ownership_v1.json`, `core/base_trader.py`, `core/execution_lane_pipeline.py`.
**Baseline:** Role catalogs and enforcement exist; completeness at every mutating entry point has not been demonstrated.
**Verify with:** C01, C07, C08, C11, C17, C19, C24.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C16.1 | Inventory mutations across direct scripts, CLI/API, retries, schedulers, launchers, storage actions, runtime overrides, training publication, broker adapters, and candidate/release changes. Give each one an action and owner contract. |
| C16.2 | Trace each entry point to a guard immediately before its effect, including alternate/imported paths. Registry coverage or a guard elsewhere in the process cannot prove the particular mutation is protected. |
| C16.3 | Validate caller role, target domain, action, mode, lease, expiry, and generation binding; missing/unknown/ambiguous values deny. Test forged, stale, wrong-owner, and replayed authority receipts. |
| C16.4 | Prove exclusive ownership across retries, concurrent requests, restarts, and handoff. A stale lease holder cannot retain mutation authority after replacement. |
| C16.5 | Verify advisory/report/repair/research components cannot mint candidate acceptance, promotion, risk expansion, or live-order authority. Test direct calls as well as normal orchestration paths. |
| C16.6 | Record allowed and denied actions with nonsecret identity and reason; link every sensitive effect to its authorization receipt. Review bypass findings and rerun the full entry-point matrix on release changes. |

**Verification start:** `tests/test_system_role_contracts.py`, `tests/test_execution_lane_pipeline.py`, `tests/test_live_order_ledger.py`.
**Closure evidence:** Mutation-to-guard coverage map, negative/direct-call/concurrency tests, and actual redacted action receipts. This work cannot self-approve new powers.

### C17 Broker And Order Lifecycle

**Lead:** `core/live_order_ledger.py`.
**Collaborators:** `core/live_execution_envelope.py`, `core/live_execution_controls.py`, `core/live_canary_preflight.py`, `core/base_trader.py`, `scripts/run_execution_lane.py`.
**Baseline:** Sealed execution, durable reservations, and reconciliation controls exist; release-level failure/restart proof remains required.
**Verify with:** C05, C06, C08, C12, C13, C16, C18, C23, C24.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C17.1 | Bind exact candidate, nonsecret account policy, route, fresh broker/account truth, quote provenance, risk decision, and immutable release to each intent. Reject mismatched, stale, expired, or missing inputs. |
| C17.2 | Verify durable intent/reservation before broker dispatch, one idempotency identity, and at-most-once mutation dispatch across concurrency/restart. Test crash before dispatch, during dispatch, and before acknowledgment persistence. |
| C17.3 | Treat network timeout or unknown broker response as ambiguous, not automatically retryable. Reconcile by authoritative broker/ledger state and keep exposure blocked until ambiguity resolves. |
| C17.4 | Verify partial fills, duplicate/out-of-order events, rejection, cancellation race, unfilled deadline, terminal state, and reconciliation gaps. Replacement remains disabled unless independently authorized and implemented. |
| C17.5 | Reconstruct open orders and cumulative fills after restart without duplicate submissions or lost reservations. Compare event chain, materialized state, broker truth, and projected account exposure. |
| C17.6 | Run release-bound offline failure fixtures and the existing read-only dress rehearsal; distinguish simulated, paper, and broker-visible evidence. Any actual live lifecycle exercise requires separate explicit order authorization. |

**Verification start:** `tests/test_live_order_ledger.py`, `tests/test_execution_lane_pipeline.py`, `tests/test_production_readiness_control.py`.
**Closure evidence:** Durable lifecycle/concurrency matrix, identity rejection tests, reconciliation/restart receipts, and clearly scoped broker-visible proof. Paper success cannot authorize or prove a live order.

### C18 Portfolio, Risk, And Accounting

**Lead:** `scripts/risk_service_boundary.py`.
**Collaborators:** `scripts/portfolio_allocator_service.py`, `scripts/ops/account_position_study.py`, `scripts/paper_performance_report.py`, `core/live_order_ledger.py`.
**Baseline:** Portfolio, risk, and accounting services exist; authoritative agreement and adversarial cases require release-level verification.
**Verify with:** C05, C06, C13, C16, C17, C21, C23.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C18.1 | Define authoritative position, settled/unsettled cash, fees, realized/unrealized P&L, currency, cost basis, and timestamp per account. Broker fields, inferred projections, and paper balances remain distinct. |
| C18.2 | Reconcile orders/fills, positions, cash movements, fees, dividends/corporate actions, transfers, and mark prices with independently derived checks. Declare units, rounding, tolerance, and missing-data behavior in owner policy. |
| C18.3 | Verify exposure, notional, concentration, correlation, leverage/collateral where supported, drawdown, and cross-sleeve limits before commitment. Include pending and ambiguous orders in conservative exposure. |
| C18.4 | Test stale/future prices, missing cash, contradictory positions, duplicate fills, partial exits, currency mismatch, and accounting disagreement. Missing financial fields cannot become favorable zeros. |
| C18.5 | Verify kill switches and reduce-only behavior across callers and restart. New exposure stops on unresolved disagreement; a permitted exit cannot cross flat into new opposite exposure. |
| C18.6 | Produce candidate/account/time-scoped post-cost reconciled results with explicit unresolved differences. Deposits, carried inventory, mixed candidates, and unrealized gains cannot masquerade as current realized strategy profit. |

**Verification start:** `tests/test_portfolio_allocator_and_risk_service.py`, `tests/test_account_position_study.py`, `tests/test_live_order_ledger.py`.
**Closure evidence:** Frozen reconciliation rules, independently checked accounting fixtures, risk/kill-switch tests, and fresh account-bound reconciliation receipts. No sizing or risk-limit increase is implied.

### C19 Security And Dependencies

**Lead:** `scripts/security_hardening_audit.py`.
**Collaborators:** `scripts/secret_scan.py`, `scripts/dependency_guard.py`, `scripts/ops/runtime_dependency_profiles.py`, `core/brokers/schwab_credentials.py`, `config/credential_runtime_policy.json`.
**Baseline:** Audit now observes the actual Git hook route and requires explicit fresh scan counts. The checked-in hook was not active in the baseline checkout; this is not a security certification.
**Verify with:** C08, C14, C15, C16, C24, C25.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C19.1 | Reproduce each runtime profile from pinned dependencies and recorded interpreter/platform/toolchain; compare installed distributions against the declared lock and explain native/optional differences. |
| C19.2 | Inventory dependencies and check current authoritative advisories with timestamp/version applicability. Review severity/exposure/remediation; an offline or failed advisory check remains unknown, not clear. |
| C19.3 | Verify secret storage, owner-only permissions, redacted errors/logs/URLs, minimum network access, and no secret-bearing runtime data in Git or release artifacts. Scan scoped safe inputs without traversing forbidden storage. |
| C19.4 | Test the effective pre-commit/CI/release scanning path, negative secret fixtures, and bypass/error behavior. Checked-in scripts alone do not establish enforcement; do not disable source-acceptance gates to install or pass a hook. |
| C19.5 | Exercise expired/rotated/revoked credential handling through documented flows under explicit authorization. Verify serialized atomic rotation, epoch adoption, bounded failure, and no stale client claiming refreshed truth. |
| C19.6 | Verify least-authority actions, artifact integrity, dependency-update rollback, and review evidence. Record unresolved vulnerabilities, exceptions, owners, expiry, and scope instead of reporting a universal security pass. |

**Verification start:** `tests/test_security_hardening_audit.py`, `tests/test_runtime_dependency_profiles.py`, `tests/test_dependency_activation_smoke.py`.
**Closure evidence:** Reproducible environment receipt, dated advisory review, negative access/secret tests, effective enforcement proof, and approved auth-recovery receipt. This definition performs no advisory lookup or credential change.

### C20 Honest Grades

**Lead:** `scripts/ops/production_resilience_control.py`.
**Collaborators:** `scripts/ops/profitability_evidence_firewall.py`, `scripts/ops/health_fast.py`, `scripts/ops/grade_regression_guard.py`, `config/production_resilience_v1.json`.
**Baseline:** Several publication/qualification and simulated/production distinctions are corrected; all grade and dashboard consumers still need semantic consistency checks.
**Verify with:** C03, C05, C09, C13, C18, C21, C22, C23, C26.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C20.1 | Define separate denominators for implementation coverage, operational availability, data quality, unresolved improvements, economic outcomes, and promotion eligibility. Never combine them into an unlabeled completion percentage. |
| C20.2 | State each metric's owner, required inputs, formula, scope, freshness, and missing-data rule. Partial coverage cannot silently shrink the denominator or become a passing default. |
| C20.3 | Keep current storage writability separate from full-restore evidence, and publication success separate from qualification. Test all combinations, including healthy collection with honest production-only debt. |
| C20.4 | Prevent stale summaries, unknown fields, synthetic fixtures, or a high structural score from cancelling hard blockers. Verify dependent dashboards retain the same reason and authority boundary. |
| C20.5 | Show raw post-cost outcomes, sample counts, uncertainty, candidate/account scope, and benchmark provenance without grade-based relabeling. Negative results remain negative after refreshing reports. |
| C20.6 | Reconcile command, dashboard, report, and readiness views against frozen owner inputs. Test missing/stale/adverse evidence and assert that no grade or score grants trading/promotion authority. |

**Verification start:** `tests/test_production_resilience_control.py`, `tests/test_platform_evidence_freshness.py`.
**Closure evidence:** Metric/denominator catalog, truth-table tests, and consistent deployed consumer receipts. A structural 100 is never a global 100-percent-complete claim.

### C21 Research Lineage And Reproducibility

**Lead:** `scripts/ops/training_lineage_manifest.py`.
**Collaborators:** `scripts/feature_store_manifest.py`, `scripts/experiment_tracker.py`, `scripts/build_runtime_training_snapshot.py`, `core/runtime_training_common.py`, `core/research_data_platform.py`.
**Baseline:** Immutable lineage and digest-bound snapshot publication exist; complete dataset/model/run binding and restart/replay proof remain incomplete.
**Verify with:** C03, C05, C06, C07, C08, C10, C13, C19, C22, C23, C24.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C21.1 | Bind raw source versions, event/availability watermarks, features/transforms, labels/outcome authority, dataset partitions, and model artifact digests to one experiment identity. Preserve unknown/unbound rows as nonqualifying. |
| C21.2 | Record exact code, dependency/configuration identities, seed, hardware/backend, numerical settings, and training parameters. Define deterministic equality or justified numerical tolerance before replay. |
| C21.3 | Enforce point-in-time joins, purged chronological validation, embargo where required, and candidate/generation separation. Inject future availability and leaked labels; prove rejection rather than silent inclusion. |
| C21.4 | Make checkpoints generation-bound and resumable with correct optimizer/model/data position state. Interrupt and resume; compare against an uninterrupted reference and invalidate changed-input checkpoints. |
| C21.5 | Verify bounded atomic publication, complete row/model hashes, reference-safe retention, and reader rejection of mixed generations or corrupted bytes. Reusing an artifact retains its original producer age. |
| C21.6 | Reproduce a release-bound experiment from immutable inputs in an isolated environment and obtain independent validation. Keep diagnostic/synthetic outcomes distinct from organic evidence and serving/promotion authority. |

**Verification start:** `tests/test_training_lineage_manifest.py`, `tests/test_feature_store_manifest.py`, `tests/test_experiment_tracker.py`, `tests/test_research_data_platform.py`.
**Closure evidence:** Complete dataset-to-model manifest, leakage tests, checkpoint comparison, deterministic/tolerance replay, and independently reviewed results.

### C22 Institutional Research Evidence

**Lead:** `core/institutional_research_extensions.py`.
**Collaborators:** `config/institutional_research_extensions_v1.json`, `scripts/ops/institutional_research_extensions_control.py`, `core/research_data_platform.py`.
**Baseline:** Eight advisory controls are implemented; seven evidence categories remain distinct obligations. Public design references and synthetic probes are not operational evidence.
**Verify with:** C05, C06, C13, C15, C18, C21, C23.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C22.1 | Candidate risk schedules: bind reviewed risk limits and time/regime applicability to the exact candidate; replay enforcement and record genuine observed schedule use without increasing authority. |
| C22.2 | Cross-engine valuation: compare independently implemented engines using aligned instruments, timestamps, conventions, and source-backed observations; record residuals, tolerances, and disagreements. Two wrappers over one engine are not independent. |
| C22.3 | Execution speed/cost frontier: measure candidate/instrument/size-bound latency, fill quality, fees, spread, and impact across eligible choices. Retain uncertainty and abstention; diagnostic latency curves cannot prove real execution economics. |
| C22.4 | Independent factor benchmarks: obtain permitted external benchmark provenance, align availability/time/currency, and calculate candidate-bound attribution with residuals and uncertainty. Internal features cannot be relabeled independent observations. |
| C22.5 | Pipeline incident ownership: link actual or explicitly labeled exercise incidents to accountable owners, lineage impact, delivery, acknowledgment, remediation, and recurrence evidence. A list of owner names is not an exercised incident workflow. |
| C22.6 | Research DAG checkpoint/resume: materialize dependency/checkpoint receipts, interrupt an actual isolated research run, resume only valid nodes, and compare output with a complete reference run. Changed inputs invalidate descendants. |
| C22.7 | Versioned dataset storage: materialize immutable content-addressed versions, schema/provenance/access metadata, and time-travel reads; reproduce a prior research result and verify reference-safe retention. A manifest with no accessible dataset is insufficient. |

**Verification start:** `tests/test_institutional_research_extensions.py`, `tests/test_research_data_platform.py`.
**Closure evidence:** Seven separately scoped genuine evidence sets with producer/input identities, independent-source proof where required, and candidate binding. New paid data, external services, or licensed storage engines need separate entitlement approval.

### C23 Economic Qualification And Promotion

**Lead:** `scripts/ops/candidate_scope_validation.py`.
**Collaborators:** `config/candidate_scope_validation_v1.json`, `scripts/ops/production_excellence_control.py`, `scripts/ops/continuous_soak_integrity_control.py`, `scripts/ops/profitability_evidence_firewall.py`.
**Baseline:** Qualification is gated by genuine candidate-bound outcomes and policy time/session requirements; engineering patches cannot manufacture them.
**Verify with:** C13, C16, C17, C18, C20, C21, C22, C24, C25.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C23.1 | Bind the accepted candidate, source/configuration/dependencies, strategy/risk/execution scope, and immutable event window. Unaccepted drift and unrelated historical generations earn no clean current-scope credit. |
| C23.2 | Collect sufficient independent post-cost outcomes and benchmarks under existing policy; retain sample counts, confidence, multiple-testing/selection-bias controls, regime coverage, and adverse results. No forced trades to fill a quota. |
| C23.3 | Satisfy drawdown, exposure, capacity, correlation, fill fidelity, fee/slippage, and operational/reconciliation gates. Preserve missing evidence as missing and test negative evidence blocks qualification. |
| C23.4 | Accrue both elapsed hours and completed XNYS sessions: operations 72/3; data/dependencies 120/5; promotion 336/10; strategy/execution/risk 720/20. Interrupted sessions and offline time receive no new credit under the existing policy. |
| C23.5 | Verify evidence invalidation after material changes and distinguish preserved cumulative history from qualifying current scope. Do not accept a candidate, rewrite clocks, or lower thresholds to obtain completion. |
| C23.6 | Produce a reviewable qualification packet with exact passing/failing gates, independent evidence, identity, and remaining approvals. Promotion, account release, and live execution remain separate explicit decisions even when qualification passes. |

**Verification start:** `tests/test_production_resilience_control.py`, `tests/test_production_readiness_control.py`; use the candidate/soak owners' existing suites for policy-clock and identity tests.
**Closure evidence:** Genuine candidate-bound forward evidence, policy-required hours/sessions, reconciled qualification packet, and recorded review. Passing qualification is not a guarantee of profit or automatic live authorization.

## Release And Ownership

### C24 Source And Release Organization

**Lead:** `scripts/ops/release_freeze_guard.py`.
**Collaborators:** `core/build_provenance.py`, `scripts/ops/production_excellence_control.py`, `scripts/ops/source_mutation_guard.py`, `scripts/ops/codex_project_guard.py`.
**Baseline:** Trading changes through the baseline commit were pushed; publication does not establish a clean accepted candidate, immutable deployed build, or rollback proof.
**Verify with:** C06, C08, C12, C16, C19, C23, C25, C26.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C24.1 | Classify tracked/untracked/ignored files as source, tests, policy, documentation, runtime evidence, secrets, cache, or separate-domain work. Track dependency-complete source explicitly; never blanket-add runtime or unrelated files. |
| C24.2 | Inventory worktrees with branch, unique commits, dirty/untracked state, active task/process use, and recoverability. Remove only approved inactive preserved trees; absent removable trees mean zero cleanup, not an invented saving. |
| C24.3 | Review changes, pass scoped secret/project/source gates, and publish to the intended remote/branch without force or hook bypass. Record commit and remote synchronization; push success alone is not release acceptance. |
| C24.4 | Build an immutable artifact with source manifest, dependency locks, interpreter/platform, redacted configuration, artifact digest, and trustworthy provenance scope. Local unsigned metadata is not a trusted CI attestation. |
| C24.5 | Verify the running process adopts the exact accepted release/configuration and that source drift is visible. Candidate acceptance remains explicitly operator-controlled; tests against a dirty checkout do not certify that release. |
| C24.6 | Exercise rollback to a known supported release with compatible data/configuration, one writer, verified restart, and retained evidence. Record failures and source/adoption state after rollback without rewriting candidate history. |

**Verification start:** `tests/test_release_freeze_guard.py`, `tests/test_build_provenance.py`; run the existing staged project/secret guards before publication.
**Closure evidence:** File/worktree classification, reviewed commit/remote receipt, immutable build provenance, accepted deployment identity, and approved rollback test. Separate-domain changes stay untouched.

### C25 Complete Test Evidence

**Lead:** `scripts/ops/chaos_drill_coordinator.py` for operational drills; each source owner owns its regression tests.
**Collaborators:** `scripts/ops/production_recovery_drill_harness.py`, repository test/CI owners, C24 release owner.
**Baseline:** Selected hardening suites pass; neither their count nor a successful full unit suite proves every production workflow.
**Verify with:** All other packages; verification can run incrementally, but final acceptance binds one release artifact.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C25.1 | Map every requirement to unit, integration, contract, migration, concurrency, failure-injection, load, restart, or soak evidence as appropriate. Mark uncovered requirements and separate discovered defects from unverified behavior. |
| C25.2 | Preserve deterministic reproduction fixtures for every corrected defect with before/after assertions. Run affected downstream contracts as well as the changed module; count unique tests and disclose scope. |
| C25.3 | Exercise the cross-boundary scenarios below under bounded CPU/memory/disk and isolated authority. Include timing, cancellation, lost acknowledgments, and corrupted/missing inputs, not only happy paths. |
| C25.4 | Run release-bound performance/load and approved long-duration observation with predefined workloads, budgets, duration, latency/backlog/error thresholds, and failure criteria. Short synthetic runs do not earn long-duration credit. |
| C25.5 | Track flaky, skipped, quarantined, unavailable, and failed tests with owner, reproduction, impact, and expiry. Rerun success cannot erase an earlier failure or silently remove a required test. |
| C25.6 | Retain commands, environment/fixture/artifact digests, exact test inventory, timestamps, results, and reviewer disposition. Rerun invalidated evidence after source/configuration/dependency changes. |

**Verification start:** `tests/test_production_recovery_drill_harness.py`, `tests/test_chaos_drill_coordinator.py`, and each package's listed suites.
**Closure evidence:** Requirement-to-test coverage matrix, immutable release-bound results, cross-boundary failures/recovery, approved load/soak observations, and no unreviewed required-test gaps.

### C26 Documentation And Operator Consistency

**Lead:** `scripts/ops/commands_hygiene_bot.py` for generated commands; `docs/architecture/SOURCE_OF_TRUTH.md` for source ownership.
**Collaborators:** `README.md`, `COMMANDS.md`, `docs/operations/PLATFORM_COMPLETION_CHECKLIST.md`, `docs/operations/PLATFORM_HARDENING_LEDGER.md`, this definition.
**Baseline:** Main owner/command docs are aligned; per-control completeness and deployed command/dashboard consistency need release-level proof.
**Verify with:** All other packages; documentation follows the implemented owner, not a desired green report.

| Requirement | Acceptance And Required Proof |
| --- | --- |
| C26.1 | Give every control one canonical owner, purpose, inputs/outputs/schema, consumers, mode, mutation/authority boundary, resource budget, failure behavior, and evidence location. Label incomplete behavior explicitly. |
| C26.2 | Document recovery prerequisites, safe diagnostic command, expected failure result, escalation owner, approval boundary, rollback, and post-action verification. Do not present a mutating command as a read-only check. |
| C26.3 | Regenerate command inventories from the owning generator and test supported flags/paths. Direct edits to generated docs cannot be the lasting source of a command contract. |
| C26.4 | Reconcile README, source map, runbooks, dashboards, and readiness labels against actual owner output. Timestamp historical examples; never describe a stale or scoped result as current platform-wide readiness. |
| C26.5 | Link each requirement, defect, test result, runtime receipt, release identity, and unresolved decision without creating conflicting status sources. Keep sensitive/runtime payloads out of tracked documentation. |
| C26.6 | Have an accountable reviewer execute the applicable runbooks against the release in a safe scope. Verify commands, failure/approval paths, and escalation contacts; record deviations before closure. |

**Verification start:** Existing command-hygiene and project guards, plus package-specific command/consumer tests.
**Closure evidence:** Source/command/link consistency checks, reviewed control/runbook inventory, safe operator walkthrough, and a release-bound completion packet. This document is a definition, not that packet.

## Cross-Boundary Acceptance

These are mandatory integration scenarios, not extra subsystem counts. Use
isolated fixtures first; production interruptions and heavy work require their
own approval. A scenario that cannot safely run remains unverified.

| Scenario | Areas | Required Outcome |
| --- | --- | --- |
| X01 Concurrent storage demand | C01, C02, C07, C09, C10 | Snapshot, vacuum, archive, and training requests cannot collectively overcommit a device; critical collection retains its policy reserve. |
| X02 Disconnected storage during commit | C01, C03, C05, C12 | Writer failure preserves acknowledged data/checkpoints and explicit route state; reconnect cannot create dual writers or a fake mounted target. |
| X03 Interrupted multi-file publication | C06, C10, C13, C21 | Readers reject mixed rows/health/model/manifest generations; prior complete evidence survives only within its original validity. |
| X04 Stale green dependency | C05, C09, C11, C13, C20 | Fresh wrappers cannot certify old inputs, clear a write failure, close a repair circuit, or upgrade readiness. |
| X05 Broker accepted, response lost | C12, C16, C17, C18 | No automatic resubmission; durable ambiguity blocks new exposure until account/order reconciliation resolves it. |
| X06 Crash after SQL commit before checkpoint | C03, C05, C06, C21 | Replay converges to the same logical data without skipped committed events or duplicate economic/training credit. |
| X07 Sleeping host and clock jump | C08, C09, C12, C13, C23 | Bounded catch-up, correct override/evidence expiry, no duplicate logical run, and no unearned market-session credit. |
| X08 Repair exhaustion plus delivery failure | C11, C14, C15, C16 | Circuit stays open, attempts stop within budget, approved alternate escalation is recorded, and history is not erased. |
| X09 Source change during qualification | C08, C13, C21, C23, C24 | Old evidence remains historical; current-scope qualification/adoption rejects identity mismatch without automatic acceptance. |
| X10 Restore with corrupt archive/reference | C03, C04, C06, C07, C21 | Restore reports exact coverage/loss, original remains preserved, references prevent unsafe retirement, and partial success never becomes full-recovery proof. |

## Execution Order And Decisions

The "Verify with" references describe shared proof obligations, not a cyclic
job schedule. Implement independently where possible; integration closure binds
the resulting interfaces to one release. The four waves in the checklist remain
the delivery order, with these concrete first actions:

1. Reproduce current-write versus recovery-evidence coupling (C05.6/C20.3),
   complete resource/route inventory (C01.1/C02.1), and define the approved
   recovery/archive exercise (C03.1-C03.2/C04.1-C04.2).
2. Finish typed compatibility/replay and retention boundaries, then verify the
   loaded job/configuration/deadline/circuit contracts in unattended operation.
3. Verify mutation/accounting/security paths, obtain approved remote-monitor
   evidence, and materialize reproducible research and independent observations.
4. Freeze the reviewed release, verify adoption/rollback and invalidated tests,
   then close only requirements whose implementation and operational proof agree.

The following decisions cannot be guessed or manufactured by a coding pass:

| Decision | Required Before |
| --- | --- |
| Named accountable maintainers, independent reviewers, and escalation contacts | Incident exercise and final reviewed closure. Source owners alone are insufficient. |
| Isolated non-protected restore/salvage destination, allowed data scope, disk/memory budget, and resource window | Production-sized copy, restore, and archive investigation. Existing BOT_LOGS use is not approval for an arbitrary new recovery workload. |
| Effective RPO/RTO and acceptance workload/SLO values frozen in their owners | Recovery/load tests. Do not relax targets after seeing failed results. |
| Interruption scope, rollback target, and maintenance window | Actual restart, logout/login, reboot, failover, or host-disappearance tests. |
| Off-host receiver, recipient, permitted payload, authentication, and test notification approval | External delivery and receiver-side deadman verification. No destination is selected by this document. |
| Entitlements and independent sources | Any new paid/licensed data, external valuation inputs, or storage-engine adoption. |
| Reviewed candidate/release acceptance | Deployment qualification and new scope clocks; a Git push cannot perform acceptance. |
| Real unchanged-candidate history and market sessions | Economic/time-based qualification. A test fixture cannot replace forward observations. |

Defects discovered while executing these requirements receive a reproduction,
source-owner fix, regression test, and ledger entry. Genuinely new requirements
get stable IDs and review; do not suppress them to preserve the count. Reopen
closure when evidence expires, identities change, assumptions fail, or new
counterexamples are reproduced. Completion is demonstrated conformance to this
agreed scope, not a promise of zero hidden defects, permanent availability, or
profitability.
