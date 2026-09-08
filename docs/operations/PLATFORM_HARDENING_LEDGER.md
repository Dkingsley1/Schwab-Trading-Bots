# Platform Hardening Ledger

Audit date: September 7, 2026. This is a scoped engineering record, not a
certificate that the platform has no defects. Implementation coverage, runtime
availability, recovery proof, and live-promotion qualification are separate
claims. The dirty working checkout is not a newly accepted release.

## Closure Standard

A finding is closed when its failure is reproduced, its owning source is fixed,
regression tests pass, affected consumers are checked, and required deployment
or operational evidence is available. Tests cannot substitute for an approved
production restore, clean release, off-host alert receipt, or organic research
evidence. This ledger adds no daemon, trading permission, promotion authority,
or deletion authority. Use the existing source-of-truth owners.

## Corrected Defects

| ID | Failure And Correction | Owner / Regression Evidence |
| --- | --- | --- |
| PH-01 | Missing required fields could produce six `needs_work` contract rows with aggregate `ok=true`. Aggregate success now requires every row to be ready. This remains a field/version-presence check, not complete schema compatibility validation. | `scripts/schema_migration_guard.py`; `tests/test_schema_migration_guard.py` |
| PH-02 | Fresh health wrappers reused old platform grades. Fast health and the architecture referee now expose producer time and classify stale, missing, invalid, or future evidence as unavailable, without substituting file mtime. | `scripts/ops/long_runtime_common.py`, `health_fast.py`, `system_architecture_hardening.py`; `tests/test_platform_evidence_freshness.py` |
| PH-03 | Platform and writer producers were absent from production refresh. Five bounded observational steps now precede dependent reports. Their 30-minute eligibility fits the existing 45-minute cadence and 60-minute consumer freshness budget. Completion still depends on scheduler health, serialization, timeouts, and load. | `scripts/ops/readiness_evidence_refresh.py`; profile ordering tests |
| PH-04 | Allowed exit codes plus existing stale files counted as successful refreshes. The runner now verifies published producer time and reads status from the artifact, not stdout. Invalid/future timestamps cannot hold the refresh in cooldown through file mtime. Publishing pending qualification is still distinct from passing qualification. | `scripts/ops/readiness_evidence_refresh.py`; timestamp, cooldown, and stdout-disagreement tests |
| PH-05 | Oversized snapshot targets received restore credit and fake SHA-256 values derived from metadata. Metadata-only rows now have no hashes and no restore credit; incomplete and empty target sets fail verification. | `scripts/daily_state_snapshot_drill.py`; snapshot regression tests |
| PH-06 | Raw SQLite main-file copies could omit committed WAL data. Snapshot drills now use read-only online backup with logical-page and time budgets, compare snapshot/restore hashes, and run SQLite quick-check. Regular-file changes during copying fail verification. | `scripts/daily_state_snapshot_drill.py`; active-WAL, logical-byte-budget, and mutation tests |
| PH-07 | Zero retention could delete the current successful restore, and arbitrary directories were pruning candidates. The current run is retained; only owned completed timestamped runs are eligible; symlink directories are excluded. Run names include microseconds. | `scripts/daily_state_snapshot_drill.py`; zero-retention and owned-run tests |
| PH-08 | Snapshot traversal lacked the storage boundary check. Sources/output routes now use bounded protected-path inspection. Resolved external sources use parent-specific namespaces, not basename alone. | `core/storage_router.py`, `scripts/daily_state_snapshot_drill.py`; protected-symlink and route-contract tests |
| PH-09 | Isolated simulated timings counted as platform recovery-time proof. A separate control-drill result now leaves full-platform RTO unverified. Negative snapshot ages cannot satisfy RPO. This fixes the false claim, not the missing production restore capability. | `scripts/ops/storage_disaster_recovery.py`; recovery-scope, negative-age, and timestamp tests |
| PH-10 | Settlement combined a current active-writer observation with writer proof over eleven hours old and recommended forced maintenance. Stale proof is now explicit, cannot certify a single writer, and requests observational refresh first. Refresh showed active progress and restored settlement readiness without a restart. | `scripts/ops/platform_settlement_stabilization.py`; stale-proof regression and live observations |
| PH-11 | The ingestion-definition command was manually documented but absent from its generator. It now belongs to the command inventory, and regenerated docs/index agree with that owner. | `scripts/ops/commands_hygiene_bot.py`; generator tests and second hygiene pass |
| PH-12 | Matching report/control hashes with no execution evidence were classified as a synchronization failure. Publication now succeeds only with matching unchanged hashes, fresh/stable input, and either grading eligibility or the explicit no-execution-evidence deficit alone. Qualification remains blocked and independently reported; stale, unstable, mismatched, or otherwise incomplete inputs still fail. | `scripts/paper_performance_report.py`; publication/qualification, wrong-hash, stale, unstable, incomplete, and child-exit tests; next unattended result still requires observation |

## Verification

- Final expanded regression suite: **515 passed**, spanning snapshot/DR, refresh,
  platform health, ingestion/storage, broker adapters, rate limiting, promotion,
  live-execution safety, paper performance/profitability, and runtime refresh.
  This is a selected 29-file suite, not the full repository test suite.
- The corrected schema guard reports six ready artifact contracts.
- Command hygiene reports 230 entries, no drift, and no issues after regeneration.
- Platform and writer observations were refreshed without applying their runtime
  recommendations. The stale writer warning cleared through observation alone.
- The ordinary DR producer published the corrected RTO separation: isolated
  drill passing, full production restore still unproven.
- At 22:06 UTC, fast health reported `guarded_ready`, guarded paper enabled,
  collection ready, calm pressure, and no global halt. This is a time-bound
  observation, not an unattended-soak or live-release guarantee.
- The 23:33 UTC grade guard and non-applying autopilot report one storage-control
  blocker after the honest snapshot result propagated. At 23:04 UTC, four of
  five selected files had verified restores; the 18,163,015,680-byte database
  exceeded the copy budget and received no credit. This is recovery-evidence
  debt, not proof of active-route failure. Fast health still reported guarded
  collection ready after this blocker appeared. No autopilot repairs were applied.
- The 22:06 accrual cycle timed out in runtime training snapshot construction.
  A later cycle produced a fresh snapshot at 23:31 UTC, but the intermittent
  timeout's cause and recurrence protection remain open. Paper synchronization's
  no-execution-evidence false failure is corrected in PH-12 and test-verified;
  the last inspected unattended receipt predated that correction.
- Staged secret scan: zero findings. Project guard: all six checks ready. Staged
  and unstaged whitespace checks passed. New source/test documentation is tracked;
  mixed existing edits were preserved. No commit or push was made during the
  initial hardening pass; subsequent publication was requested separately.

## Subsequent Publication Verification

The operator requested publication of all accumulated trading-system changes,
with the separately scoped runtime work excluded. The final staged selection
contains 368 changed paths, including the completion checklist. All 126 changed
trading-system test files passed: **2,268 tests and two subtests**. This broader
run found a formatting-sensitive assertion in the stack-control test; it now
checks the Python expressions without changing the safe runtime defaults.

The staged secret scan has zero findings, whitespace checks pass, and all seven
staged project-guard checks pass. The production-candidate source-drift report
remains a separate unresolved runtime gate; no candidate was accepted and no
hook was disabled. The repository's checked-in hook was not installed as an
active Git hook in this checkout. Git publication does not certify deployment,
an unattended soak, recovery readiness, or live-promotion qualification.

## Remaining Work

### Detailed Acceptance Definitions

The operator requested further definition of all broad areas. The versioned
`docs/operations/PLATFORM_COMPLETION_WORK_PACKAGES.md` expands C01-C26 into
157 stable requirements with lead/collaborating sources, baseline obligations,
verification starting points, closure evidence, ten cross-boundary scenarios,
and explicit external decisions. The checklist still owns status. These are
acceptance definitions, not 157 diagnosed defects, a new runtime evaluator, or
new completion credit. Defect IDs below remain reserved for reproduced failures
and their actual corrections. No runtime setting, candidate, authority, or
approval boundary changes as a result of this documentation expansion.

### All-26 Follow-Up Implementation

The following additional failures were reproduced and corrected during the
operator's all-26 request. They are implementation fixes with focused test
evidence, not completed release-level work packages.

| ID / Criteria | Reproduced Failure And Correction | Owner / Tests |
| --- | --- | --- |
| PH-13 / C02 | Same-filesystem vacuum demand used the largest individual allocation instead of simultaneous demand. Reclaim now sums rewrite and scratch requirements and rejects unknown, nonfinite, negative, or boolean capacity values. | `scripts/ops/sqlite_reclaim_control.py`; shared-volume and malformed-capacity tests |
| PH-14 / C02 | Daily restore drills lacked an aggregate capacity admission check. Two capped copies per eligible target plus the configured live-growth reserve (at least 64 GiB) are required, with per-target rechecks and the existing storage-maintenance lock. Regular-file copies have byte/time bounds and owner-only access. | `scripts/daily_state_snapshot_drill.py`; combined demand, unavailable/full disk, higher reserve, busy lane, and bounded-copy tests |
| PH-15 / C07 | Failed drill attempts could prune successful history. Retention now requires published, verified current evidence, and only successful owned historical runs are eligible. Failed/unverified history is preserved for investigation, not treated as free space. | Daily snapshot owner; failed-current and failed-history retention tests |
| PH-16 / C03 | Recovery manifest verification accepted missing sizes/hashes, duplicate rows, and receipts that were unchanged across different valid contents. It now requires complete typed unique records, exact digests, content-bound receipts, and permitted in-root physical paths. Size-only large-file observations remain unverified. | `scripts/ops/storage_disaster_recovery.py`; malformed, duplicate, same-size content, and protected-symlink tests |
| PH-17 / C13 | Recovery and immutable-control evidence could use file mtime or future producer time. Both now use the shared producer-time contract and publish freshness reasons. | DR owner; absent, invalid, and future timestamp tests |
| PH-18 / C10 | Vacuum's child process lacked an outer deadline, while training's incremental budget excluded discovery, fallback, and publication. Vacuum now has a 960-second parent timeout with an unknown-outcome receipt; snapshot CLI wraps its worker in the existing process-group deadline/cleanup owner, defaults to 150 seconds, and reports the last phase. | Reclaim and training snapshot owners; owned-hold timeout release and actual hanging-child cleanup tests |
| PH-19 / C10, C21 | Training rows were overwritten in place and readers did not verify the published digest. Rows now stage and atomically replace, health uses the existing atomic writer, and schema-v2 readers/reuse require the matching hash. The brief two-file publication gap fails closed; it is not claimed to be a multi-file transaction. | `scripts/build_runtime_training_snapshot.py`, `core/runtime_training_common.py`; interrupted publication, digest, mismatched-generation, missing-hash, and temporary-symlink tests |
| PH-20 / C19, C24 | Security audit treated a checked-in hook as active, defaulted missing scan counts to clean, and credited future/mtime-based receipts. It now observes Git's actual hook route/executability, requires an explicit valid count, uses producer time, and accepts the documented `--json` flag. It does not install hooks or accept source drift. | `scripts/security_hardening_audit.py`; actual temporary Git configuration, CLI, and malformed-time tests |
| PH-21 / C09, C13 | The shared plumbing report was almost four days old and its producer was absent from the production refresh profile. The existing read-only owner now runs after writer observation and before the architecture consumer, with 30-minute eligibility inside the 45-minute cadence. | `scripts/ops/readiness_evidence_refresh.py`; producer inclusion, ordering, read-only authority, and consumer dependency assertions |

Remaining limits: the minimum 64 GiB snapshot margin is conservative admission, not a
measured fleet-wide forecast or reservation against nonparticipating writers.
The full platform recovery protocol, damaged archive salvage, and production
restart proof are still open. Training phase diagnostics and total deadlines
contain recurrence; they do not establish the cause of the earlier observed
180-second timeout or prove repeated unattended success. Schema-v1 training
fixtures remain a legacy compatibility path, not v2 integrity evidence. Active
hook declaration is configuration evidence, not proof of every shell branch or
a security certification. No production data was deleted or writer restarted.

### Follow-Up Verification

- The 75-file cross-platform suite passed **1,011 tests and two subtests** after
  correcting a command-documentation expectation for the new deadline flag.
  The final snapshot read-boundary adjustments passed **107 focused tests**;
  the subsequent missing-producer integration passed **72 tests** spanning its
  scheduler, plumbing, freshness, and architecture consumers. These overlapping
  runs must not be added together as a unique-test count or full-platform proof.
- The bounded snapshot CLI verified and reused the actual 9,123-row, 504-sequence
  snapshot at 00:49 UTC on September 8. Its original producer timestamp and row
  hash were preserved. This checks reuse and worker dispatch, not a new full
  rebuild or repeated unattended adoption.
- At 00:46 UTC the corrected security audit reported 16 passing checks and two
  failures for inactive hook enforcement. No hook was installed or disabled.
  The non-applying regression autopilot showed storage blocked and security
  degraded, with zero repair attempts.
- At 00:54 UTC fast health was guarded-ready, with no global halt and live
  execution explicitly blocked. At 00:55 UTC the refreshed plumbing owner
  reported healthy routes, queues, and progressing writer, but unresolved
  data-plane write-recovery and global-clear blockers. That evidence remains
  open under C09/C13/C20; historical incidents were not reset to force readiness.
- Command hygiene reports 230 entries with no drift. The staged secret scan
  has zero findings and the staged project guard passes all seven checks.
  Production-sized restore, archive salvage, disruptive restart exercises,
  off-host monitoring, and candidate/release acceptance remain unperformed.

These are open implementation packages or evidence prerequisites, not defects
closed merely by documenting them. Missing evidence must not become a green grade.

| Priority / Area | Existing Owner | Required Closure Evidence |
| --- | --- | --- |
| P1: Production restore and RTO | `scripts/ops/storage_disaster_recovery.py`, `scripts/daily_state_snapshot_drill.py` | Implement/review an explicit full-scope restore protocol: manifest and model/control/data coverage, isolated permitted destination, capacity/runtime budgets, checkpoint consistency, digest/row checks, application restart checks, measured duration, cleanup ownership, immutable receipt. A production-sized run needs an approved destination and maintenance scope. |
| P1: Failed legacy archive | SQL hot retention and storage maintenance owners | Diagnose the preserved corrupt archive on an approved recovery copy, verify repaired/exported contents, and record any loss. Error isolation did not repair corruption. |
| P1: Configuration provenance/adoption | `scripts/ops/load_runtime_env.sh`, role contracts, override producers | Redacted per-setting winning source/precedence, owner, type/constraints, expiration, process adoption fingerprint, and restart debt. Declared ordering is not runtime adoption proof; never export credentials. |
| P1: Job lifecycle | Readiness refresh, production hardening watch, maintenance owners | Per-surface scheduled/eligible/deferred/started/completed/failed and next-eligible records; dependency-failure propagation, clock/restart tests, retry/timeout budgets, and observed unattended completion. Ordering tests do not prove every job ran. |
| P1: Intermittent snapshot timeout | `scripts/build_runtime_training_snapshot.py`, readiness refresh | Reproduce the observed 180-second process timeout despite the 30-second incremental scan budget; measure lock, discovery, parsing, fallback, and publication phases; bound the entire operation without relabeling old data as fresh. Verify multiple unattended cycles. A later successful refresh is recovery evidence, not a root-cause fix. |
| P1: Independent monitoring | `scripts/observability_exporter.py` | Verify local sentinel and paper-regression freshness; configure an operator-approved off-host receiver; test delivery and real failure detection. No external destination or credentials were invented. |
| P1: Runtime authority coverage | Role/operating contracts, broker and execution entry points | Prove every mutating entry point enforces its action guard, including retry and restart paths. A complete role registry is not invocation-coverage proof. Keep live/promotion locks intact. |
| P1: Release isolation | Source-mutation, immutable-release, project and candidate guards | Split mixed dirty work into dependency-complete reviewed changes, test a clean immutable build, and verify rollback. Publication follows operator instruction; candidate acceptance remains a separate explicit decision. |
| P2: Training/qualification vocabulary | Training quality, grade and promotion consumers | Separate structural coverage, current quality, unresolved improvements, and candidate-bound eligibility. A score of 100 must not imply all qualifications passed. |
| P2: Full schema compatibility | Schema migration guard and its six producers/consumers | Supported versions, required types/values, nested invariants, migration compatibility, and replay fixtures. PH-01 fixes aggregate truth, not every validation gap. |
| P2: Seven institutional evidence gaps | `scripts/ops/institutional_research_extensions_control.py` and research owners | Real risk-schedule, cross-engine valuation, execution frontier, independent factor, incident ownership, DAG checkpoint/resume, and versioned dataset evidence. Eight implemented controls and one evidenced control are different coverage measures. |
| P2: Ingestion failure behavior | Collector contracts, channel queue, SQL link/shard owners | Duplicate/out-of-order/replay, interrupted acknowledgment, poison-record, schema-drift, saturation, and restart tests with data-loss/duplication accounting. Route observations do not certify database integrity. |
| P2: Capacity and combined failures | Storage reserve, resource and maintenance owners | Shared disk/memory/CPU budgets covering WAL growth, snapshots, vacuum, archives, training, and foreground use; combined-pressure tests. Protected media is not capacity. |
| P2: Dependency/security release checks | Security audit, lockfiles, secret scanner, CI | Reproducible installation, current advisory assessment, complete changed-code scanning, and isolated clean-machine boot tests. A local secret scan is not a dependency audit. |
| P2: Alerts and incidents | Escalation, incident closeout, independent monitor owners | Deduplication, severity/tenant routing, delivery failure, escalation exhaustion, acknowledgment, and reopened-incident tests with bounded fixtures and an approved receiver. |

## Scope Boundaries

No live orders, credential changes, candidate acceptance, promotion, risk
loosening, production archive deletion, production-sized restore, or writer
restart was performed. Protected media was not inspected. Earlier disk recovery
is recorded in `docs/operations/SOURCE_AND_STORAGE_MAINTENANCE.md` and is not
counted again here. Only the main Git worktree existed at the earlier inventory;
there were no disposable worktrees to remove. Existing staged and unstaged work
was preserved.

## September 8 Degradation Repair

This is a targeted repair, not closure of the 26 platform areas.

| Defect | Implemented Correction | Remaining Boundary |
| --- | --- | --- |
| Raw compaction trusted a one-byte gzip read and prefix-only duplicate check | Full restored SHA-256 and byte count, stable source identity, no-clobber publication, synced archive publication, scratch reserve, bounded processing, and protected-route checks before release | Cooperative I/O deadlines and per-operation reserves are not a fleet-wide allocation reservation or a production power-loss drill |
| Hot retention could ignore a conflicting archive insert and then delete source rows | Compare every archived field in bounded batches and commit the archive with FULL synchronization before source deletion | Production-sized retention/reclamation, schema evolution, and existing corrupt archives remain separate work |
| Mounted BOT_LOGS was called ready for rehoming without a capacity budget | Full local-fallback census plus reserve; incomplete scans, errors, and insufficient capacity block the recommendation; partial budgets are explicitly lower bounds | This does not perform a route switch or prove a less expensive, scoped migration plan |
| A successful accrual refresh erased production failure details | Preserve failed step IDs and bounded diagnostic receipts independently for each refresh profile | Historic receipts that already lost their detail cannot be reconstructed |
| One Numbers `generated_utc` was rejected as an absent producer timestamp | Recognize the existing producer field while retaining invalid, stale, and future-time rejection and higher-priority timestamp precedence | A recently generated report does not establish post-auth measurement freshness or permit trading |

Operational observations through 14:37 UTC:

- Verified and removed 158 eligible historical raw duplicates, preserving their
  full compressed contents. Receipts record 412,299,649 raw bytes released at
  `governance/health/raw_compaction_recovery_20260908.json` and
  `governance/training/raw_compaction_recovery_20260908.json`. These are runtime
  evidence, not source files to commit. Active/current-day logs were protected.
- Internal free space rose from approximately 31 GiB at initial inspection to
  approximately 36.8 GiB. Only the 412 MB receipt is attributed to this cleanup;
  concurrent host activity accounts for the other change. The configured SQL
  writer pause below 64 GiB remains intact. Collection can continue while the
  paused writer causes queue growth: 41,143 pending lines at 14:33 UTC.
- The 19 local shards occupy approximately 278 GiB. The dominant trading and
  governance databases had no material free pages. A bounded inspection of the
  100 largest cold SQLite archives found none with more than 512 MiB of free
  pages. Vacuum alone cannot fix this capacity problem. Weeks-old hot records
  and retention scheduling/admission remain unresolved.
- BOT_LOGS had approximately 116.4 GiB free. No alternate destination was
  invented, no route was switched, and no retained SQLite archive was deleted.
  Additional permitted capacity or an explicitly reviewed archival policy is
  required before larger recovery; protected media remains excluded.
- The first bounded production refresh passed 78 of 79 operational steps and
  isolated the One Numbers timestamp/lock failure. A subsequent automatic
  accrual refresh preserved that production failure receipt, proving adoption
  of per-profile diagnostics. Post-fix production verification is separate.
- The non-applying regression guard and autopilot still report storage and
  live-canary blockers, plus training, lineage, security, autonomy, and promotion
  evidence debt. Their status was not forced green; no repair retries were
  invoked through the autopilot. Candidate acceptance, off-host monitoring,
  inactive hook enforcement, post-auth execution evidence, and full recovery
  certification are not closed by this repair.

Only the main Git worktree exists. Unrelated registry and audio-runtime changes
remain outside this repair. There was no stack restart, token modification,
order placement, candidate acceptance, or reserve/qualification relaxation.

### Final Repair Verification

- The final 14-file regression suite passed **418 tests and two subtests**.
  This covers the changed producers plus storage, shard-manager, runtime,
  plumbing, health, and architecture consumers; it is not a full-platform test.
- The corrected scheduler accepted One Numbers' real producer timestamp without
  executing a duplicate rebuild. One diagnostic training refresh exceeded the
  shorter 90-second limit; its isolated retry completed within the normal
  180-second limit and preserved the actual `paper_performance_input_not_gradeable`
  dependency blockage. It was not counted as qualification success.
- The final production pass started at 14:46 UTC with the normal 180-second
  timeout: 35 refreshed, 43 already fresh, and one failed out of 79 steps. The
  remaining failure was `health_gates`, which published current evidence and
  returned 2 for real ingestion/backpressure overload. No remaining failure in
  that pass was a timeout or the One Numbers timestamp defect.
- At 14:48 UTC fast health remained degraded: collection ready, 35,093 pending
  lines, guarded paper blocked. Storage, restart-storm, and write-path recovery
  blockers remained visible. A falling queue in one sample is not sustained
  recovery proof. Additional permitted capacity remains the main unresolved
  operational prerequisite; guard statuses and thresholds were not overridden.
- The staged project guard passed all seven checks and the staged secret scan
  found zero findings before publication. The live hook enforcement gap remains
  separate; no hook was installed, disabled, or bypassed.
