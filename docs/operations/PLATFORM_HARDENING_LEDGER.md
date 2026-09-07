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
| P1: Release isolation | Source-mutation, immutable-release, project and candidate guards | Split mixed dirty work into dependency-complete reviewed changes, test a clean immutable build, and verify rollback. Preserve existing work; no automatic commit, push, or acceptance. |
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
