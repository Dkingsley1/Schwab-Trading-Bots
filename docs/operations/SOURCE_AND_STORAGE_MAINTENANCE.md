# Source And Storage Maintenance

## Source Organization

Keep new files in the existing ownership layout. Moving modules merely to group a
Git change can break imports, operator commands, or LaunchAgent paths.

| Location | Tracked Contents |
| --- | --- |
| `config/` | Versioned policies and redacted example configuration |
| `core/` | Runtime contracts, broker adapters, and reusable decision logic |
| `scripts/ops/` | Operator commands and the producers of health evidence |
| `scripts/install_*_launchd.sh` | Service installation entry points |
| `tests/` | Tests paired with their source owners |
| `docs/architecture/` | Ownership, evidence, and authority contracts |
| `docs/operations/` | Operator procedures and recovery notes |

Real account alias and policy registries, credentials, tokens, databases, and
generated health artifacts remain ignored. Track the `.example.json` account
registries, not the operator's populated copies. Stage explicit paths; changes
to existing modules may have dependencies across several feature groups. Staging
new files does not create a commit or certify the dirty tree as a production
candidate.

## Retention Safety Audit

The September 24 retention repair changes existing owners, not the native schedule:

- Stale-manifest compaction preserves event order when a path is staged again
  after a purge. It retains active records for unavailable files without probing
  manifest-supplied paths. Corrupt or unknown records block retirement rather
  than disappearing during compaction.
- Purge rechecks current evidence protection even when an old manifest says
  otherwise. Fallback suffixes and generic filenames under evidence directories
  cannot waive protection. Integrity approval must be the boolean `true`.
- File/byte selection happens before content hashing. Only selected files are
  hashed, once; device, inode, size, modification/change times and single-link
  identity are checked before and after verification. Arbitrary linked roots and
  payloads are rejected; only the exact router-owned `data/stale_stage` alias to
  local fallback storage is accepted. Legacy reindexing also verifies identity.
- Native reaper budgets must be finite and positive after unit conversion;
  optional oversized budgets can be zero to disable that lane. Failures publish
  an explicit incomplete-work receipt, not zero-work success. A busy contender
  leaves the active owner's canonical receipt untouched.
- Cold JSONL compression checks full source/archive identity, including
  same-size rewrites. Publication cannot replace a target created concurrently.
  Archive data and directory entries are synced before original release.

Regression tests use temporary fixtures. No production history was expired and
no live compression, offload or migration was applied by this audit. Changes
remain subject to the normal source acceptance/release process. The next native
pass still requires resource admission and existing owner locks.

This is not a claim that every retention implementation is bug-free. Actual
mounted-drive disconnect/restore behavior, current reclaimable capacity, and
long-running production cleanup still require operational verification. File
metadata checks are not writer leases; compression retains its coordinated
writer handoff. Do not relax the storage reserve or erase evidence to turn a
health report green.

### September 25 Follow-Up

Live verification found the active `stale_manifest.jsonl` had itself been moved
by deep-cold offload, leaving a link to the archive. The reaper now restores this
specific owned control file locally before indexing. Recovery has a 64 MiB bound,
full copied-content SHA-256, file/link identity checks, fsync and a durable proof
in `governance/storage_recovery/stale_manifest_restore.jsonl`. The archive is
preserved. Unknown links are rejected, not adopted. The existing reaper singleton
owns recovery; no additional scheduler is installed.

Deep-cold selection excludes `stale_manifest.jsonl`,
`backlog_quarantine_manifest.jsonl`, its own manifest, latest/state JSON controls
and lock files. These control files are not bulk archival payloads.

A bounded native pass restored the 38,157,525-byte manifest, removed four
verified expired files totaling 310,997 bytes, and completed without errors.
This is cleanup-path verification, not reserve recovery: internal free space was
about 54.5 GiB against the existing 125 GiB target. The writer remained paused,
so backlog, fresh training evidence, incident closeout and promotion remained
unresolved. Do not erase historical failure debt or lower proof thresholds.

The security scan's 27 findings were in third-party code inside the preserved
Python rollback environment. Full scanning now prunes virtual environments by
their standard `pyvenv.cfg` marker before descent, including renamed backups,
while still scanning other work files. Detection rules and staged-file scanning
are unchanged. A fresh native scan returned zero findings and all 18 security
audit checks passed; this is not a security certification.

## Database Space

The analytics mirror can also consume temporary space independently of database
file growth. SQLite sorter files may be unlinked while still open, so directory
size alone cannot account for them. The mirror now streams compact projected
fields rather than grouping raw JSON, with 250,000-row, 32 MiB projection and
20-second source limits. It commits both operational summaries only after the
whole requested window completes. Partial scans roll back, not publish lower
counts as complete. The native coordinator still owns scheduling.

The singleton CLI worker has a 35-second process-group deadline with bounded
cleanup. Source/output routes are checked against protected storage; configured
external caches require a mounted volume. Admission requires 1 GiB above the
pressure floor, never less than 65 GiB free, with repeated checks on the internal
volume and output volumes. Failed or deferred reports preserve honest cache
freshness debt. These controls do not reclaim occupied history or lower the
platform's reserve targets.

SQLite file size, occupied pages, and reusable free pages are different measures.
Deleting or archiving rows usually leaves free pages inside the database file.
That space becomes reusable by SQLite but is not returned to the filesystem until
compaction. A small live-page estimate cannot establish a healthy disk reserve.

`scripts/ops/sql_link_shard_manager.py` uses occupied pages for retention budgets
and physical size plus material reclaimable pages for vacuum selection. A vacuum
candidate must still satisfy the existing free-space reserve and maintenance
controls. `scripts/sql_hot_retention.py` honors a requested vacuum when pages from
an earlier pass are reclaimable, including a pass that moves zero new rows.

Heavy compaction belongs in a coordinated maintenance window. The maintenance
owner waits for the single SQL writer and retains its lock. SQLite maintenance
accepts that owner's valid hold token; unrelated processes remain blocked. An
explicit scratch directory must have enough capacity. The protected media volume
is excluded from temporary directory selection, including explicit paths and
symlink aliases.

Use the existing operator surfaces for inspection and coordinated maintenance:

```sh
./scripts/ops/opsctl.sh local-storage-reserve-guard --json
./scripts/ops/opsctl.sh storage-retention-unison --json
./scripts/ops/opsctl.sh writer-cycle-coordinator --json
```

The SQLite LaunchAgent runs `scripts/ops/sqlite_reclaim_control.py` before its
ordinary maintenance pass. This controller requires at least 2 GiB and 10 percent
reclaimable pages, memory clearance, an owned writer hold and lock, rewrite
headroom on the database volume, and scratch capacity. External scratch must also
preserve the 125 GiB reserve. A compact database is a no-op. The LaunchAgent keeps
its 03:40 schedule and retries hourly through the existing quiet-window guard.
The older unconditional size-based auto-vacuum option remains disabled by default.

The controller writes `governance/health/sqlite_reclaim_control_latest.json`,
including physical, live, and reclaimable space; deferral reasons; and the result
of a completed vacuum. A deferred run does not claim recovered space.

## Shard Hot-Retention Coverage

The nine previously uncovered large shards now declare hot-retention defaults in
`scripts/ops/sql_link_shard_manager.py`. These are ingestion-time windows for
`jsonl_records`, not event-time expiry, JSON snapshot cleanup, or a limit on total
retained evidence. Current source filters, merge behavior, IDs and writer routes
are unchanged. Historical labeling already discovers the shard owner's archive
and cold-export roots; unavailable archives remain explicit missing coverage.

| Shards | Hot Window | Size Trigger Per Shard |
| --- | --- | --- |
| `aggressive_trading`, `trading` | 7 days | 32 GiB |
| `governance`, `crypto_governance` | 14 days | 16 GiB |
| `runtime` | 7 days | 4 GiB |
| `crypto_runtime` | 7 days | 2 GiB |
| `support_watchdog` | 14 days | 2 GiB |
| `api_ingress`, `crypto_api_ingress` | 7 days | 1 GiB |

Trading mirrors the existing seven-day crypto-trading hot window. Operational
runtime/ingress retains the same week; governance and watchdog keep two weeks of
online investigation history. Size triggers request maintenance; they are not
hard caps and cannot make younger rows eligible. Each new rule also triggers on
100,000 inserted rows or its configured growth threshold, retaining a five-minute
minimum interval and the existing empty-pass backoff.

Each new rule uses 1,000-row batches, at most 5,000 rows per invocation, daily
archives in the existing per-shard namespace, and no inline vacuum request.
`hot_retention_archive_retention_days=0` explicitly disables this owner's archive
expiry. Zero is preserved through specification and command construction, not
replaced by the legacy 365-day default. This does not modify other retention
owners or shorten any existing shard's history policy.

The existing admitted storage-maintenance lane supplies writer ownership,
120-second per-database bounds, archive reserve plus scratch checks and verified
all-column copies before retiring hot rows. Conflicting historical rows keep
their separate version partitions. Ordinary one-pass ingestion still defers
retention. No new scheduler, immediate cleanup, archive deletion, policy-force
override or source acceptance is implied by assigning defaults. Failed failback,
resource, storage or writer admission still blocks maintenance. Physical disk
recovery requires a separately admitted reclaim pass after pages become reusable.

## Automatic Recovery Controls

Storage recovery uses the existing native self-healing and SQL-writer schedules,
not a separate automation. Inspect its current assessment with:

```sh
./scripts/ops/opsctl.sh soak-self-heal --storage-recovery-only --rebuild-reserve --json
```

Add `--apply` for an admitted recovery pass. The normal low-load lane retains its
1,800-second shared deadline. Above 0.62 load per CPU, only a fresh storage-recovery
lease can admit local compression, up to the existing 0.85 limit. That lane now
has a 240-second total budget, four files and 60 seconds per compactor, one worker,
and time reserved for the final reserve check. The separate quick pressure lane
remains 90 seconds with four files and 25 seconds per compactor. Neither bounded
lane inherits database rebuilding, external offload or current-day compaction.

Admission uses the larger of one- and five-minute load. Between steps, recovery
rechecks the load, maintenance hold and any required lease. After 60 seconds it
renews the memory observation. A low-load pass cannot widen itself mid-run when
load increases. A bounded compactor is deferred when its complete time window
cannot fit; previously running children keep their owner's verification and
process-group deadline controls.

`soak_storage_recovery_latest.json` separates pressure, trigger and recovery-target
shortfalls. `recovery_effectiveness` reports measured progress, failed/deferred
owners, missing reclamation measurements and completed passes with no measured
reclamation. Per-owner retry timestamps preserve both cooldowns and failure circuits.
An empty pass backs off, not loops immediately or clears the capacity blocker.
No eligible files in one pass does not prove that the entire disk was inventoried.
Occupied active database pages are not disposable free space; additional capacity
or a writer-owned retention/migration plan may still be required.

## Archive Failures

Archive pruning isolates SQLite failures by file, preserves a failed file, and
continues processing healthy archives. Its JSON result includes per-file
`archive_pruning.errors` and returns a nonzero status if archive pruning or cold
export fails. Successful hot-row movement remains visible in the same result.
An error report is not a corruption repair or proof that the affected archive is
recoverable. Keep failed archives available for a separate integrity investigation.

Legacy local archives and external archives with the same date can contain
different records. A storage relocation must use a distinct destination, preserve
the original lookup path, verify every copied file, and record a restore receipt
before releasing the local copy. Never merge those directories by filename alone.

## September 7, 2026 Recovery

The primary local database was compacted from 58.025 GiB to 12.111 GiB, reclaiming
45.914 GiB. SQLite quick-check passed before and after. The evidence tables kept
407,501 JSONL rows, 162,401 JSON-file rows, 439 number snapshots, and 12 shard merge
state rows. The recovery receipt is
`governance/health/storage_recovery_20260907.json`.

Six stable legacy archives were copied and SHA-256 verified under
`/Volumes/BOT_LOGS/schwab_trading_bot/cold_archive/local_primary_legacy_20260907`.
Their original file paths remain available through symlinks, releasing another
23.674 GiB locally. The actively pruned August 28 archive remains local; an earlier
copied snapshot is preserved externally. No divergent date-matched archives were
overwritten.

The relocation receipt is `governance/health/archive_rehome_20260907.json`, with
status `routed_verified_local_source_released`. Total verified local reclamation
was 69.588 GiB. At the final reserve check, the internal disk had about 131 GiB
free and BOT_LOGS had about 155 GiB free. The local reserve guard reported ready,
with no pressure or hard blockers. Intermediate receipts retain a recovery copy
and must not be treated as a completed cleanup.
