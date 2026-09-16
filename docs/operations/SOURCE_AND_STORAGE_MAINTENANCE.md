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

## Database Space

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
