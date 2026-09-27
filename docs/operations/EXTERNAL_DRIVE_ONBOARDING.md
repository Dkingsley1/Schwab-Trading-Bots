# New Primary Data SSD

## Prepared Scope

The 2 TB Extreme SSD is planned as the platform's **primary bulk-data drive**,
not just an overflow archive. Verified staged copies do not activate runtime
routes or authorize deletion. A volume name is never sufficient identity.

| Placement | Planned Contents |
| --- | --- |
| New SSD: `schwab_trading_bot/data/` | Large SQL shards, approved active database families, analytics, Parquet and training datasets |
| New SSD: ordinary platform subdirectories | Logs, decisions, explanations, exports, reports and model artifacts |
| New SSD: `schwab_trading_bot/cold_archive/` | Eligible closed/compressed history and retained cold archives, using existing manifests and retention rules |
| Internal disk | Source/Git, Python environments, credentials/Keychain, current halt/maintenance flags, small control/health records and a bounded recovery buffer |
| Existing BOT_LOGS and LaCie | Selected independent-device backups and rollback copies, subject to capacity and restore verification; do not duplicate every replaceable dataset |
| VIDEO | User media; excluded from routine platform storage and cleanup. Prior copy-only exceptions do not authorize source deletion |

These are planned placements, not new environment overrides. Dataset family
names in the preflight are categories rather than an exhaustive path manifest.
The last tier report is summarized with its original observation time; an old
report is explicitly stale and never proves current copy size or reclaimable
space. Build a fresh, exact manifest before moving anything. Archives already
linked to another device are separate migration work, not files to follow and
copy automatically.

BOT_LOGS and VIDEO are partitions of the same physical LaCie device, not two
independent backups. Confirm physical-device identities before counting backup
copies. Prioritize irreplaceable raw data, decision/execution evidence and
current database recovery points; derived datasets and caches can be rebuilt.
Keep at least 200 GiB free on Extreme SSD (or the larger configured reserve),
and retain the existing internal and backup-volume reserves. Broad log, model
and dataset moves remain later, separately verified steps; the first restricted
runtime handoff covers only the SQLite paths below.

## Restricted SQLite Route

The opt-in `BOT_STORAGE_ROUTE_PROFILE=sqlite_primary` profile is implemented by
`core/sqlite_primary_storage.py`. It is not enabled by adding these source files.
The target override writer supports this profile and shell-quotes spaced volume
names. Bind the exact selected APFS volume UUID, mount and platform root using
the existing storage-target owner; do not use a blanket external switch.

Only `data/sql_link_shards` and `data/{jsonl_link,bot_channel_queue,snapshot_context}.sqlite3`
plus their WAL/SHM routes are adopted. Governance, health, maintenance, halt and
execution-lane controls stay internal. Logs, exports, models and other datasets
keep their existing routes. Current staged `migration_snapshots` are not active
database destinations, and changed sources need fresh quiet-point snapshots.

The router and failback verifier observe this profile without adopting it.
Writer admission and queue/common SQLite opens reject unavailable active routes;
explicit standby writes are forbidden. Read-only standby inspection still reads
the requested standby. An unavailable SSD defers the writer and watchdog restart
attempts without readiness credit or automatic fallback to stale local data.
Legacy broad switching, disaster recovery and source-pruning owners are blocked
for this profile rather than allowed to rearrange or retire its files.

The explicit native handoff owner is:

```sh
.venv314/bin/python scripts/ops/storage_failback_sync.py --apply --sqlite-primary-receipt /absolute/path/to/reviewed-receipt.json
```

This is a maintenance-only command, not a ready-to-run migration shortcut.
The receipt must be under the project's physical `governance/storage_recovery`
directory. It uses schema version 1, purpose `sqlite_primary_cutover`, exact
`source_root`, `target_root` and `volume_uuid`, and a complete `files` list.
Each row binds `relative`, quiesced `source_identity` (device, inode, size,
mtime-ns, ctime-ns), target `sha256`, and SQLite `quick_check: "ok"` evidence.
Source root is the project's `local_fallback_storage/data`; destination is the
configured target's `data`. Missing/extra shard entries, aliases (including
unreconciled historical lookup links), nonempty journals, unknown open-handle
results, changed files or unrecognized existing routes block the handoff.

An authorized native maintenance hold and live switch OFF are required throughout.
The owner rechecks full destination hashes, namespace and file identities under
a 900-second budget, and never copies or deletes payloads. A durable internal
transaction journal records original links and prepared/committed/rollback
phases. Caught failures restore only this transaction's links; concurrent foreign
changes remain visible as rollback conflicts. Process death can leave a prepared
journal and partial routes: leave maintenance engaged and review the journal;
there is no automatic crash replay or source retirement.

Storage owners read the persisted target contract directly, including when a
LaunchAgent has an older environment. That selection is context-local, does not
modify `os.environ`, and cannot authorize a route change. Malformed or aliased
configuration fails closed. The scheduled reserve guard also uses opsctl so its
other managed settings remain current.

Route reports retain the common `route_verification` envelope for ingestion
health, explicitly scoped to declared SQLite routes. A ready route requires the
validated volume and every managed link; it is not integrity, ingestion, warm
standby, or trading-readiness evidence. Missing or conflicting links remain
blocked.

For an operator-reviewed reversal of an already committed primary, the explicit
`core.sqlite_primary_recovery.restore_committed_routes` owner requires the old
committed journal, original handoff, documented retired standbys and reconciled
history. It runs fresh full hashes over the exact primary inventory under a
2,400-second budget by default. A supervised call may explicitly select the
7,200-second large-inventory verification budget; routine jobs do not inherit it,
and an expired or lost maintenance hold still prevents publication. Only an exact
match to the committed, integrity-checked
snapshot may reuse its structural proof; changed bytes require a fresh SQLite
integrity check. The receipt distinguishes both bases. Under the owned hold, an
orphan SHM left by a closed read-only probe may be retired by SQLite only when no
handles or nonempty WAL/rollback journal exist. Database identity must remain
unchanged, and a second quiet-point check rejects recurring interference. This
does not admit pending transactions or authorize manual sidecar deletion.
It rechecks quiet identities and
publishes through the same journaled link transaction. It never adopts foreign
links, discards conflicting rows, deletes payloads, or automatically replays a
crashed transaction. Normal ingestion must be tested again afterward.

After an approved handoff, use `./scripts/ops/opsctl.sh storage-route-verify`
for route observation, then separately test actual I/O, restart/reconnect,
ingestion continuity and restoration. Metadata readiness is not integrity,
independent backup or safe-deletion proof. No live cutover or retirement is
certified by unit tests.

### Independently Backed-Up Standby Retirement

The native `local-sql-shard-standby-prune` owner has a separate explicit
`--independent-backup-receipt PATH` mode for this profile. The default legacy
pruner still cannot delete anything under `sqlite_primary`. Review and pass an
owned `governance/storage_recovery` receipt, first without `--apply`, under an
authorized maintenance hold with trading OFF. No scheduler supplies this option.

This mode is limited to two named inactive shard originals. It requires the
complete handoff receipt, actual routed queue write/read/ack proof, unchanged
quiesced originals, and Zstandard recovery copies on a UUID-verified different
physical drive. It rehashes each original and the entire decompressed backup
within 1,800 seconds, verifies active routes, journals the selected custody,
then rechecks the hold, source identities, independent drive and idle journals
before each unlink. It preserves active databases, recovery copies, raw history
and all other originals. Protected VIDEO paths are rejected before inspection.
A partial retirement journal requires review; no automatic retry or blanket
deletion authority is granted. Measure actual free space afterward: logical
bytes retired are not necessarily physical bytes recovered.

## Read-Only Preparation Command

```sh
./scripts/ops/opsctl.sh external-drive-preflight --json
```

Before arrival this reports `awaiting_drive_selection`. It does not scan attached
volumes or create a directory below `/Volumes`.

After connecting and explicitly selecting the new volume:

```sh
./scripts/ops/opsctl.sh external-drive-preflight --mount "/Volumes/BOT_DATA" --json
```

Review its reported UUID against the selected new drive, then repeat with
`--expected-uuid` and that exact value. The targeted metadata probe rejects old
BOT_LOGS identity, protected paths, symlink aliases, unmounted directories,
non-external devices, read-only volumes, non-APFS formats, mismatched UUIDs,
unexpectedly small capacity and inadequate reserve. The preparation target
reserves the larger of 150 GiB or 10% of capacity, plus 1 GiB of probe space;
these are proposed destination budgets, not changes to live storage guards.

Even a `metadata_preflight_passed` result keeps `activation_ready=false`.
No write test, speed test, migration, format, restart, source acceptance or
trading authority is implied. The tool has no apply switch and writes no report
or configuration. Disk Utility's format/erase action remains an explicit,
separate operator decision after device identification and data review.

For Mac-only primary data, the plan requires APFS. Apple documents APFS as
optimized for SSD storage and supported on external direct-attached devices:
[Apple filesystem guidance](https://support.apple.com/guide/disk-utility/file-system-formats-dsku19ed921c/mac).
An encrypted volume must be unlocked before use; the platform must not store
its unlock password in environment files. Filesystem choice does not establish
drive reliability or independent backup coverage.

## Supervised Handoff

1. Confirm the exact new device and UUID. Check its connection and manufacturer
   firmware guidance for that actual model/serial. Do not erase, rename or
   repartition any existing drive as part of preparation.
2. On the selected SSD, approve a small disposable test area. Verify write,
   fsync, complete read-back hash, atomic rename and remount/reconnect identity.
   Measure sustained I/O and a representative SQLite workload; advertised
   sequential throughput alone is not database latency evidence.
3. Build a bounded per-file inventory and exact route map. Separate active DB
   families from closed archives. Budget destination staging, retained copies,
   growth and reserves. Inventory is not a move authorization.
4. Copy closed files first under the existing storage owners and quiet/resource
   controls. Full hashes, compressed restoration where applicable and durable
   receipts precede any source retirement. Preserve cold-history lookup paths.
5. For active databases, obtain a maintenance window, stop writers/collectors
   and reconcile checkpoints. Use each database owner's consistent backup or
   quiesced database/WAL procedure. Verify restored rows, integrity and source
   identity. Do not copy a running SQLite file independently of its WAL state.
6. Review exact target bindings before changing runtime routing. Reuse
   `core/storage_target_override.py` and `core/storage_router.py`; **do not run
   a blanket external switch** before checking the route map. Current broad
   defaults include governance, while the new plan keeps active control records
   internal. Any missing narrow-route support must be implemented and tested
   before cutover, not worked around with ad hoc symlinks.
7. Resume through native owners and verify actual reads/writes, newest ingestion,
   single-writer leases, broker-read-only/paper operation and source checkpoints.
   Test detach/reconnect safely before relying on the drive for unattended work.
   An absent drive must defer affected writers, never silently create an
   internal look-alike destination or start a restart loop.
8. Keep verified rollback copies until the new routes and restore tests pass.
   Release duplicates only through their owning verified-retirement controls.
   Do not relax retention, freshness, reserve or trading gates to manufacture
   migration or health completion.

## Existing Owners

Storage-target overrides are shell-sourced. The canonical writer quotes every
value, including mount and volume names containing spaces; do not hand-write an
unquoted target assignment. Legacy simple BOT_LOGS assignments remain unchanged.

The stateful storage repair blocks equal-size file collisions rather than
classifying them as duplicates. Both versions remain at their existing paths at
the collision; earlier actions in that repair pass are not certified by the
blocked result. Resolve custody through the verified-retirement owner before
retrying. File size alone never establishes equal contents or retirement rights.

- `storage_failback_sync.py --verify-only`: read-only route observation.
- `storage_switch_orchestrator.py`: supervised stop/switch/restart coordination,
  only after the selected target and route plan are reviewed.
- `storage_sqlite_hot_route.py`: owner-controlled hot/cold SQLite lifecycle,
  not permission to copy active shards manually.
- `deep_cold_storage_layer.py`: closed-history offload, full hashes, durable
  receipts, no-clobber publication and preserved lookup paths.
- `soak_self_healing_control.py` and existing retention jobs: ongoing bounded
  maintenance once the approved target is configured. No new scheduler is added.

Putting live data and archives on the same SSD saves internal space but does
**not** make the archive an independent backup. The old BOT_LOGS partition may
provide a separate-device copy after verification; its sibling VIDEO partition
is outside this plan. No backup, drive speed, migration completion or restored
headroom is certified until measured.
