# New Primary Data SSD

## Prepared Scope

The incoming 2 TB SSD is planned as the platform's **primary bulk-data drive**,
not just an overflow archive. Nothing is armed or migrated before it arrives.
The suggested volume name is `BOT_DATA`; a name is never sufficient identity.

| Placement | Planned Contents |
| --- | --- |
| New SSD: `schwab_trading_bot/data/` | Large SQL shards, approved active database families, analytics, Parquet and training datasets |
| New SSD: ordinary platform subdirectories | Logs, decisions, explanations, exports, reports and model artifacts |
| New SSD: `schwab_trading_bot/cold_archive/` | Eligible closed/compressed history and retained cold archives, using existing manifests and retention rules |
| Internal disk | Source/Git, Python environments, credentials/Keychain, current halt/maintenance flags, small control/health records and a bounded recovery buffer |
| Existing BOT_LOGS | Preserve its current contents and identity; consider an independent backup role only after separate capacity and verification checks |
| VIDEO | Untouched and not inspected, enumerated or used for platform migration |

These are planned placements, not new environment overrides. Dataset family
names in the preflight are categories rather than an exhaustive path manifest.
The last tier report is summarized with its original observation time; an old
report is explicitly stale and never proves current copy size or reclaimable
space. Build a fresh, exact manifest before moving anything. Archives already
linked to another device are separate migration work, not files to follow and
copy automatically.

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
