# Tiered Ingestion Storage

The storage plane uses one read-only lifecycle planner to keep continuous market-data intake from turning file count, rewrite work, or retained history into hot-path pressure. The planner is owned by `core/tiered_ingestion_lifecycle.py`, configured by `config/tiered_ingestion_lifecycle_v1.json`, and published inside `governance/health/storage_tier_policy_latest.json`.

## Lifecycle

1. Active and current-day decision tails remain on the hot path.
2. Sealed, noncritical small files are grouped by family and month into bounded compaction waves.
3. Completed larger segments become warm or cold movement candidates according to age.
4. Stateful SQLite files stay on their dedicated checkpoint, vacuum, incremental-vacuum, or verified-mirror path.
5. Stale-stage files stay under the retention owner rather than generic compaction.
6. Snapshot retention, zero references, an orphan grace period, a verified cold copy, matching SHA-256, and a restore probe are all required before a file can even enter retirement review.

The lifecycle planner never moves, rewrites, throttles, or deletes data. It cannot submit orders or change live-money authority. Existing storage owners execute only their already-bounded operations, and source retirement remains separately gated.

## Backpressure

`work_state` separates maintenance demand from control health:

- `steady`: normal intake and background maintenance.
- `compaction_catchup`: critical intake stays open while sealed-segment compaction receives priority.
- `intake_throttle_advisory`: free-space or file-count pressure warrants reducing noncritical batch admission. The planner itself has no throttle authority.

This prevents a healthy controller with queued work from being mislabeled as broken. Every compaction wave has a byte ceiling and an input ceiling so maintenance cannot consume the whole machine.

## External Patterns

- [Apache Kafka tiered storage](https://kafka.apache.org/39/operations/tiered-storage/) separates a locally retained active tail from completed segments eligible for remote storage.
- [Apache Iceberg maintenance](https://iceberg.apache.org/docs/latest/maintenance/) uses atomic snapshots, bounded file rewrites, retained snapshots, and safety intervals for orphan cleanup.
- [RocksDB compaction](https://github.com/facebook/rocksdb/wiki/Compaction) manages read, write, and space amplification through bounded compaction strategies.
- [ClickHouse TTL](https://clickhouse.com/docs/concepts/features/operations/delete/ttl) coordinates background merges with time-based movement across storage volumes.

These are design influences, not claims that the bot runs those systems. The local implementation is stdlib-only and preserves the existing manifest, retention, restore, and authority boundaries.
