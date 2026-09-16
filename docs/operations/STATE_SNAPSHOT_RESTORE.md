# Selected-File Restore Admission

Owner: `scripts/daily_state_snapshot_drill.py`, with measured capacity and restore
helpers in `scripts/ops/state_snapshot_capacity.py`. Persistent policy lives in
`config/state_snapshot_drill_v1.json`; daily verification uses that same policy
and a bounded outer command timeout.

```bash
./scripts/ops/opsctl.sh state-snapshot-drill --capacity-only --json
./scripts/ops/opsctl.sh state-snapshot-drill --json
```

The capacity-only command inspects the five declared targets and output route. It
does not copy data or publish readiness. The real command retains the shared
storage-maintenance lock, support pause, operator/maintenance holds, a 256 MiB
resident-memory ceiling, and a 1,800-second total work deadline. One worker is
cooperatively paced to 25% of one CPU core. This is not OS-enforced affinity.

## Capacity Contract

For SQLite, the planner measures page size, page count, and freelist pages through
a read-only connection. Its enforced output ceiling is occupied pages plus 64 MiB,
bounded by the configured maximum; it is not an exact size prediction. New data,
fragmentation, timeouts, or a cap overrun may still cause a safe failure. Plain-file
ceilings use observed sizes, and changing sources fail verification.

The budget includes all per-target snapshot ceilings, gzip scratch ceilings, and
restore storage. Same-volume APFS clones require a successful clone/write-isolation
probe and never fall back to physical copying without a physical-copy reservation.
The configured local reserve remains intact. A distinct archive filesystem keeps
at least 64 GiB of its own reserve; the Mac's separate target is not charged to it.
Actual free space is checked during work. Other noncooperating writers can consume
capacity, so a successful preflight is not a lasting reservation.

## Verification And Scope

[SQLite VACUUM INTO](https://www.sqlite.org/lang_vacuum.html#vacuum_with_an_into_clause)
creates a consistent, compact logical snapshot without rewriting the live source.
Committed WAL data is visible. Implicit row IDs may change, so the receipt explicitly
hashes the logical snapshot, not raw source-file bytes. It does not establish one
atomic snapshot across all five files.

The restored clone must have a distinct inode, match the full snapshot SHA-256, and
pass SQLite `quick_check` where applicable. The retained gzip is then fully decoded
and compared with the verified snapshot's hash and byte count. The archive hash and
receipt are flushed before removing only that run's verified temporary copies.
Failed or unverified copies are preserved. Retention runs only after a verified
replacement and skips failed generations.

Each archive now has a 4 GiB ceiling (or the smaller bounded raw-size estimate),
included in capacity admission. A 2 GiB cap proved too small for the compact
database in the September 11 drill. When exactly one archive-cap failure remains,
an explicitly approved `state-snapshot-drill --operator-approved-recovery
--resume-run ABSOLUTE_OWNED_RUN_DIRECTORY --json` can finish from retained copies.
It reserves the entire new archive budget above the unchanged reserve, rechecks
all previous archives and the distinct SQLite copies, writes a new no-clobber
archive, and preserves the failed partial archive. Only fully reverified temporary
copies are released. The original snapshot timestamp remains unchanged; resealing
does not create newer source data. The old manifest is retained beside the new
proof. This resume path is never selected by scheduled jobs.

These clones share physical extents on BOT_LOGS. They test selected-file restore
integrity, not independent-media disaster recovery, full-platform recovery, or RTO.
Every checked target must verify and both latest publications must succeed before
daily verification or storage resilience accepts the result. Metadata-only
observations, partial results, future/stale receipts, and capacity plans cannot
earn readiness. An initial support-pause deferral is written to a separate attempt
receipt without replacing prior restore evidence.

Failed replacement drills also preserve any complete prior latest proof and publish
the failure separately in `state_snapshot_drill_attempt_latest.json`. If an older
version overwrote that receipt, `state-snapshot-drill --recover-latest-verified
--json` rechecks the retained compressed archive hashes against the newest complete,
unexpired owned manifest with the same requested targets. Discovery is capped at
200 entries and hashing at 32 GiB/180 seconds. The original restore timestamp remains
unchanged; the separate republication timestamp is not source freshness or a new
restore. Original manifests are untouched. Archive digest mismatch, missing data,
resource withdrawal, and invalid proof fail closed.

Recovery uses the platform's expiry-aware health maintenance hold. Merely retaining
an expired hold file no longer blocks work; active, unreadable, and legacy root-level
holds, environment holds, and operator stops still block it.

The historical backfill still requires fresh storage, preparation, and resource
admission after this drill. No command here trains models, promotes candidates,
changes live-money locks, contacts a broker, or grants order authority.

## Explicit Bounded Recovery

Only after explicit operator approval, either owner accepts
`--operator-approved-recovery`. This is a one-invocation exception to the support
pause, not a persistent runtime setting. The worker and runtime pause owner share
`scripts/ops/approved_storage_recovery.py`: evidence must be no older than 120
seconds, memory must be green, thermal warnings must be absent, the fluidity band
must not be strained/protect, foreground and system CPU must each remain below
the owner's 90% pressure threshold, and existing refresh resource admission
must pass. Unknown, future, or stale observations fail closed. Operator/maintenance
holds, disk reserves, memory limits, and the 30-minute duration remain enforced.
The process exemption requires the exact script, explicit flag, bounded elapsed
time, and observed CPU limit; it does not exempt unrelated processes.
Both healthy `observe` and admitted `soft_cap` profiles are eligible. A transient
hard-resource withdrawal stops copy/hash/SQLite/compression work and permits at
most a 120-second idle wait, requiring ten seconds of stable fresh admission before
continuing. The original worker deadline is never reset; operator holds and disk
reserve breaches still abort. Longer withdrawals preserve the copies for a later
explicit attempt.

```bash
./scripts/ops/opsctl.sh raw-training-compaction --operator-approved-recovery --apply \
  --scan-root /Volumes/BOT_LOGS/schwab_trading_bot --max-files 128 --max-gb 40 \
  --jumbo-gb 0 --compaction-workers 1 --min-age-hours 24 --write-history --json
./scripts/ops/opsctl.sh state-snapshot-drill --operator-approved-recovery --json
```

Approved archive recovery restricts changes to the separate scan volume's
`local_fallback_storage/governance` history. Current-day and active-latest files
remain excluded. Small files run first to build scratch capacity; open-file probes,
stable source identity, full gzip restore hashes, no-clobber publication, and flushes
precede raw-copy release. Each file must reserve its worst-case compression scratch
above the 64 GiB floor. The pass stops at 84 GiB free, the selection budget, a hard
resource stop, or its deadline. Its per-file journal is
`governance/health/raw_training_recovery_progress_latest.json`; final evidence is
also retained in the compaction history. It takes the storage-maintenance lock and
does not run the usual post-compaction self-model refresh cascade.
