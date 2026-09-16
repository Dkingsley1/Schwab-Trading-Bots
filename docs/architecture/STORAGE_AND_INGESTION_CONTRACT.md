# Storage And Ingestion Contract

## Pressure Recovery And Preserved History

The self-healing launchd owner runs `soak-self-heal --storage-recovery-only`
before the heavy-maintenance quiet-hours gate. It measures current local free
space against the existing 64 GiB pressure threshold, shares the self-healing
lock and retry state, and checks fresh memory, load, and maintenance ownership.
Only bounded telemetry compression, verified cold compression, and verified
historical offload are permitted. Cache rebuilds, training, route migration,
candidate acceptance, and execution authority are excluded. The resulting
`soak_storage_recovery_latest.json` certifies pressure relief only, not the
125 GiB unattended reserve or complete recovery of every storage surface.

Successful compactor exit status is not recovery progress. The native owner
reads each supported compactor's reclaimed-byte field and publishes that
measurement separately from net local headroom change and target shortfall.
Empty or unmeasured successful passes back off per owner exponentially, capped
at one hour. Measured positive reclamation resets this counter; no-op results
do not open failure circuits, release storage pauses, or bypass admission.

The shared cold/lifecycle compression guard has a bounded disk-recovery CPU
allowance below 125 GiB local free space on hosts with at least eight logical
CPUs. It requires ready producer evidence no older than 90 seconds, clear
thermal/memory admission, no creative session or cooldown, no protective or
support-pause recommendation, and host saturation at most 70. Foreground CPU
must remain below 150%, system CPU below 200%, and their sum below 300% (100%
means one core); five-minute load may not exceed 0.85 per logical CPU. Otherwise
the existing 90% per-category and 0.62 normalized-load limits apply. The worker
stays paced to 25% of one core with second-scale admission checks, a 256 MiB
resident-memory limit, unchanged maintenance/operator holds, and at least
16 GiB emergency reserve plus worst-case scratch space. This allowance is not
training, SQL-writer, live-lane, retention, or reserve-readiness permission.

A fresh workload-specific recovery lease may also admit the same paced
compression worker when an aggregate protect band comes from other workloads.
Underlying thermal, memory, CPU and load checks and explicit pauses must still
clear; the adaptive batch has a 240-second ceiling. See
[Workload Admission](WORKLOAD_ADMISSION.md). Neither this lease nor a successful
compression batch bypasses the writer's independent 64 GiB pressure threshold.

`data_collection_storage_guard` separates fallback files already retained under
canonical cold/quarantine roots from active-route duplicate candidates. The
archived inventory remains visible with reconciliation unverified and deletion
disallowed. A canonical sibling merely existing cannot authorize duplicate
removal: full restored bytes and SHA-256, stable file identities, idle handles,
synced canonical data, and a durable restore receipt are required. Partial,
temporary, failed-SQLite, and failover-backup filenames do not establish expiry
or recoverability and are preserved for their owning recovery/retention process.

This document connects the existing owners. It is not a new storage policy,
scheduler, collector, or permission to delete data. Definitions, observed state,
verified durability, and trading readiness are separate claims.

The walk-forward seed queue is replaced atomically only when its serialized
contents differ. A full-byte comparison uses fixed-size buffers and checks the
source identity before and after reading; unchanged contents retain the inode
and mtime so repeated planning does not reset ingestion identity or source age.
The producer still fsyncs changed content before replacement. This optimization
neither moves a SQL checkpoint nor certifies coverage or source completeness.

## Database Direction

The adopted direction retains **SQLite and DuckDB/Parquet** for local operations
and analytics. PostgreSQL is an evaluation candidate, not an installed or
activated backend. Dedicated NoSQL integration, including Redis, is deferred.

| Technology | Role | Decision |
| --- | --- | --- |
| SQLite | Local operational state, durable queues and bounded single-writer workloads | Keep existing owners, transactions, WAL visibility and storage admission. |
| DuckDB/Parquet | Analytical summaries and verified historical columnar data | Keep analytical work separate from authoritative operational writes. Preserve atomic mirror generations and exact archive restoration. |
| PostgreSQL | Possible shared transactional state for concurrent writers or multiple hosts | Evaluate next only where measured requirements justify it; no automatic migration or new service. |
| Redis / NoSQL | Possible cache for a demonstrated read bottleneck | Deferred; dependency presence does not mean runtime integration. |

PostgreSQL's indexed JSONB can support flexible metadata, but does not preserve
the exact original JSON text. Hash-bound source bytes must remain independently
preserved where replay requires them. See the official
[PostgreSQL JSON documentation](https://www.postgresql.org/docs/current/datatype-json.html).

Redis caching adds freshness and invalidation responsibilities. Any future
cache must have bounded memory, source timestamps, expiry, invalidation and
authoritative-store fallback. It cannot be the sole source of broker truth,
order state or safety evidence. See the official
[Redis caching documentation](https://redis.io/docs/latest/develop/clients/client-side-caching/).

Evaluation starts from existing query timings, lock/busy observations, durable
ingestion lag, resource receipts and actual host-access requirements. Missing
measurements remain unknown; dependency inventory is not a performance test.
A future pilot must be bounded and non-authoritative, demonstrate a benefit
against existing service-level requirements, and pass equivalence, idempotency,
outage, restart, backup/restore and rollback tests before any owner switches.

While capacity is constrained, do not install database services, produce full
migration copies or run unadmitted history scans. Use existing protection and
verified compression/retention owners first. This direction changes no reserve,
retention floor, routing permission, qualification threshold or trading authority.

## Inspect The Definition

```sh
./scripts/ops/opsctl.sh ingestion-storage-control --definitions-only --json
```

This prints current policy definitions and bounded metadata observations without
writing a health artifact, opening databases, scanning directories, fetching data,
or changing routes. It does not run the full health calculation. An explicit
`--out-file` is not used in this mode. Exit 2 means malformed/missing definitions
or a route inspection finding; exit 0 is not a storage-integrity or runtime-health
certificate. Absent optional paths and absent SQLite sidecars can be normal.

The normal `ingestion-storage-control --json` report includes the same
`data_plane_definition` section, so existing report refreshes also refresh this
view. No additional scheduled process is required. Policy fingerprints identify
the definitions used; they are not cryptographic attestations of runtime health.

## Verify New Ingestion

```sh
./scripts/ops/opsctl.sh ingestion-storage-control --verify-new-ingestion \
  --since 2026-09-15T16:16:52Z --until 2026-09-15T16:39:28Z --json
```

Use the desired timezone-aware, whole-second interval. Start is inclusive and
end is exclusive; omitted end uses the current whole UTC second. This measures
writer-owned UTC `ingested_at`, not market/source event time. The command opens
canonical and configured local-fallback primary/shard databases read-only,
deduplicates aliases by device/inode, and requires the existing ingestion-time
index. Each database uses its own read transaction, including committed WAL.
No immutable-mode shortcut, journal-mode change, cursor advance, source write,
route change or full-table fallback is permitted.

For `jsonl_records` and `json_file_records`, the receipt reports committed rows,
checked rows and bytes, stored payload SHA-1 mismatches, invalid JSON, and checked
stream counts. This is stored-value consistency, not an independent source-byte
comparison or cryptographic authenticity proof. Primary and shard copies remain
separate counts. The source census retains its original timestamp and estimated
pending counts; per-shard producer counters describe only their latest pass,
not every pass in the audit interval.

Defaults are 90 seconds, 256 MiB of payload reads and 100,000 rows. Explicit
`--verification-max-seconds`, `--verification-max-payload-mib` and
`--verification-max-rows` allow at most 300 seconds, 1,024 MiB and 1,000,000 rows.
Each database also has a 15-second query deadline; payloads over 8 MiB remain
unchecked. Limits, missing indexes, malformed databases and inaccessible routes
produce incomplete evidence, never a sampled-success certificate. Split a large
window into contiguous smaller windows and retain every successful receipt.

The output is `governance/health/ingestion_verification_latest.json`, or an
explicit `--out-file`, separate from full ingestion/storage health. Exit 0 means
all discovered supported stored rows in this bounded scope passed; zero new rows
does not prove collection is running. Exit 2 means verification is incomplete or
found inconsistent data. Re-run to cover late commits: there is no cross-database
atomic snapshot. Unlisted/custom-table sinks, queue acknowledgment, source-to-SQL
completeness, merge equivalence and archive restoration require their independent
owners. The verification command does not start a new scheduler.

## Data Classes

| Class | Representation | Completion Evidence |
| --- | --- | --- |
| Source payload | Collector-specific snapshots and raw responses | Producer persistence, provenance and source validation |
| Append-only evidence | JSONL to `jsonl_records` | SQL commit and validated source cursor; rejected lines remain separate |
| Versioned JSON snapshot | JSON to `json_file_records` | `(source_rel, payload_sha1)` version, not a line cursor |
| Durable queue | Channel queue and owner acknowledgment | Queue-owner handoff plus destination commit |
| Analytical mirror | DuckDB/Parquet | Owner-verified analytical generation, not operational authority |
| Sealed history | Manifest-backed archive | Stable preserved bytes and verified restoration |

These definitions also appear in the existing `data_plane_definition` report.
Lane assignment still comes from the existing routing policy; organizing the
definition does not move files or reclassify pending data as complete.

## Route Vocabulary

| Term | Meaning | Not Evidence Of |
| --- | --- | --- |
| Configured intent | Target roots and preference in the calling process environment | Every daemon using that environment, or every file following that target |
| Logical path | Stable repository lookup path used by a consumer | The disk that holds its bytes |
| Observed route | Resolved symlink chain and metadata at inspection time | A writer lease, SQL integrity, or a recoverable backup |
| Active database | Database selected by its current writer owner | Disposable fallback or standby data |
| Verified standby | Separate copy with matching verification and required soak evidence | Permission to prune before its retirement guard approves |
| Sealed archive | Immutable input with verified manifest and restore evidence | Any file merely named `archive` or dated in the past |

`core/storage_router.py` owns route mutation. The bounded inspector follows each
listed path independently: a physical `data/` directory can contain a local
primary DB, external archive symlinks, and independently routed children.
Directory observations never certify all descendants. The report does not scan
all shards, archive members, arbitrary environment-overridden DB paths, or open
file descriptors. Use the writer and failback owners for those active-route
proofs. SQLite DB, WAL, and SHM observations are separate; do not move or delete
one member of an active database family independently.
Present sidecars that do not resolve beside their observed database are reported
as `sqlite_sidecar_route_mismatch`; the inspector never relocates them.

The inspector refuses the protected media volume before target metadata access,
including symlink aliases. Missing targets, loops, and access failures remain
explicit observations instead of being converted into healthy route labels.

## Storage Ownership

| Data Class | Logical Location | Writer / Lifecycle Owner | Routing Constraint |
| --- | --- | --- | --- |
| Primary SQL | `data/jsonl_link.sqlite3` | SQL shard manager | Active writer-owned state; switch only through coordinated handoff |
| SQL shards | `data/sql_link_shards/` | SQL shard manager | A shard checkpoint is not proof of primary merge |
| Channel queue | `data/bot_channel_queue.sqlite3` | Channel producers and SQL writer | Durable handoff state, not expendable cache |
| Shared context | `data/snapshot_context.sqlite3` | Snapshot context writer | Shared consumer snapshot; freshness still matters |
| Fetch/ingestion receipts | `governance/ops_data_plane.sqlite3` | `scripts/ops_data_plane.py` | Receipt persistence must be distinguished from source payload persistence |
| Analytical summaries | `data/analytics_mirror.duckdb` | `scripts/ops/sql_analytics_mirror.py`, called by the operations coordinator | Read both source summaries in one read-only SQLite transaction; atomically commit both DuckDB replacements. Failure preserves the prior mirror. Cache freshness is separate from authoritative ledger state. |
| Dispatch index | `governance/queues/ingestion_priority_queue.sqlite3` | Priority queue owner | Bounded selection from backlog reports, not a full ingestion ledger |
| Source context | `exports/external_context/`, `exports/external_feeds/`, `data/external_context/` | Collector definitions and source synchronizers | One shared producer per context route; consumers use source-backed lineage |
| Decision/evaluation evidence | `decisions/`, `decision_explanations/`, `governance/` | Producers, tier policy and retention owners | Active tails and current-day critical evidence stay protected |
| Archives/deep cold | `data/jsonl_link_archives/`, `data/deep_cold/`, configured cold roots | Hot retention and manifest-backed cold owners | Old date alone does not prove immutability; verify source stability |

After the September 7 recovery, primary SQL remained on active local storage;
six stable legacy archive files were individually rehomed to BOT_LOGS. This is an
intentional mixed layout, not evidence that the entire repository moved external.
See [SOURCE_AND_STORAGE_MAINTENANCE.md](../operations/SOURCE_AND_STORAGE_MAINTENANCE.md)
for recovery receipts and database reclamation details. Run the inspector for
current routes instead of treating that historical snapshot as live state.

## SQLite Runtime And Maintenance

`core/sqlite_runtime.py` owns shared connection settings. Resource producer and
source observations must be timezone-aware, no more than 120 seconds old, and
not future-dated; invalid or missing sensor evidence selects conservative
settings. Fresh adaptive safety holds can only tighten them. Under pressure,
environment overrides cannot restore large caches, memory temp storage, or mmap.
These settings are chosen on connection open; they do not retune existing handles.

Connection lock waits use the smaller of the caller's budget and the configured
busy timeout, including zero and subsecond budgets. Busy handling is installed
before journal setup. Writable defaults remain WAL/NORMAL; explicit durability
requirements remain caller-owned. A configured zero WAL autocheckpoint is now
applied literally and requires the existing writer-owned checkpoint lifecycle.
No reserve, writer, maintenance, or scratch-admission gate is bypassed.

Database and journal routes are checked before opening, including protected
aliases. Read-only URIs escape special filename characters and retain committed
WAL visibility without immutable mode. The primary recovery owner validates file
headers before SQLite can rebuild sidecars, then checks schema read-only before
writable initialization. This preflight is not a full integrity certificate;
read-only SQLite connections may still maintain their transient SHM index.
Integrity summaries use a cooperative SQLite VM deadline as well as bounded lock
waits, report timeout as unverified, and close handles on every exit. An OS-level
filesystem stall is not covered by a SQLite VM deadline.

Routine planner maintenance uses `PRAGMA optimize=0x10002` with an explicit
1,000-row analysis limit, inside the existing maintenance budget. Full `ANALYZE`
requires `SQLITE_ANALYZE_ENABLED=1`, fresh green resource evidence, and the
existing size/operator gates. Red or unavailable evidence disables planner work
and automatic VACUUM; explicit maintenance still requires its existing guards.

The native SQL writer launcher always passes `--once`, with or without shards;
launchd owns cadence and the lifecycle wrapper owns the deadline. A bounded
single-writer job requires terminal lifecycle receipts. Only a genuinely
persistent, zero-deadline writer can use fresh internal liveness without that
wrapper. Storage/maintenance deferrals do not certify fresh ingestion. These are
application-layer changes, not a SQLite binary upgrade or database migration.

## Paper Outcome Evidence

The paper-performance owner publishes `outcome_evidence_diagnostics` on its
existing schedule, including JSON-only refreshes. It distinguishes rows before
the candidate cutoff, missing or mismatched candidate identity, missing or
invalid post-cost fields, and active-cohort exclusions. It never reconstructs
candidate identity from time alone. The legacy `timestamp` alias is normalized
in memory, without rewriting original records or bypassing the scan watermark.

Unreadable files, invalid JSON/timestamps, and rejected routes remain explicit;
they make the source scan incomplete and block profitability evidence sufficiency.
Missing optional roots and unmaterialized legacy aliases are counted separately,
not treated as proof that historical records existed or were recovered. Discovery
does not follow child directory aliases and refuses protected targets before
metadata access. The read covers the declared report sources, not every archive
or queue in the platform.

Only fresh execution, storage, and profitability-control receipts contribute to
reported current holds. Unknown queue depth stays unknown. A no-outcomes hold
that requires outcomes is flagged for review, not automatically released.

The critical paper trade append now raises `paper_trade_evidence_persistence_failed`
when the writer reports failure. The in-memory/persisted paper book may already
contain the fill: use execution-owner failure receipts and reconciliation, never
retry the intent blindly. This closes a silent-failure path; it is not historical
data recovery or a transactional guarantee across the book and journal.

## Ingestion Boundaries

1. **Source declared.** `scripts/collector_contracts.py` supplies producer identity,
   payload and health paths, freshness limits, coverage requirements, degradation
   semantics, and the owning command. A catalog entry does not prove execution,
   entitlement, or successful collection.
2. **Transport fetched.** `core/collector_transport.py` bounds the request and
   response, records identity/digest/timestamps, and limits transient retries.
   Fetch watermarks and dead letters are best-effort telemetry and can be skipped
   or fail to persist. HTTP success does not establish SQL durability.
3. **Payload/event qualified.** Source validators own schema and field meaning.
   `qualify_transport_event` and `EventTimeGuard` separately evaluate source event
   time, observation time, duplicates, conflicting payload identities, future
   skew, and lateness. The guard is stateful in memory; its existence does not
   prove every legacy producer invokes it or that dedupe survives restart.
4. **Capability routed.** The capability router applies the decision family's
   required/optional coverage, freshness, quality, provenance, and failover rules.
   Derived features are not independent sources. Paper and live share route
   definitions but require different evidence floors.
5. **SQL checkpoint committed.** `scripts/link_jsonl_to_sql.py` commits SQL before
   emitting a checkpoint. Resume state includes source identity, inode, line and
   byte offset; source boundary validation is required after rotation or replay.
   The SQL uniqueness key is `(source_file, line_no)`, not a global exactly-once
   event identifier. A consumed-line checkpoint may include rejected lines:
   inspect inserted, invalid, oversize and receipt-write-failure counts.
   The data shard explicitly includes
   `data/jsonl_link_archives/cold_archive_compaction_manifest.jsonl`, which the
   native backlog census already discovers; other archive paths are not added.
   This shard does not merge to the primary cache. Existing governance allocator,
   archive, regime, research and risk receipts use their normal stream/filter
   rules for bounded priority selection, preserving specialized routes and
   exclusions. Indexing a receipt does not verify its referenced archive.
   Routine ordering reserves at most the second existing position within each
   lane for overdue pending work. A valid checkpoint's observed source mtime can
   supply service age even when the source continues appending; invalid/reset
   generations do not inherit that age. This is not measured event lateness or
   proof of freshness. The first-ranked file, separate lane quotas, one-file
   caps, byte limits and admission remain intact. API/ingress focus keeps an
   overdue path inside each already-selected multi-slot lane's existing cap.
   The census publishes `checkpoint_service_age_seconds` separately from source
   age and pending counts. API focus can use it for that existing tail slot;
   EOF, reset, invalid and journal-advanced checkpoints cannot contribute stale
   state age. An overdue first-ranked file does not prevent an even older source
   from receiving the second slot.
   `governance/health/jsonl_ingest_batch_journal*` and
   `governance/events/jsonl_ingest_batches_*` (including derived indexes) are
   recovery inputs, not payload sources. The writer filter rejects these owned
   families even under explicit focus, matching the existing census exclusion.
   Original journals, checkpoint recovery, ordinary receipts and health snapshots
   remain intact; this removes recursive ingestion work, not source backlog.
   Focused health-history routing follows normal shard filters: named fast-health
   snapshots stay there; other accepted health receipts retain governance ownership.
   Small overdue governance-owned receipts in deferred accounting may use the same
   single tail slot after material raw-live admission. Separate checkpoint service
   age may schedule this slot while the source appends; it is not event lateness.
   This cannot add a shard,
   replace fast-health sentinel scope, widen cold ingestion or activate on deferred
   debt alone.
   Loop-state channel files likewise retain their normal governance or
   crypto-governance owner rather than a runtime shard that rejects them. Their
   deferred accounting does not prevent use of that already-selected owner's
   existing tail slot after material admission. Actual runtime-channel routes,
   normal source filters and all work limits remain unchanged.
   The sequential writer pass lazily opens one ops receipt connection with the
   existing full SQLite quick check and reuses it across files. Receipts still
   commit at each file/checkpoint; the connection closes at pass end. Connection,
   write and commit failures remain counted and invalidate reuse for that pass,
   without reconnect storms or changed primary-source commit behavior. Dry runs
   do not open it. This does not change shared connection policy or skip integrity
   verification; it removes duplicate full scans inside one pass.
6. **Merge confirmed.** The shard manager owns applicable primary merges. Dispatch
   acknowledgments are operational state; they are not independent SQL proof.
   Reconcile them with committed merge progress. Cold-lane work can intentionally
   bypass a primary merge, so define completion for the selected destination.
7. **Archive verified.** The lifecycle planner proposes sealed segments and
   compaction groups. Actual offload/retirement owners require stable source
   fingerprints, size and digest checks, manifest/restore evidence, ownership,
   and the applicable soak/retention conditions before local release.

These stages are distinct evidence boundaries, not a claim that every producer
uses one mandatory linear pipeline. A route receipt, an HTTP watermark, a queue
acknowledgment, and a SQL commit are not interchangeable. No end-to-end exactly-once
guarantee or automatic live-execution authority is asserted.

## Lanes And Pressure

The machine-readable lane and family definitions come directly from
`config/sleeve_ingestion_routing_v2.json`, not a second copy of its thresholds.

| Lane | Priority / Temperature | Pressure And Failure Behavior |
| --- | --- | --- |
| Core | 100 / hot | Preserve decision-path collection, throttle optional context; fresh last-good only within its validity window, then collect-only/hold per family |
| Deferred | 60 / warm | Quota-limited, age-fair research work; defer without globally blocking healthy paper collection |
| Cold | 20 / cold | Pause before core/deferred work; retry through a maintenance window |

These are scheduling and lifecycle labels, not hard-coded disk assignments.
Backlog accounting also identifies support telemetry and stale-stage subsets;
do not add overlapping counters to a source-deduplicated total. The backpressure
guard and ingestion/storage control own measured queue truth; the storage
governor owns throttle overrides. A definition report never applies those values.

`config/tiered_ingestion_lifecycle_v1.json` currently plans 256 MiB segments,
bounded groups of 4-32 small inputs and at most 8 GiB per compaction wave, with
14-day cold candidates. Those are planner limits, not blanket deletion ages.
Planner free-space advice does not override local reserve, writer, memory,
scratch-capacity, or quiet-window gates. Effective reserve thresholds come from
`core/local_storage_reserve.py` plus the loaded runtime environment; consult
`local-storage-reserve-guard --json` instead of assuming library defaults apply.

## Failure And Recovery

The self-healing owner uses `BOT_LOCAL_STORAGE_PRESSURE_FREE_GB` (64 GiB in the
operating guard) to start storage recovery, not just the lower macOS swap-critical
threshold. SQL and paper pause boundaries, the 125/135 GiB unattended targets, and
memory/capacity admission remain unchanged. Recovery is serialized and subject to
the existing per-step cooldowns and retry circuit.

- Telemetry rotation preserves the canonical writer path. An open rotated segment
  is retained. Full gzip restore SHA-256, stable source identity, durable archive
  publication, and scratch admission precede raw-segment release.
- `cold-archive-compactor --filesystem-select-inactive --filesystem-compressor
  ditto` selects a bounded wave, not a complete archive census. The automatic
  wave selects at most four old 100 MiB to 2 GiB SQLite files and 8 GiB total,
  with a 600-second work deadline. Apply requires an owned maintenance hold and
  the actual SQL writer lock at publication, after the isolated copy is fully
  verified. Copying, compression and full-file hashing leave the hot writer free.
  Publication caps handoff at 60 seconds plus 30 seconds for final checks, with
  a 60-second hold-expiry cushion. Source mutation or failed handoff preserves
  the original, and failed owned-hold release cannot earn ready status.
  The built-in backend needs no optional package;
  explicit `afsctool` requests preflight both executables before any writer hold.
  Both backends act only on temporary
  copies. Original replacement requires full matching hashes, SQLite quick_check,
  no active file handles or pending journal data, and measured allocated-block
  savings. Native compression preserves SQLite bytes and paths; future writes
  may decompress a file, so hot databases are excluded. Failed compression is
  reported and leaves the original intact.
  The SQL manager checks maintenance requests during its inter-cycle wait, so
  an idle writer yields at its next poll rather than holding the lock through
  a long sleep. Active write batches still finish before handoff.
- Archive counts, export reads, integrity checks, and no-op retention probes use
  SQLite URI `mode=ro`. They do not set journal mode or open the database writable.
  Read-only probes still include committed WAL data; they do not assume immutable
  databases. Retention opens a retained archive for mutation only after finding
  actual expired rows. Verified compression savings are measured at publication,
  not permanent capacity credit: later legitimate writes may expand the file.
- `deep-cold-storage-layer --include-compressed-history` adds only dated gzip
  decisions/explanations/governance history older than 24 hours by default.
  An explicit `--closed-history-min-age-hours N` permits a one-time offload after
  at least one hour since modification; current/future/invalid dates and raw files
  remain excluded. Native recovery never supplies this override. This age is
  movement eligibility, not expiry or deletion authority. The verified copy must be
  durable before an atomic symlink replaces the local file. Critical record
  retention remains unchanged; this option is location movement, not expiry or
  deletion permission. Divergent existing archives are not overwritten.
- `--include-registry-backups` is included in the existing full native recovery
  pass, not the minute-scale compression-only pass. Only timestamp-named registry
  snapshots in `backups/` qualify; the newest per producer family stays local,
  and both filename timestamp and mtime must be at least 24 hours old. All bytes
  and original lookup paths survive verified offload under `deep_cold/registry_backups/`.
- Checkpoint reconciliation and pending-byte ordering reuse the SQL writer's
  cursor validation. Replaced inodes, truncation, rewound timestamps and offsets
  past EOF cannot retire pending work. Reconciliation rechecks source identity
  after bounded counting and rejects protected/unavailable routes before reads.
- Material raw-live focus reserves at most one existing source slot per already
  requested shard for a small tail older than 30 minutes. The normal material
  threshold still controls activation, a one-source cap is never displaced,
  and worker, source-count, byte and storage limits are not enlarged for tails.
- `BOT_DEEP_COLD_OFFLOAD_ROOT` independently selects closed-history offload;
  `BOT_SECOND_COLD_ROOT` retains its APFS compression/cache-export role. LaCie
  is configured only for closed cold archives, and linked history requires it
  to stay attached. Missing external mounts fail closed before directory creation.
  Where exclusive rename/hard links are unsupported, exclusive-create copy
  publication preserves destination scratch reserve and rechecks capacity while
  streaming. Full hashes, durable restore receipts and source stability are still
  mandatory before source replacement. Permission errors never trigger a fallback.
- The eject guard identifies BOT_LOGS by the configured UUID and performs targeted
  `diskutil info` discovery, not all-volume discovery or project-folder matching.
  Local storage pins veto automatic failback. Failed route certification stops
  the refresh/mutation cascade; it cannot be relabeled a successful transition.
  Drive availability and active hot routing remain distinct health facts.
- `sqlite-reclaim-control --db PATH --scratch-dir PATH` exposes the existing
  per-database reclamation owner. Source rewrite, shared-volume scratch, external
  reserve, memory, and writer gates are identical to scheduled reclamation.

| Condition | Required Response / Owner |
| --- | --- |
| External unavailable | Storage failover/recovery owners preserve hot collection and bounded deferred cold work; do not probe other user volumes |
| External returns | Coordinated verification and failback owner decide transitions; availability alone cannot retire active local databases |
| Backlog grows | Governor protects core, limits optional/deferred work, pauses cold work, and reopens through measured recovery conditions |
| SQLite has free pages | Guarded reclaim controller checks material reclaim, writer ownership, memory, rewrite and scratch reserves; deferred runs remain explicit |
| Archive changes during copy | Preserve source, invalidate release eligibility, and retry through its owner; no filename-based overwrite of divergent copies |
| Archive malformed | Preserve failed archive and surface per-file errors; healthy archive work can continue; this is not a corruption repair |
| Receipt missing or stale | Report unknown/incomplete evidence; never infer durable ingestion, route safety, or live readiness from a green transport result |

Any future collector or route change should identify its producer, logical and
physical destinations, schema/source timestamps, durable completion boundary,
replay identity, freshness/failure behavior, quota lane, retention owner, and
verification evidence in the existing owning contract before claiming readiness.
