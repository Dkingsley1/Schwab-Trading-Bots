# Storage And Ingestion Contract

This document connects the existing owners. It is not a new storage policy,
scheduler, collector, or permission to delete data. Definitions, observed state,
verified durability, and trading readiness are separate claims.

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
