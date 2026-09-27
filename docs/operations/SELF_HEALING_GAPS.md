# Self-Healing Gap Resolution

The September 21 report was checked against source and live observations on
September 23, 2026. Its two supplied copies were identical. These findings do
not certify a healthy platform, successful ingestion, profitability or trading
authority.

## Native Controls

```bash
./scripts/ops/opsctl.sh self-healing-gaps --json
./scripts/ops/opsctl.sh storage-route-verify --json
./scripts/ops/opsctl.sh emergency-storage-thin --json
```

The existing 15-minute accrual profile owns the census, read-only route check
and conditional emergency-thin `--apply`. There is no Codex automation. A
deferred repair is explicit, not a successful space-recovery claim. OFF blocks
the shared scheduled runner without rewriting producer evidence.

## Findings By Area

1. **Circular recovery:** the native emergency-thin owner requires a fresh
   disk-only compaction block, local free space strictly between 16 and 32 GiB,
   bounded raw backlog, fresh CPU/RAM/thermal measurements and no operator,
   maintenance or global halt. It rechecks while working. At most two old local
   JSONL files totaling 3 GiB are processed by one paced worker for 90 seconds,
   with two attempts per rolling hour. Crashes/failures consume that budget.
   Full restored gzip SHA-256, unchanged source identity and closed-handle
   checks precede raw removal. Current-day/latest, fallback, symlink and external
   sources are excluded. The audit lists reported owner/next commands; listing
   them is not proof that every referenced producer is circularity-free.
2. **Stale SQL evidence:** the SQL writer is storage-paused, not an obsolete
   producer to remove. The census lists every latest-artifact age, including SQL
   overlays, and alerts on overdue registered owners. It does not rewrite SQL
   timestamps or infer that the backlog drained. Actual writer recovery still
   requires storage admission.
3. **Route verification:** deferred failback previously omitted route coverage.
   Read-only physical observation now populates it without route mutation. The
   observed three routes had full coverage; this does not certify database
   integrity, absence of backlog or sufficient free space.
4. **Deep cold:** manifest-ready is not move-authorized. Most of the reported
   3 GiB was critical nearline retention data. A bounded LaCie preview found
   only about 0.026 GiB of eligible local relief. Retention-locked candidates
   were not moved/deleted, and no cold-copy receipt was invented. Native moving
   remains governed by the existing separate cold-storage policy.
5. **Capacity:** the September 23 verified compatibility-cache rebuild reclaimed
   62,761,041,920 bytes (58.45 GiB), taking internal free space from about 48 to
   106 GiB. All 2,032,704 cold rows passed full typed-row SHA-256 restoration
   checks on LaCie before atomic replacement and old-cache pruning. The new
   cache retains schema, indexes, sequences and merge metadata. The reserve
   owner cleared pressure and writer-pause conditions; the 125 GiB target
   still has roughly a 20 GiB deficit. Occupied authoritative SQL shards were
   not deleted. Restoration proof is
   `cache_rebuild_20260923T124725Z_verified.json` in the configured LaCie
   `cold_archive/sql_link_primary/autonomic_cache_rebuild` directory.
6. **Repair testing:** fixtures cover low-disk admission, backlog/extra-fault
   denial, protected/current sources, full restore verification, failed-attempt
   budgets, stop flags, stale evidence and no-mutation route refresh.
7. **Collector declarations:** the census classifies all declared collectors
   individually. Required collectors and explicit evidence contracts need
   runtime proof; optional context without that contract is advisory-only, not
   execution evidence. Definition completeness cannot set runtime conformance
   true. Capability materialization remains the runtime proof owner.
8. **Missing owners:** registered 22 existing artifact refresh commands.
   `broker_readiness_control` lacks a standalone refresh owner and now has an
   explicit missing-owner/overdue alert rather than a fabricated command.
   Registration is documentation, not permission to execute arbitrary repairs.
9. **Locks:** both named recovery locks use kernel `flock`. Process exit releases
   ownership even if the pathname remains. Deleting a held lock could permit
   concurrent writers; no age-based unlink was added. New power/thin controls
   use the same lifetime-lock contract and report busy states.
10. **Worker overrides:** the operator requested eight preprocessing workers.
    The selected count can be capped by the resource budget. Reports now show
    requested versus admitted workers and explicitly identify resource capping;
    no secret-bearing environment file was edited and no governor was disabled.

## Verified Self-Interference Fixes

September 23 follow-up: 49 retained BOT_LOGS archives were fully hash-verified
on LaCie before source links were published, releasing 3,281,612,962 bytes on
BOT_LOGS. Two closed local telemetry files were compressed with full gzip
restoration checks, saving 271,424,411 bytes internally. These amounts are
separate from the earlier cache rebuild. Active writes continue to consume
capacity; neither operation certifies the 125 GiB reserve target.

The restart logs identified redirected fallback directories as the immediate
launcher failure: `decision_explanations`, `decisions`, and `governance` under
the local fallback root were legacy BOT_LOGS aliases. The dedicated repair
preserves those aliases and their targets, then creates local directories.
It is singleton-owned through failback and runs in native accrual. Unknown,
protected, unavailable, or changing aliases remain blocked. The watchdog
checks pinned-local routes before another launch; accumulated restart history
is not erased to claim recovery.

At 14:22 UTC the native watchdog reported all three collectors running with
healthy heartbeats and no active restart storms. This is a point-in-time
recovery observation, not a reset of restart history or a platform all-clear.

Retention now continues independent cleanup after a failed main policy pass.
Stale repair uses one total deadline and cleans up timed-out descendants;
reported command strings cannot request arbitrary opsctl actions or arguments.
The census and freshness SLO use the canonical source-observation timestamp,
including invalid/future rejection. No age-only deletion of latest evidence
or protected history was added.

Cold SQLite compression accepts an alternate APFS scratch root, preserving
32 GiB local staging and 64 GiB external publication reserves and full raw-size
copy headroom. Cross-filesystem publication must retain compressed allocation,
match the full original SHA-256 and pass the existing ownership checks. The
attempted eligible-size archives still had uncheckpointed sidecars and were
not changed. Large occupied databases and WAL-bearing snapshots are not
disposable stale files; their archival/restore owners must handle them.
Soft scheduler termination now unwinds owned temporary-file and maintenance
hold cleanup. Forced termination or a power loss still requires owner recovery.
The optional cross-volume native compression test was skipped because its
APFS fixture probe was unavailable; a real cross-volume publication is not
claimed by the unit tests or the sidecar-blocked attempts.

- Cold exports could check BOT_LOGS capacity while writing to LaCie. Capacity
  now follows the actual archive filesystem, with independent staging reserves.
- The large cache repair sat behind the broad maintenance gate while the
  pressure pass repeatedly offered small exhausted cleanup sources. Full
  low-load pressure recovery now offers the same verified cache owner first,
  after scratch cleanup; its inner/outer deadlines fit the existing window.
- Timed-out recovery parents could leave descendant workers behind. The
  existing bounded process-group runner now owns their cleanup.
- The governor labeled the cache exporter as unknown external CPU. It now
  accounts for both the coordinator and worker as owned storage work.
- Hiding connection chatter also hid livefeed status refresh, and file selection
  never rediscovered new dated logs. Status polling is now independent, and a
  bounded reader preserves offsets while rediscovering paths each minute.
- The phone view labeled polling time as updated data. Source ages and stale or
  unknown core producers are now explicit, separate from the poll clock.

These are verified defects, not a claim that every control is conflict-free.
The feed continues to show backlog, stale producers and evidence blockers.

## Remaining Evidence

The census is bounded to 2,000 latest artifacts, 10 seconds and 2 MiB per normal
artifact; incomplete or invalid observations remain visible. Its generated
`self_healing_gap_audit_latest.json` contains the per-file ages, all discovered
owner/next-command references, collector classifications and missing-owner
alerts. Historical stale reports need owner-specific review, not blanket
refresh/delete. Source acceptance and immutable-release validation are separate
operator actions; none of these repairs grants order authority.
