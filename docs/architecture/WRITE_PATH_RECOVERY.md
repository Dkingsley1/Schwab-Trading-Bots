# Native Write-Path Recovery

The existing SQL launchd schedule publishes `sql_writer_observation_latest.json`
before early storage or maintenance exits. This is a fresh admission observation,
not a new SQL progress timestamp or a zero-error claim. Old progress stays old.

`data-plane-recovery --apply --json` also runs through the existing SQL schedule
and native artifact refresh. The recovery controller serializes its own passes.
Storage repair remains with the existing guarded self-healing owner; SQL drains
remain with the admitted single writer. No new scheduler is installed.

## Per-Path Proof

- Failure domains bind source owner, exact target, UTC day and failure generation.
- At most four paths are checked per pass within a three-second loop budget.
- Only actual requested JSONL or atomic JSON writes produce success receipts.
- Receipts require read-back, file/directory fsync, byte hash and device/inode identity.
- Oversized batches, concurrent target changes, external routes and missing proof cannot pass.
- Request receipts expire; unchanged receipts cannot advance probation.
- Two successful owner observations at least 60 seconds apart qualify path recovery.

The writer instrumentation is loaded at each process's next managed start.
Running long-lived writers are not force-restarted by this change.

## Historical Reconciliation

New append failures retain bounded stable message IDs and exact enriched-payload
hashes in the existing failure journal. Verification reads at most 1 MiB of tail
data and requires matching records without duplicate or conflicting IDs. It does
not resubmit decisions, orders or fills. Normal SQL checkpoint recovery remains
the ingestion owner's responsibility.

Path recovery and historical reconciliation are separate fields. Atomic snapshot
supersession, missing old payloads, records outside the bounded tail, and SQLite
queue history without an owner checkpoint do not prove recovery of old events.
These remain unresolved rather than being silently dropped or blindly replayed.
The queue no longer reports a locked schema query as a successful schema check.

Unresolved daily domains survive the two-family census window. A migration with
older unclassified aggregate debt retains a conservative explicitly labeled count;
it is not an exact new incident count. Incomplete scans cannot erase prior state.

## Bounds And Authority

Failed checks back off from 60 to 900 seconds and escalate after six attempts per
failure generation. A new failure generation resets probation. Recovery state is
bounded to 256 domains and 2 MiB; each failed batch checkpoint holds at most 128
record identities. Control state and proof receipts are excluded from ingestion.

No recovery receipt changes storage reserves, broker approval, trading locks,
economic evidence or execution authority. A read-only `data-plane-recovery --json`
reports state without advancing requests, retries or probation.
