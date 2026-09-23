# Operations Master Infrabot

## Role And Authority

The operations master is the coordination role of the existing
`master_infrastructure_supervisor`, not another daemon. It turns observed needs
into ordered subgroup directions with an owner, dependency, deferral reason,
bounded action and required completion evidence. Its own healthy execution does
not mean the platform is healthy.

The existing native job has a configured default cadence of five minutes.
It automatically produces directions; repair dispatch is opt-in and defaults off.
Scheduler/resource admission can delay a pass. No Codex automation is installed.
`operations-master` and the previous master-supervisor aliases address the same
owner. The canonical report remains
`governance/health/master_infrastructure_supervisor_latest.json`.

## Subgroups

| Order | Subgroup | Accountable Owner | Responsibilities And Required Proof |
| --- | --- | --- | --- |
| 1 | Runtime | runtime throttle control | Contain unnecessary concurrency, protect foreground use, identify duplicate ownership and restart storms. Fresh resource observations and owner admission are required before increasing work. |
| 2 | Storage | soak self-healing control | Coordinate verified compression, eligible retention and offload. Preserve reserves, active databases, rollback material, original lookup paths and protected volumes. Report actual reclaimed bytes and remaining capacity deficit. |
| 3 | Provider access | provider access guard | Separate auth, entitlement, rate limits and availability. Honor provider cooldowns and request admission. A successful account request cannot clear a market-data 429. Interactive reauthentication remains operator-owned. |
| 4 | Ingestion | storage backpressure autopilot | Prioritize current decision data, then drain older admitted work through the single writer. Require fresh backlog measurements and independent write admission. A requested accelerator is not an observed worker or completed drain. |
| 5 | Integrity | storage disaster recovery | Coordinate backup verification, restore drills, database integrity and durable write recovery. Require actual source-backed proof; metadata presence or an old successful receipt is insufficient. |
| 6 | Commands | command validity bot | Check command routes, usefulness, safe probes and documentation coverage. Never execute trading, auth-consent or destructive commands merely to test them. Unprobed operator-gated commands remain explicitly unproven. |
| 7 | Evidence and research | runtime artifact refresh | Order lineage, freshness, replay, analytics and training dependencies. Preserve producer timestamps and hashes. Defer costly evidence work behind capacity, ingestion and runtime needs. No invented samples, quality or profitability. |
| 8 | Assurance | master infrastructure supervisor | Review subgroup results, unmapped needs, recurring failures, operator escalation and release handoff. Keep system health, implementation completeness and economic evidence separate. |

## Required Responsibilities

### 1. Inventory And Ownership
Map every check to exactly one subgroup and accountable owner. Surface unmapped
checks instead of silently dropping them. Maintain the existing owners' locks and
boundaries; do not create a competing SQL writer or duplicate repair scheduler.

### 2. Triage And Prioritization
Contain resource and capacity problems before expanding intake or training.
Independent diagnostics may continue. Separate the root cause from downstream
symptoms, and preserve the original owner-reported condition in the report.
Use existing severity and blast-radius assessments; a shared timestamp or
dependency is not proof of causation. The current master uses fixed priorities,
not a learned root-cause model.

### 3. Dependency Direction
Publish each subgroup's upstream dependencies and why work is deferred. Storage
recovery uses its own admission lease, not normal maintenance admission; requiring
healthy storage before allowing storage recovery would create a circular block.

### 4. Resource Admission
Read fresh runtime and reserve observations, preserve workload-specific leases,
and check controls again before dispatch. Missing, future or expired controls
cannot grant admission. Power OFF, operator stop, global halt and maintenance
holds inhibit dispatch; the master cannot clear them.
Resource owners retain CPU, memory/compression, swap, thermal, disk, writer and
provider-request budgets. Protect time-sensitive work and ordinary Mac use;
maximum utilization is not the objective. Capacity forecasts must distinguish
occupied data, reclaimable data and verified destination headroom.

### 5. Delegation And Scope
Only four fixed owner actions are dispatched in this version: runtime-throttle
apply, quick bounded storage recovery, ingestion-status refresh, and safe command
audit. Other subgroup work stays with its existing scheduler or is an explicit
operator follow-up. Directions do not grant universal control over independently
scheduled owners. Suggested command strings in reports are never dispatch tokens.
Retention eligibility, compression verification, write checkpointing and backup
restore drills remain with their specialized owners. A retry must retain the
owner's idempotency rules and existing failure debt; duplicate cleanup or a second
writer is never an acceptable shortcut.

### 6. Bounded Execution
Allow at most two owner calls in a 150-second work window, reserving each child's
full configured deadline. A kernel lock prevents overlapping master dispatches.
Children run under bounded process-group cleanup, not unbounded nested retries.
Existing owner admission, locks and budgets still apply inside every child.

### 7. Recovery Verification
An exit code, refreshed wrapper or completed assessment earns no repair credit.
Require fresh owner evidence showing the original issue resolved. Capacity needs
measured headroom; ingestion needs observed backlog progress; provider recovery
needs relevant successful requests; integrity needs actual verification.
Backup owners must report achieved recovery-point and recovery-time evidence
against their existing targets, not infer recoverability from an archive's name
or existence. Unknown or partial restoration stays unresolved.

### 8. Cooldowns And Stability
Persist admission before launching a child. Every attempted owner consumes a
ten-minute cooldown, including failure, timeout and a coordinator crash. Future or
invalid retry clocks fail closed. Never erase an owner's failure history to retry.

### 9. Freshness And Provenance
Keep observation time distinct from source time and decision time. Heartbeat
health cannot certify a fresh symbol decision. Unknown, malformed or protected
input routes remain unavailable. No stale input is relabeled as current.

### 10. Audit And Explainability
Report priority, mission, issues, dependencies, selected action, deferrals and
attempt outcomes. Each subgroup has its own completion criteria and unresolved
delegated needs appear in `escalations`. Persist the last attempt per owner and the most recent 64
dispatch records in `operations_master_dispatch_state.json`. This bounded
operational history is not an immutable long-term audit archive. Canonical owner
evidence and the release acceptance chain retain their existing responsibilities.

### 11. Escalation
Distinguish repairable local defects from unavailable capacity, provider limits,
operator consent and missing qualification time. Name the remaining need and its
owner. Do not repeatedly dispatch a no-progress repair outside its cooldown.
The existing incident/notification owners remain responsible for delivery,
severity-change deduplication and acknowledgement. This version publishes
escalations in the canonical report; it does not add a pager or claim that an
operator has received or acknowledged them.

### 12. Release Handoff
Identify source drift and reviewed-release prerequisites. Acceptance, publication,
freeze changes and deployment approval remain operator-authorized actions, not
self-healing privileges. A new candidate restarts affected evidence windows;
historical losses and cumulative evidence remain intact.
Security and configuration-drift findings go to their existing audit/release
owners. Require reviewed changes, regression results and rollback evidence;
never retrieve credentials, expand permissions or silently amend a contract to
make a check pass. Adaptivity stays within the existing owners' approved bounds.

### 13. Economic Separation
Operational readiness is not proof that a bot should buy or will be profitable.
Preserve HOLD/veto decisions and all risk, market-session, quote, cash, account,
attestation and exact-order gates. Never submit, cancel or replace an order.

### 14. Self-Limits
Do not rewrite contracts, reduce reserves, clear halts, change credentials, invent
evidence, alter retention eligibility or touch `/Volumes/VIDEO`. Definition
completeness and working tests do not certify every subgroup's runtime outcome.

## Operator Commands

```bash
./scripts/ops/opsctl.sh operations-master --json
./scripts/ops/opsctl.sh operations-master --apply --json
```

The first command reports direction only. The second attempts only the fixed
admitted actions above. Both preserve trading authority. Native automatic
dispatch is controlled by `MASTER_INFRASTRUCTURE_SUPERVISOR_APPLY`; `1` opts into
bounded dispatch and `0` (the default) keeps the scheduled owner in direction-only
mode. Existing subgroup schedulers continue their separately admitted work.
