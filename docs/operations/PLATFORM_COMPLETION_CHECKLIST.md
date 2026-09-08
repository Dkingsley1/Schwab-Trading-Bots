# Platform Completion Checklist

This checklist defines what must be proven before describing the agreed release
scope as complete. It is not a claim that every listed subsystem is broken.
"100 percent" means all required acceptance criteria pass, not zero possible
future bugs, continuous availability under every failure, or guaranteed profits.

The defect history and specific open work packages are in
`docs/operations/PLATFORM_HARDENING_LEDGER.md`. Edit the existing owner named in
`docs/architecture/SOURCE_OF_TRUTH.md`, not a generated score or health report.

The detailed [work-package definitions](PLATFORM_COMPLETION_WORK_PACKAGES.md)
expand these areas into 157 stable acceptance requirements, source owners,
verification starting points, closure evidence, ten cross-boundary scenarios,
and explicit operator decisions. These are proof obligations, not 157 confirmed
bugs or completed fixes. This checklist remains the status index.

Status vocabulary:

- **Open:** a specific implementation, operational, or evidence gap was found.
- **Partial:** useful controls/tests exist, but the full claim is not established.
- **Verify:** existing behavior needs release-level proof; this is not a newly
  diagnosed defect or a request to rewrite the subsystem.

## Data And Storage

| ID | Area / Status | Required To Close |
| --- | --- | --- |
| C01 | Storage routes / Partial | Every mutable database, SQLite sidecar, shard, snapshot, model, report, cache, and archive has an explicit permitted physical destination and owner. Distinguish configured intent from actual routing. No implicit fallback onto protected storage, split-brain writers, or destructive fallback. |
| C02 | Shared capacity budget / Open | Account together for live data, WAL growth, pending queues, snapshot/restore copies, vacuum scratch, archives, training artifacts, and foreground use. Define reserves, forecasts, admission limits, deferrals, and escalation. Test combined disk/memory/CPU pressure. |
| C03 | Full recovery / Open | Restore production-scope data, models, configuration, and restart-critical control evidence to an approved isolated destination. Verify consistency, digests/rows, application restart, maximum acceptable data loss, measured recovery time, and immutable receipts. Metadata and simulated timings earn no full-restore credit. |
| C04 | Archive integrity / Open | Investigate the preserved damaged archive using an approved recovery copy. Verify salvaged/repaired contents and disclose unrecoverable records. Do not delete the original merely because another archive shares its date or filename. |
| C05 | Ingestion semantics / Partial | Define event identity, provenance, event/arrival time, schemas, quality gates, acknowledgment boundaries, replay, deduplication, out-of-order handling, quarantine, and restart behavior. Prove bounded loss/duplication behavior during interrupted ingestion and downstream outages. |
| C06 | Schema compatibility / Open | Validate required types, values, nested invariants, supported versions, migration/rollback compatibility, and point-in-time replay fixtures. The current field-presence guard is not full compatibility validation. |
| C07 | Retention and deletion / Partial | Each data family has an owner, retention policy, hot/warm/cold transition, reference checks, integrity and restore requirements, grace periods, and deletion authority. Failed, mutable, referenced, or unverified data cannot be mistaken for disposable capacity. |

## Runtime And Operations

| ID | Area / Status | Required To Close |
| --- | --- | --- |
| C08 | Configuration provenance / Open | Show the redacted winning value/source, precedence, owner, constraints, expiry, and required adoption action for each setting. Prove which configuration each process actually loaded. Never expose secret values in provenance reports. |
| C09 | Scheduled-job lifecycle / Open | Every job exposes scheduled, eligible, deferred, started, completed, failed, and next-eligible states, with reasons and timestamps. Test quiet-window deferrals, locks, dependency failure, missed runs, restarts, and clock changes. |
| C10 | Bounded execution / Open | Reproduce and fix the intermittent training-snapshot timeout. Bound discovery, lock waiting, reads, parsing, fallback, and publication, not only one scan phase. All workers need cancellation, retry limits, deadlines, and cleanup of owned temporary state. |
| C11 | Self-healing circuits / Open | Prove lightweight detection and heavier repair cooperate, obey per-surface budgets, stop after repeated failure, preserve single-writer ownership, and recover from an interrupted repair. Open circuits require evidence-based resolution, not blanket resets. |
| C12 | Startup and failover / Verify | Verify reboot, login/session changes, dependency ordering, stale locks, missing mounts, network loss, token expiration, and interrupted writes on the actual deployment. Demonstrate restart without duplicate workers/orders or silently stale evidence. |
| C13 | Evidence freshness / Partial | Extend producer-time and scope checks across all safety-relevant consumers. A fresh summary, successful exit code, file mtime, or old green status cannot establish fresh underlying evidence. Observe the newly corrected publication paths across unattended cycles. |
| C14 | Independent monitoring / Open | Verify a monitor independent of the trading runtime, an approved off-host receiver, delivery receipts, and detection when the host or main watchdog disappears. A local heartbeat cannot prove off-host alerting. |
| C15 | Alerts and incidents / Open | Define severity, owner, deduplication, delivery failure, acknowledgment, escalation, resolution, and reopening. Test terminal failure and exhausted retry budgets, not only successful notification calls. |

## Safety And Research

| ID | Area / Status | Required To Close |
| --- | --- | --- |
| C16 | Authority enforcement / Partial | Verify every mutating entry point enforces its action/ownership contract, including scripts, retries, restart paths, and broker adapters. Registry completeness is not proof that every caller uses its guard. |
| C17 | Broker/order lifecycle / Verify | Prove account binding, fresh broker truth, quote provenance, preflight, durable reservation, at-most-once submission, ambiguous-outcome reconciliation, partial fills, cancel handling, and restart recovery. Paper success cannot authorize live execution. |
| C18 | Portfolio/risk/accounting / Verify | Verify position, cash, fees, realized/unrealized P&L, exposure, drawdown, sizing, and cross-sleeve constraints agree with authoritative broker/ledger state. Test stale input, reconciliation disagreements, and kill-switch behavior. |
| C19 | Security and dependencies / Open | Reproduce the dependency environment, check current advisories, protect credentials and logs, validate access boundaries, and test auth rotation/recovery without leaking secrets. Secret scanning is only one check, not a security certification. |
| C20 | Honest grades / Open | Separately report structural coverage, operational availability, data quality, unresolved improvements, raw financial outcomes, and promotion eligibility. A structural score of 100 cannot cancel missing execution evidence or open defects. |
| C21 | Research lineage and reproducibility / Partial | Bind training inputs, feature definitions, dataset versions, model artifacts, code/configuration, and results. Prove checkpoint/resume, deterministic replay where applicable, point-in-time boundaries, leakage controls, and independent validation. |
| C22 | Institutional research evidence / Open | Produce real runtime evidence for seven gaps: candidate risk schedules, cross-engine valuation reconciliation, execution speed/cost frontier, independent factor benchmarks, pipeline incident ownership, research DAG checkpoint/resume, and versioned dataset storage. Eight implemented controls currently do not mean eight evidenced controls. |
| C23 | Economic qualification and promotion / Open | Obtain sufficient candidate-bound, independent, post-cost evidence under the existing validation policy. Preserve regime, benchmark, statistical, drawdown, and operational gates. Meet the policy's elapsed-time and market-session requirements; do not lower them to reach a score. Promotion remains an explicit separate decision. |

## Release And Ownership

| ID | Area / Status | Required To Close |
| --- | --- | --- |
| C24 | Source and release organization / Open | Publish reviewed, dependency-complete source changes; keep secrets/runtime data out of Git; make an immutable build; identify its exact code/configuration/dependency versions; verify rollback and deployment adoption. Git publication is not production-candidate acceptance. |
| C25 | Complete test evidence / Partial | Run appropriate unit, integration, contract, migration, restart, failure-injection, concurrency, load, and long-duration tests against the actual release artifact. The 515-test hardening run covered 29 files; subsequent publication verification passed 2,268 tests and two subtests across 126 changed test files. Neither proves every production workflow. Track gaps and flaky tests explicitly. |
| C26 | Documentation and operator consistency / Partial | Every control has one owner, inputs/outputs, authority boundary, failure behavior, recovery procedure, evidence path, and escalation destination. Generated commands, dashboards, runbooks, and readiness summaries must agree with the implemented behavior. |

September 8 continuation: C11's misclassified storage-memory circuit is corrected
and its unattended adoption was observed at 21:34 UTC. C10/C13/C21 now have
bounded sidecar scans, shared incremental/seed deadlines, producer-owned snapshot
publication, and separate coordinator failure receipts. Those changes do not close
the broad areas: the production snapshot verification attempt deferred for host
pressure/quiet hours, internal capacity remains below reserve, and full restore,
off-host monitoring, clean candidate acceptance, and organic qualification remain
separate prerequisites. See the hardening ledger for the dated evidence.

## Completion Rules

For each required row record the accountable owner, versioned contract, baseline
failure or proof obligation, acceptance tests, runtime evidence, and deployment
identity. A row is complete only after the deployed release satisfies its criteria.
Passing implementation tests is one stage, not the final stage.

Evaluate collection health, unattended operational readiness, research quality,
and live-promotion eligibility independently. The September 7, 23:36 UTC
resilience report counted 8 of 10 implementation sections and 3 of 10 live-ready
sections, while fast collection health could remain guarded-ready. Those are
different denominators and claims, not a global 80-percent or 30-percent bug score.

The immediate order is recovery/capacity and archive integrity; timeout,
scheduler, and self-healing reliability; monitoring and configuration adoption;
release/authority/security verification; then complete research and candidate
qualification. Parallel independent work is possible, but no row gains evidence
by being added to a checklist, pushed to GitHub, or assigned a better grade.

## Execution Waves And Hidden Defects

The operator requested work across all 26 areas. These are work packages, not
26 patches. No row is marked complete by this follow-up implementation pass.

| Wave | Checklist Coverage | Exit Evidence |
| --- | --- | --- |
| 1: Data safety | C01-C07 | Route and ingestion failure tests, shared admission, schema/replay fixtures, verified archive investigation, and an approved production-scope isolated restore. |
| 2: Unattended operation | C08-C15 | Configuration adoption, scheduled and deferred lifecycle receipts, total deadlines, bounded repairs, controlled restart/failover exercises, and approved off-host delivery/failure receipts. |
| 3: Trust and qualification | C16-C23 | Entrypoint authority tests, broker/accounting reconciliation, dependency/security review, honest grades, reproducible research, genuine institutional evidence, and policy-required candidate history. |
| 4: Release closure | C24-C26, with all earlier rows rechecked | Immutable reviewed build, rollback/deployment proof, broader integration/failure/load/soak testing, and consistent operator documentation. |

Newly reproduced defects belong under the affected row, with a regression test
and a ledger entry. Cross-subsystem, concurrency, interrupted-write, clock,
capacity, malformed-evidence, and restart tests are mandatory discovery work.
If a finding introduces a genuinely new release requirement, add it explicitly;
do not hide it to preserve the number 26. Reopen a closed row if its release
identity, evidence validity, or acceptance criteria cease to hold.

PH-13 through PH-21 in the hardening ledger record this pass's capacity,
retention, recovery-manifest, freshness, worker-timeout, atomic-publication, and
security-audit and missing scheduled-producer fixes. Those fixes advance C02, C03, C07, C09, C10, C13, C19, C21,
C24, C25, and C26, but do not close their broader proof obligations.

Production-sized restore/archive work still requires an approved isolated
destination and resource window; real reboot/failover tests require an approved
interruption scope; off-host tests require a receiver approved by the operator.
External advisories, independently sourced outcomes, market sessions, and
candidate acceptance cannot be manufactured by editing local reports. Neither
this checklist nor a successful Git push authorizes live trading or promotion.
