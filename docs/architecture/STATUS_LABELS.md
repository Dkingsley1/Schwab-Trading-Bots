# Evidence-Scoped Status Labels

`core/status_label_contract.py` owns reporting labels. It does not grant admission,
clear holds, start workers, change risk limits, accept candidates, or promote bots.

## Platform Scope

The runtime gate dashboard attaches `status_label` to every configured artifact.
Use `status_label.display` for display, `reported_status` for the last producer
verdict, and `evidence_status` for freshness. Existing machine gate fields remain
compatible; a label is not permission to execute. The coverage audit counts the
configured dashboard artifacts, not every historical report or arbitrary string
in the repository.

Labels require aware, nonfuture observation timestamps. Explicit source time
takes precedence over report time, and a new report or file mtime cannot renew an
old observation. Missing, invalid, inconsistent, future and stale evidence remain
distinct. A producer's `ok: true` means producer completion unless the producer
also supplies a scoped verdict; it does not erase an explicit degraded status.
The dashboard uses each artifact's existing age budget. Grade-guard display
labels reuse those budgets where defined; otherwise the age budget is explicitly
unspecified, not silently invented, without changing repair/admission rules.
The livefeed uses its existing per-source budgets and no mtime fallback.

Paper-hold reporting separates the executor's observed hold from current runtime
policy and local-storage reasons. Fresh policy reasons do not prove that an
executor has applied them. Market/session breaker reasons remain separate.
Generic legacy pressure receipts retain their original reason but gain a current
cause explanation in the dashboard and paper-performance report. New executor
starts also report the exact hot-reloaded reason. Running executors are not
restarted merely to adopt a label change.

## Bot Scope

The existing bot-organization refresh labels every registry assignment with:

- Declared registry/lifecycle state, not observed process activity.
- Configured collection enablement, not proof of recent collection.
- Operating-definition completeness and its actual scope.
- Source-bound implementation kind, or unverified source binding.
- Defined process versus verified runtime execution.
- Declared training-label contract versus measured training outcomes.
- Runtime and economic evidence explicitly not assessed by the static audit.

The fleet posture report also identifies registry count fields as configuration
counts. No labels, historical outcomes, source bindings or candidate windows are
rewritten to manufacture training or profitability evidence.

## Write-Path Recovery

The native recovery report retains existing counters and gates, and adds
`recovery_diagnostics`: historical failure debt, the bounded journal census's
latest observed failure and 15-minute event count, current SQL errors with fresh
source evidence, writer evidence, and explicit unmet requirements. Missing SQL
measurements are null, not zero. A default zero with no fresh SQL overlay sources
is unknown; partial overlay coverage cannot establish health for all write paths.
Incomplete scans cannot establish quietness.
Journal paths are counted as identities; this does not probe affected targets.

These diagnostics do not reconcile historical failures. In particular, successful
SQL activity alone cannot prove that a failed JSONL append or another affected
path recovered. A quiet period alone cannot release a hold.

### Next Recovery Improvements

The next implementation should give each affected route and owner a durable
recovery receipt. This work is not implemented by the reporting-label change:

1. Classify failure domains by owner, route and error type, retaining journal
   identity and duplicate provenance without losing failures beyond display caps.
2. Obtain bounded write/read-back and integrity proof through the owning writer,
   produced after the last failure and current route/auth epoch where applicable.
3. Reconcile pending or ambiguous operations from durable checkpoints and stable
   operation IDs. Never blindly replay a fill or assume a failed acknowledgment
   means the write did not commit.
4. Track detecting, isolating, repairing, verifying, probation and recovered
   states. Preserve historical debt with explicit reconciliation receipts instead
   of deleting it or treating all historical events as current errors forever.
5. Require sustained fresh progress, healthy latency and reserve headroom before
   releasing a route; reset probation on new failures. Existing owner floors,
   protected volumes, single-writer authority and trading gates remain intact.
6. Let existing native infrabots run bounded, idempotent recovery with per-owner
   retry budgets, exponential backoff and escalation for unrecoverable paths.
