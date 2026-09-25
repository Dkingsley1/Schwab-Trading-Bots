# Live Execution Control

**WARNING: REAL MONEY.** This interlock is separate from system power. It does
not grant broker authority, accept source, clear halts, modify environment flags,
or submit an order. Keep independent broker access.

The system role contract assigns the switch to the existing
`live_canary_boundary_controller`, including its source files and persisted state.
That owner cannot submit orders; execution remains with the live gateway.

The generated `COMMANDS.md` has a dedicated section above Most Used:

```bash
./scripts/ops/opsctl.sh live-execution off --json
./scripts/ops/opsctl.sh live-execution status --json
./scripts/ops/opsctl.sh live-execution on
```

OFF blocks new real-broker placements and replacements, including exit orders.
Broker reads, reconciliation and cancellation are unaffected by this switch;
their own API/authentication controls still apply. OFF does not cancel orders
already at Schwab, recall in-flight requests, liquidate holdings or stop paper
trading/data collection. An OFF persistence error must be treated as unverified,
not success. Check the broker and use existing operator emergency controls.

Each ON/OFF command uses the existing native macOS notification transport after
the transition result is known. The notification explicitly names **Live
Execution**, not platform power. Blocked/unverified transitions report failure;
notification delivery errors are returned separately and never undo OFF. Clicking
the native notification opens `COMMANDS.md`, never an order or activation action.
OS notification settings can suppress display. Read-only status sends no alert.

ON requires an interactive terminal, purpose, symbol and exact confirmation.
It grants a local permission lease for 30 minutes (explicit `--minutes 1..60`).
The default session is NORMAL; explicit `--session AM` or `--session PM` is
available only through the existing eligible supervised test policies. Current
support is Schwab; this does not enable Coinbase execution.

Production activation requires native canary preflight and allowlist readiness.
Supervised activation requires native technical readiness; attestation blockers
are deferred to the existing interactive submit dialog, which still requires
fresh account/risk review and exact confirmation for each BUY and SELL. This
avoids making the switch depend on an attestation that the blocked submit dialog
has not yet collected. No order-level checks are relaxed. Activation itself does
not fetch broker data; stale evidence remains a blocker for its native owner.

The lease binds candidate, account registry and risk/test policy file hashes.
Missing, malformed, unreadable, expired or mismatched state blocks dispatch.
Changing purpose, symbol or session requires a fresh ON confirmation. Other
execution gates are checked by the order owner at submission, so ON is not a
claim of live readiness or an authorization to trade. The switch cannot turn a
bot HOLD into BUY, guarantee a profitable exit, or enable unattended tests.

`core/live_execution_switch.py` owns
`governance/runtime/live_execution_switch_state.json` and its short local state-write
lock. No scheduler renews the lease. An OFF generation supersedes a pending ON
check. The broker dispatcher re-reads the switch after rate-limit admission and
immediately before invoking the broker. An already-dispatched request cannot be
recalled. A known switch rejection before dispatch is recorded as rejected,
not an ambiguous broker outcome; consumed intent IDs are never auto-retried.

## Deployment

Processes must load the switch-enabled source before they enforce it. Editing
files or persisting OFF does not retrofit an existing Python process. Perform
the normal reviewed release and controlled runtime adoption before relying on
this switch. Tests use temporary state and fake broker clients only. Do not
restart live processes, activate the real switch, or submit an order as a test
of this implementation.
