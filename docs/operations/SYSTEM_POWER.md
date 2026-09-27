# Platform Power

Run the desired command from the repository root. Check status:

```sh
./scripts/ops/opsctl.sh system-power status --json
```

Clear operator/global halts through their owners:

```sh
./scripts/ops/opsctl.sh system-power clear-halts --json
```

Turn on in guarded paper/live-data mode:

```sh
./scripts/ops/opsctl.sh system-power on --json
```

Turn off:

```sh
./scripts/ops/opsctl.sh system-power off --json
```

`clear-halts` combines operator release, native evidence refresh, and guarded
global-halt clearance. It never force-clears a current risk/storage/auth fault.
An explicit power-off request requires `on` instead of clearing a stop behind it.

`off` records previously loaded LaunchAgents whose program arguments belong to
this repository, writes a persistent off flag, engages the operator stop,
disables/unloads repository agents, and invokes the existing stack shutdown.
Unrelated Mac services are excluded. Failed steps are reported as incomplete.
The saved inventory survives repeated OFF calls and host restarts.

`on` checks safe halt clearance, invokes the existing `start --paper` workflow,
and restores the recorded agents. Changed/missing agents and agents explicitly
requesting live execution are not restored. New or previously unloaded agents
are not implicitly installed. A first-ever ON uses the existing core-stack
start scope; it does not invent a complete installation inventory.
Failed startup restores the OFF latch and operator stop and reports incomplete
startup, not a successful ON. The shared scheduled runner defers while OFF
without changing any producer timestamp.

This switch does not place/cancel orders, liquidate holdings, issue attestations,
accept source, or promote production. Keep independent broker access for open
orders: stopping host processes cannot guarantee brokerage orders are canceled.
Status reports requested state and the last transition, not verified overall
system health. Ad hoc processes outside the existing stack stop inventory may
need their own owner shutdown; do not infer that every Python process belongs
to this platform.
