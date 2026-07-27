# Distributed Cell Architecture

Generated: 2026-07-27T22:28:42.439293+00:00

Architecture: A+ | Score: 100.0
Guarded soak health: D | Score: 61.929 | Status: needs_work
Raw production backlog: D | Score: 61.929 | Status: needs_work
Distributed mode: drain_first

| Cell | Raw Status | Raw Grade | Needs | Stale |
| --- | --- | --- | ---: | ---: |
| Control Plane | blocked | F | 1 | 0 |
| Sleeve Cells | ready | A+ | 0 | 0 |
| Storage / Writer Cell | blocked | F | 3 | 0 |
| Training Cell | blocked | F | 1 | 0 |
| Market Data Cell | ready | A+ | 0 | 0 |
| Execution / Paper Cell | ready | A+ | 0 | 0 |
| Infra Cell | ready | A+ | 0 | 0 |

## Next Needs

- `control_plane` `system_needs_intelligence`: backlog_above_target_or_old_pending_work -> `./scripts/ops/opsctl.sh system-intelligence --apply --json`
- `storage_writer_cell` `storage_quota_guard`: storage_quota_guard is degraded -> `./scripts/ops/opsctl.sh training-drain-autopilot --apply --json`
- `storage_writer_cell` `storage_backpressure_autopilot`: storage_backpressure_autopilot is applied_with_followups -> `./scripts/ops/opsctl.sh storage-backpressure-autopilot --apply --json`
- `storage_writer_cell` `writer_cycle_coordinator`: writer_cycle_coordinator is idle -> `./scripts/ops/opsctl.sh writer-cycle-coordinator --json`
- `training_cell` `training_runtime`: autonomic_training_budget_closed -> `./scripts/ops/opsctl.sh training-runtime-control --limit 30 --json`
