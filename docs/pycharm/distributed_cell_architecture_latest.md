# Distributed Cell Architecture

Generated: 2026-07-28T15:27:34.204855+00:00

Architecture: A+ | Score: 100.0
Guarded soak health: D | Score: 65.271 | Status: needs_work
Raw production backlog: D | Score: 65.271 | Status: needs_work
Distributed mode: drain_first

| Cell | Raw Status | Raw Grade | Needs | Stale |
| --- | --- | --- | ---: | ---: |
| Control Plane | ready | A+ | 0 | 0 |
| Sleeve Cells | ready | A+ | 0 | 0 |
| Storage / Writer Cell | blocked | F | 1 | 0 |
| Training Cell | blocked | F | 3 | 2 |
| Market Data Cell | ready | A+ | 0 | 0 |
| Execution / Paper Cell | ready | A+ | 0 | 0 |
| Infra Cell | advisory | A+ | 1 | 1 |

## Next Needs

- `storage_writer_cell` `storage_quota_guard`: backlog_age_or_lines_not_green_for_p_core_acceleration -> `./scripts/ops/opsctl.sh training-drain-autopilot --apply --json`
- `training_cell` `training_runtime`: autonomic_training_budget_closed -> `./scripts/ops/opsctl.sh training-runtime-control --limit 30 --json`
- `training_cell` `training_labeling`: autonomic_training_budget_closed -> `./scripts/ops/opsctl.sh training-runtime-control --limit 30 --json`
- `training_cell` `training_probation_isolation`: autonomic_training_budget_closed -> `./scripts/ops/opsctl.sh training-runtime-control --limit 30 --json`
- `infra_cell` `watchdog_intelligence`: watchdog_intelligence is ready -> `./scripts/ops/opsctl.sh runtime-throttle --apply --json`
