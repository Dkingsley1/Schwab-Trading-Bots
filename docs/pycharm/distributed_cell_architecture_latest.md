# Distributed Cell Architecture

Generated: 2026-07-27T16:53:59.459393+00:00

Architecture: A+ | Score: 100.0
Operational health: F | Score: 23.25 | Status: blocked
Distributed mode: drain_first

| Cell | Status | Grade | Needs | Stale |
| --- | --- | --- | ---: | ---: |
| Control Plane | blocked | F | 2 | 0 |
| Sleeve Cells | blocked | F | 4 | 1 |
| Storage / Writer Cell | blocked | F | 2 | 0 |
| Training Cell | blocked | F | 3 | 1 |
| Market Data Cell | blocked | F | 3 | 2 |
| Execution / Paper Cell | blocked | F | 2 | 1 |
| Infra Cell | blocked | F | 1 | 0 |

## Next Needs

- `control_plane` `whole_system_governor`: mlx_or_gpu_lane_capped -> `./scripts/ops/opsctl.sh whole-system-governor --json`
- `control_plane` `system_needs_intelligence`: mlx_or_gpu_lane_capped -> `./scripts/ops/opsctl.sh system-intelligence --apply --json`
- `sleeve_cells` `sleeve_profitability_dashboard`: sleeve_profitability_dashboard is ready -> `./scripts/ops/opsctl.sh sleeve-profitability-dashboard --json`
- `sleeve_cells` `sleeve_ticker_universe`: sleeve_ticker_universe is missing -> `./scripts/ops/opsctl.sh sleeve-ticker-universe --json`
- `sleeve_cells` `paper_profitability_control`: paper_profitability_control is protective_tightening -> `./scripts/ops/opsctl.sh sleeve-profitability-dashboard --json`
- `sleeve_cells` `data_collection_observation_rollup`: data_collection_observation_rollup is degraded -> `./scripts/ops/opsctl.sh sleeve-profitability-dashboard --json`
- `storage_writer_cell` `storage_backpressure_autopilot`: storage_backpressure_autopilot is applied_with_followups -> `./scripts/ops/opsctl.sh storage-backpressure-autopilot --apply --json`
- `storage_writer_cell` `writer_cycle_coordinator`: writer_cycle_coordinator is applied_with_followups -> `./scripts/ops/opsctl.sh writer-cycle-coordinator --json`
- `training_cell` `training_runtime`: training_runtime is degraded -> `./scripts/ops/opsctl.sh training-runtime-control --limit 30 --json`
- `training_cell` `training_quality`: training_quality is blocked -> `./scripts/ops/opsctl.sh training-quality --json`
- `training_cell` `training_probation_isolation`: training_probation_isolation is ready -> `./scripts/ops/opsctl.sh training-runtime-control --limit 30 --json`
- `market_data_cell` `source_verification`: source_verification is degraded -> `./scripts/ops/opsctl.sh source-verification --json`
