# Distributed Cell Architecture

Generated: 2026-07-23T02:35:07.564394+00:00

Architecture: A+ | Score: 100.0
Operational health: F | Score: 20.429 | Status: blocked
Distributed mode: drain_first

| Cell | Status | Grade | Needs | Stale |
| --- | --- | --- | ---: | ---: |
| Control Plane | blocked | F | 4 | 2 |
| Sleeve Cells | blocked | F | 4 | 2 |
| Storage / Writer Cell | blocked | F | 3 | 2 |
| Training Cell | blocked | F | 2 | 1 |
| Market Data Cell | blocked | F | 3 | 2 |
| Execution / Paper Cell | blocked | F | 2 | 1 |
| Infra Cell | blocked | F | 2 | 1 |

## Next Needs

- `control_plane` `whole_system_intelligence`: mlx_or_gpu_lane_capped -> `./scripts/ops/opsctl.sh system-intelligence --apply --json`
- `control_plane` `whole_system_governor`: mlx_or_gpu_lane_capped -> `./scripts/ops/opsctl.sh whole-system-governor --json`
- `control_plane` `autonomic_resource_governor`: mlx_or_gpu_lane_capped -> `./scripts/ops/opsctl.sh system-intelligence --apply --json`
- `control_plane` `system_needs_intelligence`: mlx_or_gpu_lane_capped -> `./scripts/ops/opsctl.sh system-intelligence --apply --json`
- `sleeve_cells` `sleeve_profitability_dashboard`: sleeve_profitability_dashboard is ready -> `./scripts/ops/opsctl.sh sleeve-profitability-dashboard --json`
- `sleeve_cells` `sleeve_ticker_universe`: sleeve_ticker_universe is missing -> `./scripts/ops/opsctl.sh sleeve-ticker-universe --json`
- `sleeve_cells` `backlog_pump_infrabots`: backlog_pump_infrabots is advisory -> `./scripts/ops/opsctl.sh backlog-pump-infrabots --apply --json`
- `sleeve_cells` `paper_profitability_control`: paper_profitability_control is protective_tightening -> `./scripts/ops/opsctl.sh sleeve-profitability-dashboard --json`
- `storage_writer_cell` `storage_backpressure_autopilot`: storage_backpressure_autopilot is running -> `./scripts/ops/opsctl.sh storage-backpressure-autopilot --apply --json`
- `storage_writer_cell` `writer_process_intelligence`: writer_process_intelligence is ready -> `./scripts/ops/opsctl.sh training-drain-autopilot --apply --json`
- `storage_writer_cell` `backlog_pcore_accelerator`: backlog_pcore_accelerator is ready -> `./scripts/ops/opsctl.sh training-drain-autopilot --apply --json`
- `training_cell` `training_runtime`: autonomic_training_budget_closed -> `./scripts/ops/opsctl.sh training-runtime-control --limit 30 --json`
