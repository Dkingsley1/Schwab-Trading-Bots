# Distributed Cell Architecture

Generated: 2026-07-27T19:50:20.189649+00:00

Architecture: A+ | Score: 100.0
Guarded soak health: A+ | Score: 100.0 | Status: ready
Raw production backlog: F | Score: 35.95 | Status: blocked
Distributed mode: drain_first

| Cell | Raw Status | Raw Grade | Needs | Stale |
| --- | --- | --- | ---: | ---: |
| Control Plane | blocked | F | 3 | 1 |
| Sleeve Cells | blocked | F | 5 | 2 |
| Storage / Writer Cell | blocked | F | 2 | 1 |
| Training Cell | blocked | F | 1 | 0 |
| Market Data Cell | blocked | F | 3 | 2 |
| Execution / Paper Cell | blocked | F | 1 | 0 |
| Infra Cell | blocked | F | 1 | 0 |

## Next Needs

- `control_plane` `whole_system_governor`: mlx_or_gpu_lane_capped -> `./scripts/ops/opsctl.sh whole-system-governor --json`
- `control_plane` `autonomic_resource_governor`: mlx_or_gpu_lane_capped -> `./scripts/ops/opsctl.sh system-intelligence --apply --json`
- `control_plane` `system_needs_intelligence`: mlx_or_gpu_lane_capped -> `./scripts/ops/opsctl.sh system-intelligence --apply --json`
- `sleeve_cells` `sleeve_profitability_dashboard`: sleeve_profitability_dashboard is ready -> `./scripts/ops/opsctl.sh sleeve-profitability-dashboard --json`
- `sleeve_cells` `sleeve_ticker_universe`: sleeve_ticker_universe is missing -> `./scripts/ops/opsctl.sh sleeve-ticker-universe --json`
- `sleeve_cells` `backlog_pump_infrabots`: backlog_pump_infrabots is advisory -> `./scripts/ops/opsctl.sh backlog-pump-infrabots --apply --json`
- `sleeve_cells` `paper_profitability_control`: paper_profitability_control is protective_tightening -> `./scripts/ops/opsctl.sh sleeve-profitability-dashboard --json`
- `sleeve_cells` `data_collection_observation_rollup`: data_collection_observation_rollup is degraded -> `./scripts/ops/opsctl.sh sleeve-profitability-dashboard --json`
- `storage_writer_cell` `writer_cycle_coordinator`: writer_cycle_coordinator is idle -> `./scripts/ops/opsctl.sh writer-cycle-coordinator --json`
- `storage_writer_cell` `writer_process_intelligence`: writer_process_intelligence is ready -> `./scripts/ops/opsctl.sh training-drain-autopilot --apply --json`
- `training_cell` `training_quality`: training_quality is needs_attention -> `./scripts/ops/opsctl.sh training-quality --json`
- `market_data_cell` `provider_mesh`: provider_mesh is ready -> `./scripts/ops/opsctl.sh provider-mesh --json`
