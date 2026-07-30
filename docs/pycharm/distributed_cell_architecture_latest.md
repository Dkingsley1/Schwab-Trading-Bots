# Distributed Cell Architecture

Generated: 2026-07-30T15:04:08.957782+00:00

Architecture maturity: A+ | Score: 100.0
Guarded soak runtime health: C | Score: 73.021 | Status: needs_work
Raw production backlog visibility: C | Score: 73.021 | Status: needs_work
Distributed mode: training_ready
Sleeve guard posture: paper_repair_guarded_with_systemic_weak_point_locks | Recurrence guarded: 25/25 | Systemic weak points: 6
Sleeve profitability evidence: controlled A+ | raw D | paper_only=True | live_execution_allowed=False

| Cell | Raw Status | Raw Grade | Needs | Stale |
| --- | --- | --- | ---: | ---: |
| Control Plane | blocked | F | 4 | 3 |
| Sleeve Cells | advisory | A | 3 | 3 |
| Storage / Writer Cell | advisory | A+ | 1 | 1 |
| Training Cell | advisory | A | 3 | 3 |
| Market Data Cell | advisory | A+ | 2 | 2 |
| Execution / Paper Cell | ready | A+ | 0 | 0 |
| Infra Cell | advisory | A+ | 2 | 2 |

## Sleeve Guard Posture

- Recurrence guard ready: `True`; guarded weak profiles: `25/25`.
- Systemic guard ready: `True`; active systemic causes: `conflict:low, event_proximity:low, fill_quality:unknown, session:intraday, source_quality:low`.
- Top recurrent causes: `conflict:low, event_proximity:low, fill_quality:unknown, source_quality:low, spread_regime:unknown`.
- Rule: controlled grades describe protection strength; raw profitability only improves after fresh paper PnL evidence improves.

## Next Needs

- `control_plane` `whole_system_intelligence`: whole_system_intelligence is advisory -> `./scripts/ops/opsctl.sh system-intelligence --apply --json`
- `control_plane` `whole_system_governor`: whole_system_governor is ready -> `./scripts/ops/opsctl.sh whole-system-governor --json`
- `control_plane` `autonomic_resource_governor`: autonomic_resource_governor is ready -> `./scripts/ops/opsctl.sh system-intelligence --apply --json`
- `control_plane` `system_needs_intelligence`: system_needs_intelligence is needs_action -> `./scripts/ops/opsctl.sh system-intelligence --apply --json`
- `sleeve_cells` `sleeve_ticker_universe`: sleeve_ticker_universe is ready -> `./scripts/ops/opsctl.sh sleeve-ticker-universe --json`
- `sleeve_cells` `backlog_pump_infrabots`: backlog_pump_infrabots is advisory -> `./scripts/ops/opsctl.sh backlog-pump-infrabots --apply --json`
- `sleeve_cells` `data_collection_observation_rollup`: data_collection_observation_rollup is ready -> `./scripts/ops/opsctl.sh sleeve-profitability-dashboard --json`
- `storage_writer_cell` `backlog_pcore_accelerator`: backlog_pcore_accelerator is ready -> `./scripts/ops/opsctl.sh training-drain-autopilot --apply --json`
- `training_cell` `training_data_intake`: training_data_intake is ready -> `./scripts/ops/opsctl.sh training-data-intake --apply --json`
- `training_cell` `training_labeling`: training_labeling is ready -> `./scripts/ops/opsctl.sh training-runtime-control --limit 30 --json`
- `training_cell` `training_probation_isolation`: training_probation_isolation is ready -> `./scripts/ops/opsctl.sh training-runtime-control --limit 30 --json`
- `market_data_cell` `provider_mesh`: provider_mesh is ready -> `./scripts/ops/opsctl.sh provider-mesh --json`
