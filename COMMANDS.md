# Commands (Canonical)

Use these exact commands as the current source of truth.

This file is intentionally trimmed down:
- paper mode is the operating default
- no simulate commands are listed
- no redundant partial start commands are listed when the full stack command already covers them
- live feed refreshes and live feed views are separated clearly

## Most Used

### Keep the Mac awake
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
caffeinate -dimsu
```

### Start the full live stack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh start
```

### Stop the stack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh stop
```

### Schwab handshake
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh-interactive
```
Expected healthy result: `Handshake Successful.`

### Set the global trading halt
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/operator_control.py --set-global-halt --reason "manual_operator_halt" --json
```

### Clear the global trading halt
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/operator_control.py --clear-global-halt --json
```

### Refresh all live feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source all
```

### Live feed view for all feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source all --lines 80
```

### Phone browser mirror for the live terminal feed
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh phone-feed --host 0.0.0.0 --port 8787 --source all --lines 80 --include-decisions
```

## Data Context Syncs

### Crypto market context sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh crypto-market-sync --json
```

### Stock / crypto correlation sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh market-correlation-sync --json
```

### FX market context sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh fx-market-sync --json
```

### Macro context sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-context-sync --json
```

### Macro crosscheck
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-crosscheck --json
```

### Source verification
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh source-verification --json
```

### Tastytrade context sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh tastytrade-sync --json
```

## Keep Awake

### Keep the Mac awake during long feed, SQL, or retrain work
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
caffeinate -dimsu
```

## Live Feed Refreshes

### Refresh all live feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source all
```

### Refresh Schwab live feeds only
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

### Refresh Coinbase live feeds only
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source coinbase
```

### Refresh FX live feeds only
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source fx
```

## Live Feed Views

Light views are the default daily-use tails. They skip the heavy decision firehose.

### Light live feed view for all feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source all --lines 80
```

### Snapshot of the live feed without staying attached
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source all --lines 80 --snapshot
```

### Light live feed view for Schwab, Coinbase, and futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh main-tail --lines 80
```

### Light live feed view for Schwab
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-tail --lines 80
```

### Light live feed view for Coinbase
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-tail --lines 80
```

### Light live feed view for all futures sleeves
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh futures-tail --lines 80
```

### Light live feed view for Schwab futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-futures-tail --lines 80
```

### Light live feed view for Coinbase futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-futures-tail --lines 80
```

### Light live feed view for FX
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh fx-tail --lines 80
```

Heavy views include the decision firehose and are more expensive on memory/scrollback.

### Heavy live feed view for all feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source all --lines 80 --include-decisions
```

### Heavy live feed view for Schwab, Coinbase, and futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source main --lines 80 --include-decisions
```

### Heavy live feed view for Schwab
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-tail --lines 80 --include-decisions
```

### Heavy live feed view for Coinbase
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-tail --lines 80 --include-decisions
```

### Heavy live feed view for futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh futures-tail --lines 80 --include-decisions
```

## Report Generation

### Training report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh training-report --allow-gui-pdf-renderer --json
```

### Crash report digest
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh crash-report --lookback-days 3 --recent-limit 40 --allow-gui-pdf-renderer --json
```

### Project timeline report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh timeline-report --render-pdf --allow-gui-pdf-renderer --json
```

### Paper performance report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh paper-performance --day "$(date -u +%Y%m%d)" --week-days 7 --json
```
The lead chart in the rendered report is now a daily line graph.

### Sentiment report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh sentiment-report --day "$(date -u +%Y%m%d)" --lookback-days 180 --allow-gui-pdf-renderer --json
```

### Open the latest paper performance PDF
```bash
./scripts/ops/open_report_artifact.sh paper
```

### Open the latest sentiment report PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh sentiment
```

## Reports

### Refresh the report catalog and PDF bundle
Run this first if a PDF is missing or stale.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh report-pdfs --allow-gui-pdf-renderer --json
```

### Open the report catalog PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh bundle
```

### Open the latest crash report PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh crash
```

### Open the latest training report PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh training
```

### Open the latest paper performance PDF
Build it first if it is missing.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh paper
```

### Open the latest sentiment report PDF
Build it first if it is missing.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh sentiment
```

### Open the market correlation overlap PDF
Build the PDF bundle first if it is missing.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh correlation
```

### Open the project timeline PDF
Build it first if it is missing.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh timeline
```

### Open the project timeline printable report
Use this if the PDF renderer is unavailable for any reason.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh timeline-report --json
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/project_timeline/project_timeline_print_latest.html
```

### Build the main send-out packet first
Use this before you send reports so the latest PDFs are already rendered.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh sendout
```

### Print the ready-to-send packet path without opening it
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh --print-only sendout
```

## Retrain

### Standard retrain
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh retrain
```

## Strategy Research

### Full strategy research lane refresh
Runs attribution, counterfactual replay, the research sandbox, and promotion readiness into one canonical health artifact at `governance/health/strategy_research_latest.json`.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh strategy-research --day "$(date -u +%Y%m%d)" --json
```

### Fast strategy research snapshot
Use this during the day when you want the latest summary without rerunning the heavier research sandbox refresh. This now honors freshness TTLs for the lighter artifacts too.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh strategy-research --day "$(date -u +%Y%m%d)" --skip-sandbox --max-age-minutes 90 --sandbox-max-age-minutes 720 --json
```

### Unified derived-state snapshot
Roll allocator, portfolio risk, and execution budgets into one small canonical artifact at `governance/health/derived_state_latest.json`.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh derived-state --json
```

### Cold-lane research refresh
Use this for the heavier background pass. It checks resource guard, skips fresh artifacts, and only reruns the full strategy research lane when the summary is stale.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh cold-lane-refresh --day "$(date -u +%Y%m%d)" --strategy-max-age-minutes 180 --sandbox-max-age-minutes 720 --json
```

## Infrastructure Lanes

### Ops coordinator lane
Runs the lightweight ops-control sweep outside the trading registry. It refreshes the watchdog, derived-state snapshot, fast strategy research view, and the platform control-plane summary into one artifact at `governance/health/ops_coordinator_latest.json`.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh ops-coordinator --day "$(date -u +%Y%m%d)" --strategy-max-age-minutes 90 --sandbox-max-age-minutes 720 --json
```

### Storage maintenance lane
Runs guarded storage route sync, shard maintenance, SQLite checkpointing, the stale-artifact sweeper bot, the stale-artifact reaper bot, and retention cleanup into one artifact at `governance/health/storage_maintenance_latest.json`.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-maintenance --json
```

### Stage stale artifacts into the stale holding area
Use this when you want the dedicated infrastructure sweeper bot to move stale retention candidates into `data/stale_stage` without hard-deleting them yet.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh stale-sweeper --stale-stage-sections all --json
```

Expected useful signal: `summary.staged_files` should climb while `deleted_files` stays `0`, and the staged audit trail lands in `data/stale_stage/stale_manifest.jsonl`.

### Reap aged stale-stage artifacts
Use this when you want the dedicated deletion bot to purge files that have already been sitting in `data/stale_stage` longer than the configured review window.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh stale-reaper --stale-purge-days 30 --json
```

Expected useful signal: `summary.deleted_files` tells you how many already-staged artifacts were actually removed during the pass.

### Deep storage maintenance pass
Use this when you explicitly want the lane to run a heavier SQLite vacuum instead of checkpoint-only maintenance.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-maintenance --vacuum --json
```

## Python Runtime

### Audit Python 3.14 shadow readiness
Use this before any cutover. It now checks lock drift, MLX package presence, pytest availability, and a real import of the training stack.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
BOT_RUNTIME_LANE=shadow314 ./scripts/ops/opsctl.sh py314-canary --skip-install --json
```

Expected healthy result: `ok=true` with successful `mlx_core_import`, `mlx_lm_import`, `pytest_import`, and `indicator_bot_common_import`.

### Rebuild or resync the Python 3.14 shadow lane
Use this after the current retrain is finished, or any time you want to rebuild `.venv314` from the lockfile path and then rerun the readiness audit.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
BOT_RUNTIME_LANE=shadow314 ./scripts/ops/opsctl.sh py314-canary --refresh-deps --json
```

### Audit SQL access runtime
Checks the installed SQL access stack against the lock, runs `pip check`, and smoke-tests both DuckDB-through-SQLAlchemy and ADBC SQLite support.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh sql-audit --json
```

Expected healthy result: `ok=true` with successful `duckdb_sqlalchemy_smoke` and `adbc_sqlite_smoke`.

### Audit registry-wide training readiness
Builds a single view of which registry bots are sample-starved, quality-failing, still brand-new, or missing diagnostics so you can separate shared runtime-input problems from bot-specific labeling/filter problems.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh training-registry-audit --json
```

Expected useful signal: active bots with `inferred_cause=shared_runtime_input_gap` point to shared snapshot/loader coverage issues, while `quality_guard_failure` points to bot-specific label/threshold work.

### Audit label quality and abstention behavior
Use this after the registry audit when the issue looks like over-filtering, under-filtering, side imbalance, or runaway acted coverage instead of a shared runtime snapshot gap.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh training-label-audit --json
```

Expected useful signal: `top_actions` should tell you whether the next fix is `relax_sample_filter`, `relax_confidence_gate`, `rebalance_label_builder`, or `tighten_abstention_thresholds`.

### Build the unified training-quality control view
Use this after the registry audit and label audit when you want one canonical artifact that scores supportability, lane balance, symbol concentration, promotion coverage, ingestion health, and the targeted retrain shortlist in one place.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh training-quality --json
```

Expected useful signal: `overall_status`, `training_quality_score`, and `top_priorities` should tell you whether the next move is fixing runtime inputs, refreshing stale diagnostics, rebalancing dominant lanes, or isolating probation bots.

### Build the canonical feature-store manifest
Use this when you want one point-in-time contract for runtime training rows, event joins, feature hashes, and lane partitions instead of treating lineage as a loose collection of artifacts.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feature-store --json
```

Expected useful signal: `overall_status`, `dataset_contract.rows_sha256`, `point_in_time_contract.dataset_join_keys`, and `lane_partitions` should tell you whether lineage is clean enough for large retrains and replay.

### Build the multiple-testing guard
Use this after replay feature ablation or counterfactual searches so experimentation stays attached to an explicit hypothesis family, correction method, and regime segmentation.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh multiple-testing-guard --json
```

Expected useful signal: `family_size`, `correction_method`, `corrected_alpha`, and `regime_segments` should tell you whether research batches are disciplined enough to compare fairly.

### Build the decay monitor
Use this to turn paper and replay outcomes into a direct training signal instead of waiting for weak sleeves to silently accumulate.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh decay-monitor --json
```

Expected useful signal: `overall_status`, `weak_sleeve_count`, `pnl_slope`, and `trailing_periods` should tell you whether a sleeve needs targeted retrain, probation, or retirement work.

### Build the ingestion and storage control plane
Use this when backlog, retention debt, or stale-stage pressure feels like it is dragging down both ops and training quality. It estimates drain time, throughput, and top remediation actions in one place.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh ingestion-storage-control --json
```

Expected useful signal: `severity`, `pressure_index`, `estimated_core_drain_minutes`, `estimated_total_drain_minutes`, and `retention_debt_gb` should tell you whether to retrain, throttle, split shards, or stay in maintenance mode.

### Build or apply the memory-efficiency profile
Use this when you want the system to react more intelligently to Apple Silicon memory pressure, swap growth, and storage pressure instead of relying on static defaults alone.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh memory-efficiency status --json
./scripts/ops/opsctl.sh memory-efficiency apply --json
```

Expected useful signal: `recommended_profile`, `memory_snapshot`, and `recommended_env_overrides` should tell you whether to stay at full throughput, drop to an air-safe posture, or force a constrained profile until pressure clears.

### Audit model lifecycle hygiene
Use this to see whether active bots still have fresh training diagnostics, valid model/log artifact paths, and whether any active entries should be downgraded out of production until they are supportable again.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh model-lifecycle --json
```

Expected useful signal: `stale_active_training_diagnostics=0` for a healthy active set. If this is nonzero, refresh or downgrade those bots before trusting full-registry training conclusions.

### Morning control-plane sweep
Runs the lightweight operator pass that refreshes watchdog health, derived state, fast strategy research, registry audits, label audits, and the control-plane summary in one shot.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh ops-coordinator --day "$(date -u +%Y%m%d)" --strategy-max-age-minutes 90 --sandbox-max-age-minutes 720 --json
```

Expected useful signal: one artifact at `governance/health/ops_coordinator_latest.json` with active sample-starved counts, stale-diagnostic counts, recommended action, and pending-line pressure.

### Audit security hardening
Use this to verify RBAC, pre-commit secret scanning, paper/live separation, audit journals, and backup evidence before you trust a live expansion or a promotion packet.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh security-audit
```

Expected useful signal: `overall_status`, failed check names, `secret_scan_age_hours`, and `rbac_role_count` should tell you whether the current operator surface is actually hardened.

### Run the secret scan directly
Use this for quick repo-wide or staged-file credential hygiene checks.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh secret-scan --staged
```

Expected healthy result: `findings_count=0`.

### Institutional-readiness control plane
Use this when you want one explicit snapshot of how close the repo is to institutional-grade across the big domains: point-in-time lineage, immutable experiments, simulator fidelity, portfolio/risk layers, TCA/capacity, research discipline, model governance, security, reliability, observability, and developer process.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-control-plane --max-rows 4000 --json
```

Expected useful signal: `institutional_readiness.overall_score`, `overall_status`, `weakest_domains`, and `top_priorities` should tell you which structural upgrades matter most next instead of just whether day-to-day ops are green.

### Build the schema and migration manifest
Use this when you want one explicit inventory of operator-facing contracts, their schema versions, and which artifacts are still legacy or missing versioning.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schema-migration --json
```

Expected useful signal: `summary.missing_contracts`, `summary.legacy_unversioned_contracts`, and `contracts[*].compatibility` should tell you where schema discipline still needs tightening.

## Access Mode

### Enable portable access mode
Use this before exporting or handing the repo to someone else. It keeps storage project-local, advertises the portable SQL path, and leaves your current native defaults untouched until you flip it on.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh access-portable
```

### Restore native access mode
Use this on your Mac to go back to the current native behavior with your normal storage and runtime defaults.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh access-native
```

### Show the current access mode
This reports whether the runtime is currently in `native` or `portable` mode and which override file is controlling it.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh access-status
```

Expected healthy result: `runtime_access_mode mode=native` on your Mac by default, or `mode=portable` after you flip the export-friendly switch.

## Brain Switching

These are the friendly wrappers around the backend switch layer. They are best used for portable or shadow workflows, not as a replacement for the live MLX trading brain.

### Show the current brain/backend contract
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh brain-status --json
```

Expected useful signal: `backend_contract` shows which backend is active, which packages are installed, and whether the selected backend is live-trading capable or observation-only.

### Switch to automatic portable brain selection
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh brain-switch portable_auto --json
```

### Pin the brain switch to MLX
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh brain-switch mlx --json
```

### Pin the brain switch to PyTorch
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh brain-switch pytorch --json
```

### Pin the brain switch to ONNX
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh brain-switch onnx --json
```

### Pin the brain switch to TensorFlow
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh brain-switch tensorflow --json
```

### Pin the brain switch to JAX
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh brain-switch jax --json
```

### Restore the native MLX-default runtime
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh brain-native --json
```

### Force full retrain
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh retrain-force-full
```

### Force targeted retrain
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh retrain-force-targeted --include-bot-ids brain_refinery_v56_meta_ranker
```

### Regime validation
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh regime-validate
```

## Institutional Upgrade Lanes

These are the operator commands for the bigger platform-hardening work: queue-backed ingestion, split-brain reconciliation, content-addressed artifacts, training requalification, portfolio/risk services, execution lab work, and the new daily-verify remediation bot.

### Build the durable prioritized ingestion queue
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh ingestion-priority-queue --json
```

### Apply the ingestion/storage governor
Use this when you want the queue and storage lane to clamp deferred and cold pressure, normalize the SQL primary path, and publish the current pressure profile.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh ingestion-storage-governor apply --json
```

### Drain the external deferred and cold backlog
Use this during off-hours to raise deferred and cold drain budgets temporarily, sweep stale artifacts, and push down the external BOT_LOGS backlog without relaxing the live-time governor.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh external-backlog-drain --apply --json
```

If the SQL writer is already busy and you want the drain to keep trying automatically until it can take over, use follow-through mode.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh external-backlog-drain --apply --follow-through --wait-timeout-seconds 900 --json
```

### Run the background backlog retry bot manually
Use this if you want the infrastructure bot to decide whether the drain is actionable and then launch a follow-through pass automatically.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh external-backlog-retry-bot --apply --wait-timeout-seconds 900 --json
```

### Quarantine stale prior-day backlog during market hours
Use this to stage oversized prior-day `shadow_pnl_attribution` and explanation files into `data/stale_stage` so they stop competing with the live ingestion path before the next off-hours drain window.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh backlog-quarantine --apply --json
```

### Install the hands-off backlog retry bot
This is part of the ops automation launchd stack. Reinstalling the ops automations will register the new background retry bot and kick it off immediately.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/install_ops_automation_launchd.sh
```

### Materialize the content-addressed artifact store
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh content-store --json
```

### Review BOT_LOGS split-brain conflicts
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh split-brain-reconcile --json
```

### Check storage resilience and standby readiness
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-resilience --json
```

### Build the training requalification lane
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh training-requalification --json
```

### Seed walk-forward coverage continuously
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coverage-seed --json
```

### Publish calibration and abstention controls
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh calibration-control --json
```

### Net sleeve intents through the portfolio allocator
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh portfolio-allocator --json
```

### Show the separate risk-service boundary
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh risk-service --json
```

### Run the execution research lab
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh execution-lab --json
```

### Open the operator cockpit
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh operator-cockpit --json
```

### Run the daily-verify auto-remediation infrastructure bot
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh daily-verify-remediation --apply --json
```

## Schwab Auth

### Refresh the Schwab token / handshake
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh --always-auth
```

Expected healthy result: `premarket_token_guard ok=1`

### Interactive Schwab re-auth
Use this if the normal token refresh does not recover the handshake cleanly.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh-interactive
```

Expected healthy result: `Handshake Successful.`

### Token refresh plus Schwab feed restart
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh --always-auth
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

## SQL And One Numbers

### Quick SQL sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh sql-sync
```

### Full SQL sync + SQLite maintenance + One Numbers refresh
If `sql-sync --json` returns `{"ok": false, "reason": "writer_lock_busy", ...}`, that means the background `sql_link_shard_manager` already owns the writer lock and is actively syncing. That is expected, so continue with the next step instead of rerunning `sql-sync`.

```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh sql-sync --json
cat governance/health/sql_link_service_progress_latest.json | rg '"status"|"current_step"|"completed_shard_count"|"completed_merge_count"|"merged_rows_this_cycle"'
./scripts/ops/opsctl.sh sql-maint --json
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/build_one_numbers_report.py
```

`sql-maint` now skips auto-vacuum in the manual wrapper so this path stays usable for quick refreshes. Use `./scripts/ops/opsctl.sh sql-maint --json --vacuum` only when you intentionally want a full vacuum.

`build_one_numbers_report.py` now prints startup progress immediately. If the terminal stays quiet for a while, check the active worker list:

```bash
ps -axo pid,etime,command | rg "build_one_numbers_report.py|sqlite_performance_maintenance.py|sql_link_shard_manager.py"
```

### Daily full refresh pipeline
Use this when you want the broader daily refresh instead of only SQL.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/daily_log_refresh.sh
```

### Open the latest One Numbers CSV
`latest.csv` is only a symlink to the most recently built One Numbers export. The report now auto-resolves to the most recent linked session day with data and includes day, month-to-date, and all-time rollups in the same document. If it still looks stale, run the SQL / One Numbers refresh command above first.
```bash
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/one_numbers/latest.csv
```

### Open the latest One Numbers markdown summary
```bash
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/one_numbers/latest.md
```

## Start / Stop

### Start the full live stack
This is the canonical start command. It defaults to the live profile, enables the watchdog-backed stack, and brings up the sleeves loop, futures, live feeds, and paper-mode defaults.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh start
```

### Force-restart the live stack in place
Use this when the stack is already running and you want `opsctl` to restart the live stack cleanly instead of only refreshing feeds.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh start --force-restart
```

### Stop the stack
This stops the loop processes and disables the stack auto-restart launchd jobs so they do not come right back up.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh stop
```

## Status And Health

### Runtime status
Shows the watchdog, sleeves, dividend/bond sleeves, execution lanes, futures, FX, and SQL writer processes.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh status
```

### Health snapshot
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh health
```

### Doctor
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh doctor
```

### Current halt recovery state
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
cat governance/health/shadow_watchdog_halt_recovery_latest.json
```

### Current global trading halt flag
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
cat governance/health/GLOBAL_TRADING_HALT.flag
```

### Set the global trading halt
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/operator_control.py --set-global-halt --reason "manual_operator_halt" --json
```

### Clear the global trading halt
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/operator_control.py --clear-global-halt --json
```

## Storage Routing

### Switch collection to the Mac's internal drive
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-switch-local
```

### Watch external drive compaction progress
```bash
df -h /Volumes/BOT_LOGS
ls -lh /Volumes/BOT_LOGS/schwab_trading_bot/data/jsonl_link_archives/jsonl_link_archive_2026_03.compact.sqlite3*
ps -p 52563 -o pid=,etime=,command=
```

### Safe-eject the external BOT_LOGS drive
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-safe-eject
```

### Switch collection back to the external drive
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-switch-external
```

## BOT_LOGS Hygiene

`/Users/dankingsley/PycharmProjects/schwab_trading_bot/{data,logs,exports,governance,decisions,decision_explanations,models}` are symlinked to `/Volumes/BOT_LOGS/schwab_trading_bot`, while `local_fallback_storage` stays on the Mac's internal drive. Safe purge work here means Finder junk, stale `.local_fallback*` conflict copies, retention-managed stale artifacts, and empty fallback mirror directories. Do not blindly delete `local_fallback_storage/data/*.sqlite3`: `bot_channel_queue.sqlite3` is the active queue DB and the other SQLite files are the fallback cutover copies if the external drive disappears.

### Inspect the active storage route and fallback footprint
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
cat governance/health/storage_failback_sync_latest.json
du -sh /Volumes/BOT_LOGS/schwab_trading_bot/{data,decisions,decision_explanations,exports,governance,logs,models}
du -sh /Users/dankingsley/PycharmProjects/schwab_trading_bot/local_fallback_storage/data
ls -lh /Users/dankingsley/PycharmProjects/schwab_trading_bot/local_fallback_storage/data/*.sqlite3
```

### Preview safe BOT_LOGS purge candidates
```bash
find /Volumes/BOT_LOGS/schwab_trading_bot \
  \( -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' -o -name '.hypothesis' \) \
  -o -type f \( -name '.DS_Store' -o -name '*.pyc' -o -name '*.pyo' \) \) \
  | sort
find /Volumes/BOT_LOGS/schwab_trading_bot -type f -name '*.local_fallback*' -mtime +1 | sort
```

### Purge safe BOT_LOGS clutter
```bash
find /Volumes/BOT_LOGS/schwab_trading_bot -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' -o -name '.hypothesis' \) -prune -exec rm -rf {} +
find /Volumes/BOT_LOGS/schwab_trading_bot -type f \( -name '.DS_Store' -o -name '*.pyc' -o -name '*.pyo' \) -delete
find /Volumes/BOT_LOGS/schwab_trading_bot -type f -name '*.local_fallback*' -mtime +1 -delete
```

### Preview overdue raw BOT_LOGS files that already have `.gz` archives
These raw files are past the normal log-maintenance window and already have archived `.gz` siblings, so the raw copy is just extra space pressure.
```bash
find /Volumes/BOT_LOGS/schwab_trading_bot/{decisions,decision_explanations,governance,exports} \
  -type f \( -name '*.jsonl' -o -name '*.log' \) -mtime +0 ! -name '*.gz' -print0 \
  | while IFS= read -r -d '' path; do
      [ -e "${path}.gz" ] && printf '%s\n' "$path"
    done | sort
```

### Purge overdue raw BOT_LOGS files that already have `.gz` archives
Use this only after you preview the list. It keeps the `.gz` archives and removes the redundant raw copies.
```bash
find /Volumes/BOT_LOGS/schwab_trading_bot/{decisions,decision_explanations,governance,exports} \
  -type f \( -name '*.jsonl' -o -name '*.log' \) -mtime +0 ! -name '*.gz' -print0 \
  | while IFS= read -r -d '' path; do
      [ -e "${path}.gz" ] && rm -f "$path"
    done
```

### Inspect shard corrupt-quarantine storage
```bash
du -sh /Volumes/BOT_LOGS/schwab_trading_bot/data/sql_link_shards/corrupt_quarantine
find /Volumes/BOT_LOGS/schwab_trading_bot/data/sql_link_shards/corrupt_quarantine -maxdepth 3 | sort
```

### Purge a reviewed corrupt-quarantine shard copy
```bash
rm -rf /Volumes/BOT_LOGS/schwab_trading_bot/data/sql_link_shards/corrupt_quarantine/crypto_trading_20260331T145531Z
```

### Preview empty internal fallback mirror directories
```bash
find -P /Users/dankingsley/PycharmProjects/schwab_trading_bot/local_fallback_storage -type d -empty | sort
```

### Purge empty internal fallback mirror directories
```bash
find -P /Users/dankingsley/PycharmProjects/schwab_trading_bot/local_fallback_storage -depth -type d -empty -delete
```

### Preview retention-managed stale BOT_LOGS artifacts
This dry-run shows old logs, governance files, debug snapshots, exports, and other retention targets without deleting anything.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/data_retention_policy.py --json
```

### Stage stale BOT_LOGS artifacts into the stale holding area
This uses the sweeper bot to move stale candidates into `data/stale_stage` first, so the deletion process has a review buffer and manifest trail.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh stale-sweeper --stale-stage-sections all --json
```

### Purge aged files that are already sitting in stale_stage
This only deletes artifacts that have already been staged and aged out of the review buffer.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh stale-reaper --stale-purge-days 30 --json
```

### Apply retention-managed BOT_LOGS cleanup
This uses the repo's configured retention windows for logs, governance artifacts, debug snapshots, reports, and archive pruning.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/data_retention_policy.py --apply --json
```

## Workspace Hygiene

`/Users/dankingsley/Documents/schwab_trading_bot` is already a Finder-visible wrapper for the canonical repo at `/Users/dankingsley/PycharmProjects/schwab_trading_bot`. The repo's heavy runtime trees now live under `BOT_LOGS`, and `local_fallback_storage` is the internal safety net, so the commands below stay focused on leftover `Documents/New project` cache, archive, and merge artifacts while leaving the current `one_numbers*` work and `organize_lacie_photos.py` alone.

### Verify the Documents wrapper folder
```bash
ls -la /Users/dankingsley/Documents/schwab_trading_bot
```

### Preview the current New project top level
```bash
find "/Users/dankingsley/Documents/New project" -maxdepth 1 -mindepth 1 | sort
```

### Preview stale workspace artifacts in Documents/New project
```bash
find "/Users/dankingsley/Documents/New project" \
  \( -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '_schwab_prepurge_*' \) -o -type f -name '.DS_Store' \) \
  | sort
```

### Purge stale workspace artifacts in Documents/New project
```bash
find "/Users/dankingsley/Documents/New project" -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '_schwab_prepurge_*' \) -prune -exec rm -rf {} +
find "/Users/dankingsley/Documents/New project" -type f -name '.DS_Store' -delete
```

### Preview stale repo artifacts inside the project
This excludes virtualenv folders so you only see repo-owned cache and test artifacts.
```bash
PROJECT_ROOT="/Users/dankingsley/PycharmProjects/schwab_trading_bot"
find "$PROJECT_ROOT" \
  \( -path "$PROJECT_ROOT/.venv*" -o -path "$PROJECT_ROOT/.git" -o -path "$PROJECT_ROOT/.git/*" \) -prune \
  -o \( -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' -o -name '.hypothesis' -o -name 'htmlcov' \) -o -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '.coverage' -o -name '.DS_Store' \) \) \
  -print | sort
```

### Purge stale repo artifacts inside the project
This excludes virtualenv folders so you only remove repo-owned cache and test artifacts.
```bash
PROJECT_ROOT="/Users/dankingsley/PycharmProjects/schwab_trading_bot"
find "$PROJECT_ROOT" \
  \( -path "$PROJECT_ROOT/.venv*" -o -path "$PROJECT_ROOT/.git" -o -path "$PROJECT_ROOT/.git/*" \) -prune \
  -o -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' -o -name '.hypothesis' -o -name 'htmlcov' \) -prune -exec rm -rf {} +
find "$PROJECT_ROOT" \
  \( -path "$PROJECT_ROOT/.venv*" -o -path "$PROJECT_ROOT/.git" -o -path "$PROJECT_ROOT/.git/*" \) -prune \
  -o -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '.coverage' -o -name '.DS_Store' \) -delete
```

### Preview legacy repo-side merge artifacts
```bash
find /Users/dankingsley/PycharmProjects/schwab_trading_bot/docs -maxdepth 3 \( -name 'documents_merge' -o -name 'new_project_schwab_workspace' \) | sort
```

### Purge the legacy repo-side merge archive
```bash
rm -rf /Users/dankingsley/PycharmProjects/schwab_trading_bot/docs/documents_merge/new_project_schwab_workspace
```

### Preview legacy Schwab workspace-folder purge status
Expected current result: `eligible_for_archive` is empty because those old staging folders are already gone.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/ops/documents_workspace_purge.py plan --json
```

### Purge all legacy prepurge archives created by the helper
Dry-run is the default. Add `--execute` only after you verify the archive list.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/ops/documents_workspace_purge.py purge --json
```

### Purge only the latest legacy prepurge archive
Dry-run is the default. Add `--execute` only after you verify the archive list.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/ops/documents_workspace_purge.py purge --latest-only --json
```
