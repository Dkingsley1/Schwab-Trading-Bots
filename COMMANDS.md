# Commands (Canonical)

Use these exact commands as the current source of truth.

This file is generated from the curated operator inventory in `scripts/ops/commands_hygiene_bot.py`.
Rebuild it with `./scripts/ops/opsctl.sh commands-hygiene --apply` after changing that inventory.

This file is intentionally trimmed down:
- paper mode is the operating default
- no simulate variants are listed
- no duplicate restart commands are listed when a broader command already covers them

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

### Brain switch: launch the mode switchboard
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
SWITCHBOARD_MODES="shadow,paper" "$PY" scripts/run_mode_switchboard.py
```

Valid modes are `shadow`, `paper`, and `live`.
This launches one `main.py` child per mode and sets `BOT_MODE` automatically.

### Phone mirror view for the live feed
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh phone-feed --host 0.0.0.0 --source all --include-decisions
```

This starts the phone-friendly live feed mirror and prints the local and Tailscale URLs in the terminal.
When `--host 0.0.0.0` is used without `--token`, the server auto-generates a remote-access token for you.

### Open the One Numbers CSV in Numbers
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/one_numbers/latest.csv
```

### Open the One Numbers PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/one_numbers/one_numbers_latest.pdf
```

### Broker Truth Step 1: refresh Schwab auth
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh --always-auth
```

Use this first when broker-truth lanes start showing transient 403s or auth churn.

### Broker Truth Step 2: restart the Schwab loops
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

This forces the Schwab sleeves to pick up the refreshed token and republish their latest broker-truth snapshots.

### Broker Truth Step 3: verify broker readiness and lane statuses
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
/Users/dankingsley/PycharmProjects/schwab_trading_bot/.venv312/bin/python -c "from pathlib import Path; import json; root=Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot/governance/health'); broker=json.loads((root/'broker_readiness_latest.json').read_text()); print(f'ready_for_open={broker.get(\"ready_for_open\")} auth_ok={broker.get(\"auth_ok\")} token_warning_level={broker.get(\"token_warning_level\")}'); print('lane,status,mismatch_count,error'); [print(f'{p.name.replace(\"broker_truth_\", \"\").replace(\"_latest.json\", \"\")},{json.loads(p.read_text()).get(\"status\", \"\")},{int(json.loads(p.read_text()).get(\"mismatch_count\", 0) or 0)},{json.loads(p.read_text()).get(\"error\") or \"\"}') for p in sorted(root.glob('broker_truth_*_latest.json')) if 'shared_snapshot' not in p.name]"
```

Healthy target: `ready_for_open=True`, `auth_ok=True`, and all Schwab broker-truth lanes reporting `status=ok` with `mismatch_count=0`.

### Refresh the live loops without reinstalling the stack watchdog
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source all
```

`feed-refresh` is a live-loop restart helper, not a passive data-context sync. It kills and restarts the relevant market-data loops. If you want a full supervised stack refresh instead of a feed-loop refresh, use `./scripts/ops/opsctl.sh start --force-restart`.

### Stop the stack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh stop
```

### Validate documented commands
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh command-validity --json
```

## Storage

### Switch collection to the Mac's internal drive
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-switch-local
```

### Switch collection back to the external BOT_LOGS drive
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-switch-external
```

### Safe-eject the external BOT_LOGS drive
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-safe-eject
```

## Live Feed Refreshes

### Refresh Schwab equities, Schwab futures, and FX
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

### Refresh Coinbase spot and Coinbase futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source coinbase
```

### Refresh FX only
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source fx
```

## Live Feed Views

### Heavy live feed view across all feeds and decisions
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source all --include-decisions
```

Use this as the primary all-feeds operator view when you want the broad multi-feed tail plus decision-stream context in one window.
If the Mac is running an `air_safe` or `constrained` memory-efficiency profile, the feed automatically trims decision fanout and uses a lower default line budget unless you pass your own `--lines` or `--no-memory-aware`.

### Light live feed tail for all feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source all --lines 80
```

### Live feed tail for Schwab, Coinbase, and futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh main-tail --lines 80
```

### Live feed tail for Schwab
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-tail --lines 80
```

### Live feed tail for Coinbase
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-tail --lines 80
```

### Live feed tail for all futures sleeves
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh futures-tail --lines 80
```

### Live feed tail for Schwab futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-futures-tail --lines 80
```

### Live feed tail for Coinbase futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-futures-tail --lines 80
```

### Live feed tail for FX
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh fx-tail --lines 80
```

## Schwab Auth

Use these exact Schwab authorization commands when tokens expire, browser consent needs renewal, or broker-truth lanes start surfacing 401/403 errors.

### Schwab authorization refresh
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh --always-auth
```

Use this when the Schwab browser grant is stale or broker-truth lanes start showing auth churn.

### Interactive Schwab authorization re-consent
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh-interactive
```

Run this when you want to force the browser-based Schwab authorization flow directly.

### Schwab auth recovery plus lane restart
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh --always-auth
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

This is the paste-ready recovery pair when refreshed authorization needs to be picked up by the Schwab loops immediately.

## Status And Health

### Runtime status
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

## SQL And Reports

### Full SQL refresh pipeline
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/daily_log_refresh.sh
```

Use this when you want the full SQL/log/report refresh instead of the one-pass writer sync.

### Quick SQL sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh sql-sync
```

### Data quality refresh bundle
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source all
./scripts/daily_log_refresh.sh
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/build_one_numbers_report.py
```

Use this when One Numbers is stale or you want the latest data-quality averages and report artifacts refreshed together.

## Retrain

Use these commands when you are preparing or launching a manual retrain cycle.

### Full retrain preflight
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/daily_log_refresh.sh
./scripts/ops/opsctl.sh runtime-training-snapshot --json
./scripts/ops/opsctl.sh coverage-seed --write-queue --json
./scripts/ops/opsctl.sh coverage-gap-closer --apply-stage --launch --json
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/retrain_schema_compatibility_guard.py --json
"$PY" scripts/promotion_quality_gate.py --json
```

Run this before a manual full retrain so SQL state, runtime snapshots, coverage, and promotion gates are fresh.

### Guarded retrain orchestrator
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh retrain-orchestrate --json
```

This is the safer manual retrain entrypoint because it refreshes stale artifacts and honors freshness checks before launching weekly retrain.

### Force full retrain (bypass prechecks)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh retrain-force-full
```

Use this only when you intentionally want to bypass the normal data-quality, freshness, snapshot-sync, and sample-quota prechecks.

## Reports And PDFs

This section includes the generate commands plus direct open commands for each report PDF.

### One Numbers report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/build_one_numbers_report.py
```

### Paper performance report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh paper-performance --day "$(date -u +%Y%m%d)" --week-days 7 --json
```

### Report catalog bundle
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh report-pdfs --json
```

### Active bot stack PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh bot-stack-report --top 25 --render-pdf --allow-gui-pdf-renderer
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/bot_stack_status/latest.pdf`.

### Open the report catalog PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/report_pdf_bundle_latest.pdf
```

### Open the daily ops PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/daily_ops_report_latest.pdf
```

### Open the paper performance PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/paper_performance_latest.pdf
```

### Open the sentiment PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/sentiment_report_latest.pdf
```

### Open the strategy attribution PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/strategy_attribution_latest.pdf
```

### Open the post-trade analysis PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh posttrade
```

This refreshes the post-trade analysis source, renders the report PDF bundle, prefers the PDF artifact, and falls back to printable HTML or markdown if the PDF renderer is unavailable.

### Open the crash digest PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh crash
```

This regenerates the crash digest with a 30-day lookback by default, prefers the PDF artifact, and falls back to printable HTML if the PDF renderer is unavailable.

### Open the project timeline PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh timeline
```

This regenerates the timeline report, prefers the PDF artifact, and falls back to printable HTML if the PDF renderer is unavailable.

### Open the training report PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/training_reports/training_report_latest.pdf
```

### Open the macro crosscheck PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/macro_crosscheck_latest.pdf
```

### Open the market correlation PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/market_crypto_correlation_latest.pdf
```

### Open the source verification PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/source_verification_latest.pdf
```

### Open the retrain scorecard PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/retrain_scorecard_latest.pdf
```

### Open the daily runtime summary PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/daily_runtime_summary_latest.pdf
```

### Open the daily auto verify PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/daily_auto_verify_latest.pdf
```

### Open the model card PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/model_card_latest.pdf
```

### Open the paper execution calibration PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/paper_execution_calibration_latest.pdf
```

### Open the replay feature ablation PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/replay_feature_ablation_latest.pdf
```

### Open the one numbers PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/one_numbers/one_numbers_latest.pdf
```

### Open the state snapshot drills PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/state_snapshot_drills/state_snapshot_drills_latest.pdf
```

### Open the active bot stack PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh botstack
```

This refreshes the bot stack report, prefers the PDF artifact, and falls back to HTML or markdown if the PDF renderer is unavailable.

### Open the unified lane scorecard PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/unified_lane_scorecard_latest.pdf
```

This one is on-demand and only exists after it has been generated.

### Open the bot explainability PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/bot_explainability_latest.pdf
```

This one is on-demand and only exists after it has been generated.

## Data Context Syncs

### Options flow context sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh options-flow-sync --json
```

`options-flow-sync` is the canonical command. `tastytrade-sync` remains a legacy alias for backward compatibility.

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

### Source verification
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh source-verification --json
```

## Macro And Media

### Start the macro auto-watch lane
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-auto-start --force-restart --youtube-channel-url "https://www.youtube.com/@federalreserve" --template fed --speaker "Federal Reserve" --source "Federal Reserve"
```

### Show macro auto-watch status
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-auto-status --json
```
