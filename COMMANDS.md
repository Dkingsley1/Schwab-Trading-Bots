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

### Start or refresh the full live stack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source all
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

### Open the latest paper performance PDF
```bash
./scripts/ops/open_report_artifact.sh paper
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

## Python Runtime

### Audit Python 3.14 shadow readiness
Use this before any cutover. It now checks lock drift, MLX package presence, pytest availability, and a real import of the training stack.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
BOT_RUNTIME_LANE=shadow314 ./scripts/ops/opsctl.sh py314-canary --skip-install --json
```

### Rebuild or resync the Python 3.14 shadow lane
Use this after the current retrain is finished, or any time you want to rebuild `.venv314` from the lockfile path and then rerun the readiness audit.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
BOT_RUNTIME_LANE=shadow314 ./scripts/ops/opsctl.sh py314-canary --refresh-deps --json
```

Expected healthy result: `ok=true` with successful `mlx_core_import`, `mlx_lm_import`, `pytest_import`, and `indicator_bot_common_import`.

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

### Start or refresh the full live stack
This is the main stack command. It covers the full sleeves loop, live feeds, and paper-mode defaults.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source all
```

### Stop the stack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh stop
```

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

## Workspace Merge

`/Users/dankingsley/Documents/schwab_trading_bot` is already a Finder-visible wrapper for the canonical repo at `/Users/dankingsley/PycharmProjects/schwab_trading_bot`. The commands below are for merging the Schwab-specific workspace pieces from `/Users/dankingsley/Documents/New project` into the repo archive path while leaving unrelated `one_numbers*` work and `organize_lacie_photos.py` alone.

### Verify the Documents wrapper folder
```bash
ls -la /Users/dankingsley/Documents/schwab_trading_bot
```

### Preview the automated workspace purge plan
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/ops/documents_workspace_purge.py plan --json
```

### Automatically archive the old Schwab workspace folders out of New project
Dry-run is the default. Add `--execute` after you review the plan.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/ops/documents_workspace_purge.py archive --json
```

### Automatically purge the archived Schwab workspace folders
Dry-run is the default. Add `--execute` only after you verify the archive contents.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/ops/documents_workspace_purge.py purge --latest-only --json
```

### Preview the merged Schwab workspace archive
```bash
find /Users/dankingsley/PycharmProjects/schwab_trading_bot/docs/documents_merge/new_project_schwab_workspace -maxdepth 2 | sort | sed -n '1,200p'
```

### Re-sync the Schwab workspace from Documents/New project
```bash
SRC="/Users/dankingsley/Documents/New project"
DEST="/Users/dankingsley/PycharmProjects/schwab_trading_bot/docs/documents_merge/new_project_schwab_workspace"
items=(
  balloon_fix
  commands_alpha
  commands_most_used
  commands_remove_macro
  commands_reports_pdf
  commands_reports_timeline
  fx_patch
  stage
  stage_commands_md
  stage_dividend_paper
  stage_export_alias
  stage_external_csv
  stage_futures_tail
  stage_fx_context_launchd
  stage_fx_gate
  stage_fx_quotes
  stage_fx_session
  stage_fx_tail
  stage_fx_twelve
  stage_paper_report_all_sleeves
  stage_reports
  stage_retrain_note
  tmp_final3
  tmp_fix
  tmp_fix2
  tmp_fix_scripts
  tmp_memory_guard_patch
  tmp_reconcile_patch
  tmp_repo_patch
  tmp_resource_guard_patch
  tmp_storage_patch
)
for item in "${items[@]}"; do
  rsync -a "$SRC/$item" "$DEST/"
done
```

### Stage 1: move the old Schwab workspace folders out of New project
```bash
SRC="/Users/dankingsley/Documents/New project"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$SRC/_schwab_prepurge_$STAMP"
items=(
  balloon_fix
  commands_alpha
  commands_most_used
  commands_remove_macro
  commands_reports_pdf
  commands_reports_timeline
  fx_patch
  stage
  stage_commands_md
  stage_dividend_paper
  stage_export_alias
  stage_external_csv
  stage_futures_tail
  stage_fx_context_launchd
  stage_fx_gate
  stage_fx_quotes
  stage_fx_session
  stage_fx_tail
  stage_fx_twelve
  stage_paper_report_all_sleeves
  stage_reports
  stage_retrain_note
  tmp_final3
  tmp_fix
  tmp_fix2
  tmp_fix_scripts
  tmp_memory_guard_patch
  tmp_reconcile_patch
  tmp_repo_patch
  tmp_resource_guard_patch
  tmp_storage_patch
)
mkdir -p "$ARCHIVE"
for item in "${items[@]}"; do
  mv "$SRC/$item" "$ARCHIVE/"
done
printf '%s\n' "$ARCHIVE"
```

### Stage 2: review the prepurge archive before deleting it
```bash
find "/Users/dankingsley/Documents/New project"/_schwab_prepurge_* -maxdepth 2 | sort | sed -n '1,200p'
du -sh "/Users/dankingsley/Documents/New project"/_schwab_prepurge_*
find /Users/dankingsley/PycharmProjects/schwab_trading_bot/docs/documents_merge/new_project_schwab_workspace -maxdepth 2 | sort | sed -n '1,200p'
```

### Stage 3: purge the archived Schwab workspace folders from New project
```bash
ARCHIVE="$(ls -dt "/Users/dankingsley/Documents/New project"/_schwab_prepurge_* | head -n 1)"
rm -rf "$ARCHIVE"
```
