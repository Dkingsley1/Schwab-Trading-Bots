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

### Live feed tail / refresh view for all feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source all --lines 80
```

### Live feed tail / refresh view for Schwab
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-tail --lines 80
```

### Live feed tail / refresh view for Coinbase
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-tail --lines 80
```

### Live feed tail / refresh view for FX
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh fx-tail --lines 80
```

## Live Macro Bulletins

### Powell / Fed live bulletin
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-bulletin --template powell --headline "Jerome Powell live remarks" --summary "Paste the key line or theme here" --url "https://www.youtube.com/watch?v=-sSSzdXIlA8" --stance auto --impact high
```

### Show the active live macro bulletin
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-bulletin --status --json
```

### Start automatic macro watch
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-auto-start --force-restart --youtube-channel-url "https://www.youtube.com/@federalreserve" --template fed --speaker "Federal Reserve" --source "Federal Reserve" --correlate-with-schwab-calendar --trigger-media-ingest-on-live --trigger-media-ingest-before-minutes 10 --media-ingest-cookies-from-browser chrome
```

### Stop automatic macro watch
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-auto-stop
```

## Report Generation

### Training report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh training-report --json
```

### Crash report digest
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh crash-report --lookback-days 3 --recent-limit 40 --json
```

### Project timeline report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh timeline-report --json
```

### Paper performance report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh paper-performance --day "$(date -u +%Y%m%d)" --week-days 7 --json
```

## Reports

### Refresh the report catalog and PDF bundle
Run this first if a PDF is missing or stale.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh report-pdfs --json
```

### Open the report catalog PDF
```bash
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/report_pdf_bundle_latest.pdf
```

### Open the latest crash report PDF
```bash
open "$(ls -t /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/crash_reports/*.pdf | head -n 1)"
```

### Open the latest training report PDF
```bash
open "$(ls -t /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/training_reports/*.pdf | head -n 1)"
```

### Open the project timeline PDF
Build it first if it is missing.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh timeline-report --json
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/project_timeline/project_timeline_latest.pdf
```

## Retrain

### Standard retrain
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh retrain
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
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/link_jsonl_to_sql.py --mode sqlite
"$PY" scripts/sqlite_performance_maintenance.py
"$PY" scripts/build_one_numbers_report.py
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

## Storage Routing

### Switch collection to the Mac's internal drive
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-switch-local
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
