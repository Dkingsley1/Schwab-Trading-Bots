# Commands (Canonical)

Use these exact commands as the source of truth.
They are aligned to the current `opsctl.sh`, `start_stack.sh`, and direct script entry points in this repo.

Primary start commands below use live market data with shadow execution only. Anything with `--simulate` or `--paper` is not live data.

The old custom `run_all_sleeves.py ... --simulate` command was simulated data. The live version below removes `--simulate`.

## Token Authorization

### FIRST: refresh Schwab token / authorization handshake
Use this first when Schwab live feeds halt, the token expires, or `get_accounts_snapshot` starts failing.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh --always-auth
```

Expected healthy result: `premarket_token_guard ok=1`

### If non-interactive token refresh fails, run an interactive Schwab re-auth
Use this when `token-refresh --always-auth` fails with `unsupported_token_type` or `refresh_token_authentication_error`.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh-interactive
```

Expected healthy result: `Handshake Successful.` and `schwab_auth_refresh ok=1`

### FIRST: restart Schwab live feeds after token authorization
Run this immediately after a successful token handshake.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

### Exact Schwab feed refresh command
Copy/paste this when you only need to restart Schwab live feeds.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

### Full token-auth recovery sequence
Use this exact pair when the system is halted on Schwab auth problems.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh --always-auth
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

### Full token-auth recovery sequence when non-interactive refresh is broken
Use this exact sequence when the token refresh reports `unsupported_token_type`.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh-interactive
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

## Live Starts

### Full live feed refresh/start (Schwab sleeves + Coinbase live data)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source all
```

### Schwab live sleeves only
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

### Coinbase live shadow only
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-start --force-restart --live-data
```

### Coinbase crypto futures live shadow
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-futures-start --force-restart --live-data --profiles crypto_futures
```

### Schwab futures live shadow
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-futures-start --force-restart --live-data --profiles schwab_futures
```

### Direct all-sleeves live run with canonical symbol sets
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
source ./scripts/ops/load_runtime_env.sh live --quiet
./.venv312/bin/python scripts/run_all_sleeves.py --with-aggressive-modes
```

## Live Feed Refresh

### Refresh all live feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source all
```

### Refresh Schwab live feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

### Refresh Coinbase live feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source coinbase
```

## Live Macro Bulletins

### Powell / Fed live bulletin
Use this during Powell remarks, FOMC press conferences, or other live Fed events. The bots react to the text bulletin immediately through the news-shock and macro-event features. The YouTube link is optional metadata; the text is what matters.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-bulletin --template powell --headline "Jerome Powell live remarks" --summary "Paste the key line or theme here" --url "https://www.youtube.com/watch?v=-sSSzdXIlA8" --stance auto --impact high
```

### Hawkish Powell update
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-bulletin --template powell --headline "Powell says inflation risks remain elevated" --summary "Higher for longer tone; no rush to cut" --url "https://www.youtube.com/watch?v=-sSSzdXIlA8" --stance hawkish --impact critical
```

### Dovish Powell update
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-bulletin --template powell --headline "Powell says disinflation is continuing" --summary "Growth softening and policy can become less restrictive" --url "https://www.youtube.com/watch?v=-sSSzdXIlA8" --stance dovish --impact critical
```

### Show the active live macro bulletin
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-bulletin --status --json
```

### Start automatic YouTube caption watch for a single Powell stream
This uses YouTube auto-captions and keeps the live macro bulletin updated without manual re-entry.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-auto-start --force-restart --youtube-url "https://www.youtube.com/watch?v=-sSSzdXIlA8" --template powell --speaker "Jerome Powell" --source "Federal Reserve"
```

### Start automatic Federal Reserve channel watch
This watches the full Federal Reserve YouTube channel, checks the Schwab calendar for the matching Fed event, arms media ingest inside the pre-live window, and then rolls straight into live capture when the stream actually opens.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-auto-start --force-restart --youtube-channel-url "https://www.youtube.com/@federalreserve" --template fed --speaker "Federal Reserve" --source "Federal Reserve" --correlate-with-schwab-calendar --trigger-media-ingest-on-live --trigger-media-ingest-before-minutes 10 --media-ingest-cookies-from-browser chrome
```

### Replay a full Fed or Powell video transcript
This re-reads the entire YouTube caption track from a finished video, scores the whole transcript, and writes an overall macro stance with keyword evidence.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-replay --youtube-url "https://www.youtube.com/watch?v=-sSSzdXIlA8" --template powell --speaker "Jerome Powell" --source "Federal Reserve" --json
```

### Capture audio, archive alignment, and build training-ready macro transcript features
This downloads the event audio, tries MLX-based transcription when available, archives alignment against the caption cues already captured, and writes SQL-ingestible training feature rows.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-media-ingest --youtube-url "https://www.youtube.com/watch?v=-sSSzdXIlA8" --template powell --speaker "Jerome Powell" --source "Federal Reserve" --cookies-from-browser chrome --wait-for-live-seconds 900 --json
```

### Check automatic macro watch status
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-auto-status
```

### Stop automatic YouTube caption watch
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-auto-stop
```

### Clear the live macro bulletin after the event
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-bulletin --clear
```

### Watch all live feeds
Use `--symbol SYMBOL` only when you want a filtered tail; there is no NVIDIA-specific default in this list.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source all --lines 80
```

### Watch Schwab live feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-tail --lines 80
```

### Watch Coinbase live feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-tail --lines 80
```

### Refresh Schwab auth token before restarting live feeds
Run this before `feed-refresh --source schwab` if Schwab auth or handshake looks stale.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh --always-auth
```

### Interactive Schwab token refresh
Use this when the normal token refresh still leaves auth broken.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh-interactive
```

## Health And SQL

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

### TradingEconomics free guest-data sync
This pulls the free no-key guest sample datasets TradingEconomics exposes, archives raw rows for SQL, and refreshes the live `market_breadth` and `bond_reference` snapshots the loop already consumes.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh tradingeconomics-sync --json
```

### TradingEconomics free guest-data sync with broader country coverage
Use this when you want the widest no-key macro pull the system supports without a paid TradingEconomics key.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh tradingeconomics-sync --countries "United States,Euro Area,China,Japan,United Kingdom,Canada" --lookahead-days 45 --news-limit 40 --json
```

### SQL sync + SQLite maintenance + One Numbers refresh
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./.venv312/bin/python scripts/link_jsonl_to_sql.py --mode sqlite
./.venv312/bin/python scripts/sqlite_performance_maintenance.py
./.venv312/bin/python scripts/build_one_numbers_report.py --day "$(date -u +%Y%m%d)"
```

### Daily full refresh pipeline
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/daily_log_refresh.sh
```

### Allocator / risk / budget layer refresh
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./.venv312/bin/python scripts/sleeve_allocator.py
./.venv312/bin/python scripts/portfolio_risk_ledger.py
./.venv312/bin/python scripts/execution_budgeter.py
```

## Retrain

### Standard retrain (after-hours gate enabled)
This wrapper is blocked during the configured session window, which is currently `8:00 AM` to `8:00 PM` Eastern on March 18, 2026 unless you override the gate.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh retrain
```

### Force full retrain bypass
Runs during market hours and bypasses retrain quality/freshness/sample/precheck gates.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh retrain-force-full
```

### Force targeted retrain bypass
Use this for surgical retrains without a master-registry update.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh retrain-force-targeted --include-bot-ids brain_refinery_v56_meta_ranker,brain_refinery_v64_regime_router_layer
```

### Distillation / promotion recovery retrain
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
RETRAIN_AFTER_HOURS_ONLY=0 \
RETRAIN_MAX_TARGETS=12 \
RETRAIN_REGIME_FOCUS=mean_revert,shock,macro \
RETRAIN_CANARY_PRIORITY_TOP_N=10 \
RETRAIN_NEW_BOT_BOOST=1 \
RETRAIN_DISTILLATION_PRIORITY=1 \
RETRAIN_DISTILLATION_STUDENT_EXTRA_PASS=2 \
./.venv312/bin/python scripts/weekly_retrain.py --continue-on-error
```

### Walk-forward + promotion gate
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./.venv312/bin/python scripts/walk_forward_validate.py
./.venv312/bin/python scripts/walk_forward_promotion_gate.py
```

### Regime detection / regime validation
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh regime-validate
```

### Meta learning layer targeted refresh
The meta-learning layer already exists as `brain_refinery_v56_meta_ranker`; this retrains only that layer.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh retrain-force-targeted --include-bot-ids brain_refinery_v56_meta_ranker
```

## Model And Analysis

### Model card + lineage snapshot
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh model-card --json
```

### Bot explainability export
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh explainability --limit 12 --json
```

### Strategy attribution artifact
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh strategy-attribution --day "$(date -u +%Y%m%d)" --json
```

### Paper execution calibration
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh paper-calibration --hours 24 --json
```

### Automated post-trade analysis
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh post-trade-analysis --day "$(date -u +%Y%m%d)" --hours 24 --json
```

## Reports

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

### PDF report bundle page
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh report-pdfs --json
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/report_pdf_bundle_latest.html
```

### Open the latest PDF bundle file
```bash
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/report_pdf_bundle_latest.pdf
```

## Halts And Recovery

### Current halt recovery state
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
cat governance/health/shadow_watchdog_halt_recovery_latest.json
```

### Tail softguard global halt events
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
tail -n 40 governance/events/live_softguard_$(date -u +%Y%m%d).jsonl
```

### Tail live execution guard failures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
tail -n 40 governance/events/live_execution_guard_$(date -u +%Y%m%d).jsonl
```

### Search today for softguard/token-related halt causes
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
rg -n "softguard_api_circuit_opened|refresh_token_authentication_error|unsupported_token_type|circuit_opened" governance/events logs --glob "*$(date -u +%Y%m%d)*"
```

### Recovery sequence after token/auth-related halts
Run token authorization first. A healthy token refresh should print `premarket_token_guard ok=1`.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh --always-auth
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

### Recovery sequence when token refresh reports `unsupported_token_type`
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh-interactive
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

## Sim And Paper

### Simulated Schwab sleeves
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh start-sim --run-all-sleeves --force-restart
```

### Coinbase paper mirror
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-start --paper --force-restart --top-n 5 --min-acc 0.58 --profiles default
```

### Coinbase crypto futures paper mirror
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-futures-start --paper --force-restart --top-n 10 --min-acc 0.56 --profiles crypto_futures
```

## Common Gotchas

- If a command includes `--simulate`, it is simulated data, not live data.
- If a command includes `--paper`, it is paper execution, not live execution.
- `token-refresh` is the first fix for repeated Schwab `softguard_api_circuit_opened` halts tied to refresh-token auth failures.
- If `token-refresh --always-auth` fails with `unsupported_token_type`, use `token-refresh-interactive` before restarting Schwab feeds.
- After Schwab token authorization, a healthy refresh should print `premarket_token_guard ok=1`; if it does not, fix credentials before trusting live restarts.
- `retrain-force-targeted` skips master update on purpose; use it for surgical retrains, not promotion.
- `report-pdfs` refreshes the report catalog page even when some optional source reports are still missing.
