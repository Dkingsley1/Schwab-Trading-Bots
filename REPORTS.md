# Reports (Canonical)

Use this page as the source of truth for report generation, bundle refreshes, and report file locations.

## Freshness Warning

Do not trust a report or health gate blindly if the underlying artifact timestamp is stale. This matters most for:
- `governance/health/data_source_divergence_latest.json`
- `governance/health/daily_auto_verify_latest.json`
- `governance/health/retrain_scorecard_latest.json`
- `governance/health/model_card_latest.json`

Quick check:
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
cat governance/health/data_source_divergence_latest.json
cat governance/health/daily_auto_verify_latest.json
```

If those files are old, run:
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/daily_log_refresh.sh
```

## Bundle Page

### Refresh the report catalog page and PDF bundle
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh report-pdfs --json
```

### Open the report catalog page
```bash
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/report_pdf_bundle_latest.html
```

### Open the report catalog PDF
```bash
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/report_pdf_bundle_latest.pdf
```

## Operations

### Daily ops refresh artifacts
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/daily_log_refresh.sh
```

### Daily runtime summary
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/daily_runtime_summary.py --day "$(date -u +%Y%m%d)" --json > exports/sql_reports/daily_runtime_summary_$(date -u +%Y%m%d).json
```

### One Numbers report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/build_one_numbers_report.py
```
This writes the single canonical One Numbers report:
- `exports/one_numbers/latest.csv`
- `exports/one_numbers/latest.md`

Those files include the day, month-to-date, and all-time rollups in one document.

### Unified lane scorecard
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/unified_lane_scorecard.py --json
```

## Training

### Training report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh training-report --json
```

### Retrain scorecard
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
cat /Users/dankingsley/PycharmProjects/schwab_trading_bot/governance/health/retrain_scorecard_latest.json
```

### Model card / lineage
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh model-card --json
```

### Bot explainability
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh explainability --limit 12 --json
```

### Regime validation artifact
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh regime-validate
cat /Users/dankingsley/PycharmProjects/schwab_trading_bot/governance/walk_forward/regime_segmented_latest.json
```

## Analysis

### Strategy attribution report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh strategy-attribution --day "$(date -u +%Y%m%d)" --json
```

### Paper execution calibration report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh paper-calibration --hours 24 --json
```

### Automated post-trade analysis
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh post-trade-analysis --day "$(date -u +%Y%m%d)" --hours 24 --json
```

## Forensics

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

### Latest printable timeline outputs
```bash
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/project_timeline/project_timeline_print_latest.html
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/project_timeline/project_timeline_latest.pdf
```

## Notes

- `report-pdfs` builds the report catalog page and attempts PDF companions for every available report source.
- The report catalog now includes model card, bot explainability, paper calibration, strategy attribution, and post-trade analysis artifacts.
- Some reports are on-demand and may show `missing_source` in the bundle until they are generated at least once.
