# Schwab Trading Bot

## Runbook
- Canonical commands: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/COMMANDS.md`
- Terminal helper: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/runbook.sh`
- Canonical reports page: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/REPORTS.md`
- Report helper: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/reportbook.sh`

## Quick Usage
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/runbook.sh
./scripts/runbook.sh live
./scripts/runbook.sh retrain
./scripts/reportbook.sh bundle
```

## Notes
- Use `COMMANDS.md` as the source of truth for launch/retrain/floor/SQL commands.
- Use `REPORTS.md` as the source of truth for report generation/open commands.
- If command behavior changes, update `COMMANDS.md` first.

## Data Sources
- Ingestion source catalog + key requirements: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/DATA_INGESTION_SOURCES.md`
