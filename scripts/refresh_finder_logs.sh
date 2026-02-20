#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="/Users/dankingsley/PycharmProjects/schwab_trading_bot"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
EXPORT_SCRIPT="$PROJECT_ROOT/scripts/export_logs_to_csv.py"
OUT_DIR="$PROJECT_ROOT/exports/csv"
AUTOMATION_LOG_DIR="$PROJECT_ROOT/logs/automation"

mkdir -p "$AUTOMATION_LOG_DIR"
mkdir -p "$HOME/Desktop/Sub Logs" "$HOME/Desktop/Master Logs"

DATE_UTC="$(date -u +%Y%m%d)"

"$PYTHON_BIN" "$EXPORT_SCRIPT" --date "$DATE_UTC" --latest-aliases >> "$AUTOMATION_LOG_DIR/finder_refresh.log" 2>&1

ln -sfn "$OUT_DIR/latest_decision_explanations.csv" "$HOME/Desktop/Sub Logs/latest_decision_explanations.csv"
ln -sfn "$OUT_DIR/latest_master_control.csv" "$HOME/Desktop/Master Logs/latest_master_control.csv"
