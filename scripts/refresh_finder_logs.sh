#!/bin/zsh
set -uo pipefail

PROJECT_ROOT="/Users/dankingsley/PycharmProjects/schwab_trading_bot"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
EXPORT_SCRIPT="$PROJECT_ROOT/scripts/export_logs_to_csv.py"
OUT_DIR="$PROJECT_ROOT/exports/csv"
AUTOMATION_LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
HEALTH_FILE="$PROJECT_ROOT/governance/health/finder_log_refresh_latest.json"
PUBLISH_DESKTOP_SHORTCUTS="${FINDER_LOG_REFRESH_DESKTOP_SHORTCUTS:-0}"

mkdir -p "$AUTOMATION_LOG_DIR"
mkdir -p "$(dirname "$HEALTH_FILE")"

DATE_UTC="$(date -u +%Y%m%d)"
TIMESTAMP_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

refresh_ok=1
error_msg=""

if ! "$PYTHON_BIN" "$EXPORT_SCRIPT" --date "$DATE_UTC" --latest-aliases >> "$AUTOMATION_LOG_DIR/finder_refresh.log" 2>&1; then
  refresh_ok=0
  error_msg="export_logs_to_csv_failed"
fi

if [[ "$PUBLISH_DESKTOP_SHORTCUTS" == "1" ]]; then
  mkdir -p "$HOME/Desktop/Sub Logs" "$HOME/Desktop/Master Logs"
  ln -sfn "$OUT_DIR/latest_decision_explanations.csv" "$HOME/Desktop/Sub Logs/latest_decision_explanations.csv" || true
  ln -sfn "$OUT_DIR/latest_master_control.csv" "$HOME/Desktop/Master Logs/latest_master_control.csv" || true
fi

cat > "$HEALTH_FILE" <<JSON
{"timestamp_utc":"$TIMESTAMP_UTC","ok":$refresh_ok,"date_utc":"$DATE_UTC","error":"$error_msg","publish_desktop_shortcuts":$([[ "$PUBLISH_DESKTOP_SHORTCUTS" == "1" ]] && echo true || echo false)}
JSON

if [[ "$refresh_ok" != "1" ]]; then
  echo "finder_log_refresh_warn error=$error_msg date_utc=$DATE_UTC" >> "$AUTOMATION_LOG_DIR/finder_refresh.log"
fi

exit 0
