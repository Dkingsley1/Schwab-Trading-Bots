#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/Users/dankingsley/PycharmProjects/schwab_trading_bot}"
cd "$PROJECT_ROOT"

TS="${2:-$(date -u +%Y%m%d_%H%M%S)}"
LABEL="${1:-com.dankingsley.schwab.fullretrain.${TS}}"
LOG="$PROJECT_ROOT/logs/full_retrain_overnight_${TS}.log"
PID_FILE="$PROJECT_ROOT/governance/health/full_retrain_overnight_latest.pid"
STATUS_FILE="$PROJECT_ROOT/governance/health/full_retrain_overnight_latest.json"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/governance/health"

printf "%s\n" "$$" > "$PID_FILE"
printf '{"timestamp_utc":"%s","pid":%s,"launch_label":"%s","log":"%s","status":"running","mode":"force_all_full_overnight"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$LABEL" "$LOG" > "$STATUS_FILE"
printf '[full-retrain] started utc=%s pid=%s label=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$LABEL" >> "$LOG"

set +e
env \
  RETRAIN_ACTIVE_ONLY=0 \
  RETRAIN_MAX_TARGETS=0 \
  RETRAIN_FORCE_ALL_TARGETS=1 \
  PYTHONUNBUFFERED=1 \
  ./scripts/ops/opsctl.sh retrain-force-full \
    --force-all-targets \
    --include-deleted \
    --retrain-profile full_overnight \
    --runtime-training-snapshot-lookback-days 30 \
    --runtime-training-snapshot-reuse-if-fresh-minutes 360 \
    --target-timeout-seconds 2400 \
    --memory-max-wait-seconds 21600 \
    --between-target-sleep-seconds 20 \
    --thread-cap 2 \
    --max-swap-gb 4.5 \
    --adaptive-swap-max-gb 8.0 \
  >> "$LOG" 2>&1
RC=$?
set -e

if [[ "$RC" -eq 0 ]]; then
  FINAL_STATUS="completed"
else
  FINAL_STATUS="failed"
fi
printf '{"timestamp_utc":"%s","pid":%s,"launch_label":"%s","log":"%s","status":"%s","exit_code":%s,"mode":"force_all_full_overnight"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$LABEL" "$LOG" "$FINAL_STATUS" "$RC" > "$STATUS_FILE"
printf '[full-retrain] finished utc=%s pid=%s rc=%s status=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$RC" "$FINAL_STATUS" >> "$LOG"
exit "$RC"
