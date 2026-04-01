#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$("$PROJECT_ROOT/scripts/ops/runtime_python.sh")"
TODAY_UTC="$(date -u +%Y%m%d)"
export DAILY_AUTO_VERIFY_CMD_TIMEOUT_SEC="${DAILY_AUTO_VERIFY_CMD_TIMEOUT_SEC:-90}"
export DAILY_AUTO_VERIFY_SLOW_CMD_TIMEOUT_SEC="${DAILY_AUTO_VERIFY_SLOW_CMD_TIMEOUT_SEC:-300}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"

echo "daily_auto_verify_start day=$TODAY_UTC timeout_s=$DAILY_AUTO_VERIFY_CMD_TIMEOUT_SEC slow_timeout_s=$DAILY_AUTO_VERIFY_SLOW_CMD_TIMEOUT_SEC"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/daily_auto_verify.py" --day "$TODAY_UTC" "$@" || true
echo "daily_auto_verify_end day=$TODAY_UTC"
