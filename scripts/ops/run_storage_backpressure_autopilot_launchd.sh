#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" storage_backpressure_autopilot \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/storage_backpressure_autopilot.py" \
  --apply \
  --poll-seconds "${STORAGE_BACKPRESSURE_AUTOPILOT_POLL_SECONDS:-20}" \
  --wait-timeout-seconds "${STORAGE_BACKPRESSURE_AUTOPILOT_WAIT_TIMEOUT_SECONDS:-900}" \
  --command-timeout-seconds "${STORAGE_BACKPRESSURE_AUTOPILOT_TIMEOUT_SECONDS:-2400}" \
  --backpressure-command-timeout-seconds "${STORAGE_BACKPRESSURE_AUTOPILOT_BACKPRESSURE_TIMEOUT_SECONDS:-900}" \
  --max-cycles "${STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES:-3}" \
  --target-pending-lines "${STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_PENDING_LINES:-20000}" \
  --target-retention-debt-gb "${STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_RETENTION_DEBT_GB:-0.25}" \
  --json
