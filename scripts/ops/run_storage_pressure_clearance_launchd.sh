#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv314/bin/python"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" storage_pressure_clearance \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/storage_pressure_clearance_bot.py" \
  --apply \
  --force-clear-stale-gate \
  --checkpoint-mode "${STORAGE_PRESSURE_CLEARANCE_CHECKPOINT_MODE:-passive}" \
  --max-cycles "${STORAGE_PRESSURE_CLEARANCE_MAX_CYCLES:-1}" \
  --poll-seconds "${STORAGE_PRESSURE_CLEARANCE_POLL_SECONDS:-10}" \
  --wait-timeout-seconds "${STORAGE_PRESSURE_CLEARANCE_WAIT_TIMEOUT_SECONDS:-180}" \
  --command-timeout-seconds "${STORAGE_PRESSURE_CLEARANCE_TIMEOUT_SECONDS:-900}" \
  --json
