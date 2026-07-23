#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"
source "$PROJECT_ROOT/scripts/ops/runtime_python.sh"
PYTHON_BIN="$(resolve_runtime_python)"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"
export MARKET_DATA_ONLY=1
export ALLOW_ORDER_EXECUTION=0
export BOT_LIVE_MONEY_LOCKED_DURING_SOAK=1
export BOT_UNATTENDED_SOAK_ACTIVE=1

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" soak_self_healing \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/soak_self_healing_control.py" \
  --apply \
  --target-days "${SOAK_SELF_HEAL_TARGET_DAYS:-30}" \
  --daily-max-age-minutes "${SOAK_SELF_HEAL_DAILY_MAX_AGE_MINUTES:-360}" \
  --step-timeout-sec "${SOAK_SELF_HEAL_STEP_TIMEOUT_SECONDS:-120}" \
  --storage-cooldown-minutes "${SOAK_SELF_HEAL_STORAGE_COOLDOWN_MINUTES:-60}" \
  --storage-cleanup-max-delete-gb "${SOAK_SELF_HEAL_STORAGE_CLEANUP_MAX_DELETE_GB:-16}" \
  --storage-target-free-gb "${SOAK_SELF_HEAL_STORAGE_TARGET_FREE_GB:-125}" \
  --ingestion-repair-cooldown-minutes "${SOAK_SELF_HEAL_INGESTION_REPAIR_COOLDOWN_MINUTES:-20}" \
  --include-adaptive-governor \
  --max-adaptive-repairs "${SOAK_SELF_HEAL_MAX_ADAPTIVE_REPAIRS:-3}" \
  --json
