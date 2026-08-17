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

# The always-on sentinel is intentionally outside the maintenance-slot gate.
# It performs only bounded allowlisted refreshes and publishes whether heavy
# repair is actually required.
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/soak_reliability_sentinel.py" --apply --json

if [[ -f "$PROJECT_ROOT/governance/runtime/soak_self_healing_request.json" ]] && \
  command -v jq >/dev/null 2>&1 && \
  jq -e '.active == true and .heavy_repair_required == true and (.severity == "critical" or .severity == "proactive")' "$PROJECT_ROOT/governance/runtime/soak_self_healing_request.json" >/dev/null 2>&1; then
  export MAINTENANCE_SLOT_DEFER_OUTSIDE_QUIET_WINDOW=0
fi

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" soak_self_healing \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/soak_self_healing_control.py" \
  --apply \
  --target-days "${SOAK_SELF_HEAL_TARGET_DAYS:-30}" \
  --daily-max-age-minutes "${SOAK_SELF_HEAL_DAILY_MAX_AGE_MINUTES:-360}" \
  --step-timeout-sec "${SOAK_SELF_HEAL_STEP_TIMEOUT_SECONDS:-120}" \
  --storage-cooldown-minutes "${SOAK_SELF_HEAL_STORAGE_COOLDOWN_MINUTES:-60}" \
  --storage-cleanup-max-delete-gb "${SOAK_SELF_HEAL_STORAGE_CLEANUP_MAX_DELETE_GB:-16}" \
  --storage-target-free-gb "${SOAK_SELF_HEAL_STORAGE_TARGET_FREE_GB:-135}" \
  --ingestion-repair-cooldown-minutes "${SOAK_SELF_HEAL_INGESTION_REPAIR_COOLDOWN_MINUTES:-20}" \
  --include-adaptive-governor \
  --max-adaptive-repairs "${SOAK_SELF_HEAL_MAX_ADAPTIVE_REPAIRS:-3}" \
  --json
