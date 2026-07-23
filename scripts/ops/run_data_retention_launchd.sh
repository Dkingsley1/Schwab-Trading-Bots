#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
PYTHON_BIN="$PROJECT_ROOT/.venv314/bin/python"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"

if [[ "${OPS_SUPPORT_MAINTENANCE_FREEZE:-0:l}" == "1" ]]; then
  echo "data_retention skip support_maintenance_freeze=1"
  exit 0
fi

if ! guard_output="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/resource_guard.py" --profile optional --json 2>&1)"; then
  echo "data_retention skip resource_guard_blocked detail=${guard_output:-resource_guard_blocked}"
  exit 0
fi

if ! slot_output="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/maintenance_slot_guard.py" --slot data_retention --begin --json 2>&1)"; then
  echo "data_retention skip maintenance_slot_blocked detail=${slot_output:-maintenance_slot_blocked}"
  exit 0
fi

finish_slot() {
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/maintenance_slot_guard.py" --slot data_retention --end --json >/dev/null 2>&1 || true
}
trap finish_slot EXIT INT TERM

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/data_retention_policy.py" --apply --skip-sqlite-vacuum --json
