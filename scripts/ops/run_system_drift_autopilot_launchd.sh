#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"
PYTHON_BIN="$("$PROJECT_ROOT/scripts/ops/runtime_python.sh")"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" system_drift_autopilot \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/system_drift_autopilot.py" \
  --apply \
  --max-steps "${SYSTEM_DRIFT_AUTOPILOT_MAX_STEPS:-12}" \
  --json
