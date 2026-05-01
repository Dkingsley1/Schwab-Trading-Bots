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

ARGS=(--timeout-sec "${MASTER_INFRASTRUCTURE_SUPERVISOR_TIMEOUT_SECONDS:-900}" --json)
if [[ "${MASTER_INFRASTRUCTURE_SUPERVISOR_APPLY:-0}" == "1" ]]; then
  ARGS=(--apply "${ARGS[@]}")
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/master_infrastructure_supervisor.py" "${ARGS[@]}"
