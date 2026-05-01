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

ARGS=(--apply --json --min-route-soak-hours "${BOT_LOGS_STANDBY_PRUNE_MIN_ROUTE_SOAK_HOURS:-2}")
if [[ "${BOT_LOGS_STANDBY_PRUNE_INCLUDE_CURATED_AUTO:-0}" == "1" ]]; then
  ARGS+=(--include-curated-standby)
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/storage_standby_prune.py" "${ARGS[@]}"
