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

ARGS=(--json)
if [[ "${PROCESS_FANOUT_GUARD_APPLY:-1}" == "1" ]]; then
  ARGS=(--apply "${ARGS[@]}")
fi

set +e
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/process_fanout_guard.py" "${ARGS[@]}"
RC=$?
set -e

if [[ "${GUARD_INTELLIGENCE_AUTO_APPLY:-1}" == "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/guard_intelligence_layer.py" --apply --json || true
fi

exit "$RC"
