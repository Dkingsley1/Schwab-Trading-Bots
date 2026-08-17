#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/ops/runtime_python.sh"
PYTHON_BIN="$(resolve_runtime_python)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/premarket_token_guard.py" "$@"
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "[WARN] premarket_token_guard exit_code=$rc"
fi
exit 0
