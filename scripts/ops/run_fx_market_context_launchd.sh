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

guard_output=""
if ! guard_output="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/resource_guard.py" --profile optional)"; then
  echo "fx_market_context skip resource_guard_blocked detail=${guard_output:-resource_guard_blocked}"
  exit 0
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_fx_market_context.py" --json
