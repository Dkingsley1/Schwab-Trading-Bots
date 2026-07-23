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

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" infrastructure_autofix \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/infrastructure_autofix_bot.py" \
  --apply \
  --timeout-sec "${INFRASTRUCTURE_AUTOFIX_TIMEOUT_SECONDS:-1200}" \
  --json
