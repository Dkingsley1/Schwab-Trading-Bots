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

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" storage_reconnect_infrabot \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/storage_reconnect_infrabot.py" \
  --apply \
  --timeout-sec "${STORAGE_RECONNECT_INFRABOT_TIMEOUT_SECONDS:-900}" \
  --json
