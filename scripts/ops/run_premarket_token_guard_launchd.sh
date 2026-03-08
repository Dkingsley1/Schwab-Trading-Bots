#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${BOT_PYTHON_BIN:-$PROJECT_ROOT/.venv312/bin/python}"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/premarket_token_guard.py" "$@"
