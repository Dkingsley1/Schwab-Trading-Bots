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
export BOT_PROTECTED_VOLUME_DENYLIST="${BOT_PROTECTED_VOLUME_DENYLIST:-/Volumes/VIDEO}"

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" distributed_cell_architecture \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/distributed_cell_architecture.py" \
  --apply \
  --json

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" cell_federation_intelligence \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/cell_federation_intelligence.py" \
  --apply \
  --json
