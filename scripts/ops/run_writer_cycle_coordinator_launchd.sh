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

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/writer_cycle_coordinator.py" \
  --apply \
  --poll-seconds "${WRITER_CYCLE_COORDINATOR_POLL_SECONDS:-20}" \
  --wait-timeout-seconds "${WRITER_CYCLE_COORDINATOR_WAIT_TIMEOUT_SECONDS:-900}" \
  --command-timeout-seconds "${WRITER_CYCLE_COORDINATOR_COMMAND_TIMEOUT_SECONDS:-2400}" \
  --json
