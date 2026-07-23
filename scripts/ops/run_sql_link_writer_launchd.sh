#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
PYTHON_BIN="$PROJECT_ROOT/.venv314/bin/python"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"
export MAINTENANCE_SLOT_NICE_LEVEL="${SQL_LINK_WRITER_NICE:-${OPS_SQL_WRITER_NICE:-${OPS_SUPPORT_JOB_NICE:-5}}}"
export MAINTENANCE_SLOT_BACKGROUND_POLICY="${SQL_LINK_WRITER_BACKGROUND_POLICY:-${OPS_SQL_WRITER_BACKGROUND_POLICY:-0}}"

if [[ -n "${SQL_LINK_SERVICE_SHARDS:-}" ]]; then
  "$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" sql_link_writer \
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/sql_link_shard_manager.py"
  exit $?
fi

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" sql_link_writer \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/sql_link_writer_service.py"
