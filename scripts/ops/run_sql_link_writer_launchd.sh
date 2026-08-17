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
export MAINTENANCE_SLOT_SQL_LINK_WRITER_MAX_RUNTIME_SECONDS="${MAINTENANCE_SLOT_SQL_LINK_WRITER_MAX_RUNTIME_SECONDS:-900}"
export SQL_LINK_SERVICE_FORCE_LOCAL_FALLBACK="${SQL_LINK_SERVICE_FORCE_LOCAL_FALLBACK:-1}"
# Cold archive export is a separate support lane; the hot SQLite writer never opens VIDEO-backed files.
export BOT_ALLOW_VIDEO_COLD_ARCHIVE=0

if [[ "${SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE:-0:l}" == "1" \
  || "${SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE:-0:l}" == "true" \
  || "${SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE:-0:l}" == "yes" \
  || "${SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE:-0:l}" == "on" ]]; then
  print -r -- "sql_link_writer status=deferred reason=local_storage_reserve_pressure"
  exit 0
fi

if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/runtime_maintenance_hold.py" --json \
  | "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("active") else 1)'; then
  print -r -- "sql_link_writer status=deferred reason=runtime_maintenance_hold"
  exit 0
fi

if [[ -n "${SQL_LINK_SERVICE_SHARDS:-}" ]]; then
  "$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" sql_link_writer \
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/sql_link_shard_manager.py"
  exit $?
fi

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" sql_link_writer \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/sql_link_writer_service.py"
