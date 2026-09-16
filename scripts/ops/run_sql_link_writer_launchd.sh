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

# Publish fresh admission even when the scheduled writer exits before starting SQL.
"$PYTHON_BIN" -m scripts.ops.sql_writer_admission >/dev/null || print -r -- "sql_link_writer observation=failed"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/data_plane_recovery_controller.py" --apply --json >/dev/null \
  || print -r -- "sql_link_writer write_path_recovery=guarded"

if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/runtime_maintenance_hold.py" --json \
  | "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("active") else 1)'; then
  print -r -- "sql_link_writer status=deferred reason=runtime_maintenance_hold"
  exit 0
fi

refresh_backlog_plan() {
  # Keep requests and activation evidence fresh even while SQL is storage-paused.
  if ! "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/backpressure_drainer_fleet.py" \
    --apply --refresh-backlog --refresh-accelerator --ttl-seconds 120 --json >/dev/null; then
    print -r -- "sql_link_writer backlog_plan=deferred preserving_existing_request=1"
  fi
}
refresh_backlog_plan

if [[ "${SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE:-0:l}" == "1" \
  || "${SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE:-0:l}" == "true" \
  || "${SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE:-0:l}" == "yes" \
  || "${SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE:-0:l}" == "on" ]]; then
  if ! nice -n 15 "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/soak_self_healing_control.py" \
    --storage-recovery-only --quick-storage-recovery --apply --json >/dev/null; then
    print -r -- "sql_link_writer storage_recovery=deferred"
  fi
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
  if [[ "${SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE:-0:l}" == "1" \
    || "${SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE:-0:l}" == "true" \
    || "${SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE:-0:l}" == "yes" \
    || "${SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE:-0:l}" == "on" ]]; then
    print -r -- "sql_link_writer status=deferred reason=local_storage_reserve_pressure"
    exit 0
  fi
  # Recovery may outlive the request or change which debt should go first.
  refresh_backlog_plan
fi

if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/runtime_maintenance_hold.py" --json \
  | "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("active") else 1)'; then
  print -r -- "sql_link_writer status=deferred reason=runtime_maintenance_hold"
  exit 0
fi

# launchd owns the deadline; useful passes may follow through inside one bounded run.
SQL_LINK_MANAGER_ARGS=(--once)

if [[ -n "${SQL_LINK_SERVICE_SHARDS:-}" ]]; then
  SQL_LINK_MANAGER_ARGS+=(--scheduled-drain)
  "$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" sql_link_writer \
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/sql_link_shard_manager.py" "${SQL_LINK_MANAGER_ARGS[@]}"
  exit $?
fi

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" sql_link_writer \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/sql_link_writer_service.py" "${SQL_LINK_MANAGER_ARGS[@]}"
