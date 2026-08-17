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

if [[ "${OPS_SUPPORT_MAINTENANCE_FREEZE:-0:l}" == "1" ]]; then
  echo "data_retention skip support_maintenance_freeze=1"
  exit 0
fi

if ! guard_output="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/resource_guard.py" --profile optional --json 2>&1)"; then
  echo "data_retention skip resource_guard_blocked detail=${guard_output:-resource_guard_blocked}"
  exit 0
fi

export RETENTION_STALE_PCORE_GUARD_PASSED=1

if ! slot_output="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/maintenance_slot_guard.py" --slot data_retention --begin --json 2>&1)"; then
  echo "data_retention skip maintenance_slot_blocked detail=${slot_output:-maintenance_slot_blocked}"
  exit 0
fi

finish_slot() {
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/maintenance_slot_guard.py" --slot data_retention --end --json >/dev/null 2>&1 || true
}
trap finish_slot EXIT INT TERM

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/data_retention_policy.py" --apply --skip-sqlite-vacuum --json

if [[ "${RETENTION_INCLUDE_EXTERNAL_STALE_ROOT:-1}" != "0" ]]; then
  stale_reaper_cmd=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/stale_artifact_reaper_bot.py"
    --include-external-stale-root
    --max-reindex-files "${RETENTION_STALE_REINDEX_MAX_FILES:-2048}"
    --max-reindex-gb "${RETENTION_STALE_REINDEX_MAX_GB:-4}"
    --max-oversized-reindex-files "${RETENTION_STALE_REINDEX_OVERSIZED_MAX_FILES:-1}"
    --max-oversized-reindex-gb "${RETENTION_STALE_REINDEX_OVERSIZED_MAX_GB:-64}"
    --oversized-reindex-min-age-days "${RETENTION_STALE_REINDEX_OVERSIZED_MIN_AGE_DAYS:-3}"
    --max-oversized-delete-files "${RETENTION_STALE_PURGE_OVERSIZED_MAX_FILES:-1}"
    --max-oversized-delete-gb "${RETENTION_STALE_PURGE_OVERSIZED_MAX_GB:-64}"
    --json
  )
  if [[ "$(uname -s)" == "Darwin" && "${RETENTION_STALE_PCORE_ENABLED:-1}" != "0" && "${RETENTION_STALE_PCORE_TASKPOLICY_APPLICATION:-1}" != "0" && -x /usr/sbin/taskpolicy ]]; then
    stale_reaper_cmd=(/usr/sbin/taskpolicy -a "${stale_reaper_cmd[@]}")
    export RETENTION_STALE_PCORE_TASKPOLICY_APPLIED=1
  else
    export RETENTION_STALE_PCORE_TASKPOLICY_APPLIED=0
  fi
  if ! BOT_WORKLOAD_CLASS=maintenance_accelerated \
      BOT_CPU_ALLOCATION_POLICY=performance_core_preferred_pressure_gated \
      BOT_CPU_QOS_POLICY=darwin_user_initiated_when_guarded \
      "${stale_reaper_cmd[@]}" >/dev/null; then
    echo "data_retention stale_artifact_reaper=degraded detail=health_artifact_written"
  fi
fi

if [[ "${BOT_COLD_ARCHIVE_COMPACTION_ON_RETENTION:-1}" != "0" ]]; then
  if ! archive_output="$(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/cold_archive_compactor.py" \
      --apply \
      --coordinate-writer-handoff \
      --archive-root "${BOT_SECOND_COLD_ROOT:-${BOT_LOGS_EXTERNAL_PROJECT_ROOT:-$PROJECT_ROOT}/cold_archive}" \
      --min-age-hours "${BOT_COLD_ARCHIVE_MIN_AGE_HOURS:-24}" \
      --max-files "${BOT_COLD_ARCHIVE_MAX_FILES:-8}" \
      --max-raw-gb "${BOT_COLD_ARCHIVE_MAX_RAW_GB:-16}" \
      --compression-level "${BOT_COLD_ARCHIVE_COMPRESSION_LEVEL:-3}" \
      2>&1
  )"; then
    echo "data_retention cold_archive_compaction=degraded detail=${archive_output:-unknown_error}"
  else
    echo "$archive_output"
  fi
fi
