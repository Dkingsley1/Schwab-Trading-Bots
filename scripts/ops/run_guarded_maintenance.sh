#!/bin/zsh
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: run_guarded_maintenance.sh SLOT COMMAND [ARGS...]" >&2
  exit 64
fi

SLOT="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"
PRESSURE_OVERRIDE_FILE="$PROJECT_ROOT/config/.env.pressure_relief_override"
if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
elif [[ -f "$PRESSURE_OVERRIDE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$PRESSURE_OVERRIDE_FILE"
fi
PYTHON_BIN="$("$PROJECT_ROOT/scripts/ops/runtime_python.sh")"
GUARD="$PROJECT_ROOT/scripts/ops/maintenance_slot_guard.py"
SKIP_RC="${MAINTENANCE_SLOT_SKIP_EXIT_CODE:-75}"
JITTER_MAX_SECONDS="${MAINTENANCE_SLOT_JITTER_MAX_SECONDS:-90}"
if [[ "$SLOT" == "infrastructure_observe" ]]; then
  JITTER_MAX_SECONDS="${MAINTENANCE_SLOT_OBSERVER_JITTER_MAX_SECONDS:-5}"
fi
if [[ "$SLOT" == "sql_link_writer" ]]; then
  JITTER_MAX_SECONDS="${MAINTENANCE_SLOT_SQL_LINK_WRITER_JITTER_MAX_SECONDS:-0}"
  NICE_LEVEL="${MAINTENANCE_SLOT_NICE_LEVEL:-${SQL_LINK_WRITER_NICE:-${OPS_SQL_WRITER_NICE:-0}}}"
  BACKGROUND_POLICY="${MAINTENANCE_SLOT_BACKGROUND_POLICY:-${SQL_LINK_WRITER_BACKGROUND_POLICY:-${OPS_SQL_WRITER_BACKGROUND_POLICY:-0}}}"
else
  NICE_LEVEL="${MAINTENANCE_SLOT_NICE_LEVEL:-${OPS_SUPPORT_JOB_NICE:-15}}"
  BACKGROUND_POLICY="${MAINTENANCE_SLOT_BACKGROUND_POLICY:-${OPS_SUPPORT_JOBS_BACKGROUND_POLICY:-1}}"
fi
case "$SLOT" in
  sql_link_writer)
    DEFAULT_MAX_RUNTIME_SECONDS="${MAINTENANCE_SLOT_SQL_LINK_WRITER_MAX_RUNTIME_SECONDS:-900}"
    ;;
  sqlite_maintenance)
    DEFAULT_MAX_RUNTIME_SECONDS="${SQLITE_MAINTENANCE_SLOT_MAX_RUNTIME_SECONDS:-14400}"
    ;;
  storage_backpressure_autopilot|storage_pressure_clearance)
    DEFAULT_MAX_RUNTIME_SECONDS="${MAINTENANCE_SLOT_STORAGE_MAX_RUNTIME_SECONDS:-1800}"
    ;;
  *)
    DEFAULT_MAX_RUNTIME_SECONDS="0"
    ;;
esac
MAX_RUNTIME_SECONDS="${MAINTENANCE_SLOT_MAX_RUNTIME_SECONDS:-$DEFAULT_MAX_RUNTIME_SECONDS}"
TIMEOUT_TERM_GRACE_SECONDS="${MAINTENANCE_SLOT_TIMEOUT_TERM_GRACE_SECONDS:-30}"
STORAGE_RECOVERY_FAST_MODE=0
STORAGE_RECOVERY_MIN_INTERVAL_SECONDS=""
case "$SLOT" in
  storage_backpressure_autopilot)
    STORAGE_RECOVERY_FAST_MODE="${MAINTENANCE_SLOT_STORAGE_RECOVERY_FAST_MODE:-0}"
    STORAGE_RECOVERY_MIN_INTERVAL_SECONDS="${MAINTENANCE_SLOT_STORAGE_BACKPRESSURE_AUTOPILOT_MIN_INTERVAL_SECONDS:-}"
    ;;
  storage_pressure_clearance)
    STORAGE_RECOVERY_FAST_MODE="${MAINTENANCE_SLOT_STORAGE_RECOVERY_FAST_MODE:-0}"
    STORAGE_RECOVERY_MIN_INTERVAL_SECONDS="${MAINTENANCE_SLOT_STORAGE_PRESSURE_CLEARANCE_MIN_INTERVAL_SECONDS:-}"
    ;;
esac

if [[ "${MAINTENANCE_SLOT_DISABLE_JITTER:-0}" != "1" ]] && [[ "$JITTER_MAX_SECONDS" == <-> ]] && (( JITTER_MAX_SECONDS > 0 )); then
  sleep $(( RANDOM % (JITTER_MAX_SECONDS + 1) ))
fi

guard_args=(--slot "$SLOT" --execute --runtime-limit "$MAX_RUNTIME_SECONDS" --terminate-grace "$TIMEOUT_TERM_GRACE_SECONDS")
guard_args+=(--lease-wait-seconds "${MAINTENANCE_SLOT_LEASE_WAIT_SECONDS:-0}")
if [[ "${MAINTENANCE_SLOT_ALLOW_DURING_MACRO_EVENT:-0}" == "1" ]]; then
  guard_args+=(--allow-during-macro-event)
fi
if [[ "$SLOT" == "strategy_market_fit_infrabot" || "$SLOT" == "infrastructure_observe" ]]; then
  guard_args+=(--no-defer-outside-quiet-window --no-defer-while-sql-link-active)
fi
if [[ "$STORAGE_RECOVERY_FAST_MODE" == "1" ]]; then
  if [[ "${MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_OUTSIDE_QUIET_WINDOW:-1}" == "0" ]]; then
    guard_args+=(--no-defer-outside-quiet-window)
  fi
  if [[ "${MAINTENANCE_SLOT_STORAGE_RECOVERY_DEFER_WHILE_SQL_LINK_ACTIVE:-1}" == "0" ]]; then
    guard_args+=(--no-defer-while-sql-link-active)
  else
    guard_args+=(--defer-while-sql-link-active)
  fi
  if [[ -n "$STORAGE_RECOVERY_MIN_INTERVAL_SECONDS" ]]; then
    guard_args+=(--min-interval-seconds "$STORAGE_RECOVERY_MIN_INTERVAL_SECONDS")
  fi
fi
cmd_prefix=()
if [[ "$BACKGROUND_POLICY" == "1" ]] && command -v taskpolicy >/dev/null 2>&1; then
  cmd_prefix=(taskpolicy -b nice -n "$NICE_LEVEL")
else
  cmd_prefix=(nice -n "$NICE_LEVEL")
fi

set +e
"$PYTHON_BIN" "$GUARD" "${guard_args[@]}" --command "${cmd_prefix[@]}" "$@"
rc=$?
set -e
if [[ "$rc" == "$SKIP_RC" ]]; then
  exit 0
fi
exit "$rc"
