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
if [[ "$SLOT" == "sql_link_writer" ]]; then
  JITTER_MAX_SECONDS="${MAINTENANCE_SLOT_SQL_LINK_WRITER_JITTER_MAX_SECONDS:-0}"
  NICE_LEVEL="${MAINTENANCE_SLOT_NICE_LEVEL:-${SQL_LINK_WRITER_NICE:-${OPS_SQL_WRITER_NICE:-0}}}"
  BACKGROUND_POLICY="${MAINTENANCE_SLOT_BACKGROUND_POLICY:-${SQL_LINK_WRITER_BACKGROUND_POLICY:-${OPS_SQL_WRITER_BACKGROUND_POLICY:-0}}}"
else
  NICE_LEVEL="${MAINTENANCE_SLOT_NICE_LEVEL:-${OPS_SUPPORT_JOB_NICE:-15}}"
  BACKGROUND_POLICY="${MAINTENANCE_SLOT_BACKGROUND_POLICY:-${OPS_SUPPORT_JOBS_BACKGROUND_POLICY:-1}}"
fi
case "$SLOT" in
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

if [[ "${MAINTENANCE_SLOT_DISABLE_JITTER:-0}" != "1" ]] && [[ "$JITTER_MAX_SECONDS" == <-> ]] && (( JITTER_MAX_SECONDS > 0 )); then
  sleep $(( RANDOM % (JITTER_MAX_SECONDS + 1) ))
fi

set +e
guard_args=(--slot "$SLOT" --begin)
if [[ "${MAINTENANCE_SLOT_ALLOW_DURING_MACRO_EVENT:-0}" == "1" ]]; then
  guard_args+=(--allow-during-macro-event)
fi
"$PYTHON_BIN" "$GUARD" "${guard_args[@]}"
guard_rc=$?
set -e
if [[ "$guard_rc" != "0" ]]; then
  if [[ "$guard_rc" == "$SKIP_RC" ]]; then
    exit 0
  fi
  exit "$guard_rc"
fi

cleanup() {
  "$PYTHON_BIN" "$GUARD" --slot "$SLOT" --end >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cmd_prefix=()
if [[ "$BACKGROUND_POLICY" == "1" ]] && command -v taskpolicy >/dev/null 2>&1; then
  cmd_prefix=(taskpolicy -b nice -n "$NICE_LEVEL")
else
  cmd_prefix=(nice -n "$NICE_LEVEL")
fi

if [[ "$MAX_RUNTIME_SECONDS" == <-> ]] && (( MAX_RUNTIME_SECONDS > 0 )); then
  set +e
  "${cmd_prefix[@]}" "$@" &
  child_pid=$!
  set -e
  start_seconds=$SECONDS
  while kill -0 "$child_pid" >/dev/null 2>&1; do
    if (( SECONDS - start_seconds >= MAX_RUNTIME_SECONDS )); then
      echo "maintenance_slot_timeout slot=$SLOT pid=$child_pid max_runtime_seconds=$MAX_RUNTIME_SECONDS" >&2
      kill -TERM "$child_pid" >/dev/null 2>&1 || true
      grace_start=$SECONDS
      while kill -0 "$child_pid" >/dev/null 2>&1; do
        if (( SECONDS - grace_start >= TIMEOUT_TERM_GRACE_SECONDS )); then
          echo "maintenance_slot_timeout_force_kill slot=$SLOT pid=$child_pid" >&2
          kill -KILL "$child_pid" >/dev/null 2>&1 || true
          break
        fi
        sleep 1
      done
      set +e
      wait "$child_pid" >/dev/null 2>&1
      set -e
      exit 124
    fi
    sleep 1
  done
  set +e
  wait "$child_pid"
  rc=$?
  set -e
  exit "$rc"
fi

"${cmd_prefix[@]}" "$@"
