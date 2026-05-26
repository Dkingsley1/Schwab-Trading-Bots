#!/bin/zsh
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: run_guarded_maintenance.sh SLOT COMMAND [ARGS...]" >&2
  exit 64
fi

SLOT="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PRESSURE_OVERRIDE_FILE="$PROJECT_ROOT/config/.env.pressure_relief_override"
if [[ -f "$PRESSURE_OVERRIDE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$PRESSURE_OVERRIDE_FILE"
fi
PYTHON_BIN="$("$PROJECT_ROOT/scripts/ops/runtime_python.sh")"
GUARD="$PROJECT_ROOT/scripts/ops/maintenance_slot_guard.py"
SKIP_RC="${MAINTENANCE_SLOT_SKIP_EXIT_CODE:-75}"
JITTER_MAX_SECONDS="${MAINTENANCE_SLOT_JITTER_MAX_SECONDS:-90}"
NICE_LEVEL="${MAINTENANCE_SLOT_NICE_LEVEL:-${OPS_SUPPORT_JOB_NICE:-15}}"
BACKGROUND_POLICY="${MAINTENANCE_SLOT_BACKGROUND_POLICY:-${OPS_SUPPORT_JOBS_BACKGROUND_POLICY:-1}}"

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

if [[ "$BACKGROUND_POLICY" == "1" ]] && command -v taskpolicy >/dev/null 2>&1; then
  taskpolicy -b nice -n "$NICE_LEVEL" "$@"
else
  nice -n "$NICE_LEVEL" "$@"
fi
