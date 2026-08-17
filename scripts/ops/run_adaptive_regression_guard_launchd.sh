#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"
PYTHON_BIN="$("$PROJECT_ROOT/scripts/ops/runtime_python.sh")"
export PYTHONUNBUFFERED=1
SKIP_RC="${MAINTENANCE_SLOT_SKIP_EXIT_CODE:-75}"
JITTER_MAX_SECONDS="${ADAPTIVE_REGRESSION_GUARD_JITTER_MAX_SECONDS:-30}"
NICE_LEVEL="${ADAPTIVE_REGRESSION_GUARD_NICE:-15}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

if [[ "${MAINTENANCE_SLOT_DISABLE_JITTER:-0}" != "1" ]] && [[ "$JITTER_MAX_SECONDS" == <-> ]] && (( JITTER_MAX_SECONDS > 0 )); then
  sleep $(( RANDOM % (JITTER_MAX_SECONDS + 1) ))
fi

set +e
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/maintenance_slot_guard.py" \
  --slot adaptive_regression_guard \
  --begin \
  --max-load-ratio "${ADAPTIVE_REGRESSION_MAINTENANCE_MAX_LOAD_RATIO:-4.0}" \
  --max-five-min-load-ratio "${ADAPTIVE_REGRESSION_MAINTENANCE_MAX_FIVE_MIN_LOAD_RATIO:-4.0}" \
  --min-interval-seconds "${ADAPTIVE_REGRESSION_GUARD_MIN_INTERVAL_SECONDS:-240}" \
  --no-defer-while-sql-link-active \
  --no-defer-outside-quiet-window \
  --smooth-gate-exempt-slots "${MAINTENANCE_SLOT_SMOOTH_GATE_EXEMPT_SLOTS:-sql_link_writer,storage_backpressure_autopilot,storage_pressure_clearance,storage_reconnect_infrabot,storage_eject_guard,runtime_smooth_mode,failover_watch,shadow_watchdog,mac_notification_watch,observability_exporter,premarket_token_guard,adaptive_regression_guard}"
guard_rc=$?
set -e
if [[ "$guard_rc" != "0" ]]; then
  if [[ "$guard_rc" == "$SKIP_RC" ]]; then
    exit 0
  fi
  exit "$guard_rc"
fi

cleanup() {
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/maintenance_slot_guard.py" --slot adaptive_regression_guard --end >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

ADAPTIVE_ARGS=(
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/adaptive_regression_guard.py"
  --apply
  --persistence-threshold "${ADAPTIVE_REGRESSION_PERSISTENCE_THRESHOLD:-3}"
  --blocked-escalation-threshold "${ADAPTIVE_REGRESSION_BLOCKED_ESCALATION_THRESHOLD:-2}"
  --max-artifact-age-minutes "${ADAPTIVE_REGRESSION_MAX_ARTIFACT_AGE_MINUTES:-60}"
  --json
)

set +e
if command -v taskpolicy >/dev/null 2>&1; then
  taskpolicy -b nice -n "$NICE_LEVEL" "${ADAPTIVE_ARGS[@]}"
  adaptive_rc=$?
else
  nice -n "$NICE_LEVEL" "${ADAPTIVE_ARGS[@]}"
  adaptive_rc=$?
fi
set -e

if [[ "$adaptive_rc" == "0" || "$adaptive_rc" == "2" ]]; then
  exit 0
fi
exit "$adaptive_rc"
