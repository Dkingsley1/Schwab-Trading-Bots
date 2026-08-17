#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"
WATCH_NICE="${PRODUCTION_HARDENING_WATCH_NICE:-15}"
LOCK_ROOT="${PRODUCTION_HARDENING_WATCH_LOCK_ROOT:-${TMPDIR:-/tmp}/schwab_trading_bot}"
LOCK_DIR="$LOCK_ROOT/production_hardening_watch_launchd.lock"
LOCK_STALE_SECONDS="${PRODUCTION_HARDENING_WATCH_LOCK_STALE_SECONDS:-300}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"
export MARKET_DATA_ONLY=1
export ALLOW_ORDER_EXECUTION=0
export TOP_BOT_ENABLE_LIVE_EXECUTION=0
export EXECUTION_LANE_LIVE_ENABLED=0
export BOT_LIVE_MONEY_LOCKED_DURING_SOAK=1
export BOT_UNATTENDED_SOAK_ACTIVE="${BOT_UNATTENDED_SOAK_ACTIVE:-1}"

mkdir -p "$LOCK_ROOT"

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    return 0
  fi

  local now_epoch lock_epoch
  now_epoch="$(date +%s)"
  lock_epoch="$(stat -f %m "$LOCK_DIR" 2>/dev/null || stat -c %Y "$LOCK_DIR" 2>/dev/null || echo 0)"
  if [[ "$now_epoch" == <-> ]] && [[ "$lock_epoch" == <-> ]] \
    && (( now_epoch - lock_epoch > LOCK_STALE_SECONDS )); then
    rmdir "$LOCK_DIR" >/dev/null 2>&1 || true
    mkdir "$LOCK_DIR" 2>/dev/null
    return $?
  fi
  return 1
}

if ! acquire_lock; then
  exit 0
fi

cleanup() {
  rmdir "$LOCK_DIR" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

WATCH_ARGS=(
  production-hardening-watch
  --apply
  --max-actions "${PRODUCTION_HARDENING_WATCH_MAX_ACTIONS:-8}"
  --max-execute-actions "${PRODUCTION_HARDENING_WATCH_MAX_EXECUTE_ACTIONS:-2}"
  --command-timeout-seconds "${PRODUCTION_HARDENING_WATCH_COMMAND_TIMEOUT_SECONDS:-240}"
  --json
)

if [[ "${PRODUCTION_HARDENING_WATCH_EXECUTE_SAFE_REPAIRS:-0}" == "1" ]]; then
  WATCH_ARGS+=(--execute-safe-repairs)
fi

if [[ "${PRODUCTION_HARDENING_WATCH_EXECUTE_ON_WATCH:-0}" == "1" ]]; then
  WATCH_ARGS+=(--execute-on-watch)
fi

if [[ "${PRODUCTION_HARDENING_WATCH_REFRESH_EVIDENCE:-1}" == "1" ]]; then
  /usr/bin/nice -n "$WATCH_NICE" "$PROJECT_ROOT/scripts/ops/opsctl.sh" \
    readiness-evidence-refresh \
    --apply \
    --profile "${READINESS_EVIDENCE_REFRESH_PROFILE:-accrual}" \
    --cooldown-minutes "${READINESS_EVIDENCE_REFRESH_COOLDOWN_MINUTES:-15}" \
    --timeout-seconds "${READINESS_EVIDENCE_REFRESH_STEP_TIMEOUT_SECONDS:-180}" \
    --json
fi

if [[ "${PRODUCTION_PILLAR_REFRESH_ENABLED:-1}" == "1" ]]; then
  /usr/bin/nice -n "$WATCH_NICE" "$PROJECT_ROOT/scripts/ops/opsctl.sh" \
    readiness-evidence-refresh \
    --apply \
    --profile production \
    --cooldown-minutes "${PRODUCTION_PILLAR_REFRESH_COOLDOWN_MINUTES:-45}" \
    --timeout-seconds "${PRODUCTION_PILLAR_REFRESH_STEP_TIMEOUT_SECONDS:-300}" \
    --json
fi

/usr/bin/nice -n "$WATCH_NICE" "$PROJECT_ROOT/scripts/ops/opsctl.sh" "${WATCH_ARGS[@]}"
