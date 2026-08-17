#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"
SMOOTH_NICE="${RUNTIME_SMOOTH_MODE_NICE:-20}"
LOCK_ROOT="${RUNTIME_SMOOTH_MODE_LOCK_ROOT:-${TMPDIR:-/tmp}/schwab_trading_bot}"
LOCK_DIR="$LOCK_ROOT/runtime_smooth_mode_launchd.lock"
LOCK_STALE_SECONDS="${RUNTIME_SMOOTH_MODE_LOCK_STALE_SECONDS:-300}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"
export BOT_PROTECTED_VOLUME_DENYLIST="${BOT_PROTECTED_VOLUME_DENYLIST:-/Volumes/VIDEO}"
export RUNTIME_SMOOTH_MODE_AUTOMATIC="${RUNTIME_SMOOTH_MODE_AUTOMATIC:-1}"

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

if [[ "${RUNTIME_SMOOTH_MODE_MEMORY_REFRESH:-1}" == "1" ]]; then
  /usr/bin/nice -n "$SMOOTH_NICE" "$PROJECT_ROOT/scripts/ops/opsctl.sh" \
    memory-pressure-intelligence --apply --json >/dev/null || true
fi

/usr/bin/nice -n "$SMOOTH_NICE" "$PROJECT_ROOT/scripts/ops/opsctl.sh" \
  runtime-throttle --apply --json >/dev/null

if [[ "${RUNTIME_SMOOTH_MODE_PAPER_REFRESH:-1}" == "1" ]]; then
  /usr/bin/nice -n "$SMOOTH_NICE" "$PROJECT_ROOT/scripts/ops/opsctl.sh" \
    paper-400-ramp --apply --json >/dev/null || true
  /usr/bin/nice -n "$SMOOTH_NICE" "$PROJECT_ROOT/scripts/ops/opsctl.sh" \
    paper-live-data-standard --apply --json >/dev/null || true
  /usr/bin/nice -n "$SMOOTH_NICE" "$PROJECT_ROOT/scripts/ops/opsctl.sh" \
    paper-trade-lock-infrabot --apply --json >/dev/null || true
fi

if [[ "${RUNTIME_SMOOTH_MODE_ADAPTIVE_NEEDS:-1}" == "1" ]]; then
  /usr/bin/nice -n "$SMOOTH_NICE" "$PROJECT_ROOT/scripts/ops/opsctl.sh" \
    infrabot-adaptive-governor \
    --apply \
    --refresh-needs \
    --max-actions "${RUNTIME_SMOOTH_MODE_ADAPTIVE_MAX_ACTIONS:-4}" \
    --execute-safe-repairs \
    --max-execute-actions "${RUNTIME_SMOOTH_MODE_ADAPTIVE_MAX_EXECUTE_ACTIONS:-3}" \
    --command-timeout-seconds "${RUNTIME_SMOOTH_MODE_ADAPTIVE_COMMAND_TIMEOUT_SECONDS:-240}" \
    --json >/dev/null || true
fi

print -r -- "runtime_smooth_mode status=complete profile=$BOT_RUNTIME_PROFILE"
