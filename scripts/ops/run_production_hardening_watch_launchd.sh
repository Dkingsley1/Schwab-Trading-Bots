#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"
WATCH_NICE="${PRODUCTION_HARDENING_WATCH_NICE:-15}"
LOCK_ROOT="${PRODUCTION_HARDENING_WATCH_LOCK_ROOT:-${TMPDIR:-/tmp}/schwab_trading_bot}"
LOCK_DIR="$LOCK_ROOT/production_hardening_watch_launchd.lock"
LOCK_FILE="$LOCK_ROOT/production_hardening_watch_launchd.lockfile"

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

# An older wrapper may still own the directory lease. Never steal it by age.
if [[ -d "$LOCK_DIR" ]]; then
  print -u2 "production_hardening_watch deferred=legacy_lock_present"
  exit 75
fi
zmodload zsh/system
: >> "$LOCK_FILE"
if zsystem flock -t 0.01 -f LOCK_FD "$LOCK_FILE"; then
  :
else
  lock_rc=$?
  print -u2 "production_hardening_watch deferred=wrapper_lock_unavailable rc=$lock_rc"
  if [[ "$lock_rc" == "2" ]]; then
    exit 0
  fi
  exit "$lock_rc"
fi

trap 'exit 130' INT
trap 'exit 143' TERM

cycle_rc=0
run_observation_stage() {
  local stage="$1" stage_rc=0
  shift
  if "$@"; then
    print "production_hardening_watch stage=$stage rc=0"
  else
    stage_rc=$?
    print -u2 "production_hardening_watch stage=$stage rc=$stage_rc continuing_independent_observations=1"
    if [[ "$cycle_rc" == "0" ]]; then
      cycle_rc=$stage_rc
    fi
  fi
  return 0
}

WATCH_ARGS=(
  production-hardening-watch
  --apply
  --max-actions "${PRODUCTION_HARDENING_WATCH_MAX_ACTIONS:-8}"
  --max-execute-actions "${PRODUCTION_HARDENING_WATCH_MAX_EXECUTE_ACTIONS:-2}"
  --command-timeout-seconds "${PRODUCTION_HARDENING_WATCH_COMMAND_TIMEOUT_SECONDS:-240}"
  --json
)

if [[ "${PRODUCTION_HARDENING_WATCH_REFRESH_EVIDENCE:-1}" == "1" ]]; then
  run_observation_stage accrual /usr/bin/nice -n "$WATCH_NICE" "$PROJECT_ROOT/scripts/ops/opsctl.sh" \
    readiness-evidence-refresh \
    --apply \
    --profile "${READINESS_EVIDENCE_REFRESH_PROFILE:-accrual}" \
    --cooldown-minutes "${READINESS_EVIDENCE_REFRESH_COOLDOWN_MINUTES:-15}" \
    --timeout-seconds "${READINESS_EVIDENCE_REFRESH_STEP_TIMEOUT_SECONDS:-180}" \
    --json
fi

if [[ "${PRODUCTION_PILLAR_REFRESH_ENABLED:-1}" == "1" ]]; then
  run_observation_stage production /usr/bin/nice -n "$WATCH_NICE" "$PROJECT_ROOT/scripts/ops/opsctl.sh" \
    readiness-evidence-refresh \
    --apply \
    --profile production \
    --cooldown-minutes "${PRODUCTION_PILLAR_REFRESH_COOLDOWN_MINUTES:-45}" \
    --timeout-seconds "${PRODUCTION_PILLAR_REFRESH_STEP_TIMEOUT_SECONDS:-300}" \
    --json
fi

if [[ "$cycle_rc" == "0" ]]; then
  if [[ "${PRODUCTION_HARDENING_WATCH_EXECUTE_SAFE_REPAIRS:-0}" == "1" ]]; then
    WATCH_ARGS+=(--execute-safe-repairs)
  fi
  if [[ "${PRODUCTION_HARDENING_WATCH_EXECUTE_ON_WATCH:-0}" == "1" ]]; then
    WATCH_ARGS+=(--execute-on-watch)
  fi
else
  print -u2 "production_hardening_watch repair_execution=disabled upstream_refresh_failed=1"
fi
run_observation_stage watcher /usr/bin/nice -n "$WATCH_NICE" "$PROJECT_ROOT/scripts/ops/opsctl.sh" "${WATCH_ARGS[@]}"
exit "$cycle_rc"
