#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"
PYTHON_BIN="$PROJECT_ROOT/.venv314/bin/python"
cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi
export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"
export RUNTIME_SMOOTH_MODE_AUTOMATIC=1
export MARKET_DATA_ONLY=1
export ALLOW_ORDER_EXECUTION=0
export TOP_BOT_ENABLE_LIVE_EXECUTION=0
export EXECUTION_LANE_LIVE_ENABLED=0

# Legacy locks are not stolen by age. The new owner uses a kernel-held flock.
LEGACY_LOCK="${RUNTIME_SMOOTH_MODE_LOCK_ROOT:-${TMPDIR:-/tmp}/schwab_trading_bot}/runtime_smooth_mode_launchd.lock"
if [[ -d "$LEGACY_LOCK" ]]; then
  print -u2 "runtime_smooth_mode deferred=legacy_owner_lock_present"
  exit 75
fi
# Short observation/decision work uses observability priority; bulk repairs stay
# with production_hardening_watch and never occupy this 60-second control loop.
exec /usr/bin/nice -n "${RUNTIME_SMOOTH_MODE_NICE:-10}" "$PYTHON_BIN" \
  "$PROJECT_ROOT/scripts/ops/governor_refresh.py" --scheduled --json
