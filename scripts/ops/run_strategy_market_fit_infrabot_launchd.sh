#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

PYTHON_BIN="$($PROJECT_ROOT/scripts/ops/runtime_python.sh)"
export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"
export MARKET_DATA_ONLY=1
export ALLOW_ORDER_EXECUTION=0
export BOT_LIVE_MONEY_LOCKED_DURING_SOAK=1
export BOT_UNATTENDED_SOAK_ACTIVE=1
export MAINTENANCE_SLOT_DEFER_OUTSIDE_QUIET_WINDOW=0
export MAINTENANCE_SLOT_DEFER_WHILE_SQL_LINK_ACTIVE=0
export MAINTENANCE_SLOT_MAX_RUNTIME_SECONDS="${STRATEGY_MARKET_FIT_MAX_RUNTIME_SECONDS:-180}"
export MAINTENANCE_SLOT_JITTER_MAX_SECONDS="${STRATEGY_MARKET_FIT_JITTER_MAX_SECONDS:-60}"
export MAINTENANCE_SLOT_NICE_LEVEL="${STRATEGY_MARKET_FIT_NICE_LEVEL:-15}"
export MAINTENANCE_SLOT_BACKGROUND_POLICY=1

exec "$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" \
  strategy_market_fit_infrabot \
  "$PYTHON_BIN" \
  "$PROJECT_ROOT/scripts/ops/strategy_market_fit_infrabot.py" \
  --json
