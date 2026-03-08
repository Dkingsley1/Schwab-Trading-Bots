#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"
export MARKET_DATA_ONLY="${MARKET_DATA_ONLY:-1}"
export ALLOW_ORDER_EXECUTION="${ALLOW_ORDER_EXECUTION:-0}"
export MARKET_SESSION_START_HOUR="${MARKET_SESSION_START_HOUR:-4}"
export SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT="${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT:-1}"
export SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_MIN_AGE_SECONDS="${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_MIN_AGE_SECONDS:-300}"
export SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_ALLOWED_REASONS="${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_ALLOWED_REASONS:-incident_auto_halt,global_risk_killswitch,repeated_hard_gates}"
export SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_REQUIRE_PAPER_ONLY="${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_REQUIRE_PAPER_ONLY:-1}"

COINBASE_TOP_N="${TOP_BOT_PAPER_TRADING_TOP_N:-5}"
COINBASE_MIN_ACC="${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.55}"
COINBASE_PROFILES="${TOP_BOT_PAPER_TRADING_PROFILES:-default}"

COINBASE_FUTURES_PROFILE="${COINBASE_FUTURES_PROFILE:-crypto_futures}"
COINBASE_FUTURES_TOP_N="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}"
COINBASE_FUTURES_MIN_ACC="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.53}"
COINBASE_FUTURES_PROFILES="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-$COINBASE_FUTURES_PROFILE}"

SCHWAB_START_CMD="$PYTHON_BIN $PROJECT_ROOT/scripts/run_parallel_shadows.py"
AGGRESSIVE_START_CMD="$PYTHON_BIN $PROJECT_ROOT/scripts/run_parallel_aggressive_modes.py"
DIVIDEND_START_CMD="$PYTHON_BIN $PROJECT_ROOT/scripts/run_dividend_shadow.py --interval-seconds 60"
BOND_START_CMD="$PYTHON_BIN $PROJECT_ROOT/scripts/run_bond_shadow.py --interval-seconds 90"
COINBASE_START_CMD="$PROJECT_ROOT/scripts/ops/opsctl.sh coinbase-start --paper --live-data --top-n $COINBASE_TOP_N --min-acc $COINBASE_MIN_ACC --profiles $COINBASE_PROFILES"
COINBASE_FUTURES_START_CMD="$PROJECT_ROOT/scripts/ops/opsctl.sh coinbase-futures-start --paper --live-data --top-n $COINBASE_FUTURES_TOP_N --min-acc $COINBASE_FUTURES_MIN_ACC --profiles $COINBASE_FUTURES_PROFILES"

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/shadow_watchdog.py" \
  --watch-coinbase \
  --watch-coinbase-futures \
  --watch-aggressive-modes \
  --watch-dividend \
  --watch-bond \
  --interval-seconds "${SHADOW_WATCHDOG_INTERVAL_SECONDS:-20}" \
  --max-restarts-per-window "${SHADOW_WATCHDOG_MAX_RESTARTS_PER_WINDOW:-12}" \
  --restart-window-seconds "${SHADOW_WATCHDOG_RESTART_WINDOW_SECONDS:-3600}" \
  --schwab-heartbeat-stale-seconds "${SHADOW_WATCHDOG_SCHWAB_HEARTBEAT_STALE_SECONDS:-180}" \
  --coinbase-heartbeat-stale-seconds "${SHADOW_WATCHDOG_COINBASE_HEARTBEAT_STALE_SECONDS:-210}" \
  --schwab-start-cmd "$SCHWAB_START_CMD" \
  --aggressive-modes-start-cmd "$AGGRESSIVE_START_CMD" \
  --dividend-start-cmd "$DIVIDEND_START_CMD" \
  --bond-start-cmd "$BOND_START_CMD" \
  --coinbase-start-cmd "$COINBASE_START_CMD" \
  --coinbase-futures-start-cmd "$COINBASE_FUTURES_START_CMD" \
  --no-event-log
