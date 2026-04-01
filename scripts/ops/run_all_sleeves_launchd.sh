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
export LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"
export LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"
export LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"
export LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"
export LOG_FUTURES_MASTER_DECISIONS="${LOG_FUTURES_MASTER_DECISIONS:-1}"

# Enable paper mirror for Schwab sleeves while keeping live execution disabled.
export TOP_BOT_PAPER_TRADING_ENABLED="1"
export TOP_BOT_PAPER_TRADING_TOP_N="${TOP_BOT_PAPER_TRADING_TOP_N:-5}"
export TOP_BOT_PAPER_TRADING_MIN_ACC="${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.55}"
export TOP_BOT_PAPER_TRADING_PROFILES="default,conservative,aggressive,intraday_aggressive,swing_aggressive,dividend,bond"
export TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED="${TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED:-1}"
export TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N="${TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N:-2}"
export TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC="${TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC:-0.55}"
export TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES="${TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES:-default,conservative,aggressive,intraday_aggressive,swing_aggressive,dividend,bond}"
export PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"
export PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_all_sleeves.py" \
  --with-aggressive-modes \
  --disable-circuit-breakers
