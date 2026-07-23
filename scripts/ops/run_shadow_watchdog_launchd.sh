#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export BOT_RUNTIME_LANE="${BOT_RUNTIME_LANE:-${BOT_SHADOW_RUNTIME_LANE:-canary314}}"
export BOT_PYTHON_VERSION="${BOT_PYTHON_VERSION:-3.14.5}"
export BOT_TRAINING_RUNTIME_LANE="${BOT_TRAINING_RUNTIME_LANE:-canary314}"
export BOT_TRAINING_PYTHON_VERSION="${BOT_TRAINING_PYTHON_VERSION:-3.14.5}"
export PY314_RUNTIME_FLIP_APPROVED="${PY314_RUNTIME_FLIP_APPROVED:-1}"
export PY314_RETIRE_312_ANCHOR="${PY314_RETIRE_312_ANCHOR:-1}"
unset __PYVENV_LAUNCHER__
BOOT_LOG="${SHADOW_WATCHDOG_BOOT_LOG:-$PROJECT_ROOT/logs/launchd_watchdog/shadow_watchdog.boot.log}"
mkdir -p "$(dirname "$BOOT_LOG")"
printf 'timestamp_utc=%s pid=%s ppid=%s profile=%s xpc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$PPID" "${BOT_RUNTIME_PROFILE:-live}" "${XPC_SERVICE_NAME:-}" >> "$BOOT_LOG" 2>/dev/null || true
source "$PROJECT_ROOT/scripts/ops/runtime_python.sh"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"

"$PROJECT_ROOT/.venv314/bin/python" "$PROJECT_ROOT/scripts/ops/apple_silicon_profile.py" apply >/dev/null 2>&1 || true
"$PROJECT_ROOT/.venv314/bin/python" "$PROJECT_ROOT/scripts/ops/computer_task_intelligence.py" --apply --json >/dev/null 2>&1 || true

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"
PYTHON_BIN="$(resolve_runtime_python)"
export MARKET_DATA_ONLY="${MARKET_DATA_ONLY:-1}"
export ALLOW_ORDER_EXECUTION="${ALLOW_ORDER_EXECUTION:-0}"
export PAPER_TRADE_LOCK_PATH="${PAPER_TRADE_LOCK_PATH:-$PROJECT_ROOT/governance/health/PAPER_TRADE_LOCK.flag}"
mkdir -p "$(dirname "$PAPER_TRADE_LOCK_PATH")"
printf 'enabled_at_utc=%s\npolicy=live_data_paper_trade_only\nmanaged_by=run_shadow_watchdog_launchd\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$PAPER_TRADE_LOCK_PATH"
export PAPER_TRADE_LOCK="${PAPER_TRADE_LOCK:-1}"
export TOP_BOT_ENABLE_LIVE_EXECUTION="${TOP_BOT_ENABLE_LIVE_EXECUTION:-0}"
export EXECUTION_LANE_LIVE_ENABLED="${EXECUTION_LANE_LIVE_ENABLED:-0}"
export RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR="${RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR:-0}"
export INLINE_PAPER_EXECUTION_ENABLED="${INLINE_PAPER_EXECUTION_ENABLED:-0}"
export MARKET_SESSION_START_HOUR="${MARKET_SESSION_START_HOUR:-4}"
export TOP_BOT_PAPER_TRADING_ENABLED="${TOP_BOT_PAPER_TRADING_ENABLED:-1}"
export TOP_BOT_PAPER_TRADING_TOP_N="${TOP_BOT_PAPER_TRADING_TOP_N:-5}"
export TOP_BOT_PAPER_TRADING_MIN_ACC="${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.55}"
export TOP_BOT_PAPER_TRADING_PROFILES="${TOP_BOT_PAPER_TRADING_PROFILES:-default,conservative,aggressive,intraday_aggressive,swing_aggressive,dividend,bond,fx}"
export TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED="${TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED:-1}"
export TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N="${TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N:-2}"
export TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC="${TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC:-0.55}"
export TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES="${TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES:-default,aggressive,intraday_aggressive,swing_aggressive,options_on_futures,options_on_futures_aggressive}"
export SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}"
export SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.53}"
export SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-schwab_futures}"
export COINBASE_TOP_BOT_PAPER_TRADING_TOP_N="${COINBASE_TOP_BOT_PAPER_TRADING_TOP_N:-${TOP_BOT_PAPER_TRADING_TOP_N:-5}}"
export COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC="${COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC:-${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.58}}"
export COINBASE_TOP_BOT_PAPER_TRADING_PROFILES="${COINBASE_TOP_BOT_PAPER_TRADING_PROFILES:-default}"
export COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}"
export COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.56}"
export COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-crypto_futures}"
export FX_WATCHDOG_PAPER_MODE="${FX_WATCHDOG_PAPER_MODE:-1}"
export SCHWAB_FUTURES_WATCHDOG_PAPER_MODE="${SCHWAB_FUTURES_WATCHDOG_PAPER_MODE:-1}"
export COINBASE_WATCHDOG_PAPER_MODE="${COINBASE_WATCHDOG_PAPER_MODE:-1}"
export COINBASE_FUTURES_WATCHDOG_PAPER_MODE="${COINBASE_FUTURES_WATCHDOG_PAPER_MODE:-1}"
export PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"
export PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"
export LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"
export LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"
export LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"
export LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"
export LOG_FUTURES_MASTER_DECISIONS="${LOG_FUTURES_MASTER_DECISIONS:-1}"
export SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT="${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT:-1}"
export SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_MIN_AGE_SECONDS="${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_MIN_AGE_SECONDS:-60}"
export SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_ALLOWED_REASONS="${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_ALLOWED_REASONS:-incident_auto_halt,global_risk_killswitch,repeated_hard_gates,softguard_api_circuit_opened}"
export SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_REQUIRE_PAPER_ONLY="${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_REQUIRE_PAPER_ONLY:-1}"
export SHADOW_WATCHDOG_ALLOW_SCHWAB_STANDBY_HEARTBEATS="${SHADOW_WATCHDOG_ALLOW_SCHWAB_STANDBY_HEARTBEATS:-1}"
export DIVIDEND_CAPTURE_SHADOW_ENABLED="${DIVIDEND_CAPTURE_SHADOW_ENABLED:-1}"

COINBASE_TOP_N="${COINBASE_TOP_BOT_PAPER_TRADING_TOP_N}"
COINBASE_MIN_ACC="${COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC}"
COINBASE_PROFILES="${COINBASE_TOP_BOT_PAPER_TRADING_PROFILES}"

SCHWAB_FUTURES_PROFILE="${SCHWAB_FUTURES_PROFILE:-schwab_futures}"
SCHWAB_FUTURES_TOP_N="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}"
SCHWAB_FUTURES_MIN_ACC="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.53}"
SCHWAB_FUTURES_PROFILES="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-$SCHWAB_FUTURES_PROFILE}"

COINBASE_FUTURES_PROFILE="${COINBASE_FUTURES_PROFILE:-crypto_futures}"
COINBASE_FUTURES_TOP_N="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-6}"
COINBASE_FUTURES_MIN_ACC="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.56}"
COINBASE_FUTURES_PROFILES="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-$COINBASE_FUTURES_PROFILE}"

json_argv() {
  "$PYTHON_BIN" - "$@" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:]))
PY
}

coinbase_watchdog_args() {
  local mode_var="$1"
  local top_n="$2"
  local min_acc="$3"
  local profiles="$4"
  local -a args=(--live-data --top-n "$top_n" --min-acc "$min_acc" --profiles "$profiles")
  if [[ "${mode_var}" == "1" ]]; then
    args=(--paper "${args[@]}")
  fi
  printf '%s\n' "${args[@]}"
}

SCHWAB_START_CMD="$(json_argv "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_all_sleeves.py" --with-aggressive-modes)"
AGGRESSIVE_START_CMD="$(json_argv "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_parallel_aggressive_modes.py")"
DIVIDEND_START_CMD="$(json_argv "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_dividend_shadow.py" --interval-seconds 60)"
DIVIDEND_CAPTURE_START_CMD="$(json_argv "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_dividend_capture_shadow.py" --interval-seconds 60)"
BOND_START_CMD="$(json_argv "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_bond_shadow.py" --interval-seconds 120)"
FX_START_CMD="$(json_argv "$PROJECT_ROOT/scripts/ops/opsctl.sh" fx-start --paper --live-data)"
SCHWAB_FUTURES_START_CMD="$(json_argv "$PROJECT_ROOT/scripts/ops/opsctl.sh" schwab-futures-start $(coinbase_watchdog_args "$SCHWAB_FUTURES_WATCHDOG_PAPER_MODE" "$SCHWAB_FUTURES_TOP_N" "$SCHWAB_FUTURES_MIN_ACC" "$SCHWAB_FUTURES_PROFILES"))"
COINBASE_START_CMD="$(json_argv "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-start $(coinbase_watchdog_args "$COINBASE_WATCHDOG_PAPER_MODE" "$COINBASE_TOP_N" "$COINBASE_MIN_ACC" "$COINBASE_PROFILES"))"
COINBASE_FUTURES_START_CMD="$(json_argv "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-futures-start $(coinbase_watchdog_args "$COINBASE_FUTURES_WATCHDOG_PAPER_MODE" "$COINBASE_FUTURES_TOP_N" "$COINBASE_FUTURES_MIN_ACC" "$COINBASE_FUTURES_PROFILES"))"

WATCH_ARGS=(
  --watch-schwab-futures
  --watch-coinbase
  --watch-coinbase-futures
)

schwab_credentials_ready_for_watchdog() {
  local key="${SCHWAB_API_KEY:-}"
  local secret="${SCHWAB_SECRET:-}"
  case "$key" in
    ""|"YOUR_KEY_HERE"|"YOUR_REAL_KEY"|"<real_key>") return 1 ;;
  esac
  case "$secret" in
    ""|"YOUR_SECRET_HERE"|"YOUR_REAL_SECRET"|"<real_secret>") return 1 ;;
  esac
  return 0
}

if ! schwab_credentials_ready_for_watchdog; then
  WATCH_ARGS+=(--schwab-futures-optional)
fi

if [[ "${SHADOW_WATCHDOG_DIRECT_CHILD_SLEEVES:-0}" == "1" ]]; then
  WATCH_ARGS+=(
    --watch-aggressive-modes
    --watch-dividend
    --watch-bond
    --watch-fx
  )
  if [[ "$DIVIDEND_CAPTURE_SHADOW_ENABLED" == "1" ]]; then
    WATCH_ARGS+=(--watch-dividend-capture)
  fi
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/shadow_watchdog.py" \
  "${WATCH_ARGS[@]}" \
  --interval-seconds "${SHADOW_WATCHDOG_INTERVAL_SECONDS:-20}" \
  --max-restarts-per-window "${SHADOW_WATCHDOG_MAX_RESTARTS_PER_WINDOW:-12}" \
  --restart-window-seconds "${SHADOW_WATCHDOG_RESTART_WINDOW_SECONDS:-3600}" \
  --schwab-heartbeat-stale-seconds "${SHADOW_WATCHDOG_SCHWAB_HEARTBEAT_STALE_SECONDS:-180}" \
  --coinbase-heartbeat-stale-seconds "${SHADOW_WATCHDOG_COINBASE_HEARTBEAT_STALE_SECONDS:-210}" \
  --schwab-start-cmd "$SCHWAB_START_CMD" \
  --schwab-futures-start-cmd "$SCHWAB_FUTURES_START_CMD" \
  --aggressive-modes-start-cmd "$AGGRESSIVE_START_CMD" \
  --dividend-start-cmd "$DIVIDEND_START_CMD" \
  --dividend-capture-start-cmd "$DIVIDEND_CAPTURE_START_CMD" \
  --bond-start-cmd "$BOND_START_CMD" \
  --fx-start-cmd "$FX_START_CMD" \
  --coinbase-start-cmd "$COINBASE_START_CMD" \
  --coinbase-futures-start-cmd "$COINBASE_FUTURES_START_CMD" \
  --no-event-log
