#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$PROJECT_ROOT/.venv312/bin/python"

FORCE_RESTART=0
WITH_COINBASE=1
SIMULATE=0
DISABLE_BREAKERS=0
COINBASE_PAPER=0
COINBASE_SIMULATE="${COINBASE_START_SIMULATE:-1}"
PROFILE="${BOT_RUNTIME_PROFILE:-}"
ORCHESTRATOR_MODE="${STACK_ORCHESTRATOR_MODE:-watchdog}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-restart) FORCE_RESTART=1 ;;
    --no-coinbase) WITH_COINBASE=0 ;;
    --simulate) SIMULATE=1 ;;
    --disable-circuit-breakers) DISABLE_BREAKERS=1 ;;
    --coinbase-paper) COINBASE_PAPER=1; COINBASE_SIMULATE=0 ;;
    --coinbase-live-data|--coinbase-no-simulate) COINBASE_SIMULATE=0 ;;
    --coinbase-simulate) COINBASE_SIMULATE=1 ;;
    --profile) PROFILE="${2:-$PROFILE}"; shift ;;
    --orchestrator-mode) ORCHESTRATOR_MODE="${2:-$ORCHESTRATOR_MODE}"; shift ;;
    --watchdog-only) ORCHESTRATOR_MODE="watchdog" ;;
    --run-all-sleeves) ORCHESTRATOR_MODE="all_sleeves" ;;
  esac
  shift
done

cd "$PROJECT_ROOT"

if [[ -z "$PROFILE" ]]; then
  if [[ "$SIMULATE" == "1" ]]; then
    PROFILE="sim"
  else
    PROFILE="live"
  fi
fi

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

echo "runtime_profile=$PROFILE"
echo "orchestrator_mode=$ORCHESTRATOR_MODE"

if [[ "$FORCE_RESTART" == "1" ]]; then
  # Clean sweep so stale wrappers/children do not keep locks and destabilize the supervisor.
  pkill -f "scripts/run_all_sleeves.py" || true
  pkill -f "scripts/run_parallel_shadows.py" || true
  pkill -f "scripts/run_parallel_aggressive_modes.py" || true
  pkill -f "scripts/run_dividend_shadow.py" || true
  pkill -f "scripts/run_bond_shadow.py" || true
  pkill -f "scripts/run_shadow_training_loop.py --broker schwab" || true
  pkill -f "scripts/run_shadow_training_loop.py --broker coinbase" || true
  sleep 1
fi

"$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true
"$PY" "$PROJECT_ROOT/scripts/ops/storage_failback_sync.py" --json >/dev/null 2>&1 || true
PREFLIGHT_ARGS=(--broker "${DATA_BROKER:-schwab}" --json)
if [[ "$SIMULATE" == "1" ]]; then
  PREFLIGHT_ARGS+=(--simulate)
fi
if [[ "${OPS_PREFLIGHT_APPLY_KILL_DUPLICATES:-1}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--apply-kill-duplicates)
fi
"$PY" "$PROJECT_ROOT/scripts/ops/preflight_autofix.py" "${PREFLIGHT_ARGS[@]}" || true

export MARKET_DATA_ONLY="${MARKET_DATA_ONLY:-1}"
export ALLOW_ORDER_EXECUTION="${ALLOW_ORDER_EXECUTION:-0}"

if [[ "$ORCHESTRATOR_MODE" == "watchdog" ]]; then
  WD_MATCH="scripts/shadow_watchdog.py"
  WD_PLIST="$HOME/Library/LaunchAgents/com.dankingsley.shadow_watchdog.plist"
  if [[ "$FORCE_RESTART" == "1" ]]; then
    pkill -f "$WD_MATCH" >/dev/null 2>&1 || true
    sleep 1
    if [[ -x "$PROJECT_ROOT/scripts/install_shadow_watchdog_launchd.sh" ]]; then
      "$PROJECT_ROOT/scripts/install_shadow_watchdog_launchd.sh" >/dev/null 2>&1 || true
    elif [[ -f "$WD_PLIST" ]]; then
      launchctl unload "$WD_PLIST" >/dev/null 2>&1 || true
      launchctl load "$WD_PLIST" >/dev/null 2>&1 || true
    elif [[ -x "$PROJECT_ROOT/scripts/ops/run_shadow_watchdog_launchd.sh" ]]; then
      WD_LOG="logs/shadow_watchdog_manual_$(date -u +%Y%m%d_%H%M%S).log"
      PYTHONUNBUFFERED=1 nohup "$PROJECT_ROOT/scripts/ops/run_shadow_watchdog_launchd.sh" > "$WD_LOG" 2>&1 & disown
      echo "shadow_watchdog_log=$WD_LOG"
    fi
    sleep 2
    if ps -axo command | grep -F "$WD_MATCH" | grep -v grep >/dev/null 2>&1; then
      WD_PID="$(ps -axo pid,command | grep -F "$WD_MATCH" | grep -v grep | awk 'NR==1{print $1}')"
      echo "shadow_watchdog=reloaded pid=$WD_PID"
    else
      echo "shadow_watchdog=failed_to_restart"
      exit 1
    fi
  elif ps -axo command | grep -F "$WD_MATCH" | grep -v grep >/dev/null 2>&1; then
    WD_PID="$(ps -axo pid,command | grep -F "$WD_MATCH" | grep -v grep | awk 'NR==1{print $1}')"
    echo "shadow_watchdog=already_running pid=$WD_PID"
  else
    if [[ -x "$PROJECT_ROOT/scripts/install_shadow_watchdog_launchd.sh" ]]; then
      "$PROJECT_ROOT/scripts/install_shadow_watchdog_launchd.sh" >/dev/null 2>&1 || true
    elif [[ -f "$WD_PLIST" ]]; then
      launchctl unload "$WD_PLIST" >/dev/null 2>&1 || true
      launchctl load "$WD_PLIST" >/dev/null 2>&1 || true
    elif [[ -x "$PROJECT_ROOT/scripts/ops/run_shadow_watchdog_launchd.sh" ]]; then
      WD_LOG="logs/shadow_watchdog_manual_$(date -u +%Y%m%d_%H%M%S).log"
      PYTHONUNBUFFERED=1 nohup "$PROJECT_ROOT/scripts/ops/run_shadow_watchdog_launchd.sh" > "$WD_LOG" 2>&1 & disown
      echo "shadow_watchdog_log=$WD_LOG"
    fi

    sleep 2
    if ps -axo command | grep -F "$WD_MATCH" | grep -v grep >/dev/null 2>&1; then
      WD_PID="$(ps -axo pid,command | grep -F "$WD_MATCH" | grep -v grep | awk 'NR==1{print $1}')"
      echo "shadow_watchdog=started pid=$WD_PID"
    else
      echo "shadow_watchdog=failed_to_start"
      exit 1
    fi
  fi

  OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
  echo "stack_start_delegated_to=shadow_watchdog"
  exit 0
fi

LOG_ALL="logs/all_sleeves_$(date -u +%Y%m%d_%H%M%S).log"
CMD=("$PY" "$PROJECT_ROOT/scripts/run_all_sleeves.py" --with-aggressive-modes)
if [[ "$SIMULATE" == "1" ]]; then
  CMD+=(--simulate)
fi
if [[ "$DISABLE_BREAKERS" == "1" ]]; then
  CMD+=(--disable-circuit-breakers)
fi

PYTHONUNBUFFERED=1 nohup "${CMD[@]}" > "$LOG_ALL" 2>&1 & disown

echo "all_sleeves_log=$LOG_ALL"

if [[ "$WITH_COINBASE" == "1" ]]; then
  if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase" | grep -v grep >/dev/null 2>&1; then
    EXISTING_PID="$(ps -axo pid,command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase" | grep -v grep | awk 'NR==1{print $1}')"
    echo "coinbase_loop=already_running pid=$EXISTING_PID"
  else
    LOG_CB="logs/coinbase_live_$(date -u +%Y%m%d_%H%M%S).log"
    CB_CMD=(
      "$PY" "$PROJECT_ROOT/scripts/run_shadow_training_loop.py"
      --broker coinbase
      --symbols "${COINBASE_WATCH_SYMBOLS:-BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD}"
      --interval-seconds "${COINBASE_WATCH_INTERVAL_SECONDS:-20}"
      --max-iterations 0
    )
    if [[ "$COINBASE_SIMULATE" == "1" ]]; then
      CB_CMD+=(--simulate)
    fi

    if [[ "$COINBASE_PAPER" == "1" ]]; then
      COINBASE_PAPER_TOP_N="${COINBASE_TOP_BOT_PAPER_TRADING_TOP_N:-${TOP_BOT_PAPER_TRADING_TOP_N:-5}}"
      COINBASE_PAPER_MIN_ACC="${COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC:-${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.58}}"
      COINBASE_PAPER_PROFILES="${COINBASE_TOP_BOT_PAPER_TRADING_PROFILES:-${TOP_BOT_PAPER_TRADING_PROFILES:-default}}"
      echo "coinbase_paper=enabled top_n=$COINBASE_PAPER_TOP_N min_acc=$COINBASE_PAPER_MIN_ACC"
      TOP_BOT_PAPER_TRADING_ENABLED=1       TOP_BOT_PAPER_TRADING_TOP_N="$COINBASE_PAPER_TOP_N"       TOP_BOT_PAPER_TRADING_MIN_ACC="$COINBASE_PAPER_MIN_ACC"       TOP_BOT_PAPER_TRADING_PROFILES="$COINBASE_PAPER_PROFILES"       PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"       PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"       ADAPTIVE_INTERVAL_ENABLED="${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-1}"       PYTHONUNBUFFERED=1 nohup "${CB_CMD[@]}" > "$LOG_CB" 2>&1 & disown
    else
      ADAPTIVE_INTERVAL_ENABLED="${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-1}"       PYTHONUNBUFFERED=1 nohup "${CB_CMD[@]}" > "$LOG_CB" 2>&1 & disown
    fi

    sleep 2
    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase" | grep -v grep >/dev/null 2>&1; then
      echo "coinbase_log=$LOG_CB"
      echo "coinbase_mode simulate=$COINBASE_SIMULATE paper=$COINBASE_PAPER"
    else
      echo "coinbase_loop=failed_to_start log=$LOG_CB"
      tail -n 40 "$LOG_CB" || true
    fi
  fi
fi

OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
