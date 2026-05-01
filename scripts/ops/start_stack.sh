#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/ops/runtime_python.sh"
PY="$(resolve_runtime_python)"
HEALTH_DIR="$PROJECT_ROOT/governance/health"
OPERATOR_STOP_FLAG="$HEALTH_DIR/OPERATOR_STOP.flag"
GLOBAL_HALT_FLAG="$HEALTH_DIR/GLOBAL_TRADING_HALT.flag"
PAPER_TRADE_LOCK_FILE="$HEALTH_DIR/PAPER_TRADE_LOCK.flag"

FORCE_RESTART=0
WITH_COINBASE=1
SIMULATE=0
DISABLE_BREAKERS=0
SCHWAB_PAPER=1
COINBASE_PAPER=1
COINBASE_SIMULATE="${COINBASE_START_SIMULATE:-0}"
PROFILE="${BOT_RUNTIME_PROFILE:-}"
ORCHESTRATOR_MODE="${STACK_ORCHESTRATOR_MODE:-watchdog}"
DRY_RUN=0

flag_reason() {
  local path="$1"
  "$PY" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    payload = {}

reason = str(payload.get("reason") or "").strip()
timestamp = str(payload.get("timestamp_utc") or "").strip()
operator = str(payload.get("operator") or "").strip()
parts = []
if reason:
    parts.append(f"reason={reason}")
if operator:
    parts.append(f"operator={operator}")
if timestamp:
    parts.append(f"timestamp_utc={timestamp}")
print(" ".join(parts))
PY
}

abort_for_safety_flags() {
  local blocked=0

  if [[ -f "$OPERATOR_STOP_FLAG" ]]; then
    blocked=1
    echo "stack_start_blocked=operator_stop"
    echo "operator_stop_flag=$OPERATOR_STOP_FLAG"
    echo "operator_stop_detail=$(flag_reason "$OPERATOR_STOP_FLAG")"
  fi

  if [[ -f "$GLOBAL_HALT_FLAG" ]]; then
    blocked=1
    echo "stack_start_blocked=global_halt"
    echo "global_halt_flag=$GLOBAL_HALT_FLAG"
    echo "global_halt_detail=$(flag_reason "$GLOBAL_HALT_FLAG")"
  fi

  if [[ "$blocked" == "1" ]]; then
    echo "stack_start_status=blocked_by_safety_flags"
    echo "review_halt_status=./scripts/ops/opsctl.sh global-halt-status --json"
    echo "refresh_global_halt_blockers=./scripts/ops/opsctl.sh global-halt-refresh --json"
    echo "release_operator_stop=./scripts/ops/opsctl.sh operator-release --json"
    echo "attempt_safe_global_halt_clear=./scripts/ops/opsctl.sh global-halt-auto-clear --json"
    echo "manual_clear_all_halts=./scripts/ops/opsctl.sh clear-all-halts --json"
    exit 2
  fi
}

enable_paper_trade_lock() {
  mkdir -p "$(dirname "$PAPER_TRADE_LOCK_FILE")"
  printf 'enabled_at_utc=%s\npolicy=live_data_paper_trade_only\nmanaged_by=start_stack\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$PAPER_TRADE_LOCK_FILE"
}

paper_trade_lock_env() {
  enable_paper_trade_lock
  export PAPER_TRADE_LOCK=1
  export PAPER_TRADE_LOCK_PATH="$PAPER_TRADE_LOCK_FILE"
  export MARKET_DATA_ONLY=1
  export ALLOW_ORDER_EXECUTION=0
  export TOP_BOT_ENABLE_LIVE_EXECUTION=0
  export EXECUTION_LANE_LIVE_ENABLED=0
  export RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR=0
  export INLINE_PAPER_EXECUTION_ENABLED=0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-restart) FORCE_RESTART=1 ;;
    --no-coinbase) WITH_COINBASE=0 ;;
    --simulate) SIMULATE=1 ;;
    --disable-circuit-breakers) DISABLE_BREAKERS=1 ;;
    --paper|--schwab-paper) SCHWAB_PAPER=1 ;;
    --coinbase-paper) COINBASE_PAPER=1; COINBASE_SIMULATE=0 ;;
    --coinbase-live-data|--coinbase-no-simulate) COINBASE_SIMULATE=0 ;;
    --coinbase-simulate) COINBASE_SIMULATE=1 ;;
    --profile) PROFILE="${2:-$PROFILE}"; shift ;;
    --orchestrator-mode) ORCHESTRATOR_MODE="${2:-$ORCHESTRATOR_MODE}"; shift ;;
    --watchdog-only) ORCHESTRATOR_MODE="watchdog" ;;
    --run-all-sleeves) ORCHESTRATOR_MODE="all_sleeves" ;;
    --dry-run) DRY_RUN=1 ;;
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

abort_for_safety_flags

if [[ "$DRY_RUN" == "1" ]]; then
  echo "stack_start_dry_run=1"
  echo "stack_start_status=ready_to_launch"
  echo "runtime_profile=$PROFILE"
  echo "orchestrator_mode=$ORCHESTRATOR_MODE"
  echo "with_coinbase=$WITH_COINBASE"
  echo "simulate=$SIMULATE"
  echo "schwab_paper=$SCHWAB_PAPER"
  echo "coinbase_paper=$COINBASE_PAPER"
  echo "coinbase_simulate=$COINBASE_SIMULATE"
  exit 0
fi

"$PY" "$PROJECT_ROOT/scripts/ops/apple_silicon_profile.py" apply >/dev/null 2>&1 || true
"$PY" "$PROJECT_ROOT/scripts/ops/portable_brain_contract.py" apply >/dev/null 2>&1 || true
"$PY" "$PROJECT_ROOT/scripts/resource_guard.py" --profile refresh --json >/dev/null 2>&1 || true
"$PY" "$PROJECT_ROOT/scripts/ops/memory_efficiency_control.py" apply >/dev/null 2>&1 || true

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

coinbase_spot_process_lines() {
  ps -axo pid,command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase" | grep -v " --profile crypto_futures" | grep -v grep || true
}

coinbase_spot_running() {
  coinbase_spot_process_lines | grep -q .
}

kill_coinbase_spot_loops() {
  local pids
  pids="$(coinbase_spot_process_lines | awk '{print $1}')"
  if [[ -n "${pids//[[:space:]]/}" ]]; then
    while IFS= read -r pid; do
      [[ -n "$pid" ]] && kill "$pid" >/dev/null 2>&1 || true
    done <<< "$pids"
  fi
}

echo "runtime_profile=$PROFILE"
echo "orchestrator_mode=$ORCHESTRATOR_MODE"

if [[ "$FORCE_RESTART" == "1" ]]; then
  # Clean sweep so stale wrappers/children do not keep locks and destabilize the supervisor.
  pkill -f "scripts/run_all_sleeves.py" || true
  pkill -f "scripts/run_parallel_shadows.py" || true
  pkill -f "scripts/run_parallel_aggressive_modes.py" || true
  pkill -f "scripts/run_dividend_shadow.py" || true
  pkill -f "scripts/run_dividend_capture_shadow.py" || true
  pkill -f "scripts/run_bond_shadow.py" || true
  pkill -f "scripts/run_fx_shadow.py" || true
  pkill -f "scripts/run_shadow_training_loop.py --broker schwab" || true
  kill_coinbase_spot_loops
  pkill -f "scripts/run_shadow_training_loop.py --broker coinbase --profile crypto_futures" || true
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
export TOP_BOT_PAPER_TRADING_ENABLED="${TOP_BOT_PAPER_TRADING_ENABLED:-1}"
export TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED="${TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED:-1}"
export PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS="${PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS:-1}"
export PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"
export PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"
export LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"
export LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"
export LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"
export LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"
export LOG_FUTURES_MASTER_DECISIONS="${LOG_FUTURES_MASTER_DECISIONS:-1}"

if [[ "$SCHWAB_PAPER" == "1" || "$COINBASE_PAPER" == "1" ]]; then
  paper_trade_lock_env
fi

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

if [[ "$SCHWAB_PAPER" == "1" ]]; then
  SCHWAB_PAPER_TOP_N="${SCHWAB_TOP_BOT_PAPER_TRADING_TOP_N:-${TOP_BOT_PAPER_TRADING_TOP_N:-5}}"
  SCHWAB_PAPER_MIN_ACC="${SCHWAB_TOP_BOT_PAPER_TRADING_MIN_ACC:-${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.58}}"
  SCHWAB_PAPER_PROFILES="${SCHWAB_TOP_BOT_PAPER_TRADING_PROFILES:-${TOP_BOT_PAPER_TRADING_PROFILES:-}}"
  SCHWAB_OPTIONS_PAPER_TOP_N="${SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_TOP_N:-${TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N:-2}}"
  SCHWAB_OPTIONS_PAPER_MIN_ACC="${SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_MIN_ACC:-${TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC:-$SCHWAB_PAPER_MIN_ACC}}"
  SCHWAB_OPTIONS_PAPER_PROFILES="${SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_PROFILES:-${TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES:-default,aggressive,intraday_aggressive,swing_aggressive,options_on_futures,options_on_futures_aggressive}}"
  echo "schwab_paper=enabled top_n=$SCHWAB_PAPER_TOP_N min_acc=$SCHWAB_PAPER_MIN_ACC profiles=${SCHWAB_PAPER_PROFILES:-all}"
  echo "schwab_options_paper=enabled top_n=$SCHWAB_OPTIONS_PAPER_TOP_N min_acc=$SCHWAB_OPTIONS_PAPER_MIN_ACC profiles=${SCHWAB_OPTIONS_PAPER_PROFILES:-all}"
  TOP_BOT_PAPER_TRADING_ENABLED=1 \
  TOP_BOT_PAPER_TRADING_TOP_N="$SCHWAB_PAPER_TOP_N" \
  TOP_BOT_PAPER_TRADING_MIN_ACC="$SCHWAB_PAPER_MIN_ACC" \
  TOP_BOT_PAPER_TRADING_PROFILES="$SCHWAB_PAPER_PROFILES" \
  TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED="${TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED:-1}" \
  TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N="$SCHWAB_OPTIONS_PAPER_TOP_N" \
  TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC="$SCHWAB_OPTIONS_PAPER_MIN_ACC" \
  TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES="$SCHWAB_OPTIONS_PAPER_PROFILES" \
  PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}" \
  PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}" \
  PYTHONUNBUFFERED=1 nohup "${CMD[@]}" > "$LOG_ALL" 2>&1 & disown
else
  PYTHONUNBUFFERED=1 nohup "${CMD[@]}" > "$LOG_ALL" 2>&1 & disown
fi

echo "all_sleeves_log=$LOG_ALL"

if [[ "$WITH_COINBASE" == "1" ]]; then
  if coinbase_spot_running; then
    EXISTING_PID="$(coinbase_spot_process_lines | awk 'NR==1{print $1}')"
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
    if coinbase_spot_running; then
      echo "coinbase_log=$LOG_CB"
      echo "coinbase_mode simulate=$COINBASE_SIMULATE paper=$COINBASE_PAPER"
    else
      echo "coinbase_loop=failed_to_start log=$LOG_CB"
      tail -n 40 "$LOG_CB" || true
    fi
  fi
fi

OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
