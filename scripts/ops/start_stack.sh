#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/ops/runtime_python.sh"
PY="$(resolve_runtime_python)"
HEALTH_DIR="$PROJECT_ROOT/governance/health"
OPERATOR_STOP_FLAG="$HEALTH_DIR/OPERATOR_STOP.flag"
GLOBAL_HALT_FLAG="$HEALTH_DIR/GLOBAL_TRADING_HALT.flag"
RUNTIME_MAINTENANCE_HOLD_FLAG="$HEALTH_DIR/RUNTIME_MAINTENANCE_HOLD.flag"
PAPER_TRADE_LOCK_FILE="$HEALTH_DIR/PAPER_TRADE_LOCK.flag"
STACK_STOPPED_FLAG="$HEALTH_DIR/STACK_STOPPED.flag"

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
SHADOW_WATCHDOG_PAUSED_FOR_RESTART=0

load_stack_runtime_env() {
  if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
  fi
  PY="$(resolve_runtime_python)"
}

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

  if [[ -f "$RUNTIME_MAINTENANCE_HOLD_FLAG" ]]; then
    if "$PY" "$PROJECT_ROOT/scripts/ops/runtime_maintenance_hold.py" --json | "$PY" -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("active") else 1)'; then
      blocked=1
      echo "stack_start_blocked=runtime_maintenance_hold"
      echo "runtime_maintenance_hold_flag=$RUNTIME_MAINTENANCE_HOLD_FLAG"
    fi
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

wait_for_process_match() {
  local match="$1"
  local timeout_seconds="${2:-45}"
  local poll_seconds="${3:-1}"
  local started_at="$SECONDS"
  while (( SECONDS - started_at < timeout_seconds )); do
    if ps -axo command | grep -F "$match" | grep -v grep >/dev/null 2>&1; then
      return 0
    fi
    sleep "$poll_seconds"
  done
  return 1
}

wait_for_process_absent() {
  local match="$1"
  local timeout_seconds="${2:-45}"
  local poll_seconds="${3:-1}"
  local started_at="$SECONDS"
  while (( SECONDS - started_at < timeout_seconds )); do
    if ! ps -axo command | grep -F "$match" | grep -v grep >/dev/null 2>&1; then
      return 0
    fi
    sleep "$poll_seconds"
  done
  return 1
}

wait_for_process_stable() {
  local match="$1"
  local timeout_seconds="${2:-45}"
  local stable_seconds="${3:-5}"
  local started_at="$SECONDS"
  local stable_since=-1
  while (( SECONDS - started_at < timeout_seconds )); do
    if ps -axo command | grep -F "$match" | grep -v grep >/dev/null 2>&1; then
      if (( stable_since < 0 )); then
        stable_since="$SECONDS"
      fi
      if (( SECONDS - stable_since >= stable_seconds )); then
        return 0
      fi
    else
      stable_since=-1
    fi
    sleep 1
  done
  return 1
}

recover_launchd_label() {
  local label="$1"
  local required="${2:-0}"
  local domain="gui/$(id -u)"
  local plist="$HOME/Library/LaunchAgents/${label}.plist"

  launchctl enable "$domain/$label" >/dev/null 2>&1 || true
  if ! launchctl print "$domain/$label" >/dev/null 2>&1; then
    if [[ ! -f "$plist" ]]; then
      echo "launchd_recovery_missing_plist=$label path=$plist"
      [[ "$required" == "1" ]] && return 1
      return 0
    fi
    launchctl bootstrap "$domain" "$plist" >/dev/null 2>&1 || true
  fi
  if launchctl print "$domain/$label" >/dev/null 2>&1; then
    echo "launchd_recovery_ready=$label"
    return 0
  fi
  echo "launchd_recovery_failed=$label"
  [[ "$required" == "1" ]] && return 1
  return 0
}

restore_unattended_support_services() {
  local failed=0
  recover_launchd_label "com.dankingsley.ops.watchdog" 1 || failed=1
  recover_launchd_label "com.dankingsley.ops.sql_link_writer" 1 || failed=1
  recover_launchd_label "com.dankingsley.failover_hot_standby" 1 || failed=1
  recover_launchd_label "com.dankingsley.caffeinate_guard" 1 || failed=1
  recover_launchd_label "com.dankingsley.observability_exporter" 0 || true
  recover_launchd_label "com.dankingsley.livefeed-local" 0 || true
  recover_launchd_label "com.dankingsley.premarket_token_guard" 0 || true
  recover_launchd_label "com.dankingsley.ops.schwab_auth_supervisor" 0 || true
  recover_launchd_label "com.dankingsley.reboot_resilience_guard" 1 || failed=1
  return "$failed"
}

pause_shadow_watchdog_for_restart() {
  local label="com.dankingsley.shadow_watchdog"
  local plist="$HOME/Library/LaunchAgents/${label}.plist"
  local domain="gui/$(id -u)"

  if [[ -f "$plist" ]]; then
    launchctl bootout "$domain" "$plist" >/dev/null 2>&1 || true
  fi
  pkill -f "scripts/shadow_watchdog.py" >/dev/null 2>&1 || true
  if wait_for_process_absent "scripts/shadow_watchdog.py" "${SHADOW_WATCHDOG_STOP_TIMEOUT_SECONDS:-20}"; then
    echo "shadow_watchdog=paused_for_restart"
    return 0
  fi
  echo "shadow_watchdog=failed_to_pause_before_restart"
  return 1
}

resume_shadow_watchdog_after_restart() {
  if [[ "$SHADOW_WATCHDOG_PAUSED_FOR_RESTART" != "1" ]]; then
    return 0
  fi

  local label="com.dankingsley.shadow_watchdog"
  local plist="$HOME/Library/LaunchAgents/${label}.plist"
  local domain="gui/$(id -u)"
  if [[ -x "$PROJECT_ROOT/scripts/install_shadow_watchdog_launchd.sh" ]]; then
    "$PROJECT_ROOT/scripts/install_shadow_watchdog_launchd.sh" >/dev/null 2>&1 || true
  elif [[ -f "$plist" ]]; then
    launchctl bootstrap "$domain" "$plist" >/dev/null 2>&1 || true
    launchctl kickstart -k "$domain/$label" >/dev/null 2>&1 || true
  elif [[ -x "$PROJECT_ROOT/scripts/ops/run_shadow_watchdog_launchd.sh" ]]; then
    PYTHONUNBUFFERED=1 nohup "$PROJECT_ROOT/scripts/ops/run_shadow_watchdog_launchd.sh" \
      > "logs/shadow_watchdog_restart_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 & disown
  fi

  if wait_for_process_match "scripts/shadow_watchdog.py" "${SHADOW_WATCHDOG_START_TIMEOUT_SECONDS:-45}"; then
    SHADOW_WATCHDOG_PAUSED_FOR_RESTART=0
    echo "shadow_watchdog=resumed_after_restart"
    return 0
  fi
  echo "shadow_watchdog=failed_to_resume_after_restart"
  return 1
}

restart_exit_cleanup() {
  local rc=$?
  if [[ "$SHADOW_WATCHDOG_PAUSED_FOR_RESTART" == "1" ]]; then
    resume_shadow_watchdog_after_restart || true
  fi
  return "$rc"
}

trap restart_exit_cleanup EXIT

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

load_stack_runtime_env
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

if [[ -f "$STACK_STOPPED_FLAG" && "${BOT_OPS_DATA_PLANE_STARTUP_COMPACTION:-1}" != "0" ]]; then
  "$PY" "$PROJECT_ROOT/scripts/ops/ops_data_plane_compactor.py" --apply --json >/dev/null
fi

rm -f "$STACK_STOPPED_FLAG"

"$PY" "$PROJECT_ROOT/scripts/ops/apple_silicon_profile.py" apply >/dev/null 2>&1 || true
"$PY" "$PROJECT_ROOT/scripts/ops/portable_brain_contract.py" apply >/dev/null 2>&1 || true
"$PY" "$PROJECT_ROOT/scripts/resource_guard.py" --profile refresh --json >/dev/null 2>&1 || true
"$PY" "$PROJECT_ROOT/scripts/ops/memory_efficiency_control.py" apply >/dev/null 2>&1 || true
"$PY" "$PROJECT_ROOT/scripts/ops/computer_task_intelligence.py" --apply --json >/dev/null 2>&1 || true

load_stack_runtime_env

if [[ "${PAPER_400_RAMP_AUTO_APPLY:-1}" != "0" && -f "$PROJECT_ROOT/scripts/ops/paper_400_ramp_control.py" ]]; then
  "$PY" "$PROJECT_ROOT/scripts/ops/paper_400_ramp_control.py" --apply --json >/dev/null 2>&1 || true
  if [[ -f "$PROJECT_ROOT/config/.env.paper_400_ramp_override" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/config/.env.paper_400_ramp_override"
    set +a
  fi
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
  if ! pause_shadow_watchdog_for_restart; then
    exit 1
  fi
  SHADOW_WATCHDOG_PAUSED_FOR_RESTART=1
  # Clean sweep so stale wrappers/children do not keep locks and destabilize the supervisor.
  pkill -f "scripts/run_all_sleeves.py" || true
  pkill -f "scripts/run_parallel_shadows.py" || true
  pkill -f "scripts/run_parallel_aggressive_modes.py" || true
  pkill -f "scripts/run_dividend_shadow.py" || true
  pkill -f "scripts/run_dividend_capture_shadow.py" || true
  pkill -f "scripts/run_bond_shadow.py" || true
  pkill -f "scripts/run_fx_shadow.py" || true
  pkill -f "scripts/run_.*_shadow.py" || true
  pkill -f "scripts/run_shadow_training_loop.py --broker schwab" || true
  kill_coinbase_spot_loops
  pkill -f "scripts/run_shadow_training_loop.py --broker coinbase --profile crypto_futures" || true
  if ! wait_for_process_absent "scripts/run_all_sleeves.py" "${ALL_SLEEVES_STOP_TIMEOUT_SECONDS:-45}"; then
    echo "all_sleeves=failed_to_stop_before_restart"
    exit 1
  fi
  sleep 1
fi

"$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true
"$PY" "$PROJECT_ROOT/scripts/ops/storage_failback_sync.py" --json >/dev/null 2>&1 || true
load_stack_runtime_env
PREFLIGHT_ARGS=(--broker "${DATA_BROKER:-schwab}" --json)
if [[ "$SIMULATE" == "1" ]]; then
  PREFLIGHT_ARGS+=(--simulate)
fi
if ps -axo command | grep -F "scripts/run_all_sleeves.py" | grep -v grep >/dev/null 2>&1; then
  # An idempotent start audits the managed stack in place; its single healthy
  # child launchers are expected and must not be killed as pre-start debris.
  PREFLIGHT_ARGS+=(--allow-running)
elif [[ "${OPS_PREFLIGHT_APPLY_KILL_DUPLICATES:-1}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--apply-kill-duplicates)
fi
"$PY" "$PROJECT_ROOT/scripts/ops/preflight_autofix.py" "${PREFLIGHT_ARGS[@]}" || true

export MARKET_DATA_ONLY="${MARKET_DATA_ONLY:-1}"
export ALLOW_ORDER_EXECUTION="${ALLOW_ORDER_EXECUTION:-0}"
export TOP_BOT_PAPER_TRADING_ENABLED="${TOP_BOT_PAPER_TRADING_ENABLED:-1}"
export TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED="${TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED:-1}"
export PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS="${PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS:-0}"
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
    if resume_shadow_watchdog_after_restart; then
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

    if wait_for_process_match "$WD_MATCH" "${SHADOW_WATCHDOG_START_TIMEOUT_SECONDS:-45}"; then
      WD_PID="$(ps -axo pid,command | grep -F "$WD_MATCH" | grep -v grep | awk 'NR==1{print $1}')"
      echo "shadow_watchdog=started pid=$WD_PID"
    else
      echo "shadow_watchdog=failed_to_start"
      exit 1
    fi
  fi

  OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
  if [[ "$FORCE_RESTART" == "1" ]]; then
    if ! wait_for_process_stable \
      "scripts/run_all_sleeves.py" \
      "${ALL_SLEEVES_START_TIMEOUT_SECONDS:-60}" \
      "${ALL_SLEEVES_START_STABLE_SECONDS:-5}"; then
      echo "all_sleeves=missing_after_restart"
      exit 1
    fi
    if wait_for_process_match "scripts/run_execution_lane.py --mode paper" "${PAPER_EXECUTION_LANE_START_TIMEOUT_SECONDS:-45}"; then
      "$PY" "$PROJECT_ROOT/scripts/ops/creative_cotenant_guard.py" apply --paper-lane-only --json >/dev/null
      echo "paper_execution_lane=singleton_verified_after_restart"
    else
      echo "paper_execution_lane=missing_after_restart"
      exit 1
    fi
  fi
  if ! restore_unattended_support_services; then
    echo "stack_start_status=failed_to_restore_unattended_supervisors"
    exit 1
  fi
  "$PY" "$PROJECT_ROOT/scripts/ops/health_fast.py" --json >/dev/null 2>&1 || true
  echo "stack_start_delegated_to=shadow_watchdog"
  exit 0
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
fi

if [[ "$WITH_COINBASE" == "1" && "$COINBASE_PAPER" == "1" ]]; then
  COINBASE_PAPER_TOP_N="${COINBASE_TOP_BOT_PAPER_TRADING_TOP_N:-${TOP_BOT_PAPER_TRADING_TOP_N:-5}}"
  COINBASE_PAPER_MIN_ACC="${COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC:-${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.58}}"
  echo "coinbase_paper=enabled top_n=$COINBASE_PAPER_TOP_N min_acc=$COINBASE_PAPER_MIN_ACC"
fi

WATCHDOG_HANDOFF_LOG="logs/process_watchdog_handoff_$(date -u +%Y%m%d_%H%M%S).log"
run_process_watchdog_handoff() {
  OPS_WATCHDOG_REFRESH_REPORTS=0 \
  OPS_WATCHDOG_REQUIRE_ALL_SLEEVES=1 \
  OPS_WATCHDOG_REQUIRE_COINBASE="$WITH_COINBASE" \
  OPS_WATCHDOG_REQUIRE_COINBASE_FUTURES="$WITH_COINBASE" \
  OPS_WATCHDOG_ALL_SLEEVES_SIMULATE="$SIMULATE" \
  OPS_WATCHDOG_ALL_SLEEVES_DISABLE_BREAKERS="$DISABLE_BREAKERS" \
  OPS_WATCHDOG_COINBASE_SIMULATE="$COINBASE_SIMULATE" \
  OPS_WATCHDOG_COINBASE_FUTURES_SIMULATE="$COINBASE_SIMULATE" \
  "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >> "$WATCHDOG_HANDOFF_LOG" 2>&1
}

if ! run_process_watchdog_handoff; then
  echo "stack_start_handoff=watchdog_command_failed log=$WATCHDOG_HANDOFF_LOG"
  tail -n 80 "$WATCHDOG_HANDOFF_LOG" || true
  exit 1
fi

if ! wait_for_process_stable \
  "scripts/run_all_sleeves.py" \
  "${ALL_SLEEVES_START_TIMEOUT_SECONDS:-60}" \
  "${ALL_SLEEVES_START_STABLE_SECONDS:-5}"; then
  # A concurrently running singleton may have completed a differently-timed pass.
  run_process_watchdog_handoff || true
fi

if wait_for_process_stable \
  "scripts/run_all_sleeves.py" \
  "${ALL_SLEEVES_START_TIMEOUT_SECONDS:-60}" \
  "${ALL_SLEEVES_START_STABLE_SECONDS:-5}"; then
  ALL_SLEEVES_RUNNING_PID="$(ps -axo pid,command | grep -F "scripts/run_all_sleeves.py" | grep -v grep | awk 'NR==1{print $1}')"
  echo "all_sleeves=started pid=$ALL_SLEEVES_RUNNING_PID"
  echo "all_sleeves_log=logs/watchdog_all_sleeves.log"
else
  echo "all_sleeves=failed_to_start owner=process_watchdog log=$WATCHDOG_HANDOFF_LOG"
  tail -n 80 "$WATCHDOG_HANDOFF_LOG" || true
  tail -n 80 "logs/watchdog_all_sleeves.log" || true
  exit 1
fi

if [[ "$FORCE_RESTART" == "1" ]]; then
  if wait_for_process_match "scripts/run_execution_lane.py --mode paper" "${PAPER_EXECUTION_LANE_START_TIMEOUT_SECONDS:-45}"; then
    "$PY" "$PROJECT_ROOT/scripts/ops/creative_cotenant_guard.py" apply --paper-lane-only --json >/dev/null
    echo "paper_execution_lane=singleton_verified_after_restart"
  else
    echo "paper_execution_lane=missing_after_restart"
    exit 1
  fi
fi

if [[ "$WITH_COINBASE" == "1" ]]; then
  if ! wait_for_process_stable \
    "scripts/run_shadow_training_loop.py --broker coinbase --symbols" \
    "${COINBASE_START_TIMEOUT_SECONDS:-60}" \
    "${COINBASE_START_STABLE_SECONDS:-5}"; then
    echo "coinbase_loop=failed_to_start owner=process_watchdog log=$WATCHDOG_HANDOFF_LOG"
    tail -n 80 "logs/watchdog_coinbase_loop.log" || true
    exit 1
  fi
  COINBASE_RUNNING_PID="$(coinbase_spot_process_lines | awk 'NR==1{print $1}')"
  echo "coinbase_loop=started pid=$COINBASE_RUNNING_PID"
  echo "coinbase_log=logs/watchdog_coinbase_loop.log"
  echo "coinbase_mode simulate=$COINBASE_SIMULATE paper=$COINBASE_PAPER"

  if ! wait_for_process_stable \
    "scripts/run_shadow_training_loop.py --broker coinbase --profile crypto_futures" \
    "${COINBASE_START_TIMEOUT_SECONDS:-60}" \
    "${COINBASE_START_STABLE_SECONDS:-5}"; then
    echo "coinbase_futures_loop=failed_to_start owner=process_watchdog log=$WATCHDOG_HANDOFF_LOG"
    tail -n 80 "logs/watchdog_coinbase_futures_loop.log" || true
    exit 1
  fi
  COINBASE_FUTURES_RUNNING_PID="$(ps -axo pid,command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase --profile crypto_futures" | grep -v grep | awk 'NR==1{print $1}')"
  echo "coinbase_futures_loop=started pid=$COINBASE_FUTURES_RUNNING_PID"
  echo "coinbase_futures_log=logs/watchdog_coinbase_futures_loop.log"
fi

if ! resume_shadow_watchdog_after_restart; then
  echo "stack_start_status=failed_to_restore_shadow_watchdog"
  exit 1
fi

if ! restore_unattended_support_services; then
  echo "stack_start_status=failed_to_restore_unattended_supervisors"
  exit 1
fi
"$PY" "$PROJECT_ROOT/scripts/ops/health_fast.py" --json >/dev/null 2>&1 || true

echo "stack_start_owner=process_watchdog"
echo "process_watchdog_handoff_log=$WATCHDOG_HANDOFF_LOG"
