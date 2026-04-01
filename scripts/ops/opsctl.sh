#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/ops/runtime_python.sh"
PY="$(resolve_runtime_python)"

cmd="${1:-help}"
shift || true

PROFILE="${BOT_RUNTIME_PROFILE:-live}"
case "$cmd" in
  status|start|start-sim|start-live|sql-sync|tradingeconomics-sync|macro-context-sync|market-micro-sync|sec-edgar-sync|extended-quant-sync|tastytrade-sync|crypto-market-sync|market-correlation-sync|fx-market-sync|dividend-drip-sync|showcase-refresh|macro-crosscheck|source-verification|coinbase-start|coinbase-futures-start|schwab-futures-start|fx-start|feed-refresh|storage-switch-local|storage-switch-external|storage-safe-eject|retrain-force-full|retrain-force-targeted|token-refresh|token-refresh-interactive|macro-bulletin|macro-auto-start|macro-replay|macro-media-ingest|macro-auto-stop|macro-auto-status|futures-tail|schwab-futures-tail|coinbase-futures-tail)
    if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
      # shellcheck disable=SC1091
      source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
    fi
    ;;
esac

load_runtime_profile() {
  local profile_name="${1:-live}"
  if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$profile_name" --quiet
  fi
  export BOT_RUNTIME_PROFILE="$profile_name"
  export MARKET_DATA_ONLY="${MARKET_DATA_ONLY:-1}"
  export ALLOW_ORDER_EXECUTION="${ALLOW_ORDER_EXECUTION:-0}"
  export TOP_BOT_PAPER_TRADING_ENABLED="${TOP_BOT_PAPER_TRADING_ENABLED:-1}"
  export TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED="${TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED:-1}"
  export PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"
  export PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"
  export LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"
  export LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"
  export LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"
  export LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"
  export LOG_FUTURES_MASTER_DECISIONS="${LOG_FUTURES_MASTER_DECISIONS:-1}"
}

STORAGE_OVERRIDE_FILE="$PROJECT_ROOT/config/.env.storage_override"

write_storage_override() {
  local mode="${1:-external}"
  mkdir -p "$(dirname "$STORAGE_OVERRIDE_FILE")"
  case "$mode" in
    local)
      cat > "$STORAGE_OVERRIDE_FILE" <<'EOF'
# Auto-managed by scripts/ops/opsctl.sh
BOT_LOGS_PREFER_EXTERNAL=0
EOF
      ;;
    external)
      rm -f "$STORAGE_OVERRIDE_FILE"
      ;;
    *)
      echo "unknown storage override mode: $mode" >&2
      return 2
      ;;
  esac
}

apply_storage_route_mode() {
  local mode="${1:-external}"
  local prefer_external="1"
  if [[ "$mode" == "local" ]]; then
    prefer_external="0"
  fi
  BOT_LOGS_PREFER_EXTERNAL="$prefer_external" \
    "$PY" "$PROJECT_ROOT/scripts/ops/storage_failback_sync.py" --json
}

restart_collection_after_storage_switch() {
  "$PROJECT_ROOT/scripts/ops/opsctl.sh" stop >/dev/null 2>&1 || true
  sleep 1
  "$PROJECT_ROOT/scripts/ops/opsctl.sh" feed-refresh --source all
  OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
}

kill_schwab_live_loops() {
  pkill -f "scripts/run_all_sleeves.py" || true
  pkill -f "scripts/run_parallel_shadows.py" || true
  pkill -f "scripts/run_parallel_aggressive_modes.py" || true
  pkill -f "scripts/run_dividend_shadow.py" || true
  pkill -f "scripts/run_bond_shadow.py" || true
  pkill -f "scripts/run_shadow_training_loop.py --broker schwab" || true
}

start_schwab_live_loops() {
  local paper_mode="${1:-0}"
  load_runtime_profile live
  "$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true
  "$PY" "$PROJECT_ROOT/scripts/ops/storage_failback_sync.py" --json >/dev/null 2>&1 || true

  local log_file="$PROJECT_ROOT/logs/all_sleeves_$(date -u +%Y%m%d_%H%M%S).log"
  local -a cmd=(
    "$PY" "$PROJECT_ROOT/scripts/run_all_sleeves.py"
    --with-aggressive-modes
  )

  if [[ "$paper_mode" == "1" ]]; then
    local paper_top_n="${SCHWAB_TOP_BOT_PAPER_TRADING_TOP_N:-${TOP_BOT_PAPER_TRADING_TOP_N:-5}}"
    local paper_min_acc="${SCHWAB_TOP_BOT_PAPER_TRADING_MIN_ACC:-${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.58}}"
    local paper_profiles="${SCHWAB_TOP_BOT_PAPER_TRADING_PROFILES:-${TOP_BOT_PAPER_TRADING_PROFILES:-}}"
    local options_paper_top_n="${SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_TOP_N:-${TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N:-2}}"
    local options_paper_min_acc="${SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_MIN_ACC:-${TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC:-$paper_min_acc}}"
    local options_paper_profiles="${SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_PROFILES:-${TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES:-${paper_profiles:-}}}"
    echo "schwab_paper=enabled top_n=$paper_top_n min_acc=$paper_min_acc profiles=${paper_profiles:-all}"
    echo "schwab_options_paper=enabled top_n=$options_paper_top_n min_acc=$options_paper_min_acc profiles=${options_paper_profiles:-all}"
    TOP_BOT_PAPER_TRADING_ENABLED=1 \
    TOP_BOT_PAPER_TRADING_TOP_N="$paper_top_n" \
    TOP_BOT_PAPER_TRADING_MIN_ACC="$paper_min_acc" \
    TOP_BOT_PAPER_TRADING_PROFILES="$paper_profiles" \
    TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED="${TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED:-1}" \
    TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N="$options_paper_top_n" \
    TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC="$options_paper_min_acc" \
    TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES="$options_paper_profiles" \
    PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}" \
    PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}" \
    PYTHONUNBUFFERED=1 nohup "${cmd[@]}" > "$log_file" 2>&1 & disown
  else
    PYTHONUNBUFFERED=1 nohup "${cmd[@]}" > "$log_file" 2>&1 & disown
  fi
  sleep 2

  if ps -axo command | grep -F "scripts/run_all_sleeves.py --with-aggressive-modes" | grep -v grep >/dev/null 2>&1; then
    echo "$log_file"
    echo "schwab_live_loops_started simulate=0 paper_mode=$paper_mode"
    OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
    return 0
  fi

  echo "schwab_live_loops_failed_to_start log=$log_file" >&2
  tail -n 60 "$log_file" || true
  return 1
}

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

fx_process_lines() {
  ps -axo pid,command | grep -E "scripts/run_fx_shadow.py|scripts/run_shadow_training_loop.py --broker .* --profile fx" | grep -v grep || true
}

fx_running() {
  fx_process_lines | grep -q .
}

kill_fx_loops() {
  local pids
  pids="$(fx_process_lines | awk '{print $1}')"
  if [[ -n "${pids//[[:space:]]/}" ]]; then
    while IFS= read -r pid; do
      [[ -n "$pid" ]] && kill "$pid" >/dev/null 2>&1 || true
    done <<< "$pids"
  fi
}

case "$cmd" in
  start)
    exec "$PROJECT_ROOT/scripts/ops/start_stack.sh" "$@"
    ;;
  start-sim)
    exec "$PROJECT_ROOT/scripts/ops/start_stack.sh" --profile sim --simulate "$@"
    ;;
  start-live)
    exec "$PROJECT_ROOT/scripts/ops/start_stack.sh" --profile live "$@"
    ;;
  stop)
    pkill -f "scripts/run_all_sleeves.py --with-aggressive-modes" || true
    pkill -f "scripts/run_parallel_shadows.py" || true
    pkill -f "scripts/run_parallel_aggressive_modes.py" || true
    kill_coinbase_spot_loops
    kill_fx_loops
    pkill -f "scripts/run_shadow_training_loop.py --broker coinbase --profile crypto_futures" || true
    pkill -f "scripts/run_shadow_training_loop.py --broker schwab --profile schwab_futures" || true
    pkill -f "scripts/ops/sql_link_shard_manager.py" || true
    pkill -f "scripts/ops/sql_link_writer_service.py" || true
    pkill -f "scripts/link_jsonl_to_sql.py --project-root $PROJECT_ROOT --mode sqlite --sqlite-db $PROJECT_ROOT/data/sql_link_shards/" || true
    echo "stopped core loops"
    ;;
  status)
    ps -axo pid,etime,command | grep -E "run_all_sleeves.py|run_parallel_shadows.py|run_parallel_aggressive_modes.py|run_fx_shadow.py|run_shadow_training_loop.py --broker coinbase|run_shadow_training_loop.py --broker .* --profile fx|run_shadow_training_loop.py --broker schwab --profile schwab_futures|sql_link_shard_manager.py|sql_link_writer_service.py|link_jsonl_to_sql.py --project-root .*sql_link_shards" | grep -v grep || true
    PROFILE="${BOT_RUNTIME_PROFILE:-live}"
    if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
      # shellcheck disable=SC1091
      source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
    fi
    PREFLIGHT_ARGS=(--broker "${DATA_BROKER:-schwab}" --allow-running --json)
    if [[ "$PROFILE" == "sim" ]]; then
      PREFLIGHT_ARGS+=(--simulate)
    fi
    "$PY" "$PROJECT_ROOT/scripts/ops/preflight_autofix.py" "${PREFLIGHT_ARGS[@]}" || true
    ;;
  retrain)
    # MLX Metal JIT can intermittently crash in some launch contexts; keep a stable default.
    MLX_METAL_JIT="${MLX_METAL_JIT:-0}" exec "$PY" "$PROJECT_ROOT/scripts/weekly_retrain.py" --continue-on-error "$@"
    ;;
  retrain-orchestrate)
    exec "$PY" "$PROJECT_ROOT/scripts/retrain_orchestrator.py" "$@"
    ;;
  scorecard)
    exec "$PY" "$PROJECT_ROOT/scripts/unified_lane_scorecard.py" "$@"
    ;;
  sql-sync)
    if [[ -n "${SQL_LINK_SERVICE_SHARDS:-}" ]]; then
      exec "$PY" "$PROJECT_ROOT/scripts/ops/sql_link_shard_manager.py" --once "$@"
    fi
    exec "$PY" "$PROJECT_ROOT/scripts/ops/sql_link_writer_service.py" --once "$@"
    ;;
  tradingeconomics-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_tradingeconomics_guest_data.py" "$@"
    ;;
  macro-context-sync)
    "$PY" "$PROJECT_ROOT/scripts/collect_bls_census_data.py" "$@" || true
    exec "$PY" "$PROJECT_ROOT/scripts/collect_official_macro_context.py" "$@"
    ;;
  market-micro-sync)
    if [[ $# -eq 0 ]]; then
      exec "$PY" "$PROJECT_ROOT/scripts/collect_market_micro_context.py" \
        --lookback-days "${MARKET_MICRO_LOOKBACK_DAYS:-21}" \
        --finra-lookback-days "${MARKET_MICRO_FINRA_LOOKBACK_DAYS:-15}" \
        --symbols "${MARKET_MICRO_SYMBOLS:-}" 
    fi
    exec "$PY" "$PROJECT_ROOT/scripts/collect_market_micro_context.py" "$@"
    ;;
  sec-edgar-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_sec_edgar_context.py" "$@"
    ;;
  extended-quant-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_extended_quant_context.py" "$@"
    ;;
  tastytrade-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_tastytrade_context.py" "$@"
    ;;
  crypto-market-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_crypto_market_context.py" "$@"
    ;;
  market-correlation-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_market_crypto_correlation_context.py" "$@"
    ;;
  fx-market-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_fx_market_context.py" "$@"
    ;;
  dividend-drip-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_dividend_drip_state.py" "$@"
    ;;
  showcase-refresh)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/update_showcase_highlights.py" "$@"
    ;;
  macro-crosscheck)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/macro_crosscheck_report.py" "$@"
    ;;
  source-verification)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/source_verification_report.py" "$@"
    ;;
  sql-maint|sql-maintenance|sqlite-maint|sqlite-maintenance)
    sql_maint_args=("$@")
    wants_explicit_vacuum=0
    for arg in "${sql_maint_args[@]}"; do
      case "$arg" in
        --vacuum|--no-auto-vacuum|--auto-vacuum-over-gb|--checkpoint-only)
          wants_explicit_vacuum=1
          ;;
      esac
    done
    if [[ "$wants_explicit_vacuum" == "0" ]]; then
      sql_maint_args+=(--no-auto-vacuum)
    fi
    exec "$PY" "$PROJECT_ROOT/scripts/sqlite_performance_maintenance.py" "${sql_maint_args[@]}"
    ;;
  health)
    exec "$PY" "$PROJECT_ROOT/scripts/daily_auto_verify.py" --json "$@"
    ;;
  dashboard)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/runtime_gate_dashboard.py" --json "$@"
    ;;
  phone-feed)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/live_feed_phone_server.py" "$@"
    ;;
  py314-canary|py314-ready|python314-canary|python314-ready)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/python314_canary.py" --json "$@"
    ;;
  doctor)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/doctor.py" "$@"
    ;;
  schwab-futures-start)
    FORCE_RESTART=0
    SCHWAB_SIMULATE="${SCHWAB_FUTURES_SIMULATE:-0}"
    PAPER_MODE=1
    FUTURES_PROFILE="${SCHWAB_FUTURES_PROFILE:-schwab_futures}"
    PAPER_TOP_N="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}"
    PAPER_MIN_ACC="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.53}"
    PAPER_PROFILES="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-$FUTURES_PROFILE}"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --force-restart) FORCE_RESTART=1 ;;
        --paper) PAPER_MODE=1; SCHWAB_SIMULATE=0 ;;
        --simulate) SCHWAB_SIMULATE=1 ;;
        --live-data|--no-simulate) SCHWAB_SIMULATE=0 ;;
        --top-n) PAPER_TOP_N="${2:-$PAPER_TOP_N}"; shift ;;
        --min-acc) PAPER_MIN_ACC="${2:-$PAPER_MIN_ACC}"; shift ;;
        --profiles) PAPER_PROFILES="${2:-$PAPER_PROFILES}"; shift ;;
        *) echo "unknown schwab-futures-start arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    if [[ "$FORCE_RESTART" == "1" ]]; then
      pkill -f "scripts/run_shadow_training_loop.py --broker schwab --profile $FUTURES_PROFILE" || true
      sleep 1
    fi

    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker schwab --profile $FUTURES_PROFILE" | grep -v grep >/dev/null 2>&1; then
      PID="$(ps -axo pid,command | grep -F "scripts/run_shadow_training_loop.py --broker schwab --profile $FUTURES_PROFILE" | grep -v grep | awk 'NR==1{print $1}')"
      LATEST_LOG="$(ls -1t "$PROJECT_ROOT"/logs/schwab_futures_live_*.log 2>/dev/null | head -n 1)"
      echo "schwab_futures_loop already running pid=$PID profile=$FUTURES_PROFILE"
      [[ -n "$LATEST_LOG" ]] && echo "$LATEST_LOG"
      exit 0
    fi

    "$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true

    LOG="$PROJECT_ROOT/logs/schwab_futures_live_$(date -u +%Y%m%d_%H%M%S).log"
    SCHWAB_CMD=(
      "$PY" "$PROJECT_ROOT/scripts/run_shadow_training_loop.py"
      --broker schwab
      --profile "$FUTURES_PROFILE"
      --domain equities
      --symbols "${SCHWAB_FUTURES_WATCH_SYMBOLS:-/ES,/NQ,/YM,/RTY,/CL,/GC,/ZN}"
      --context-symbols "${SCHWAB_FUTURES_CONTEXT_SYMBOLS:-SPY,UUP,GLD}"
      --interval-seconds "${SCHWAB_FUTURES_WATCH_INTERVAL_SECONDS:-12}"
      --max-iterations 0
    )
    if [[ "$SCHWAB_SIMULATE" == "1" ]]; then
      SCHWAB_CMD+=(--simulate)
    fi

    if [[ "$PAPER_MODE" == "1" ]]; then
      echo "schwab_futures_paper=enabled profile=$FUTURES_PROFILE top_n=$PAPER_TOP_N min_acc=$PAPER_MIN_ACC profiles=$PAPER_PROFILES"
      SHADOW_PROFILE="$FUTURES_PROFILE"       SHADOW_DOMAIN=equities       LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"       LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"       LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"       LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"       TOP_BOT_PAPER_TRADING_ENABLED=1       TOP_BOT_PAPER_TRADING_TOP_N="$PAPER_TOP_N"       TOP_BOT_PAPER_TRADING_MIN_ACC="$PAPER_MIN_ACC"       TOP_BOT_PAPER_TRADING_PROFILES="$PAPER_PROFILES"       PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"       PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"       nohup "${SCHWAB_CMD[@]}" > "$LOG" 2>&1 & disown
    else
      SHADOW_PROFILE="$FUTURES_PROFILE"       SHADOW_DOMAIN=equities       LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"       LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"       LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"       LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"       nohup "${SCHWAB_CMD[@]}" > "$LOG" 2>&1 & disown
    fi

    sleep 2
    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker schwab --profile $FUTURES_PROFILE" | grep -v grep >/dev/null 2>&1; then
      echo "$LOG"
      echo "schwab_futures_loop_started profile=$FUTURES_PROFILE simulate=$SCHWAB_SIMULATE paper_mode=$PAPER_MODE"
      OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
    else
      echo "schwab_futures_loop failed_to_start profile=$FUTURES_PROFILE"
      tail -n 60 "$LOG" || true
      exit 1
    fi
    ;;
  schwab-futures-stop)
    FUTURES_PROFILE="${SCHWAB_FUTURES_PROFILE:-schwab_futures}"
    pkill -f "scripts/run_shadow_training_loop.py --broker schwab --profile $FUTURES_PROFILE" || true
    echo "schwab futures loop stopped profile=$FUTURES_PROFILE"
    ;;
  coinbase-start)
    FORCE_RESTART=0
    COINBASE_SIMULATE="${COINBASE_START_SIMULATE:-0}"
    PAPER_MODE=1
    PAPER_TOP_N="${COINBASE_TOP_BOT_PAPER_TRADING_TOP_N:-${TOP_BOT_PAPER_TRADING_TOP_N:-5}}"
    PAPER_MIN_ACC="${COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC:-${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.58}}"
    PAPER_PROFILES="${COINBASE_TOP_BOT_PAPER_TRADING_PROFILES:-${TOP_BOT_PAPER_TRADING_PROFILES:-default}}"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --force-restart) FORCE_RESTART=1 ;;
        --paper) PAPER_MODE=1; COINBASE_SIMULATE=0 ;;
        --simulate) COINBASE_SIMULATE=1 ;;
        --live-data|--no-simulate) COINBASE_SIMULATE=0 ;;
        --top-n) PAPER_TOP_N="${2:-$PAPER_TOP_N}"; shift ;;
        --min-acc) PAPER_MIN_ACC="${2:-$PAPER_MIN_ACC}"; shift ;;
        --profiles) PAPER_PROFILES="${2:-$PAPER_PROFILES}"; shift ;;
        *) echo "unknown coinbase-start arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    if [[ "$FORCE_RESTART" == "1" ]]; then
      kill_coinbase_spot_loops
      sleep 1
    fi

    if coinbase_spot_running; then
      PID="$(coinbase_spot_process_lines | awk 'NR==1{print $1}')"
      LATEST_LOG="$(ls -1t "$PROJECT_ROOT"/logs/coinbase_live_*.log 2>/dev/null | head -n 1)"
      echo "coinbase_loop already running pid=$PID"
      [[ -n "$LATEST_LOG" ]] && echo "$LATEST_LOG"
      exit 0
    fi

    "$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true

    LOG="$PROJECT_ROOT/logs/coinbase_live_$(date -u +%Y%m%d_%H%M%S).log"
    COINBASE_CMD=(
      "$PY" "$PROJECT_ROOT/scripts/run_shadow_training_loop.py"
      --broker coinbase
      --symbols "${COINBASE_WATCH_SYMBOLS:-BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD}"
      --context-symbols "${COINBASE_CONTEXT_SYMBOLS:-BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD}"
      --interval-seconds "${COINBASE_WATCH_INTERVAL_SECONDS:-20}"
      --max-iterations 0
    )
    if [[ "$COINBASE_SIMULATE" == "1" ]]; then
      COINBASE_CMD+=(--simulate)
    fi

    if [[ "$PAPER_MODE" == "1" ]]; then
      echo "coinbase_paper=enabled top_n=$PAPER_TOP_N min_acc=$PAPER_MIN_ACC profiles=$PAPER_PROFILES"
      TOP_BOT_PAPER_TRADING_ENABLED=1       TOP_BOT_PAPER_TRADING_TOP_N="$PAPER_TOP_N"       TOP_BOT_PAPER_TRADING_MIN_ACC="$PAPER_MIN_ACC"       TOP_BOT_PAPER_TRADING_PROFILES="$PAPER_PROFILES"       PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"       PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"       ADAPTIVE_INTERVAL_ENABLED="${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-1}"       nohup "${COINBASE_CMD[@]}" > "$LOG" 2>&1 & disown
    else
      ADAPTIVE_INTERVAL_ENABLED="${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-1}"       nohup "${COINBASE_CMD[@]}" > "$LOG" 2>&1 & disown
    fi

    sleep 2
    if coinbase_spot_running; then
      echo "$LOG"
      echo "coinbase_loop_started simulate=$COINBASE_SIMULATE paper_mode=$PAPER_MODE"
      OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --require-coinbase --json >/dev/null 2>&1 || true
    else
      echo "coinbase_loop failed_to_start"
      tail -n 60 "$LOG" || true
      exit 1
    fi
    ;;
  coinbase-futures-start)
    FORCE_RESTART=0
    COINBASE_SIMULATE="${COINBASE_FUTURES_SIMULATE:-0}"
    PAPER_MODE=1
    FUTURES_PROFILE="${COINBASE_FUTURES_PROFILE:-crypto_futures}"
    PAPER_TOP_N="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}"
    PAPER_MIN_ACC="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.56}"
    PAPER_PROFILES="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-$FUTURES_PROFILE}"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --force-restart) FORCE_RESTART=1 ;;
        --paper) PAPER_MODE=1; COINBASE_SIMULATE=0 ;;
        --simulate) COINBASE_SIMULATE=1 ;;
        --live-data|--no-simulate) COINBASE_SIMULATE=0 ;;
        --top-n) PAPER_TOP_N="${2:-$PAPER_TOP_N}"; shift ;;
        --min-acc) PAPER_MIN_ACC="${2:-$PAPER_MIN_ACC}"; shift ;;
        --profiles) PAPER_PROFILES="${2:-$PAPER_PROFILES}"; shift ;;
        *) echo "unknown coinbase-futures-start arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    if [[ "$FORCE_RESTART" == "1" ]]; then
      pkill -f "scripts/run_shadow_training_loop.py --broker coinbase --profile $FUTURES_PROFILE" || true
      sleep 1
    fi

    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase --profile $FUTURES_PROFILE" | grep -v grep >/dev/null 2>&1; then
      PID="$(ps -axo pid,command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase --profile $FUTURES_PROFILE" | grep -v grep | awk 'NR==1{print $1}')"
      LATEST_LOG="$(ls -1t "$PROJECT_ROOT"/logs/coinbase_futures_live_*.log 2>/dev/null | head -n 1)"
      echo "coinbase_futures_loop already running pid=$PID profile=$FUTURES_PROFILE"
      [[ -n "$LATEST_LOG" ]] && echo "$LATEST_LOG"
      exit 0
    fi

    "$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true

    LOG="$PROJECT_ROOT/logs/coinbase_futures_live_$(date -u +%Y%m%d_%H%M%S).log"
    COINBASE_CMD=(
      "$PY" "$PROJECT_ROOT/scripts/run_shadow_training_loop.py"
      --broker coinbase
      --profile "$FUTURES_PROFILE"
      --domain crypto
      --symbols "${COINBASE_FUTURES_WATCH_SYMBOLS:-BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LINK-USD,DOGE-USD}"
      --context-symbols "${COINBASE_FUTURES_CONTEXT_SYMBOLS:-BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD}"
      --interval-seconds "${COINBASE_FUTURES_WATCH_INTERVAL_SECONDS:-20}"
      --max-iterations 0
    )
    if [[ "$COINBASE_SIMULATE" == "1" ]]; then
      COINBASE_CMD+=(--simulate)
    fi

    if [[ "$PAPER_MODE" == "1" ]]; then
      echo "coinbase_futures_paper=enabled profile=$FUTURES_PROFILE top_n=$PAPER_TOP_N min_acc=$PAPER_MIN_ACC profiles=$PAPER_PROFILES"
      SHADOW_PROFILE="$FUTURES_PROFILE"       SHADOW_DOMAIN=crypto       SHADOW_THRESHOLD_SHIFT="${COINBASE_FUTURES_THRESHOLD_SHIFT:-0.02}"       SIZING_MAX_NOTIONAL_PCT="${COINBASE_FUTURES_MAX_NOTIONAL_PCT:-0.03}"       PORTFOLIO_BASE_BUDGET="${COINBASE_FUTURES_BASE_BUDGET:-0.50}"       CROSS_SYMBOL_MAX_LONG="${COINBASE_FUTURES_MAX_LONG:-4}"       CROSS_SYMBOL_MAX_SHORT="${COINBASE_FUTURES_MAX_SHORT:-4}"       RISK_MAX_DAILY_LOSS_PROXY="${COINBASE_FUTURES_MAX_DAILY_LOSS_PROXY:-0.03}"       LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"       LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"       LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"       LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"       TOP_BOT_PAPER_TRADING_ENABLED=1       TOP_BOT_PAPER_TRADING_TOP_N="$PAPER_TOP_N"       TOP_BOT_PAPER_TRADING_MIN_ACC="$PAPER_MIN_ACC"       TOP_BOT_PAPER_TRADING_PROFILES="$PAPER_PROFILES"       PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"       PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"       ADAPTIVE_INTERVAL_ENABLED="${COINBASE_FUTURES_ADAPTIVE_INTERVAL_ENABLED:-${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-1}}"       nohup "${COINBASE_CMD[@]}" > "$LOG" 2>&1 & disown
    else
      SHADOW_PROFILE="$FUTURES_PROFILE"       SHADOW_DOMAIN=crypto       SHADOW_THRESHOLD_SHIFT="${COINBASE_FUTURES_THRESHOLD_SHIFT:-0.02}"       SIZING_MAX_NOTIONAL_PCT="${COINBASE_FUTURES_MAX_NOTIONAL_PCT:-0.03}"       PORTFOLIO_BASE_BUDGET="${COINBASE_FUTURES_BASE_BUDGET:-0.50}"       CROSS_SYMBOL_MAX_LONG="${COINBASE_FUTURES_MAX_LONG:-4}"       CROSS_SYMBOL_MAX_SHORT="${COINBASE_FUTURES_MAX_SHORT:-4}"       RISK_MAX_DAILY_LOSS_PROXY="${COINBASE_FUTURES_MAX_DAILY_LOSS_PROXY:-0.03}"       LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"       LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"       LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"       LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"       ADAPTIVE_INTERVAL_ENABLED="${COINBASE_FUTURES_ADAPTIVE_INTERVAL_ENABLED:-${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-1}}"       nohup "${COINBASE_CMD[@]}" > "$LOG" 2>&1 & disown
    fi

    sleep 2
    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase --profile $FUTURES_PROFILE" | grep -v grep >/dev/null 2>&1; then
      echo "$LOG"
      echo "coinbase_futures_loop_started profile=$FUTURES_PROFILE simulate=$COINBASE_SIMULATE paper_mode=$PAPER_MODE"
      OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --require-coinbase --json >/dev/null 2>&1 || true
    else
      echo "coinbase_futures_loop failed_to_start profile=$FUTURES_PROFILE"
      tail -n 60 "$LOG" || true
      exit 1
    fi
    ;;
  coinbase-futures-stop)
    FUTURES_PROFILE="${COINBASE_FUTURES_PROFILE:-crypto_futures}"
    pkill -f "scripts/run_shadow_training_loop.py --broker coinbase --profile $FUTURES_PROFILE" || true
    echo "coinbase futures loop stopped profile=$FUTURES_PROFILE"
    ;;
  fx-start)
    FORCE_RESTART=0
    FX_SIMULATE="${FX_START_SIMULATE:-0}"
    PAPER_MODE=1
    DIRECT_EXECUTION=0
    FX_SYMBOL_SET="${FX_SYMBOLS:-UUP,FXE,FXY,FXB,FXC,FXA,CYB,EUO,YCS,UDN}"
    FX_CONTEXT_SET="${FX_CONTEXT_SYMBOLS:-SPY,QQQ,TLT,GLD,UUP,FXE,FXY,FXB,FXC,FXA}"
    FX_INTERVAL="${FX_SHADOW_INTERVAL:-45}"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --force-restart) FORCE_RESTART=1 ;;
        --paper) PAPER_MODE=1; FX_SIMULATE=0 ;;
        --simulate) FX_SIMULATE=1 ;;
        --live-data|--no-simulate) FX_SIMULATE=0 ;;
        --symbols) FX_SYMBOL_SET="${2:-$FX_SYMBOL_SET}"; shift ;;
        --context-symbols) FX_CONTEXT_SET="${2:-$FX_CONTEXT_SET}"; shift ;;
        --interval-seconds) FX_INTERVAL="${2:-$FX_INTERVAL}"; shift ;;
        --direct-execution) DIRECT_EXECUTION=1 ;;
        *) echo "unknown fx-start arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    if [[ "$DIRECT_EXECUTION" == "1" ]]; then
      if [[ "${SCHWAB_FOREX_API_VERIFIED:-0}" != "1" || "${FX_DIRECT_EXECUTION_ENABLED:-0}" != "1" ]]; then
        echo "fx-start direct execution blocked: Schwab forex API support is not officially verified for this stack" >&2
        exit 2
      fi
      echo "fx-start direct execution blocked: no direct Schwab forex execution implementation is enabled in this repo yet" >&2
      exit 2
    fi

    if [[ "$PAPER_MODE" != "1" ]]; then
      echo "fx-start only supports paper-only proxy mode" >&2
      exit 2
    fi

    if [[ "$FORCE_RESTART" == "1" ]]; then
      kill_fx_loops
      sleep 1
    fi

    if fx_running; then
      PID="$(fx_process_lines | awk 'NR==1{print $1}')"
      LATEST_LOG="$(ls -1t "$PROJECT_ROOT"/logs/fx_live_*.log 2>/dev/null | head -n 1)"
      echo "fx_loop already running pid=$PID"
      [[ -n "$LATEST_LOG" ]] && echo "$LATEST_LOG"
      exit 0
    fi

    "$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true

    LOG="$PROJECT_ROOT/logs/fx_live_$(date -u +%Y%m%d_%H%M%S).log"
    FX_CMD=(
      "$PY" "$PROJECT_ROOT/scripts/run_fx_shadow.py"
      --broker schwab
      --symbols "$FX_SYMBOL_SET"
      --context-symbols "$FX_CONTEXT_SET"
      --interval-seconds "$FX_INTERVAL"
      --max-iterations "${FX_SHADOW_MAX_ITERS:-0}"
    )
    if [[ "$FX_SIMULATE" == "1" ]]; then
      FX_CMD+=(--simulate)
    fi

    echo "fx_paper=enabled symbols=$FX_SYMBOL_SET context=$FX_CONTEXT_SET interval=$FX_INTERVAL"
    MARKET_DATA_ONLY=1 \
    ALLOW_ORDER_EXECUTION=0 \
    FX_DIRECT_EXECUTION_ENABLED=0 \
    SCHWAB_FOREX_API_VERIFIED="${SCHWAB_FOREX_API_VERIFIED:-0}" \
    nohup "${FX_CMD[@]}" > "$LOG" 2>&1 & disown

    sleep 2
    if fx_running; then
      echo "$LOG"
      echo "fx_loop_started simulate=$FX_SIMULATE paper_mode=$PAPER_MODE"
      OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
    else
      echo "fx_loop failed_to_start"
      tail -n 60 "$LOG" || true
      exit 1
    fi
    ;;
  fx-stop)
    kill_fx_loops
    echo "fx loop stopped"
    ;;
  coinbase-stop)
    kill_coinbase_spot_loops
    echo "coinbase loop stopped"
    ;;
  feed-refresh)
    SOURCE="all"
    SCHWAB_PAPER=1
    COINBASE_PAPER=1
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --source) SOURCE="${2:-all}"; shift ;;
        --paper|--schwab-paper) SCHWAB_PAPER=1 ;;
        --coinbase-paper) COINBASE_PAPER=1 ;;
        *) echo "unknown feed-refresh arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    if [[ "$SOURCE" != "all" && "$SOURCE" != "schwab" && "$SOURCE" != "coinbase" && "$SOURCE" != "fx" ]]; then
      echo "--source must be all, schwab, coinbase, or fx" >&2
      exit 2
    fi

    if [[ "$SOURCE" == "schwab" || "$SOURCE" == "all" ]]; then
      "$PROJECT_ROOT/scripts/ops/opsctl.sh" dividend-drip-sync --json >/dev/null 2>&1 || true
      kill_schwab_live_loops
      sleep 1
      start_schwab_live_loops "$SCHWAB_PAPER"
    fi

    if [[ "$SOURCE" == "coinbase" || "$SOURCE" == "all" ]]; then
      if [[ "$COINBASE_PAPER" == "1" ]]; then
        "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-start --paper --force-restart --live-data
        "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-futures-start --paper --force-restart --live-data
      else
        "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-start --force-restart --live-data
        "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-futures-start --force-restart --live-data
      fi
    fi

    if [[ "$SOURCE" == "fx" || "$SOURCE" == "schwab" || "$SOURCE" == "all" ]]; then
      "$PROJECT_ROOT/scripts/ops/opsctl.sh" fx-market-sync --json >/dev/null 2>&1 || true
      "$PROJECT_ROOT/scripts/ops/opsctl.sh" fx-start --paper --force-restart --live-data
    fi

    "$PROJECT_ROOT/scripts/ops/opsctl.sh" market-correlation-sync --json >/dev/null 2>&1 || true
    ;;
  storage-switch-local|storage-safe-eject)
    DO_REFRESH=1
    DO_EJECT=0
    if [[ "$cmd" == "storage-safe-eject" ]]; then
      DO_EJECT=1
    fi

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --no-refresh) DO_REFRESH=0 ;;
        --eject) DO_EJECT=1 ;;
        --no-eject) DO_EJECT=0 ;;
        *) echo "unknown $cmd arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    write_storage_override local
    apply_storage_route_mode local
    if [[ "$DO_REFRESH" == "1" ]]; then
      restart_collection_after_storage_switch
    fi

    if [[ "$DO_EJECT" == "1" ]]; then
      MOUNT_ROOT="${BOT_LOGS_EXTERNAL_MOUNT:-/Volumes/BOT_LOGS}"
      if ! command -v diskutil >/dev/null 2>&1; then
        echo "diskutil not found; local switch complete, eject manually from Finder" >&2
        exit 1
      fi
      diskutil eject "$MOUNT_ROOT"
    fi
    ;;
  storage-switch-external)
    DO_REFRESH=1
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --no-refresh) DO_REFRESH=0 ;;
        *) echo "unknown storage-switch-external arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    write_storage_override external
    apply_storage_route_mode external
    if [[ "$DO_REFRESH" == "1" ]]; then
      restart_collection_after_storage_switch
    fi
    ;;
  feed)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" "$@"
    ;;
  schwab-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source schwab "$@"
    ;;
  coinbase-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source coinbase "$@"
    ;;
  main-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source main "$@"
    ;;
  futures-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source futures "$@"
    ;;
  schwab-futures-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source schwab_futures "$@"
    ;;
  coinbase-futures-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source coinbase_futures "$@"
    ;;
  fx-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source fx "$@"
    ;;
  timeline-report)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/project_timeline_report.py" "$@"
    ;;
  crash-report)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/crash_report_digest.py" "$@"
    ;;
  training-report)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/training_report.py" "$@"
    ;;
  report-pdfs)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/report_pdf_bundle.py" "$@"
    ;;
  model-card)
    exec "$PY" "$PROJECT_ROOT/scripts/export_model_card.py" "$@"
    ;;
  explainability)
    exec "$PY" "$PROJECT_ROOT/scripts/export_bot_explainability.py" "$@"
    ;;
  strategy-attribution)
    exec "$PY" "$PROJECT_ROOT/scripts/strategy_attribution_report.py" "$@"
    ;;
  paper-calibration)
    exec "$PY" "$PROJECT_ROOT/scripts/paper_execution_calibration_report.py" "$@"
    ;;
  paper-performance)
    exec "$PY" "$PROJECT_ROOT/scripts/paper_performance_report.py" "$@"
    ;;
  post-trade-analysis)
    exec "$PY" "$PROJECT_ROOT/scripts/post_trade_analysis.py" "$@"
    ;;
  macro-bulletin)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/live_macro_bulletin.py" "$@"
    ;;
  macro-replay)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/live_macro_auto_watch.py" --once --replay-full-video "$@"
    ;;
  macro-media-ingest)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/live_macro_media_ingest.py" "$@"
    ;;
  macro-auto-start)
    FORCE_RESTART=0
    RUN_ONCE=0
    PASS_ARGS=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --force-restart) FORCE_RESTART=1 ;;
        --once) RUN_ONCE=1; PASS_ARGS+=("$1") ;;
        *) PASS_ARGS+=("$1") ;;
      esac
      shift
    done

    if [[ "$FORCE_RESTART" == "1" ]]; then
      pkill -f "scripts/ops/live_macro_auto_watch.py" || true
      rm -f "$PROJECT_ROOT/governance/health/macro_auto_watch.pid"
      sleep 1
    fi

    if [[ "$RUN_ONCE" == "1" ]]; then
      exec "$PY" "$PROJECT_ROOT/scripts/ops/live_macro_auto_watch.py" "${PASS_ARGS[@]}"
    fi

    if ps -axo command | grep -F "scripts/ops/live_macro_auto_watch.py" | grep -v grep >/dev/null 2>&1; then
      PID="$(ps -axo pid,command | grep -F "scripts/ops/live_macro_auto_watch.py" | grep -v grep | awk 'NR==1{print $1}')"
      LATEST_LOG="$(ls -1t "$PROJECT_ROOT"/logs/macro_auto_watch_*.log 2>/dev/null | head -n 1)"
      echo "macro_auto_watch already running pid=$PID"
      [[ -n "$LATEST_LOG" ]] && echo "$LATEST_LOG"
      exit 0
    fi

    LOG="$PROJECT_ROOT/logs/macro_auto_watch_$(date -u +%Y%m%d_%H%M%S).log"
    PYTHONUNBUFFERED=1 nohup "$PY" "$PROJECT_ROOT/scripts/ops/live_macro_auto_watch.py" "${PASS_ARGS[@]}" > "$LOG" 2>&1 & disown
    sleep 2
    if ps -axo command | grep -F "scripts/ops/live_macro_auto_watch.py" | grep -v grep >/dev/null 2>&1; then
      echo "$LOG"
      echo "macro_auto_watch_started"
    else
      echo "macro_auto_watch failed_to_start log=$LOG" >&2
      tail -n 60 "$LOG" || true
      exit 1
    fi
    ;;
  macro-auto-stop)
    pkill -f "scripts/ops/live_macro_auto_watch.py" || true
    rm -f "$PROJECT_ROOT/governance/health/macro_auto_watch.pid"
    echo "macro_auto_watch stopped"
    ;;
  macro-auto-status)
    STATUS_PATH="$PROJECT_ROOT/governance/health/macro_auto_watch_status.json"
    PID_PATH="$PROJECT_ROOT/governance/health/macro_auto_watch.pid"
    if [[ -f "$STATUS_PATH" ]]; then
      cat "$STATUS_PATH"
    else
      echo "{\"ok\":false,\"reason\":\"status_missing\",\"status_path\":\"$STATUS_PATH\"}"
    fi
    if [[ -f "$PID_PATH" ]]; then
      echo
      echo "pid=$(cat "$PID_PATH")"
    fi
    ;;
  regime-validate)
    exec "$PY" "$PROJECT_ROOT/scripts/regime_segmented_validate.py" "$@"
    ;;
  retrain-force-full)
    load_runtime_profile live
    RETRAIN_AFTER_HOURS_ONLY=0 \
    RETRAIN_REQUIRE_DATA_QUALITY_FLOOR=0 \
    RETRAIN_REQUIRE_ARTIFACT_FRESHNESS=0 \
    RETRAIN_REQUIRE_SAMPLE_QUOTAS=0 \
    RETRAIN_REQUIRE_FULL_SNAPSHOT_SYNC=0 \
    RETRAIN_REFRESH_PROMOTION_ARTIFACTS=0 \
    RETRAIN_ALLOW_PRECHECK_FAILURES=1 \
    RETRAIN_THERMAL_GUARD=0 \
    RETRAIN_RETIRE_PERSISTENT_LOSERS=0 \
    MLX_METAL_JIT="${MLX_METAL_JIT:-0}" \
    exec "$PY" "$PROJECT_ROOT/scripts/weekly_retrain.py" --continue-on-error "$@"
    ;;
  retrain-force-targeted)
    if [[ $# -eq 0 ]]; then
      echo "retrain-force-targeted requires selector args such as --include-bot-ids or --regime-focus" >&2
      exit 2
    fi
    load_runtime_profile live
    RETRAIN_AFTER_HOURS_ONLY=0 \
    RETRAIN_REQUIRE_DATA_QUALITY_FLOOR=0 \
    RETRAIN_REQUIRE_ARTIFACT_FRESHNESS=0 \
    RETRAIN_REQUIRE_SAMPLE_QUOTAS=0 \
    RETRAIN_REQUIRE_FULL_SNAPSHOT_SYNC=0 \
    RETRAIN_REFRESH_PROMOTION_ARTIFACTS=0 \
    RETRAIN_ALLOW_PRECHECK_FAILURES=1 \
    RETRAIN_THERMAL_GUARD=0 \
    ENABLE_TRADE_BEHAVIOR_RETRAIN=0 \
    RETRAIN_DISTILLATION_STUDENT_EXTRA_PASS=0 \
    RETRAIN_NEW_BOT_EXTRA_PASS=0 \
    RETRAIN_RETIRE_PERSISTENT_LOSERS=0 \
    MLX_METAL_JIT="${MLX_METAL_JIT:-0}" \
    exec "$PY" "$PROJECT_ROOT/scripts/weekly_retrain.py" --continue-on-error "$@"
    ;;
  timeline-install-autoupdate)
    exec "$PROJECT_ROOT/scripts/install_project_timeline_autoupdate_launchd.sh" "$@"
    ;;
  token-refresh)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/premarket_token_guard.py" "$@"
    ;;
  token-refresh-interactive)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/schwab_auth_refresh.py" "$@"
    ;;
  token-install-autorefresh)
    exec "$PROJECT_ROOT/scripts/install_premarket_token_guard_launchd.sh" "$@"
    ;;
  notify-watch)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/mac_notification_watch.py" "$@"
    ;;
  notify-start)
    exec "$PROJECT_ROOT/scripts/install_mac_notification_watch_launchd.sh" "$@"
    ;;
  notify-stop)
    LABEL="com.dankingsley.mac_notification_watch"
    UID_NUM="$(id -u)"
    PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
    launchctl bootout "gui/$UID_NUM" "$PLIST_PATH" >/dev/null 2>&1 || true
    launchctl disable "gui/$UID_NUM/$LABEL" >/dev/null 2>&1 || true
    pkill -f "scripts/ops/mac_notification_watch.py" || true
    rm -f "$PROJECT_ROOT/governance/health/mac_notification_watch.pid"
    echo "notification_watch stopped"
    ;;
  notify-test)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/mac_notification_watch.py" --test "$@"
    ;;
  help|*)
    cat <<'EOF'
opsctl commands:
  start [--profile sim|live] [--force-restart] [--no-coinbase] [--simulate] [--paper|--schwab-paper] [--coinbase-paper] [--coinbase-live-data] [--disable-circuit-breakers] [--run-all-sleeves]
  start-sim [--force-restart] [--no-coinbase] [--disable-circuit-breakers] [--run-all-sleeves]
  start-live [--force-restart] [--no-coinbase] [--paper|--schwab-paper] [--coinbase-paper] [--coinbase-live-data] [--disable-circuit-breakers] [--run-all-sleeves]
  stop
  status
  retrain
  retrain-force-full [extra weekly_retrain args...]
  retrain-force-targeted --include-bot-ids CSV [extra weekly_retrain args...]
  retrain-orchestrate [--bypass-market-guard] [--json]
  scorecard [--lookback-hours 24] [--json]
  sql-sync
  tradingeconomics-sync [--countries CSV] [--market-symbols CSV] [--lookahead-days N] [--news-limit N] [--json]
  macro-context-sync [--json]
  sec-edgar-sync [--symbols CSV] [--timeout N] [--pause-seconds N] [--json]
  extended-quant-sync [--symbols CSV] [--timeout N] [--json]
  tastytrade-sync [--symbols CSV] [--timeout-seconds N] [--sandbox] [--json]
  crypto-market-sync [--symbols CSV] [--timeout N] [--json]
  market-correlation-sync [--lookback-days N] [--bucket-seconds N] [--min-points N] [--json]
  fx-market-sync [--timeout N] [--json]
  dividend-drip-sync [--lookback-days N] [--recent-window-days N] [--json]
  showcase-refresh
  macro-crosscheck [--json]
  source-verification [--json]
  storage-switch-local [--no-refresh]
  storage-switch-external [--no-refresh]
  storage-safe-eject [--no-refresh] [--no-eject]
  sql-maint|sqlite-maint [--vacuum] [--json]
  health
  py314-canary|py314-ready [--refresh-deps] [--skip-install] [--json]
  doctor
  coinbase-start [paper default] [--paper] [--force-restart] [--live-data|--simulate] [--top-n N] [--min-acc X] [--profiles default]
  schwab-futures-start [paper default] [--paper] [--force-restart] [--live-data|--simulate] [--top-n N] [--min-acc X] [--profiles schwab_futures]
  schwab-futures-stop
  coinbase-stop
  coinbase-futures-start [paper default] [--paper] [--force-restart] [--live-data|--simulate] [--top-n N] [--min-acc X] [--profiles crypto_futures]
  coinbase-futures-stop
  fx-start [paper only] [--paper] [--force-restart] [--live-data|--simulate] [--symbols CSV] [--context-symbols CSV] [--interval-seconds N]
  fx-stop
  feed-refresh [paper default] [--source schwab|coinbase|fx|all] [--paper|--schwab-paper] [--coinbase-paper]
  feed [--source schwab|coinbase|fx|futures|schwab_futures|coinbase_futures|main|all] [--symbol SYMBOL] [--lines 40] [--raw] [--include-decisions]
  phone-feed [--host 127.0.0.1|0.0.0.0] [--port 8787] [--source all] [--lines 80] [--include-decisions] [--token TOKEN]
  schwab-tail [--symbol SYMBOL] [--lines 40]
  coinbase-tail [--symbol SYMBOL] [--lines 40]
  main-tail [--symbol SYMBOL] [--lines 40]
  futures-tail [--symbol SYMBOL] [--lines 40]
  schwab-futures-tail [--symbol SYMBOL] [--lines 40]
  coinbase-futures-tail [--symbol SYMBOL] [--lines 40]
  fx-tail [--symbol SYMBOL] [--lines 40]
  timeline-report [--auto] [--json]
  crash-report [--lookback-days N] [--recent-limit N] [--json]
  training-report [--render-pdf] [--allow-gui-pdf-renderer] [--json]
  report-pdfs [--allow-gui-pdf-renderer] [--json]
  model-card [--json]
  explainability [--limit N] [--bot-ids CSV] [--json]
  strategy-attribution [--day YYYYMMDD] [--json]
  paper-calibration [--hours N] [--json]
  paper-performance [--day YYYYMMDD] [--week-days N] [--json]
  post-trade-analysis [--day YYYYMMDD] [--hours N] [--json]
  macro-bulletin [--template powell|fed|generic] [--headline TEXT] [--summary TEXT] [--content TEXT] [--url URL] [--stance auto|hawkish|dovish|neutral|mixed] [--impact low|medium|high|critical] [--expires-hours N] [--status] [--clear] [--json]
  macro-auto-start (--youtube-url URL | --youtube-channel-url URL) [--template powell|fed|generic] [--speaker NAME] [--source NAME] [--symbols CSV] [--poll-seconds N] [--lookback-seconds N] [--expires-hours N] [--correlate-with-schwab-calendar] [--trigger-media-ingest-on-live] [--trigger-media-ingest-before-minutes N] [--media-ingest-cookies-from-browser chrome|safari] [--once] [--force-restart] [--json]
  macro-replay --youtube-url URL [--template powell|fed|generic] [--speaker NAME] [--source NAME] [--symbols CSV] [--replay-window-seconds N] [--expires-hours N] [--json]
  macro-media-ingest --youtube-url URL [--template powell|fed|generic] [--speaker NAME] [--source NAME] [--language en] [--audio-format mp3] [--asr-backend auto|mlx_whisper] [--asr-model MODEL] [--cookies-from-browser chrome|safari] [--wait-for-live-seconds N] [--retry-interval-seconds N] [--force-redownload] [--json]
  macro-auto-stop
  macro-auto-status
  regime-validate [--out-file PATH]
  timeline-install-autoupdate
  token-refresh [--always-auth] [--json]
  token-refresh-interactive [--callback-timeout-seconds N] [--requested-browser BROWSER] [--skip-account-probe] [--json]
  token-install-autorefresh
  notify-watch [--poll-seconds N] [--enable-imessage] [--imessage-recipient DEST] [--imessage-min-severity info|warn|critical] [--imessage-event-allowlist CSV]
  notify-start [--poll-seconds N] [--enable-imessage] [--imessage-recipient DEST] [--imessage-min-severity info|warn|critical] [--imessage-event-allowlist CSV]
  notify-stop
  notify-test [--enable-imessage] [--imessage-recipient DEST] [--imessage-min-severity info|warn|critical] [--imessage-event-allowlist CSV]
EOF
    ;;
esac
