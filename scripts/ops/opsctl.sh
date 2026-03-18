#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${BOT_PYTHON_BIN:-$PROJECT_ROOT/.venv312/bin/python}"

cmd="${1:-help}"
shift || true

PROFILE="${BOT_RUNTIME_PROFILE:-live}"
case "$cmd" in
  status|start|start-sim|start-live|sql-sync|tradingeconomics-sync|coinbase-start|coinbase-futures-start|schwab-futures-start|feed-refresh|retrain-force-full|retrain-force-targeted|token-refresh|token-refresh-interactive|macro-bulletin|macro-auto-start|macro-replay|macro-media-ingest|macro-auto-stop|macro-auto-status)
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
  load_runtime_profile live
  "$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true
  "$PY" "$PROJECT_ROOT/scripts/ops/storage_failback_sync.py" --json >/dev/null 2>&1 || true

  local log_file="$PROJECT_ROOT/logs/all_sleeves_$(date -u +%Y%m%d_%H%M%S).log"
  local -a cmd=(
    "$PY" "$PROJECT_ROOT/scripts/run_all_sleeves.py"
    --with-aggressive-modes
  )

  PYTHONUNBUFFERED=1 nohup "${cmd[@]}" > "$log_file" 2>&1 & disown
  sleep 2

  if ps -axo command | grep -F "scripts/run_all_sleeves.py --with-aggressive-modes" | grep -v grep >/dev/null 2>&1; then
    echo "$log_file"
    echo "schwab_live_loops_started simulate=0"
    OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
    return 0
  fi

  echo "schwab_live_loops_failed_to_start log=$log_file" >&2
  tail -n 60 "$log_file" || true
  return 1
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
    pkill -f "scripts/run_shadow_training_loop.py --broker coinbase" || true
    pkill -f "scripts/run_shadow_training_loop.py --broker schwab --profile schwab_futures" || true
    pkill -f "scripts/ops/sql_link_writer_service.py" || true
    echo "stopped core loops"
    ;;
  status)
    ps -axo pid,etime,command | grep -E "run_all_sleeves.py|run_parallel_shadows.py|run_parallel_aggressive_modes.py|run_shadow_training_loop.py --broker coinbase|run_shadow_training_loop.py --broker schwab --profile schwab_futures|sql_link_writer_service.py" | grep -v grep || true
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
    exec "$PY" "$PROJECT_ROOT/scripts/ops/sql_link_writer_service.py" --once "$@"
    ;;
  tradingeconomics-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_tradingeconomics_guest_data.py" "$@"
    ;;
  sqlite-maint)
    exec "$PY" "$PROJECT_ROOT/scripts/sqlite_performance_maintenance.py" "$@"
    ;;
  health)
    exec "$PY" "$PROJECT_ROOT/scripts/daily_auto_verify.py" --json "$@"
    ;;
  py314-canary)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/python314_canary.py" --json "$@"
    ;;
  doctor)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/doctor.py" "$@"
    ;;
  schwab-futures-start)
    FORCE_RESTART=0
    SCHWAB_SIMULATE="${SCHWAB_FUTURES_SIMULATE:-1}"
    PAPER_MODE=0
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
    COINBASE_SIMULATE="${COINBASE_START_SIMULATE:-1}"
    PAPER_MODE=0
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
      pkill -f "scripts/run_shadow_training_loop.py --broker coinbase" || true
      sleep 1
    fi

    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase" | grep -v grep >/dev/null 2>&1; then
      PID="$(ps -axo pid,command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase" | grep -v grep | awk 'NR==1{print $1}')"
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
    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase" | grep -v grep >/dev/null 2>&1; then
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
    COINBASE_SIMULATE="${COINBASE_FUTURES_SIMULATE:-1}"
    PAPER_MODE=0
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
      --context-symbols "${COINBASE_FUTURES_CONTEXT_SYMBOLS:-BTC-USD,ETH-USD,SOL-USD}"
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
  coinbase-stop)
    pkill -f "scripts/run_shadow_training_loop.py --broker coinbase" || true
    echo "coinbase loop stopped"
    ;;
  feed-refresh)
    SOURCE="all"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --source) SOURCE="${2:-all}"; shift ;;
        *) echo "unknown feed-refresh arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    if [[ "$SOURCE" != "all" && "$SOURCE" != "schwab" && "$SOURCE" != "coinbase" ]]; then
      echo "--source must be all, schwab, or coinbase" >&2
      exit 2
    fi

    if [[ "$SOURCE" == "schwab" || "$SOURCE" == "all" ]]; then
      kill_schwab_live_loops
      sleep 1
      start_schwab_live_loops
    fi

    if [[ "$SOURCE" == "coinbase" || "$SOURCE" == "all" ]]; then
      "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-start --force-restart --live-data
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
    RETRAIN_RETIRE_PERSISTENT_LOSERS=0 \
    MLX_METAL_JIT="${MLX_METAL_JIT:-0}" \
    exec "$PY" "$PROJECT_ROOT/scripts/weekly_retrain.py" --continue-on-error --skip-master-update "$@"
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
  start [--profile sim|live] [--force-restart] [--no-coinbase] [--simulate] [--coinbase-live-data] [--disable-circuit-breakers] [--run-all-sleeves]
  start-sim [--force-restart] [--no-coinbase] [--disable-circuit-breakers] [--run-all-sleeves]
  start-live [--force-restart] [--no-coinbase] [--coinbase-live-data] [--disable-circuit-breakers] [--run-all-sleeves]
  stop
  status
  retrain
  retrain-force-full [extra weekly_retrain args...]
  retrain-force-targeted --include-bot-ids CSV [extra weekly_retrain args...]
  retrain-orchestrate [--bypass-market-guard] [--json]
  scorecard [--lookback-hours 24] [--json]
  sql-sync
  tradingeconomics-sync [--countries CSV] [--market-symbols CSV] [--lookahead-days N] [--news-limit N] [--json]
  sqlite-maint [--vacuum] [--json]
  health
  py314-canary [--refresh-deps] [--skip-install] [--json]
  doctor
  coinbase-start [--paper] [--force-restart] [--live-data|--simulate] [--top-n N] [--min-acc X] [--profiles default]
  schwab-futures-start [--paper] [--force-restart] [--live-data|--simulate] [--top-n N] [--min-acc X] [--profiles schwab_futures]
  schwab-futures-stop
  coinbase-stop
  coinbase-futures-start [--paper] [--force-restart] [--live-data|--simulate] [--top-n N] [--min-acc X] [--profiles crypto_futures]
  coinbase-futures-stop
  feed-refresh [--source schwab|coinbase|all]
  feed [--source schwab|coinbase|all] [--symbol SYMBOL] [--lines 40] [--raw]
  schwab-tail [--symbol SYMBOL] [--lines 40]
  coinbase-tail [--symbol SYMBOL] [--lines 40]
  timeline-report [--auto] [--json]
  crash-report [--lookback-days N] [--recent-limit N] [--json]
  training-report [--render-pdf] [--allow-gui-pdf-renderer] [--json]
  report-pdfs [--allow-gui-pdf-renderer] [--json]
  model-card [--json]
  explainability [--limit N] [--bot-ids CSV] [--json]
  strategy-attribution [--day YYYYMMDD] [--json]
  paper-calibration [--hours N] [--json]
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
