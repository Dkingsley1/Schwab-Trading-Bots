#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${BOT_PYTHON_BIN:-$PROJECT_ROOT/.venv312/bin/python}"

cmd="${1:-help}"
shift || true

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
    PAPER_TOP_N="${TOP_BOT_PAPER_TRADING_TOP_N:-2}"
    PAPER_MIN_ACC="${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.55}"
    PAPER_PROFILES="${TOP_BOT_PAPER_TRADING_PROFILES:-default}"

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
      TOP_BOT_PAPER_TRADING_ENABLED=1       TOP_BOT_PAPER_TRADING_TOP_N="$PAPER_TOP_N"       TOP_BOT_PAPER_TRADING_MIN_ACC="$PAPER_MIN_ACC"       TOP_BOT_PAPER_TRADING_PROFILES="$PAPER_PROFILES"       PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"       PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"       ADAPTIVE_INTERVAL_ENABLED="${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-0}"       nohup "${COINBASE_CMD[@]}" > "$LOG" 2>&1 & disown
    else
      ADAPTIVE_INTERVAL_ENABLED="${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-0}"       nohup "${COINBASE_CMD[@]}" > "$LOG" 2>&1 & disown
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
    PAPER_MIN_ACC="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.53}"
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
      SHADOW_PROFILE="$FUTURES_PROFILE"       SHADOW_DOMAIN=crypto       SHADOW_THRESHOLD_SHIFT="${COINBASE_FUTURES_THRESHOLD_SHIFT:-0.02}"       SIZING_MAX_NOTIONAL_PCT="${COINBASE_FUTURES_MAX_NOTIONAL_PCT:-0.03}"       PORTFOLIO_BASE_BUDGET="${COINBASE_FUTURES_BASE_BUDGET:-0.50}"       CROSS_SYMBOL_MAX_LONG="${COINBASE_FUTURES_MAX_LONG:-4}"       CROSS_SYMBOL_MAX_SHORT="${COINBASE_FUTURES_MAX_SHORT:-4}"       RISK_MAX_DAILY_LOSS_PROXY="${COINBASE_FUTURES_MAX_DAILY_LOSS_PROXY:-0.03}"       LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"       LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"       LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"       LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"       TOP_BOT_PAPER_TRADING_ENABLED=1       TOP_BOT_PAPER_TRADING_TOP_N="$PAPER_TOP_N"       TOP_BOT_PAPER_TRADING_MIN_ACC="$PAPER_MIN_ACC"       TOP_BOT_PAPER_TRADING_PROFILES="$PAPER_PROFILES"       PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"       PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"       ADAPTIVE_INTERVAL_ENABLED="${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-0}"       nohup "${COINBASE_CMD[@]}" > "$LOG" 2>&1 & disown
    else
      SHADOW_PROFILE="$FUTURES_PROFILE"       SHADOW_DOMAIN=crypto       SHADOW_THRESHOLD_SHIFT="${COINBASE_FUTURES_THRESHOLD_SHIFT:-0.02}"       SIZING_MAX_NOTIONAL_PCT="${COINBASE_FUTURES_MAX_NOTIONAL_PCT:-0.03}"       PORTFOLIO_BASE_BUDGET="${COINBASE_FUTURES_BASE_BUDGET:-0.50}"       CROSS_SYMBOL_MAX_LONG="${COINBASE_FUTURES_MAX_LONG:-4}"       CROSS_SYMBOL_MAX_SHORT="${COINBASE_FUTURES_MAX_SHORT:-4}"       RISK_MAX_DAILY_LOSS_PROXY="${COINBASE_FUTURES_MAX_DAILY_LOSS_PROXY:-0.03}"       LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"       LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"       LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"       LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"       ADAPTIVE_INTERVAL_ENABLED="${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-0}"       nohup "${COINBASE_CMD[@]}" > "$LOG" 2>&1 & disown
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
  timeline-install-autoupdate)
    exec "$PROJECT_ROOT/scripts/install_project_timeline_autoupdate_launchd.sh" "$@"
    ;;
  token-refresh)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/premarket_token_guard.py" "$@"
    ;;
  token-install-autorefresh)
    exec "$PROJECT_ROOT/scripts/install_premarket_token_guard_launchd.sh" "$@"
    ;;
  help|*)
    cat <<'EOF'
opsctl commands:
  start [--profile sim|live] [--force-restart] [--no-coinbase] [--simulate] [--disable-circuit-breakers]
  start-sim [--force-restart] [--no-coinbase] [--disable-circuit-breakers]
  start-live [--force-restart] [--no-coinbase] [--disable-circuit-breakers]
  stop
  status
  retrain
  retrain-orchestrate [--bypass-market-guard] [--json]
  scorecard [--lookback-hours 24] [--json]
  sql-sync
  sqlite-maint [--vacuum] [--json]
  health
  py314-canary [--refresh-deps] [--skip-install] [--json]
  doctor
  coinbase-start [--paper] [--force-restart] [--top-n N] [--min-acc X] [--profiles default]
  schwab-futures-start [--paper] [--force-restart] [--top-n N] [--min-acc X] [--profiles schwab_futures]
  schwab-futures-stop
  coinbase-stop
  coinbase-futures-start [--paper] [--force-restart] [--top-n N] [--min-acc X] [--profiles crypto_futures]
  coinbase-futures-stop
  feed [--source schwab|coinbase|all] [--symbol NVDA] [--lines 40] [--raw]
  schwab-tail [--symbol NVDA] [--lines 40]
  coinbase-tail [--symbol BTC-USD] [--lines 40]
  timeline-report [--auto] [--json]
  timeline-install-autoupdate
  token-refresh [--always-auth] [--json]
  token-install-autorefresh
EOF
    ;;
esac
