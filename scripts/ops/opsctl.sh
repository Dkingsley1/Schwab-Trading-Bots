#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$PROJECT_ROOT/.venv312/bin/python"

cmd="${1:-help}"
shift || true

case "$cmd" in
  start)
    exec "$PROJECT_ROOT/scripts/ops/start_stack.sh" "$@"
    ;;
  stop)
    pkill -f "scripts/run_all_sleeves.py --with-aggressive-modes" || true
    pkill -f "scripts/run_parallel_shadows.py" || true
    pkill -f "scripts/run_parallel_aggressive_modes.py" || true
    pkill -f "scripts/run_shadow_training_loop.py --broker coinbase" || true
    pkill -f "scripts/ops/sql_link_writer_service.py" || true
    echo "stopped core loops"
    ;;
  status)
    ps -axo pid,etime,command | grep -E "run_all_sleeves.py|run_parallel_shadows.py|run_parallel_aggressive_modes.py|run_shadow_training_loop.py --broker coinbase|sql_link_writer_service.py" | grep -v grep || true
    "$PY" "$PROJECT_ROOT/scripts/ops/preflight_autofix.py" --broker "${DATA_BROKER:-schwab}" --json || true
    ;;
  retrain)
    exec "$PY" "$PROJECT_ROOT/scripts/weekly_retrain.py" --continue-on-error "$@"
    ;;
  sql-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/sql_link_writer_service.py" --once "$@"
    ;;
  health)
    exec "$PY" "$PROJECT_ROOT/scripts/daily_auto_verify.py" --json "$@"
    ;;
  coinbase-start)
    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase" | grep -v grep >/dev/null 2>&1; then
      PID="$(ps -axo pid,command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase" | grep -v grep | awk 'NR==1{print $1}')"
      LATEST_LOG="$(ls -1t "$PROJECT_ROOT"/logs/coinbase_live_*.log 2>/dev/null | head -n 1)"
      echo "coinbase_loop already running pid=$PID"
      [[ -n "$LATEST_LOG" ]] && echo "$LATEST_LOG"
      exit 0
    fi

    "$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true

    LOG="$PROJECT_ROOT/logs/coinbase_live_$(date -u +%Y%m%d_%H%M%S).log"
    ADAPTIVE_INTERVAL_ENABLED="${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-0}" nohup "$PY" "$PROJECT_ROOT/scripts/run_shadow_training_loop.py" \
      --broker coinbase \
      --symbols "${COINBASE_WATCH_SYMBOLS:-BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD}" \
      --interval-seconds "${COINBASE_WATCH_INTERVAL_SECONDS:-20}" \
      --max-iterations 0 \
      --simulate \
      > "$LOG" 2>&1 & disown

    sleep 2
    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase" | grep -v grep >/dev/null 2>&1; then
      echo "$LOG"
      "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --require-coinbase --json >/dev/null 2>&1 || true
    else
      echo "coinbase_loop failed_to_start"
      tail -n 40 "$LOG" || true
      exit 1
    fi
    ;;
  coinbase-stop)
    pkill -f "scripts/run_shadow_training_loop.py --broker coinbase" || true
    echo "coinbase loop stopped"
    ;;
  coinbase-tail)
    LOG="$(ls -1t "$PROJECT_ROOT"/logs/coinbase_live_*.log 2>/dev/null | head -n 1)"
    if [[ -z "$LOG" ]]; then
      echo "no coinbase_live log found"
      exit 1
    fi
    echo "tailing: $LOG"
    exec tail -f "$LOG"
    ;;
  help|*)
    cat <<'EOF'
opsctl commands:
  start [--force-restart] [--no-coinbase] [--simulate] [--disable-circuit-breakers]
  stop
  status
  retrain
  sql-sync
  health
  coinbase-start
  coinbase-stop
  coinbase-tail
EOF
    ;;
esac
