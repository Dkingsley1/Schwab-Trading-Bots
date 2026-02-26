#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$PROJECT_ROOT/.venv312/bin/python"

FORCE_RESTART=0
WITH_COINBASE=1
SIMULATE=0
DISABLE_BREAKERS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-restart) FORCE_RESTART=1 ;;
    --no-coinbase) WITH_COINBASE=0 ;;
    --simulate) SIMULATE=1 ;;
    --disable-circuit-breakers) DISABLE_BREAKERS=1 ;;
  esac
  shift
done

cd "$PROJECT_ROOT"

"$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true
"$PY" "$PROJECT_ROOT/scripts/ops/storage_failback_sync.py" --json >/dev/null 2>&1 || true
"$PY" "$PROJECT_ROOT/scripts/ops/preflight_autofix.py" --broker "${DATA_BROKER:-schwab}" $([[ "$SIMULATE" == "1" ]] && echo --simulate) --json || true

if [[ "$FORCE_RESTART" == "1" ]]; then
  pkill -f "scripts/run_all_sleeves.py --with-aggressive-modes" || true
  pkill -f "scripts/run_parallel_shadows.py" || true
  pkill -f "scripts/run_parallel_aggressive_modes.py" || true
  pkill -f "scripts/run_shadow_training_loop.py --broker coinbase" || true
  sleep 1
fi

export MARKET_DATA_ONLY="${MARKET_DATA_ONLY:-1}"
export ALLOW_ORDER_EXECUTION="${ALLOW_ORDER_EXECUTION:-0}"

LOG_ALL="logs/all_sleeves_$(date -u +%Y%m%d_%H%M%S).log"
CMD=("$PY" "$PROJECT_ROOT/scripts/run_all_sleeves.py" --with-aggressive-modes)
if [[ "$SIMULATE" == "1" ]]; then
  CMD+=(--simulate)
fi
if [[ "$DISABLE_BREAKERS" == "1" ]]; then
  CMD+=(--disable-circuit-breakers)
fi

nohup "${CMD[@]}" > "$LOG_ALL" 2>&1 & disown

echo "all_sleeves_log=$LOG_ALL"

if [[ "$WITH_COINBASE" == "1" ]]; then
  LOG_CB="logs/coinbase_live_$(date -u +%Y%m%d_%H%M%S).log"
  nohup "$PY" "$PROJECT_ROOT/scripts/run_shadow_training_loop.py"     --broker coinbase     --symbols "${COINBASE_WATCH_SYMBOLS:-BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD}"     --interval-seconds "${COINBASE_WATCH_INTERVAL_SECONDS:-20}"     --max-iterations 0     --simulate     > "$LOG_CB" 2>&1 & disown
  echo "coinbase_log=$LOG_CB"
fi

"$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
