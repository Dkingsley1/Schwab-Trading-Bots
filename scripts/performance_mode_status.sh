#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE_FILE="$PROJECT_ROOT/governance/perf/heavy_load_mode.json"

if [[ -f "$MODE_FILE" ]]; then
  echo "HEAVY_MODE=ON"
  cat "$MODE_FILE"
else
  echo "HEAVY_MODE=OFF"
fi

echo ""
echo "Launchd jobs:"
launchctl list | grep "com.dankingsley.retrain.daily_small\|com.dankingsley.retrain.weekly_full\|com.dankingsley.daily_log_refresh\|com.dankingsley.shadow_watchdog" || true

echo ""
echo "Bot/watchdog processes:"
ps -axo pid,ni,etime,command | grep -E "run_parallel_shadows.py|run_parallel_aggressive_modes.py|run_shadow_training_loop.py|shadow_watchdog.py" | grep -v grep || true
