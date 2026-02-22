#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE_DIR="$PROJECT_ROOT/governance/perf"
MODE_FILE="$MODE_DIR/heavy_load_mode.json"
DRY_RUN="${1:-}"

run_cmd() {
  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "[dry-run] $*"
  else
    eval "$*"
  fi
}

mkdir -p "$MODE_DIR"

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PAYLOAD="{\"enabled\":true,\"mode\":\"heavy_load\",\"timestamp_utc\":\"$TS\"}"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[dry-run] write $MODE_FILE => $PAYLOAD"
else
  printf '%s\n' "$PAYLOAD" > "$MODE_FILE"
fi

for label in \
  com.dankingsley.retrain.daily_small \
  com.dankingsley.retrain.weekly_full \
  com.dankingsley.daily_log_refresh
 do
  run_cmd "launchctl unload \"$HOME/Library/LaunchAgents/${label}.plist\" >/dev/null 2>&1 || true"
  echo "paused launchd job: $label"
done

# Stop aggressive launcher + its child profiles to free CPU quickly.
AGG_PIDS=$(ps -axo pid,command | grep "run_parallel_aggressive_modes.py" | grep -v grep | awk '{print $1}')
for pid in $AGG_PIDS; do
  run_cmd "kill $pid"
  echo "stopped aggressive launcher pid=$pid"
done

AGG_CHILD_PIDS=$(ps -axo pid,command | grep "run_shadow_training_loop.py" | grep -E "intraday_aggressive|swing_aggressive" | grep -v grep | awk '{print $1}')
for pid in $AGG_CHILD_PIDS; do
  run_cmd "kill $pid"
  echo "stopped aggressive child pid=$pid"
done

# Lower scheduler priority for remaining bot processes.
TARGET_PIDS=$(ps -axo pid,command | grep -E "run_parallel_shadows.py|run_shadow_training_loop.py --broker coinbase|shadow_watchdog.py" | grep -v grep | awk '{print $1}')
for pid in $TARGET_PIDS; do
  run_cmd "renice +15 -p $pid >/dev/null 2>&1 || true"
  echo "reniced pid=$pid to +15"
done

echo "heavy load mode enabled"
