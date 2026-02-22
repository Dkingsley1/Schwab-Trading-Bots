#!/bin/zsh
set -euo pipefail

THRESHOLD_CPU=120
ON_STREAK_REQ=3
OFF_STREAK_REQ=8

STATE_DIR="$HOME/.local/state/schwab_perf"
MODE_FILE="$STATE_DIR/mode"
HIGH_FILE="$STATE_DIR/high_streak"
LOW_FILE="$STATE_DIR/low_streak"
LOG_FILE="$STATE_DIR/auto_switch.log"

mkdir -p "$STATE_DIR"
[[ -f "$MODE_FILE" ]] || echo "normal" > "$MODE_FILE"
[[ -f "$HIGH_FILE" ]] || echo "0" > "$HIGH_FILE"
[[ -f "$LOW_FILE" ]] || echo "0" > "$LOW_FILE"

cpu_sum=$(/bin/ps -axo %cpu,command | /usr/bin/awk '
/Final Cut Pro|Logic Pro/ {sum += $1}
END {printf "%.0f", sum+0}
')

mode=$(cat "$MODE_FILE")
high=$(cat "$HIGH_FILE")
low=$(cat "$LOW_FILE")

if (( cpu_sum >= THRESHOLD_CPU )); then
  high=$((high+1)); low=0
else
  low=$((low+1)); high=0
fi

echo "$high" > "$HIGH_FILE"
echo "$low" > "$LOW_FILE"

heavy_on() {
  /bin/launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.dankingsley.schwab.logrefresh.plist" 2>/dev/null || true
  /bin/launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.dankingsley.retrain.daily_small.plist" 2>/dev/null || true
  /bin/launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.dankingsley.retrain.weekly_full.plist" 2>/dev/null || true
  /usr/bin/pkill -f "run_parallel_aggressive_modes.py" 2>/dev/null || true
  for p in $(/usr/bin/pgrep -f "run_parallel_shadows.py|run_shadow_training_loop.py|shadow_watchdog.py" || true); do
    /usr/bin/renice +10 -p "$p" >/dev/null || true
  done
  echo "heavy" > "$MODE_FILE"
}

heavy_off() {
  /bin/launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.dankingsley.schwab.logrefresh.plist" 2>/dev/null || true
  /bin/launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.dankingsley.retrain.daily_small.plist" 2>/dev/null || true
  /bin/launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.dankingsley.retrain.weekly_full.plist" 2>/dev/null || true
  echo "normal" > "$MODE_FILE"
}

if [[ "$mode" == "normal" && "$high" -ge "$ON_STREAK_REQ" ]]; then
  heavy_on
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=heavy cpu_sum=$cpu_sum" >> "$LOG_FILE"
elif [[ "$mode" == "heavy" && "$low" -ge "$OFF_STREAK_REQ" ]]; then
  heavy_off
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=normal cpu_sum=$cpu_sum" >> "$LOG_FILE"
else
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=$mode cpu_sum=$cpu_sum high=$high low=$low" >> "$LOG_FILE"
fi
