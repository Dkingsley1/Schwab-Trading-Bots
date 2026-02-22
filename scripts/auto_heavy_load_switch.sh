#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEAVY_ON_SCRIPT="$PROJECT_ROOT/scripts/performance_mode_heavy_on.sh"
HEAVY_OFF_SCRIPT="$PROJECT_ROOT/scripts/performance_mode_heavy_off.sh"

THRESHOLD_CPU="${AUTO_HEAVY_THRESHOLD_CPU:-120}"
ON_STREAK_REQ="${AUTO_HEAVY_ON_STREAK_REQ:-3}"
OFF_STREAK_REQ="${AUTO_HEAVY_OFF_STREAK_REQ:-8}"
MEMORY_PRESSURE_TRIP="${AUTO_HEAVY_MEMORY_PRESSURE_TRIP:-1}"

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

memory_pressure_high=$(/usr/bin/memory_pressure 2>/dev/null | /usr/bin/awk '/System-wide memory free percentage/ {if ($5+0 < 8) print 1; else print 0}' | tail -n1)
[[ -n "$memory_pressure_high" ]] || memory_pressure_high=0

mode=$(cat "$MODE_FILE")
high=$(cat "$HIGH_FILE")
low=$(cat "$LOW_FILE")

if (( cpu_sum >= THRESHOLD_CPU )) || (( MEMORY_PRESSURE_TRIP == 1 && memory_pressure_high == 1 )); then
  high=$((high+1)); low=0
  reason="editing_cpu_or_memory_pressure"
else
  low=$((low+1)); high=0
  reason="normal_load"
fi

echo "$high" > "$HIGH_FILE"
echo "$low" > "$LOW_FILE"

heavy_on() {
  if [[ -x "$HEAVY_ON_SCRIPT" ]]; then
    "$HEAVY_ON_SCRIPT" >/dev/null 2>&1 || true
  fi
  echo "heavy" > "$MODE_FILE"
}

heavy_off() {
  if [[ -x "$HEAVY_OFF_SCRIPT" ]]; then
    "$HEAVY_OFF_SCRIPT" >/dev/null 2>&1 || true
  fi
  echo "normal" > "$MODE_FILE"
}

if [[ "$mode" == "normal" && "$high" -ge "$ON_STREAK_REQ" ]]; then
  heavy_on
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=heavy cpu_sum=$cpu_sum mem_pressure_high=$memory_pressure_high reason=$reason" >> "$LOG_FILE"
elif [[ "$mode" == "heavy" && "$low" -ge "$OFF_STREAK_REQ" ]]; then
  heavy_off
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=normal cpu_sum=$cpu_sum mem_pressure_high=$memory_pressure_high reason=$reason" >> "$LOG_FILE"
else
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=$mode cpu_sum=$cpu_sum mem_pressure_high=$memory_pressure_high high=$high low=$low reason=$reason" >> "$LOG_FILE"
fi
