#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="$PROJECT_ROOT/exports/incidents/$STAMP"
mkdir -p "$OUT_DIR"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "project_root=$PROJECT_ROOT"
} > "$OUT_DIR/README.txt"

uptime > "$OUT_DIR/uptime.txt" 2>&1 || true
memory_pressure > "$OUT_DIR/memory_pressure.txt" 2>&1 || true
vm_stat > "$OUT_DIR/vm_stat.txt" 2>&1 || true
ps -axo pid,ppid,etime,%cpu,%mem,rss,command > "$OUT_DIR/ps_full.txt" 2>&1 || true
ps -axo pid,etime,%cpu,%mem,command | grep -E "run_parallel_shadows.py|run_parallel_aggressive_modes.py|run_shadow_training_loop.py|shadow_watchdog.py" | grep -v grep > "$OUT_DIR/ps_trading.txt" 2>&1 || true
if [ -f "$PROJECT_ROOT/governance/watchdog/watchdog_events_$(date -u +%Y%m%d).jsonl" ]; then
  tail -n 200 "$PROJECT_ROOT/governance/watchdog/watchdog_events_$(date -u +%Y%m%d).jsonl" > "$OUT_DIR/watchdog_events_tail.jsonl" || true
fi
for f in ingestion_backpressure_latest.json health_gates_latest.json resource_guard_latest.json; do
  if [ -f "$PROJECT_ROOT/governance/health/$f" ]; then
    cp "$PROJECT_ROOT/governance/health/$f" "$OUT_DIR/"
  fi
done

echo "$OUT_DIR"
