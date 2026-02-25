#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="$PROJECT_ROOT/exports/debug_snapshots"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_ROOT/$STAMP"

mkdir -p "$OUT_DIR"

echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT_DIR/meta.txt"
echo "project_root=$PROJECT_ROOT" >> "$OUT_DIR/meta.txt"
echo "cwd=$(pwd)" >> "$OUT_DIR/meta.txt"

# Core process views
ps -axo pid,etime,%cpu,%mem,command > "$OUT_DIR/ps_full.txt" || true
ps -axo pid,etime,%cpu,%mem,command | grep -E "run_all_sleeves.py|run_parallel_shadows.py|run_parallel_aggressive_modes.py|run_shadow_training_loop.py|shadow_watchdog.py|weekly_retrain.py" | grep -v grep > "$OUT_DIR/ps_trading.txt" || true

# Locks
ls -la "$PROJECT_ROOT/governance" > "$OUT_DIR/governance_ls.txt" || true
ls -la "$PROJECT_ROOT/governance/locks" > "$OUT_DIR/governance_locks_ls.txt" || true
[[ -f "$PROJECT_ROOT/governance/parallel_shadow.lock" ]] && cat "$PROJECT_ROOT/governance/parallel_shadow.lock" > "$OUT_DIR/parallel_shadow_lock.txt" || true
[[ -f "$PROJECT_ROOT/governance/all_sleeves.lock" ]] && cat "$PROJECT_ROOT/governance/all_sleeves.lock" > "$OUT_DIR/all_sleeves_lock.txt" || true
[[ -f "$PROJECT_ROOT/governance/mlx_retrain.lock" ]] && cat "$PROJECT_ROOT/governance/mlx_retrain.lock" > "$OUT_DIR/mlx_retrain_lock.txt" || true

# Resource/system quick snapshot
uptime > "$OUT_DIR/uptime.txt" || true
vm_stat > "$OUT_DIR/vm_stat.txt" || true

# Key health files
for f in \
  "$PROJECT_ROOT/governance/health/daily_auto_verify_latest.json" \
  "$PROJECT_ROOT/governance/health/session_ready_latest.json" \
  "$PROJECT_ROOT/governance/watchdog/sleeve_slo_latest.json" \
  "$PROJECT_ROOT/governance/walk_forward/promotion_gate_latest.json" \
  "$PROJECT_ROOT/exports/one_numbers/one_numbers_summary.json" \
  "$PROJECT_ROOT/governance/allocator/sleeve_allocator_latest.json" \
  "$PROJECT_ROOT/governance/risk/portfolio_risk_latest.json" \
  "$PROJECT_ROOT/governance/risk/execution_budget_latest.json" \
  "$PROJECT_ROOT/exports/bot_stack_status/latest.json"
do
  if [[ -f "$f" ]]; then
    cp "$f" "$OUT_DIR/"
  fi
done

# Recent watchdog / backup events
for g in \
  "$PROJECT_ROOT/governance/watchdog/watchdog_events_$(date -u +%Y%m%d).jsonl" \
  "$PROJECT_ROOT/governance/watchdog/sleeve_slo_events_$(date -u +%Y%m%d).jsonl" \
  "$PROJECT_ROOT/governance/watchdog/backup_restore_events.jsonl" \
  "$PROJECT_ROOT/governance/watchdog/state_snapshot_drill_events.jsonl"
do
  if [[ -f "$g" ]]; then
    tail -n 200 "$g" > "$OUT_DIR/$(basename "$g")"
  fi
done

# Latest retrain and bot logs pointers
ls -1t "$PROJECT_ROOT/logs"/trade_behavior_policy_*.json 2>/dev/null | head -n 5 > "$OUT_DIR/latest_trade_behavior_logs.txt" || true
ls -1t "$PROJECT_ROOT/logs"/brain_refinery_*.json 2>/dev/null | head -n 20 > "$OUT_DIR/latest_brain_logs.txt" || true

# Git context
(git -C "$PROJECT_ROOT" status --short && echo && git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD && git -C "$PROJECT_ROOT" rev-parse HEAD) > "$OUT_DIR/git_status.txt" || true

# Summary index
{
  echo "Debug snapshot created"
  echo "path=$OUT_DIR"
  echo "files=$(ls -1 "$OUT_DIR" | wc -l | tr -d ' ')"
  echo "top files:"
  ls -1 "$OUT_DIR" | sed -n '1,25p'
} > "$OUT_DIR/README.txt"

# latest symlink
ln -sfn "$OUT_DIR" "$OUT_ROOT/latest"

echo "snapshot_ok path=$OUT_DIR"
echo "latest=$OUT_ROOT/latest"
