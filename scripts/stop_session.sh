#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME_PY_HELPER="$PROJECT_ROOT/scripts/ops/runtime_python.sh"
PY="$PROJECT_ROOT/.venv314/bin/python"
DAY_UTC="$(date -u +%Y%m%d)"

cd "$PROJECT_ROOT"
[[ -f "$PROJECT_ROOT/scripts/load_ops_thresholds_env.sh" ]] && source "$PROJECT_ROOT/scripts/load_ops_thresholds_env.sh"
[[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]] && source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" live --quiet
if [[ -x "$RUNTIME_PY_HELPER" ]]; then
  PY="$(
    BOT_RUNTIME_LANE="${BOT_OPS_RUNTIME_LANE:-${BOT_RUNTIME_LANE:-canary314}}" /bin/zsh "$RUNTIME_PY_HELPER"
  )"
fi

pkill -f "run_parallel_shadows.py" 2>/dev/null || true
pkill -f "run_parallel_aggressive_modes.py" 2>/dev/null || true
pkill -f "run_shadow_training_loop.py" 2>/dev/null || true
pkill -f "shadow_watchdog.py" 2>/dev/null || true
sleep 2

"$PY" "$PROJECT_ROOT/scripts/ops/sql_link_shard_manager.py" --once --json
"$PY" "$PROJECT_ROOT/scripts/build_one_numbers_report.py"
"$PY" "$PROJECT_ROOT/scripts/daily_auto_verify.py" --day "$DAY_UTC" || true
"$PY" "$PROJECT_ROOT/scripts/safe_mode_guard.py" --trip-streak "${SAFE_MODE_TRIP_STREAK_REQUIRED:-3}" --clear-streak "${SAFE_MODE_CLEAR_STREAK_REQUIRED:-2}" || true
"$PY" "$PROJECT_ROOT/scripts/global_risk_killswitch.py" || true
"$PY" "$PROJECT_ROOT/scripts/observability_exporter.py"
"$PY" "$PROJECT_ROOT/scripts/experiment_tracker.py" --name "runtime_session" --status "completed" --notes "stop_session" || true

echo "session_stop_complete"
