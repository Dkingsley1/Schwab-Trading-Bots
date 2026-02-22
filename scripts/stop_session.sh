#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$PROJECT_ROOT/.venv312/bin/python"
DAY_UTC="$(date -u +%Y%m%d)"

cd "$PROJECT_ROOT"
[[ -f "$PROJECT_ROOT/scripts/load_ops_thresholds_env.sh" ]] && source "$PROJECT_ROOT/scripts/load_ops_thresholds_env.sh"

# graceful stop
pkill -f "run_parallel_shadows.py" 2>/dev/null || true
pkill -f "run_parallel_aggressive_modes.py" 2>/dev/null || true
pkill -f "run_shadow_training_loop.py" 2>/dev/null || true
pkill -f "shadow_watchdog.py" 2>/dev/null || true
sleep 2

# post-session golden report
"$PY" "$PROJECT_ROOT/scripts/link_jsonl_to_sql.py" --mode sqlite
"$PY" "$PROJECT_ROOT/scripts/build_one_numbers_report.py" --day "$DAY_UTC"
"$PY" "$PROJECT_ROOT/scripts/daily_auto_verify.py" --day "$DAY_UTC" || true
"$PY" "$PROJECT_ROOT/scripts/safe_mode_guard.py" --trip-streak "${SAFE_MODE_TRIP_STREAK_REQUIRED:-3}" --clear-streak "${SAFE_MODE_CLEAR_STREAK_REQUIRED:-2}" || true
"$PY" "$PROJECT_ROOT/scripts/observability_exporter.py"
"$PY" "$PROJECT_ROOT/scripts/experiment_tracker.py" --name "runtime_session" --status "completed" --notes "stop_session" || true

echo "session_stop_complete"
