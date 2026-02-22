#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
TODAY_UTC="$(date -u +%Y%m%d)"

cd "$PROJECT_ROOT"

[[ -f "$PROJECT_ROOT/scripts/load_ops_thresholds_env.sh" ]] && source "$PROJECT_ROOT/scripts/load_ops_thresholds_env.sh"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/link_jsonl_to_sql.py" --mode sqlite
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/sqlite_performance_maintenance.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ingestion_backpressure_guard.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/sql_runtime_report.py" --day "$TODAY_UTC"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/daily_runtime_summary.py" --day "$TODAY_UTC" --json \
  > "$PROJECT_ROOT/exports/sql_reports/daily_runtime_summary_${TODAY_UTC}.json"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_data_center.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_one_numbers_report.py" --day "$TODAY_UTC"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/health_gates.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/canary_rollout_guard.py" --day "$TODAY_UTC" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/walk_forward_validate.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/walk_forward_promotion_gate.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/regime_segmented_validate.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/champion_challenger_registry.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/data_retention_policy.py" --apply
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/daily_auto_verify.py" --day "$TODAY_UTC" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/observability_exporter.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/event_bus_relay.py" --day "$TODAY_UTC" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/safe_mode_guard.py" --trip-streak "${SAFE_MODE_TRIP_STREAK_REQUIRED:-3}" --clear-streak "${SAFE_MODE_CLEAR_STREAK_REQUIRED:-2}" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/slo_burn_rate_guard.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/security_hardening_audit.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/backup_restore_verify.py" || true

echo "daily_log_refresh complete day_utc=$TODAY_UTC"
