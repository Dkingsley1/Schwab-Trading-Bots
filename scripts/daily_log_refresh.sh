#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
TODAY_UTC="$(date -u +%Y%m%d)"

cd "$PROJECT_ROOT"

[[ -f "$PROJECT_ROOT/scripts/load_ops_thresholds_env.sh" ]] && source "$PROJECT_ROOT/scripts/load_ops_thresholds_env.sh"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/link_jsonl_to_sql.py" --mode sqlite

# Explicit SQLite maintenance step (non-fatal so the rest of daily refresh still runs).
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/sqlite_performance_maintenance.py" \
  --auto-vacuum-over-gb "${SQLITE_AUTO_VACUUM_OVER_GB:-35}" \
  --vacuum-min-interval-hours "${SQLITE_VACUUM_MIN_INTERVAL_HOURS:-24}" \
  --json \
  || echo "[WARN] sqlite_performance_maintenance failed; continuing daily refresh"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ingestion_backpressure_guard.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/sql_runtime_report.py" --day "$TODAY_UTC"
DAILY_RUNTIME_SUMMARY_JSON="$PROJECT_ROOT/exports/sql_reports/daily_runtime_summary_${TODAY_UTC}.json"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/daily_runtime_summary.py" --day "$TODAY_UTC" --json \
  > "$DAILY_RUNTIME_SUMMARY_JSON"
cp "$DAILY_RUNTIME_SUMMARY_JSON" "$PROJECT_ROOT/exports/sql_reports/daily_runtime_summary_latest.json" || true
mkdir -p "$PROJECT_ROOT/governance/health"
cp "$DAILY_RUNTIME_SUMMARY_JSON" "$PROJECT_ROOT/governance/health/daily_runtime_summary_latest.json" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_data_center.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_one_numbers_report.py" --day "$TODAY_UTC"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/bot_stack_status_report.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/sleeve_slo_guard.py" --day "$TODAY_UTC" --once || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/sleeve_allocator.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/portfolio_risk_ledger.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/execution_budgeter.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/distill_new_bots.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/session_ready_check.py" --json || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/daily_state_snapshot_drill.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_executive_dashboard.py" --day "$TODAY_UTC"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/health_gates.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/canary_rollout_guard.py" --day "$TODAY_UTC" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/walk_forward_validate.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/walk_forward_promotion_gate.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/lane_promotion_gate.py" --json || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/regime_segmented_validate.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/new_bot_graduation_gate.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/leak_overfit_guard.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/promotion_readiness_summary.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/canary_diagnostics_loop.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/promotion_bottleneck_focus.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/weekly_gate_blocker_report.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/model_lifecycle_hygiene.py" \
  --keep-backups "${MODEL_LIFECYCLE_KEEP_BACKUPS:-25}" \
  --min-free-gb "${MODEL_LIFECYCLE_MIN_FREE_GB:-10}" \
  --apply-prune \
  --repair-stale-artifacts \
  --apply-repair \
  --json \
  || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/champion_challenger_registry.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/sync_snapshot_health_to_sql.py" --json || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/data_retention_policy.py" --apply --skip-sqlite-vacuum || echo "[WARN] data_retention_policy failed; continuing daily refresh"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/dependency_guard.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/secret_scan.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/daily_auto_verify.py" --day "$TODAY_UTC" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/observability_exporter.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/event_bus_relay.py" --day "$TODAY_UTC" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/safe_mode_guard.py" --trip-streak "${SAFE_MODE_TRIP_STREAK_REQUIRED:-3}" --clear-streak "${SAFE_MODE_CLEAR_STREAK_REQUIRED:-2}" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/global_risk_killswitch.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/slo_burn_rate_guard.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/security_hardening_audit.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/backup_restore_verify.py" || true

# AUTO_PRUNE_STATE_SNAPSHOT_DRILLS
if [[ -x "scripts/prune_state_snapshot_drills.sh" ]]; then
  ./scripts/prune_state_snapshot_drills.sh "/Users/dankingsley/PycharmProjects/schwab_trading_bot" || true
fi

# Repair known launchd agents if they have non-zero last exit code.
if command -v launchctl >/dev/null 2>&1; then
  USER_DOMAIN="gui/$(id -u)"
  for label in com.dankingsley.project_timeline_autoupdate com.dankingsley.schwab.logrefresh; do
    if launchctl print "$USER_DOMAIN/$label" 2>/dev/null | grep -Eq "last exit code = [1-9][0-9]*"; then
      launchctl kickstart -k "$USER_DOMAIN/$label" >/dev/null 2>&1 || true
      echo "launchd_repair label=$label action=kickstart"
    fi
  done
fi

echo "daily_log_refresh complete day_utc=$TODAY_UTC"
