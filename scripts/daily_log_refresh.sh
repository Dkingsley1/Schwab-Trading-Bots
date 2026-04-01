#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_ROOT/scripts/ops/runtime_python.sh"
PYTHON_BIN="$(resolve_runtime_python)"
TODAY_UTC="$(date -u +%Y%m%d)"

cd "$PROJECT_ROOT"

[[ -f "$PROJECT_ROOT/scripts/load_ops_thresholds_env.sh" ]] && source "$PROJECT_ROOT/scripts/load_ops_thresholds_env.sh"
[[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]] && source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" live --quiet

wait_for_sqlite_maintenance() {
  local timeout_s="${DAILY_SQLITE_MAINT_WAIT_SECONDS:-300}"
  local poll_s="${DAILY_SQLITE_MAINT_WAIT_POLL_SECONDS:-5}"
  local waited=0
  while (( waited < timeout_s )); do
    if ! pgrep -f "scripts/sql_hot_retention.py|scripts/sqlite_performance_maintenance.py" >/dev/null 2>&1; then
      return 0
    fi
    echo "sqlite_maintenance_busy waited=${waited}s"
    sleep "$poll_s"
    waited=$((waited + poll_s))
  done
  echo "[WARN] sqlite maintenance still active after ${timeout_s}s; continuing anyway"
  return 0
}

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_tradingeconomics_guest_data.py" --json \
  || echo "[WARN] collect_tradingeconomics_guest_data failed; continuing daily refresh"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_bls_census_data.py" \
  || echo "[WARN] collect_bls_census_data failed; continuing daily refresh"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_official_macro_context.py" --json \
  || echo "[WARN] collect_official_macro_context failed; continuing daily refresh"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_market_micro_context.py" \
  --lookback-days "${MARKET_MICRO_LOOKBACK_DAYS:-21}" \
  --finra-lookback-days "${MARKET_MICRO_FINRA_LOOKBACK_DAYS:-15}" \
  --symbols "${MARKET_MICRO_SYMBOLS:-}" \
  --json \
  || echo "[WARN] collect_market_micro_context failed; continuing daily refresh"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_sec_edgar_context.py" --json \
  || echo "[WARN] collect_sec_edgar_context failed; continuing daily refresh"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_extended_quant_context.py" --json \
  || echo "[WARN] collect_extended_quant_context failed; continuing daily refresh"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_tastytrade_context.py" --json \
  || echo "[WARN] collect_tastytrade_context failed; continuing daily refresh"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_crypto_market_context.py" --json \
  || echo "[WARN] collect_crypto_market_context failed; continuing daily refresh"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_market_crypto_correlation_context.py" --json \
  || echo "[WARN] collect_market_crypto_correlation_context failed; continuing daily refresh"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_fx_market_context.py" --json \
  || echo "[WARN] collect_fx_market_context failed; continuing daily refresh"

wait_for_sqlite_maintenance
# Keep all ingestion on the shard-managed path so refresh jobs do not compete
# with the long-running SQL writer service or rebuild stale backpressure.
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/sql_link_shard_manager.py" --once --json

# Explicit SQLite maintenance step (non-fatal so the rest of daily refresh still runs).
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/sqlite_performance_maintenance.py" \
  --auto-vacuum-over-gb "${SQLITE_AUTO_VACUUM_OVER_GB:-35}" \
  --vacuum-min-interval-hours "${SQLITE_VACUUM_MIN_INTERVAL_HOURS:-24}" \
  --json \
  || echo "[WARN] sqlite_performance_maintenance failed; continuing daily refresh"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ingestion_backpressure_guard.py"
if [[ "${DAILY_ENABLE_SQL_RUNTIME_REPORT:-0}" == "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/sql_runtime_report.py" --day "$TODAY_UTC" \
    || echo "[WARN] sql_runtime_report failed; continuing daily refresh"
else
  echo "sql_runtime_report skipped (set DAILY_ENABLE_SQL_RUNTIME_REPORT=1 to enable)"
fi
DAILY_RUNTIME_SUMMARY_JSON="$PROJECT_ROOT/exports/sql_reports/daily_runtime_summary_${TODAY_UTC}.json"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/daily_runtime_summary.py" --day "$TODAY_UTC" --json \
  > "$DAILY_RUNTIME_SUMMARY_JSON"
cp "$DAILY_RUNTIME_SUMMARY_JSON" "$PROJECT_ROOT/exports/sql_reports/daily_runtime_summary_latest.json" || true
mkdir -p "$PROJECT_ROOT/governance/health"
cp "$DAILY_RUNTIME_SUMMARY_JSON" "$PROJECT_ROOT/governance/health/daily_runtime_summary_latest.json" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_data_center.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_one_numbers_report.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/bot_stack_status_report.py"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/export_model_card.py" --json || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/export_bot_explainability.py" --json || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/paper_execution_calibration_report.py" --hours 24 --json || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/paper_performance_report.py" --day "$TODAY_UTC" --week-days 7 --json || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/strategy_attribution_report.py" --day "$TODAY_UTC" --json || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/post_trade_analysis.py" --day "$TODAY_UTC" --hours 24 --json || true
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
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/data_source_divergence_bot.py" --json || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/macro_crosscheck_report.py" --json || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/source_verification_report.py" --json || true
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
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/report_pdf_bundle.py" --json || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/update_showcase_highlights.py" || true
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
