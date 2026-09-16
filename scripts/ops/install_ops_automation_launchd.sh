#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"
export BOT_RUNTIME_LANE="${BOT_RUNTIME_LANE:-${BOT_SHADOW_RUNTIME_LANE:-canary314}}"
export BOT_PYTHON_VERSION="${BOT_PYTHON_VERSION:-3.14.5}"
export BOT_TRAINING_RUNTIME_LANE="${BOT_TRAINING_RUNTIME_LANE:-canary314}"
export BOT_TRAINING_PYTHON_VERSION="${BOT_TRAINING_PYTHON_VERSION:-3.14.5}"
export PY314_RUNTIME_FLIP_APPROVED="${PY314_RUNTIME_FLIP_APPROVED:-1}"
export PY314_RETIRE_312_ANCHOR="${PY314_RETIRE_312_ANCHOR:-1}"
unset __PYVENV_LAUNCHER__
source "$PROJECT_ROOT/scripts/ops/runtime_python.sh"
if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$RUNTIME_PROFILE" --quiet
fi
PY="$(resolve_runtime_python)"
SCHEDULED_LIFECYCLE_RUNNER="$PROJECT_ROOT/scripts/ops/run_scheduled_lifecycle_job.py"
SQL_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_sql_link_writer_launchd.sh"
FX_MARKET_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_fx_market_context_launchd.sh"
OPTIONS_FLOW_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_options_flow_context_launchd.sh"
OFFICIAL_MACRO_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_official_macro_context_launchd.sh"
SCHWAB_EDUCATION_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_schwab_education_context_launchd.sh"
MARKET_CORR_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_market_crypto_correlation_launchd.sh"
RETENTION_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_data_retention_launchd.sh"
ONE_NUMBERS_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_one_numbers_refresh_launchd.sh"
ONE_NUMBERS_REGRESSION_GUARD_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_one_numbers_regression_guard_launchd.sh"
BACKLOG_RETRY_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_external_backlog_retry_launchd.sh"
STORAGE_BACKPRESSURE_AUTOPILOT_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_storage_backpressure_autopilot_launchd.sh"
TRAINING_DRAIN_AUTOPILOT_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_training_drain_autopilot_launchd.sh"
STORAGE_PRESSURE_CLEARANCE_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_storage_pressure_clearance_launchd.sh"
LOCAL_SQL_SHARD_STANDBY_PRUNE_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_local_sql_shard_standby_prune_launchd.sh"
STORAGE_RECONNECT_INFRABOT_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_storage_reconnect_infrabot_launchd.sh"
INFRA_AUTOFIX_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_infrastructure_autofix_launchd.sh"
MASTER_INFRA_SUPERVISOR_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_master_infrastructure_supervisor_launchd.sh"
SCHWAB_AUTH_SUPERVISOR_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_schwab_auth_supervisor_launchd.sh"
COMMAND_VALIDITY_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_command_validity_launchd.sh"
SYSTEM_DRIFT_GUARD_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_system_drift_guard_launchd.sh"
SYSTEM_DRIFT_AUTOPILOT_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_system_drift_autopilot_launchd.sh"
SYSTEM_CELL_FEDERATION_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_distributed_cell_architecture_launchd.sh"
BOT_QUALITY_AUTOPILOT_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_bot_quality_autopilot_launchd.sh"
STORAGE_STANDBY_PRUNE_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_storage_standby_prune_launchd.sh"
GRADE_REGRESSION_AUTOPILOT_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_grade_regression_autopilot_launchd.sh"
ADAPTIVE_REGRESSION_GUARD_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_adaptive_regression_guard_launchd.sh"
SECTION_GRADE_AUTOPILOT_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_section_grade_autopilot_launchd.sh"
CHROME_HEADLESS_GUARD_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_chrome_headless_guard_launchd.sh"
SYSTEM_SUMMARY_AUTOPILOT_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_system_summary_autopilot_launchd.sh"
CREATIVE_COTENANT_GUARD_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_creative_cotenant_guard_launchd.sh"
SWAP_PRESSURE_GOVERNOR_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_swap_pressure_governor_launchd.sh"
RUNTIME_SMOOTH_MODE_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_runtime_smooth_mode_launchd.sh"
SOAK_SELF_HEAL_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_soak_self_healing_launchd.sh"
SOAK_RELIABILITY_SENTINEL_SCRIPT="$PROJECT_ROOT/scripts/ops/soak_reliability_sentinel.py"
PRODUCTION_RESILIENCE_CONTROL_SCRIPT="$PROJECT_ROOT/scripts/ops/production_resilience_control.py"
PRODUCTION_HARDENING_WATCH_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_production_hardening_watch_launchd.sh"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="${BOT_OPS_LAUNCHD_LOG_DIR:-/tmp/schwab_trading_bot/launchd_ops}"
UID_NUM="$(id -u)"
mkdir -p "$AGENTS_DIR"
mkdir -p "$LOG_DIR"

chmod +x "$SQL_RUN_SCRIPT"
chmod +x "$FX_MARKET_RUN_SCRIPT"
chmod +x "$OPTIONS_FLOW_RUN_SCRIPT"
chmod +x "$OFFICIAL_MACRO_RUN_SCRIPT"
chmod +x "$SCHWAB_EDUCATION_RUN_SCRIPT"
chmod +x "$MARKET_CORR_RUN_SCRIPT"
chmod +x "$RETENTION_RUN_SCRIPT"
chmod +x "$ONE_NUMBERS_RUN_SCRIPT"
chmod +x "$ONE_NUMBERS_REGRESSION_GUARD_RUN_SCRIPT"
chmod +x "$BACKLOG_RETRY_RUN_SCRIPT"
chmod +x "$STORAGE_BACKPRESSURE_AUTOPILOT_RUN_SCRIPT"
chmod +x "$TRAINING_DRAIN_AUTOPILOT_RUN_SCRIPT"
chmod +x "$STORAGE_PRESSURE_CLEARANCE_RUN_SCRIPT"
chmod +x "$LOCAL_SQL_SHARD_STANDBY_PRUNE_RUN_SCRIPT"
chmod +x "$STORAGE_RECONNECT_INFRABOT_RUN_SCRIPT"
chmod +x "$INFRA_AUTOFIX_RUN_SCRIPT"
chmod +x "$MASTER_INFRA_SUPERVISOR_RUN_SCRIPT"
chmod +x "$SCHWAB_AUTH_SUPERVISOR_RUN_SCRIPT"
chmod +x "$COMMAND_VALIDITY_RUN_SCRIPT"
chmod +x "$SYSTEM_DRIFT_GUARD_RUN_SCRIPT"
chmod +x "$SYSTEM_DRIFT_AUTOPILOT_RUN_SCRIPT"
chmod +x "$SYSTEM_CELL_FEDERATION_RUN_SCRIPT"
chmod +x "$BOT_QUALITY_AUTOPILOT_RUN_SCRIPT"
chmod +x "$STORAGE_STANDBY_PRUNE_RUN_SCRIPT"
chmod +x "$GRADE_REGRESSION_AUTOPILOT_RUN_SCRIPT"
chmod +x "$ADAPTIVE_REGRESSION_GUARD_RUN_SCRIPT"
chmod +x "$SECTION_GRADE_AUTOPILOT_RUN_SCRIPT"
chmod +x "$CHROME_HEADLESS_GUARD_RUN_SCRIPT"
chmod +x "$SYSTEM_SUMMARY_AUTOPILOT_RUN_SCRIPT"
chmod +x "$CREATIVE_COTENANT_GUARD_RUN_SCRIPT"
chmod +x "$SWAP_PRESSURE_GOVERNOR_RUN_SCRIPT"
chmod +x "$RUNTIME_SMOOTH_MODE_RUN_SCRIPT"
chmod +x "$SOAK_SELF_HEAL_RUN_SCRIPT"
chmod +x "$PRODUCTION_HARDENING_WATCH_RUN_SCRIPT"

xml_escape() {
  local value="$1"
  value="${value//&/&amp;}"
  value="${value//</&lt;}"
  value="${value//>/&gt;}"
  value="${value//\"/&quot;}"
  value="${value//\'/&apos;}"
  print -rn -- "$value"
}

scheduled_program_arguments() {
  local job_id="$1"
  local artifact="$2"
  local interval="$3"
  local deadline="$4"
  shift 4
  local arg
  print -r -- "  <key>ProgramArguments</key><array>"
  for arg in "$PY" "$SCHEDULED_LIFECYCLE_RUNNER" \
    "--job-id" "$job_id" \
    "--artifact" "$artifact" \
    "--schedule-interval-seconds" "$interval" \
    "--deadline-seconds" "$deadline" \
    "--"; do
    print -r -- "    <string>$(xml_escape "$arg")</string>"
  done
  for arg in "$@"; do
    print -r -- "    <string>$(xml_escape "$arg")</string>"
  done
  print -r -- "  </array>"
}

WATCHDOG_PLIST="$AGENTS_DIR/com.dankingsley.ops.watchdog.plist"
REPORT_PLIST="$AGENTS_DIR/com.dankingsley.ops.daily_report.plist"
CANARY_PLIST="$AGENTS_DIR/com.dankingsley.ops.canary_tuner.plist"
SQL_PLIST="$AGENTS_DIR/com.dankingsley.ops.sql_link_writer.plist"
PROMO_PLIST="$AGENTS_DIR/com.dankingsley.ops.promotion_pipeline.plist"
MARKET_CORR_PLIST="$AGENTS_DIR/com.dankingsley.ops.market_crypto_correlation.plist"
MARKET_CORR_INTERVAL="${MARKET_CRYPTO_CORRELATION_REFRESH_INTERVAL_SECONDS:-300}"
FX_MARKET_PLIST="$AGENTS_DIR/com.dankingsley.ops.fx_market_context.plist"
FX_MARKET_INTERVAL="${FX_MARKET_CONTEXT_REFRESH_INTERVAL_SECONDS:-900}"
OPTIONS_FLOW_PLIST="$AGENTS_DIR/com.dankingsley.ops.options_flow_context.plist"
OPTIONS_FLOW_INTERVAL="${OPTIONS_FLOW_EFFICIENCY_INTERVAL_SECONDS:-${OPTIONS_FLOW_REFRESH_INTERVAL_SECONDS:-3600}}"
OFFICIAL_MACRO_PLIST="$AGENTS_DIR/com.dankingsley.ops.official_macro_context.plist"
OFFICIAL_MACRO_INTERVAL="${OFFICIAL_MACRO_CONTEXT_REFRESH_INTERVAL_SECONDS:-21600}"
SCHWAB_EDUCATION_PLIST="$AGENTS_DIR/com.dankingsley.ops.schwab_education_context.plist"
SCHWAB_EDUCATION_INTERVAL="${SCHWAB_EDUCATION_CONTEXT_REFRESH_INTERVAL_SECONDS:-3600}"
ONE_NUMBERS_PLIST="$AGENTS_DIR/com.dankingsley.ops.one_numbers_refresh.plist"
ONE_NUMBERS_REGRESSION_GUARD_PLIST="$AGENTS_DIR/com.dankingsley.ops.one_numbers_regression_guard.plist"
ONE_NUMBERS_INTERVAL="${ONE_NUMBERS_REFRESH_LAUNCHD_INTERVAL_SECONDS:-180}"
ONE_NUMBERS_REGRESSION_GUARD_INTERVAL="${ONE_NUMBERS_REGRESSION_GUARD_INTERVAL_SECONDS:-300}"
WATCHDOG_INTERVAL="${OPS_WATCHDOG_LAUNCHD_INTERVAL_SECONDS:-180}"
ADAPTIVE_OPS_POLICY_INTERVAL="${ADAPTIVE_OPS_POLICY_INTERVAL_SECONDS:-60}"
MAINT_STRATEGY_PLIST="$AGENTS_DIR/com.dankingsley.ops.maintenance_strategy_reloader.plist"
ADAPTIVE_OPS_POLICY_PLIST="$AGENTS_DIR/com.dankingsley.ops.adaptive_ops_recovery_policy.plist"
RETENTION_PLIST="$AGENTS_DIR/com.dankingsley.ops.data_retention.plist"
RETENTION_INTERVAL="${RETENTION_REFRESH_INTERVAL_SECONDS:-3600}"
BACKLOG_RETRY_PLIST="$AGENTS_DIR/com.dankingsley.ops.external_backlog_retry.plist"
BACKLOG_RETRY_INTERVAL="${EXTERNAL_BACKLOG_RETRY_LAUNCHD_INTERVAL_SECONDS:-300}"
SQL_LINK_WRITER_INTERVAL="${SQL_LINK_WRITER_INTERVAL_SECONDS:-60}"
STORAGE_BACKPRESSURE_AUTOPILOT_PLIST="$AGENTS_DIR/com.dankingsley.ops.storage_backpressure_autopilot.plist"
STORAGE_BACKPRESSURE_AUTOPILOT_INTERVAL="${STORAGE_BACKPRESSURE_AUTOPILOT_INTERVAL_SECONDS:-60}"
TRAINING_DRAIN_AUTOPILOT_PLIST="$AGENTS_DIR/com.dankingsley.ops.training_drain_autopilot.plist"
TRAINING_DRAIN_AUTOPILOT_INTERVAL="${TRAINING_DRAIN_AUTOPILOT_INTERVAL_SECONDS:-300}"
STORAGE_PRESSURE_CLEARANCE_PLIST="$AGENTS_DIR/com.dankingsley.ops.storage_pressure_clearance.plist"
STORAGE_PRESSURE_CLEARANCE_INTERVAL="${STORAGE_PRESSURE_CLEARANCE_INTERVAL_SECONDS:-180}"
LOCAL_SQL_SHARD_STANDBY_PRUNE_PLIST="$AGENTS_DIR/com.dankingsley.ops.local_sql_shard_standby_prune.plist"
LOCAL_SQL_SHARD_STANDBY_PRUNE_INTERVAL="${LOCAL_SQL_SHARD_STANDBY_PRUNE_INTERVAL_SECONDS:-300}"
STORAGE_RECONNECT_INFRABOT_PLIST="$AGENTS_DIR/com.dankingsley.ops.storage_reconnect_infrabot.plist"
STORAGE_RECONNECT_INFRABOT_INTERVAL="${STORAGE_RECONNECT_INFRABOT_INTERVAL_SECONDS:-240}"
WRITER_COORDINATOR_PLIST="$AGENTS_DIR/com.dankingsley.ops.writer_cycle_coordinator.plist"
RETENTION_SHERIFF_PLIST="$AGENTS_DIR/com.dankingsley.ops.retention_debt_sheriff.plist"
BACKPRESSURE_SLO_PLIST="$AGENTS_DIR/com.dankingsley.ops.backpressure_slo_bot.plist"
INFRA_AUTOFIX_PLIST="$AGENTS_DIR/com.dankingsley.ops.infrastructure_autofix.plist"
INFRA_AUTOFIX_INTERVAL="${INFRASTRUCTURE_AUTOFIX_INTERVAL_SECONDS:-300}"
MASTER_INFRA_SUPERVISOR_PLIST="$AGENTS_DIR/com.dankingsley.ops.master_infrastructure_supervisor.plist"
MASTER_INFRA_SUPERVISOR_INTERVAL="${MASTER_INFRASTRUCTURE_SUPERVISOR_INTERVAL_SECONDS:-300}"
SCHWAB_AUTH_SUPERVISOR_PLIST="$AGENTS_DIR/com.dankingsley.ops.schwab_auth_supervisor.plist"
SCHWAB_AUTH_SUPERVISOR_INTERVAL="${SCHWAB_AUTH_SUPERVISOR_INTERVAL_SECONDS:-120}"
COMMAND_VALIDITY_PLIST="$AGENTS_DIR/com.dankingsley.ops.command_validity.plist"
COMMAND_VALIDITY_INTERVAL="${COMMAND_VALIDITY_INTERVAL_SECONDS:-600}"
SYSTEM_DRIFT_GUARD_PLIST="$AGENTS_DIR/com.dankingsley.ops.system_drift_guard.plist"
SYSTEM_DRIFT_GUARD_INTERVAL="${SYSTEM_DRIFT_GUARD_INTERVAL_SECONDS:-600}"
SYSTEM_DRIFT_AUTOPILOT_PLIST="$AGENTS_DIR/com.dankingsley.ops.system_drift_autopilot.plist"
SYSTEM_DRIFT_AUTOPILOT_INTERVAL="${SYSTEM_DRIFT_AUTOPILOT_INTERVAL_SECONDS:-600}"
SYSTEM_CELL_FEDERATION_PLIST="$AGENTS_DIR/com.dankingsley.ops.system_cell_federation.plist"
SYSTEM_CELL_FEDERATION_INTERVAL="${DISTRIBUTED_CELL_ARCHITECTURE_INTERVAL_SECONDS:-300}"
BOT_QUALITY_AUTOPILOT_PLIST="$AGENTS_DIR/com.dankingsley.ops.bot_quality_autopilot.plist"
BOT_QUALITY_AUTOPILOT_INTERVAL="${BOT_QUALITY_AUTOPILOT_INTERVAL_SECONDS:-1800}"
STORAGE_STANDBY_PRUNE_PLIST="$AGENTS_DIR/com.dankingsley.ops.storage_standby_prune.plist"
STORAGE_STANDBY_PRUNE_INTERVAL="${BOT_LOGS_STANDBY_PRUNE_INTERVAL_SECONDS:-300}"
GRADE_REGRESSION_AUTOPILOT_PLIST="$AGENTS_DIR/com.dankingsley.ops.grade_regression_autopilot.plist"
GRADE_REGRESSION_AUTOPILOT_INTERVAL="${GRADE_REGRESSION_AUTOPILOT_INTERVAL_SECONDS:-600}"
ADAPTIVE_REGRESSION_GUARD_PLIST="$AGENTS_DIR/com.dankingsley.ops.adaptive_regression_guard.plist"
ADAPTIVE_REGRESSION_GUARD_INTERVAL="${ADAPTIVE_REGRESSION_GUARD_INTERVAL_SECONDS:-300}"
SECTION_GRADE_AUTOPILOT_PLIST="$AGENTS_DIR/com.dankingsley.ops.section_grade_autopilot.plist"
SECTION_GRADE_AUTOPILOT_INTERVAL="${SECTION_GRADE_AUTOPILOT_INTERVAL_SECONDS:-600}"
CHROME_HEADLESS_GUARD_PLIST="$AGENTS_DIR/com.dankingsley.ops.chrome_headless_guard.plist"
CHROME_HEADLESS_GUARD_INTERVAL="${CHROME_HEADLESS_GUARD_INTERVAL_SECONDS:-300}"
SYSTEM_SUMMARY_AUTOPILOT_PLIST="$AGENTS_DIR/com.dankingsley.ops.system_summary_autopilot.plist"
SYSTEM_SUMMARY_AUTOPILOT_INTERVAL="${SYSTEM_SUMMARY_AUTOPILOT_INTERVAL_SECONDS:-1800}"
CREATIVE_COTENANT_GUARD_PLIST="$AGENTS_DIR/com.dankingsley.ops.creative_cotenant_guard.plist"
CREATIVE_COTENANT_GUARD_INTERVAL="${CREATIVE_COTENANT_GUARD_INTERVAL_SECONDS:-20}"
SWAP_PRESSURE_GOVERNOR_PLIST="$AGENTS_DIR/com.dankingsley.ops.swap_pressure_governor.plist"
SWAP_PRESSURE_GOVERNOR_INTERVAL="${SWAP_PRESSURE_GOVERNOR_INTERVAL_SECONDS:-60}"
RUNTIME_SMOOTH_MODE_PLIST="$AGENTS_DIR/com.dankingsley.ops.runtime_smooth_mode.plist"
RUNTIME_SMOOTH_MODE_INTERVAL="${RUNTIME_SMOOTH_MODE_INTERVAL_SECONDS:-20}"
SOAK_SELF_HEAL_PLIST="$AGENTS_DIR/com.dankingsley.ops.soak_self_healing.plist"
SOAK_SELF_HEAL_INTERVAL="${SOAK_SELF_HEAL_INTERVAL_SECONDS:-900}"
SOAK_RELIABILITY_SENTINEL_PLIST="$AGENTS_DIR/com.dankingsley.ops.soak_reliability_sentinel.plist"
SOAK_RELIABILITY_SENTINEL_INTERVAL="${SOAK_RELIABILITY_SENTINEL_INTERVAL_SECONDS:-300}"
PRODUCTION_RESILIENCE_CONTROL_PLIST="$AGENTS_DIR/com.dankingsley.ops.production_resilience_control.plist"
PRODUCTION_RESILIENCE_CONTROL_INTERVAL="${PRODUCTION_RESILIENCE_CONTROL_INTERVAL_SECONDS:-300}"
PRODUCTION_HARDENING_WATCH_PLIST="$AGENTS_DIR/com.dankingsley.ops.production_hardening_watch.plist"
PRODUCTION_HARDENING_WATCH_INTERVAL="${PRODUCTION_HARDENING_WATCH_INTERVAL_SECONDS:-300}"

cat > "$WATCHDOG_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.watchdog</string>
$(scheduled_program_arguments watchdog "governance/health/process_watchdog_latest.json" "$WATCHDOG_INTERVAL" 120 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py")
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>Nice</key><integer>0</integer>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$WATCHDOG_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_watchdog.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_watchdog.err.log</string>
</dict></plist>
PLIST

cat > "$REPORT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.daily_report</string>
$(scheduled_program_arguments daily_report "exports/reports/daily_ops_report_latest.json" 86400 1800 "$PY" "$PROJECT_ROOT/scripts/ops/daily_ops_report.py")
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>21</integer><key>Minute</key><integer>10</integer></dict>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_daily_report.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_daily_report.err.log</string>
</dict></plist>
PLIST

cat > "$CANARY_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.canary_tuner</string>
$(scheduled_program_arguments canary_tuner "governance/health/canary_auto_tuner_latest.json" 1800 300 "$PY" "$PROJECT_ROOT/scripts/ops/canary_auto_tuner.py")
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>1800</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_canary_tuner.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_canary_tuner.err.log</string>
</dict></plist>
PLIST

cat > "$SQL_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.sql_link_writer</string>
$(scheduled_program_arguments sql_link_writer "governance/health/sql_link_service_latest.json" "$SQL_LINK_WRITER_INTERVAL" 900 /bin/zsh "$SQL_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$SQL_LINK_WRITER_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_sql_link_writer.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_sql_link_writer.err.log</string>
</dict></plist>
PLIST

cat > "$PROMO_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.promotion_pipeline</string>
$(scheduled_program_arguments promotion_pipeline "governance/walk_forward/promotion_pipeline_latest.json" 86400 1800 "$PY" "$PROJECT_ROOT/scripts/ops/promotion_pipeline.py")
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>21</integer><key>Minute</key><integer>0</integer></dict>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_promotion_pipeline.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_promotion_pipeline.err.log</string>
</dict></plist>
PLIST

cat > "$MARKET_CORR_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.market_crypto_correlation</string>
$(scheduled_program_arguments market_crypto_correlation "governance/health/market_crypto_correlation_sync_latest.json" "$MARKET_CORR_INTERVAL" 240 /bin/zsh "$MARKET_CORR_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$MARKET_CORR_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_market_crypto_correlation.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_market_crypto_correlation.err.log</string>
</dict></plist>
PLIST

cat > "$FX_MARKET_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.fx_market_context</string>
$(scheduled_program_arguments fx_market_context "governance/health/fx_market_context_sync_latest.json" "$FX_MARKET_INTERVAL" 240 /bin/zsh "$FX_MARKET_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$FX_MARKET_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_fx_market_context.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_fx_market_context.err.log</string>
</dict></plist>
PLIST

cat > "$OPTIONS_FLOW_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.options_flow_context</string>
$(scheduled_program_arguments options_flow_context "governance/health/options_flow_context_sync_latest.json" "$OPTIONS_FLOW_INTERVAL" 900 /bin/zsh "$OPTIONS_FLOW_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$OPTIONS_FLOW_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_options_flow_context.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_options_flow_context.err.log</string>
</dict></plist>
PLIST

cat > "$OFFICIAL_MACRO_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.official_macro_context</string>
$(scheduled_program_arguments official_macro_context "governance/health/official_macro_context_sync_latest.json" "$OFFICIAL_MACRO_INTERVAL" 900 /bin/zsh "$OFFICIAL_MACRO_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$OFFICIAL_MACRO_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_official_macro_context.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_official_macro_context.err.log</string>
</dict></plist>
PLIST

cat > "$SCHWAB_EDUCATION_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.schwab_education_context</string>
$(scheduled_program_arguments schwab_education_context "governance/health/schwab_education_context_sync_latest.json" "$SCHWAB_EDUCATION_INTERVAL" 900 /bin/zsh "$SCHWAB_EDUCATION_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$SCHWAB_EDUCATION_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_schwab_education_context.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_schwab_education_context.err.log</string>
</dict></plist>
PLIST

cat > "$ONE_NUMBERS_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.one_numbers_refresh</string>
$(scheduled_program_arguments one_numbers_refresh "governance/health/one_numbers_latest.json" "$ONE_NUMBERS_INTERVAL" 900 /bin/zsh "$ONE_NUMBERS_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$ONE_NUMBERS_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_one_numbers_refresh.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_one_numbers_refresh.err.log</string>
</dict></plist>
PLIST

cat > "$ONE_NUMBERS_REGRESSION_GUARD_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.one_numbers_regression_guard</string>
$(scheduled_program_arguments one_numbers_regression_guard "governance/health/one_numbers_regression_guard_latest.json" "$ONE_NUMBERS_REGRESSION_GUARD_INTERVAL" 900 /bin/zsh "$ONE_NUMBERS_REGRESSION_GUARD_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$ONE_NUMBERS_REGRESSION_GUARD_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_one_numbers_regression_guard.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_one_numbers_regression_guard.err.log</string>
</dict></plist>
PLIST

cat > "$COMMAND_VALIDITY_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.command_validity</string>
$(scheduled_program_arguments command_validity "governance/health/command_validity_latest.json" "$COMMAND_VALIDITY_INTERVAL" 300 /bin/zsh "$COMMAND_VALIDITY_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$COMMAND_VALIDITY_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_command_validity.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_command_validity.err.log</string>
</dict></plist>
PLIST

cat > "$SYSTEM_DRIFT_GUARD_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.system_drift_guard</string>
$(scheduled_program_arguments system_drift_guard "governance/health/system_drift_guard_latest.json" "$SYSTEM_DRIFT_GUARD_INTERVAL" 300 /bin/zsh "$SYSTEM_DRIFT_GUARD_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$SYSTEM_DRIFT_GUARD_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_system_drift_guard.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_system_drift_guard.err.log</string>
</dict></plist>
PLIST

cat > "$SYSTEM_DRIFT_AUTOPILOT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.system_drift_autopilot</string>
$(scheduled_program_arguments system_drift_autopilot "governance/health/system_drift_autopilot_latest.json" "$SYSTEM_DRIFT_AUTOPILOT_INTERVAL" 900 /bin/zsh "$SYSTEM_DRIFT_AUTOPILOT_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$SYSTEM_DRIFT_AUTOPILOT_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_system_drift_autopilot.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_system_drift_autopilot.err.log</string>
</dict></plist>
PLIST

cat > "$SYSTEM_CELL_FEDERATION_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.system_cell_federation</string>
$(scheduled_program_arguments system_cell_federation "governance/health/system_cell_federation_latest.json" "$SYSTEM_CELL_FEDERATION_INTERVAL" 900 /bin/zsh "$SYSTEM_CELL_FEDERATION_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict>
    <key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string>
    <key>BOT_PROTECTED_VOLUME_DENYLIST</key><string>/Volumes/VIDEO</string>
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$SYSTEM_CELL_FEDERATION_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_system_cell_federation.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_system_cell_federation.err.log</string>
</dict></plist>
PLIST

cat > "$MAINT_STRATEGY_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.maintenance_strategy_reloader</string>
$(scheduled_program_arguments maintenance_strategy_reloader "governance/health/maintenance_strategy_reloader_latest.json" 300 300 "$PY" "$PROJECT_ROOT/scripts/ops/maintenance_strategy_reloader.py")
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>300</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_maintenance_strategy_reloader.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_maintenance_strategy_reloader.err.log</string>
</dict></plist>
PLIST

cat > "$ADAPTIVE_OPS_POLICY_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.adaptive_ops_recovery_policy</string>
$(scheduled_program_arguments adaptive_ops_recovery_policy "governance/health/adaptive_ops_recovery_policy_latest.json" "$ADAPTIVE_OPS_POLICY_INTERVAL" 240 "$PY" "$PROJECT_ROOT/scripts/ops/adaptive_ops_recovery_policy.py" --apply --json)
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$ADAPTIVE_OPS_POLICY_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_adaptive_ops_recovery_policy.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_adaptive_ops_recovery_policy.err.log</string>
</dict></plist>
PLIST

cat > "$RETENTION_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.data_retention</string>
$(scheduled_program_arguments data_retention "governance/health/data_retention_latest.json" "$RETENTION_INTERVAL" 1800 /bin/zsh "$RETENTION_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$RETENTION_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_data_retention.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_data_retention.err.log</string>
</dict></plist>
PLIST

cat > "$BACKLOG_RETRY_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.external_backlog_retry</string>
$(scheduled_program_arguments external_backlog_retry "governance/health/external_backlog_retry_bot_latest.json" "$BACKLOG_RETRY_INTERVAL" 1800 /bin/zsh "$BACKLOG_RETRY_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$BACKLOG_RETRY_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_external_backlog_retry.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_external_backlog_retry.err.log</string>
</dict></plist>
PLIST

cat > "$STORAGE_BACKPRESSURE_AUTOPILOT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.storage_backpressure_autopilot</string>
$(scheduled_program_arguments storage_backpressure_autopilot "governance/health/storage_backpressure_autopilot_latest.json" "$STORAGE_BACKPRESSURE_AUTOPILOT_INTERVAL" 600 /bin/zsh "$STORAGE_BACKPRESSURE_AUTOPILOT_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$STORAGE_BACKPRESSURE_AUTOPILOT_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_storage_backpressure_autopilot.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_storage_backpressure_autopilot.err.log</string>
</dict></plist>
PLIST

cat > "$TRAINING_DRAIN_AUTOPILOT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.training_drain_autopilot</string>
$(scheduled_program_arguments training_drain_autopilot "governance/health/training_drain_autopilot_latest.json" "$TRAINING_DRAIN_AUTOPILOT_INTERVAL" 1800 /bin/zsh "$TRAINING_DRAIN_AUTOPILOT_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict>
    <key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string>
    <key>BOT_PROTECTED_VOLUME_DENYLIST</key><string>/Volumes/VIDEO</string>
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$TRAINING_DRAIN_AUTOPILOT_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_training_drain_autopilot.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_training_drain_autopilot.err.log</string>
</dict></plist>
PLIST

cat > "$STORAGE_PRESSURE_CLEARANCE_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.storage_pressure_clearance</string>
$(scheduled_program_arguments storage_pressure_clearance "governance/health/storage_pressure_clearance_latest.json" "$STORAGE_PRESSURE_CLEARANCE_INTERVAL" 600 /bin/zsh "$STORAGE_PRESSURE_CLEARANCE_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$STORAGE_PRESSURE_CLEARANCE_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_storage_pressure_clearance.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_storage_pressure_clearance.err.log</string>
</dict></plist>
PLIST

cat > "$STORAGE_RECONNECT_INFRABOT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.storage_reconnect_infrabot</string>
$(scheduled_program_arguments storage_reconnect_infrabot "governance/health/storage_reconnect_infrabot_latest.json" "$STORAGE_RECONNECT_INFRABOT_INTERVAL" 900 /bin/zsh "$STORAGE_RECONNECT_INFRABOT_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$STORAGE_RECONNECT_INFRABOT_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_storage_reconnect_infrabot.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_storage_reconnect_infrabot.err.log</string>
</dict></plist>
PLIST

cat > "$INFRA_AUTOFIX_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.infrastructure_autofix</string>
$(scheduled_program_arguments infrastructure_autofix "governance/health/infrastructure_autofix_bot_latest.json" "$INFRA_AUTOFIX_INTERVAL" 900 /bin/zsh "$INFRA_AUTOFIX_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$INFRA_AUTOFIX_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_infrastructure_autofix.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_infrastructure_autofix.err.log</string>
</dict></plist>
PLIST

cat > "$BOT_QUALITY_AUTOPILOT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.bot_quality_autopilot</string>
$(scheduled_program_arguments bot_quality_autopilot "governance/health/bot_quality_autopilot_latest.json" "$BOT_QUALITY_AUTOPILOT_INTERVAL" 1800 /bin/zsh "$BOT_QUALITY_AUTOPILOT_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$BOT_QUALITY_AUTOPILOT_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_bot_quality_autopilot.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_bot_quality_autopilot.err.log</string>
</dict></plist>
PLIST

cat > "$MASTER_INFRA_SUPERVISOR_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.master_infrastructure_supervisor</string>
$(scheduled_program_arguments master_infrastructure_supervisor "governance/health/master_infrastructure_supervisor_latest.json" "$MASTER_INFRA_SUPERVISOR_INTERVAL" 900 /bin/zsh "$MASTER_INFRA_SUPERVISOR_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$MASTER_INFRA_SUPERVISOR_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_master_infrastructure_supervisor.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_master_infrastructure_supervisor.err.log</string>
</dict></plist>
PLIST

cat > "$SCHWAB_AUTH_SUPERVISOR_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.schwab_auth_supervisor</string>
$(scheduled_program_arguments schwab_auth_supervisor "governance/health/schwab_auth_supervisor_latest.json" "$SCHWAB_AUTH_SUPERVISOR_INTERVAL" 300 /bin/zsh "$SCHWAB_AUTH_SUPERVISOR_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$SCHWAB_AUTH_SUPERVISOR_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_schwab_auth_supervisor.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_schwab_auth_supervisor.err.log</string>
</dict></plist>
PLIST

cat > "$STORAGE_STANDBY_PRUNE_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.storage_standby_prune</string>
$(scheduled_program_arguments storage_standby_prune "governance/health/storage_standby_prune_latest.json" "$STORAGE_STANDBY_PRUNE_INTERVAL" 900 /bin/zsh "$STORAGE_STANDBY_PRUNE_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$STORAGE_STANDBY_PRUNE_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_storage_standby_prune.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_storage_standby_prune.err.log</string>
</dict></plist>
PLIST

cat > "$LOCAL_SQL_SHARD_STANDBY_PRUNE_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.local_sql_shard_standby_prune</string>
$(scheduled_program_arguments local_sql_shard_standby_prune "governance/health/local_sql_shard_standby_prune_latest.json" "$LOCAL_SQL_SHARD_STANDBY_PRUNE_INTERVAL" 900 /bin/zsh "$LOCAL_SQL_SHARD_STANDBY_PRUNE_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$LOCAL_SQL_SHARD_STANDBY_PRUNE_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_local_sql_shard_standby_prune.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_local_sql_shard_standby_prune.err.log</string>
</dict></plist>
PLIST

cat > "$GRADE_REGRESSION_AUTOPILOT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.grade_regression_autopilot</string>
$(scheduled_program_arguments grade_regression_autopilot "governance/health/grade_regression_autopilot_latest.json" "$GRADE_REGRESSION_AUTOPILOT_INTERVAL" 900 /bin/zsh "$GRADE_REGRESSION_AUTOPILOT_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$GRADE_REGRESSION_AUTOPILOT_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_grade_regression_autopilot.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_grade_regression_autopilot.err.log</string>
</dict></plist>
PLIST

cat > "$ADAPTIVE_REGRESSION_GUARD_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.adaptive_regression_guard</string>
$(scheduled_program_arguments adaptive_regression_guard "governance/health/adaptive_regression_guard_latest.json" "$ADAPTIVE_REGRESSION_GUARD_INTERVAL" 900 /bin/zsh "$ADAPTIVE_REGRESSION_GUARD_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$ADAPTIVE_REGRESSION_GUARD_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_adaptive_regression_guard.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_adaptive_regression_guard.err.log</string>
</dict></plist>
PLIST

cat > "$SECTION_GRADE_AUTOPILOT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.section_grade_autopilot</string>
$(scheduled_program_arguments section_grade_autopilot "governance/health/section_grade_autopilot_latest.json" "$SECTION_GRADE_AUTOPILOT_INTERVAL" 900 /bin/zsh "$SECTION_GRADE_AUTOPILOT_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$SECTION_GRADE_AUTOPILOT_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_section_grade_autopilot.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_section_grade_autopilot.err.log</string>
</dict></plist>
PLIST

cat > "$CHROME_HEADLESS_GUARD_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.chrome_headless_guard</string>
$(scheduled_program_arguments chrome_headless_guard "governance/health/chrome_headless_guard_latest.json" "$CHROME_HEADLESS_GUARD_INTERVAL" 300 /bin/zsh "$CHROME_HEADLESS_GUARD_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$CHROME_HEADLESS_GUARD_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_chrome_headless_guard.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_chrome_headless_guard.err.log</string>
</dict></plist>
PLIST

cat > "$SYSTEM_SUMMARY_AUTOPILOT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.system_summary_autopilot</string>
$(scheduled_program_arguments system_summary_autopilot "governance/health/system_summary_autopilot_latest.json" "$SYSTEM_SUMMARY_AUTOPILOT_INTERVAL" 900 /bin/zsh "$SYSTEM_SUMMARY_AUTOPILOT_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$SYSTEM_SUMMARY_AUTOPILOT_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_system_summary_autopilot.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_system_summary_autopilot.err.log</string>
</dict></plist>
PLIST

cat > "$SWAP_PRESSURE_GOVERNOR_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.swap_pressure_governor</string>
$(scheduled_program_arguments swap_pressure_governor "governance/health/swap_pressure_governor_latest.json" "$SWAP_PRESSURE_GOVERNOR_INTERVAL" 240 /bin/zsh "$SWAP_PRESSURE_GOVERNOR_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$SWAP_PRESSURE_GOVERNOR_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_swap_pressure_governor.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_swap_pressure_governor.err.log</string>
</dict></plist>
PLIST

cat > "$RUNTIME_SMOOTH_MODE_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.runtime_smooth_mode</string>
$(scheduled_program_arguments runtime_smooth_mode "governance/health/runtime_smooth_mode_latest.json" "$RUNTIME_SMOOTH_MODE_INTERVAL" 60 /bin/zsh "$RUNTIME_SMOOTH_MODE_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict>
    <key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string>
    <key>BOT_PROTECTED_VOLUME_DENYLIST</key><string>/Volumes/VIDEO</string>
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$RUNTIME_SMOOTH_MODE_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_runtime_smooth_mode.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_runtime_smooth_mode.err.log</string>
</dict></plist>
PLIST

cat > "$SOAK_SELF_HEAL_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.soak_self_healing</string>
$(scheduled_program_arguments soak_self_healing "governance/health/soak_self_healing_control_latest.json" "$SOAK_SELF_HEAL_INTERVAL" 2400 /bin/zsh "$SOAK_SELF_HEAL_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict>
    <key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string>
    <key>MARKET_DATA_ONLY</key><string>1</string>
    <key>ALLOW_ORDER_EXECUTION</key><string>0</string>
    <key>BOT_LIVE_MONEY_LOCKED_DURING_SOAK</key><string>1</string>
    <key>BOT_UNATTENDED_SOAK_ACTIVE</key><string>1</string>
    <key>READINESS_EVIDENCE_REFRESH_PROFILE</key><string>accrual</string>
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$SOAK_SELF_HEAL_INTERVAL</integer>
  <key>WatchPaths</key><array>
    <string>$PROJECT_ROOT/governance/runtime/soak_self_healing.trigger</string>
  </array>
  <key>ProcessType</key><string>Background</string>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_soak_self_healing.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_soak_self_healing.err.log</string>
</dict></plist>
PLIST

cat > "$SOAK_RELIABILITY_SENTINEL_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.soak_reliability_sentinel</string>
$(scheduled_program_arguments soak_reliability_sentinel "governance/health/soak_reliability_sentinel_latest.json" "$SOAK_RELIABILITY_SENTINEL_INTERVAL" 300 "$PY" "$SOAK_RELIABILITY_SENTINEL_SCRIPT" --apply --json)
  <key>EnvironmentVariables</key><dict>
    <key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string>
    <key>MARKET_DATA_ONLY</key><string>1</string>
    <key>ALLOW_ORDER_EXECUTION</key><string>0</string>
    <key>BOT_LIVE_MONEY_LOCKED_DURING_SOAK</key><string>1</string>
    <key>BOT_UNATTENDED_SOAK_ACTIVE</key><string>1</string>
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$SOAK_RELIABILITY_SENTINEL_INTERVAL</integer>
  <key>ProcessType</key><string>Background</string>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_soak_reliability_sentinel.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_soak_reliability_sentinel.err.log</string>
</dict></plist>
PLIST

cat > "$PRODUCTION_RESILIENCE_CONTROL_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.production_resilience_control</string>
$(scheduled_program_arguments production_resilience_control "governance/health/production_resilience_control_latest.json" "$PRODUCTION_RESILIENCE_CONTROL_INTERVAL" 300 "$PY" "$PRODUCTION_RESILIENCE_CONTROL_SCRIPT" --json)
  <key>EnvironmentVariables</key><dict>
    <key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string>
    <key>MARKET_DATA_ONLY</key><string>1</string>
    <key>ALLOW_ORDER_EXECUTION</key><string>0</string>
    <key>BOT_LIVE_MONEY_LOCKED_DURING_SOAK</key><string>1</string>
    <key>BOT_UNATTENDED_SOAK_ACTIVE</key><string>1</string>
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$PRODUCTION_RESILIENCE_CONTROL_INTERVAL</integer>
  <key>ProcessType</key><string>Background</string>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_production_resilience_control.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_production_resilience_control.err.log</string>
</dict></plist>
PLIST

cat > "$PRODUCTION_HARDENING_WATCH_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.production_hardening_watch</string>
$(scheduled_program_arguments production_hardening_watch "governance/health/production_hardening_watch_latest.json" "$PRODUCTION_HARDENING_WATCH_INTERVAL" 1800 /bin/zsh "$PRODUCTION_HARDENING_WATCH_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict>
    <key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string>
    <key>MARKET_DATA_ONLY</key><string>1</string>
    <key>ALLOW_ORDER_EXECUTION</key><string>0</string>
    <key>BOT_LIVE_MONEY_LOCKED_DURING_SOAK</key><string>1</string>
    <key>BOT_UNATTENDED_SOAK_ACTIVE</key><string>1</string>
    <key>READINESS_EVIDENCE_REFRESH_PROFILE</key><string>accrual</string>
    <key>PRODUCTION_PILLAR_REFRESH_ENABLED</key><string>${PRODUCTION_PILLAR_REFRESH_ENABLED:-1}</string>
    <key>PRODUCTION_PILLAR_REFRESH_COOLDOWN_MINUTES</key><string>${PRODUCTION_PILLAR_REFRESH_COOLDOWN_MINUTES:-45}</string>
    <key>PRODUCTION_PILLAR_REFRESH_STEP_TIMEOUT_SECONDS</key><string>${PRODUCTION_PILLAR_REFRESH_STEP_TIMEOUT_SECONDS:-300}</string>
    <key>PRODUCTION_HARDENING_WATCH_EXECUTE_SAFE_REPAIRS</key><string>${PRODUCTION_HARDENING_WATCH_EXECUTE_SAFE_REPAIRS:-0}</string>
    <key>PRODUCTION_HARDENING_WATCH_EXECUTE_ON_WATCH</key><string>${PRODUCTION_HARDENING_WATCH_EXECUTE_ON_WATCH:-0}</string>
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$PRODUCTION_HARDENING_WATCH_INTERVAL</integer>
  <key>ProcessType</key><string>Background</string>
  <key>LowPriorityIO</key><true/>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_production_hardening_watch.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_production_hardening_watch.err.log</string>
</dict></plist>
PLIST

cat > "$CREATIVE_COTENANT_GUARD_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.creative_cotenant_guard</string>
$(scheduled_program_arguments creative_cotenant_guard "governance/health/creative_cotenant_guard_latest.json" "$CREATIVE_COTENANT_GUARD_INTERVAL" 60 /bin/zsh "$CREATIVE_COTENANT_GUARD_RUN_SCRIPT")
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$CREATIVE_COTENANT_GUARD_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_creative_cotenant_guard.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_creative_cotenant_guard.err.log</string>
</dict></plist>
PLIST

install_job() {
  local label="$1"
  local plist="$2"
  launchctl bootout "gui/$UID_NUM" "$plist" >/dev/null 2>&1 || true
  launchctl enable "gui/$UID_NUM/$label" || true
  launchctl bootstrap "gui/$UID_NUM" "$plist"
  launchctl kickstart -k "gui/$UID_NUM/$label" || true
  echo "Installed: $plist"
}

remove_job() {
  local label="$1"
  local plist="$2"
  launchctl bootout "gui/$UID_NUM" "$plist" >/dev/null 2>&1 || true
  launchctl disable "gui/$UID_NUM/$label" >/dev/null 2>&1 || true
  rm -f "$plist"
  echo "Removed legacy: $plist"
}

install_job "com.dankingsley.ops.watchdog" "$WATCHDOG_PLIST"
install_job "com.dankingsley.ops.daily_report" "$REPORT_PLIST"
install_job "com.dankingsley.ops.canary_tuner" "$CANARY_PLIST"
install_job "com.dankingsley.ops.sql_link_writer" "$SQL_PLIST"
install_job "com.dankingsley.ops.promotion_pipeline" "$PROMO_PLIST"
install_job "com.dankingsley.ops.market_crypto_correlation" "$MARKET_CORR_PLIST"
install_job "com.dankingsley.ops.fx_market_context" "$FX_MARKET_PLIST"
install_job "com.dankingsley.ops.options_flow_context" "$OPTIONS_FLOW_PLIST"
install_job "com.dankingsley.ops.official_macro_context" "$OFFICIAL_MACRO_PLIST"
install_job "com.dankingsley.ops.schwab_education_context" "$SCHWAB_EDUCATION_PLIST"
install_job "com.dankingsley.ops.one_numbers_refresh" "$ONE_NUMBERS_PLIST"
install_job "com.dankingsley.ops.one_numbers_regression_guard" "$ONE_NUMBERS_REGRESSION_GUARD_PLIST"
install_job "com.dankingsley.ops.command_validity" "$COMMAND_VALIDITY_PLIST"
install_job "com.dankingsley.ops.system_drift_guard" "$SYSTEM_DRIFT_GUARD_PLIST"
install_job "com.dankingsley.ops.system_drift_autopilot" "$SYSTEM_DRIFT_AUTOPILOT_PLIST"
install_job "com.dankingsley.ops.system_cell_federation" "$SYSTEM_CELL_FEDERATION_PLIST"
install_job "com.dankingsley.ops.maintenance_strategy_reloader" "$MAINT_STRATEGY_PLIST"
install_job "com.dankingsley.ops.adaptive_ops_recovery_policy" "$ADAPTIVE_OPS_POLICY_PLIST"
install_job "com.dankingsley.ops.data_retention" "$RETENTION_PLIST"
install_job "com.dankingsley.ops.external_backlog_retry" "$BACKLOG_RETRY_PLIST"
install_job "com.dankingsley.ops.storage_backpressure_autopilot" "$STORAGE_BACKPRESSURE_AUTOPILOT_PLIST"
install_job "com.dankingsley.ops.training_drain_autopilot" "$TRAINING_DRAIN_AUTOPILOT_PLIST"
install_job "com.dankingsley.ops.storage_pressure_clearance" "$STORAGE_PRESSURE_CLEARANCE_PLIST"
install_job "com.dankingsley.ops.local_sql_shard_standby_prune" "$LOCAL_SQL_SHARD_STANDBY_PRUNE_PLIST"
install_job "com.dankingsley.ops.storage_reconnect_infrabot" "$STORAGE_RECONNECT_INFRABOT_PLIST"
remove_job "com.dankingsley.ops.writer_cycle_coordinator" "$WRITER_COORDINATOR_PLIST"
remove_job "com.dankingsley.ops.retention_debt_sheriff" "$RETENTION_SHERIFF_PLIST"
remove_job "com.dankingsley.ops.backpressure_slo_bot" "$BACKPRESSURE_SLO_PLIST"
install_job "com.dankingsley.ops.infrastructure_autofix" "$INFRA_AUTOFIX_PLIST"
install_job "com.dankingsley.ops.master_infrastructure_supervisor" "$MASTER_INFRA_SUPERVISOR_PLIST"
install_job "com.dankingsley.ops.schwab_auth_supervisor" "$SCHWAB_AUTH_SUPERVISOR_PLIST"
install_job "com.dankingsley.ops.bot_quality_autopilot" "$BOT_QUALITY_AUTOPILOT_PLIST"
install_job "com.dankingsley.ops.storage_standby_prune" "$STORAGE_STANDBY_PRUNE_PLIST"
install_job "com.dankingsley.ops.grade_regression_autopilot" "$GRADE_REGRESSION_AUTOPILOT_PLIST"
install_job "com.dankingsley.ops.adaptive_regression_guard" "$ADAPTIVE_REGRESSION_GUARD_PLIST"
install_job "com.dankingsley.ops.section_grade_autopilot" "$SECTION_GRADE_AUTOPILOT_PLIST"
install_job "com.dankingsley.ops.chrome_headless_guard" "$CHROME_HEADLESS_GUARD_PLIST"
install_job "com.dankingsley.ops.system_summary_autopilot" "$SYSTEM_SUMMARY_AUTOPILOT_PLIST"
install_job "com.dankingsley.ops.swap_pressure_governor" "$SWAP_PRESSURE_GOVERNOR_PLIST"
install_job "com.dankingsley.ops.runtime_smooth_mode" "$RUNTIME_SMOOTH_MODE_PLIST"
install_job "com.dankingsley.ops.soak_self_healing" "$SOAK_SELF_HEAL_PLIST"
install_job "com.dankingsley.ops.soak_reliability_sentinel" "$SOAK_RELIABILITY_SENTINEL_PLIST"
install_job "com.dankingsley.ops.production_resilience_control" "$PRODUCTION_RESILIENCE_CONTROL_PLIST"
install_job "com.dankingsley.ops.production_hardening_watch" "$PRODUCTION_HARDENING_WATCH_PLIST"
install_job "com.dankingsley.ops.creative_cotenant_guard" "$CREATIVE_COTENANT_GUARD_PLIST"

echo "Ops automations installed."
