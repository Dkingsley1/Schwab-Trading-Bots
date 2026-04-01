#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$PROJECT_ROOT/.venv312/bin/python"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"
SQL_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_sql_link_writer_launchd.sh"
FX_MARKET_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_fx_market_context_launchd.sh"
OFFICIAL_MACRO_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_official_macro_context_launchd.sh"
MARKET_CORR_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_market_crypto_correlation_launchd.sh"
RETENTION_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_data_retention_launchd.sh"
ONE_NUMBERS_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_one_numbers_refresh_launchd.sh"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="/tmp"
UID_NUM="$(id -u)"
mkdir -p "$AGENTS_DIR"

chmod +x "$SQL_RUN_SCRIPT"
chmod +x "$FX_MARKET_RUN_SCRIPT"
chmod +x "$OFFICIAL_MACRO_RUN_SCRIPT"
chmod +x "$MARKET_CORR_RUN_SCRIPT"
chmod +x "$RETENTION_RUN_SCRIPT"
chmod +x "$ONE_NUMBERS_RUN_SCRIPT"

WATCHDOG_PLIST="$AGENTS_DIR/com.dankingsley.ops.watchdog.plist"
REPORT_PLIST="$AGENTS_DIR/com.dankingsley.ops.daily_report.plist"
CANARY_PLIST="$AGENTS_DIR/com.dankingsley.ops.canary_tuner.plist"
SQL_PLIST="$AGENTS_DIR/com.dankingsley.ops.sql_link_writer.plist"
PROMO_PLIST="$AGENTS_DIR/com.dankingsley.ops.promotion_pipeline.plist"
MARKET_CORR_PLIST="$AGENTS_DIR/com.dankingsley.ops.market_crypto_correlation.plist"
MARKET_CORR_INTERVAL="${MARKET_CRYPTO_CORRELATION_REFRESH_INTERVAL_SECONDS:-300}"
FX_MARKET_PLIST="$AGENTS_DIR/com.dankingsley.ops.fx_market_context.plist"
FX_MARKET_INTERVAL="${FX_MARKET_CONTEXT_REFRESH_INTERVAL_SECONDS:-900}"
OFFICIAL_MACRO_PLIST="$AGENTS_DIR/com.dankingsley.ops.official_macro_context.plist"
OFFICIAL_MACRO_INTERVAL="${OFFICIAL_MACRO_CONTEXT_REFRESH_INTERVAL_SECONDS:-21600}"
ONE_NUMBERS_PLIST="$AGENTS_DIR/com.dankingsley.ops.one_numbers_refresh.plist"
ONE_NUMBERS_INTERVAL="${ONE_NUMBERS_REFRESH_LAUNCHD_INTERVAL_SECONDS:-180}"
WATCHDOG_INTERVAL="${OPS_WATCHDOG_LAUNCHD_INTERVAL_SECONDS:-180}"
MAINT_STRATEGY_PLIST="$AGENTS_DIR/com.dankingsley.ops.maintenance_strategy_reloader.plist"
RETENTION_PLIST="$AGENTS_DIR/com.dankingsley.ops.data_retention.plist"
RETENTION_INTERVAL="${RETENTION_REFRESH_INTERVAL_SECONDS:-3600}"

cat > "$WATCHDOG_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.watchdog</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$PROJECT_ROOT/scripts/ops/process_watchdog.py</string></array>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
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
  <key>ProgramArguments</key><array><string>$PY</string><string>$PROJECT_ROOT/scripts/ops/daily_ops_report.py</string></array>
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
  <key>ProgramArguments</key><array><string>$PY</string><string>$PROJECT_ROOT/scripts/ops/canary_auto_tuner.py</string></array>
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
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>$SQL_RUN_SCRIPT</string></array>
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_sql_link_writer.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_sql_link_writer.err.log</string>
</dict></plist>
PLIST

cat > "$PROMO_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.promotion_pipeline</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$PROJECT_ROOT/scripts/ops/promotion_pipeline.py</string></array>
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
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>$MARKET_CORR_RUN_SCRIPT</string></array>
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
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>$FX_MARKET_RUN_SCRIPT</string></array>
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$FX_MARKET_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_fx_market_context.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_fx_market_context.err.log</string>
</dict></plist>
PLIST

cat > "$OFFICIAL_MACRO_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.official_macro_context</string>
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>$OFFICIAL_MACRO_RUN_SCRIPT</string></array>
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$OFFICIAL_MACRO_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_official_macro_context.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_official_macro_context.err.log</string>
</dict></plist>
PLIST

cat > "$ONE_NUMBERS_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.one_numbers_refresh</string>
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>$ONE_NUMBERS_RUN_SCRIPT</string></array>
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$ONE_NUMBERS_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_one_numbers_refresh.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_one_numbers_refresh.err.log</string>
</dict></plist>
PLIST

cat > "$MAINT_STRATEGY_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.maintenance_strategy_reloader</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$PROJECT_ROOT/scripts/ops/maintenance_strategy_reloader.py</string></array>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>300</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_maintenance_strategy_reloader.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_maintenance_strategy_reloader.err.log</string>
</dict></plist>
PLIST

cat > "$RETENTION_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.data_retention</string>
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>$RETENTION_RUN_SCRIPT</string></array>
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$RETENTION_INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_data_retention.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_data_retention.err.log</string>
</dict></plist>
PLIST

install_job() {
  local label="$1"
  local plist="$2"
  launchctl bootout "gui/$UID_NUM" "$plist" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$UID_NUM" "$plist"
  launchctl enable "gui/$UID_NUM/$label" || true
  launchctl kickstart -k "gui/$UID_NUM/$label" || true
  echo "Installed: $plist"
}

install_job "com.dankingsley.ops.watchdog" "$WATCHDOG_PLIST"
install_job "com.dankingsley.ops.daily_report" "$REPORT_PLIST"
install_job "com.dankingsley.ops.canary_tuner" "$CANARY_PLIST"
install_job "com.dankingsley.ops.sql_link_writer" "$SQL_PLIST"
install_job "com.dankingsley.ops.promotion_pipeline" "$PROMO_PLIST"
install_job "com.dankingsley.ops.market_crypto_correlation" "$MARKET_CORR_PLIST"
install_job "com.dankingsley.ops.fx_market_context" "$FX_MARKET_PLIST"
install_job "com.dankingsley.ops.official_macro_context" "$OFFICIAL_MACRO_PLIST"
install_job "com.dankingsley.ops.one_numbers_refresh" "$ONE_NUMBERS_PLIST"
install_job "com.dankingsley.ops.maintenance_strategy_reloader" "$MAINT_STRATEGY_PLIST"
install_job "com.dankingsley.ops.data_retention" "$RETENTION_PLIST"

echo "Ops automations installed."
