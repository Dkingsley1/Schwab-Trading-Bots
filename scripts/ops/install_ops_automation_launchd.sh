#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$PROJECT_ROOT/.venv312/bin/python"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"
SQL_RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_sql_link_writer_launchd.sh"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="/tmp"
UID_NUM="$(id -u)"
mkdir -p "$AGENTS_DIR"

chmod +x "$SQL_RUN_SCRIPT"

WATCHDOG_PLIST="$AGENTS_DIR/com.dankingsley.ops.watchdog.plist"
REPORT_PLIST="$AGENTS_DIR/com.dankingsley.ops.daily_report.plist"
CANARY_PLIST="$AGENTS_DIR/com.dankingsley.ops.canary_tuner.plist"
SQL_PLIST="$AGENTS_DIR/com.dankingsley.ops.sql_link_writer.plist"
PROMO_PLIST="$AGENTS_DIR/com.dankingsley.ops.promotion_pipeline.plist"

cat > "$WATCHDOG_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.watchdog</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$PROJECT_ROOT/scripts/ops/process_watchdog.py</string></array>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>300</integer>
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

echo "Ops automations installed."
