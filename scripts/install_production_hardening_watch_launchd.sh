#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_production_hardening_watch_launchd.sh"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.ops.production_hardening_watch.plist"
LABEL="com.dankingsley.ops.production_hardening_watch"
UID_NUM="$(id -u)"
LOG_DIR="${BOT_OPS_LAUNCHD_LOG_DIR:-$HOME/Library/Logs/schwab_trading_bot}"
OUT_LOG="$LOG_DIR/ops_production_hardening_watch.out.log"
ERR_LOG="$LOG_DIR/ops_production_hardening_watch.err.log"
INTERVAL_SECONDS="${PRODUCTION_HARDENING_WATCH_INTERVAL_SECONDS:-300}"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
chmod +x "$RUN_SCRIPT"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$RUN_SCRIPT</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
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
    <key>PRODUCTION_HARDENING_WATCH_MAX_EXECUTE_ACTIONS</key><string>${PRODUCTION_HARDENING_WATCH_MAX_EXECUTE_ACTIONS:-2}</string>
    <key>PRODUCTION_HARDENING_WATCH_COMMAND_TIMEOUT_SECONDS</key><string>${PRODUCTION_HARDENING_WATCH_COMMAND_TIMEOUT_SECONDS:-240}</string>
  </dict>
  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key>
  <true/>
  <key>StartInterval</key>
  <integer>$INTERVAL_SECONDS</integer>
  <key>ProcessType</key><string>Background</string>
  <key>LowPriorityIO</key><true/>
  <key>StandardOutPath</key><string>$OUT_LOG</string>
  <key>StandardErrorPath</key><string>$ERR_LOG</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$UID_NUM" "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_PATH"
launchctl enable "gui/$UID_NUM/$LABEL" || true
launchctl kickstart -k "gui/$UID_NUM/$LABEL" || true

echo "Installed and loaded: $PLIST_PATH"
echo "Label: $LABEL"
echo "Schedule: every $INTERVAL_SECONDS seconds"
echo "Safe repairs: ${PRODUCTION_HARDENING_WATCH_EXECUTE_SAFE_REPAIRS:-0}"
echo "Logs: $OUT_LOG and $ERR_LOG"
