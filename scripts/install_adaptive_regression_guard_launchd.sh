#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_adaptive_regression_guard_launchd.sh"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.ops.adaptive_regression_guard.plist"
LABEL="com.dankingsley.ops.adaptive_regression_guard"
UID_NUM="$(id -u)"
LOG_DIR="${BOT_OPS_LAUNCHD_LOG_DIR:-$HOME/Library/Logs/schwab_trading_bot}"
OUT_LOG="$LOG_DIR/ops_adaptive_regression_guard.out.log"
ERR_LOG="$LOG_DIR/ops_adaptive_regression_guard.err.log"
INTERVAL_SECONDS="${ADAPTIVE_REGRESSION_GUARD_INTERVAL_SECONDS:-300}"
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
    <key>ADAPTIVE_REGRESSION_PERSISTENCE_THRESHOLD</key><string>${ADAPTIVE_REGRESSION_PERSISTENCE_THRESHOLD:-3}</string>
    <key>ADAPTIVE_REGRESSION_BLOCKED_ESCALATION_THRESHOLD</key><string>${ADAPTIVE_REGRESSION_BLOCKED_ESCALATION_THRESHOLD:-2}</string>
    <key>ADAPTIVE_REGRESSION_MAX_ARTIFACT_AGE_MINUTES</key><string>${ADAPTIVE_REGRESSION_MAX_ARTIFACT_AGE_MINUTES:-60}</string>
  </dict>
  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key>
  <true/>
  <key>StartInterval</key>
  <integer>$INTERVAL_SECONDS</integer>
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
echo "Logs: $OUT_LOG and $ERR_LOG"
