#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/daily_log_refresh.sh"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.daily_log_refresh.plist"
LABEL="com.dankingsley.daily_log_refresh"
UID_NUM="$(id -u)"
LAUNCHD_LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
OUT_LOG="$LAUNCHD_LOG_DIR/daily_log_refresh.out.log"
ERR_LOG="$LAUNCHD_LOG_DIR/daily_log_refresh.err.log"

mkdir -p "$HOME/Library/LaunchAgents" "$LAUNCHD_LOG_DIR"
chmod +x "$RUN_SCRIPT"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.dankingsley.daily_log_refresh</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$RUN_SCRIPT</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>

  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>23</integer>
    <key>Minute</key>
    <integer>55</integer>
  </dict>

  <key>RunAtLoad</key>
  <false/>

  <key>StandardOutPath</key>
  <string>$OUT_LOG</string>
  <key>StandardErrorPath</key>
  <string>$ERR_LOG</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$UID_NUM" "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_PATH"
launchctl enable "gui/$UID_NUM/$LABEL" || true
launchctl kickstart -k "gui/$UID_NUM/$LABEL" || true

echo "Installed and loaded: $PLIST_PATH"
echo "Schedule: daily at 23:55 local time"
echo "Logs: $OUT_LOG and $ERR_LOG"
