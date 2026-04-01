#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_premarket_token_guard_launchd.sh"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.premarket_token_guard.plist"
LABEL="com.dankingsley.premarket_token_guard"
UID_NUM="$(id -u)"
LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
OUT_LOG="$LOG_DIR/premarket_token_guard.out.log"
ERR_LOG="$LOG_DIR/premarket_token_guard.err.log"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"
CHECK_INTERVAL_SECONDS="${PREMARKET_TOKEN_CHECK_INTERVAL_SECONDS:-600}"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
chmod +x "$RUN_SCRIPT"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
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
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$CHECK_INTERVAL_SECONDS</integer>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>3</integer><key>Minute</key><integer>50</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>3</integer><key>Minute</key><integer>50</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>3</integer><key>Minute</key><integer>50</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>3</integer><key>Minute</key><integer>50</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>3</integer><key>Minute</key><integer>50</integer></dict>
  </array>
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
echo "Profile: $RUNTIME_PROFILE"
echo "Check interval seconds: $CHECK_INTERVAL_SECONDS"
echo "Logs: $OUT_LOG and $ERR_LOG"
