#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/refresh_finder_logs.sh"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.schwab.logrefresh.plist"
LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
OUT_LOG="$LOG_DIR/logrefresh_launchd.out.log"
ERR_LOG="$LOG_DIR/logrefresh_launchd.err.log"
REFRESH_ENABLED="${FINDER_LOG_REFRESH_LAUNCHD_ENABLED:-0}"
REFRESH_INTERVAL_SECONDS="${FINDER_LOG_REFRESH_INTERVAL_SECONDS:-1800}"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

if [[ "$REFRESH_ENABLED" != "1" ]]; then
  launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
  rm -f "$PLIST_PATH"
  echo "Legacy finder log refresh is disabled."
  echo "Set FINDER_LOG_REFRESH_LAUNCHD_ENABLED=1 to re-enable it."
  exit 0
fi

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.dankingsley.schwab.logrefresh</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$RUN_SCRIPT</string>
  </array>

  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key>
    <string>$HOME</string>
    <key>FINDER_LOG_REFRESH_DESKTOP_SHORTCUTS</key>
    <string>${FINDER_LOG_REFRESH_DESKTOP_SHORTCUTS:-0}</string>
  </dict>

  <key>RunAtLoad</key>
  <true/>

  <key>StartInterval</key>
  <integer>$REFRESH_INTERVAL_SECONDS</integer>

  <key>StandardOutPath</key>
  <string>$OUT_LOG</string>
  <key>StandardErrorPath</key>
  <string>$ERR_LOG</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl load "$PLIST_PATH"

echo "Installed and loaded: $PLIST_PATH"
echo "Schedule: every $REFRESH_INTERVAL_SECONDS seconds"
echo "Logs: $OUT_LOG and $ERR_LOG"
