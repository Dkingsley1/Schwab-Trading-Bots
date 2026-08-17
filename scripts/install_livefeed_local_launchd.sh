#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
LABEL="com.dankingsley.livefeed-local"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$PROJECT_ROOT/governance/local_livefeed_logs"
STDOUT_LOG="$LOG_DIR/livefeed_local_launchd.out.log"
STDERR_LOG="$LOG_DIR/livefeed_local_launchd.err.log"
GUI_DOMAIN="gui/$(id -u)"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$PROJECT_ROOT/scripts/ops/run_livefeed_local_launchd.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>BOT_RUNTIME_PROFILE</key>
    <string>${BOT_RUNTIME_PROFILE:-live}</string>
    <key>LIVEFEED_LOCAL_SOURCE</key>
    <string>${LIVEFEED_LOCAL_SOURCE:-main}</string>
    <key>LIVEFEED_LOCAL_LINES</key>
    <string>${LIVEFEED_LOCAL_LINES:-120}</string>
    <key>LIVEFEED_LOCAL_HEAVY</key>
    <string>${LIVEFEED_LOCAL_HEAVY:-0}</string>
    <key>LIVEFEED_LOCAL_COLOR</key>
    <string>${LIVEFEED_LOCAL_COLOR:-never}</string>
    <key>LIVEFEED_LOCAL_INCLUDE_COINBASE_WATCHDOG_LOG</key>
    <string>${LIVEFEED_LOCAL_INCLUDE_COINBASE_WATCHDOG_LOG:-0}</string>
    <key>LIVE_FEED_MIRROR_RESTART_SECONDS</key>
    <string>${LIVE_FEED_MIRROR_RESTART_SECONDS:-5}</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ProcessType</key>
  <string>Background</string>
  <key>StandardOutPath</key>
  <string>$STDOUT_LOG</string>
  <key>StandardErrorPath</key>
  <string>$STDERR_LOG</string>
</dict>
</plist>
EOF

launchctl bootout "$GUI_DOMAIN" "$PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "$GUI_DOMAIN" "$PLIST"
launchctl kickstart -k "$GUI_DOMAIN/$LABEL" >/dev/null 2>&1 || true

echo "livefeed_local_launchd_installed label=$LABEL plist=$PLIST stdout=$STDOUT_LOG stderr=$STDERR_LOG"
