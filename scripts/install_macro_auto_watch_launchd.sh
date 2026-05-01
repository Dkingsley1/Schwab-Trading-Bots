#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME_PY_SCRIPT="$PROJECT_ROOT/scripts/ops/runtime_python.sh"
PYTHON_BIN="$($RUNTIME_PY_SCRIPT)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/live_macro_auto_watch.py"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.macro_auto_watch.plist"
LABEL="com.dankingsley.macro_auto_watch"
UID_NUM="$(id -u)"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR" "$PROJECT_ROOT/governance/health"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON_BIN</string>
    <string>$RUN_SCRIPT</string>
    <string>--channel-preset</string>
    <string>fed_policy</string>
    <string>--correlate-with-schwab-calendar</string>
    <string>--trigger-media-ingest-on-live</string>
    <string>--trigger-media-ingest-before-minutes</string>
    <string>20</string>
    <string>--poll-seconds</string>
    <string>45</string>
    <string>--json</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>EnvironmentVariables</key>
  <dict>
    <key>BOT_LOGS_AUTO_SYNC_PRUNE_LOCAL</key><string>0</string>
    <key>PYTHONUNBUFFERED</key><string>1</string>
  </dict>
  <key>StandardOutPath</key><string>$LOG_DIR/macro_auto_watch_launchd.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/macro_auto_watch_launchd.err.log</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$UID_NUM" "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_PATH"
launchctl enable "gui/$UID_NUM/$LABEL" || true
launchctl kickstart -k "gui/$UID_NUM/$LABEL" || true
echo "Installed and loaded: $PLIST_PATH"
