#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
RUN_SCRIPT="$PROJECT_ROOT/scripts/failover_hot_standby.py"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.failover_hot_standby.plist"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.dankingsley.failover_hot_standby</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON_BIN</string>
    <string>$RUN_SCRIPT</string>
    <string>--interval-seconds</string>
    <string>20</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOG_DIR/failover_hot_standby.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/failover_hot_standby.err.log</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl load "$PLIST_PATH"
echo "Installed and loaded: $PLIST_PATH"
