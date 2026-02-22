#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/daily_auto_verify.py"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.daily_auto_verify.plist"
LOG_DIR="$PROJECT_ROOT/logs"
OUT_LOG="$LOG_DIR/daily_auto_verify.out.log"
ERR_LOG="$LOG_DIR/daily_auto_verify.err.log"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.dankingsley.daily_auto_verify</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON_BIN</string>
    <string>$RUN_SCRIPT</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>23</integer>
    <key>Minute</key><integer>40</integer>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>$OUT_LOG</string>
  <key>StandardErrorPath</key><string>$ERR_LOG</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl load "$PLIST_PATH"

echo "Installed and loaded: $PLIST_PATH"
echo "Schedule: daily at 23:40 local time"
