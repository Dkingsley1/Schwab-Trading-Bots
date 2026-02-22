#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$PROJECT_ROOT/.venv312/bin/python"
SCRIPT="$PROJECT_ROOT/scripts/backup_restore_verify.py"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.weekly_dr_drill.plist"
LOG_DIR="$PROJECT_ROOT/logs"
OUT_LOG="$LOG_DIR/weekly_dr_drill.out.log"
ERR_LOG="$LOG_DIR/weekly_dr_drill.err.log"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.dankingsley.weekly_dr_drill</string>

  <key>ProgramArguments</key>
  <array>
    <string>$PY</string>
    <string>$SCRIPT</string>
  </array>

  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key>
    <integer>1</integer>
    <key>Hour</key>
    <integer>3</integer>
    <key>Minute</key>
    <integer>30</integer>
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

launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl load "$PLIST_PATH"

echo "Installed and loaded: $PLIST_PATH"
echo "Schedule: weekly Monday at 03:30 local time"
echo "Logs: $OUT_LOG and $ERR_LOG"
