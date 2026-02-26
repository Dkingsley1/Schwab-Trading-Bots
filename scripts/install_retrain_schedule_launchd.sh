#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DAILY_SCRIPT="$PROJECT_ROOT/scripts/retrain_daily_small_batch.sh"
WEEKLY_SCRIPT="$PROJECT_ROOT/scripts/retrain_weekly_full_sweep.sh"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$PROJECT_ROOT/logs"

DAILY_PLIST="$AGENTS_DIR/com.dankingsley.retrain.daily_small.plist"
WEEKLY_PLIST="$AGENTS_DIR/com.dankingsley.retrain.weekly_full.plist"

mkdir -p "$AGENTS_DIR" "$LOG_DIR"

cat > "$DAILY_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.dankingsley.retrain.daily_small</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$DAILY_SCRIPT</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>21</integer>
    <key>Minute</key><integer>20</integer>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>$LOG_DIR/retrain_daily_small.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/retrain_daily_small.err.log</string>
</dict>
</plist>
PLIST

cat > "$WEEKLY_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.dankingsley.retrain.weekly_full</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$WEEKLY_SCRIPT</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>6</integer>
    <key>Hour</key><integer>20</integer>
    <key>Minute</key><integer>15</integer>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>$LOG_DIR/retrain_weekly_full.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/retrain_weekly_full.err.log</string>
</dict>
</plist>
PLIST

launchctl unload "$DAILY_PLIST" >/dev/null 2>&1 || true
launchctl unload "$WEEKLY_PLIST" >/dev/null 2>&1 || true
launchctl load "$DAILY_PLIST"
launchctl load "$WEEKLY_PLIST"

echo "Installed: $DAILY_PLIST"
echo "Installed: $WEEKLY_PLIST"
echo "Daily small batch: 21:20 local"
echo "Weekly full sweep: Friday 20:15 local"
echo "Logs: $LOG_DIR/retrain_daily_small.*.log and $LOG_DIR/retrain_weekly_full.*.log"
