#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/project_timeline_report.py"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.project_timeline_autoupdate.plist"
LOG_DIR="$PROJECT_ROOT/logs"
OUT_LOG="$LOG_DIR/project_timeline_autoupdate.out.log"
ERR_LOG="$LOG_DIR/project_timeline_autoupdate.err.log"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.dankingsley.project_timeline_autoupdate</string>

  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON_BIN</string>
    <string>$RUN_SCRIPT</string>
    <string>--auto</string>
    <string>--json</string>
  </array>

  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>

  <key>RunAtLoad</key>
  <true/>

  <key>StartInterval</key>
  <integer>120</integer>

  <key>WatchPaths</key>
  <array>
    <string>$PROJECT_ROOT/.git/HEAD</string>
    <string>$PROJECT_ROOT/.git/index</string>
  </array>

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
echo "Auto-update cadence: every 120 seconds + on git index/head changes"
echo "Logs: $OUT_LOG and $ERR_LOG"
echo "Latest markdown: $PROJECT_ROOT/exports/reports/project_timeline/project_timeline_latest.md"
echo "Latest printable html: $PROJECT_ROOT/exports/reports/project_timeline/project_timeline_print_latest.html"
echo "Latest pdf: $PROJECT_ROOT/exports/reports/project_timeline/project_timeline_latest.pdf"
