#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/project_timeline_report.py"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.project_timeline_autoupdate.plist"
PRUNE_DAYS="${PROJECT_TIMELINE_PRUNE_OLDER_DAYS:-7}"
PRUNE_KEEP_RUNS="${PROJECT_TIMELINE_PRUNE_KEEP_RUNS:-20}"
UPDATE_SECONDS="${PROJECT_TIMELINE_UPDATE_SECONDS:-300}"
ACTIVITY_HOURS="${PROJECT_TIMELINE_ACTIVITY_HOURS:-72}"
ACTIVITY_LIMIT="${PROJECT_TIMELINE_ACTIVITY_LIMIT:-600}"
PDF_AUTO_RENDER="${PROJECT_TIMELINE_AUTO_RENDER_PDF:-0}"
ALLOW_GUI_PDF_RENDERER="${PROJECT_TIMELINE_ALLOW_GUI_PDF_RENDERER:-0}"
LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
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
    <string>--prune-auto</string>
    <string>--prune-older-days</string>
    <string>$PRUNE_DAYS</string>
    <string>--prune-keep-runs</string>
    <string>$PRUNE_KEEP_RUNS</string>
    <string>--activity-hours</string>
    <string>$ACTIVITY_HOURS</string>
    <string>--activity-limit</string>
    <string>$ACTIVITY_LIMIT</string>
  </array>

  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key>
    <string>$HOME</string>
    <key>PROJECT_TIMELINE_AUTO_RENDER_PDF</key>
    <string>$PDF_AUTO_RENDER</string>
    <key>PROJECT_TIMELINE_ALLOW_GUI_PDF_RENDERER</key>
    <string>$ALLOW_GUI_PDF_RENDERER</string>
  </dict>

  <key>RunAtLoad</key>
  <true/>

  <key>StartInterval</key>
  <integer>$UPDATE_SECONDS</integer>

  <key>WatchPaths</key>
  <array>
    <string>$PROJECT_ROOT/.git/HEAD</string>
    <string>$PROJECT_ROOT/.git/index</string>
    <string>$PROJECT_ROOT/master_bot_registry.json</string>
    <string>$PROJECT_ROOT/README.md</string>
    <string>$PROJECT_ROOT/COMMANDS.md</string>
    <string>$PROJECT_ROOT/scripts</string>
    <string>$PROJECT_ROOT/core</string>
    <string>$PROJECT_ROOT/config</string>
    <string>$PROJECT_ROOT/tests</string>
    <string>$PROJECT_ROOT/governance/health</string>
    <string>$PROJECT_ROOT/governance/walk_forward</string>
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
echo "Auto-update cadence: every ${UPDATE_SECONDS} seconds + on watched project paths"
echo "Logs: $OUT_LOG and $ERR_LOG"
echo "Latest markdown: $PROJECT_ROOT/exports/reports/project_timeline/project_timeline_latest.md"
echo "Latest printable html: $PROJECT_ROOT/exports/reports/project_timeline/project_timeline_print_latest.html"
echo "Latest pdf: $PROJECT_ROOT/exports/reports/project_timeline/project_timeline_latest.pdf"
