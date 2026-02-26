#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$AGENTS_DIR" "$LOG_DIR"

PY="$PROJECT_ROOT/.venv312/bin/python"
RUN_ALL="$PROJECT_ROOT/scripts/run_all_sleeves.py"
ALL_SLEEVES_PLIST="$AGENTS_DIR/com.dankingsley.all_sleeves.plist"

cat > "$ALL_SLEEVES_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.dankingsley.all_sleeves</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PY</string>
    <string>$RUN_ALL</string>
    <string>--simulate</string>
    <string>--with-aggressive-modes</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOG_DIR/all_sleeves.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/all_sleeves.err.log</string>
</dict>
</plist>
PLIST

launchctl unload "$ALL_SLEEVES_PLIST" >/dev/null 2>&1 || true
launchctl load "$ALL_SLEEVES_PLIST"

# Install companion jobs via existing installers
[[ -x "$PROJECT_ROOT/scripts/install_shadow_watchdog_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_shadow_watchdog_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/install_daily_log_refresh_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_daily_log_refresh_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/install_daily_auto_verify_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_daily_auto_verify_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/install_retrain_schedule_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_retrain_schedule_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/install_daily_retirement_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_daily_retirement_launchd.sh"

echo "Installed all-sleeves launchd stack"
echo "Main: $ALL_SLEEVES_PLIST"
echo "Logs: $LOG_DIR/all_sleeves.out.log and $LOG_DIR/all_sleeves.err.log"
