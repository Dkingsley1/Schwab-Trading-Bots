#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
UID_NUM="$(id -u)"
LOG_DIR="${BOT_OPS_LAUNCHD_LOG_DIR:-/tmp/schwab_trading_bot/launchd_ops}"
RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_process_fanout_guard_launchd.sh"
PLIST="$AGENTS_DIR/com.dankingsley.ops.process_fanout_guard.plist"
INTERVAL="${PROCESS_FANOUT_GUARD_INTERVAL_SECONDS:-45}"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"

mkdir -p "$AGENTS_DIR" "$LOG_DIR"
chmod +x "$RUN_SCRIPT"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.dankingsley.ops.process_fanout_guard</string>
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>$RUN_SCRIPT</string></array>
  <key>EnvironmentVariables</key><dict><key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string></dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_process_fanout_guard.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_process_fanout_guard.err.log</string>
</dict></plist>
PLIST

launchctl bootout "gui/$UID_NUM" "$PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST"
launchctl enable "gui/$UID_NUM/com.dankingsley.ops.process_fanout_guard" || true
launchctl kickstart -k "gui/$UID_NUM/com.dankingsley.ops.process_fanout_guard" || true
echo "Installed process fanout guard: $PLIST"
