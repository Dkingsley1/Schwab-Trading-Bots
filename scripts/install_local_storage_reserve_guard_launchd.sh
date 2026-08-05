#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
UID_NUM="$(id -u)"
LOG_DIR="${BOT_OPS_LAUNCHD_LOG_DIR:-/tmp/schwab_trading_bot/launchd_ops}"
PLIST="$AGENTS_DIR/com.dankingsley.ops.local_storage_reserve_guard.plist"
LABEL="com.dankingsley.ops.local_storage_reserve_guard"
PYTHON_BIN="$PROJECT_ROOT/.venv314/bin/python"
INTERVAL="${BOT_LOCAL_STORAGE_GUARD_INTERVAL_SECONDS:-60}"

mkdir -p "$AGENTS_DIR" "$LOG_DIR"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$PYTHON_BIN</string>
    <string>$PROJECT_ROOT/scripts/ops/local_storage_reserve_guard.py</string>
    <string>--apply</string>
    <string>--json</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>MARKET_DATA_ONLY</key><string>1</string>
    <key>ALLOW_ORDER_EXECUTION</key><string>0</string>
    <key>BOT_LIVE_MONEY_LOCKED_DURING_SOAK</key><string>1</string>
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$INTERVAL</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_local_storage_reserve_guard.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_local_storage_reserve_guard.err.log</string>
</dict></plist>
PLIST

launchctl bootout "gui/$UID_NUM" "$PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST"
launchctl enable "gui/$UID_NUM/$LABEL" || true
launchctl kickstart -k "gui/$UID_NUM/$LABEL" || true

echo "Installed local storage reserve guard: $PLIST"
