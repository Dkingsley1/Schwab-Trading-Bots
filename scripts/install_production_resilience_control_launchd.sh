#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_ROOT/scripts/ops/runtime_python.sh"
PYTHON_BIN="$(resolve_runtime_python)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/production_resilience_control.py"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.ops.production_resilience_control.plist"
LABEL="com.dankingsley.ops.production_resilience_control"
UID_NUM="$(id -u)"
LOG_DIR="${BOT_OPS_LAUNCHD_LOG_DIR:-/tmp/schwab_trading_bot/launchd_ops}"
INTERVAL_SECONDS="${PRODUCTION_RESILIENCE_CONTROL_INTERVAL_SECONDS:-300}"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$PYTHON_BIN</string>
    <string>$RUN_SCRIPT</string>
    <string>--json</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>MARKET_DATA_ONLY</key><string>1</string>
    <key>ALLOW_ORDER_EXECUTION</key><string>0</string>
    <key>BOT_LIVE_MONEY_LOCKED_DURING_SOAK</key><string>1</string>
    <key>BOT_UNATTENDED_SOAK_ACTIVE</key><string>1</string>
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>$INTERVAL_SECONDS</integer>
  <key>ProcessType</key><string>Background</string>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/ops_production_resilience_control.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/ops_production_resilience_control.err.log</string>
</dict></plist>
PLIST

launchctl bootout "gui/$UID_NUM" "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_PATH"
launchctl enable "gui/$UID_NUM/$LABEL" || true
launchctl kickstart -k "gui/$UID_NUM/$LABEL" || true

echo "Installed and loaded: $PLIST_PATH"
