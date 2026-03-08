#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$AGENTS_DIR" "$LOG_DIR"

RUN_ALL_LAUNCHER="$PROJECT_ROOT/scripts/ops/run_all_sleeves_launchd.sh"
ALL_SLEEVES_PLIST="$AGENTS_DIR/com.dankingsley.all_sleeves.plist"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"
MARKET_OPEN_HOUR="${MARKET_SESSION_START_HOUR:-4}"
OUT_LOG="/tmp/com.dankingsley.all_sleeves.out.log"
ERR_LOG="/tmp/com.dankingsley.all_sleeves.err.log"
ORCHESTRATOR_MODE="${STACK_ORCHESTRATOR_MODE:-watchdog}"

if [[ "$ORCHESTRATOR_MODE" == "all_sleeves" ]]; then
  chmod +x "$RUN_ALL_LAUNCHER"

  cat > "$ALL_SLEEVES_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.dankingsley.all_sleeves</string>
  <key>ProgramArguments</key>
  <array>
    <string>$RUN_ALL_LAUNCHER</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string>
    <key>MARKET_SESSION_START_HOUR</key><string>$MARKET_OPEN_HOUR</string>
    <key>MARKET_DATA_ONLY</key><string>1</string>
    <key>ALLOW_ORDER_EXECUTION</key><string>0</string>
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$OUT_LOG</string>
  <key>StandardErrorPath</key><string>$ERR_LOG</string>
</dict>
</plist>
PLIST

  launchctl unload "$ALL_SLEEVES_PLIST" >/dev/null 2>&1 || true
  launchctl load "$ALL_SLEEVES_PLIST"
  echo "Installed all-sleeves launcher (orchestrator_mode=all_sleeves)"
else
  launchctl bootout "gui/$(id -u)" "$ALL_SLEEVES_PLIST" >/dev/null 2>&1 || launchctl unload "$ALL_SLEEVES_PLIST" >/dev/null 2>&1 || true
  rm -f "$ALL_SLEEVES_PLIST" >/dev/null 2>&1 || true
  echo "Skipped all-sleeves launcher (orchestrator_mode=$ORCHESTRATOR_MODE)"
fi

# Install companion jobs via existing installers
[[ -x "$PROJECT_ROOT/scripts/install_shadow_watchdog_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_shadow_watchdog_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/install_caffeinate_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_caffeinate_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/install_daily_log_refresh_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_daily_log_refresh_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/install_daily_auto_verify_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_daily_auto_verify_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/install_retrain_schedule_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_retrain_schedule_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/install_daily_retirement_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_daily_retirement_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/ops/install_ops_automation_launchd.sh" ]] && "$PROJECT_ROOT/scripts/ops/install_ops_automation_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/install_failover_watch_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_failover_watch_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/install_premarket_token_guard_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_premarket_token_guard_launchd.sh"
[[ -x "$PROJECT_ROOT/scripts/install_reboot_resilience_launchd.sh" ]] && "$PROJECT_ROOT/scripts/install_reboot_resilience_launchd.sh"

echo "Installed infra launchd stack"
if [[ "$ORCHESTRATOR_MODE" == "all_sleeves" ]]; then
  echo "Main: $ALL_SLEEVES_PLIST"
  echo "Logs: $OUT_LOG and $ERR_LOG"
else
  echo "Main: shadow_watchdog launchd (all_sleeves disabled)"
fi
