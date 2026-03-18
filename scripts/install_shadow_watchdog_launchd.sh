#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER_SCRIPT="$PROJECT_ROOT/scripts/ops/run_shadow_watchdog_launchd.sh"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.shadow_watchdog.plist"
LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
OUT_LOG="$LOG_DIR/shadow_watchdog.out.log"
ERR_LOG="$LOG_DIR/shadow_watchdog.err.log"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"
MARKET_OPEN_HOUR="${MARKET_SESSION_START_HOUR:-4}"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
chmod +x "$RUNNER_SCRIPT"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.dankingsley.shadow_watchdog</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$RUNNER_SCRIPT</string>
  </array>

  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
    <key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string>
    <key>MARKET_SESSION_START_HOUR</key><string>$MARKET_OPEN_HOUR</string>
    <key>MARKET_DATA_ONLY</key><string>1</string>
    <key>ALLOW_ORDER_EXECUTION</key><string>0</string>
    <key>TOP_BOT_PAPER_TRADING_TOP_N</key><string>${TOP_BOT_PAPER_TRADING_TOP_N:-5}</string>
    <key>TOP_BOT_PAPER_TRADING_MIN_ACC</key><string>${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.55}</string>
    <key>TOP_BOT_PAPER_TRADING_PROFILES</key><string>${TOP_BOT_PAPER_TRADING_PROFILES:-default}</string>
    <key>COINBASE_TOP_BOT_PAPER_TRADING_TOP_N</key><string>${COINBASE_TOP_BOT_PAPER_TRADING_TOP_N:-5}</string>
    <key>COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC</key><string>${COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.58}</string>
    <key>COINBASE_TOP_BOT_PAPER_TRADING_PROFILES</key><string>${COINBASE_TOP_BOT_PAPER_TRADING_PROFILES:-default}</string>
    <key>COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N</key><string>${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}</string>
    <key>COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC</key><string>${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.56}</string>
    <key>COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES</key><string>${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-crypto_futures}</string>
    <key>SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT</key><string>${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT:-1}</string>
    <key>SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_MIN_AGE_SECONDS</key><string>${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_MIN_AGE_SECONDS:-60}</string>
    <key>SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_ALLOWED_REASONS</key><string>${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_ALLOWED_REASONS:-incident_auto_halt,global_risk_killswitch,repeated_hard_gates,softguard_api_circuit_opened}</string>
    <key>SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_REQUIRE_PAPER_ONLY</key><string>${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_REQUIRE_PAPER_ONLY:-1}</string>
  </dict>

  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>

  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>

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
echo "Logs: $OUT_LOG and $ERR_LOG"
