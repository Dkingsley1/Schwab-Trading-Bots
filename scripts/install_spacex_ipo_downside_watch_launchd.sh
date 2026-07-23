#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_spacex_ipo_downside_watch_launchd.sh"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.spacex_ipo_downside_watch.plist"
LABEL="com.dankingsley.spacex_ipo_downside_watch"
UID_NUM="$(id -u)"
LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
OUT_LOG="$LOG_DIR/spacex_ipo_downside_watch.out.log"
ERR_LOG="$LOG_DIR/spacex_ipo_downside_watch.err.log"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"
POLL_SECONDS="${SPACEX_IPO_WATCH_POLL_SECONDS:-30}"
WATCH_SYMBOL="${SPACEX_IPO_WATCH_SYMBOL:-SPCX}"
WATCH_UNTIL="${SPACEX_IPO_WATCH_UNTIL_UTC:-2026-06-13T01:00:00+00:00}"
DRAWDOWN_BANDS="${SPACEX_IPO_DRAWDOWN_BANDS:-0.05,0.10,0.15,0.20}"
SPREAD_BPS_ALERT="${SPACEX_IPO_SPREAD_BPS_ALERT:-500}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll-seconds)
      POLL_SECONDS="${2:-$POLL_SECONDS}"
      shift
      ;;
    --symbol)
      WATCH_SYMBOL="${2:-$WATCH_SYMBOL}"
      shift
      ;;
    --until-utc)
      WATCH_UNTIL="${2:-$WATCH_UNTIL}"
      shift
      ;;
    --drawdown-bands)
      DRAWDOWN_BANDS="${2:-$DRAWDOWN_BANDS}"
      shift
      ;;
    --spread-bps-alert)
      SPREAD_BPS_ALERT="${2:-$SPREAD_BPS_ALERT}"
      shift
      ;;
    *)
      echo "unknown install_spacex_ipo_downside_watch_launchd arg: $1" >&2
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR" "$PROJECT_ROOT/governance/health"
chmod +x "$RUN_SCRIPT" "$PROJECT_ROOT/scripts/ops/spacex_ipo_downside_watch.py"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$RUN_SCRIPT</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
    <key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string>
    <key>SPACEX_IPO_WATCH_POLL_SECONDS</key><string>$POLL_SECONDS</string>
    <key>SPACEX_IPO_WATCH_SYMBOL</key><string>$WATCH_SYMBOL</string>
    <key>SPACEX_IPO_WATCH_UNTIL_UTC</key><string>$WATCH_UNTIL</string>
    <key>SPACEX_IPO_DRAWDOWN_BANDS</key><string>$DRAWDOWN_BANDS</string>
    <key>SPACEX_IPO_SPREAD_BPS_ALERT</key><string>$SPREAD_BPS_ALERT</string>
    <key>MARKET_DATA_ONLY</key><string>1</string>
    <key>ALLOW_ORDER_EXECUTION</key><string>0</string>
    <key>PYTHONUNBUFFERED</key><string>1</string>
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>$OUT_LOG</string>
  <key>StandardErrorPath</key><string>$ERR_LOG</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$UID_NUM" "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_PATH"
launchctl enable "gui/$UID_NUM/$LABEL" || true
launchctl kickstart -k "gui/$UID_NUM/$LABEL" || true

echo "Installed and loaded: $PLIST_PATH"
echo "Profile: $RUNTIME_PROFILE"
echo "Symbol: $WATCH_SYMBOL"
echo "Poll seconds: $POLL_SECONDS"
echo "Until UTC: $WATCH_UNTIL"
echo "Drawdown bands: $DRAWDOWN_BANDS"
echo "Spread bps alert: $SPREAD_BPS_ALERT"
echo "Logs: $OUT_LOG and $ERR_LOG"
