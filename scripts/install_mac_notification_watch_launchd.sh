#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_mac_notification_watch_launchd.sh"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.mac_notification_watch.plist"
LABEL="com.dankingsley.mac_notification_watch"
UID_NUM="$(id -u)"
LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
OUT_LOG="$LOG_DIR/mac_notification_watch.out.log"
ERR_LOG="$LOG_DIR/mac_notification_watch.err.log"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"
POLL_SECONDS="${MAC_NOTIFICATION_WATCH_POLL_SECONDS:-30}"
IMESSAGE_ENABLED="${MAC_NOTIFICATION_WATCH_IMESSAGE_ENABLED:-0}"
IMESSAGE_RECIPIENT="${MAC_NOTIFICATION_WATCH_IMESSAGE_RECIPIENT:-}"
IMESSAGE_MIN_SEVERITY="${MAC_NOTIFICATION_WATCH_IMESSAGE_MIN_SEVERITY:-warn}"
IMESSAGE_EVENT_ALLOWLIST="${MAC_NOTIFICATION_WATCH_IMESSAGE_EVENT_ALLOWLIST:-}"
MIN_REPEAT_SECONDS="${MAC_NOTIFICATION_WATCH_MIN_REPEAT_SECONDS:-300}"
POWER_EVENTS_ENABLED="${MAC_NOTIFICATION_WATCH_POWER_EVENTS_ENABLED:-1}"
PMSET_CACHE_SECONDS="${MAC_NOTIFICATION_WATCH_PMSET_CACHE_SECONDS:-900}"
SKIP_PMSET_UNDER_PRESSURE="${MAC_NOTIFICATION_WATCH_SKIP_PMSET_UNDER_PRESSURE:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll-seconds)
      POLL_SECONDS="${2:-$POLL_SECONDS}"
      shift
      ;;
    --enable-imessage)
      IMESSAGE_ENABLED=1
      ;;
    --disable-imessage)
      IMESSAGE_ENABLED=0
      ;;
    --imessage-recipient)
      IMESSAGE_RECIPIENT="${2:-$IMESSAGE_RECIPIENT}"
      IMESSAGE_ENABLED=1
      shift
      ;;
    --imessage-min-severity)
      IMESSAGE_MIN_SEVERITY="${2:-$IMESSAGE_MIN_SEVERITY}"
      shift
      ;;
    --imessage-event-allowlist)
      IMESSAGE_EVENT_ALLOWLIST="${2:-$IMESSAGE_EVENT_ALLOWLIST}"
      shift
      ;;
    --min-repeat-seconds)
      MIN_REPEAT_SECONDS="${2:-$MIN_REPEAT_SECONDS}"
      shift
      ;;
    --disable-power-events)
      POWER_EVENTS_ENABLED=0
      ;;
    --enable-power-events)
      POWER_EVENTS_ENABLED=1
      ;;
    --pmset-cache-seconds)
      PMSET_CACHE_SECONDS="${2:-$PMSET_CACHE_SECONDS}"
      shift
      ;;
    --disable-pmset-pressure-skip)
      SKIP_PMSET_UNDER_PRESSURE=0
      ;;
    --enable-pmset-pressure-skip)
      SKIP_PMSET_UNDER_PRESSURE=1
      ;;
    *)
      echo "unknown install_mac_notification_watch_launchd arg: $1" >&2
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
chmod +x "$RUN_SCRIPT"

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
    <key>MAC_NOTIFICATION_WATCH_POLL_SECONDS</key><string>$POLL_SECONDS</string>
    <key>MAC_NOTIFICATION_WATCH_IMESSAGE_ENABLED</key><string>$IMESSAGE_ENABLED</string>
    <key>MAC_NOTIFICATION_WATCH_IMESSAGE_RECIPIENT</key><string>$IMESSAGE_RECIPIENT</string>
    <key>MAC_NOTIFICATION_WATCH_IMESSAGE_MIN_SEVERITY</key><string>$IMESSAGE_MIN_SEVERITY</string>
    <key>MAC_NOTIFICATION_WATCH_IMESSAGE_EVENT_ALLOWLIST</key><string>$IMESSAGE_EVENT_ALLOWLIST</string>
    <key>MAC_NOTIFICATION_WATCH_MIN_REPEAT_SECONDS</key><string>$MIN_REPEAT_SECONDS</string>
    <key>MAC_NOTIFICATION_WATCH_POWER_EVENTS_ENABLED</key><string>$POWER_EVENTS_ENABLED</string>
    <key>MAC_NOTIFICATION_WATCH_PMSET_CACHE_SECONDS</key><string>$PMSET_CACHE_SECONDS</string>
    <key>MAC_NOTIFICATION_WATCH_SKIP_PMSET_UNDER_PRESSURE</key><string>$SKIP_PMSET_UNDER_PRESSURE</string>
  </dict>
  <key>WorkingDirectory</key><string>$PROJECT_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$OUT_LOG</string>
  <key>StandardErrorPath</key><string>$ERR_LOG</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$UID_NUM" "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl enable "gui/$UID_NUM/$LABEL" || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_PATH"
launchctl kickstart -k "gui/$UID_NUM/$LABEL" || true

echo "Installed and loaded: $PLIST_PATH"
echo "Profile: $RUNTIME_PROFILE"
echo "Poll seconds: $POLL_SECONDS"
echo "iMessage enabled: $IMESSAGE_ENABLED"
echo "iMessage recipient configured: $([[ -n "$IMESSAGE_RECIPIENT" ]] && echo yes || echo no)"
echo "iMessage min severity: $IMESSAGE_MIN_SEVERITY"
echo "iMessage event allowlist: ${IMESSAGE_EVENT_ALLOWLIST:-ALL}"
echo "Min repeat seconds: $MIN_REPEAT_SECONDS"
echo "Power events enabled: $POWER_EVENTS_ENABLED"
echo "pmset cache seconds: $PMSET_CACHE_SECONDS"
echo "Skip pmset under pressure: $SKIP_PMSET_UNDER_PRESSURE"
echo "Logs: $OUT_LOG and $ERR_LOG"
