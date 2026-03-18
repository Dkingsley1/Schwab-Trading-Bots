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
POLL_SECONDS="${MAC_NOTIFICATION_WATCH_POLL_SECONDS:-8}"
IMESSAGE_ENABLED="${MAC_NOTIFICATION_WATCH_IMESSAGE_ENABLED:-0}"
IMESSAGE_RECIPIENT="${MAC_NOTIFICATION_WATCH_IMESSAGE_RECIPIENT:-}"
IMESSAGE_MIN_SEVERITY="${MAC_NOTIFICATION_WATCH_IMESSAGE_MIN_SEVERITY:-warn}"
IMESSAGE_EVENT_ALLOWLIST="${MAC_NOTIFICATION_WATCH_IMESSAGE_EVENT_ALLOWLIST:-}"

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
echo "Logs: $OUT_LOG and $ERR_LOG"
