#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
RUN_SCRIPT="$PROJECT_ROOT/scripts/ops/run_startup_start_prompt_launchd.sh"
LABEL="com.dankingsley.startup_start_prompt"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
UID_NUM="$(id -u)"
GUI_DOMAIN="gui/$UID_NUM"
LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
OUT_LOG="$LOG_DIR/startup_start_prompt.out.log"
ERR_LOG="$LOG_DIR/startup_start_prompt.err.log"

ACTION="install"
KICKSTART_NOW=0
DELAY_SECONDS="${STARTUP_START_PROMPT_DELAY_SECONDS:-20}"
TIMEOUT_SECONDS="${STARTUP_START_PROMPT_TIMEOUT_SECONDS:-600}"
FORCE_RESTART="${STARTUP_START_PROMPT_FORCE_RESTART:-0}"
APPLY_PAPER_LOCK="${STARTUP_START_PROMPT_APPLY_PAPER_LOCK:-1}"
NO_BROWSER="${STARTUP_START_PROMPT_NO_BROWSER:-1}"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)
      ACTION="install"
      ;;
    --uninstall)
      ACTION="uninstall"
      ;;
    --kickstart|--kickstart-now|--test-now)
      KICKSTART_NOW=1
      ;;
    --no-kickstart|--next-login-only)
      KICKSTART_NOW=0
      ;;
    --delay-seconds)
      DELAY_SECONDS="${2:-$DELAY_SECONDS}"
      shift
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="${2:-$TIMEOUT_SECONDS}"
      shift
      ;;
    --force-restart)
      FORCE_RESTART=1
      ;;
    --no-force-restart)
      FORCE_RESTART=0
      ;;
    --no-paper-lock)
      APPLY_PAPER_LOCK=0
      ;;
    --allow-browser)
      NO_BROWSER=0
      ;;
    --no-browser)
      NO_BROWSER=1
      ;;
    *)
      echo "unknown install_startup_start_prompt_launchd arg: $1" >&2
      exit 2
      ;;
  esac
  shift
done

uninstall_prompt() {
  launchctl bootout "$GUI_DOMAIN" "$PLIST_PATH" >/dev/null 2>&1 || true
  launchctl disable "$GUI_DOMAIN/$LABEL" >/dev/null 2>&1 || true
  rm -f "$PLIST_PATH"
  echo "startup_start_prompt_uninstalled label=$LABEL plist=$PLIST_PATH"
}

if [[ "$ACTION" == "uninstall" ]]; then
  uninstall_prompt
  exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
chmod +x "$RUN_SCRIPT"

launchctl bootout "$GUI_DOMAIN" "$PLIST_PATH" >/dev/null 2>&1 || true

BROWSER_OPEN_ALLOWED=1
HEADLESS_RENDER_ENABLED=1
if [[ "$NO_BROWSER" == "1" ]]; then
  BROWSER_OPEN_ALLOWED=0
  HEADLESS_RENDER_ENABLED=0
fi

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$RUN_SCRIPT</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key>
    <string>$HOME</string>
    <key>BOT_RUNTIME_PROFILE</key>
    <string>$RUNTIME_PROFILE</string>
    <key>STARTUP_START_PROMPT_DELAY_SECONDS</key>
    <string>$DELAY_SECONDS</string>
    <key>STARTUP_START_PROMPT_TIMEOUT_SECONDS</key>
    <string>$TIMEOUT_SECONDS</string>
    <key>STARTUP_START_PROMPT_FORCE_RESTART</key>
    <string>$FORCE_RESTART</string>
    <key>STARTUP_START_PROMPT_APPLY_PAPER_LOCK</key>
    <string>$APPLY_PAPER_LOCK</string>
    <key>STARTUP_START_PROMPT_NO_BROWSER</key>
    <string>$NO_BROWSER</string>
    <key>SCHWAB_AUTH_BROWSER_DISABLED</key>
    <string>$NO_BROWSER</string>
    <key>SCHWAB_AUTH_ALLOW_BROWSER_OPEN</key>
    <string>$BROWSER_OPEN_ALLOWED</string>
    <key>PREMARKET_TOKEN_BROWSER_AUTH_DISABLED</key>
    <string>$NO_BROWSER</string>
    <key>CHROME_HEADLESS_QUIET_MODE</key>
    <string>$NO_BROWSER</string>
    <key>REPORT_HEADLESS_BROWSER_RENDER_ENABLED</key>
    <string>$HEADLESS_RENDER_ENABLED</string>
    <key>PROJECT_TIMELINE_AUTO_RENDER_PDF</key>
    <string>$HEADLESS_RENDER_ENABLED</string>
    <key>PROJECT_TIMELINE_ALLOW_GUI_PDF_RENDERER</key>
    <string>$BROWSER_OPEN_ALLOWED</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
  <key>ProcessType</key>
  <string>Interactive</string>
  <key>StandardOutPath</key>
  <string>$OUT_LOG</string>
  <key>StandardErrorPath</key>
  <string>$ERR_LOG</string>
</dict>
</plist>
PLIST

plutil -lint "$PLIST_PATH" >/dev/null
launchctl enable "$GUI_DOMAIN/$LABEL" >/dev/null 2>&1 || true

if [[ "$KICKSTART_NOW" == "1" ]]; then
  launchctl bootstrap "$GUI_DOMAIN" "$PLIST_PATH"
  launchctl kickstart -k "$GUI_DOMAIN/$LABEL" >/dev/null 2>&1 || true
  echo "startup_start_prompt_installed_and_started label=$LABEL plist=$PLIST_PATH"
else
  echo "startup_start_prompt_installed_next_login label=$LABEL plist=$PLIST_PATH"
fi
echo "delay_seconds=$DELAY_SECONDS timeout_seconds=$TIMEOUT_SECONDS force_restart=$FORCE_RESTART paper_lock=$APPLY_PAPER_LOCK no_browser=$NO_BROWSER"
echo "logs=$OUT_LOG $ERR_LOG"
