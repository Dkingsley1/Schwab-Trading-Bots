#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER_SCRIPT="$PROJECT_ROOT/scripts/ops/run_storage_eject_guard_launchd.sh"
GUARD_SOURCE="$PROJECT_ROOT/scripts/ops/storage_eject_guard.swift"
GUARD_BIN_DIR="$HOME/Library/Application Support/schwab_trading_bot/bin"
GUARD_BINARY="$GUARD_BIN_DIR/storage_eject_guard"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.storage_eject_guard.plist"
LABEL="com.dankingsley.storage_eject_guard"
UID_NUM="$(id -u)"
LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
OUT_LOG="$LOG_DIR/storage_eject_guard.out.log"
ERR_LOG="$LOG_DIR/storage_eject_guard.err.log"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
chmod +x "$RUNNER_SCRIPT"
mkdir -p "$GUARD_BIN_DIR"
GUARD_BINARY_TMP="$GUARD_BINARY.tmp.$$"
SWIFT_CACHE_DIR="${TMPDIR:-/tmp}/schwab_trading_bot_swift_module_cache"
mkdir -p "$SWIFT_CACHE_DIR"
CLANG_MODULE_CACHE_PATH="$SWIFT_CACHE_DIR" SWIFT_MODULE_CACHE_PATH="$SWIFT_CACHE_DIR" \
  /usr/bin/swiftc -O "$GUARD_SOURCE" -o "$GUARD_BINARY_TMP"
chmod +x "$GUARD_BINARY_TMP"
mv "$GUARD_BINARY_TMP" "$GUARD_BINARY"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.dankingsley.storage_eject_guard</string>

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
    <key>STORAGE_EJECT_GUARD_BINARY</key><string>$GUARD_BINARY</string>
  </dict>

  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>

  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>60</integer>

  <key>StandardOutPath</key>
  <string>$OUT_LOG</string>
  <key>StandardErrorPath</key>
  <string>$ERR_LOG</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$UID_NUM" "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_PATH"
launchctl enable "gui/$UID_NUM/$LABEL" || true
launchctl kickstart -k "gui/$UID_NUM/$LABEL" || true

echo "Installed and loaded: $PLIST_PATH"
echo "Compiled guard: $GUARD_BINARY"
echo "Logs: $OUT_LOG and $ERR_LOG"
