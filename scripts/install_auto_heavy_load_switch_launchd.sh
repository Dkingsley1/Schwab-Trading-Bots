#!/bin/zsh
set -euo pipefail
LABEL="com.dankingsley.heavyload.autoswitch"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
SCRIPT="/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/auto_heavy_load_switch.sh"

mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key><string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
      <string>/bin/zsh</string>
      <string>-lc</string>
      <string>${SCRIPT}</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>StartInterval</key><integer>20</integer>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/heavyload_autoswitch.out</string>
    <key>StandardErrorPath</key><string>/tmp/heavyload_autoswitch.err</string>
  </dict>
</plist>
PLISTEOF

/bin/launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
/bin/launchctl bootstrap "gui/$(id -u)" "$PLIST"
/bin/launchctl kickstart -k "gui/$(id -u)/${LABEL}"
echo "installed: ${LABEL}"
