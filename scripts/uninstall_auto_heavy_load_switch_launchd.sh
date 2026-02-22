#!/bin/zsh
set -euo pipefail
LABEL="com.dankingsley.heavyload.autoswitch"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
/bin/launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "uninstalled: ${LABEL}"
