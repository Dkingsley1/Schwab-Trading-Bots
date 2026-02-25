#!/bin/zsh
set -euo pipefail

BASE_DIR="${1:-/Users/dankingsley/PycharmProjects/schwab_trading_bot}"
TARGET_DIR="${BASE_DIR}/exports/state_snapshot_drills"
RETENTION_DAYS="${STATE_SNAPSHOT_RETENTION_DAYS:-2}"

[[ -d "$TARGET_DIR" ]] || exit 0

# Delete files older than retention window, then remove empty directories.
find "$TARGET_DIR" -type f -mtime +"$RETENTION_DAYS" -delete
find "$TARGET_DIR" -type d -empty -delete
