#!/bin/zsh
set -euo pipefail

BASE_DIR="${1:-/Users/dankingsley/PycharmProjects/schwab_trading_bot}"
TARGET_DIR="${BASE_DIR}/exports/state_snapshot_drills"
RETENTION_DAYS="${STATE_SNAPSHOT_RETENTION_DAYS:-2}"
SNAPSHOT_DRILLS_KEEP="${SNAPSHOT_DRILLS_KEEP:-5}"

[[ -d "$TARGET_DIR" ]] || exit 0

KEEP_N="$SNAPSHOT_DRILLS_KEEP"
[[ "$KEEP_N" =~ ^[0-9]+$ ]] || KEEP_N=5

# Newest-first run directories based on lexicographic timestamp folder names.
RUN_DIRS=( ${(f)$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d -print | sort -r)} )
COUNT=${#RUN_DIRS[@]}

if (( COUNT > KEEP_N )); then
  for d in "${RUN_DIRS[@]:$KEEP_N}"; do
    rm -rf "$d"
  done
fi

# Optional age-based cleanup for residual files.
find "$TARGET_DIR" -type f -mtime +"$RETENTION_DAYS" -delete || true

# Remove empty directories.
find "$TARGET_DIR" -type d -empty -delete || true
