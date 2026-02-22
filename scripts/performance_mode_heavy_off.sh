#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE_FILE="$PROJECT_ROOT/governance/perf/heavy_load_mode.json"
DRY_RUN="${1:-}"

run_cmd() {
  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "[dry-run] $*"
  else
    eval "$*"
  fi
}

if [[ -f "$MODE_FILE" ]]; then
  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "[dry-run] rm $MODE_FILE"
  else
    rm -f "$MODE_FILE"
  fi
fi

for label in \
  com.dankingsley.retrain.daily_small \
  com.dankingsley.retrain.weekly_full \
  com.dankingsley.daily_log_refresh
 do
  run_cmd "launchctl load \"$HOME/Library/LaunchAgents/${label}.plist\" >/dev/null 2>&1 || true"
  echo "resumed launchd job: $label"
done

echo "heavy load mode disabled"
echo "note: aggressive intraday/swing launcher stays off until you start it again manually"
