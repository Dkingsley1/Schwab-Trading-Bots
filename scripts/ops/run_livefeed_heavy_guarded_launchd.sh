#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh"
fi

SOURCE="${LIVEFEED_HEAVY_GUARDED_SOURCE:-main}"
LINES="${LIVEFEED_HEAVY_GUARDED_LINES:-80}"
TTL_SECONDS="${LIVEFEED_HEAVY_GUARDED_TTL_SECONDS:-900}"
CHECK_INTERVAL_SECONDS="${LIVEFEED_HEAVY_GUARDED_CHECK_INTERVAL_SECONDS:-20}"
WAIT_SECONDS="${LIVEFEED_HEAVY_GUARDED_WAIT_SECONDS:-30}"
TAIL_LINES="${LIVEFEED_HEAVY_GUARDED_TAIL_LINES:-120}"

exec "$PROJECT_ROOT/scripts/ops/live_feed_heavy_guarded.sh" \
  --source "$SOURCE" \
  --lines "$LINES" \
  --ttl-seconds "$TTL_SECONDS" \
  --check-interval-seconds "$CHECK_INTERVAL_SECONDS" \
  --wait-seconds "$WAIT_SECONDS" \
  --tail-lines "$TAIL_LINES"
