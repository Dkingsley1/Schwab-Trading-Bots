#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"
SOURCE="${LIVEFEED_LOCAL_SOURCE:-main}"
LINES="${LIVEFEED_LOCAL_LINES:-80}"
HEAVY="${LIVEFEED_LOCAL_HEAVY:-0}"
COLOR="${LIVEFEED_LOCAL_COLOR:-never}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"
export LIVEFEED_HEALTH_WRITER=1
export LIVE_FEED_INCLUDE_COINBASE_WATCHDOG_LOG="${LIVEFEED_LOCAL_INCLUDE_COINBASE_WATCHDOG_LOG:-0}"

args=(--source "$SOURCE" --lines "$LINES")
if [[ "$HEAVY" == "1" || "${HEAVY:l}" == "true" || "${HEAVY:l}" == "yes" || "${HEAVY:l}" == "on" ]]; then
  args+=(--heavy)
fi

case "${COLOR:l}" in
  always|1|true|yes|on)
    args+=(--color)
    ;;
  auto)
    ;;
  never|0|false|no|off|"")
    args+=(--no-color)
    ;;
  *)
    echo "invalid LIVEFEED_LOCAL_COLOR=$COLOR; expected auto, always, or never" >&2
    exit 2
    ;;
esac

exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" "${args[@]}"
