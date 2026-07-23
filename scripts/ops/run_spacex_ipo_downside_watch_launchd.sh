#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
PYTHON_BIN="$("$PROJECT_ROOT/scripts/ops/runtime_python.sh")"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"
export MARKET_DATA_ONLY=1
export ALLOW_ORDER_EXECUTION=0
export TOP_BOT_ENABLE_LIVE_EXECUTION=0
export EXECUTION_LANE_LIVE_ENABLED=0
export RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR=0
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/spacex_ipo_downside_watch.py" \
  --loop \
  --json \
  --symbol "${SPACEX_IPO_WATCH_SYMBOL:-SPCX}" \
  --proxy-symbols "${SPACEX_IPO_WATCH_PROXY_SYMBOLS:-TSLA,RKLB,ASTS,LUNR,ARKX,XAR,ITA,QQQ,SMH,VIXY,UUP}" \
  --drawdown-bands "${SPACEX_IPO_DRAWDOWN_BANDS:-0.05,0.10,0.15,0.20}" \
  --spread-bps-alert "${SPACEX_IPO_SPREAD_BPS_ALERT:-500}" \
  --poll-seconds "${SPACEX_IPO_WATCH_POLL_SECONDS:-30}" \
  --until-utc "${SPACEX_IPO_WATCH_UNTIL_UTC:-2026-06-13T01:00:00+00:00}"
