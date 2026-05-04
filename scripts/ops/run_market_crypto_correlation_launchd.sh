#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"

guard_output=""
if ! guard_output="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/resource_guard.py" --profile optional)"; then
  echo "market_crypto_correlation skip resource_guard_blocked detail=${guard_output:-resource_guard_blocked}"
  exit 0
fi

exec "$PROJECT_ROOT/scripts/ops/opsctl.sh" market-correlation-sync \
  --lookback-days "${MARKET_CRYPTO_CORRELATION_LOOKBACK_DAYS:-1}" \
  --bucket-seconds "${MARKET_CRYPTO_CORRELATION_BUCKET_SECONDS:-300}" \
  --min-points "${MARKET_CRYPTO_CORRELATION_MIN_POINTS:-3}" \
  --timeout-seconds "${MARKET_CRYPTO_CORRELATION_TIMEOUT_SECONDS:-90}" \
  --json
