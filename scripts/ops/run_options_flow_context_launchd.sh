#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv314/bin/python"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"

STATUS_FILE="$PROJECT_ROOT/governance/health/options_flow_context_sync_latest.json"
MAX_STALE_SECONDS="${OPTIONS_FLOW_MAX_STALE_SECONDS:-14400}"

artifact_age_seconds() {
  if [[ ! -f "$STATUS_FILE" ]]; then
    echo 999999999
    return 0
  fi
  local now_ts mtime_ts
  now_ts="$(date +%s)"
  mtime_ts="$(stat -f %m "$STATUS_FILE" 2>/dev/null || echo 0)"
  if [[ -z "$mtime_ts" || "$mtime_ts" == "0" ]]; then
    echo 999999999
    return 0
  fi
  echo $(( now_ts - mtime_ts ))
}

if [[ -z "${POLYGON_API_KEY:-}" && -z "${UNUSUAL_WHALES_API_KEY:-}" && -z "${UNUSUAL_WHALES_EXPORT_PATH:-}" ]]; then
  echo "options_flow_context skip credentials_missing"
  exit 0
fi

age_seconds="$(artifact_age_seconds)"
guard_output=""
if ! guard_output="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/resource_guard.py" --profile optional)"; then
  if (( age_seconds <= ${MAX_STALE_SECONDS:-14400} )); then
    echo "options_flow_context skip resource_guard_blocked age_seconds=${age_seconds} max_stale_seconds=${MAX_STALE_SECONDS} detail=${guard_output:-resource_guard_blocked}"
    exit 0
  fi
  echo "options_flow_context stale_override age_seconds=${age_seconds} max_stale_seconds=${MAX_STALE_SECONDS} detail=${guard_output:-resource_guard_blocked}"
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/options_flow_efficiency_bot.py" \
  --apply \
  --timeout-sec "${OPTIONS_FLOW_EFFICIENCY_TIMEOUT_SECONDS:-900}" \
  --json
