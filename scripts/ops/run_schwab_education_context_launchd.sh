#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"

cd "$PROJECT_ROOT"
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH:-/usr/bin:/bin:/usr/sbin:/sbin}"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"

STATUS_FILE="$PROJECT_ROOT/governance/health/schwab_education_context_sync_latest.json"
MAX_STALE_SECONDS="${SCHWAB_EDUCATION_CONTEXT_MAX_STALE_SECONDS:-14400}"
REFRESH_COOLDOWN_SECONDS="${SCHWAB_EDUCATION_CONTEXT_REFRESH_COOLDOWN_SECONDS:-1800}"

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

age_seconds="$(artifact_age_seconds)"
if (( age_seconds < ${REFRESH_COOLDOWN_SECONDS:-1800} )); then
  echo "schwab_education_context skip recent_refresh age_seconds=${age_seconds} cooldown_seconds=${REFRESH_COOLDOWN_SECONDS}"
  exit 0
fi

guard_output=""
if ! guard_output="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/resource_guard.py" --profile optional)"; then
  if (( age_seconds <= ${MAX_STALE_SECONDS:-14400} )); then
    echo "schwab_education_context skip resource_guard_blocked age_seconds=${age_seconds} max_stale_seconds=${MAX_STALE_SECONDS} detail=${guard_output:-resource_guard_blocked}"
    exit 0
  fi
  echo "schwab_education_context stale_override age_seconds=${age_seconds} max_stale_seconds=${MAX_STALE_SECONDS} detail=${guard_output:-resource_guard_blocked}"
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/collect_schwab_education_context.py" --json
