#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"
PYTHON_BIN="$("$PROJECT_ROOT/scripts/ops/runtime_python.sh")"
export PYTHONUNBUFFERED=1
RUNAWAY_AGE_SECONDS="${CHROME_HEADLESS_RUNAWAY_AGE_SECONDS:-45}"
ORPHAN_GRACE_SECONDS="${CHROME_HEADLESS_ORPHAN_GRACE_SECONDS:-45}"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/chrome_headless_guard.py" \
  --apply \
  --runaway-headless-age-seconds "$RUNAWAY_AGE_SECONDS" \
  --orphan-grace-seconds "$ORPHAN_GRACE_SECONDS" \
  --json
