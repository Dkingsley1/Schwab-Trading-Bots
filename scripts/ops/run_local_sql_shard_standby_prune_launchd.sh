#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"
PYTHON_BIN="$("$PROJECT_ROOT/scripts/ops/runtime_python.sh")"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/local_sql_shard_standby_prune.py" \
  --apply \
  --json \
  --max-delete-gb "${LOCAL_SQL_SHARD_STANDBY_PRUNE_MAX_DELETE_GB:-512}" \
  --min-age-minutes "${LOCAL_SQL_SHARD_STANDBY_PRUNE_MIN_AGE_MINUTES:-0}"
