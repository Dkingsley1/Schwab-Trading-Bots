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

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/sqlite_performance_maintenance.py" \
  --wal-checkpoint-threshold-gb "${SQLITE_WAL_CHECKPOINT_THRESHOLD_GB:-${SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB:-2}}" \
  --wal-truncate-max-gb "${SQLITE_WAL_TRUNCATE_MAX_GB:-${SQL_LINK_SERVICE_WAL_TRUNCATE_MAX_GB:-8}}" \
  --wal-checkpoint-mode "${SQLITE_WAL_CHECKPOINT_MODE:-${SQL_LINK_SERVICE_WAL_CHECKPOINT_MODE:-auto}}" \
  --auto-vacuum-over-gb "${SQLITE_AUTO_VACUUM_OVER_GB:-35}" \
  --vacuum-min-interval-hours "${SQLITE_VACUUM_MIN_INTERVAL_HOURS:-24}" \
  --json
