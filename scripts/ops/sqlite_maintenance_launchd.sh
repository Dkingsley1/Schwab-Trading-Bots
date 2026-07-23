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

sqlite_args=(
  --wal-checkpoint-threshold-gb "${SQLITE_WAL_CHECKPOINT_THRESHOLD_GB:-${SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB:-2}}" \
  --wal-truncate-max-gb "${SQLITE_WAL_TRUNCATE_MAX_GB:-${SQL_LINK_SERVICE_WAL_TRUNCATE_MAX_GB:-8}}" \
  --wal-checkpoint-mode "${SQLITE_WAL_CHECKPOINT_MODE:-${SQL_LINK_SERVICE_WAL_CHECKPOINT_MODE:-auto}}" \
  --auto-vacuum-over-gb "${SQLITE_AUTO_VACUUM_OVER_GB:-35}" \
  --vacuum-min-interval-hours "${SQLITE_VACUUM_MIN_INTERVAL_HOURS:-24}" \
  --max-runtime-seconds "${SQLITE_MAINTENANCE_MAX_RUNTIME_SECONDS:-7200}" \
  --json
)

if [[ "${SQLITE_LAUNCHD_ALLOW_AUTO_VACUUM:-0}" != "1" ]]; then
  sqlite_args+=(--no-auto-vacuum)
fi

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" sqlite_maintenance \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/sqlite_performance_maintenance.py" \
  "${sqlite_args[@]}"
