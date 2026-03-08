#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/sqlite_performance_maintenance.py" \
  --auto-vacuum-over-gb "${SQLITE_AUTO_VACUUM_OVER_GB:-35}" \
  --vacuum-min-interval-hours "${SQLITE_VACUUM_MIN_INTERVAL_HOURS:-24}" \
  --json
