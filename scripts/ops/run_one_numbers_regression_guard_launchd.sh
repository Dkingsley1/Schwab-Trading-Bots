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

SUMMARY_PATH="$PROJECT_ROOT/exports/one_numbers/one_numbers_summary.json"
LATEST_CSV_PATH="$PROJECT_ROOT/exports/one_numbers/latest.csv"
LATEST_METRICS_PATH="$PROJECT_ROOT/exports/one_numbers/latest_metrics.csv"
if [[ ! -s "$SUMMARY_PATH" || ! -s "$LATEST_CSV_PATH" || ! -s "$LATEST_METRICS_PATH" ]]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/one_numbers_regression_guard.py" --apply --json
  exit $?
fi

# One Numbers is market-session observability, so quiet-hours and the continuous
# SQL writer must not suppress its bounded read/repair pass.
export MAINTENANCE_SLOT_DEFER_OUTSIDE_QUIET_WINDOW=0
export MAINTENANCE_SLOT_DEFER_WHILE_SQL_LINK_ACTIVE=0

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" one_numbers_regression_guard \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/one_numbers_regression_guard.py" \
  --apply \
  --json
