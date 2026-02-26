#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/retire_persistent_losers.py" \
  --lookback-days "${RETRAIN_RETIRE_LOOKBACK_DAYS:-14}" \
  --min-fail-days "${RETRAIN_RETIRE_MIN_FAIL_DAYS:-5}" \
  --min-no-improvement-streak "${RETRAIN_RETIRE_MIN_NO_IMPROVEMENT_STREAK:-2}" \
  --max-retire-per-run "${RETRAIN_RETIRE_MAX_PER_RUN:-6}" \
  --apply \
  --json

echo "daily_retirement complete utc=$STAMP"
