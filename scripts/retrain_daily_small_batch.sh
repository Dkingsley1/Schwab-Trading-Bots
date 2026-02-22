#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

cd "$PROJECT_ROOT"

export RETRAIN_ACTIVE_ONLY=1
export RETRAIN_MAX_TARGETS=12
export RETRAIN_MIN_MODEL_AGE_HOURS=18
export RETRAIN_AFTER_HOURS_ONLY=1

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/weekly_retrain.py" \
  --continue-on-error \
  --active-only \
  --max-targets 12 \
  --min-model-age-hours 18

echo "daily_small_retrain complete utc=$STAMP"
