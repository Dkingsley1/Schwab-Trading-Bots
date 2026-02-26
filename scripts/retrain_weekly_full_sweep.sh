#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/resource_guard.py" || {
  echo "retrain skipped by resource_guard"
  exit 0
}

export RETRAIN_ACTIVE_ONLY=1
export RETRAIN_MAX_TARGETS=0
export RETRAIN_MIN_MODEL_AGE_HOURS=0
export RETRAIN_AFTER_HOURS_ONLY=1
export RETRAIN_RETIRE_PERSISTENT_LOSERS=1
export RETRAIN_RETIRE_APPLY=1
export RETRAIN_RETIRE_MIN_FAIL_DAYS=5
export RETRAIN_RETIRE_MIN_NO_IMPROVEMENT_STREAK=2
export RETRAIN_RETIRE_MAX_PER_RUN=6
export RETRAIN_REGIME_BALANCE=1
export RETRAIN_PROMOTION_BOTTLENECK_PRIORITY=1
export TRADE_BEHAVIOR_CLASS_BALANCE_CAP=5.5
export TRADE_BEHAVIOR_NEUTRAL_WEIGHT_FLOOR=1.35
export TRADE_BEHAVIOR_POSITIVE_WEIGHT_FLOOR=2.00
export TRADE_BEHAVIOR_NEGATIVE_WEIGHT_CAP=0.85
export TRADE_BEHAVIOR_REGIME_BALANCE_CAP=3.0
export TRADE_BEHAVIOR_REQUIRE_WALK_FORWARD_OK=1
export TRADE_BEHAVIOR_STRICT_PROMOTION_GATE=1

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/weekly_retrain.py" \
  --continue-on-error \
  --active-only \
  --max-targets 0 \
  --min-model-age-hours 0

echo "weekly_full_sweep complete utc=$STAMP"
