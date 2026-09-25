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
if [[ -f "$PROJECT_ROOT/scripts/ops/adaptive_ops_recovery_policy.py" ]]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/adaptive_ops_recovery_policy.py" --apply --no-control-plane-refresh --out-file "$PROJECT_ROOT/governance/health/adaptive_ops_recovery_inputs_latest.json" --json >/dev/null 2>&1 || true
  if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
  fi
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/backlog_drain_uniform_process.py" --apply --json >/dev/null 2>&1 || true

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

# A completed writer lock must be handed off before the busy-writer maintenance gate.
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/writer_cycle_coordinator.py" --apply --handoff-only --handoff-grace-seconds 1 --json >/dev/null 2>&1 || true

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" storage_backpressure_autopilot \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/storage_backpressure_autopilot.py" \
  --apply \
  --poll-seconds "${STORAGE_BACKPRESSURE_AUTOPILOT_POLL_SECONDS:-20}" \
  --wait-timeout-seconds "${STORAGE_BACKPRESSURE_AUTOPILOT_WAIT_TIMEOUT_SECONDS:-900}" \
  --command-timeout-seconds "${STORAGE_BACKPRESSURE_AUTOPILOT_TIMEOUT_SECONDS:-2400}" \
  --backpressure-command-timeout-seconds "${STORAGE_BACKPRESSURE_AUTOPILOT_BACKPRESSURE_TIMEOUT_SECONDS:-900}" \
  --max-cycles "${STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES:-3}" \
  --target-pending-lines "${STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_PENDING_LINES:-20000}" \
  --target-retention-debt-gb "${STORAGE_BACKPRESSURE_AUTOPILOT_TARGET_RETENTION_DEBT_GB:-0.25}" \
  --json
