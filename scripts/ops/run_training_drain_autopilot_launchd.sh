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

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"
export BOT_PROTECTED_VOLUME_DENYLIST="${BOT_PROTECTED_VOLUME_DENYLIST:-/Volumes/VIDEO}"

"$PROJECT_ROOT/scripts/ops/run_guarded_maintenance.sh" training_drain_autopilot \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/training_drain_autopilot.py" \
  --apply \
  --limit "${TRAINING_DRAIN_AUTOPILOT_LIMIT:-4}" \
  --max-cycles "${TRAINING_DRAIN_AUTOPILOT_MAX_CYCLES:-1}" \
  --poll-seconds "${TRAINING_DRAIN_AUTOPILOT_POLL_SECONDS:-120}" \
  --command-timeout-seconds "${TRAINING_DRAIN_AUTOPILOT_COMMAND_TIMEOUT_SECONDS:-900}" \
  --wait-timeout-seconds "${TRAINING_DRAIN_AUTOPILOT_WAIT_TIMEOUT_SECONDS:-900}" \
  --storage-autopilot-cycles "${TRAINING_DRAIN_AUTOPILOT_STORAGE_CYCLES:-1}" \
  --target-free-gb "${TRAINING_DRAIN_AUTOPILOT_TELEMETRY_TARGET_FREE_GB:-32}" \
  --min-telemetry-file-mb "${TRAINING_DRAIN_AUTOPILOT_MIN_TELEMETRY_FILE_MB:-64}" \
  --max-telemetry-files "${TRAINING_DRAIN_AUTOPILOT_MAX_TELEMETRY_FILES:-64}" \
  --json
