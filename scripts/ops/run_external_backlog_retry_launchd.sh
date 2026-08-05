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

if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/external_backlog_retry_bot.py" \
  --apply \
  --poll-seconds "${EXTERNAL_BACKLOG_RETRY_POLL_SECONDS:-20}" \
  --wait-timeout-seconds "${EXTERNAL_BACKLOG_RETRY_WAIT_TIMEOUT_SECONDS:-900}" \
  --command-timeout-seconds "${EXTERNAL_BACKLOG_RETRY_COMMAND_TIMEOUT_SECONDS:-1800}" \
  --json >/dev/null; then
  print -r -- "external_backlog_retry status=complete"
else
  rc=$?
  print -u2 -r -- "external_backlog_retry status=failed rc=$rc"
  exit "$rc"
fi
