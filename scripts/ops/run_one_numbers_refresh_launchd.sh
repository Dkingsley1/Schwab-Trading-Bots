#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
PROFILE="${BOT_RUNTIME_PROFILE:-live}"
SUMMARY_PATH="$PROJECT_ROOT/exports/one_numbers/one_numbers_summary.json"
BACKPRESSURE_PATH="$PROJECT_ROOT/governance/health/ingestion_backpressure_latest.json"
DIVERGENCE_PATH="$PROJECT_ROOT/governance/health/data_source_divergence_latest.json"

cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
fi

export BOT_RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-$PROFILE}"

SESSION_TZ="${ONE_NUMBERS_SESSION_TIMEZONE:-${ONE_NUMBERS_REPORT_TIMEZONE:-America/New_York}}"
SESSION_START="${ONE_NUMBERS_SESSION_START:-09:30}"
SESSION_END="${ONE_NUMBERS_SESSION_END:-16:00}"
SESSION_INTERVAL="${ONE_NUMBERS_REFRESH_INTERVAL_SECONDS:-300}"
OFF_HOURS_INTERVAL="${ONE_NUMBERS_OFF_HOURS_REFRESH_INTERVAL_SECONDS:-3600}"

clock_now=(${(s: :)$(TZ="$SESSION_TZ" date "+%u %H %M")})
iso_weekday="${clock_now[1]:-7}"
hour_now="${clock_now[2]:-0}"
minute_now="${clock_now[3]:-0}"

age_seconds_for() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY'
from pathlib import Path
import sys
import time

path = Path(sys.argv[1])
if not path.exists():
    print(999999999)
else:
    print(max(int(time.time() - path.stat().st_mtime), 0))
PY
}

session_start_hour="${SESSION_START%%:*}"
session_start_minute="${SESSION_START##*:}"
session_end_hour="${SESSION_END%%:*}"
session_end_minute="${SESSION_END##*:}"

current_minutes=$(( 10#$hour_now * 60 + 10#$minute_now ))
start_minutes=$(( 10#${session_start_hour:-9} * 60 + 10#${session_start_minute:-30} ))
end_minutes=$(( 10#${session_end_hour:-16} * 60 + 10#${session_end_minute:-0} ))

session_open=0
if (( 10#$iso_weekday <= 5 )) && (( current_minutes >= start_minutes )) && (( current_minutes < end_minutes )); then
  session_open=1
fi

target_interval="$OFF_HOURS_INTERVAL"
if (( session_open == 1 )); then
  target_interval="$SESSION_INTERVAL"
fi

summary_age_seconds="$(age_seconds_for "$SUMMARY_PATH")"

guard_output=""
if ! guard_output="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/resource_guard.py" --profile refresh)"; then
  echo "one_numbers_refresh skip resource_guard_blocked session_open=$session_open detail=${guard_output:-resource_guard_blocked}"
  exit 0
fi

if (( summary_age_seconds >= ${target_interval:-300} )); then
  if ps -axo command | grep -q "[b]uild_one_numbers_report.py"; then
    echo "one_numbers_refresh skip refresh_already_running session_open=$session_open"
  else
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_one_numbers_report.py" --lightweight --no-sql-write
  fi
else
  echo "one_numbers_refresh skip age_seconds=$summary_age_seconds target_interval=$target_interval session_open=$session_open"
fi

backpressure_age_seconds="$(age_seconds_for "$BACKPRESSURE_PATH")"
if (( backpressure_age_seconds >= ${INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS:-300} )); then
  if ! ps -axo command | grep -q "[i]ngestion_backpressure_guard.py"; then
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/ingestion_backpressure_guard.py" --json || true
  else
    echo "ingestion_backpressure_refresh skip refresh_already_running age_seconds=$backpressure_age_seconds"
  fi
fi

divergence_age_seconds="$(age_seconds_for "$DIVERGENCE_PATH")"
if (( divergence_age_seconds >= ${DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS:-600} )); then
  if ! ps -axo command | grep -q "[d]ata_source_divergence_bot.py"; then
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/data_source_divergence_bot.py" --json || true
  else
    echo "data_source_divergence_refresh skip refresh_already_running age_seconds=$divergence_age_seconds"
  fi
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/health_gates.py" || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/session_ready_check.py" --json || true
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/ops/runtime_gate_dashboard.py" --json || true
