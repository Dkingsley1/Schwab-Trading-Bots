#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
OPSCTL="$PROJECT_ROOT/scripts/ops/opsctl.sh"
HEALTH_DIR="$PROJECT_ROOT/governance/health"
STATE_FILE="$HEALTH_DIR/startup_start_prompt_latest.json"
LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"

DELAY_SECONDS="${STARTUP_START_PROMPT_DELAY_SECONDS:-20}"
TIMEOUT_SECONDS="${STARTUP_START_PROMPT_TIMEOUT_SECONDS:-600}"
FORCE_RESTART="${STARTUP_START_PROMPT_FORCE_RESTART:-0}"
APPLY_PAPER_LOCK="${STARTUP_START_PROMPT_APPLY_PAPER_LOCK:-1}"
NO_BROWSER="${STARTUP_START_PROMPT_NO_BROWSER:-1}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --delay-seconds)
      DELAY_SECONDS="${2:-$DELAY_SECONDS}"
      shift
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="${2:-$TIMEOUT_SECONDS}"
      shift
      ;;
    --force-restart)
      FORCE_RESTART=1
      ;;
    --no-force-restart)
      FORCE_RESTART=0
      ;;
    --no-paper-lock)
      APPLY_PAPER_LOCK=0
      ;;
    --allow-browser)
      NO_BROWSER=0
      ;;
    --no-browser)
      NO_BROWSER=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      echo "unknown startup start prompt arg: $1" >&2
      exit 2
      ;;
  esac
  shift
done

normalize_int() {
  local value="${1:-0}"
  case "$value" in
    ''|*[!0-9]*)
      print -r -- "0"
      ;;
    *)
      print -r -- "$value"
      ;;
  esac
}

DELAY_SECONDS="$(normalize_int "$DELAY_SECONDS")"
TIMEOUT_SECONDS="$(normalize_int "$TIMEOUT_SECONDS")"
if [[ "$TIMEOUT_SECONDS" == "0" ]]; then
  TIMEOUT_SECONDS=600
fi

mkdir -p "$HEALTH_DIR" "$LOG_DIR"

write_state() {
  local state_status="$1"
  local decision="$2"
  local detail="${3:-}"
  local rc="${4:-0}"
  /usr/bin/python3 - "$STATE_FILE" "$state_status" "$decision" "$detail" "$rc" "$DRY_RUN" "$DELAY_SECONDS" "$TIMEOUT_SECONDS" "$FORCE_RESTART" "$APPLY_PAPER_LOCK" "$NO_BROWSER" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

state_path = Path(sys.argv[1])
payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "status": sys.argv[2],
    "decision": sys.argv[3],
    "detail": sys.argv[4],
    "return_code": int(sys.argv[5] or 0),
    "dry_run": sys.argv[6] == "1",
    "delay_seconds": int(sys.argv[7] or 0),
    "timeout_seconds": int(sys.argv[8] or 0),
    "force_restart": sys.argv[9] == "1",
    "paper_lock_applied_before_start": sys.argv[10] == "1",
    "no_browser_mode": sys.argv[11] == "1",
    "managed_by": "run_startup_start_prompt_launchd.sh",
}
state_path.parent.mkdir(parents=True, exist_ok=True)
state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

show_banner() {
  /usr/bin/osascript -e 'display notification "Choose Yes to start the guarded paper-trading stack, or No to leave it off." with title "Schwab Trading Bot" subtitle "Start at login?"' >/dev/null 2>&1 || true
}

ask_to_start() {
  set +e
  local result
  result="$(/usr/bin/osascript <<APPLESCRIPT
try
  set dialogResult to display dialog "Start schwab_trading_bot now?" buttons {"No", "Yes"} default button "Yes" cancel button "No" with title "Schwab Trading Bot" giving up after $TIMEOUT_SECONDS
  if gave up of dialogResult then
    return "timeout"
  end if
  return button returned of dialogResult
on error number -128
  return "No"
on error errText number errNum
  return "error:" & errNum & ":" & errText
end try
APPLESCRIPT
)"
  local rc=$?
  set -e
  if [[ "$rc" != "0" ]]; then
    print -r -- "error:$rc:osascript prompt failed"
    return 0
  fi
  print -r -- "$result"
}

start_stack() {
  local -a start_args
  start_args=(start)
  if [[ "$FORCE_RESTART" == "1" ]]; then
    start_args+=(--force-restart)
  fi
  if [[ "$APPLY_PAPER_LOCK" == "1" ]]; then
    "$OPSCTL" paper-lock --apply --json >/dev/null 2>&1 || true
  fi
  "$OPSCTL" "${start_args[@]}"
}

apply_no_browser_startup_env() {
  if [[ "$NO_BROWSER" != "1" ]]; then
    return 0
  fi
  export STARTUP_START_PROMPT_NO_BROWSER=1
  export SCHWAB_AUTH_BROWSER_DISABLED=1
  export SCHWAB_AUTH_ALLOW_BROWSER_OPEN=0
  export SCHWAB_AUTH_INTERACTIVE=0
  export SCHWAB_AUTH_REQUESTED_BROWSER=none
  export PREMARKET_TOKEN_BROWSER_AUTH_DISABLED=1
  export CHROME_HEADLESS_QUIET_MODE=1
  export REPORT_HEADLESS_BROWSER_RENDER_ENABLED=0
  export PROJECT_TIMELINE_AUTO_RENDER_PDF=0
  export PROJECT_TIMELINE_ALLOW_GUI_PDF_RENDERER=0
  export REPORT_PDF_BUNDLE_ALLOW_GUI_PDF_RENDERER=0
  export CRASH_REPORT_ALLOW_GUI_PDF_RENDERER=0
  export TRAINING_REPORT_ALLOW_GUI_PDF_RENDERER=0
  export BROWSER=/usr/bin/false
}

if [[ "$DRY_RUN" == "1" ]]; then
  apply_no_browser_startup_env
  write_state "dry_run" "not_prompted" "startup prompt dry-run completed without showing the dialog or starting the stack" 0
  echo "startup_start_prompt_status=dry_run state_file=$STATE_FILE"
  exit 0
fi

apply_no_browser_startup_env

if [[ "$DELAY_SECONDS" != "0" ]]; then
  sleep "$DELAY_SECONDS"
fi

show_banner
decision="$(ask_to_start)"

case "$decision" in
  Yes)
    write_state "starting" "yes" "operator accepted startup prompt" 0
    set +e
    start_stack
    rc=$?
    set -e
    if [[ "$rc" == "0" ]]; then
      write_state "started" "yes" "opsctl start completed" 0
    else
      write_state "failed" "yes" "opsctl start failed" "$rc"
    fi
    exit "$rc"
    ;;
  timeout)
    write_state "skipped" "timeout" "startup prompt timed out without starting the stack" 0
    echo "startup_start_prompt_status=skipped reason=timeout state_file=$STATE_FILE"
    ;;
  No)
    write_state "skipped" "no" "operator declined startup prompt" 0
    echo "startup_start_prompt_status=skipped reason=no state_file=$STATE_FILE"
    ;;
  error:*)
    write_state "prompt_unavailable" "error" "$decision" 0
    echo "startup_start_prompt_status=prompt_unavailable detail=$decision state_file=$STATE_FILE"
    ;;
  *)
    write_state "skipped" "unknown" "unexpected prompt response: $decision" 0
    echo "startup_start_prompt_status=skipped reason=unknown decision=$decision state_file=$STATE_FILE"
    ;;
esac
