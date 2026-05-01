#!/bin/zsh
set -uo pipefail

PROJECT_ROOT="/Users/dankingsley/PycharmProjects/schwab_trading_bot"
PYTHON_BIN="$PROJECT_ROOT/.venv312/bin/python"
EXPORT_SCRIPT="$PROJECT_ROOT/scripts/export_logs_to_csv.py"
AUTOMATION_LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
HEALTH_FILE="$PROJECT_ROOT/governance/health/finder_log_refresh_latest.json"
PUBLISH_DESKTOP_SHORTCUTS="${FINDER_LOG_REFRESH_DESKTOP_SHORTCUTS:-0}"
BOT_LOGS_FINDER_ALIAS_PATH="${BOT_LOGS_FINDER_ALIAS_PATH:-$HOME/bot_logs}"
BOT_LOGS_FINDER_DESKTOP_PATH="${BOT_LOGS_FINDER_DESKTOP_PATH:-$HOME/Desktop/Bot Logs}"
BOT_LOGS_FINDER_DESKTOP_SHORTCUTS="${BOT_LOGS_FINDER_DESKTOP_SHORTCUTS:-1}"

[[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]] && source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" live --quiet

resolve_external_project_root() {
  local configured_root="${BOT_LOGS_EXTERNAL_PROJECT_ROOT:-}"
  local mount_root="${BOT_LOGS_EXTERNAL_MOUNT:-/Volumes/BOT_LOGS}"
  local mount_candidates="${BOT_LOGS_EXTERNAL_MOUNT_CANDIDATES:-$mount_root}"
  local project_dir="${BOT_LOGS_EXTERNAL_PROJECT_DIR:-schwab_trading_bot}"
  local candidate
  if [[ -n "$configured_root" ]]; then
    if [[ -d "$configured_root" ]]; then
      printf '%s\n' "$configured_root"
      return 0
    fi
  fi

  IFS=',' read -rA _mount_roots <<< "$mount_candidates"
  for candidate in "${_mount_roots[@]}"; do
    [[ -n "$candidate" ]] || continue
    if [[ -d "$candidate/$project_dir" ]]; then
      printf '%s\n' "$candidate/$project_dir"
      return 0
    fi
  done

  if [[ -n "$configured_root" ]]; then
    printf '%s\n' "$configured_root"
    return 0
  fi

  printf '%s\n' "$mount_root/$project_dir"
}

resolve_external_min_free_bytes() {
  local raw_bytes="${FINDER_LOG_REFRESH_EXTERNAL_MIN_FREE_BYTES:-${BOT_LOGS_EXTERNAL_MIN_FREE_BYTES:-}}"
  local raw_gb="${FINDER_LOG_REFRESH_EXTERNAL_MIN_FREE_GB:-${BOT_LOGS_EXTERNAL_MIN_FREE_GB:-0}}"
  if [[ -n "$raw_bytes" ]]; then
    printf '%s\n' "$raw_bytes"
    return 0
  fi
  awk "BEGIN {printf \"%d\", (($raw_gb+0) * 1024 * 1024 * 1024)}"
}

resolve_output_dir() {
  local prefer_external="${FINDER_LOG_REFRESH_PREFER_EXTERNAL:-1}"
  local external_root
  local free_kb
  local free_bytes
  local min_free_bytes

  if [[ "$prefer_external" != "1" ]]; then
    printf '%s\n' "$PROJECT_ROOT/exports/csv"
    return 0
  fi

  external_root="$(resolve_external_project_root)"
  if [[ ! -d "$external_root" || ! -w "$external_root" ]]; then
    printf '%s\n' "$PROJECT_ROOT/exports/csv"
    return 0
  fi

  min_free_bytes="$(resolve_external_min_free_bytes)"
  if [[ "${min_free_bytes:-0}" != "0" ]]; then
    free_kb="$(df -Pk "$external_root" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [[ -n "$free_kb" ]]; then
      free_bytes=$(( free_kb * 1024 ))
      if (( free_bytes < min_free_bytes )); then
        printf '%s\n' "$PROJECT_ROOT/exports/csv"
        return 0
      fi
    fi
  fi

  printf '%s\n' "$external_root/exports/csv"
}

OUT_DIR="$(resolve_output_dir)"
STORAGE_TARGET="project_exports"
if [[ "$OUT_DIR" == "$(resolve_external_project_root)"/* ]]; then
  STORAGE_TARGET="external_bot_logs"
fi
ACTIVE_LOG_DIR="$PROJECT_ROOT/logs"
BOT_LOGS_ALIAS_TARGET="$ACTIVE_LOG_DIR"
BOT_LOGS_ALIAS_OK=0
BOT_LOGS_DESKTOP_SHORTCUT_OK=0

mkdir -p "$AUTOMATION_LOG_DIR"
mkdir -p "$(dirname "$HEALTH_FILE")"
mkdir -p "$OUT_DIR"

DATE_UTC="$(date -u +%Y%m%d)"
TIMESTAMP_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

refresh_ok=1
error_msg=""

if ! "$PYTHON_BIN" "$EXPORT_SCRIPT" --date "$DATE_UTC" --out-dir "$OUT_DIR" --latest-aliases >> "$AUTOMATION_LOG_DIR/finder_refresh.log" 2>&1; then
  refresh_ok=0
  error_msg="export_logs_to_csv_failed"
fi

mkdir -p "$(dirname "$BOT_LOGS_FINDER_ALIAS_PATH")"
if ln -sfn "$BOT_LOGS_ALIAS_TARGET" "$BOT_LOGS_FINDER_ALIAS_PATH"; then
  BOT_LOGS_ALIAS_OK=1
else
  refresh_ok=0
  error_msg="${error_msg:-bot_logs_alias_failed}"
fi

if [[ "$PUBLISH_DESKTOP_SHORTCUTS" == "1" ]]; then
  mkdir -p "$HOME/Desktop/Sub Logs" "$HOME/Desktop/Master Logs"
  ln -sfn "$OUT_DIR/latest_decision_explanations.csv" "$HOME/Desktop/Sub Logs/latest_decision_explanations.csv" || true
  ln -sfn "$OUT_DIR/latest_master_control.csv" "$HOME/Desktop/Master Logs/latest_master_control.csv" || true
fi

if [[ "$BOT_LOGS_FINDER_DESKTOP_SHORTCUTS" == "1" ]]; then
  mkdir -p "$(dirname "$BOT_LOGS_FINDER_DESKTOP_PATH")"
  if ln -sfn "$BOT_LOGS_ALIAS_TARGET" "$BOT_LOGS_FINDER_DESKTOP_PATH"; then
    BOT_LOGS_DESKTOP_SHORTCUT_OK=1
  fi
fi

cat > "$HEALTH_FILE" <<JSON
{"timestamp_utc":"$TIMESTAMP_UTC","ok":$refresh_ok,"date_utc":"$DATE_UTC","error":"$error_msg","publish_desktop_shortcuts":$([[ "$PUBLISH_DESKTOP_SHORTCUTS" == "1" ]] && echo true || echo false),"publish_bot_logs_desktop_shortcut":$([[ "$BOT_LOGS_FINDER_DESKTOP_SHORTCUTS" == "1" ]] && echo true || echo false),"out_dir":"$OUT_DIR","storage_target":"$STORAGE_TARGET","bot_logs_alias_path":"$BOT_LOGS_FINDER_ALIAS_PATH","bot_logs_alias_target":"$BOT_LOGS_ALIAS_TARGET","bot_logs_alias_ok":$([[ "$BOT_LOGS_ALIAS_OK" == "1" ]] && echo true || echo false),"bot_logs_desktop_shortcut_path":"$BOT_LOGS_FINDER_DESKTOP_PATH","bot_logs_desktop_shortcut_ok":$([[ "$BOT_LOGS_DESKTOP_SHORTCUT_OK" == "1" ]] && echo true || echo false)}
JSON

if [[ "$refresh_ok" != "1" ]]; then
  echo "finder_log_refresh_warn error=$error_msg date_utc=$DATE_UTC" >> "$AUTOMATION_LOG_DIR/finder_refresh.log"
fi

exit 0
