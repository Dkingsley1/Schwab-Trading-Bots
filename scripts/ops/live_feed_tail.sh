#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WATCHDOG_LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
MEMORY_OVERRIDE_FILE="$PROJECT_ROOT/config/.env.memory_efficiency_override"
PRESSURE_OVERRIDE_FILE="$PROJECT_ROOT/config/.env.pressure_relief_override"
HEALTH_DIR="$PROJECT_ROOT/governance/health"
HEAVY_MARKER_FILE="$HEALTH_DIR/live_feed_heavy_view_latest.json"
LIVEFEED_HEALTH_FILE="$HEALTH_DIR/livefeed_local_latest.json"
LIVE_FEED_MAIN_PID="$$"

if [[ -f "$MEMORY_OVERRIDE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$MEMORY_OVERRIDE_FILE"
fi
if [[ -f "$PRESSURE_OVERRIDE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$PRESSURE_OVERRIDE_FILE"
fi

LIVEFEED_HEALTH_WRITER="${LIVEFEED_HEALTH_WRITER:-0}"
case "${LIVEFEED_HEALTH_WRITER:l}" in
  1|true|yes|on)
    LIVEFEED_HEALTH_WRITER="1"
    ;;
  *)
    LIVEFEED_HEALTH_WRITER="0"
    ;;
esac

SOURCE="schwab"
SYMBOL=""
LINES="40"
LINES_EXPLICIT="0"
RAW="0"
SNAPSHOT="0"
INCLUDE_DECISIONS="${LIVE_FEED_INCLUDE_DECISIONS_DEFAULT:-0}"
MEMORY_AWARE="${LIVE_FEED_MEMORY_AWARE_DEFAULT:-1}"
DECISION_FILE_MODE="${LIVE_FEED_DECISION_FILE_MODE:-day_plus_latest}"
INCLUDE_WATCHDOG_LOG="${LIVE_FEED_INCLUDE_WATCHDOG_LOG_DEFAULT:-0}"
INCLUDE_WATCHDOG_LOG_EXPLICIT="0"
INCLUDE_COINBASE_WATCHDOG_LOG="${LIVE_FEED_INCLUDE_COINBASE_WATCHDOG_LOG:-1}"
STATUS_SNAPSHOT="${LIVE_FEED_STATUS_SNAPSHOT_DEFAULT:-1}"
SHOW_FILE_LIST="${LIVE_FEED_SHOW_FILE_LIST_DEFAULT:-0}"
SUPPRESS_FUTURES_SPECIALIST_INTENTS="${LIVE_FEED_SUPPRESS_FUTURES_SPECIALIST_INTENTS_DEFAULT:-1}"
SUPPRESS_JSON_FRAGMENTS="${LIVE_FEED_SUPPRESS_JSON_FRAGMENTS_DEFAULT:-1}"
SUPPRESS_TAIL_HEADERS="${LIVE_FEED_SUPPRESS_TAIL_HEADERS_DEFAULT:-1}"
DEDUP_REPEATED_LINES="${LIVE_FEED_DEDUP_REPEATED_LINES_DEFAULT:-1}"
SHOW_KEEPALIVE="${LIVE_FEED_SHOW_KEEPALIVE_DEFAULT:-0}"
SHOW_KEEPALIVE_EXPLICIT="0"
VISIBLE_KEEPALIVE_ALLOWED="${LIVE_FEED_VISIBLE_KEEPALIVE_ALLOWED:-0}"
IMPORTANT_ONLY="${LIVE_FEED_IMPORTANT_ONLY_DEFAULT:-auto}"
HEAVY_REQUESTED="${LIVE_FEED_HEAVY_DEFAULT:-0}"
COLOR_MODE="${LIVE_FEED_COLOR:-auto}"
COLOR_PALETTE="${LIVE_FEED_COLOR_PALETTE:-semantic}"
HEAVY_TTL_ENABLED_OVERRIDE=""
HEAVY_TTL_SECONDS_OVERRIDE=""
FOLLOW_RESTART_COUNT=0
FOLLOW_LAST_RC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="${2:-}"
      shift 2
      ;;
    --symbol)
      SYMBOL="${2:-}"
      shift 2
      ;;
    --lines)
      LINES="${2:-40}"
      LINES_EXPLICIT="1"
      shift 2
      ;;
    --raw)
      RAW="1"
      shift
      ;;
    --color|--highlight)
      COLOR_MODE="always"
      shift
      ;;
    --no-color|--no-highlight)
      COLOR_MODE="never"
      shift
      ;;
    --red-only|--red)
      COLOR_PALETTE="red"
      shift
      ;;
    --red-actions|--red-action-color|--red-action-colors)
      COLOR_PALETTE="red_actions"
      shift
      ;;
    --semantic-color|--semantic-colors)
      COLOR_PALETTE="semantic"
      shift
      ;;
    --snapshot)
      SNAPSHOT="1"
      shift
      ;;
    --include-decisions)
      INCLUDE_DECISIONS="1"
      shift
      ;;
    --heavy)
      HEAVY_REQUESTED="1"
      INCLUDE_DECISIONS="1"
      shift
      ;;
    --no-decisions)
      INCLUDE_DECISIONS="0"
      shift
      ;;
    --memory-aware)
      MEMORY_AWARE="1"
      shift
      ;;
    --no-memory-aware)
      MEMORY_AWARE="0"
      shift
      ;;
    --include-watchdog-log)
      INCLUDE_WATCHDOG_LOG="1"
      INCLUDE_WATCHDOG_LOG_EXPLICIT="1"
      shift
      ;;
    --no-watchdog-log)
      INCLUDE_WATCHDOG_LOG="0"
      INCLUDE_WATCHDOG_LOG_EXPLICIT="1"
      shift
      ;;
    --status-snapshot)
      STATUS_SNAPSHOT="1"
      shift
      ;;
    --no-status-snapshot)
      STATUS_SNAPSHOT="0"
      shift
      ;;
    --show-files)
      SHOW_FILE_LIST="1"
      shift
      ;;
    --hide-files)
      SHOW_FILE_LIST="0"
      shift
      ;;
    --show-futures-specialist-intents)
      SUPPRESS_FUTURES_SPECIALIST_INTENTS="0"
      shift
      ;;
    --hide-futures-specialist-intents)
      SUPPRESS_FUTURES_SPECIALIST_INTENTS="1"
      shift
      ;;
    --show-json-fragments)
      SUPPRESS_JSON_FRAGMENTS="0"
      shift
      ;;
    --hide-json-fragments)
      SUPPRESS_JSON_FRAGMENTS="1"
      shift
      ;;
    --show-tail-headers)
      SUPPRESS_TAIL_HEADERS="0"
      shift
      ;;
    --hide-tail-headers)
      SUPPRESS_TAIL_HEADERS="1"
      shift
      ;;
    --dedupe-repeats)
      DEDUP_REPEATED_LINES="1"
      shift
      ;;
    --no-dedupe-repeats)
      DEDUP_REPEATED_LINES="0"
      shift
      ;;
    --show-keepalive)
      SHOW_KEEPALIVE="1"
      SHOW_KEEPALIVE_EXPLICIT="1"
      shift
      ;;
    --hide-keepalive)
      SHOW_KEEPALIVE="0"
      SHOW_KEEPALIVE_EXPLICIT="1"
      shift
      ;;
    --important-only|--operator-important-only)
      IMPORTANT_ONLY="1"
      shift
      ;;
    --all-feed-events|--no-important-only)
      IMPORTANT_ONLY="0"
      shift
      ;;
    --heavy-ttl)
      HEAVY_TTL_ENABLED_OVERRIDE="1"
      shift
      ;;
    --no-heavy-ttl|--disable-heavy-ttl)
      HEAVY_TTL_ENABLED_OVERRIDE="0"
      HEAVY_TTL_SECONDS_OVERRIDE="0"
      shift
      ;;
    --heavy-ttl-seconds)
      HEAVY_TTL_ENABLED_OVERRIDE="1"
      HEAVY_TTL_SECONDS_OVERRIDE="${2:-0}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/ops/live_feed_tail.sh [--source schwab|coinbase|fx|futures|schwab_futures|coinbase_futures|main|infra|all] [--symbol SYMBOL] [--lines 40] [--raw] [--color|--no-color] [--red-only|--red-actions|--semantic-color] [--snapshot] [--include-decisions|--heavy] [--memory-aware|--no-memory-aware] [--include-watchdog-log|--no-watchdog-log] [--status-snapshot|--no-status-snapshot] [--show-files|--hide-files] [--show-futures-specialist-intents|--hide-futures-specialist-intents] [--show-json-fragments|--hide-json-fragments] [--show-tail-headers|--hide-tail-headers] [--dedupe-repeats|--no-dedupe-repeats] [--show-keepalive|--hide-keepalive] [--important-only|--all-feed-events] [--heavy-ttl|--no-heavy-ttl|--heavy-ttl-seconds N]

Examples:
  scripts/ops/live_feed_tail.sh
  scripts/ops/live_feed_tail.sh --symbol SPY
  scripts/ops/live_feed_tail.sh --source coinbase
  scripts/ops/live_feed_tail.sh --source fx
  scripts/ops/live_feed_tail.sh --source futures
  scripts/ops/live_feed_tail.sh --source infra --heavy
  scripts/ops/live_feed_tail.sh --source main --lines 80
  scripts/ops/live_feed_tail.sh --source all --lines 80
  scripts/ops/live_feed_tail.sh --source all --heavy
  scripts/ops/live_feed_tail.sh --source all --heavy --red-only
  scripts/ops/live_feed_tail.sh --source all --heavy --red-actions
  scripts/ops/live_feed_tail.sh --source all --heavy --color
  scripts/ops/live_feed_tail.sh --source all --heavy --no-memory-aware
  scripts/ops/live_feed_tail.sh --source all --lines 80 --snapshot
  scripts/ops/live_feed_tail.sh --source all --snapshot --include-watchdog-log
  scripts/ops/live_feed_tail.sh --source all --heavy --no-status-snapshot
  scripts/ops/live_feed_tail.sh --source all --heavy --show-files
  scripts/ops/live_feed_tail.sh --source all --heavy --show-files --no-heavy-ttl
  scripts/ops/live_feed_tail.sh --source all --heavy --show-futures-specialist-intents
  scripts/ops/live_feed_tail.sh --source all --heavy --show-json-fragments --show-tail-headers --all-feed-events
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if ! [[ "$LINES" =~ ^[0-9]+$ ]]; then
  echo "--lines must be an integer" >&2
  exit 2
fi

if [[ "$SOURCE" != "schwab" && "$SOURCE" != "coinbase" && "$SOURCE" != "fx" && "$SOURCE" != "futures" && "$SOURCE" != "schwab_futures" && "$SOURCE" != "coinbase_futures" && "$SOURCE" != "main" && "$SOURCE" != "infra" && "$SOURCE" != "all" ]]; then
  echo "--source must be schwab, coinbase, fx, futures, schwab_futures, coinbase_futures, main, infra, or all" >&2
  exit 2
fi

COLOR_ENABLED="0"
case "$COLOR_MODE" in
  always|1|true|yes|on)
    COLOR_ENABLED="1"
    ;;
  never|0|false|no|off)
    COLOR_ENABLED="0"
    ;;
  auto|"")
    if [[ "$RAW" != "1" && ( -t 1 || "$HEAVY_REQUESTED" == "1" ) && ( -z "${NO_COLOR:-}" || "$HEAVY_REQUESTED" == "1" ) ]]; then
      COLOR_ENABLED="1"
    fi
    ;;
  *)
    echo "--color mode must be auto, always, or never" >&2
    exit 2
    ;;
esac
if [[ "$RAW" == "1" ]]; then
  COLOR_ENABLED="0"
fi
case "$COLOR_PALETTE" in
  red|red_only|red-only|red_actions|red-actions|mono|monochrome|semantic)
    ;;
  *)
    echo "--color palette must be red, red-actions, or semantic" >&2
    exit 2
    ;;
esac

if [[ "$SOURCE" == "all" && "$INCLUDE_DECISIONS" == "1" ]]; then
  HEAVY_REQUESTED="1"
fi

case "${IMPORTANT_ONLY:l}" in
  auto|"")
    IMPORTANT_ONLY="auto"
    ;;
  1|true|yes|on)
    IMPORTANT_ONLY="1"
    ;;
  0|false|no|off)
    IMPORTANT_ONLY="0"
    ;;
  *)
    echo "--important-only mode must be auto, on, or off" >&2
    exit 2
    ;;
esac
if [[ "$IMPORTANT_ONLY" == "auto" ]]; then
  if [[ "$HEAVY_REQUESTED" == "1" ]]; then
    IMPORTANT_ONLY="1"
  else
    IMPORTANT_ONLY="0"
  fi
fi
case "${VISIBLE_KEEPALIVE_ALLOWED:l}" in
  1|true|yes|on)
    VISIBLE_KEEPALIVE_ALLOWED="1"
    ;;
  *)
    VISIBLE_KEEPALIVE_ALLOWED="0"
    ;;
esac

if [[ "$HEAVY_REQUESTED" == "1" ]]; then
  INCLUDE_DECISIONS="1"
  if [[ "$INCLUDE_WATCHDOG_LOG_EXPLICIT" != "1" && "$IMPORTANT_ONLY" != "1" ]]; then
    INCLUDE_WATCHDOG_LOG="1"
  fi
fi

MEMORY_PROFILE="${BOT_MEMORY_EFFICIENCY_PROFILE:-}"
HEAVY_DEFAULT_LINES="${LIVE_FEED_HEAVY_DEFAULT_LINES:-120}"
HEAVY_PRESSURE_LINES="${LIVE_FEED_HEAVY_PRESSURE_LINES:-80}"
DECISION_FILE_MODE_PRESSURE="${LIVE_FEED_DECISION_FILE_MODE_PRESSURE:-latest_only}"
HEAVY_SELF_THROTTLE="${LIVE_FEED_HEAVY_SELF_THROTTLE:-1}"
HEAVY_NICE="${LIVE_FEED_HEAVY_NICE:-10}"
HEAVY_BACKGROUND_POLICY="${LIVE_FEED_HEAVY_BACKGROUND_POLICY:-1}"
HEAVY_INCLUDE_ALL_DECISION_DIRS="${LIVE_FEED_HEAVY_INCLUDE_ALL_DECISION_DIRS:-0}"
HEAVY_MAX_FOLLOW_FILES="${LIVE_FEED_HEAVY_MAX_FOLLOW_FILES:-36}"
HEAVY_TAIL_BYTES="${LIVE_FEED_HEAVY_TAIL_BYTES:-262144}"
HEAVY_BOOTSTRAP_SNAPSHOT="${LIVE_FEED_HEAVY_BOOTSTRAP_SNAPSHOT:-1}"
HEAVY_BOOTSTRAP_MAX_LINES="${LIVE_FEED_HEAVY_BOOTSTRAP_MAX_LINES:-80}"
HEAVY_SNAPSHOT_MAX_LINES="${LIVE_FEED_HEAVY_SNAPSHOT_MAX_LINES:-180}"
HEAVY_VISIBLE_KEEPALIVE_DEFAULT="${LIVE_FEED_HEAVY_VISIBLE_KEEPALIVE_DEFAULT:-1}"
HEAVY_KEEPALIVE_SECONDS_DEFAULT="${LIVE_FEED_HEAVY_KEEPALIVE_SECONDS_DEFAULT:-5}"
KEEPALIVE_STATUS_EVERY="${LIVE_FEED_KEEPALIVE_STATUS_EVERY:-4}"
KEEPALIVE_DECISION_SNAPSHOT="${LIVE_FEED_KEEPALIVE_DECISION_SNAPSHOT:-1}"
KEEPALIVE_DECISION_EVERY="${LIVE_FEED_KEEPALIVE_DECISION_EVERY:-1}"
DECISION_SNAPSHOT_MAX_LINES="${LIVE_FEED_DECISION_SNAPSHOT_MAX_LINES:-4}"
DECISION_SNAPSHOT_TAIL_BYTES="${LIVE_FEED_DECISION_SNAPSHOT_TAIL_BYTES:-4194304}"
HEAVY_TTL_ENABLED="${LIVE_FEED_HEAVY_TTL_ENABLED:-0}"
HEAVY_TTL_SECONDS="${LIVE_FEED_HEAVY_TTL_SECONDS:-0}"
if [[ -n "$HEAVY_TTL_ENABLED_OVERRIDE" ]]; then
  HEAVY_TTL_ENABLED="$HEAVY_TTL_ENABLED_OVERRIDE"
fi
if [[ -n "$HEAVY_TTL_SECONDS_OVERRIDE" ]]; then
  HEAVY_TTL_SECONDS="$HEAVY_TTL_SECONDS_OVERRIDE"
fi
MAX_LINE_CHARS="${LIVE_FEED_MAX_LINE_CHARS:-1400}"
KEEPALIVE_ENABLED="${LIVE_FEED_KEEPALIVE_ENABLED:-1}"
KEEPALIVE_SECONDS="${LIVE_FEED_KEEPALIVE_SECONDS:-15}"
STARTUP_STATUS_ENABLED="${LIVE_FEED_STARTUP_STATUS_ENABLED:-1}"
DECISION_MAX_AGE_HOURS="${LIVE_FEED_DECISION_MAX_AGE_HOURS:-48}"
PRESSURE_OPTIMIZED="0"

if [[ "$HEAVY_REQUESTED" == "1" && "$LINES_EXPLICIT" != "1" ]]; then
  LINES="$HEAVY_DEFAULT_LINES"
fi

if [[ "$HEAVY_REQUESTED" == "1" && "$MEMORY_AWARE" == "1" ]]; then
  case "$MEMORY_PROFILE" in
    constrained|air_safe)
      if [[ "$LINES_EXPLICIT" != "1" ]]; then
        LINES="$HEAVY_PRESSURE_LINES"
      fi
      DECISION_FILE_MODE="$DECISION_FILE_MODE_PRESSURE"
      PRESSURE_OPTIMIZED="1"
      ;;
  esac
fi

if [[ "$HEAVY_REQUESTED" == "1" && "$HEAVY_SELF_THROTTLE" == "1" ]]; then
  renice -n "$HEAVY_NICE" -p "$LIVE_FEED_MAIN_PID" >/dev/null 2>&1 || true
  if [[ "$HEAVY_BACKGROUND_POLICY" == "1" ]] && command -v taskpolicy >/dev/null 2>&1; then
    taskpolicy -b -p "$LIVE_FEED_MAIN_PID" >/dev/null 2>&1 || true
  fi
fi

if ! [[ "$HEAVY_TAIL_BYTES" =~ ^[0-9]+$ ]]; then
  HEAVY_TAIL_BYTES="262144"
fi
if ! [[ "$MAX_LINE_CHARS" =~ ^[0-9]+$ ]]; then
  MAX_LINE_CHARS="1400"
fi
if ! [[ "$HEAVY_BOOTSTRAP_MAX_LINES" =~ ^[0-9]+$ ]]; then
  HEAVY_BOOTSTRAP_MAX_LINES="80"
fi
if ! [[ "$HEAVY_SNAPSHOT_MAX_LINES" =~ ^[0-9]+$ ]]; then
  HEAVY_SNAPSHOT_MAX_LINES="180"
fi
if ! [[ "$HEAVY_TTL_SECONDS" =~ ^[0-9]+$ ]]; then
  HEAVY_TTL_SECONDS="0"
fi
if ! [[ "$KEEPALIVE_SECONDS" =~ ^[0-9]+$ ]]; then
  KEEPALIVE_SECONDS="15"
fi
if [[ "$HEAVY_REQUESTED" == "1" && -z "${LIVE_FEED_KEEPALIVE_SECONDS+x}" ]]; then
  if [[ "$HEAVY_KEEPALIVE_SECONDS_DEFAULT" =~ ^[0-9]+$ && "$HEAVY_KEEPALIVE_SECONDS_DEFAULT" -gt 0 ]]; then
    KEEPALIVE_SECONDS="$HEAVY_KEEPALIVE_SECONDS_DEFAULT"
  fi
fi
if ! [[ "$KEEPALIVE_STATUS_EVERY" =~ ^[0-9]+$ ]]; then
  KEEPALIVE_STATUS_EVERY="4"
fi
if ! [[ "$KEEPALIVE_DECISION_EVERY" =~ ^[0-9]+$ ]]; then
  KEEPALIVE_DECISION_EVERY="1"
fi
if ! [[ "$DECISION_SNAPSHOT_MAX_LINES" =~ ^[0-9]+$ ]]; then
  DECISION_SNAPSHOT_MAX_LINES="4"
fi
if ! [[ "$DECISION_SNAPSHOT_TAIL_BYTES" =~ ^[0-9]+$ ]]; then
  DECISION_SNAPSHOT_TAIL_BYTES="4194304"
fi
if ! [[ "$DECISION_MAX_AGE_HOURS" =~ ^[0-9]+$ ]]; then
  DECISION_MAX_AGE_HOURS="48"
fi

if [[ "$HEAVY_REQUESTED" == "1" && "$SHOW_KEEPALIVE_EXPLICIT" != "1" ]]; then
  case "${HEAVY_VISIBLE_KEEPALIVE_DEFAULT:l}" in
    1|true|yes|on)
      SHOW_KEEPALIVE="1"
      if [[ -z "${LIVE_FEED_VISIBLE_KEEPALIVE_ALLOWED+x}" ]]; then
        VISIBLE_KEEPALIVE_ALLOWED="1"
      fi
      ;;
  esac
fi

if [[ "$SNAPSHOT" != "1" && "$STARTUP_STATUS_ENABLED" == "1" ]]; then
  echo "live_feed_starting source=$SOURCE heavy=$HEAVY_REQUESTED important_only=$IMPORTANT_ONLY include_decisions=$INCLUDE_DECISIONS memory_profile=${MEMORY_PROFILE:-default} memory_aware=$MEMORY_AWARE max_follow_files=$HEAVY_MAX_FOLLOW_FILES"
fi

DAY_UTC="$(date -u +%Y%m%d)"
DAY_LOCAL="$(date +%Y%m%d)"

typeset -a files
typeset -a skipped_files
typeset -a skipped_file_reasons
typeset -A seen
typeset -A skipped_seen

tail_probe_ok() {
  local f="$1"
  tail -n 0 "$f" >/dev/null 2>&1
}

skip_file() {
  local f="$1"
  local reason="$2"
  [[ -n "$f" && -z "${skipped_seen[$f]:-}" ]] || return 0
  skipped_files+=("$f")
  skipped_file_reasons+=("$reason")
  skipped_seen[$f]=1
}

append_file() {
  local f="$1"
  [[ -f "$f" && -z "${seen[$f]:-}" ]] || return 0
  if ! tail_probe_ok "$f"; then
    skip_file "$f" "tail_unreadable"
    return 0
  fi
  files+=("$f")
  seen[$f]=1
}

append_decision_file() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  if [[ "$HEAVY_REQUESTED" == "1" && "$DECISION_MAX_AGE_HOURS" -gt 0 ]]; then
    local mtime now age max_age
    mtime="$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null || printf 0)"
    now="$(date +%s)"
    age=$((now - mtime))
    max_age=$((DECISION_MAX_AGE_HOURS * 3600))
    if [[ "$mtime" -gt 0 && "$age" -gt "$max_age" ]]; then
      return 0
    fi
  fi
  append_file "$f"
}

latest_log() {
  local pattern="$1"
  local -a matches
  setopt localoptions extendedglob
  # ${~pattern} forces zsh glob expansion from a variable.
  matches=(${~pattern}(N.om[1]))
  if (( ${#matches} > 0 )); then
    print -r -- "$matches[1]"
  fi
}

append_decision_json_dir() {
  local dir="$1"
  if [[ "$DECISION_FILE_MODE" == "latest_only" ]]; then
    append_decision_file "$(latest_log "$PROJECT_ROOT/decision_explanations/$dir/decision_explanations_*.jsonl")"
    return
  fi
  append_decision_file "$PROJECT_ROOT/decision_explanations/$dir/decision_explanations_${DAY_LOCAL}.jsonl"
  append_decision_file "$PROJECT_ROOT/decision_explanations/$dir/decision_explanations_${DAY_UTC}.jsonl"
  append_decision_file "$(latest_log "$PROJECT_ROOT/decision_explanations/$dir/decision_explanations_*.jsonl")"
}

append_decision_channel_dir() {
  local dir="$1"
  if [[ "$DECISION_FILE_MODE" == "latest_only" ]]; then
    append_decision_file "$(latest_log "$PROJECT_ROOT/governance/channels/decision/$dir/decision_*.jsonl")"
    return
  fi
  append_decision_file "$PROJECT_ROOT/governance/channels/decision/$dir/decision_${DAY_LOCAL}.jsonl"
  append_decision_file "$PROJECT_ROOT/governance/channels/decision/$dir/decision_${DAY_UTC}.jsonl"
  append_decision_file "$(latest_log "$PROJECT_ROOT/governance/channels/decision/$dir/decision_*.jsonl")"
}

append_trade_decision_dir() {
  local dir="$1"
  if [[ "$DECISION_FILE_MODE" == "latest_only" ]]; then
    append_decision_file "$(latest_log "$PROJECT_ROOT/local_fallback_storage/decisions/$dir/trade_decisions_*.jsonl")"
    return
  fi
  append_decision_file "$PROJECT_ROOT/local_fallback_storage/decisions/$dir/trade_decisions_${DAY_LOCAL}.jsonl"
  append_decision_file "$PROJECT_ROOT/local_fallback_storage/decisions/$dir/trade_decisions_${DAY_UTC}.jsonl"
  append_decision_file "$(latest_log "$PROJECT_ROOT/local_fallback_storage/decisions/$dir/trade_decisions_*.jsonl")"
}

append_all_decision_json_dirs() {
  setopt localoptions null_glob
  local dir_path
  for dir_path in "$PROJECT_ROOT"/decision_explanations/*(/N); do
    append_decision_json_dir "${dir_path:t}"
  done
}

append_all_decision_channel_dirs() {
  setopt localoptions null_glob
  local dir_path
  for dir_path in "$PROJECT_ROOT"/governance/channels/decision/*(/N); do
    append_decision_channel_dir "${dir_path:t}"
  done
}

append_health_file() {
  local f="$1"
  append_file "$PROJECT_ROOT/governance/health/$f"
}

append_heavy_health_files() {
  local f
  for f in \
    one_numbers_latest.json \
    one_numbers_rollup_history.json \
    health_gates_latest.json \
    global_killswitch_latest.json \
    auth_lease_manager_latest.json \
    schwab_auth_supervisor_latest.json \
    coinbase_api_health_latest.json \
    process_watchdog_latest.json \
    broker_readiness_latest.json \
    paper_live_data_standard_latest.json \
    execution_lane_paper_latest.json \
    paper_runtime_profitability_controls_latest.json \
    paper_profitability_control_latest.json \
    paper_performance_latest.json \
    paper_execution_truth_layer_latest.json \
    paper_execution_calibration_latest.json \
    runtime_paper_regression_guard_latest.json \
    paper_400_ramp_latest.json \
    mac_notification_watch_state.json \
    notification_escalation_ladder_latest.json \
    remote_alert_control_latest.json \
    data_plane_recovery_controller_latest.json \
    live_runtime_separation_control_latest.json \
    ingestion_backpressure_latest.json \
    ingestion_storage_control_latest.json \
    runtime_throttle_control_latest.json \
    coordination_state_latest.json \
    storage_pressure_clearance_latest.json \
    storage_backpressure_autopilot_latest.json \
    storage_failback_sync_latest.json \
    storage_mount_guard_latest.json \
    storage_disaster_recovery_latest.json \
    storage_resilience_latest.json \
    command_validity_latest.json \
    commands_hygiene_latest.json \
    master_infrastructure_supervisor_latest.json \
    system_drift_guard_latest.json \
    system_drift_autopilot_latest.json \
    runtime_gate_dashboard_latest.json \
    operator_cockpit_latest.json \
    macro_event_intelligence_latest.json \
    spacex_ipo_downside_watch_latest.json \
    spacex_ipo_downside_watch_state.json \
    training_runtime_control_latest.json \
    training_quality_control_latest.json \
    training_report_latest.json \
    training_success_latest.json \
    one_numbers_regression_guard_latest.json \
    chrome_headless_guard_latest.json \
    incident_review_packet_latest.json \
    incident_closeout_autopilot_latest.json; do
    append_health_file "$f"
  done
  append_file "$PROJECT_ROOT/data/external_context/live_macro_latest.json"
  append_file "$(latest_log "$PROJECT_ROOT/logs/all_sleeves_*.log")"
  append_file "$(latest_log "$PROJECT_ROOT/logs/shadow_watchdog*.log")"
  if [[ "$INCLUDE_WATCHDOG_LOG" == "1" ]]; then
    append_file "$WATCHDOG_LOG_DIR/shadow_watchdog.out.log"
  fi
}

if [[ "$INCLUDE_DECISIONS" == "1" && ( "$SOURCE" == "all" || "$HEAVY_REQUESTED" == "1" ) ]]; then
  append_trade_decision_dir "paper"
fi

if [[ "$SOURCE" == "schwab" || "$SOURCE" == "main" || "$SOURCE" == "all" ]]; then
  if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
    for d in aggressive_equities_schwab conservative_equities_schwab dividend_equities_schwab bond_equities_schwab; do
      append_decision_channel_dir "$d"
    done
    for d in shadow_equities shadow_aggressive_equities shadow_conservative_equities shadow_dividend_equities shadow_dividend_capture_equities shadow_bond_equities shadow_intraday_aggressive_equities shadow_swing_aggressive_equities; do
      append_decision_json_dir "$d"
    done
  fi
  append_file "$(latest_log "$PROJECT_ROOT/logs/schwab_live_*.log")"
fi

if [[ "$SOURCE" == "schwab" || "$SOURCE" == "futures" || "$SOURCE" == "schwab_futures" || "$SOURCE" == "main" || "$SOURCE" == "all" ]]; then
  if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
    append_decision_channel_dir "schwab_futures_equities_schwab"
    append_decision_json_dir "shadow_schwab_futures_equities"
  fi
  append_file "$(latest_log "$PROJECT_ROOT/logs/schwab_futures_live_*.log")"
  append_health_file "data_ingress_latest_schwab_futures_equities_schwab.json"
  append_health_file "broker_truth_schwab_futures_equities_schwab_latest.json"
  append_file "$(latest_log "$PROJECT_ROOT/governance/health/shadow_loop_schwab_futures_equities_schwab_*.json")"
fi

if [[ "$SOURCE" == "coinbase" || "$SOURCE" == "main" || "$SOURCE" == "all" ]]; then
  if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
    append_decision_channel_dir "default_crypto_schwab"
    append_trade_decision_dir "shadow_crypto"
    for d in shadow_crypto shadow_coinbase; do
      append_decision_json_dir "$d"
    done
  fi
  append_file "$(latest_log "$PROJECT_ROOT/logs/coinbase_live_*.log")"
  if [[ "$INCLUDE_COINBASE_WATCHDOG_LOG" == "1" ]]; then
    append_file "$PROJECT_ROOT/logs/watchdog_coinbase_loop.log"
  fi
  if [[ "$INCLUDE_WATCHDOG_LOG" == "1" ]]; then
    append_file "$WATCHDOG_LOG_DIR/shadow_watchdog.out.log"
  fi
  append_health_file "data_ingress_latest_crypto_coinbase.json"
  append_health_file "process_watchdog_latest.json"
  append_health_file "shadow_watchdog_tripwire_latest.json"
fi

if [[ "$SOURCE" == "coinbase" || "$SOURCE" == "futures" || "$SOURCE" == "coinbase_futures" || "$SOURCE" == "main" || "$SOURCE" == "all" ]]; then
  if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
    append_decision_channel_dir "crypto_futures_crypto_schwab"
    append_trade_decision_dir "shadow_crypto_futures_crypto"
    append_decision_json_dir "shadow_crypto_futures_crypto"
  fi
  append_file "$(latest_log "$PROJECT_ROOT/logs/coinbase_futures_live_*.log")"
  append_health_file "data_ingress_latest_crypto_futures_crypto_coinbase.json"
  append_file "$(latest_log "$PROJECT_ROOT/governance/health/shadow_loop_crypto_futures_crypto_coinbase_*.json")"
fi

if [[ "$SOURCE" == "fx" || "$SOURCE" == "all" ]]; then
  if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
    append_decision_channel_dir "fx_equities_schwab"
    append_trade_decision_dir "shadow_fx_equities"
    append_decision_json_dir "shadow_fx_equities"
  fi
  append_file "$(latest_log "$PROJECT_ROOT/logs/fx_live_*.log")"
  append_health_file "data_ingress_latest_fx_equities_schwab.json"
  append_health_file "broker_truth_fx_equities_schwab_latest.json"
  append_health_file "fx_shadow_session_latest.json"
  append_health_file "fx_market_context_sync_latest.json"
  append_file "$(latest_log "$PROJECT_ROOT/governance/health/shadow_loop_fx_equities_schwab_*.json")"
fi

if [[ "$HEAVY_REQUESTED" == "1" && "$INCLUDE_DECISIONS" == "1" && "$SOURCE" == "all" && "$HEAVY_INCLUDE_ALL_DECISION_DIRS" == "1" ]]; then
  append_all_decision_channel_dirs
  append_all_decision_json_dirs
fi

if [[ "$SOURCE" == "infra" || "$SOURCE" == "all" || "$HEAVY_REQUESTED" == "1" ]]; then
  append_heavy_health_files
fi

prioritize_heavy_livefeed_files() {
  [[ "$HEAVY_REQUESTED" == "1" ]] || return 0
  typeset -a paper_files decision_files live_log_files core_health_files rest_files
  local f
  for f in "${files[@]}"; do
    case "$f" in
      */governance/health/broker_readiness_latest.json|*/governance/health/paper_live_data_standard_latest.json|*/governance/health/execution_lane_paper_latest.json|*/governance/health/paper_runtime_profitability_controls_latest.json|*/governance/health/paper_profitability_control_latest.json|*/governance/health/paper_performance_latest.json|*/governance/health/paper_execution_truth_layer_latest.json|*/governance/health/paper_execution_calibration_latest.json|*/governance/health/runtime_paper_regression_guard_latest.json|*/governance/health/paper_400_ramp_latest.json)
        paper_files+=("$f")
        ;;
      */governance/channels/decision/*/decision_*.jsonl|*/decision_explanations/*/decision_explanations_*.jsonl|*/local_fallback_storage/decisions/*/trade_decisions_*.jsonl)
        decision_files+=("$f")
        ;;
      */logs/coinbase_live_*.log|*/logs/coinbase_futures_live_*.log|*/logs/schwab_live_*.log|*/logs/schwab_futures_live_*.log|*/logs/fx_live_*.log|*/logs/all_sleeves_*.log)
        live_log_files+=("$f")
        ;;
      */governance/health/auth_lease_manager_latest.json|*/governance/health/schwab_auth_supervisor_latest.json|*/governance/health/runtime_gate_dashboard_latest.json|*/governance/health/runtime_throttle_control_latest.json|*/governance/health/ingestion_storage_control_latest.json|*/governance/health/coordination_state_latest.json|*/governance/health/hdf5_training_cache_latest.json|*/governance/health/one_numbers_latest.json)
        core_health_files+=("$f")
        ;;
      *)
        rest_files+=("$f")
        ;;
    esac
  done
  files=("${paper_files[@]}" "${decision_files[@]}" "${live_log_files[@]}" "${core_health_files[@]}" "${rest_files[@]}")
}

prioritize_heavy_livefeed_files

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No live feed files found for source=$SOURCE local_day=$DAY_LOCAL utc_day=$DAY_UTC" >&2
  echo "Start loops first with: $PROJECT_ROOT/scripts/ops/opsctl.sh start" >&2
  exit 1
fi

if [[ "$HEAVY_REQUESTED" == "1" && "$SNAPSHOT" != "1" && "$HEAVY_MAX_FOLLOW_FILES" =~ ^[0-9]+$ && "$HEAVY_MAX_FOLLOW_FILES" -gt 0 && ${#files[@]} -gt "$HEAVY_MAX_FOLLOW_FILES" ]]; then
  typeset -a capped_files
  local_i=1
  while [[ "$local_i" -le "$HEAVY_MAX_FOLLOW_FILES" && "$local_i" -le ${#files[@]} ]]; do
    capped_files+=("${files[$local_i]}")
    local_i=$((local_i + 1))
  done
  files=("${capped_files[@]}")
fi

TAIL_START_MODE="lines"
if [[ "$HEAVY_REQUESTED" == "1" && "$RAW" != "1" ]]; then
  TAIL_START_MODE="bytes"
fi

write_heavy_marker() {
  [[ "$HEAVY_REQUESTED" == "1" ]] || return 0
  local active="true"
  if [[ "$SNAPSHOT" == "1" ]]; then
    active="false"
  fi
  mkdir -p "$HEALTH_DIR"
  {
    printf '{'
    printf '"timestamp_utc":"%s",' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '"schema_version":1,'
    printf '"active":%s,' "$active"
    printf '"mode":"%s",' "$([[ "$SNAPSHOT" == "1" ]] && printf snapshot || printf live_tail)"
    printf '"pid":%s,' "$LIVE_FEED_MAIN_PID"
    printf '"started_epoch":%s,' "$(date +%s)"
    printf '"ttl_enabled":%s,' "$([[ "$HEAVY_TTL_ENABLED" == "1" ]] && printf true || printf false)"
    printf '"ttl_seconds":%s,' "$HEAVY_TTL_SECONDS"
    printf '"source":"%s",' "$SOURCE"
    printf '"lines":%s,' "$LINES"
    printf '"include_decisions":%s,' "$INCLUDE_DECISIONS"
    printf '"include_watchdog_log":%s,' "$INCLUDE_WATCHDOG_LOG"
    printf '"include_coinbase_watchdog_log":%s,' "$INCLUDE_COINBASE_WATCHDOG_LOG"
    printf '"status_snapshot":%s,' "$STATUS_SNAPSHOT"
    printf '"important_only":%s,' "$IMPORTANT_ONLY"
    printf '"show_file_list":%s,' "$SHOW_FILE_LIST"
    printf '"suppress_futures_specialist_intents":%s,' "$SUPPRESS_FUTURES_SPECIALIST_INTENTS"
    printf '"suppress_json_fragments":%s,' "$SUPPRESS_JSON_FRAGMENTS"
    printf '"suppress_tail_headers":%s,' "$SUPPRESS_TAIL_HEADERS"
    printf '"dedup_repeated_lines":%s,' "$DEDUP_REPEATED_LINES"
    printf '"show_keepalive":%s,' "$SHOW_KEEPALIVE"
    printf '"visible_keepalive_allowed":%s,' "$VISIBLE_KEEPALIVE_ALLOWED"
    printf '"memory_aware":%s,' "$MEMORY_AWARE"
    printf '"highlight_enabled":%s,' "$COLOR_ENABLED"
    printf '"highlight_mode":"%s",' "$COLOR_MODE"
    printf '"highlight_palette":"%s",' "$COLOR_PALETTE"
    printf '"pressure_optimized":%s,' "$PRESSURE_OPTIMIZED"
    printf '"decision_file_mode":"%s",' "$DECISION_FILE_MODE"
    printf '"file_count":%s,' "${#files[@]}"
    printf '"skipped_file_count":%s,' "${#skipped_files[@]}"
    printf '"max_follow_files":%s,' "$HEAVY_MAX_FOLLOW_FILES"
    printf '"tail_start_mode":"%s",' "$TAIL_START_MODE"
    printf '"tail_start_bytes":%s,' "$HEAVY_TAIL_BYTES"
    printf '"bootstrap_snapshot":%s,' "$HEAVY_BOOTSTRAP_SNAPSHOT"
    printf '"bootstrap_max_lines":%s,' "$HEAVY_BOOTSTRAP_MAX_LINES"
    printf '"snapshot_max_lines":%s,' "$HEAVY_SNAPSHOT_MAX_LINES"
    printf '"decision_max_age_hours":%s,' "$DECISION_MAX_AGE_HOURS"
    printf '"include_all_decision_dirs":%s,' "$HEAVY_INCLUDE_ALL_DECISION_DIRS"
    printf '"self_throttle":%s,' "$HEAVY_SELF_THROTTLE"
    printf '"nice":%s,' "$HEAVY_NICE"
    printf '"background_policy":%s,' "$HEAVY_BACKGROUND_POLICY"
    printf '"contract":"operator_requested_heavy_observability_view"'
    printf '}\n'
  } > "$HEAVY_MARKER_FILE"
}

write_heavy_marker

write_livefeed_health() {
  [[ "$LIVEFEED_HEALTH_WRITER" == "1" ]] || return 0
  local feed_state="${1:-running}"
  local alive="true"
  if [[ "$feed_state" != "running" ]]; then
    alive="false"
  fi
  mkdir -p "$HEALTH_DIR"
  {
    printf '{'
    printf '"timestamp_utc":"%s",' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '"schema_version":1,'
    printf '"status":"%s",' "$feed_state"
    printf '"alive":%s,' "$alive"
    printf '"pid":%s,' "$LIVE_FEED_MAIN_PID"
    printf '"health_writer":true,'
    printf '"writer_mode":"local_mirror",'
    printf '"source":"%s",' "$SOURCE"
    printf '"heavy":%s,' "$HEAVY_REQUESTED"
    printf '"lines":%s,' "$LINES"
    printf '"file_count":%s,' "${#files[@]}"
    printf '"skipped_file_count":%s,' "${#skipped_files[@]}"
    printf '"idle_heartbeat_seconds":0,'
    printf '"include_decisions":%s,' "$INCLUDE_DECISIONS"
    printf '"include_watchdog_log":%s,' "$INCLUDE_WATCHDOG_LOG"
    printf '"include_coinbase_watchdog_log":%s,' "$INCLUDE_COINBASE_WATCHDOG_LOG"
    printf '"status_snapshot":%s,' "$STATUS_SNAPSHOT"
    printf '"important_only":%s,' "$IMPORTANT_ONLY"
    printf '"show_file_list":%s,' "$SHOW_FILE_LIST"
    printf '"suppress_futures_specialist_intents":%s,' "$SUPPRESS_FUTURES_SPECIALIST_INTENTS"
    printf '"suppress_json_fragments":%s,' "$SUPPRESS_JSON_FRAGMENTS"
    printf '"suppress_tail_headers":%s,' "$SUPPRESS_TAIL_HEADERS"
    printf '"dedup_repeated_lines":%s,' "$DEDUP_REPEATED_LINES"
    printf '"show_keepalive":%s,' "$SHOW_KEEPALIVE"
    printf '"visible_keepalive_allowed":%s,' "$VISIBLE_KEEPALIVE_ALLOWED"
    printf '"memory_aware":%s,' "$MEMORY_AWARE"
    printf '"pressure_optimized":%s,' "$PRESSURE_OPTIMIZED"
    printf '"tail_start_mode":"%s",' "$TAIL_START_MODE"
    printf '"follow_restart_count":%s,' "$FOLLOW_RESTART_COUNT"
    printf '"follow_last_rc":%s,' "$FOLLOW_LAST_RC"
    printf '"contract":"launchd_local_livefeed_mirror"'
    printf '}\n'
  } > "$LIVEFEED_HEALTH_FILE"
}

write_livefeed_health "running"

mark_heavy_inactive() {
  [[ "$HEAVY_REQUESTED" == "1" ]] || return 0
  mkdir -p "$HEALTH_DIR"
  {
    printf '{'
    printf '"timestamp_utc":"%s",' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '"schema_version":1,'
    printf '"active":false,'
    printf '"mode":"expired_or_closed",'
    printf '"pid":%s,' "$LIVE_FEED_MAIN_PID"
    printf '"source":"%s",' "$SOURCE"
    printf '"ttl_enabled":%s,' "$([[ "$HEAVY_TTL_ENABLED" == "1" ]] && printf true || printf false)"
    printf '"ttl_seconds":%s,' "$HEAVY_TTL_SECONDS"
    printf '"contract":"operator_requested_heavy_observability_view"'
    printf '}\n'
  } > "$HEAVY_MARKER_FILE"
}

cleanup_live_feed() {
  if [[ -n "${LIVE_FEED_KEEPALIVE_PID:-}" ]]; then
    kill "$LIVE_FEED_KEEPALIVE_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${HEAVY_TTL_PID:-}" ]]; then
    kill "$HEAVY_TTL_PID" >/dev/null 2>&1 || true
  fi
  write_livefeed_health "stopped"
  mark_heavy_inactive
  true
}

install_live_feed_trap() {
  [[ "$SNAPSHOT" != "1" ]] || return 0
  # Do not trap EXIT here: zsh pipeline stages can inherit the trap and mark the
  # heavy feed closed while the parent tail is still alive.
  trap 'cleanup_live_feed; exit 129' HUP
  trap 'cleanup_live_feed; exit 130' INT
  trap 'cleanup_live_feed; exit 143' TERM
}

start_live_feed_keepalive() {
  [[ "$SNAPSHOT" != "1" ]] || return 0
  [[ "$KEEPALIVE_ENABLED" == "1" ]] || return 0
  [[ "$KEEPALIVE_SECONDS" -gt 0 ]] || return 0
  (
    keepalive_count=0
    while true; do
      sleep "$KEEPALIVE_SECONDS" || exit 0
      write_livefeed_health "running"
      keepalive_count=$((keepalive_count + 1))
      emit_live_feed_keepalive "$keepalive_count"
    done
  ) &
  LIVE_FEED_KEEPALIVE_PID=$!
}

emit_live_feed_keepalive() {
  local keepalive_count="${1:-0}"
  [[ "$SHOW_KEEPALIVE" == "1" && "$VISIBLE_KEEPALIVE_ALLOWED" == "1" ]] || return 0
  printf 'live_feed_keepalive timestamp_utc=%s keepalive_count=%s source=%s heavy=%s files=%s following=1 important_only=%s waiting_for_new_matching_lines=1 next_keepalive_seconds=%s interrupt=ctrl-c\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$keepalive_count" "$SOURCE" "$HEAVY_REQUESTED" "${#files[@]}" "$IMPORTANT_ONLY" "$KEEPALIVE_SECONDS"
  if [[ "$HEAVY_REQUESTED" == "1" && "$INCLUDE_DECISIONS" == "1" && "$KEEPALIVE_DECISION_SNAPSHOT" == "1" && "$KEEPALIVE_DECISION_EVERY" -gt 0 && ( "$keepalive_count" -eq 0 || $((keepalive_count % KEEPALIVE_DECISION_EVERY)) -eq 0 ) ]]; then
    emit_livefeed_decision_paper_snapshot | truncate_live_lines 0 0 || true
  fi
  if [[ "$HEAVY_REQUESTED" == "1" && "$STATUS_SNAPSHOT" == "1" && "$KEEPALIVE_STATUS_EVERY" -gt 0 && "$keepalive_count" -gt 0 && $((keepalive_count % KEEPALIVE_STATUS_EVERY)) -eq 0 ]]; then
    echo "live_feed_keepalive_status_snapshot=begin every=${KEEPALIVE_STATUS_EVERY} keepalive_count=${keepalive_count}"
    emit_livefeed_status_snapshot | truncate_live_lines 40 || true
    echo "live_feed_keepalive_status_snapshot=end"
  fi
}

start_heavy_ttl_guard() {
  [[ "$HEAVY_REQUESTED" == "1" ]] || return 0
  [[ "$SNAPSHOT" != "1" ]] || return 0
  [[ "$HEAVY_TTL_ENABLED" == "1" ]] || return 0
  [[ "$HEAVY_TTL_SECONDS" -gt 0 ]] || return 0
  (
    zmodload zsh/system 2>/dev/null || true
    ttl_guard_pid="${sysparams[pid]:-0}"
    terminate_livefeed_descendants() {
      local parent_pid="${1:-}"
      local child_pid
      [[ "$parent_pid" =~ ^[0-9]+$ ]] || return 0
      for child_pid in ${(f)"$(pgrep -P "$parent_pid" 2>/dev/null || true)"}; do
        [[ "$child_pid" =~ ^[0-9]+$ ]] || continue
        [[ "$child_pid" == "$ttl_guard_pid" ]] && continue
        terminate_livefeed_descendants "$child_pid"
        kill -TERM "$child_pid" >/dev/null 2>&1 || true
      done
    }
    sleep "$HEAVY_TTL_SECONDS"
    printf 'live_feed_heavy_ttl_expired ttl_seconds=%s source=%s\n' "$HEAVY_TTL_SECONDS" "$SOURCE" >&2
    terminate_livefeed_descendants "$LIVE_FEED_MAIN_PID"
    kill -TERM "$LIVE_FEED_MAIN_PID" >/dev/null 2>&1 || true
  ) &
  HEAVY_TTL_PID=$!
}

echo "live_feed source=$SOURCE local_day=$DAY_LOCAL utc_day=$DAY_UTC symbol=${SYMBOL:-ALL} lines=$LINES heavy=$HEAVY_REQUESTED health_writer=$LIVEFEED_HEALTH_WRITER include_decisions=$INCLUDE_DECISIONS include_watchdog_log=$INCLUDE_WATCHDOG_LOG status_snapshot=$STATUS_SNAPSHOT important_only=$IMPORTANT_ONLY suppress_futures_specialist_intents=$SUPPRESS_FUTURES_SPECIALIST_INTENTS suppress_json_fragments=$SUPPRESS_JSON_FRAGMENTS suppress_tail_headers=$SUPPRESS_TAIL_HEADERS dedup_repeated_lines=$DEDUP_REPEATED_LINES show_keepalive=$SHOW_KEEPALIVE visible_keepalive_allowed=$VISIBLE_KEEPALIVE_ALLOWED memory_profile=${MEMORY_PROFILE:-default} memory_aware=$MEMORY_AWARE highlight=$COLOR_ENABLED highlight_palette=$COLOR_PALETTE decision_mode=$DECISION_FILE_MODE pressure_optimized=$PRESSURE_OPTIMIZED file_count=${#files[@]} skipped_file_count=${#skipped_files[@]} tail_start_mode=$TAIL_START_MODE tail_start_bytes=$HEAVY_TAIL_BYTES max_line_chars=$MAX_LINE_CHARS"
if [[ "$SHOW_FILE_LIST" == "1" ]]; then
  for f in "${files[@]}"; do
    echo " - $f"
  done
  if (( ${#skipped_files[@]} > 0 )); then
    local_i=1
    while [[ "$local_i" -le ${#skipped_files[@]} ]]; do
      echo " - skipped ${skipped_file_reasons[$local_i]} ${skipped_files[$local_i]}"
      local_i=$((local_i + 1))
    done
  fi
else
  echo "live_feed_files_hidden count=${#files[@]} skipped=${#skipped_files[@]} use=--show-files"
fi
if (( ${#skipped_files[@]} > 0 )); then
  echo "live_feed_files_skipped count=${#skipped_files[@]} reason=tail_unreadable use=--show-files"
fi

if [[ "$RAW" == "1" ]]; then
  if [[ "$SNAPSHOT" == "1" ]]; then
    tail -n "$LINES" "${files[@]}"
    exit $?
  fi
  install_live_feed_trap
  start_heavy_ttl_guard
  tail -n "$LINES" -F "${files[@]}"
  exit $?
fi

tail_source_snapshot() {
  if [[ "$TAIL_START_MODE" == "bytes" ]]; then
    tail -c "$HEAVY_TAIL_BYTES" "${files[@]}"
    return $?
  fi
  tail -n "$LINES" "${files[@]}"
}

tail_source_follow() {
  if [[ "$TAIL_START_MODE" == "bytes" ]]; then
    if [[ "$HEAVY_REQUESTED" == "1" && "$HEAVY_BOOTSTRAP_SNAPSHOT" == "1" ]]; then
      tail -n 0 -F "${files[@]}"
      return $?
    fi
    tail -c "$HEAVY_TAIL_BYTES" -F "${files[@]}"
    return $?
  fi
  tail -n "$LINES" -F "${files[@]}"
}

truncate_live_lines() {
  local line_limit="${1:-0}"
  local important_override="${2:-$IMPORTANT_ONLY}"
  awk -v max="$MAX_LINE_CHARS" -v limit="$line_limit" -v color="$COLOR_ENABLED" -v palette="$COLOR_PALETTE" -v suppress_fut_intents="$SUPPRESS_FUTURES_SPECIALIST_INTENTS" -v suppress_json_fragments="$SUPPRESS_JSON_FRAGMENTS" -v suppress_tail_headers="$SUPPRESS_TAIL_HEADERS" -v dedup_repeats="$DEDUP_REPEATED_LINES" -v important_only="$important_override" '
  BEGIN {
    if (color == "1") {
      esc = sprintf("%c", 27)
      reset = esc "[0m"
      bold = esc "[1m"
      dim = esc "[2m"
      red = esc "[31m"
      green = esc "[32m"
      yellow = esc "[33m"
      blue = esc "[34m"
      magenta = esc "[35m"
      cyan = esc "[36m"
      action_green = green
      action_red = red
      if (palette == "red" || palette == "red_only" || palette == "red-only" || palette == "mono" || palette == "monochrome") {
        green = red
        yellow = red
        blue = red
        magenta = red
        cyan = red
        action_green = red
      } else if (palette == "red_actions" || palette == "red-actions") {
        yellow = red
        blue = red
        magenta = red
        cyan = red
      }
    } else {
      reset = bold = dim = red = green = yellow = blue = magenta = cyan = action_green = action_red = ""
    }
  }
  function text_field(line, key, pattern, raw) {
    pattern = "\"" key "\"[[:space:]]*:[[:space:]]*\"[^\"]*\""
    if (match(line, pattern)) {
      raw = substr(line, RSTART, RLENGTH)
      sub("^\"" key "\"[[:space:]]*:[[:space:]]*\"", "", raw)
      sub("\"$", "", raw)
      return raw
    }
    return ""
  }
  function num_field(line, key, pattern, raw) {
    pattern = "\"" key "\"[[:space:]]*:[[:space:]]*-?[0-9.]+"
    if (match(line, pattern)) {
      raw = substr(line, RSTART, RLENGTH)
      sub("^\"" key "\"[[:space:]]*:[[:space:]]*", "", raw)
      return raw
    }
    return ""
  }
  function bool_field(line, key, pattern, raw) {
    pattern = "\"" key "\"[[:space:]]*:[[:space:]]*(true|false)"
    if (match(line, pattern)) {
      raw = substr(line, RSTART, RLENGTH)
      sub("^\"" key "\"[[:space:]]*:[[:space:]]*", "", raw)
      return raw
    }
    return ""
  }
  function token_value(line, key, pattern, raw) {
    pattern = key "=[^[:space:]]+"
    if (match(line, pattern)) {
      raw = substr(line, RSTART, RLENGTH)
      sub("^" key "=", "", raw)
      return raw
    }
    return ""
  }
  function token_num_value(line, key, pattern, raw) {
    pattern = key "=-?[0-9.]+"
    if (match(line, pattern)) {
      raw = substr(line, RSTART, RLENGTH)
      sub("^" key "=", "", raw)
      return raw
    }
    return ""
  }
  function arg_value(line, flag, pattern, raw) {
    pattern = flag "[[:space:]]+[^[:space:]]+"
    if (match(line, pattern)) {
      raw = substr(line, RSTART, RLENGTH)
      sub("^" flag "[[:space:]]+", "", raw)
      return raw
    }
    return ""
  }
  function first_text(line, a, b, c, d, e, f, value) {
    if (a != "") { value = text_field(line, a); if (value != "") return value }
    if (b != "") { value = text_field(line, b); if (value != "") return value }
    if (c != "") { value = text_field(line, c); if (value != "") return value }
    if (d != "") { value = text_field(line, d); if (value != "") return value }
    if (e != "") { value = text_field(line, e); if (value != "") return value }
    if (f != "") { value = text_field(line, f); if (value != "") return value }
    return ""
  }
  function first_num(line, a, b, c, d, e, f, value) {
    if (a != "") { value = num_field(line, a); if (value != "") return value }
    if (b != "") { value = num_field(line, b); if (value != "") return value }
    if (c != "") { value = num_field(line, c); if (value != "") return value }
    if (d != "") { value = num_field(line, d); if (value != "") return value }
    if (e != "") { value = num_field(line, e); if (value != "") return value }
    if (f != "") { value = num_field(line, f); if (value != "") return value }
    return ""
  }
  function short_value(value, max_len) {
    gsub(/[[:space:]]+/, "_", value)
    if (max_len > 0 && length(value) > max_len) return substr(value, 1, max_len - 3) "..."
    return value
  }
  function short_num(value) {
    if (value == "") return ""
    return sprintf("%.4f", value + 0.0)
  }
  function bytes_to_gb(value) {
    if (value == "") return ""
    return sprintf("%.1f", (value + 0.0) / 1073741824.0)
  }
  function basename_value(value, parts, n) {
    if (value == "") return ""
    n = split(value, parts, "/")
    return parts[n]
  }
  function csv_count(value, parts) {
    if (value == "") return ""
    return split(value, parts, ",")
  }
  function human_length(value) {
    if (value >= 1000000) return sprintf("%.1fM", value / 1000000.0)
    if (value >= 1000) return sprintf("%.1fk", value / 1000.0)
    return value
  }
  function append_token(line, key, value) {
    if (value == "") return line
    return line " " key "=" short_value(value, 96)
  }
  function looks_like_escaped_json_fragment(line, trimmed) {
    trimmed = line
    sub(/^[[:space:]]+/, "", trimmed)
    return trimmed ~ /^\\\"[A-Za-z0-9_:-]+\\\"[[:space:]]*:/ || trimmed ~ /^\{\\\"[A-Za-z0-9_:-]+\\\"[[:space:]]*:/ || line ~ /\\",[[:space:]]*\\\"[A-Za-z0-9_:-]+\\\"[[:space:]]*:/ || line ~ /\\\"(token_before|token_after|stdout_tail|stderr_tail|finder_sync|route_verification|active_repo|external|local|entries|broker_readiness)\\\"[[:space:]]*:/ || line ~ /\/schwab_trading_bot\/[^[:space:]]+\\",[[:space:]]*\\\"(exists|size_bytes|mtime_utc|age_seconds|expires_at|expires_in_seconds)\\\"[[:space:]]*:/
  }
  function looks_like_json_fragment(line) {
    return looks_like_escaped_json_fragment(line) || line ~ /^[[:space:]]*[,}\]]/ || line ~ /^[[:space:]]*"[A-Za-z0-9_:-]+"[[:space:]]*:/ || line ~ /^[[:space:]]*"[^\"]+"[[:space:]]*,?[[:space:]]*$/ || line ~ /^[[:space:]]*[A-Za-z0-9_:-]+"[[:space:]]*:/ || line ~ /",[[:space:]]*"[A-Za-z0-9_:-]+"[[:space:]]*:/ || line ~ /"bot_id"[[:space:]]*:|"observer_meta"[[:space:]]*:|"sleeve_profile"[[:space:]]*:/
  }
  function suppressible_futures_specialist_intent(line) {
    return (line ~ /\[ExecutionIntent\]/ && line ~ /paper_mirror_futures::futures_specialist_/) || line ~ /\[PaperMirrorFutures\]/
  }
  function intentional_data_only_blocked(lower) {
    return lower ~ /data_only_blocked/ && lower ~ /market_data_only=1|execution_enabled=0|collection_only_no_master_vote/
  }
  function paper_mirror_selection_line(line) {
    return line ~ /^\[PaperMirror\][[:space:]]+selected=/
  }
  function important_operator_line(line, lower) {
    lower = tolower(line)
    if (paper_mirror_selection_line(line)) return 0
    if (line ~ /^\[(status-contract|system|collection|fx-provider|auth|schwab-auth|storage|throttle|soak|dashboard|paper|paper-data|paper-profit|paper-truth|decision-latest|decision-route)\]/) return 1
    if (line ~ /\[Decision\]|\[decision\]|ExecutionIntent|ShadowLoop|RegimeCooldown|AdaptiveInterval/) return 1
    if (line ~ /"symbol"[[:space:]]*:/ && line ~ /"action"[[:space:]]*:|"master_action"[[:space:]]*:|"master_intent_action"[[:space:]]*:|"grand_action"[[:space:]]*:/) return 1
    if (line ~ /symbol=[^[:space:]]+/ && line ~ /action=|grand_action=|futures_action=|options_action=|master_action=/) return 1
    if (lower ~ /global_trading_halt|operator_stop|halt_state|hard_gate|critical|failed|failure|tripwire|killswitch|kill_switch|margin_guard/) return 1
    if (lower ~ /degraded|warning|backpressure|storage_pressure|storage_backpressure|low_space|external_low_space|free_below|sql_wal_pressure|split_brain|route_verification|auth|oauth|lease_state|broker_ready|coinbase_api|schwab_auth|timeout|stale/) return 1
    if (lower ~ /blocked/ && !blocked_metric_is_clear(lower)) return 1
    if (lower ~ /\[hdf5\]/) return 1
    if (lower ~ /\[storage\]/) return 1
    if (lower ~ /\[alerts\]|\[notify\]/ && lower !~ /status=ready|ok=true/) return 1
    return 0
  }
  function compact_infra_noise_line(line, mode, root, copied, errors, pruned, skipped, conflicts, free_bytes, min_free_bytes, out, owner, lock_path, cmd, symbols, interval) {
    if (line ~ /^\[StorageRoute\]/) {
      mode = token_value(line, "mode")
      root = token_value(line, "active_root")
      copied = token_num_value(line, "copied")
      errors = token_num_value(line, "errors")
      pruned = token_num_value(line, "pruned")
      skipped = token_value(line, "skipped")
      if (skipped ~ /autosync_skipped_external_low_space/) skipped = "external_low_space"
      conflicts = token_num_value(line, "split_brain_conflicts")
      free_bytes = token_num_value(line, "free_bytes")
      min_free_bytes = token_num_value(line, "min_free_bytes")
      out = "[StorageRoute]"
      out = append_token(out, "mode", mode)
      out = append_token(out, "root", root)
      out = append_token(out, "copied", copied)
      out = append_token(out, "errors", errors)
      out = append_token(out, "pruned", pruned)
      out = append_token(out, "skipped", skipped)
      out = append_token(out, "free_gb", bytes_to_gb(free_bytes))
      out = append_token(out, "min_gb", bytes_to_gb(min_free_bytes))
      out = append_token(out, "conflicts", conflicts)
      out = append_token(out, "len", human_length(length(line)))
      return out
    }
    if (line ~ /^\[ShadowLock\][[:space:]]+busy/) {
      owner = token_value(line, "owner")
      sub("^pid=", "", owner)
      lock_path = token_value(line, "lock_path")
      cmd = token_value(line, "cmd")
      symbols = arg_value(line, "--symbols")
      interval = arg_value(line, "--interval-seconds")
      out = "[ShadowLock] busy"
      out = append_token(out, "broker", token_value(line, "broker"))
      out = append_token(out, "profile", token_value(line, "profile"))
      out = append_token(out, "owner_pid", owner)
      out = append_token(out, "lock", basename_value(lock_path))
      out = append_token(out, "cmd", basename_value(cmd))
      out = append_token(out, "symbols", csv_count(symbols))
      if (interval != "") out = append_token(out, "interval", interval "s")
      out = append_token(out, "len", human_length(length(line)))
      return out
    }
    return line
  }
  function tail_file_header(line) {
    return line ~ /^==> .* <==$/
  }
  function normalized_repeat_key(line, key) {
    key = line
    sub(/ timestamp_utc=[^[:space:]]+/, " timestamp_utc=?", key)
    sub(/ snapshot_id=[^[:space:]]+/, " snapshot_id=?", key)
    sub(/ iter=[0-9]+/, " iter=?", key)
    if (key ~ /^\[StorageRoute\]/) {
      sub(/ free_gb=[^[:space:]]+/, " free_gb=?", key)
      sub(/ min_gb=[^[:space:]]+/, " min_gb=?", key)
      sub(/ len=[^[:space:]]+/, " len=?", key)
    }
    if (key ~ /^\[ShadowLock\]/) {
      sub(/ len=[^[:space:]]+/, " len=?", key)
    }
    return key
  }
  function highlight_token(line, pattern, paint) {
    if (color != "1") return line
    gsub(pattern, paint "&" reset, line)
    return line
  }
  function blocked_metric_is_clear(lower) {
    return lower ~ /blocked/ && (lower ~ /false/ || lower ~ /"[[:space:]]*0[[:space:]]*"/ || lower ~ /"[[:space:]]*0\.0+[[:space:]]*"/ || lower ~ /:[[:space:]]*0([,}[:space:]]|$)/ || lower ~ /:[[:space:]]*0\.0+([,}[:space:]]|$)/)
  }
  function colorize_line(line, lower, prefix, clear_blocked_metric, red_alert, explicit_level, explicit_status) {
    if (color != "1") return line
    lower = tolower(line)
    prefix = ""
    clear_blocked_metric = blocked_metric_is_clear(lower)
    explicit_level = token_value(lower, "level")
    explicit_status = token_value(lower, "status")
    red_alert = explicit_level == "alert" || (!paper_mirror_selection_line(line) && lower ~ /global_trading_halt|operator_stop|halt=true|halt_state[^a-z0-9_]*active|hard_gate|critical|failed=[^[:space:]][^[:space:]]*|failed_checks=[^[:space:]][^[:space:]]*|tripwire|killswitch|kill_switch|margin_guard/)
    if (explicit_level == "ok") {
      red_alert = 0
      prefix = bold green "[OK] " reset
    } else if (explicit_level == "flow") {
      red_alert = 0
      prefix = bold cyan "[FLOW] " reset
    } else if (explicit_level == "watch") {
      red_alert = 0
      prefix = bold yellow "[WATCH] " reset
    } else if (!clear_blocked_metric && lower ~ /blocked/ && !intentional_data_only_blocked(lower) && explicit_status !~ /blocked_read_only|locked/) {
      red_alert = 1
    }
    if (red_alert) {
      prefix = bold red "[ALERT] " reset
    } else if (prefix == "" && lower ~ /degraded|warning=[^[:space:]][^[:space:]]*|warnings=[^[:space:]][^[:space:]]*|backpressure|storage_pressure|sql_wal_pressure|awaiting|timeout|pressure|stale|low_grade_blockers=[1-9]/) {
      prefix = bold yellow "[WATCH] " reset
    } else if (prefix == "" && lower ~ /clear_ready|healthy|heartbeat_ok|broker_ready|ready|status[=:][[:space:]]*ok|\"ok\"[[:space:]]*:[[:space:]]*true/) {
      prefix = bold green "[OK] " reset
    } else if (prefix == "" && line ~ /\[Decision\]|\[decision\]|\[decision-latest\]|ShadowLoop|ExecutionIntent|RegimeCooldown|AdaptiveInterval/) {
      prefix = bold cyan "[FLOW] " reset
    }
    line = highlight_token(line, "GLOBAL_TRADING_HALT|OPERATOR_STOP|halt=true|critical|tripwire|killswitch|margin_guard", bold red)
    if (!clear_blocked_metric && !intentional_data_only_blocked(lower)) {
      line = highlight_token(line, "blocked", bold red)
    }
    line = highlight_token(line, "degraded|warning|backpressure|storage_pressure|sql_wal_pressure|awaiting|timeout|throttle|pressure|stale|low_grade_blockers=[1-9][0-9]*", bold yellow)
    line = highlight_token(line, "clear_ready|healthy|heartbeat_ok|broker_ready|ready", bold green)
    line = highlight_token(line, "BUY", bold action_green)
    line = highlight_token(line, "SELL", bold action_red)
    line = highlight_token(line, "HOLD", bold blue)
    line = highlight_token(line, "symbol=|action=|mode=|status=|score=", cyan)
    return prefix line
  }
  {
    if (suppress_fut_intents == "1" && suppressible_futures_specialist_intent($0)) {
      next
    }
    if (suppress_json_fragments == "1" && looks_like_json_fragment($0)) {
      next
    }
    if (suppress_tail_headers == "1" && tail_file_header($0)) {
      next
    }
    if (important_only == "1" && !important_operator_line($0)) {
      next
    }
    line = compact_infra_noise_line($0)
    repeat_key = normalized_repeat_key(line)
    if (dedup_repeats == "1" && repeat_key == last_repeat_key) {
      next
    }
    last_repeat_key = repeat_key
    count += 1
    if (limit > 0 && count > limit) {
      print colorize_line("live_feed_output_truncated line_limit=" limit)
      fflush()
      exit 0
    }
    if (length(line) > max && line ~ /^\{/ && line ~ /"symbol"[[:space:]]*:/ && line ~ /"action"[[:space:]]*:|"master_action"[[:space:]]*:|"master_intent_action"[[:space:]]*:/) {
      has_master_decision = line ~ /"master_action"[[:space:]]*:|"master_intent_action"[[:space:]]*:/
      ts = text_field(line, "timestamp_utc")
      mode = first_text(line, "mode", "decision_mode", "execution_mode", "", "", "")
      profile = first_text(line, "shadow_profile", "profile", "source_profile", "paper_profile", "", "")
      broker = first_text(line, "broker", "shadow_domain", "market", "", "", "")
      status = ""
      if (!has_master_decision) status = first_text(line, "status", "state", "loop_state", "", "", "")
      symbol = text_field(line, "symbol")
      action = first_text(line, "action", "master_intent_action", "master_action", "decision", "", "")
      driver = first_text(line, "strategy", "bot_id", "source_strategy", "leader_strategy", "top_strategy", "")
      score = first_num(line, "model_score", "master_intent_score", "master_score", "score", "decision_score", "")
      threshold = first_num(line, "threshold", "master_threshold", "decision_threshold", "", "", "")
      guard_ok = bool_field(line, "ok")
      schema_valid = bool_field(line, "schema_valid")
      blocked_intent = bool_field(line, "master_guard_blocked_intent")
      reason = text_field(line, "reason")
      if (mode == "none" || mode == "null") mode = ""
      if (status == "" && blocked_intent == "true") status = "blocked"
      if (status == "" && schema_valid == "true") status = "ok"
      if (status == "" && guard_ok == "true") status = "ok"
      if (guard_ok == "true") guard = "ok"
      else if (guard_ok == "false") guard = "check"
      else guard = ""
      out = "[decision]"
      out = append_token(out, "ts", ts)
      out = append_token(out, "profile", profile)
      out = append_token(out, "broker", broker)
      out = append_token(out, "symbol", symbol)
      out = append_token(out, "action", action)
      out = append_token(out, "status", status)
      out = append_token(out, "score", short_num(score))
      out = append_token(out, "threshold", short_num(threshold))
      out = append_token(out, "mode", mode)
      out = append_token(out, "driver", driver)
      out = append_token(out, "guard", guard)
      out = append_token(out, "reason", reason)
      out = append_token(out, "len", human_length(length(line)))
      print colorize_line(out)
      fflush()
      next
    }
    if (length(line) > max && looks_like_json_fragment(line)) {
      print colorize_line("[json-fragment skipped len=" human_length(length(line)) " source=byte_tail_midline]")
    } else if (length(line) > max) {
      print colorize_line(substr(line, 1, max) "... [truncated length=" length(line) "]")
    } else {
      print colorize_line(line)
    }
    fflush()
  }'
}

emit_livefeed_status_snapshot() {
  [[ "$STATUS_SNAPSHOT" == "1" ]] || return 0
  [[ "$RAW" != "1" ]] || return 0
  local status_py="$PROJECT_ROOT/.venv314/bin/python"
  if [[ ! -x "$status_py" ]]; then
    status_py="$(command -v python3 || true)"
  fi
  [[ -n "$status_py" ]] || return 0
  "$status_py" - "$PROJECT_ROOT" "$SOURCE" <<'PY'
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
source = sys.argv[2]
health = root / "governance" / "health"
external = root / "data" / "external_context"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


def load(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def nested(payload: dict, *keys: str):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def as_bool(value) -> str:
    return "true" if bool(value) else "false"


def compact(value, max_len: int = 96) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if max_len > 0 and len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def joined(values, max_len: int = 120) -> str:
    if not isinstance(values, list):
        return ""
    return compact(",".join(str(item) for item in values if item is not None), max_len)


def env_broker_config() -> dict:
    default_broker = compact(os.getenv("DATA_BROKER", "schwab")).lower() or "schwab"
    market_data = compact(os.getenv("MARKET_DATA_PROVIDER") or os.getenv("SHADOW_MARKET_DATA_PROVIDER") or default_broker).lower()
    live_execution = compact(os.getenv("LIVE_EXECUTION_BROKER") or os.getenv("EXECUTION_BROKER") or default_broker).lower()
    paper_execution = compact(os.getenv("PAPER_EXECUTION_BROKER") or live_execution).lower()
    auth = compact(os.getenv("AUTH_BROKER") or live_execution or default_broker).lower()
    return {
        "market_data": market_data,
        "paper_execution": paper_execution,
        "auth": auth,
        "live_execution": live_execution,
        "market_data_only": os.getenv("MARKET_DATA_ONLY", "1"),
        "live_orders": os.getenv("ALLOW_ORDER_EXECUTION", "0"),
    }


def seconds(value) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return ""


def as_num(value) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    return f"{number:.3f}".rstrip("0").rstrip(".")


print(f"live_feed_status_snapshot=begin source={source} timestamp_utc={datetime.now(timezone.utc).isoformat()}")

feed = load(health / "livefeed_local_latest.json")
if feed:
    print(
        "[feed] "
        f"status={feed.get('status', 'unknown')} "
        f"alive={as_bool(feed.get('alive'))} "
        f"heavy={feed.get('heavy', '')} "
        f"files={feed.get('file_count', '')}"
    )

spcx = load(health / "spacex_ipo_downside_watch_latest.json")
if spcx:
    quote = spcx.get("quote") if isinstance(spcx.get("quote"), dict) else {}
    alert = spcx.get("alert") if isinstance(spcx.get("alert"), dict) else {}
    print(
        "[spcx] "
        f"status={spcx.get('overall_status', 'unknown')} "
        f"symbol={spcx.get('symbol', 'SPCX')} "
        f"quote_error={quote.get('error', '')} "
        f"alert={as_bool(alert.get('triggered'))} "
        f"policy={spcx.get('policy', '')}"
    )

macro = load(external / "live_macro_latest.json")
if macro:
    items = macro.get("items") if isinstance(macro.get("items"), list) else []
    item = items[0] if items and isinstance(items[0], dict) else {}
    print(
        "[macro] "
        f"headline={compact(item.get('headline') or macro.get('headline'), 130)} "
        f"shock={macro.get('shock_hint', '')} "
        f"sentiment={macro.get('sentiment_hint', '')} "
        f"expires={macro.get('expires_at_utc', '')}"
    )

macro_intel = load(health / "macro_event_intelligence_latest.json")
if macro_intel:
    calendar = macro_intel.get("calendar_verification") if isinstance(macro_intel.get("calendar_verification"), dict) else {}
    print(
        "[macro-intel] "
        f"status={macro_intel.get('overall_status', 'unknown')} "
        f"relevance={macro_intel.get('market_relevance', '')} "
        f"calendar={calendar.get('status', '')} "
        f"mismatch={joined(calendar.get('mismatch_expected_terms') or [], 80)}"
    )

notify = load(health / "mac_notification_watch_state.json")
if notify:
    allowlist = notify.get("imessage_event_allowlist") if isinstance(notify.get("imessage_event_allowlist"), list) else []
    min_severity = str(notify.get("imessage_min_severity", "") or "")
    if min_severity == "critical":
        min_severity = "crit"
    print(
        "[notify] "
        f"imessage={as_bool(notify.get('imessage_enabled') and notify.get('imessage_recipient_configured'))} "
        f"min={min_severity} "
        f"allow_count={len(allowlist)}"
    )

broker_cfg = env_broker_config()
print(
    "[broker] "
    f"market_data={broker_cfg['market_data']} "
    f"paper_execution={broker_cfg['paper_execution']} "
    f"auth={broker_cfg['auth']} "
    f"live_execution={broker_cfg['live_execution']} "
    f"market_data_only={broker_cfg['market_data_only']} "
    f"live_orders={broker_cfg['live_orders']}"
)

try:
    from scripts.ops.live_feed_status_contract import build_status_snapshot, format_status_lines

    for status_line in format_status_lines(build_status_snapshot(root, source=source)):
        print(status_line)
except Exception as exc:
    print(
        f"[status-contract] level=watch schema=3 status=degraded visibility=degraded "
        f"operational=unknown paper=unknown walkaway=false cause=status_contract_error "
        f"owner=livefeed impact=paper_unverified error={compact(exc, 140)} action=livefeed-status-contract"
    )

remote = load(health / "remote_alert_control_latest.json")
if remote:
    backlog = remote.get("critical_backlog") if isinstance(remote.get("critical_backlog"), dict) else {}
    channels = remote.get("channels") if isinstance(remote.get("channels"), dict) else {}
    print(
        "[alerts] "
        f"status={remote.get('overall_status', 'unknown')} "
        f"imessage={as_bool(channels.get('imessage_bridge'))} "
        f"unsent={backlog.get('unsent_count', '')} "
        f"unacked={backlog.get('unacked_count', '')}"
    )

watchdog = load(health / "process_watchdog_latest.json")
if watchdog:
    intel = watchdog.get("watchdog_intelligence") if isinstance(watchdog.get("watchdog_intelligence"), dict) else {}
    restarts = watchdog.get("restarts") if isinstance(watchdog.get("restarts"), list) else []
    print(
        "[watchdog] "
        f"status={watchdog.get('overall_status', 'unknown')} "
        f"grade={intel.get('grade', '')} "
        f"active_issues={intel.get('active_issue_count', '')} "
        f"restarts={len(restarts)}"
    )

dashboard = load(health / "runtime_gate_dashboard_latest.json")
if dashboard:
    overall = dashboard.get("overall") if isinstance(dashboard.get("overall"), dict) else {}
    active_attention = overall.get("attention") if isinstance(overall.get("attention"), list) else []
    managed_attention = overall.get("managed_attention") if isinstance(overall.get("managed_attention"), list) else []
    dashboard_status = str(overall.get("status") or "unknown").strip().lower()
    dashboard_level = "alert" if dashboard_status in {"critical", "blocked", "failed"} else ("watch" if dashboard_status in {"degraded", "warn", "warning"} else "ok")
    forensic_attention = overall.get("forensic_attention") if isinstance(overall.get("forensic_attention"), list) else []
    promotion_state = "evidence_pending" if "promotion_not_ready" in forensic_attention else "ready"
    print(
        "[dashboard] "
        f"level={dashboard_level} "
        f"status={dashboard_status} "
        f"ok={as_bool(overall.get('ok'))} "
        f"active={len(active_attention)} "
        f"managed={len(managed_attention)} "
        f"promotion={promotion_state} "
        f"attention={joined(active_attention, 160)}"
    )

hdf5 = load(health / "hdf5_training_cache_latest.json")
if hdf5:
    cache = hdf5.get("cache") if isinstance(hdf5.get("cache"), dict) else {}
    freshness = hdf5.get("freshness_gate") if isinstance(hdf5.get("freshness_gate"), dict) else {}
    schema = hdf5.get("schema_validation") if isinstance(hdf5.get("schema_validation"), dict) else {}
    bench = hdf5.get("performance_benchmark") if isinstance(hdf5.get("performance_benchmark"), dict) else {}
    speedup = bench.get("speedup_ratio", "")
    print(
        "[hdf5] "
        f"status={hdf5.get('overall_status', 'unknown')} "
        f"fresh={as_bool(freshness.get('fresh'))} "
        f"schema={as_bool(schema.get('ok'))} "
        f"rows={cache.get('row_count', '')} "
        f"features={cache.get('feature_count', '')} "
        f"speedup={speedup}"
    )

coord = load(health / "coordination_state_latest.json")
if coord:
    policies = coord.get("policies") if isinstance(coord.get("policies"), dict) else {}
    live = policies.get("live_orders") if isinstance(policies.get("live_orders"), dict) else {}
    paper = policies.get("paper_execution") if isinstance(policies.get("paper_execution"), dict) else {}
    heavy = policies.get("heavy_viewer") if isinstance(policies.get("heavy_viewer"), dict) else {}
    training = policies.get("training_launch") if isinstance(policies.get("training_launch"), dict) else {}
    terminal = policies.get("terminal_restart") if isinstance(policies.get("terminal_restart"), dict) else {}
    print(
        "[coord] "
        f"status={coord.get('overall_status', 'unknown')} "
        f"mode={coord.get('coordination_mode', '')} "
        f"live={as_bool(live.get('allowed'))} "
        f"paper={as_bool(paper.get('allowed'))} "
        f"heavy={as_bool(heavy.get('allowed'))} "
        f"train={as_bool(training.get('allowed'))} "
        f"terminal_restart={as_bool(terminal.get('safe'))}"
    )

print("live_feed_status_snapshot=end")
PY
}

emit_livefeed_decision_paper_snapshot() {
  [[ "$RAW" != "1" ]] || return 0
  local snapshot_py="$PROJECT_ROOT/.venv314/bin/python"
  if [[ ! -x "$snapshot_py" ]]; then
    snapshot_py="$(command -v python3 || true)"
  fi
  [[ -n "$snapshot_py" ]] || return 0
  "$snapshot_py" - "$PROJECT_ROOT" "$SOURCE" "$DECISION_SNAPSHOT_MAX_LINES" "$DECISION_SNAPSHOT_TAIL_BYTES" <<'PY'
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

root = Path(sys.argv[1])
source = str(sys.argv[2] or "main")
try:
    max_lines = max(int(float(sys.argv[3])), 1)
except Exception:
    max_lines = 4
try:
    tail_bytes = max(int(float(sys.argv[4])), 65536)
except Exception:
    tail_bytes = 4194304

health = root / "governance" / "health"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
try:
    from core.accountability import enrich_log_row as normalize_decision_record
except Exception:
    normalize_decision_record = None

ROUTE_PLACEHOLDERS = {"", "unknown", "unclassified", "none", "na", "n_a"}


def compact(value: Any, max_len: int = 96) -> str:
    text = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
    if max_len > 0 and len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def age_text(raw: Any) -> str:
    parsed = parse_ts(raw)
    if not parsed:
        return "unknown"
    age = max((datetime.now(timezone.utc) - parsed).total_seconds(), 0.0)
    if age < 120:
        return f"{int(age)}s"
    if age < 7200:
        return f"{int(age // 60)}m"
    return f"{age / 3600:.1f}h"


def as_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def as_num(value: Any) -> str:
    try:
        num = float(value)
    except Exception:
        return ""
    if abs(num) >= 100:
        return f"{num:.2f}"
    return f"{num:.4f}".rstrip("0").rstrip(".")


def nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def usable_route_label(value: Any) -> str:
    label = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return "" if label in ROUTE_PLACEHOLDERS else label


def normalized_decision(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    if normalize_decision_record is not None:
        try:
            return normalize_decision_record(
                payload,
                include_correlation=False,
                include_schema=False,
                path_hint=str(path),
                channel="decision" if "/decision/" in str(path).replace("\\", "/") else "",
            )
        except Exception:
            pass
    return dict(payload)


def decision_contract_state(original: dict[str, Any], normalized: dict[str, Any]) -> str:
    if original.get("schema_valid") is True:
        return "valid"
    errors = original.get("schema_errors") if isinstance(original.get("schema_errors"), list) else []
    repaired_action = bool(str(normalized.get("action") or "").strip())
    if repaired_action and errors and set(str(item) for item in errors).issubset({"missing:action"}):
        return "legacy_repaired"
    if original.get("schema_valid") is False or errors:
        return "invalid"
    return "unchecked"


def newest_jsonl(pattern: str) -> Path | None:
    matches = [path for path in root.glob(pattern) if path.is_file() and path.suffix == ".jsonl"]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def decision_candidates() -> list[Path]:
    patterns: list[str] = []
    if source in {"schwab", "main", "all"}:
        patterns.extend(
            [
                "governance/channels/decision/aggressive_equities_schwab/decision_*.jsonl",
                "governance/channels/decision/conservative_equities_schwab/decision_*.jsonl",
                "governance/channels/decision/dividend_equities_schwab/decision_*.jsonl",
                "governance/channels/decision/bond_equities_schwab/decision_*.jsonl",
                "governance/channels/decision/schwab_futures_equities_schwab/decision_*.jsonl",
                "decision_explanations/shadow_equities/decision_explanations_*.jsonl",
                "decision_explanations/shadow_schwab_futures_equities/decision_explanations_*.jsonl",
            ]
        )
    if source in {"coinbase", "main", "all"}:
        patterns.extend(
            [
                "governance/channels/decision/default_crypto_schwab/decision_*.jsonl",
                "governance/channels/decision/crypto_futures_crypto_schwab/decision_*.jsonl",
                "decision_explanations/shadow_crypto/decision_explanations_*.jsonl",
                "decision_explanations/shadow_coinbase/decision_explanations_*.jsonl",
                "decision_explanations/shadow_crypto_futures_crypto/decision_explanations_*.jsonl",
            ]
        )
    if source in {"fx", "all"}:
        patterns.extend(
            [
                "governance/channels/decision/fx_equities_schwab/decision_*.jsonl",
                "decision_explanations/shadow_fx_equities/decision_explanations_*.jsonl",
            ]
        )
    paths = [path for path in (newest_jsonl(pattern) for pattern in patterns) if path is not None]
    deduped: dict[str, Path] = {str(path): path for path in paths}
    return sorted(deduped.values(), key=lambda path: path.stat().st_mtime, reverse=True)[:12]


def recent_lines(path: Path) -> list[str]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            start = max(size - tail_bytes, 0)
            handle.seek(start)
            raw = handle.read()
        if start > 0:
            first_newline = raw.find(b"\n")
            if first_newline >= 0:
                raw = raw[first_newline + 1 :]
        return [line.decode("utf-8", errors="ignore") for line in raw.splitlines() if line.strip()]
    except Exception:
        return []


def first_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    return None


def decision_action(payload: dict[str, Any]) -> Any:
    value = first_value(payload, ("action", "master_action", "master_intent_action", "decision"))
    if value not in (None, ""):
        return value
    for path in (
        ("execution_guard", "action"),
        ("options_margin_guard", "action"),
        ("futures_margin_guard", "action"),
        ("execution_intent", "action"),
        ("intent", "action"),
    ):
        value = nested(payload, *path)
        if value not in (None, ""):
            return value
    return ""


def emit_decisions() -> None:
    rows: list[tuple[datetime, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    unresolved_routes: list[str] = []
    for path in decision_candidates():
        for line in reversed(recent_lines(path)):
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            normalized = normalized_decision(payload, path)
            timestamp = first_value(normalized, ("timestamp_utc", "ts_utc", "created_at_utc")) or ""
            file_timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            parsed_timestamp = parse_ts(timestamp)
            parsed = parsed_timestamp or file_timestamp
            age_source = "record_timestamp" if parsed_timestamp else "file_mtime"
            symbol = compact(first_value(normalized, ("symbol", "ticker", "underlying")) or "?", 28)
            profile = compact(first_value(normalized, ("shadow_profile", "profile", "mode", "source_profile")) or path.parent.name, 48)
            status = compact(first_value(normalized, ("status", "state", "loop_state")) or "", 32)
            action = compact(decision_action(normalized) or "?", 24)
            broker = compact(first_value(normalized, ("broker", "source_broker", "source_provider")) or "", 24)
            score = as_num(first_value(normalized, ("model_score", "master_intent_score", "master_score", "score", "decision_score")))
            threshold = as_num(first_value(normalized, ("threshold", "master_threshold", "decision_threshold")))
            original_route = payload.get("data_route") if isinstance(payload.get("data_route"), dict) else {}
            raw_lane = usable_route_label(original_route.get("routing_lane") or payload.get("routing_lane"))
            route = normalized.get("data_route") if isinstance(normalized.get("data_route"), dict) else {}
            lane = compact(usable_route_label(route.get("routing_lane") or normalized.get("routing_lane")), 42)
            lane_source = "record" if raw_lane else ("inferred" if lane else "unresolved")
            asset_class = compact(usable_route_label(route.get("asset_class") or normalized.get("asset_class")), 24)
            quality = compact(usable_route_label(route.get("source_quality_label") or normalized.get("source_quality_label")), 28)
            contract_state = decision_contract_state(payload, normalized)
            if not lane:
                unresolved_routes.append(f"{profile}:{symbol}:{path.parent.name}")
                lane = "unknown"
            key = (symbol, profile, status, action)
            if key in seen:
                continue
            seen.add(key)
            parts = [
                "[decision-latest]",
                "level=flow",
                f"age={age_text(timestamp) if parsed_timestamp else age_text(file_timestamp.isoformat())}",
                f"age_source={age_source}",
                f"file_age={age_text(file_timestamp.isoformat())}",
                f"profile={profile}",
                f"broker={broker}" if broker else "",
                f"lane={lane}",
                f"lane_source={lane_source}",
                f"asset={asset_class}" if asset_class else "",
                f"quality={quality}" if quality else "",
                f"symbol={symbol}",
                f"status={status}" if status else "",
                f"action={action}",
                f"score={score}" if score else "",
                f"threshold={threshold}" if threshold else "",
                f"schema={contract_state}",
                f"file={path.parent.name}/{path.name}",
            ]
            rows.append((parsed, " ".join(part for part in parts if part)))
            if len(rows) >= max_lines:
                break
        if len(rows) >= max_lines:
            break
    if not rows:
        print("[decision-latest] status=none reason=no_recent_decision_jsonl")
        return
    for _timestamp, line in sorted(rows, key=lambda item: item[0], reverse=True)[:max_lines]:
        print(line)
    if unresolved_routes:
        examples = ",".join(dict.fromkeys(unresolved_routes))
        print(
            "[decision-route] level=watch status=degraded "
            f"unresolved_count={len(unresolved_routes)} "
            f"examples={compact(examples, 160)} "
            "action=inspect_data_route_contract"
        )


def emit_paper() -> None:
    lane = load_json(health / "execution_lane_paper_latest.json")
    if lane:
        gateway = lane.get("execution_gateway") if isinstance(lane.get("execution_gateway"), dict) else {}
        print(
            "[paper] "
            f"level={'ok' if lane.get('auth_ok') and not lane.get('auth_error') else 'alert'} "
            f"age={age_text(lane.get('timestamp_utc'))} "
            f"mode={lane.get('mode', 'paper')} "
            f"processed={lane.get('processed_count', '')} "
            f"pending={lane.get('pending_rows', '')} "
            f"pending_unknown={as_bool(lane.get('pending_rows_unknown'))} "
            f"approved_intents={gateway.get('approved_intents', '')} "
            f"pre_trade_orders={gateway.get('pre_trade_orders', '')} "
            f"auth_ok={as_bool(lane.get('auth_ok'))} "
            f"auth_error={compact(lane.get('auth_error'), 72)}"
        )
    standard = load_json(health / "paper_live_data_standard_latest.json")
    if standard:
        counts = standard.get("counts_after") if isinstance(standard.get("counts_after"), dict) else {}
        safety = standard.get("safety_contract") if isinstance(standard.get("safety_contract"), dict) else {}
        print(
            "[paper-data] "
            f"level={'ok' if str(standard.get('overall_status') or '').lower() == 'ready' else 'watch'} "
            f"age={age_text(standard.get('timestamp_utc'))} "
            f"status={standard.get('overall_status', '')} "
            f"collectors={counts.get('data_collection_active_bots', '')} "
            f"paper_enabled={counts.get('paper_live_data_enabled_bots', '')} "
            f"live_execution={as_bool(safety.get('live_execution_allowed'))} "
            f"market_data_only={safety.get('market_data_only', '')}"
        )
    profit = load_json(health / "paper_runtime_profitability_controls_latest.json")
    if profit:
        current = profit.get("current") if isinstance(profit.get("current"), dict) else {}
        low = profit.get("low_grade_layer_summary") if isinstance(profit.get("low_grade_layer_summary"), dict) else {}
        controlled_contract = (
            profit.get("controlled_profitability_grade_contract")
            if isinstance(profit.get("controlled_profitability_grade_contract"), dict)
            else {}
        )
        raw_recovery = (
            profit.get("raw_profitability_a_recovery_contract")
            if isinstance(profit.get("raw_profitability_a_recovery_contract"), dict)
            else {}
        )
        raw_improvement = (
            profit.get("raw_profitability_improvement_contract")
            if isinstance(profit.get("raw_profitability_improvement_contract"), dict)
            else {}
        )
        exact_gate = (
            controlled_contract.get("exact_raw_upgrade_gate")
            if isinstance(controlled_contract.get("exact_raw_upgrade_gate"), dict)
            else {}
        )
        current_gap = (
            exact_gate.get("current_gap_to_next_grade")
            if isinstance(exact_gate.get("current_gap_to_next_grade"), dict)
            else {}
        )
        gap_to_raw_a = (
            raw_recovery.get("gap_to_raw_a")
            if isinstance(raw_recovery.get("gap_to_raw_a"), dict)
            else {}
        )
        burn_down = (
            raw_improvement.get("burn_down_contract")
            if isinstance(raw_improvement.get("burn_down_contract"), dict)
            else {}
        )
        recovery_current = (
            raw_recovery.get("current")
            if isinstance(raw_recovery.get("current"), dict)
            else {}
        )
        burn_down_current = (
            burn_down.get("current")
            if isinstance(burn_down.get("current"), dict)
            else {}
        )
        runtime_enforcement = (
            raw_improvement.get("runtime_enforcement")
            if isinstance(raw_improvement.get("runtime_enforcement"), dict)
            else {}
        )
        raw_grade = str(profit.get("raw_profitability_grade") or "").strip().upper()
        controlled_grade = str(
            profit.get("controlled_profitability_grade")
            or profit.get("controlled_financial_grade")
            or low.get("control_posture_grade")
            or ""
        ).strip().upper()
        raw_evidence_based = bool(
            raw_recovery.get("raw_grade_remains_evidence_based")
            or raw_improvement.get("raw_grade_remains_evidence_based")
            or (
                isinstance(controlled_contract.get("runtime_enforcement"), dict)
                and controlled_contract["runtime_enforcement"].get("do_not_raise_raw_financial_grade_without_pnl_evidence")
            )
        )
        control_ready = bool(
            controlled_grade in {"A", "A+", "A++"}
            and (
                controlled_contract.get("control_ready", True)
                or raw_improvement.get("control_ready", False)
                or raw_recovery.get("active", False)
            )
        )
        raw_gap_to_a = (
            current_gap.get("net_pnl_needed")
            if current_gap.get("net_pnl_needed") is not None
            else gap_to_raw_a.get("net_pnl_gap")
        )
        if raw_gap_to_a is None:
            raw_gap_to_a = burn_down.get("net_pnl_gap_to_raw_a")
        raw_state = "ready"
        if raw_grade and raw_grade not in {"A", "A+", "A++"}:
            raw_state = "recovery_debt" if control_ready and raw_evidence_based else "needs_attention"
        raw_blocking_soak = raw_state == "needs_attention"
        print(
            "[paper-profit] "
            f"level={'alert' if raw_state == 'needs_attention' else ('watch' if raw_state == 'recovery_debt' else 'ok')} "
            f"age={age_text(profit.get('timestamp_utc'))} "
            f"grade={profit.get('profitability_display_grade') or profit.get('profitability_grade', '')} "
            f"raw={raw_grade} "
            f"raw_state={raw_state} "
            f"raw_blocking_soak={as_bool(raw_blocking_soak)} "
            f"raw_gap_to_a={as_num(raw_gap_to_a)} "
            f"control={controlled_grade or low.get('control_posture_grade', '')} "
            f"weak_zero_entry={as_bool(runtime_enforcement.get('block_new_entries_on_weak_profiles'))} "
            f"reduce_only_open={as_bool(runtime_enforcement.get('keep_sells_and_reduce_only_paths_open'))} "
            f"net_pnl={as_num(current.get('portfolio_net_pnl_total') or current.get('net_pnl') or recovery_current.get('net_pnl') or burn_down_current.get('net_pnl'))} "
            f"low_grade_blockers={low.get('active_blocker_count', '')}"
        )
    truth = load_json(health / "paper_execution_truth_layer_latest.json")
    if truth:
        failed = truth.get("failed_checks") if isinstance(truth.get("failed_checks"), list) else []
        warnings = truth.get("warnings") if isinstance(truth.get("warnings"), list) else []
        advisories = truth.get("advisory_warnings") if isinstance(truth.get("advisory_warnings"), list) else []
        print(
            "[paper-truth] "
            f"level={'ok' if str(truth.get('overall_status') or '').lower() == 'ready' and not failed else ('alert' if failed else 'watch')} "
            f"age={age_text(truth.get('timestamp_utc'))} "
            f"status={truth.get('overall_status', '')} "
            f"grade={truth.get('grade', '')} "
            f"score={as_num(truth.get('score'))} "
            f"score_dimension={truth.get('score_dimension', 'legacy')} "
            f"raw_metric={as_num(truth.get('raw_metric_score'))} "
            f"promotion={truth.get('promotion_status', '')} "
            f"promotion_score={as_num(truth.get('promotion_evidence_score'))} "
            f"failed={compact(','.join(str(item) for item in failed), 90)} "
            f"warnings={compact(','.join(str(item) for item in warnings), 90)} "
            f"advisories={compact(','.join(str(item) for item in advisories), 120)}"
        )


emit_paper()
emit_decisions()
PY
}

ops_pat='AllSleevesLock|PREFLIGHT|IncidentSnapshot|process_watchdog|sql_link_writer_service|ShadowLoop|AdaptiveInterval|IngestionBackpressure'
fx_ops_pat='FXSession|Starting FX shadow profile|ShadowLoop|AdaptiveInterval|broker_truth|context_only_off_hours'
fx_json_pat='"loop_state":|"state":|"mode":|"off_hours_reason":|"open_now":|"profile": "fx"|"profile":"fx"|"broker":|"symbols_total":|"context_total":|"ok":|"warning_count":|"error":|"reason":|"status":'
json_pat='"timestamp_utc":|"mode":|"status":|"symbol":|"action":|mode=|status=|symbol=|action='
infra_pat='GLOBAL_TRADING_HALT|OPERATOR_STOP|global_halt|halt_state|clear_blockers|hard_gate|health_gate|coordination_state|coordination_status|backpressure|storage_pressure|storage_backpressure|sql_wal_pressure|split_brain|route_verification|auth|OAuth|lease_state|broker_ready|coinbase_api|schwab_auth|one_numbers|command_validity|commands_hygiene|master_infrastructure|system_drift|runtime_gate|operator_cockpit|training_|hdf5|h5|incident_review|chrome_headless|overall_status|recommended_actions|recommended_commands|blocked|degraded|guarded|ready|warning|error|reason|ok'
important_pat='\[Decision\]|\[decision\]|ExecutionIntent|ShadowLoop|RegimeCooldown|AdaptiveInterval|"symbol"[[:space:]]*:|symbol=|GLOBAL_TRADING_HALT|OPERATOR_STOP|global_halt|halt_state|hard_gate|critical|failed|failure|tripwire|killswitch|kill_switch|margin_guard|coordination_status|degraded|guarded|warning|backpressure|storage_pressure|storage_backpressure|low_space|external_low_space|free_below|sql_wal_pressure|split_brain|route_verification|auth|OAuth|lease_state|broker_ready|coinbase_api|schwab_auth|timeout|stale|blocked|\[coord\]|\[storage\]|\[hdf5\]|\[alerts\]|\[notify\]'
if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
  json_pat="$json_pat|\\[Decision\\]"
fi

run_filtered_tail() {
  local pattern="$1"
  if command -v rg >/dev/null 2>&1; then
    if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
      tail_source_follow | rg --line-buffered -i -e "$pattern" | truncate_live_lines
    else
      tail_source_follow | rg --line-buffered -i -e "$pattern" | rg --line-buffered -v '^\[Decision\]' | truncate_live_lines
    fi
  else
    if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
      tail_source_follow | grep --line-buffered -Ei "$pattern" | truncate_live_lines
    else
      tail_source_follow | grep --line-buffered -Ei "$pattern" | grep --line-buffered -Ev '^\[Decision\]' | truncate_live_lines
    fi
  fi
}

run_filtered_snapshot() {
  local pattern="$1"
  local line_limit="${2:-0}"
  if command -v rg >/dev/null 2>&1; then
    if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
      tail_source_snapshot | rg --line-buffered -i -e "$pattern" | truncate_live_lines "$line_limit"
    else
      tail_source_snapshot | rg --line-buffered -i -e "$pattern" | rg --line-buffered -v '^\[Decision\]' | truncate_live_lines "$line_limit"
    fi
  else
    if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
      tail_source_snapshot | grep --line-buffered -Ei "$pattern" | truncate_live_lines "$line_limit"
    else
      tail_source_snapshot | grep --line-buffered -Ei "$pattern" | grep --line-buffered -Ev '^\[Decision\]' | truncate_live_lines "$line_limit"
    fi
  fi
}

drop_stale_bootstrap_state_lines() {
  awk '
    /^\[(ALERT|WATCH|OK|INFO|FLOW)\][[:space:]]+\[(dashboard|storage|throttle|broker|auth|schwab-auth)\]/ { next }
    /^\[(dashboard|storage|throttle|broker|auth|schwab-auth)\]/ { next }
    /^\[(BrokerConfig|StorageRoute)\]/ { next }
    /BrokerConfig/ { next }
    /StorageRoute/ { next }
    { print; fflush() }
  '
}

run_filtered_state_safe_snapshot() {
  local pattern="$1"
  local line_limit="${2:-0}"
  if command -v rg >/dev/null 2>&1; then
    if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
      tail_source_snapshot | rg --line-buffered -i -e "$pattern" | drop_stale_bootstrap_state_lines | truncate_live_lines "$line_limit"
    else
      tail_source_snapshot | rg --line-buffered -i -e "$pattern" | rg --line-buffered -v '^\[Decision\]' | drop_stale_bootstrap_state_lines | truncate_live_lines "$line_limit"
    fi
  else
    if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
      tail_source_snapshot | grep --line-buffered -Ei "$pattern" | drop_stale_bootstrap_state_lines | truncate_live_lines "$line_limit"
    else
      tail_source_snapshot | grep --line-buffered -Ei "$pattern" | grep --line-buffered -Ev '^\[Decision\]' | drop_stale_bootstrap_state_lines | truncate_live_lines "$line_limit"
    fi
  fi
}

filter_pat="$ops_pat|$json_pat"
if [[ "$SOURCE" == "fx" || "$SOURCE" == "all" || "$HEAVY_REQUESTED" == "1" ]]; then
  filter_pat="$filter_pat|$fx_ops_pat|$fx_json_pat"
fi
if [[ "$SOURCE" == "infra" || "$SOURCE" == "all" || "$HEAVY_REQUESTED" == "1" ]]; then
  filter_pat="$filter_pat|$infra_pat"
fi
if [[ "$IMPORTANT_ONLY" == "1" ]]; then
  filter_pat="$important_pat"
fi

if [[ -n "$SYMBOL" ]]; then
  sym_pat='"symbol": "'"$SYMBOL"'"|symbol='"$SYMBOL"
  filter_pat="$filter_pat|$sym_pat"
fi

if [[ "$SNAPSHOT" == "1" ]]; then
  snapshot_line_limit="0"
  if [[ "$HEAVY_REQUESTED" == "1" ]]; then
    snapshot_line_limit="$HEAVY_SNAPSHOT_MAX_LINES"
  fi
  emit_livefeed_status_snapshot | truncate_live_lines 80 || true
  if [[ "$HEAVY_REQUESTED" == "1" && "$INCLUDE_DECISIONS" == "1" ]]; then
    emit_livefeed_decision_paper_snapshot | truncate_live_lines 0 0 || true
  fi
  if [[ "$HEAVY_REQUESTED" == "1" ]]; then
    run_filtered_state_safe_snapshot "$filter_pat" "$snapshot_line_limit" || true
  else
    run_filtered_snapshot "$filter_pat" "$snapshot_line_limit" || true
  fi
  exit 0
fi

if [[ "$HEAVY_REQUESTED" == "1" && "$HEAVY_BOOTSTRAP_SNAPSHOT" == "1" ]]; then
  emit_livefeed_status_snapshot | truncate_live_lines 80 || true
  echo "live_feed_bootstrap_snapshot=begin mode=$TAIL_START_MODE bytes=$HEAVY_TAIL_BYTES"
  run_filtered_state_safe_snapshot "$filter_pat" "$HEAVY_BOOTSTRAP_MAX_LINES" || true
  echo "live_feed_following=1 interrupt=ctrl-c"
else
  emit_livefeed_status_snapshot | truncate_live_lines 80 || true
fi
install_live_feed_trap
start_live_feed_keepalive
if [[ "$HEAVY_REQUESTED" == "1" ]]; then
  emit_live_feed_keepalive "0"
fi
start_heavy_ttl_guard
if [[ "$LIVEFEED_HEALTH_WRITER" == "1" ]]; then
  mirror_restart_sleep="${LIVE_FEED_MIRROR_RESTART_SECONDS:-5}"
  if ! [[ "$mirror_restart_sleep" =~ ^[0-9]+$ ]]; then
    mirror_restart_sleep="5"
  fi
  while true; do
    set +e
    run_filtered_tail "$filter_pat"
    tail_rc=$?
    set -e
    FOLLOW_LAST_RC="$tail_rc"
    FOLLOW_RESTART_COUNT=$((FOLLOW_RESTART_COUNT + 1))
    write_livefeed_health "running"
    printf 'live_feed_follow_restarting rc=%s restart_count=%s sleep_seconds=%s source=%s files=%s\n' \
      "$tail_rc" "$FOLLOW_RESTART_COUNT" "$mirror_restart_sleep" "$SOURCE" "${#files[@]}" >&2
    sleep "$mirror_restart_sleep"
  done
else
  set +e
  run_filtered_tail "$filter_pat"
  tail_rc=$?
  set -e
  cleanup_live_feed
  exit "$tail_rc"
fi
