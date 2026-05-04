#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WATCHDOG_LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"
MEMORY_OVERRIDE_FILE="$PROJECT_ROOT/config/.env.memory_efficiency_override"
HEALTH_DIR="$PROJECT_ROOT/governance/health"
HEAVY_MARKER_FILE="$HEALTH_DIR/live_feed_heavy_view_latest.json"

if [[ -f "$MEMORY_OVERRIDE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$MEMORY_OVERRIDE_FILE"
fi

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
HEAVY_REQUESTED="${LIVE_FEED_HEAVY_DEFAULT:-0}"
COLOR_MODE="${LIVE_FEED_COLOR:-auto}"
COLOR_PALETTE="${LIVE_FEED_COLOR_PALETTE:-red}"

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
    -h|--help)
      cat <<'EOF'
Usage: scripts/ops/live_feed_tail.sh [--source schwab|coinbase|fx|futures|schwab_futures|coinbase_futures|main|infra|all] [--symbol SYMBOL] [--lines 40] [--raw] [--color|--no-color] [--red-only|--semantic-color] [--snapshot] [--include-decisions|--heavy] [--memory-aware|--no-memory-aware] [--include-watchdog-log|--no-watchdog-log]

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
  scripts/ops/live_feed_tail.sh --source all --heavy --color
  scripts/ops/live_feed_tail.sh --source all --heavy --no-memory-aware
  scripts/ops/live_feed_tail.sh --source all --lines 80 --snapshot
  scripts/ops/live_feed_tail.sh --source all --snapshot --include-watchdog-log
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
  red|red_only|red-only|mono|monochrome|semantic)
    ;;
  *)
    echo "--color palette must be red or semantic" >&2
    exit 2
    ;;
esac

if [[ "$SOURCE" == "all" && "$INCLUDE_DECISIONS" == "1" ]]; then
  HEAVY_REQUESTED="1"
fi

if [[ "$HEAVY_REQUESTED" == "1" ]]; then
  INCLUDE_DECISIONS="1"
  if [[ "$INCLUDE_WATCHDOG_LOG_EXPLICIT" != "1" ]]; then
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
MAX_LINE_CHARS="${LIVE_FEED_MAX_LINE_CHARS:-1400}"
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
  renice -n "$HEAVY_NICE" -p $$ >/dev/null 2>&1 || true
  if [[ "$HEAVY_BACKGROUND_POLICY" == "1" ]] && command -v taskpolicy >/dev/null 2>&1; then
    taskpolicy -b -p $$ >/dev/null 2>&1 || true
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
if ! [[ "$DECISION_MAX_AGE_HOURS" =~ ^[0-9]+$ ]]; then
  DECISION_MAX_AGE_HOURS="48"
fi

DAY_UTC="$(date -u +%Y%m%d)"
DAY_LOCAL="$(date +%Y%m%d)"

typeset -a files
typeset -A seen

append_file() {
  local f="$1"
  if [[ -f "$f" && -z "${seen[$f]:-}" ]]; then
    files+=("$f")
    seen[$f]=1
  fi
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
  local out
  setopt localoptions nonomatch
  unsetopt null_glob csh_null_glob
  # ${~pattern} forces zsh glob expansion from a variable.
  out="$(ls -1t ${~pattern} 2>/dev/null | head -n 1 || true)"
  echo "$out"
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

append_all_decision_json_dirs() {
  setopt localoptions null_glob
  local dir_path
  for dir_path in "$PROJECT_ROOT"/decision_explanations/*(/N); do
    append_decision_json_dir "${dir_path:t}"
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
    data_plane_recovery_controller_latest.json \
    live_runtime_separation_control_latest.json \
    ingestion_backpressure_latest.json \
    ingestion_storage_control_latest.json \
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
  append_file "$(latest_log "$PROJECT_ROOT/logs/all_sleeves_*.log")"
  append_file "$(latest_log "$PROJECT_ROOT/logs/shadow_watchdog*.log")"
  if [[ "$INCLUDE_WATCHDOG_LOG" == "1" ]]; then
    append_file "$WATCHDOG_LOG_DIR/shadow_watchdog.out.log"
  fi
}

if [[ "$SOURCE" == "schwab" || "$SOURCE" == "main" || "$SOURCE" == "all" ]]; then
  append_file "$(latest_log "$PROJECT_ROOT/logs/schwab_live_*.log")"
  if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
    for d in shadow_equities shadow_aggressive_equities shadow_conservative_equities shadow_dividend_equities shadow_dividend_capture_equities shadow_bond_equities shadow_intraday_aggressive_equities shadow_swing_aggressive_equities; do
      append_decision_json_dir "$d"
    done
  fi
fi

if [[ "$SOURCE" == "schwab" || "$SOURCE" == "futures" || "$SOURCE" == "schwab_futures" || "$SOURCE" == "main" || "$SOURCE" == "all" ]]; then
  append_file "$(latest_log "$PROJECT_ROOT/logs/schwab_futures_live_*.log")"
  append_health_file "data_ingress_latest_schwab_futures_equities_schwab.json"
  append_health_file "broker_truth_schwab_futures_equities_schwab_latest.json"
  append_file "$(latest_log "$PROJECT_ROOT/governance/health/shadow_loop_schwab_futures_equities_schwab_*.json")"
  if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
    append_decision_json_dir "shadow_schwab_futures_equities"
  fi
fi

if [[ "$SOURCE" == "coinbase" || "$SOURCE" == "main" || "$SOURCE" == "all" ]]; then
  append_file "$(latest_log "$PROJECT_ROOT/logs/coinbase_live_*.log")"
  append_file "$PROJECT_ROOT/logs/watchdog_coinbase_loop.log"
  if [[ "$INCLUDE_WATCHDOG_LOG" == "1" ]]; then
    append_file "$WATCHDOG_LOG_DIR/shadow_watchdog.out.log"
  fi
  append_health_file "data_ingress_latest_crypto_coinbase.json"
  append_health_file "process_watchdog_latest.json"
  append_health_file "shadow_watchdog_tripwire_latest.json"
  if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
    for d in shadow_crypto shadow_coinbase; do
      append_decision_json_dir "$d"
    done
  fi
fi

if [[ "$SOURCE" == "coinbase" || "$SOURCE" == "futures" || "$SOURCE" == "coinbase_futures" || "$SOURCE" == "main" || "$SOURCE" == "all" ]]; then
  append_file "$(latest_log "$PROJECT_ROOT/logs/coinbase_futures_live_*.log")"
  append_health_file "data_ingress_latest_crypto_futures_crypto_coinbase.json"
  append_file "$(latest_log "$PROJECT_ROOT/governance/health/shadow_loop_crypto_futures_crypto_coinbase_*.json")"
  if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
    append_decision_json_dir "shadow_crypto_futures_crypto"
  fi
fi

if [[ "$SOURCE" == "fx" || "$SOURCE" == "all" ]]; then
  append_file "$(latest_log "$PROJECT_ROOT/logs/fx_live_*.log")"
  append_health_file "data_ingress_latest_fx_equities_schwab.json"
  append_health_file "broker_truth_fx_equities_schwab_latest.json"
  append_health_file "fx_shadow_session_latest.json"
  append_health_file "fx_market_context_sync_latest.json"
  append_file "$(latest_log "$PROJECT_ROOT/governance/health/shadow_loop_fx_equities_schwab_*.json")"
  if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
    append_decision_json_dir "shadow_fx_equities"
  fi
fi

if [[ "$HEAVY_REQUESTED" == "1" && "$INCLUDE_DECISIONS" == "1" && "$SOURCE" == "all" && "$HEAVY_INCLUDE_ALL_DECISION_DIRS" == "1" ]]; then
  append_all_decision_json_dirs
fi

if [[ "$SOURCE" == "infra" || "$SOURCE" == "all" || "$HEAVY_REQUESTED" == "1" ]]; then
  append_heavy_health_files
fi

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
    printf '"pid":%s,' "$$"
    printf '"source":"%s",' "$SOURCE"
    printf '"lines":%s,' "$LINES"
    printf '"include_decisions":%s,' "$INCLUDE_DECISIONS"
    printf '"include_watchdog_log":%s,' "$INCLUDE_WATCHDOG_LOG"
    printf '"memory_aware":%s,' "$MEMORY_AWARE"
    printf '"highlight_enabled":%s,' "$COLOR_ENABLED"
    printf '"highlight_mode":"%s",' "$COLOR_MODE"
    printf '"highlight_palette":"%s",' "$COLOR_PALETTE"
    printf '"pressure_optimized":%s,' "$PRESSURE_OPTIMIZED"
    printf '"decision_file_mode":"%s",' "$DECISION_FILE_MODE"
    printf '"file_count":%s,' "${#files[@]}"
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

echo "live_feed source=$SOURCE local_day=$DAY_LOCAL utc_day=$DAY_UTC symbol=${SYMBOL:-ALL} lines=$LINES heavy=$HEAVY_REQUESTED include_decisions=$INCLUDE_DECISIONS include_watchdog_log=$INCLUDE_WATCHDOG_LOG memory_profile=${MEMORY_PROFILE:-default} memory_aware=$MEMORY_AWARE highlight=$COLOR_ENABLED highlight_palette=$COLOR_PALETTE decision_mode=$DECISION_FILE_MODE pressure_optimized=$PRESSURE_OPTIMIZED file_count=${#files[@]} tail_start_mode=$TAIL_START_MODE tail_start_bytes=$HEAVY_TAIL_BYTES max_line_chars=$MAX_LINE_CHARS"
for f in "${files[@]}"; do
  echo " - $f"
done

if [[ "$RAW" == "1" ]]; then
  if [[ "$SNAPSHOT" == "1" ]]; then
    tail -n "$LINES" "${files[@]}"
    exit $?
  fi
  exec tail -n "$LINES" -F "${files[@]}"
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
    tail -c "$HEAVY_TAIL_BYTES" -F "${files[@]}"
    return $?
  fi
  tail -n "$LINES" -F "${files[@]}"
}

truncate_live_lines() {
  local line_limit="${1:-0}"
  awk -v max="$MAX_LINE_CHARS" -v limit="$line_limit" -v color="$COLOR_ENABLED" -v palette="$COLOR_PALETTE" '
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
      if (palette == "red" || palette == "red_only" || palette == "red-only" || palette == "mono" || palette == "monochrome") {
        green = red
        yellow = red
        blue = red
        magenta = red
        cyan = red
      }
    } else {
      reset = bold = dim = red = green = yellow = blue = magenta = cyan = ""
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
  function highlight_token(line, pattern, paint) {
    if (color != "1") return line
    gsub(pattern, paint "&" reset, line)
    return line
  }
  function blocked_metric_is_clear(lower) {
    return lower ~ /blocked/ && (lower ~ /false/ || lower ~ /"[[:space:]]*0[[:space:]]*"/ || lower ~ /"[[:space:]]*0\.0+[[:space:]]*"/ || lower ~ /:[[:space:]]*0([,}[:space:]]|$)/ || lower ~ /:[[:space:]]*0\.0+([,}[:space:]]|$)/)
  }
  function colorize_line(line, lower, prefix, clear_blocked_metric, red_alert) {
    if (color != "1") return line
    lower = tolower(line)
    prefix = ""
    clear_blocked_metric = blocked_metric_is_clear(lower)
    red_alert = lower ~ /global_trading_halt|operator_stop|halt=true|halt_state[^a-z0-9_]*active|hard_gate|critical|failed|tripwire|killswitch|kill_switch|margin_guard/
    if (!clear_blocked_metric && lower ~ /blocked/) {
      red_alert = 1
    }
    if (red_alert) {
      prefix = bold red "[ALERT] " reset
    } else if (lower ~ /degraded|warning|backpressure|storage_pressure|sql_wal_pressure|awaiting|timeout|throttle|pressure|stale/) {
      prefix = bold yellow "[WATCH] " reset
    } else if (lower ~ /clear_ready|healthy|heartbeat_ok|broker_ready|ready|status[=:][[:space:]]*ok|\"ok\"[[:space:]]*:[[:space:]]*true/) {
      prefix = bold green "[OK] " reset
    } else if (line ~ /\[Decision\]|\[decision\]|ShadowLoop|ExecutionIntent|RegimeCooldown|AdaptiveInterval/) {
      prefix = bold cyan "[FLOW] " reset
    }
    line = highlight_token(line, "GLOBAL_TRADING_HALT|OPERATOR_STOP|halt=true|failed|critical|tripwire|killswitch|margin_guard", bold red)
    if (!clear_blocked_metric) {
      line = highlight_token(line, "blocked", bold red)
    }
    line = highlight_token(line, "degraded|warning|backpressure|storage_pressure|sql_wal_pressure|awaiting|timeout|throttle|pressure|stale", bold yellow)
    line = highlight_token(line, "clear_ready|healthy|heartbeat_ok|broker_ready|ready", bold green)
    line = highlight_token(line, "BUY", bold green)
    line = highlight_token(line, "SELL", bold red)
    line = highlight_token(line, "HOLD", bold blue)
    line = highlight_token(line, "symbol=|action=|mode=|status=|score=", cyan)
    return prefix line
  }
  {
    count += 1
    if (limit > 0 && count > limit) {
      print colorize_line("live_feed_output_truncated line_limit=" limit)
      fflush()
      exit 0
    }
    if (length($0) > max && $0 ~ /^\{/) {
      ts = text_field($0, "timestamp_utc")
      mode = text_field($0, "mode")
      status = text_field($0, "status")
      symbol = text_field($0, "symbol")
      action = text_field($0, "action")
      strategy = text_field($0, "strategy")
      score = num_field($0, "model_score")
      if (ts == "") ts = "?"
      if (mode == "") mode = "?"
      if (status == "") status = "?"
      if (symbol == "") symbol = "?"
      if (action == "") action = "?"
      if (strategy == "") strategy = "?"
      if (score == "") score = "?"
      print colorize_line("[decision] ts=" ts " mode=" mode " status=" status " symbol=" symbol " action=" action " strategy=" strategy " score=" score " [compacted length=" length($0) "]")
      fflush()
      next
    }
    if (length($0) > max) {
      print colorize_line(substr($0, 1, max) "... [truncated length=" length($0) "]")
    } else {
      print colorize_line($0)
    }
    fflush()
  }'
}

ops_pat='AllSleevesLock|PREFLIGHT|IncidentSnapshot|process_watchdog|sql_link_writer_service|ShadowLoop|AdaptiveInterval|IngestionBackpressure'
fx_ops_pat='FXSession|Starting FX shadow profile|ShadowLoop|AdaptiveInterval|broker_truth|context_only_off_hours'
fx_json_pat='"loop_state":|"state":|"mode":|"off_hours_reason":|"open_now":|"profile": "fx"|"profile":"fx"|"broker":|"symbols_total":|"context_total":|"ok":|"warning_count":|"error":|"reason":|"status":'
json_pat='"timestamp_utc":|"mode":|"status":|"symbol":|"action":|mode=|status=|symbol=|action='
infra_pat='GLOBAL_TRADING_HALT|OPERATOR_STOP|global_halt|halt_state|clear_blockers|hard_gate|health_gate|backpressure|storage_pressure|storage_backpressure|sql_wal_pressure|split_brain|route_verification|auth|OAuth|lease_state|broker_ready|coinbase_api|schwab_auth|one_numbers|command_validity|commands_hygiene|master_infrastructure|system_drift|runtime_gate|operator_cockpit|training_|incident_review|chrome_headless|overall_status|recommended_actions|recommended_commands|blocked|degraded|ready|warning|error|reason|ok'
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

filter_pat="$ops_pat|$json_pat"
if [[ "$SOURCE" == "fx" || "$SOURCE" == "all" || "$HEAVY_REQUESTED" == "1" ]]; then
  filter_pat="$filter_pat|$fx_ops_pat|$fx_json_pat"
fi
if [[ "$SOURCE" == "infra" || "$SOURCE" == "all" || "$HEAVY_REQUESTED" == "1" ]]; then
  filter_pat="$filter_pat|$infra_pat"
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
  run_filtered_snapshot "$filter_pat" "$snapshot_line_limit" || true
  exit 0
fi

if [[ "$HEAVY_REQUESTED" == "1" && "$HEAVY_BOOTSTRAP_SNAPSHOT" == "1" ]]; then
  echo "live_feed_bootstrap_snapshot=begin mode=$TAIL_START_MODE bytes=$HEAVY_TAIL_BYTES"
  run_filtered_snapshot "$filter_pat" "$HEAVY_BOOTSTRAP_MAX_LINES" || true
  echo "live_feed_following=1 interrupt=ctrl-c"
fi
run_filtered_tail "$filter_pat"
