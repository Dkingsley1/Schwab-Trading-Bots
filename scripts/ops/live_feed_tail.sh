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
Usage: scripts/ops/live_feed_tail.sh [--source schwab|coinbase|fx|futures|schwab_futures|coinbase_futures|main|infra|all] [--symbol SYMBOL] [--lines 40] [--raw] [--snapshot] [--include-decisions|--heavy] [--memory-aware|--no-memory-aware] [--include-watchdog-log|--no-watchdog-log]

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
    append_file "$(latest_log "$PROJECT_ROOT/decision_explanations/$dir/decision_explanations_*.jsonl")"
    return
  fi
  append_file "$PROJECT_ROOT/decision_explanations/$dir/decision_explanations_${DAY_LOCAL}.jsonl"
  append_file "$PROJECT_ROOT/decision_explanations/$dir/decision_explanations_${DAY_UTC}.jsonl"
  append_file "$(latest_log "$PROJECT_ROOT/decision_explanations/$dir/decision_explanations_*.jsonl")"
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

if [[ "$HEAVY_REQUESTED" == "1" && "$INCLUDE_DECISIONS" == "1" && "$SOURCE" == "all" ]]; then
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
    printf '"pressure_optimized":%s,' "$PRESSURE_OPTIMIZED"
    printf '"decision_file_mode":"%s",' "$DECISION_FILE_MODE"
    printf '"file_count":%s,' "${#files[@]}"
    printf '"self_throttle":%s,' "$HEAVY_SELF_THROTTLE"
    printf '"nice":%s,' "$HEAVY_NICE"
    printf '"background_policy":%s,' "$HEAVY_BACKGROUND_POLICY"
    printf '"contract":"operator_requested_heavy_observability_view"'
    printf '}\n'
  } > "$HEAVY_MARKER_FILE"
}

write_heavy_marker

echo "live_feed source=$SOURCE local_day=$DAY_LOCAL utc_day=$DAY_UTC symbol=${SYMBOL:-ALL} lines=$LINES heavy=$HEAVY_REQUESTED include_decisions=$INCLUDE_DECISIONS include_watchdog_log=$INCLUDE_WATCHDOG_LOG memory_profile=${MEMORY_PROFILE:-default} memory_aware=$MEMORY_AWARE decision_mode=$DECISION_FILE_MODE pressure_optimized=$PRESSURE_OPTIMIZED file_count=${#files[@]}"
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
      exec tail -n "$LINES" -F "${files[@]}" | rg --line-buffered -i -e "$pattern"
    else
      exec tail -n "$LINES" -F "${files[@]}" | rg --line-buffered -i -e "$pattern" | rg --line-buffered -v '^\[Decision\]'
    fi
  else
    if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
      exec tail -n "$LINES" -F "${files[@]}" | grep --line-buffered -Ei "$pattern"
    else
      exec tail -n "$LINES" -F "${files[@]}" | grep --line-buffered -Ei "$pattern" | grep --line-buffered -Ev '^\[Decision\]'
    fi
  fi
}

run_filtered_snapshot() {
  local pattern="$1"
  if command -v rg >/dev/null 2>&1; then
    if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
      tail -n "$LINES" "${files[@]}" | rg --line-buffered -i -e "$pattern"
    else
      tail -n "$LINES" "${files[@]}" | rg --line-buffered -i -e "$pattern" | rg --line-buffered -v '^\[Decision\]'
    fi
  else
    if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
      tail -n "$LINES" "${files[@]}" | grep --line-buffered -Ei "$pattern"
    else
      tail -n "$LINES" "${files[@]}" | grep --line-buffered -Ei "$pattern" | grep --line-buffered -Ev '^\[Decision\]'
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
  run_filtered_snapshot "$filter_pat"
  exit $?
fi
run_filtered_tail "$filter_pat"
