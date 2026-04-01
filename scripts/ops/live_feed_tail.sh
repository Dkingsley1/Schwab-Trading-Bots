#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WATCHDOG_LOG_DIR="$HOME/Library/Logs/schwab_trading_bot"

SOURCE="schwab"
SYMBOL=""
LINES="40"
RAW="0"
SNAPSHOT="0"
INCLUDE_DECISIONS="${LIVE_FEED_INCLUDE_DECISIONS_DEFAULT:-0}"

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
    --no-decisions)
      INCLUDE_DECISIONS="0"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/ops/live_feed_tail.sh [--source schwab|coinbase|fx|futures|schwab_futures|coinbase_futures|main|all] [--symbol SYMBOL] [--lines 40] [--raw] [--snapshot] [--include-decisions]

Examples:
  scripts/ops/live_feed_tail.sh
  scripts/ops/live_feed_tail.sh --symbol SPY
  scripts/ops/live_feed_tail.sh --source coinbase
  scripts/ops/live_feed_tail.sh --source fx
  scripts/ops/live_feed_tail.sh --source futures
  scripts/ops/live_feed_tail.sh --source main --lines 80
  scripts/ops/live_feed_tail.sh --source all --lines 80
  scripts/ops/live_feed_tail.sh --source all --lines 80 --include-decisions
  scripts/ops/live_feed_tail.sh --source all --lines 80 --snapshot
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

if [[ "$SOURCE" != "schwab" && "$SOURCE" != "coinbase" && "$SOURCE" != "fx" && "$SOURCE" != "futures" && "$SOURCE" != "schwab_futures" && "$SOURCE" != "coinbase_futures" && "$SOURCE" != "main" && "$SOURCE" != "all" ]]; then
  echo "--source must be schwab, coinbase, fx, futures, schwab_futures, coinbase_futures, main, or all" >&2
  exit 2
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
  # ${~pattern} forces zsh glob expansion from a variable.
  out="$(ls -1t ${~pattern} 2>/dev/null | head -n 1 || true)"
  echo "$out"
}

append_decision_json_dir() {
  local dir="$1"
  append_file "$PROJECT_ROOT/decision_explanations/$dir/decision_explanations_${DAY_LOCAL}.jsonl"
  append_file "$PROJECT_ROOT/decision_explanations/$dir/decision_explanations_${DAY_UTC}.jsonl"
  append_file "$(latest_log "$PROJECT_ROOT/decision_explanations/$dir/decision_explanations_*.jsonl")"
}

append_health_file() {
  local f="$1"
  append_file "$PROJECT_ROOT/governance/health/$f"
}

if [[ "$SOURCE" == "schwab" || "$SOURCE" == "main" || "$SOURCE" == "all" ]]; then
  append_file "$(latest_log "$PROJECT_ROOT/logs/schwab_live_*.log")"
  if [[ "$INCLUDE_DECISIONS" == "1" ]]; then
    for d in shadow_equities shadow_aggressive_equities shadow_dividend_equities shadow_bond_equities shadow_intraday_aggressive_equities shadow_swing_aggressive_equities; do
      append_decision_json_dir "$d"
    done
  fi
fi

if [[ "$SOURCE" == "schwab" || "$SOURCE" == "futures" || "$SOURCE" == "schwab_futures" || "$SOURCE" == "main" || "$SOURCE" == "all" ]]; then
  append_file "$(latest_log "$PROJECT_ROOT/logs/schwab_futures_live_*.log")"
  append_health_file "data_ingress_latest_schwab_futures_equities_schwab.json"
  append_health_file "broker_truth_schwab_futures_equities_schwab_latest.json"
  append_file "$(latest_log "$PROJECT_ROOT/governance/health/shadow_loop_schwab_futures_equities_schwab_*.json")"
fi

if [[ "$SOURCE" == "coinbase" || "$SOURCE" == "main" || "$SOURCE" == "all" ]]; then
  append_file "$(latest_log "$PROJECT_ROOT/logs/coinbase_live_*.log")"
  append_file "$PROJECT_ROOT/logs/watchdog_coinbase_loop.log"
  append_file "$WATCHDOG_LOG_DIR/shadow_watchdog.out.log"
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
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No live feed files found for source=$SOURCE local_day=$DAY_LOCAL utc_day=$DAY_UTC" >&2
  echo "Start loops first with: $PROJECT_ROOT/scripts/ops/opsctl.sh start" >&2
  exit 1
fi

echo "live_feed source=$SOURCE local_day=$DAY_LOCAL utc_day=$DAY_UTC symbol=${SYMBOL:-ALL} lines=$LINES include_decisions=$INCLUDE_DECISIONS"
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

if [[ -n "$SYMBOL" ]]; then
  sym_pat='"symbol": "'"$SYMBOL"'"|symbol='"$SYMBOL"
  if [[ "$SNAPSHOT" == "1" ]]; then
    run_filtered_snapshot "$ops_pat|$sym_pat"
    exit $?
  fi
  run_filtered_tail "$ops_pat|$sym_pat"
else
  if [[ "$SOURCE" == "fx" ]]; then
    if [[ "$SNAPSHOT" == "1" ]]; then
      run_filtered_snapshot "$fx_ops_pat|$fx_json_pat"
      exit $?
    fi
    run_filtered_tail "$fx_ops_pat|$fx_json_pat"
  fi
  if [[ "$SNAPSHOT" == "1" ]]; then
    run_filtered_snapshot "$ops_pat|$json_pat"
    exit $?
  fi
  run_filtered_tail "$ops_pat|$json_pat"
fi
