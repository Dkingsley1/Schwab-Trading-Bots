#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

SOURCE="schwab"
SYMBOL=""
LINES="40"
RAW="0"

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
    -h|--help)
      cat <<'EOF'
Usage: scripts/ops/live_feed_tail.sh [--source schwab|coinbase|all] [--symbol NVDA] [--lines 40] [--raw]

Examples:
  scripts/ops/live_feed_tail.sh
  scripts/ops/live_feed_tail.sh --symbol NVDA
  scripts/ops/live_feed_tail.sh --source coinbase
  scripts/ops/live_feed_tail.sh --source all --lines 80
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

if [[ "$SOURCE" != "schwab" && "$SOURCE" != "coinbase" && "$SOURCE" != "all" ]]; then
  echo "--source must be schwab, coinbase, or all" >&2
  exit 2
fi

DAY="$(date -u +%Y%m%d)"

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

if [[ "$SOURCE" == "schwab" || "$SOURCE" == "all" ]]; then
  append_file "$(latest_log "$PROJECT_ROOT/logs/schwab_live_*.log")"
  for d in     shadow_equities     shadow_aggressive_equities     shadow_dividend_equities     shadow_bond_equities     shadow_intraday_aggressive_equities     shadow_swing_aggressive_equities; do
    append_file "$PROJECT_ROOT/decision_explanations/$d/decision_explanations_${DAY}.jsonl"
  done
fi

if [[ "$SOURCE" == "coinbase" || "$SOURCE" == "all" ]]; then
  append_file "$(latest_log "$PROJECT_ROOT/logs/coinbase_live_*.log")"
  append_file "$PROJECT_ROOT/logs/watchdog_coinbase_loop.log"
  append_file "$PROJECT_ROOT/logs/shadow_watchdog.out.log"
  append_file "$(latest_log "$PROJECT_ROOT/logs/coinbase_futures_live_*.log")"
  for d in shadow_crypto shadow_coinbase shadow_crypto_futures_crypto; do
    append_file "$PROJECT_ROOT/decision_explanations/$d/decision_explanations_${DAY}.jsonl"
  done
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No live feed files found for source=$SOURCE day=$DAY" >&2
  echo "Start loops first with: $PROJECT_ROOT/scripts/ops/opsctl.sh start" >&2
  exit 1
fi

echo "live_feed source=$SOURCE day=$DAY symbol=${SYMBOL:-ALL} lines=$LINES"
for f in "${files[@]}"; do
  echo " - $f"
done

if [[ "$RAW" == "1" ]]; then
  exec tail -n "$LINES" -F "${files[@]}"
fi

ops_pat='AllSleevesLock|PREFLIGHT|IncidentSnapshot|StorageRoute|process_watchdog|sql_link_writer_service'
json_pat='"timestamp_utc":|"mode":|"status":|"symbol":|"action":'

if [[ -n "$SYMBOL" ]]; then
  sym_pat='"symbol": "'"$SYMBOL"'"|symbol='"$SYMBOL"
  if command -v rg >/dev/null 2>&1; then
    exec tail -n "$LINES" -F "${files[@]}" | rg --line-buffered -i -e "$ops_pat|$sym_pat"
  else
    exec tail -n "$LINES" -F "${files[@]}" | grep --line-buffered -Ei "$ops_pat|$sym_pat"
  fi
else
  if command -v rg >/dev/null 2>&1; then
    exec tail -n "$LINES" -F "${files[@]}" | rg --line-buffered -i -e "$ops_pat|$json_pat"
  else
    exec tail -n "$LINES" -F "${files[@]}" | grep --line-buffered -Ei "$ops_pat|$json_pat"
  fi
fi
