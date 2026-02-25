#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNBOOK="$PROJECT_ROOT/COMMANDS.md"

if [[ ! -f "$RUNBOOK" ]]; then
  echo "Missing runbook: $RUNBOOK"
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  ./scripts/runbook.sh                 # show section list
  ./scripts/runbook.sh all             # show full COMMANDS.md
  ./scripts/runbook.sh <section>

Sections:
  start
  watchdog
  retrain
  retrain-override
  validate
  floor
  floor-override
  sql
  portfolio
  dashboard
  daily
  locks
  gotchas
  distill
EOF
}

extract_heading() {
  local heading="$1"
  heading="${heading//\\/}"
  awk -v h="$heading" '
    $0 == "## " h {show=1; print; next}
    /^## / && show==1 {exit}
    show==1 {print}
  ' "$RUNBOOK"
}

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

case "$1" in
  all)
    cat "$RUNBOOK"
    ;;
  start)
    extract_heading "1\) Start All Sleeves \(single live terminal feed\)"
    ;;
  watchdog)
    extract_heading "2\) Watchdog Health \(one-shot\)"
    ;;
  retrain)
    extract_heading "3\) Retrain \(normal, after-hours gate enabled\)"
    ;;
  retrain-override)
    extract_heading "4\) Retrain Manual Override \(run during market hours\)"
    ;;
  validate)
    extract_heading "5\) Walk-Forward + Promotion Gate"
    ;;
  floor)
    extract_heading "6\) Raise Floor \(canary gate enforced\)"
    ;;
  floor-override)
    extract_heading "7\) Raise Floor \(manual override if canary is blocked\)"
    ;;
  sql)
    extract_heading "8\) SQL + One Numbers Refresh"
    ;;
  portfolio)
    extract_heading "9\) Allocator / Risk / Budget \(portfolio control layer\)"
    ;;
  dashboard)
    extract_heading "10\) Executive Dashboard \(Numbers-friendly\)"
    ;;
  daily)
    extract_heading "11\) Daily Full Pipeline"
    ;;
  locks)
    extract_heading "12\) Lock Troubleshooting"
    ;;
  gotchas)
    extract_heading "13\) Common Gotchas"
    ;;
  distill)
    extract_heading "14\) Distillation-Enabled Retrain \(teachers -> new bots\)"
    ;;
  *)
    echo "Unknown section: $1"
    usage
    exit 2
    ;;
esac
