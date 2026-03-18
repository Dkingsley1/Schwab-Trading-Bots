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
  live
  refresh
  health
  retrain
  analysis
  reports
  halts
  sim-paper
  gotchas
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
  live)
    extract_heading "Live Starts"
    ;;
  refresh)
    extract_heading "Live Feed Refresh"
    ;;
  retrain)
    extract_heading "Retrain"
    ;;
  health)
    extract_heading "Health And SQL"
    ;;
  analysis)
    extract_heading "Model And Analysis"
    ;;
  reports)
    extract_heading "Reports"
    ;;
  halts)
    extract_heading "Halts And Recovery"
    ;;
  sim-paper)
    extract_heading "Sim And Paper"
    ;;
  gotchas)
    extract_heading "Common Gotchas"
    ;;
  *)
    echo "Unknown section: $1"
    usage
    exit 2
    ;;
esac
