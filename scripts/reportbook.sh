#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNBOOK="$PROJECT_ROOT/REPORTS.md"

if [[ ! -f "$RUNBOOK" ]]; then
  echo "Missing reportbook: $RUNBOOK"
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  ./scripts/reportbook.sh                 # show section list
  ./scripts/reportbook.sh all             # show full REPORTS.md
  ./scripts/reportbook.sh <section>

Sections:
  bundle
  ops
  training
  analysis
  forensics
  notes
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
  bundle)
    extract_heading "Bundle Page"
    ;;
  ops)
    extract_heading "Operations"
    ;;
  training)
    extract_heading "Training"
    ;;
  analysis)
    extract_heading "Analysis"
    ;;
  forensics)
    extract_heading "Forensics"
    ;;
  notes)
    extract_heading "Notes"
    ;;
  *)
    echo "Unknown section: $1"
    usage
    exit 2
    ;;
esac
