#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNBOOK="$PROJECT_ROOT/COMMANDS.md"

if [[ ! -f "$RUNBOOK" ]]; then
  echo "Missing runbook: $RUNBOOK"
  exit 1
fi

slugify() {
  print -r -- "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

list_sections() {
  awk '/^## / { print substr($0, 4) }' "$RUNBOOK"
}

resolve_section() {
  local raw="${1:-}"
  case "$raw" in
    live) print -r -- "Live Feed Views" ;;
    refresh) print -r -- "Live Feed Refreshes" ;;
    health) print -r -- "Status And Health" ;;
    retrain) print -r -- "Retrain" ;;
    analysis) print -r -- "Strategy Research" ;;
    reports) print -r -- "Reports And PDFs" ;;
    halts) print -r -- "Status And Health" ;;
    sim-paper) print -r -- "Most Used" ;;
    *) print -r -- "$raw" ;;
  esac
}

find_section_heading() {
  local requested="$1"
  local resolved
  resolved="$(resolve_section "$requested")"
  while IFS= read -r heading; do
    [[ -n "$heading" ]] || continue
    if [[ "$heading" == "$resolved" || "$(slugify "$heading")" == "$requested" || "$(slugify "$heading")" == "$(slugify "$resolved")" ]]; then
      print -r -- "$heading"
      return 0
    fi
  done < <(list_sections)
  return 1
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/runbook.sh                 # show section list
  ./scripts/runbook.sh all             # show full COMMANDS.md
  ./scripts/runbook.sh <section>

Sections:
EOF
  while IFS= read -r heading; do
    [[ -n "$heading" ]] || continue
    echo "  $(slugify "$heading")"
  done < <(list_sections)
}

extract_heading() {
  local heading="$1"
  awk -v h="$heading" '
    $0 == "## " h { show=1; print; next }
    /^## / && show == 1 { exit }
    show == 1 { print }
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
  *)
    if ! heading="$(find_section_heading "$1")"; then
      echo "Unknown section: $1"
      usage
      exit 2
    fi
    extract_heading "$heading"
    ;;
esac
