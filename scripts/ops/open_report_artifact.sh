#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Users/dankingsley/PycharmProjects/schwab_trading_bot"
PRINT_ONLY=0
REPORT_KIND=""

usage() {
  cat <<'EOF'
Usage: open_report_artifact.sh [--print-only] <report>

Reports:
  crash
  training
  timeline
  paper
  bundle
  correlation
  sendout
EOF
}

pick_existing() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

open_or_print() {
  local target="$1"
  if [[ "$PRINT_ONLY" -eq 1 ]]; then
    printf '%s\n' "$target"
    return 0
  fi
  /usr/bin/open "$target"
}

run_opsctl() {
  (cd "$PROJECT_ROOT" && ./scripts/ops/opsctl.sh "$@" >/dev/null)
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-only)
      PRINT_ONLY=1
      ;;
    crash|training|timeline|paper|bundle|correlation|sendout)
      REPORT_KIND="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -z "$REPORT_KIND" ]]; then
  usage >&2
  exit 2
fi

REPORT=""
case "$REPORT_KIND" in
  crash)
    run_opsctl crash-report --lookback-days 3 --recent-limit 40 --allow-gui-pdf-renderer --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/crash_reports/crash_report_digest_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/crash_reports/crash_report_digest_print_latest.html" \
      "$PROJECT_ROOT/exports/reports/crash_reports/crash_report_digest_latest.md")"
    ;;
  training)
    run_opsctl training-report --allow-gui-pdf-renderer --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/training_reports/training_report_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/training_reports/training_report_print_latest.html" \
      "$PROJECT_ROOT/exports/reports/training_reports/training_report_latest.md")"
    ;;
  timeline)
    run_opsctl timeline-report --render-pdf --allow-gui-pdf-renderer --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/project_timeline/project_timeline_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/project_timeline/project_timeline_print_latest.html" \
      "$PROJECT_ROOT/exports/reports/project_timeline/project_timeline_latest.md")"
    ;;
  paper)
    run_opsctl paper-performance --day "$(date -u +%Y%m%d)" --week-days 7 --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/paper_performance_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/paper_performance_latest.html" \
      "$PROJECT_ROOT/exports/reports/paper_performance_latest.md")"
    ;;
  bundle)
    run_opsctl report-pdfs --allow-gui-pdf-renderer --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/report_pdf_bundle_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/report_pdf_bundle_latest.html")"
    ;;
  correlation)
    run_opsctl report-pdfs --allow-gui-pdf-renderer --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/market_crypto_correlation_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/market_crypto_correlation_latest.md")"
    ;;
  sendout)
    run_opsctl crash-report --lookback-days 3 --recent-limit 40 --allow-gui-pdf-renderer --json
    run_opsctl training-report --allow-gui-pdf-renderer --json
    run_opsctl timeline-report --render-pdf --allow-gui-pdf-renderer --json
    run_opsctl paper-performance --day "$(date -u +%Y%m%d)" --week-days 7 --json
    run_opsctl report-pdfs --allow-gui-pdf-renderer --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/report_pdf_bundle_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/report_pdf_bundle_latest.html")"
    ;;
esac

if [[ -z "$REPORT" ]]; then
  echo "No report artifact found for: $REPORT_KIND" >&2
  exit 1
fi

open_or_print "$REPORT"
