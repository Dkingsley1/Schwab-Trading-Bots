#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Users/dankingsley/PycharmProjects/schwab_trading_bot"
PRINT_ONLY=0
REPORT_KIND=""
CRASH_LOOKBACK_DAYS="${CRASH_REPORT_OPEN_LOOKBACK_DAYS:-30}"

usage() {
  cat <<'EOF'
Usage: open_report_artifact.sh [--print-only] <report>

Reports:
  summary
  crash
  framework
  special
  posttrade
  training
  retrain
  timeline
  incident
  incident-packet
  paper
  daily-ops
  daily-runtime
  strategy-attribution
  calibration
  daily-auto-verify
  modelcard
  quant
  sentiment
  macro
  source
  replay
  unified
  explainability
  bundle
  report-catalog
  correlation
  botstack
  state-snapshot
  system-overview
  one-numbers
  one-numbers-csv
  strategy-inventory
  expansions
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
  if [[ "$target" == *.pdf ]]; then
    if /usr/bin/open -a Preview "$target"; then
      /usr/bin/osascript -e 'tell application "Preview" to activate' >/dev/null 2>&1 || true
      return 0
    fi
    if /usr/bin/open "$target"; then
      return 0
    fi
    /usr/bin/open -R "$target" >/dev/null 2>&1 || true
    printf 'Could not open PDF automatically. Finder target: %s\n' "$target" >&2
    return 1
  fi
  if /usr/bin/open "$target"; then
    return 0
  fi
  /usr/bin/open -R "$target" >/dev/null 2>&1 || true
  printf 'Could not open artifact automatically. Finder target: %s\n' "$target" >&2
  return 1
}

run_opsctl() {
  if ! (cd "$PROJECT_ROOT" && ./scripts/ops/opsctl.sh "$@" >/dev/null); then
    return 0
  fi
}

run_opsctl_checked() {
  (cd "$PROJECT_ROOT" && ./scripts/ops/opsctl.sh "$@" >/dev/null)
}

run_python_script() {
  local script="$1"
  shift
  local py
  py="$(cd "$PROJECT_ROOT" && zsh ./scripts/ops/runtime_python.sh)"
  if ! (cd "$PROJECT_ROOT" && "$py" "$script" "$@" >/dev/null); then
    return 0
  fi
}

run_python_script_checked() {
  local script="$1"
  shift
  local py
  py="$(cd "$PROJECT_ROOT" && zsh ./scripts/ops/runtime_python.sh)"
  (cd "$PROJECT_ROOT" && "$py" "$script" "$@" >/dev/null)
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-only)
      PRINT_ONLY=1
      ;;
    summary|crash|framework|special|posttrade|training|timeline|incident|incident-packet|paper|daily-ops|daily-runtime|strategy-attribution|calibration|daily-auto-verify|modelcard|quant|sentiment|macro|source|replay|unified|explainability|bundle|report-catalog|correlation|botstack|state-snapshot|system-overview|one-numbers|one-numbers-csv|strategy-inventory|expansions|sendout)
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
  summary)
    run_opsctl system-summary --refresh-supporting-artifacts --render-pdf --allow-gui-pdf-renderer --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/system_summary/system_summary_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/system_summary/system_summary_latest.html")"
    ;;
  crash)
    run_opsctl report-pdfs --only crash_report_digest --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/crash_reports/crash_report_digest_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/crash_reports/crash_report_digest_print_latest.html" \
      "$PROJECT_ROOT/exports/reports/crash_reports/crash_report_digest_latest.md")"
    ;;
  framework)
    run_opsctl system-explainers
    run_opsctl report-pdfs --only framework_map_v2 --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/system_explainers/framework_map_v2_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/system_explainers/framework_map_v2_latest.html")"
    ;;
  special)
    run_opsctl report-pdfs --only special_features --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/showcase/special_features_latest.pdf" \
      "$PROJECT_ROOT/docs/showcase/generated/special_features_latest.html")"
    ;;
  posttrade)
    POST_TRADE_ANALYSIS_SUBCOMMAND_TIMEOUT_SECONDS="${POST_TRADE_ANALYSIS_SUBCOMMAND_TIMEOUT_SECONDS:-12}" run_opsctl post-trade-analysis --day "$(date -u +%Y%m%d)" --hours 24 --json
    run_opsctl report-pdfs --only post_trade_analysis --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/post_trade_analysis_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/pdf_render_sources/post_trade_analysis_latest.html" \
      "$PROJECT_ROOT/exports/reports/post_trade_analysis_latest.md")"
    ;;
  training)
    run_opsctl report-pdfs --only training_report --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/training_reports/training_report_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/training_reports/training_report_print_latest.html" \
      "$PROJECT_ROOT/exports/reports/training_reports/training_report_latest.md")"
    ;;
  retrain)
    run_opsctl report-pdfs --only retrain_scorecard --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/sql_reports/retrain_scorecard_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/pdf_render_sources/retrain_scorecard_latest.html" \
      "$PROJECT_ROOT/exports/sql_reports/retrain_scorecard_latest.md" \
      "$PROJECT_ROOT/governance/health/retrain_scorecard_latest.json")"
    ;;
  timeline)
    run_opsctl report-pdfs --only project_timeline --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/project_timeline/project_timeline_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/project_timeline/project_timeline_print_latest.html" \
      "$PROJECT_ROOT/exports/reports/project_timeline/project_timeline_latest.md")"
    ;;
  incident)
    run_opsctl report-pdfs --only incident_report --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/incident_report_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/incident_report_latest.html" \
      "$PROJECT_ROOT/exports/reports/incident_report_latest.md")"
    ;;
  incident-packet)
    run_opsctl report-pdfs --only incident_review_packet --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/incident_review_packet_latest.pdf" \
      "$PROJECT_ROOT/governance/health/incident_review_packet_latest.json")"
    ;;
  paper)
    run_opsctl paper-performance --day "$(date -u +%Y%m%d)" --week-days 7 --no-allow-gui-pdf-renderer --json
    run_opsctl report-pdfs --only paper_performance --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/paper_performance_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/paper_performance_latest.html" \
      "$PROJECT_ROOT/exports/reports/paper_performance_latest.md")"
    ;;
  daily-ops)
    run_python_script scripts/ops/daily_ops_report.py --json
    run_opsctl report-pdfs --only daily_ops_report --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/daily_ops_report_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/daily_ops_report_latest.md" \
      "$PROJECT_ROOT/exports/reports/daily_ops_report_latest.json")"
    ;;
  daily-runtime)
    run_opsctl report-pdfs --only daily_runtime_summary --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/sql_reports/daily_runtime_summary_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/pdf_render_sources/daily_runtime_summary_latest.html" \
      "$PROJECT_ROOT/governance/health/daily_runtime_summary_latest.json")"
    ;;
  strategy-attribution)
    run_opsctl strategy-attribution --json
    run_opsctl report-pdfs --only strategy_attribution --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/strategy_attribution_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/strategy_attribution_latest.md" \
      "$PROJECT_ROOT/governance/health/strategy_attribution_latest.json")"
    ;;
  calibration)
    run_opsctl report-pdfs --only paper_execution_calibration --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/sql_reports/paper_execution_calibration_latest.pdf" \
      "$PROJECT_ROOT/governance/health/paper_execution_calibration_latest.json")"
    ;;
  daily-auto-verify)
    run_opsctl daily-auto-verify --json
    run_opsctl report-pdfs --only daily_auto_verify --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/sql_reports/daily_auto_verify_latest.pdf" \
      "$PROJECT_ROOT/governance/health/daily_auto_verify_latest.json")"
    ;;
  modelcard)
    run_opsctl report-pdfs --only model_card --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/sql_reports/model_card_latest.pdf" \
      "$PROJECT_ROOT/exports/sql_reports/model_card_latest.json" \
      "$PROJECT_ROOT/governance/health/model_card_latest.json")"
    ;;
  quant)
    run_opsctl quant-model-control --json
    run_opsctl report-pdfs --only quant_model_control --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/quant_model_control/quant_model_control_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/quant_model_control/quant_model_control_latest.md" \
      "$PROJECT_ROOT/governance/health/quant_model_control_latest.json")"
    ;;
  sentiment)
    run_opsctl report-pdfs --only sentiment_report --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/sentiment_report_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/sentiment_report_latest.html" \
      "$PROJECT_ROOT/exports/reports/sentiment_report_latest.md")"
    ;;
  macro)
    run_opsctl macro-crosscheck --json
    run_opsctl report-pdfs --only macro_crosscheck --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/macro_crosscheck_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/macro_crosscheck_latest.md")"
    ;;
  source)
    run_opsctl source-verification --json
    run_opsctl report-pdfs --only source_verification --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/source_verification_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/source_verification_latest.md")"
    ;;
  replay)
    run_python_script scripts/replay_feature_ablation_report.py --json
    run_opsctl report-pdfs --only replay_feature_ablation --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/sql_reports/replay_feature_ablation_latest.pdf" \
      "$PROJECT_ROOT/exports/sql_reports/replay_feature_ablation_latest.md" \
      "$PROJECT_ROOT/exports/sql_reports/replay_feature_ablation_latest.json")"
    ;;
  unified)
    run_opsctl scorecard --json
    run_opsctl report-pdfs --only unified_lane_scorecard --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/sql_reports/unified_lane_scorecard_latest.pdf" \
      "$PROJECT_ROOT/exports/sql_reports/unified_lane_scorecard_latest.md")"
    ;;
  explainability)
    run_opsctl explainability --json
    run_opsctl report-pdfs --only bot_explainability --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/sql_reports/bot_explainability_latest.pdf" \
      "$PROJECT_ROOT/exports/sql_reports/bot_explainability_latest.json" \
      "$PROJECT_ROOT/governance/health/bot_explainability_latest.json")"
    ;;
  bundle|report-catalog)
    run_opsctl report-pdfs --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/report_pdf_bundle_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/report_pdf_bundle_latest.html")"
    ;;
  correlation)
    run_opsctl report-pdfs --only market_crypto_correlation --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/market_crypto_correlation_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/market_crypto_correlation_latest.md")"
    ;;
  botstack)
    run_opsctl report-pdfs --only active_bot_stack --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/bot_stack_status/latest.pdf" \
      "$PROJECT_ROOT/exports/bot_stack_status/latest.html" \
      "$PROJECT_ROOT/exports/bot_stack_status/latest.md")"
    ;;
  state-snapshot)
    run_opsctl report-pdfs --only state_snapshot_drills --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/state_snapshot_drills/state_snapshot_drills_latest.pdf" \
      "$PROJECT_ROOT/exports/state_snapshot_drills/latest.json")"
    ;;
  system-overview)
    run_opsctl report-pdfs --only system_overview --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/system_overview/system_overview_weekly_platform_history_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/system_overview/system_overview_weekly_platform_history_latest.md")"
    ;;
  one-numbers)
    run_python_script scripts/build_one_numbers_report.py
    run_opsctl report-pdfs --only one_numbers --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/one_numbers/one_numbers_latest.pdf" \
      "$PROJECT_ROOT/exports/one_numbers/latest.md" \
      "$PROJECT_ROOT/exports/one_numbers/latest/one_numbers_latest.md" \
      "$PROJECT_ROOT/governance/health/one_numbers_latest.json")"
    ;;
  one-numbers-csv)
    run_python_script scripts/build_one_numbers_report.py
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/one_numbers/latest.csv" \
      "$PROJECT_ROOT/exports/one_numbers/latest/one_numbers_latest.csv")"
    ;;
  strategy-inventory)
    run_opsctl strategy-inventory --json
    run_opsctl report-pdfs --only strategy_inventory --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/strategy_inventory/strategy_inventory_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/strategy_inventory/strategy_inventory_latest.md" \
      "$PROJECT_ROOT/governance/health/strategy_inventory_latest.json")"
    ;;
  expansions)
    run_opsctl expansion-list-report --json
    run_opsctl report-pdfs --only expansion_inventory --json
    REPORT="$(pick_existing \
      "$PROJECT_ROOT/exports/reports/expansion_inventory/expansion_inventory_latest.pdf" \
      "$PROJECT_ROOT/exports/reports/expansion_inventory/expansion_inventory_latest.md" \
      "$PROJECT_ROOT/governance/health/expansion_inventory_latest.json")"
    ;;
  sendout)
    run_opsctl report-pdfs --json
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
