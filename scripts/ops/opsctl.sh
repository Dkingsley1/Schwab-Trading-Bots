#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/ops/runtime_python.sh"
PY="$(resolve_runtime_python)"
TRAINING_PY="$(resolve_training_python)"

cmd="${1:-help}"
shift || true

PROFILE="${BOT_RUNTIME_PROFILE:-live}"
case "$cmd" in
  status|start|start-sim|start-live|restart-sanity|post-restart-settle|post-restart-settlement|restart-settlement|sql-sync|tradingeconomics-sync|macro-context-sync|market-micro-sync|sec-edgar-sync|extended-quant-sync|quant-model-control|quant-engine-control|quant-models|pricing-grad|gpu-mc-sim|kalman-parallel|options-flow-sync|tastytrade-sync|options-flow-export-hygiene|options-flow-efficiency|bot-stack-report|active-bot-report|core-bot-catalog|bot-catalog|core-bot-materialize|materialize-core-bots|core-bot-materialization-guard|core-bot-file-guard|core-bot-materialization-infrabot|core-bot-file-infrabot|core-bot-tier-organizer|organize-core-bot-tiers|bot-tier-organizer|crypto-market-sync|market-correlation-sync|fx-market-sync|dividend-drip-sync|showcase-refresh|system-explainers|system-explainer-docs|system-summary|executive-summary|system-summary-autopilot|master-infra-supervisor|master-infrastructure-supervisor|infra-supervisor|macro-crosscheck|source-verification|collector-contracts|sleeve-strategy-coverage|sleeve-coverage|strategy-coverage|mlx-audit|mlx-library-upgrade|mlx-upgrade|mlx-audio-audit|mlx-intelligence-router|mlx-compute-brain|mlx-utilization|library-utilization-router|library-router|non-mlx-library-router|dependency-utilization|onnx-audit|pytorch-audit|torch-audit|pytorch-replay-canary|torch-replay-canary|sql-audit|training-registry-audit|training-label-audit|training-quality|feature-store|multiple-testing-guard|decay-monitor|security-audit|security-evidence-autofix|secret-scan|schema-migration|ingestion-storage-control|ingestion-storage-governor|external-backlog-drain|backpressure-drainer-fleet|backpressure-drainers|drainer-fleet|external-backlog-retry-bot|storage-backpressure-autopilot|storage-pressure-clearance|storage-pressure-clear|storage-pressure-supervisor|storage-clearance|storage-reconnect-regression-guard|storage-reconnect-guard|storage-reconnect-infrabot|storage-recovery-infrabot|storage-auto-recovery-bot|stateful-storage-regression-guard|stateful-storage-guard|local-stateful-guard|data-retention|retention-policy|data-retention-policy|stale-sweeper|stale-artifact-sweeper|stale-stage-bot|stale-reaper|stale-artifact-reaper|stale-delete-bot|data-collection-storage-guard|collector-storage-guard|data-collection-observation-rollup|collector-observation-rollup|writer-cycle-coordinator|retention-debt-sheriff|backpressure-slo-bot|backlog-quarantine|ingestion-priority-queue|content-store|split-brain-reconcile|storage-resilience|storage-tier-policy|runtime-training-snapshot|training-runtime-control|training-requalification|coverage-seed|coverage-gap-closer|regime-control|supportability-control|teacher-quality|bot-quality-autopilot|commands-hygiene|runbook-hygiene|command-validity|commands-verify|codex-project-guard|codex-guard|infrastructure-autofix|system-cleanliness-autopilot|cleanliness-autopilot|system-cleanliness-infrabot|cleanliness-infrabot|report-quality-guard|reporter-quality|reporter-infrabot|report-infrabot|system-drift-guard|drift-guard|drift-mesh|system-drift-autopilot|drift-autopilot|drift-mesh-autopilot|global-halt|global-halt-status|halt-status|global-halt-refresh|global-halt-clear-blockers|halt-clear-blockers|global-halt-auto-clear|halt-auto-clear|clear-all-halts|clear-global-halts|global-halt-force-clear|halt-force-clear|operator-control|operator-release|operator-stop-status|live-runtime-separation|live-canary-control|rolling-restart|schwab-auth-supervisor|schwab-auth-guard|auth-supervisor|auth-lease|incident-timeline|incident-closeout|incident-review-packet|incident-packet|promotion-autopilot|autonomy-control|runtime-throttle|throttle-control|throttle-bot|mode-switchboard|mode-switchboard-mission-control|decision-provenance|decision-provenance-cards|blackstart-recovery|sleeve-isolation|artifact-freshness-slo|runtime-snapshot-cache|remote-alert-control|storage-quota-guard|release-freeze|legacy-bot-harmonizer|legacy-bot-harmonize|legacy-v107-harmonizer|roster-expansion|roster-resilience|chaos-drills|calibration-control|portfolio-allocator|portfolio-capacity-curves|capacity-curves|risk-service|execution-lab|operator-cockpit|daily-verify-remediation|memory-efficiency|creative-cotenant-guard|creative-cotenant|cotenant-guard|model-lifecycle|grade-lift-hardening|a-grade-hardening|grade-up|grade-regression-guard|regression-guard|grade-regression-autopilot|grade-regression-bot|regression-upgrade|section-grade-guard|section-floor-guard|grade-floor-guard|section-grade-autopilot|section-floor-autopilot|grade-floor-autopilot|chrome-headless-guard|chrome-guard|chrome-pdf-guard|cost-telemetry|cross-host-parity|parity-report|experiment-ledger|immutable-experiment-ledger|one-numbers-regression-guard|one-numbers-guard|point-in-time-event-store|pit-event-store|replay-hash-registry|replay-hash-guard|golden-replay-regression|golden-replay-guard|paper-trade-lock-infrabot|paper-lock-infrabot|paper-trade-lock-guard|paper-lock|coinbase-api-health|coinbase-health|coinbase-start|coinbase-futures-start|schwab-futures-start|fx-start|feed-refresh|livefeed-refresh|live-feed-refresh|feed|phone-feed|schwab-tail|coinbase-tail|main-tail|futures-tail|schwab-futures-tail|coinbase-futures-tail|fx-tail|infra-tail|storage-switch-local|storage-switch-external|storage-safe-eject|storage-prune-standby|storage-standby-prune|storage-transition-coordinator|storage-transition-bots|storage-disaster-recovery|storage-recovery-bot|storage-maintenance|ops-coordinator|platform-control-plane|institutional-readiness|intelligence-capability-expansion|capability-expansion|platform-intelligence|platform-intelligence-expansion|bot-admission|new-bot-admission|new-bot-admission-guard|admission-guard|sleeve-masters|quality-score-system|market-regime-router|capacity-planner|expansion-capacity|expansion-capacity-planner|growth-capacity|research-pipeline|correlation-governor|model-decay-detector|system-dashboard|big-platform-brain|platform-brain|system-self-model|self-model|self-awareness|metacognition|self-awareness-infrabots|system-self-awareness|alpha-intelligence-evolution|alpha-advancement|alpha-evolution|alpha-intelligence|training-readiness-brain|execution-reality-lab|portfolio-exposure-brain|data-source-confidence-engine|research-intake-pipeline|duplicate-alpha-novelty|professional-dashboard-v2|broker-adapter-mesh|adapter-mesh|cleanup-governor|intelligence-layer-advancement|intelligence-layer-v2|intelligence-upgrade|meta-intelligence-v2|metacognitive-routing-v2|world-model-lab|alpha-benchmark-suite|memory-retrieval-v2|critic-board|active-learning-v2|ensemble-uncertainty|tool-intelligence-router|safety-invariants-v2|self-improvement-backlog|adaptive-intelligence-kernel|intelligence-kernel|meta-learning-kernel|bot-founder-dna|founder-dna|bot-genome|lineage-dna|retrain-force-full|retrain-force-targeted|token-refresh|token-refresh-interactive|macro-bulletin|macro-auto-start|macro-replay|macro-media-ingest|macro-auto-stop|macro-auto-status|access-status|apple-profile|apple-silicon-profile)
    if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
      # shellcheck disable=SC1091
      source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
    fi
    ;;
esac

load_runtime_profile() {
  local profile_name="${1:-live}"
  if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$profile_name" --quiet
  fi
  export BOT_RUNTIME_PROFILE="$profile_name"
  export MARKET_DATA_ONLY="${MARKET_DATA_ONLY:-1}"
  export ALLOW_ORDER_EXECUTION="${ALLOW_ORDER_EXECUTION:-0}"
  export TOP_BOT_PAPER_TRADING_ENABLED="${TOP_BOT_PAPER_TRADING_ENABLED:-1}"
  export TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED="${TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED:-1}"
  export PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS="${PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS:-0}"
  export PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"
  export PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"
  export LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"
  export LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"
  export LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"
  export LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"
  export LOG_FUTURES_MASTER_DECISIONS="${LOG_FUTURES_MASTER_DECISIONS:-1}"
}

STORAGE_OVERRIDE_FILE="$PROJECT_ROOT/config/.env.storage_override"
HEALTH_DIR="$PROJECT_ROOT/governance/health"
OPERATOR_STOP_FLAG="$HEALTH_DIR/OPERATOR_STOP.flag"
GLOBAL_HALT_FLAG="$HEALTH_DIR/GLOBAL_TRADING_HALT.flag"
PAPER_TRADE_LOCK_FILE="$PROJECT_ROOT/governance/health/PAPER_TRADE_LOCK.flag"

flag_summary() {
  local path="$1"
  "$PY" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    payload = {}

reason = str(payload.get("reason") or "").strip()
timestamp = str(payload.get("timestamp_utc") or "").strip()
operator = str(payload.get("operator") or "").strip()
parts = []
if reason:
    parts.append(f"reason={reason}")
if operator:
    parts.append(f"operator={operator}")
if timestamp:
    parts.append(f"timestamp_utc={timestamp}")
print(" ".join(parts))
PY
}

enable_paper_trade_lock() {
  mkdir -p "$(dirname "$PAPER_TRADE_LOCK_FILE")"
  printf 'enabled_at_utc=%s\npolicy=live_data_paper_trade_only\nmanaged_by=opsctl\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$PAPER_TRADE_LOCK_FILE"
}

paper_trade_lock_env() {
  enable_paper_trade_lock
  export PAPER_TRADE_LOCK=1
  export PAPER_TRADE_LOCK_PATH="$PAPER_TRADE_LOCK_FILE"
  export MARKET_DATA_ONLY=1
  export ALLOW_ORDER_EXECUTION=0
  export TOP_BOT_ENABLE_LIVE_EXECUTION=0
  export EXECUTION_LANE_LIVE_ENABLED=0
  export RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR=0
  export INLINE_PAPER_EXECUTION_ENABLED=0
}

abort_loop_refresh_if_safety_flags_active() {
  local action_name="${1:-loop_refresh}"
  local blocked=0

  if [[ -f "$OPERATOR_STOP_FLAG" ]]; then
    blocked=1
    echo "${action_name}_blocked=operator_stop" >&2
    echo "operator_stop_flag=$OPERATOR_STOP_FLAG" >&2
    echo "operator_stop_detail=$(flag_summary "$OPERATOR_STOP_FLAG")" >&2
  fi

  if [[ -f "$GLOBAL_HALT_FLAG" ]]; then
    blocked=1
    echo "${action_name}_blocked=global_halt" >&2
    echo "global_halt_flag=$GLOBAL_HALT_FLAG" >&2
    echo "global_halt_detail=$(flag_summary "$GLOBAL_HALT_FLAG")" >&2
  fi

  if [[ "$blocked" == "1" ]]; then
    echo "${action_name}_status=blocked_by_safety_flags" >&2
    echo "show_global_halt_status=./scripts/ops/opsctl.sh global-halt-status --json" >&2
    echo "refresh_global_halt_blockers=./scripts/ops/opsctl.sh global-halt-refresh --json" >&2
    echo "release_operator_stop=./scripts/ops/opsctl.sh operator-release --json" >&2
    echo "attempt_safe_global_halt_clear=./scripts/ops/opsctl.sh global-halt-auto-clear --json" >&2
    echo "manual_clear_all_halts=./scripts/ops/opsctl.sh clear-all-halts --json" >&2
    exit 2
  fi
}

refresh_system_self_model_quietly() {
  if [[ "${OPSCTL_SELF_MODEL_AUTO_REFRESH:-1}" == "0" ]]; then
    return 0
  fi
  "$PY" "$PROJECT_ROOT/scripts/ops/system_self_model.py" --json >/dev/null 2>&1 || true
}

run_then_refresh_self_model() {
  set +e
  "$@"
  local rc=$?
  set -e
  refresh_system_self_model_quietly
  exit "$rc"
}

write_storage_override() {
  local mode="${1:-external}"
  mkdir -p "$(dirname "$STORAGE_OVERRIDE_FILE")"
  case "$mode" in
    local)
      cat > "$STORAGE_OVERRIDE_FILE" <<'EOF'
# Auto-managed by scripts/ops/opsctl.sh
BOT_LOGS_PREFER_EXTERNAL=0
EOF
      ;;
    external)
      rm -f "$STORAGE_OVERRIDE_FILE"
      ;;
    *)
      echo "unknown storage override mode: $mode" >&2
      return 2
      ;;
  esac
}

apply_storage_route_mode() {
  local mode="${1:-external}"
  local prefer_external="1"
  if [[ "$mode" == "local" ]]; then
    prefer_external="0"
  fi
  BOT_LOGS_PREFER_EXTERNAL="$prefer_external" \
    "$PY" "$PROJECT_ROOT/scripts/ops/storage_failback_sync.py" --json
}

run_storage_transition_coordinator() {
  local mode="${1:-external}"
  "$PY" "$PROJECT_ROOT/scripts/ops/storage_transition_coordinator.py" --transition-mode "$mode" --apply --json >/dev/null 2>&1 || true
}

restart_collection_after_storage_switch() {
  local mode="${1:-external}"
  "$PROJECT_ROOT/scripts/ops/opsctl.sh" stop >/dev/null 2>&1 || true
  sleep 1
  "$PROJECT_ROOT/scripts/ops/opsctl.sh" feed-refresh --source all
  OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
  run_storage_transition_coordinator "$mode"
}

stop_launchd_service() {
  local label="$1"
  local plist_path="${2:-$HOME/Library/LaunchAgents/${label}.plist}"
  local uid_num
  uid_num="$(id -u)"
  launchctl bootout "gui/$uid_num" "$plist_path" >/dev/null 2>&1 || launchctl unload "$plist_path" >/dev/null 2>&1 || true
  launchctl disable "gui/$uid_num/$label" >/dev/null 2>&1 || true
}

stop_core_stack_supervisors() {
  stop_launchd_service "com.dankingsley.shadow_watchdog"
  stop_launchd_service "com.dankingsley.all_sleeves"
  stop_launchd_service "com.dankingsley.ops.watchdog"
}

runtime_status_lines() {
  ps -axo pid,etime,command | awk '
    /grep/ { next }
    {
      cmd = $0
      if (index(cmd, "scripts/shadow_watchdog.py") > 0) {
        if (!seen[cmd]++) {
          print
        }
        next
      }

      if (index(cmd, "scripts/run_all_sleeves.py") > 0 || index(cmd, "scripts/run_parallel_shadows.py") > 0 || index(cmd, "scripts/run_parallel_aggressive_modes.py") > 0 || index(cmd, "scripts/run_dividend_shadow.py") > 0 || index(cmd, "scripts/run_bond_shadow.py") > 0 || index(cmd, "scripts/run_fx_shadow.py") > 0 || index(cmd, "scripts/run_specialized_sleeve_shadow.py") > 0 || index(cmd, "scripts/run_execution_lane.py") > 0 || index(cmd, "scripts/run_shadow_training_loop.py --broker coinbase") > 0 || index(cmd, "scripts/run_shadow_training_loop.py --broker schwab --profile schwab_futures") > 0 || index(cmd, "scripts/run_shadow_training_loop.py --broker schwab --profile fx") > 0 || index(cmd, "scripts/ops/sql_link_shard_manager.py") > 0 || index(cmd, "scripts/ops/sql_link_writer_service.py") > 0 || index(cmd, "scripts/link_jsonl_to_sql.py --project-root") > 0) {
        if (!seen[cmd]++) {
          print
        }
      }
    }
  '
}

kill_schwab_live_loops() {
  pkill -f "scripts/run_all_sleeves.py" || true
  pkill -f "scripts/run_parallel_shadows.py" || true
  pkill -f "scripts/run_parallel_aggressive_modes.py" || true
  pkill -f "scripts/run_dividend_shadow.py" || true
  pkill -f "scripts/run_dividend_capture_shadow.py" || true
  pkill -f "scripts/run_bond_shadow.py" || true
  pkill -f "scripts/run_fx_shadow.py" || true
  pkill -f "scripts/run_specialized_sleeve_shadow.py" || true
  pkill -f "scripts/run_execution_lane.py" || true
  pkill -f "scripts/run_shadow_training_loop.py --broker schwab" || true
}

start_schwab_live_loops() {
  local paper_mode="${1:-0}"
  load_runtime_profile live
  "$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true
  "$PY" "$PROJECT_ROOT/scripts/ops/storage_failback_sync.py" --json >/dev/null 2>&1 || true

  local log_file="$PROJECT_ROOT/logs/all_sleeves_$(date -u +%Y%m%d_%H%M%S).log"
  local -a cmd=(
    "$PY" "$PROJECT_ROOT/scripts/run_all_sleeves.py"
    --with-aggressive-modes
  )
  local specialized_interval="${SPECIALIZED_SLEEVE_INTERVAL:-300}"
  local specialized_workers="${SLEEVE_WORKERS_SPECIALIZED:-1}"
  local specialized_nice="${SLEEVE_NICE_SPECIALIZED:-14}"
  local breaker_startup_grace="${ALL_SLEEVES_BREAKER_STARTUP_GRACE_SECONDS:-600}"
  local breaker_data_quality_grace="${ALL_SLEEVES_BREAKER_DATA_QUALITY_GRACE_SECONDS:-1800}"
  local breaker_streak="${ALL_SLEEVES_BREAKER_STREAK:-3}"
  local breaker_cooldown="${ALL_SLEEVES_BREAKER_COOLDOWN:-120}"
  echo "livefeed_expansion=enabled specialized_interval=$specialized_interval specialized_workers=$specialized_workers specialized_nice=$specialized_nice breaker_startup_grace=$breaker_startup_grace breaker_data_quality_grace=$breaker_data_quality_grace"

  if [[ "$paper_mode" == "1" ]]; then
    paper_trade_lock_env
    local paper_top_n="${SCHWAB_TOP_BOT_PAPER_TRADING_TOP_N:-${TOP_BOT_PAPER_TRADING_TOP_N:-5}}"
    local paper_min_acc="${SCHWAB_TOP_BOT_PAPER_TRADING_MIN_ACC:-${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.58}}"
    local paper_profiles="${SCHWAB_TOP_BOT_PAPER_TRADING_PROFILES:-${TOP_BOT_PAPER_TRADING_PROFILES:-}}"
    local options_paper_top_n="${SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_TOP_N:-${TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N:-2}}"
    local options_paper_min_acc="${SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_MIN_ACC:-${TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC:-$paper_min_acc}}"
    local options_paper_profiles="${SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_PROFILES:-${TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES:-default,aggressive,intraday_aggressive,swing_aggressive,options_on_futures,options_on_futures_aggressive}}"
    echo "schwab_paper=enabled top_n=$paper_top_n min_acc=$paper_min_acc profiles=${paper_profiles:-all}"
    echo "schwab_options_paper=enabled top_n=$options_paper_top_n min_acc=$options_paper_min_acc profiles=${options_paper_profiles:-all}"
    TOP_BOT_PAPER_TRADING_ENABLED=1 \
    TOP_BOT_PAPER_TRADING_TOP_N="$paper_top_n" \
    TOP_BOT_PAPER_TRADING_MIN_ACC="$paper_min_acc" \
    TOP_BOT_PAPER_TRADING_PROFILES="$paper_profiles" \
    TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED="${TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED:-1}" \
    TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N="$options_paper_top_n" \
    TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC="$options_paper_min_acc" \
    TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES="$options_paper_profiles" \
    PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}" \
    PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}" \
    PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS="${PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS:-0}" \
    RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES="${RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES:-1}" \
    SPECIALIZED_SLEEVE_INTERVAL="$specialized_interval" \
    SLEEVE_WORKERS_SPECIALIZED="$specialized_workers" \
    SLEEVE_NICE_SPECIALIZED="$specialized_nice" \
    ALL_SLEEVES_BREAKER_STARTUP_GRACE_SECONDS="$breaker_startup_grace" \
    ALL_SLEEVES_BREAKER_DATA_QUALITY_GRACE_SECONDS="$breaker_data_quality_grace" \
    ALL_SLEEVES_BREAKER_STREAK="$breaker_streak" \
    ALL_SLEEVES_BREAKER_COOLDOWN="$breaker_cooldown" \
    TRAINING_EXCLUDED_UNTIL_READY="${TRAINING_EXCLUDED_UNTIL_READY:-1}" \
    SOURCE_CREDENTIAL_GATED="${SOURCE_CREDENTIAL_GATED:-1}" \
    ALLOW_SYNTHETIC_VENDOR_PULLS="${ALLOW_SYNTHETIC_VENDOR_PULLS:-0}" \
    PYTORCH_REPLAY_CANARY_ENABLED="${PYTORCH_REPLAY_CANARY_ENABLED:-0}" \
    PYTHONUNBUFFERED=1 nohup "${cmd[@]}" > "$log_file" 2>&1 & disown
  else
    PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS="${PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS:-0}" \
    RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES="${RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES:-1}" \
    SPECIALIZED_SLEEVE_INTERVAL="$specialized_interval" \
    SLEEVE_WORKERS_SPECIALIZED="$specialized_workers" \
    SLEEVE_NICE_SPECIALIZED="$specialized_nice" \
    ALL_SLEEVES_BREAKER_STARTUP_GRACE_SECONDS="$breaker_startup_grace" \
    ALL_SLEEVES_BREAKER_DATA_QUALITY_GRACE_SECONDS="$breaker_data_quality_grace" \
    ALL_SLEEVES_BREAKER_STREAK="$breaker_streak" \
    ALL_SLEEVES_BREAKER_COOLDOWN="$breaker_cooldown" \
    TRAINING_EXCLUDED_UNTIL_READY="${TRAINING_EXCLUDED_UNTIL_READY:-1}" \
    SOURCE_CREDENTIAL_GATED="${SOURCE_CREDENTIAL_GATED:-1}" \
    ALLOW_SYNTHETIC_VENDOR_PULLS="${ALLOW_SYNTHETIC_VENDOR_PULLS:-0}" \
    PYTORCH_REPLAY_CANARY_ENABLED="${PYTORCH_REPLAY_CANARY_ENABLED:-0}" \
    PYTHONUNBUFFERED=1 nohup "${cmd[@]}" > "$log_file" 2>&1 & disown
  fi
  sleep 2

  if ps -axo command | grep -F "scripts/run_all_sleeves.py --with-aggressive-modes" | grep -v grep >/dev/null 2>&1; then
    echo "$log_file"
    echo "schwab_live_loops_started simulate=0 paper_mode=$paper_mode"
    OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
    return 0
  fi

  echo "schwab_live_loops_failed_to_start log=$log_file" >&2
  tail -n 60 "$log_file" || true
  return 1
}

coinbase_spot_process_lines() {
  ps -axo pid,command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase" | grep -v " --profile crypto_futures" | grep -v grep || true
}

coinbase_spot_running() {
  coinbase_spot_process_lines | grep -q .
}

kill_coinbase_spot_loops() {
  local pids
  pids="$(coinbase_spot_process_lines | awk '{print $1}')"
  if [[ -n "${pids//[[:space:]]/}" ]]; then
    while IFS= read -r pid; do
      [[ -n "$pid" ]] && kill "$pid" >/dev/null 2>&1 || true
    done <<< "$pids"
  fi
}

kill_coinbase_futures_loops() {
  local futures_profile="${COINBASE_FUTURES_PROFILE:-crypto_futures}"
  pkill -f "scripts/run_shadow_training_loop.py --broker coinbase --profile $futures_profile" || true
}

market_correlation_process_lines() {
  ps -axo pid,command | grep -F "scripts/collect_market_crypto_correlation_context.py" | grep -v grep || true
}

market_correlation_running() {
  market_correlation_process_lines | grep -q .
}

fx_process_lines() {
  ps -axo pid,command | grep -E "scripts/run_fx_shadow.py|scripts/run_shadow_training_loop.py --broker .* --profile fx" | grep -v grep || true
}

fx_running() {
  fx_process_lines | grep -q .
}

kill_fx_loops() {
  local pids
  pids="$(fx_process_lines | awk '{print $1}')"
  if [[ -n "${pids//[[:space:]]/}" ]]; then
    while IFS= read -r pid; do
      [[ -n "$pid" ]] && kill "$pid" >/dev/null 2>&1 || true
    done <<< "$pids"
  fi
}

stop_all_runtime_loops() {
  pkill -f "scripts/shadow_watchdog.py" || true
  pkill -f "scripts/ops/process_watchdog.py" || true
  kill_schwab_live_loops
  kill_coinbase_spot_loops
  kill_coinbase_futures_loops
  kill_fx_loops
  pkill -f "scripts/run_shadow_training_loop.py --broker coinbase --profile crypto_futures" || true
  pkill -f "scripts/ops/sql_link_shard_manager.py" || true
  pkill -f "scripts/ops/sql_link_writer_service.py" || true
  pkill -f "scripts/link_jsonl_to_sql.py --project-root $PROJECT_ROOT --mode sqlite --sqlite-db $PROJECT_ROOT/data/sql_link_shards/" || true
  pkill -f "scripts/data_retention_policy.py" || true
  pkill -f "scripts/ops/storage_maintenance_lane.py" || true
  pkill -f "scripts/ops/external_backlog_drain.py" || true
}

case "$cmd" in
  start)
    exec "$PROJECT_ROOT/scripts/ops/start_stack.sh" "$@"
    ;;
  start-sim)
    exec "$PROJECT_ROOT/scripts/ops/start_stack.sh" --profile sim --simulate "$@"
    ;;
  start-live)
    exec "$PROJECT_ROOT/scripts/ops/start_stack.sh" --profile live "$@"
    ;;
  stop)
    DRY_RUN=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --dry-run) DRY_RUN=1 ;;
      esac
      shift
    done
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "stack_stop_dry_run=1"
      echo "stack_stop_status=ready_to_stop"
      echo "stop_targets=core_stack_supervisors,runtime_loops"
      exit 0
    fi
    stop_core_stack_supervisors
    stop_all_runtime_loops
    echo "stopped stack services"
    ;;
  status)
    runtime_status_lines || true
    PROFILE="${BOT_RUNTIME_PROFILE:-live}"
    if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
      # shellcheck disable=SC1091
      source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$PROFILE" --quiet
    fi
    PREFLIGHT_ARGS=(--broker "${DATA_BROKER:-schwab}" --allow-running --json)
    if [[ "$PROFILE" == "sim" ]]; then
      PREFLIGHT_ARGS+=(--simulate)
    fi
    "$PY" "$PROJECT_ROOT/scripts/ops/preflight_autofix.py" "${PREFLIGHT_ARGS[@]}" || true
    ;;
  restart-sanity)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/restart_sanity_bundle.py" "$@"
    ;;
  post-restart-settle|post-restart-settlement|restart-settlement)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/post_restart_settlement.py" "$@"
    ;;
  retrain)
    # MLX Metal JIT can intermittently crash in some launch contexts; keep a stable default.
    RETRAIN_TRIGGER_SOURCE="opsctl_retrain" \
    RETRAIN_TRIGGER_LABEL="opsctl:retrain" \
    RETRAIN_TRIGGER_CONTEXT="manual_opsctl" \
    MLX_METAL_JIT="${MLX_METAL_JIT:-0}" \
    exec "$TRAINING_PY" "$PROJECT_ROOT/scripts/weekly_retrain.py" --continue-on-error "$@"
    ;;
  retrain-orchestrate)
    exec "$TRAINING_PY" "$PROJECT_ROOT/scripts/retrain_orchestrator.py" "$@"
    ;;
  scorecard)
    exec "$PY" "$PROJECT_ROOT/scripts/unified_lane_scorecard.py" "$@"
    ;;
  sql-sync)
    if [[ -n "${SQL_LINK_SERVICE_SHARDS:-}" ]]; then
      exec "$PY" "$PROJECT_ROOT/scripts/ops/sql_link_shard_manager.py" --once "$@"
    fi
    exec "$PY" "$PROJECT_ROOT/scripts/ops/sql_link_writer_service.py" --once "$@"
    ;;
  tradingeconomics-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_tradingeconomics_guest_data.py" "$@"
    ;;
  macro-context-sync)
    bls_args=()
    for arg in "$@"; do
      case "$arg" in
        --json) ;;
        *) bls_args+=("$arg") ;;
      esac
    done
    "$PY" "$PROJECT_ROOT/scripts/collect_bls_census_data.py" "${bls_args[@]}" || true
    exec "$PY" "$PROJECT_ROOT/scripts/collect_official_macro_context.py" "$@"
    ;;
  schwab-education-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_schwab_education_context.py" "$@"
    ;;
  market-micro-sync)
    mm_args=("$@")
    mm_has_timeout=0
    mm_has_max_runtime=0
    mm_has_lookback=0
    mm_has_finra=0
    mm_has_symbols=0
    for arg in "$@"; do
      case "$arg" in
        --timeout-seconds) mm_has_timeout=1 ;;
        --max-runtime-seconds) mm_has_max_runtime=1 ;;
        --lookback-days) mm_has_lookback=1 ;;
        --finra-lookback-days) mm_has_finra=1 ;;
        --symbols) mm_has_symbols=1 ;;
      esac
    done
    if [[ "$mm_has_timeout" == "0" ]]; then
      mm_args+=(--timeout-seconds "${MARKET_MICRO_TIMEOUT_SECONDS:-4}")
    fi
    if [[ "$mm_has_max_runtime" == "0" ]]; then
      mm_args+=(--max-runtime-seconds "${MARKET_MICRO_MAX_RUNTIME_SECONDS:-75}")
    fi
    if [[ "$mm_has_lookback" == "0" ]]; then
      mm_args+=(--lookback-days "${MARKET_MICRO_LOOKBACK_DAYS:-21}")
    fi
    if [[ "$mm_has_finra" == "0" ]]; then
      mm_args+=(--finra-lookback-days "${MARKET_MICRO_FINRA_LOOKBACK_DAYS:-5}")
    fi
    if [[ "$mm_has_symbols" == "0" ]]; then
      mm_args+=(--symbols "${MARKET_MICRO_SYMBOLS:-}")
    fi
    exec "$PY" "$PROJECT_ROOT/scripts/ops/bounded_market_micro_sync.py" \
      --outer-timeout-seconds "${MARKET_MICRO_OUTER_TIMEOUT_SECONDS:-90}" \
      "${mm_args[@]}"
    ;;
  sec-edgar-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_sec_edgar_context.py" "$@"
    ;;
  extended-quant-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_extended_quant_context.py" "$@"
    ;;
  quant-model-control|quant-engine-control|quant-models)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/quant_model_control.py" "$@"
    ;;
  pricing-grad)
    exec "$PY" "$PROJECT_ROOT/scripts/quant_models/pricing_grad.py" "$@"
    ;;
  gpu-mc-sim)
    exec "$PY" "$PROJECT_ROOT/scripts/quant_models/gpu_mc_sim.py" "$@"
    ;;
  kalman-parallel)
    exec "$PY" "$PROJECT_ROOT/scripts/quant_models/kalman_parallel.py" "$@"
    ;;
  options-flow-sync|tastytrade-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_options_flow_context.py" "$@"
    ;;
  options-flow-export-hygiene|unusual-whales-export-hygiene)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/options_flow_export_hygiene_bot.py" "$@"
    ;;
  options-flow-efficiency|options-flow-efficiency-bot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/options_flow_efficiency_bot.py" "$@"
    ;;
  bot-stack-report|active-bot-report)
    exec "$PY" "$PROJECT_ROOT/scripts/bot_stack_status_report.py" "$@"
    ;;
  core-bot-catalog|bot-catalog)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/build_core_bot_catalog.py" "$@"
    ;;
  core-bot-materialize|materialize-core-bots)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/materialize_core_bot_modules.py" "$@"
    ;;
  core-bot-materialization-guard|core-bot-file-guard)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/core_bot_materialization_guard.py" "$@"
    ;;
  core-bot-materialization-infrabot|core-bot-file-infrabot)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/core_bot_materialization_infrabot.py" "$@"
    ;;
  core-bot-tier-organizer|organize-core-bot-tiers|bot-tier-organizer)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/organize_core_bot_tiers.py" "$@"
    ;;
  crypto-market-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_crypto_market_context.py" "$@"
    ;;
  market-correlation-sync)
    corr_args=("$@")
    corr_has_timeout=0
    corr_has_lookback=0
    corr_has_bucket=0
    corr_has_min_points=0
    for arg in "$@"; do
      case "$arg" in
        --timeout-seconds) corr_has_timeout=1 ;;
        --lookback-days) corr_has_lookback=1 ;;
        --bucket-seconds) corr_has_bucket=1 ;;
        --min-points) corr_has_min_points=1 ;;
      esac
    done
    if [[ "$corr_has_lookback" == "0" ]]; then
      corr_args+=(--lookback-days "${MARKET_CRYPTO_CORRELATION_LOOKBACK_DAYS:-1}")
    fi
    if [[ "$corr_has_bucket" == "0" ]]; then
      corr_args+=(--bucket-seconds "${MARKET_CRYPTO_CORRELATION_BUCKET_SECONDS:-300}")
    fi
    if [[ "$corr_has_min_points" == "0" ]]; then
      corr_args+=(--min-points "${MARKET_CRYPTO_CORRELATION_MIN_POINTS:-3}")
    fi
    if [[ "$corr_has_timeout" == "0" ]]; then
      corr_args+=(--timeout-seconds "${MARKET_CRYPTO_CORRELATION_TIMEOUT_SECONDS:-90}")
    fi
    exec "$PY" "$PROJECT_ROOT/scripts/ops/bounded_market_crypto_correlation_sync.py" \
      --outer-timeout-seconds "${MARKET_CRYPTO_CORRELATION_OUTER_TIMEOUT_SECONDS:-100}" \
      "${corr_args[@]}"
    ;;
  fx-market-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_fx_market_context.py" "$@"
    ;;
  dividend-drip-sync)
    exec "$PY" "$PROJECT_ROOT/scripts/collect_dividend_drip_state.py" "$@"
    ;;
  showcase-refresh)
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
      echo "Usage: opsctl.sh showcase-refresh"
      echo "Refresh docs/showcase generated highlights and README snippets."
      exit 0
    fi
    exec "$PY" "$PROJECT_ROOT/scripts/ops/update_showcase_highlights.py" "$@"
    ;;
  macro-crosscheck)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/macro_crosscheck_report.py" "$@"
    ;;
  source-verification)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/source_verification_report.py" "$@"
    ;;
  collector-contracts|collector-contract)
    exec "$PY" "$PROJECT_ROOT/scripts/collector_contracts.py" "$@"
    ;;
  sleeve-strategy-coverage|sleeve-coverage|strategy-coverage)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/sleeve_strategy_coverage_guard.py" "$@"
    ;;
  global-halt|global-halt-status|halt-status)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/global_risk_killswitch.py" --status-only --exit-zero "$@"
    ;;
  global-halt-refresh|global-halt-clear-blockers|halt-clear-blockers)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/global_risk_killswitch.py" --clear-blockers --status-only --exit-zero "$@"
    ;;
  global-halt-auto-clear|halt-auto-clear)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/global_risk_killswitch.py" --auto-clear "$@"
    ;;
  clear-all-halts|clear-global-halts|global-halt-force-clear|halt-force-clear)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/operator_control.py" --release --clear-global-halt "$@"
    ;;
  operator-control|operator-stop-status)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/operator_control.py" "$@"
    ;;
  operator-release)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/operator_control.py" --release "$@"
    ;;
  mlx-audit|mlx-runtime-audit)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/mlx_runtime_audit.py" "$@"
    ;;
  mlx-library-upgrade|mlx-upgrade)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/mlx_library_upgrade.py" "$@"
    ;;
  mlx-audio-audit|mlx-audio-runtime-audit)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/mlx_audio_runtime_audit.py" "$@"
    ;;
  mlx-intelligence-router|mlx-compute-brain|mlx-utilization)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/mlx_intelligence_router.py" "$@"
    ;;
  library-utilization-router|library-router|non-mlx-library-router|dependency-utilization)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/library_utilization_router.py" "$@"
    ;;
  onnx-audit|onnx-runtime-audit)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/onnx_runtime_audit.py" "$@"
    ;;
  pytorch-audit|torch-audit|pytorch-runtime-audit)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/pytorch_runtime_audit.py" "$@"
    ;;
  pytorch-replay-canary|torch-replay-canary)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/pytorch_replay_canary.py" "$@"
    ;;
  sql-audit|sql-runtime-audit|sql-access-audit)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/sql_access_runtime_audit.py" "$@"
    ;;
  training-registry-audit|registry-training-audit)
    exec "$PY" "$PROJECT_ROOT/scripts/training_registry_audit.py" "$@"
    ;;
  training-label-audit|label-audit)
    exec "$PY" "$PROJECT_ROOT/scripts/training_label_audit.py" "$@"
    ;;
  training-quality|training-quality-control)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/training_quality_control.py" "$@"
    ;;
  feature-store|feature-store-manifest)
    exec "$PY" "$PROJECT_ROOT/scripts/feature_store_manifest.py" "$@"
    ;;
  multiple-testing|multiple-testing-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/multiple_testing_guard.py" "$@"
    ;;
  decay-monitor)
    exec "$PY" "$PROJECT_ROOT/scripts/decay_monitor.py" "$@"
    ;;
  security-audit|security-hardening-audit)
    exec "$PY" "$PROJECT_ROOT/scripts/security_hardening_audit.py" "$@"
    ;;
  security-evidence-autofix|security-evidence|security-refresh)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/security_evidence_autofix.py" "$@"
    ;;
  secret-scan)
    exec "$PY" "$PROJECT_ROOT/scripts/secret_scan.py" "$@"
    ;;
  schema-migration|migration-manifest|schema-migration-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/schema_migration_guard.py" "$@"
    ;;
  ingestion-storage|ingestion-storage-control)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/ingestion_storage_control.py" "$@"
    ;;
  ingestion-storage-governor|storage-pressure-governor|governor)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/ingestion_storage_governor.py" "$@"
    ;;
  external-backlog-drain|backlog-drain|external-drain)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/external_backlog_drain.py" "$@"
    ;;
  backpressure-drainer-fleet|backpressure-drainers|drainer-fleet)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/backpressure_drainer_fleet.py" "$@"
    ;;
  external-backlog-retry-bot|backlog-retry-bot|external-backlog-bot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/external_backlog_retry_bot.py" "$@"
    ;;
  storage-backpressure-autopilot|storage-backpressure-bot|storage-backpressure-control)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_backpressure_autopilot.py" "$@"
    ;;
  storage-pressure-clearance|storage-pressure-clear|storage-pressure-supervisor|storage-clearance)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_pressure_clearance_bot.py" "$@"
    ;;
  storage-reconnect-regression-guard|storage-reconnect-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_reconnect_regression_guard.py" "$@"
    ;;
  storage-reconnect-infrabot|storage-recovery-infrabot|storage-auto-recovery-bot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_reconnect_infrabot.py" "$@"
    ;;
  stateful-storage-regression-guard|stateful-storage-guard|local-stateful-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/stateful_storage_regression_guard.py" "$@"
    ;;
  data-retention|retention-policy|data-retention-policy)
    exec "$PY" "$PROJECT_ROOT/scripts/data_retention_policy.py" "$@"
    ;;
  data-collection-storage-guard|collector-storage-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/data_collection_storage_guard.py" "$@"
    ;;
  data-collection-observation-rollup|collector-observation-rollup)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/data_collection_observation_rollup.py" "$@"
    ;;
  writer-cycle-coordinator|writer-handoff|writer-cycle-bot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/writer_cycle_coordinator.py" "$@"
    ;;
  retention-debt-sheriff|retention-sheriff|explanation-retention-bot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/retention_debt_sheriff.py" "$@"
    ;;
  backpressure-slo-bot|backpressure-slo|slo-bot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/backpressure_slo_bot.py" "$@"
    ;;
  backlog-quarantine|backlog-quarantine-bot|cold-backlog-quarantine)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/backlog_quarantine_bot.py" "$@"
    ;;
  ingestion-priority-queue|queue-backed-ingestion|queue-control)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/ingestion_priority_queue.py" "$@"
    ;;
  content-store|content-address-store|cas-store)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/content_addressed_artifact_store.py" "$@"
    ;;
  split-brain-reconcile|storage-split-brain|split-brain)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_split_brain_reconciler.py" "$@"
    ;;
  storage-resilience|storage-resilience-control)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_resilience_control.py" "$@"
    ;;
  storage-tier-policy|storage-tier)
    exec "$PY" "$PROJECT_ROOT/scripts/storage_tier_policy.py" "$@"
    ;;
  runtime-training-snapshot|training-snapshot-runtime)
    exec "$PY" "$PROJECT_ROOT/scripts/build_runtime_training_snapshot.py" "$@"
    ;;
  training-runtime-control|runtime-training-control)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/training_runtime_control.py" "$@"
    ;;
  training-requalification|requalification-lane)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/training_requalification_lane.py" "$@"
    ;;
  coverage-seed|walk-forward-coverage-seed)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/walk_forward_coverage_seed.py" "$@"
    ;;
  coverage-gap-closer|gap-closer|promotion-gap-closer)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/coverage_gap_closer.py" "$@"
    ;;
  regime-control|regime-control-plane)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/regime_control_plane.py" "$@"
    ;;
  supportability-control|supportability|lifecycle-control)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/supportability_control.py" "$@"
    ;;
  teacher-quality|teacher-quality-guard|teacher-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/teacher_quality_guard.py" "$@"
    ;;
  bot-quality-autopilot|quality-autopilot|quality-coach)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/bot_quality_autopilot.py" "$@"
    ;;
  commands-hygiene|runbook-hygiene|commands-cleanup)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/commands_hygiene_bot.py" "$@"
    ;;
  command-validity|commands-verify|command-audit)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/command_validity_bot.py" "$@"
    ;;
  codex-project-guard|codex-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/codex_project_guard.py" "$@"
    ;;
  infrastructure-autofix|infra-autofix|infra-remediation)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/infrastructure_autofix_bot.py" "$@"
    ;;
  system-cleanliness-autopilot|cleanliness-autopilot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/system_cleanliness_autopilot.py" "$@"
    ;;
  system-cleanliness-infrabot|cleanliness-infrabot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/system_cleanliness_infrabot.py" "$@"
    ;;
  master-infra-supervisor|master-infrastructure-supervisor|infra-supervisor)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/master_infrastructure_supervisor.py" "$@"
    ;;
  coinbase-api-health|coinbase-health)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/coinbase_api_health.py" "$@"
    ;;
  system-drift-guard|drift-guard|drift-mesh)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/system_drift_guard.py" "$@"
    ;;
  system-drift-autopilot|drift-autopilot|drift-mesh-autopilot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/system_drift_autopilot.py" "$@"
    ;;
  live-runtime-separation|runtime-separation|live-runtime-control)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/live_runtime_separation_control.py" "$@"
    ;;
  rolling-restart|rolling-restart-controller|restart-controller)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/rolling_restart_controller.py" "$@"
    ;;
  schwab-auth-supervisor|schwab-auth-guard|auth-supervisor)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/schwab_auth_supervisor.py" "$@"
    ;;
  auth-lease|auth-lease-manager|lease-manager)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/auth_lease_manager.py" "$@"
    ;;
  incident-timeline|incident-log|incident-control)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/incident_timeline.py" "$@"
    ;;
  incident-closeout|incident-closeout-autopilot|incident-closeout-control)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/incident_closeout_autopilot.py" "$@"
    ;;
  incident-review-packet|incident-packet)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/incident_review_packet.py" "$@"
    ;;
  incident-report|incident-review-report|incident-brief)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/incident_report.py" "$@"
    ;;
  promotion-autopilot|promotion-packet-autopilot|promotion-autopilot-packet)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/promotion_autopilot_packet.py" "$@"
    ;;
  autonomy-control|autonomy-control-plane|autonomy-plane)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/autonomy_control_plane.py" "$@"
    ;;
  live-canary-control|canary-control|supervised-canary)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/live_canary_control.py" "$@"
    ;;
  runtime-throttle|throttle-control|throttle-bot)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/runtime_throttle_control.py" "$@"
    ;;
  mode-switchboard|mode-switchboard-mission-control)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/mode_switchboard_mission_control.py" "$@"
    ;;
  decision-provenance|decision-provenance-cards)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/decision_provenance_cards.py" "$@"
    ;;
  blackstart-recovery|blackstart|reboot-blackstart)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/blackstart_recovery.py" "$@"
    ;;
  sleeve-isolation|sleeve-isolation-guard|quarantine-isolation)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/sleeve_isolation_guard.py" "$@"
    ;;
  artifact-freshness-slo|freshness-slo|artifact-slo)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/artifact_freshness_slo.py" "$@"
    ;;
  runtime-snapshot-cache|snapshot-cache-control|runtime-cache)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/runtime_snapshot_cache_control.py" "$@"
    ;;
  remote-alert-control|remote-alerts|alert-control)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/remote_alert_control.py" "$@"
    ;;
  storage-quota-guard|storage-quotas|quota-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_quota_guard.py" "$@"
    ;;
  release-freeze|freeze-window|runtime-freeze)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/release_freeze_guard.py" "$@"
    ;;
  legacy-bot-harmonizer|legacy-bot-harmonize|legacy-v107-harmonizer)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/legacy_bot_harmonizer.py" "$@"
    ;;
  roster-expansion|roster-slots|bot-slots)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/roster_expansion_slots.py" "$@"
    ;;
  roster-resilience|roster-planner|bench-depth)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/roster_resilience_planner.py" "$@"
    ;;
  chaos-drills|chaos-drill-coordinator|chaos-drill)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/chaos_drill_coordinator.py" "$@"
    ;;
  calibration-control|abstention-control|calibration-abstention)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/calibration_abstention_control.py" "$@"
    ;;
  portfolio-allocator|portfolio-allocator-service)
    exec "$PY" "$PROJECT_ROOT/scripts/portfolio_allocator_service.py" "$@"
    ;;
  portfolio-capacity-curves|capacity-curves)
    exec "$PY" "$PROJECT_ROOT/scripts/portfolio_capacity_curve_report.py" "$@"
    ;;
  risk-service|risk-service-boundary)
    exec "$PY" "$PROJECT_ROOT/scripts/risk_service_boundary.py" "$@"
    ;;
  execution-lab)
    exec "$PY" "$PROJECT_ROOT/scripts/execution_lab.py" "$@"
    ;;
  operator-cockpit|cockpit)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/operator_cockpit.py" "$@"
    ;;
  daily-verify-remediation|daily-verify-auto-remediation|daily-verify-remediation-bot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/daily_verify_auto_remediation_bot.py" "$@"
    ;;
  memory-efficiency|memory-efficiency-control)
    subcmd="${1:-status}"
    shift || true
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/memory_efficiency_control.py" "$subcmd" "$@"
    ;;
  swap-pressure-governor|swap-pressure|swap-governor)
    subcmd="${1:-status}"
    shift || true
    exec "$PY" "$PROJECT_ROOT/scripts/ops/swap_pressure_governor.py" "$subcmd" "$@"
    ;;
  creative-cotenant-guard|creative-cotenant|cotenant-guard)
    subcmd="${1:-status}"
    shift || true
    exec "$PY" "$PROJECT_ROOT/scripts/ops/creative_cotenant_guard.py" "$subcmd" "$@"
    ;;
  platform-control-plane|platform-control|control-plane|institutional-readiness)
    exec "$PY" "$PROJECT_ROOT/scripts/platform_control_plane_report.py" "$@"
    ;;
  new-bot-admission|new-bot-admission-guard|admission-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/new_bot_admission_guard.py" "$@"
    ;;
  intelligence-capability-expansion|capability-expansion|platform-intelligence|platform-intelligence-expansion|bot-admission|sleeve-masters|quality-score-system|market-regime-router|capacity-planner|research-pipeline|correlation-governor|model-decay-detector|system-dashboard)
    if [[ "$cmd" == "intelligence-capability-expansion" || "$cmd" == "capability-expansion" ]]; then
      run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/intelligence_capability_expansion.py" "$@"
    fi
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/platform_intelligence_expansion.py" "$@"
    ;;
  expansion-capacity|expansion-capacity-planner|growth-capacity)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/expansion_capacity_planner_bot.py" "$@"
    ;;
  advanced-intelligence-mesh|intelligence-mesh|meta-intelligence)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/advanced_intelligence_mesh_expansion.py" "$@"
    ;;
  cognitive-control-plane|cognitive-control|cognition-plane)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/cognitive_control_plane_expansion.py" "$@"
    ;;
  recursive-research-foundry|research-foundry|recursive-foundry)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/recursive_research_foundry_expansion.py" "$@"
    ;;
  coordination-intelligence|coordination-control|strategy-conflict-resolver|capital-simulator|market-regime-memory|research-to-bot-pipeline|feature-quality-layer|adversarial-paper-lab|sleeve-master-upgrade|bot-admission-committee|system-explainability)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/coordination_intelligence_expansion.py" "$@"
    ;;
  adaptive-intelligence-kernel|intelligence-kernel|meta-learning-kernel)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/adaptive_intelligence_kernel_expansion.py" "$@"
    ;;
  big-platform-brain|platform-brain|system-self-model|self-model|self-awareness|metacognition)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/system_self_model.py" "$@"
    ;;
  self-awareness-infrabots|system-self-awareness)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/system_self_awareness_expansion.py" "$@"
    ;;
  alpha-intelligence-evolution|alpha-advancement|alpha-evolution|alpha-intelligence|training-readiness-brain|execution-reality-lab|portfolio-exposure-brain|data-source-confidence-engine|research-intake-pipeline|duplicate-alpha-novelty|professional-dashboard-v2|broker-adapter-mesh|adapter-mesh|cleanup-governor)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/alpha_intelligence_evolution_expansion.py" "$@"
    ;;
  intelligence-layer-advancement|intelligence-layer-v2|intelligence-upgrade|meta-intelligence-v2|metacognitive-routing-v2|world-model-lab|alpha-benchmark-suite|memory-retrieval-v2|critic-board|active-learning-v2|ensemble-uncertainty|tool-intelligence-router|safety-invariants-v2|self-improvement-backlog)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/intelligence_layer_advancement_expansion.py" "$@"
    ;;
  bot-founder-dna|founder-dna|bot-genome|lineage-dna)
    run_then_refresh_self_model "$PY" "$PROJECT_ROOT/scripts/ops/bot_founder_dna_lineage.py" "$@"
    ;;
  stale-sweeper|stale-artifact-sweeper|stale-stage-bot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/stale_artifact_sweeper_bot.py" "$@"
    ;;
  stale-reaper|stale-artifact-reaper|stale-delete-bot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/stale_artifact_reaper_bot.py" "$@"
    ;;
  model-lifecycle|lifecycle-audit)
    exec "$PY" "$PROJECT_ROOT/scripts/model_lifecycle_hygiene.py" "$@"
    ;;
  grade-lift-hardening|a-grade-hardening|grade-up)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/grade_lift_hardening.py" "$@"
    ;;
  grade-regression-guard|regression-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/grade_regression_guard.py" "$@"
    ;;
  grade-regression-autopilot|grade-regression-bot|regression-upgrade)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/grade_regression_autopilot.py" "$@"
    ;;
  section-grade-guard|section-floor-guard|grade-floor-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/section_grade_guard.py" "$@"
    ;;
  section-grade-autopilot|section-floor-autopilot|grade-floor-autopilot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/section_grade_autopilot.py" "$@"
    ;;
  access-portable)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/runtime_access_mode.py" set portable "$@"
    ;;
  access-native)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/runtime_access_mode.py" set native "$@"
    ;;
  access-status)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/runtime_access_mode.py" status "$@"
    ;;
  runtime-backend-switch|runtime-access-switch)
    backend="${1:-portable_auto}"
    shift || true
    exec "$PY" "$PROJECT_ROOT/scripts/ops/runtime_access_mode.py" set portable --ml-backend "$backend" "$@"
    ;;
  runtime-backend-status)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/runtime_access_mode.py" status "$@"
    ;;
  runtime-backend-native)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/runtime_access_mode.py" set native "$@"
    ;;
  apple-profile|apple-silicon-profile)
    subcmd="${1:-status}"
    shift || true
    exec "$PY" "$PROJECT_ROOT/scripts/ops/apple_silicon_profile.py" "$subcmd" "$@"
    ;;
  paper-trade-lock-infrabot|paper-lock-infrabot|paper-trade-lock-guard|paper-lock)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/paper_trade_lock_infrabot.py" "$@"
    ;;
  sql-maint|sql-maintenance|sqlite-maint|sqlite-maintenance)
    sql_maint_args=("$@")
    wants_explicit_vacuum=0
    for arg in "${sql_maint_args[@]}"; do
      case "$arg" in
        --vacuum|--no-auto-vacuum|--auto-vacuum-over-gb|--checkpoint-only)
          wants_explicit_vacuum=1
          ;;
      esac
    done
    if [[ "$wants_explicit_vacuum" == "0" ]]; then
      sql_maint_args+=(--no-auto-vacuum)
    fi
    exec "$PY" "$PROJECT_ROOT/scripts/sqlite_performance_maintenance.py" "${sql_maint_args[@]}"
    ;;
  health)
    exec "$PY" "$PROJECT_ROOT/scripts/daily_auto_verify.py" --json "$@"
    ;;
  dashboard-refresh|runtime-artifact-refresh|runtime-contract-refresh)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/runtime_artifact_refresh.py" --json "$@"
    ;;
  dashboard)
    dashboard_refresh=1
    dashboard_args=()
    for arg in "$@"; do
      case "$arg" in
        --skip-refresh|--no-refresh)
          dashboard_refresh=0
          ;;
        *)
          dashboard_args+=("$arg")
          ;;
      esac
    done
    if [[ "$dashboard_refresh" == "1" ]]; then
      "$PY" "$PROJECT_ROOT/scripts/ops/runtime_artifact_refresh.py" --json >/dev/null 2>&1 || true
    fi
    exec "$PY" "$PROJECT_ROOT/scripts/ops/runtime_gate_dashboard.py" --json "${dashboard_args[@]}"
    ;;
  phone-feed)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/live_feed_phone_server.py" "$@"
    ;;
  py314-canary|py314-ready|python314-canary|python314-ready)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/python314_canary.py" --json "$@"
    ;;
  doctor)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/doctor.py" "$@"
    ;;
  schwab-futures-start)
    FORCE_RESTART=0
    SCHWAB_SIMULATE="${SCHWAB_FUTURES_SIMULATE:-0}"
    PAPER_MODE=1
    FUTURES_PROFILE="${SCHWAB_FUTURES_PROFILE:-schwab_futures}"
    PAPER_TOP_N="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}"
    PAPER_MIN_ACC="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.53}"
    PAPER_PROFILES="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-$FUTURES_PROFILE}"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --force-restart) FORCE_RESTART=1 ;;
        --paper) PAPER_MODE=1; SCHWAB_SIMULATE=0 ;;
        --simulate) SCHWAB_SIMULATE=1 ;;
        --live-data|--no-simulate) SCHWAB_SIMULATE=0 ;;
        --top-n) PAPER_TOP_N="${2:-$PAPER_TOP_N}"; shift ;;
        --min-acc) PAPER_MIN_ACC="${2:-$PAPER_MIN_ACC}"; shift ;;
        --profiles) PAPER_PROFILES="${2:-$PAPER_PROFILES}"; shift ;;
        *) echo "unknown schwab-futures-start arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    if [[ "$FORCE_RESTART" == "1" ]]; then
      pkill -f "scripts/run_shadow_training_loop.py --broker schwab --profile $FUTURES_PROFILE" || true
      sleep 1
    fi

    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker schwab --profile $FUTURES_PROFILE" | grep -v grep >/dev/null 2>&1; then
      PID="$(ps -axo pid,command | grep -F "scripts/run_shadow_training_loop.py --broker schwab --profile $FUTURES_PROFILE" | grep -v grep | awk 'NR==1{print $1}')"
      LATEST_LOG="$(ls -1t "$PROJECT_ROOT"/logs/schwab_futures_live_*.log 2>/dev/null | head -n 1)"
      echo "schwab_futures_loop already running pid=$PID profile=$FUTURES_PROFILE"
      [[ -n "$LATEST_LOG" ]] && echo "$LATEST_LOG"
      exit 0
    fi

    "$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true

    LOG="$PROJECT_ROOT/logs/schwab_futures_live_$(date -u +%Y%m%d_%H%M%S).log"
    SCHWAB_CMD=(
      "$PY" "$PROJECT_ROOT/scripts/run_shadow_training_loop.py"
      --broker schwab
      --profile "$FUTURES_PROFILE"
      --domain equities
      --symbols "${SCHWAB_FUTURES_WATCH_SYMBOLS:-/ES,/NQ,/YM,/RTY,/CL,/GC,/ZN}"
      --context-symbols "${SCHWAB_FUTURES_CONTEXT_SYMBOLS:-SPY,UUP,GLD}"
      --interval-seconds "${SCHWAB_FUTURES_WATCH_INTERVAL_SECONDS:-12}"
      --max-iterations 0
    )
    if [[ "$SCHWAB_SIMULATE" == "1" ]]; then
      SCHWAB_CMD+=(--simulate)
    fi

    if [[ "$PAPER_MODE" == "1" ]]; then
      paper_trade_lock_env
      echo "schwab_futures_paper=enabled profile=$FUTURES_PROFILE top_n=$PAPER_TOP_N min_acc=$PAPER_MIN_ACC profiles=$PAPER_PROFILES"
      SHADOW_PROFILE="$FUTURES_PROFILE"       SHADOW_DOMAIN=equities       LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"       LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"       LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"       LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"       TOP_BOT_PAPER_TRADING_ENABLED=1       TOP_BOT_PAPER_TRADING_TOP_N="$PAPER_TOP_N"       TOP_BOT_PAPER_TRADING_MIN_ACC="$PAPER_MIN_ACC"       TOP_BOT_PAPER_TRADING_PROFILES="$PAPER_PROFILES"       PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"       PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"       nohup "${SCHWAB_CMD[@]}" > "$LOG" 2>&1 & disown
    else
      SHADOW_PROFILE="$FUTURES_PROFILE"       SHADOW_DOMAIN=equities       LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"       LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"       LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"       LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"       nohup "${SCHWAB_CMD[@]}" > "$LOG" 2>&1 & disown
    fi

    sleep 2
    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker schwab --profile $FUTURES_PROFILE" | grep -v grep >/dev/null 2>&1; then
      echo "$LOG"
      echo "schwab_futures_loop_started profile=$FUTURES_PROFILE simulate=$SCHWAB_SIMULATE paper_mode=$PAPER_MODE"
      OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
    else
      echo "schwab_futures_loop failed_to_start profile=$FUTURES_PROFILE"
      tail -n 60 "$LOG" || true
      exit 1
    fi
    ;;
  schwab-futures-stop)
    FUTURES_PROFILE="${SCHWAB_FUTURES_PROFILE:-schwab_futures}"
    pkill -f "scripts/run_shadow_training_loop.py --broker schwab --profile $FUTURES_PROFILE" || true
    echo "schwab futures loop stopped profile=$FUTURES_PROFILE"
    ;;
  coinbase-start)
    FORCE_RESTART=0
    COINBASE_SIMULATE="${COINBASE_START_SIMULATE:-0}"
    PAPER_MODE=1
    PAPER_TOP_N="${COINBASE_TOP_BOT_PAPER_TRADING_TOP_N:-${TOP_BOT_PAPER_TRADING_TOP_N:-5}}"
    PAPER_MIN_ACC="${COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC:-${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.58}}"
    PAPER_PROFILES="${COINBASE_TOP_BOT_PAPER_TRADING_PROFILES:-${TOP_BOT_PAPER_TRADING_PROFILES:-default}}"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --force-restart) FORCE_RESTART=1 ;;
        --paper) PAPER_MODE=1; COINBASE_SIMULATE=0 ;;
        --simulate) COINBASE_SIMULATE=1 ;;
        --live-data|--no-simulate) COINBASE_SIMULATE=0 ;;
        --top-n) PAPER_TOP_N="${2:-$PAPER_TOP_N}"; shift ;;
        --min-acc) PAPER_MIN_ACC="${2:-$PAPER_MIN_ACC}"; shift ;;
        --profiles) PAPER_PROFILES="${2:-$PAPER_PROFILES}"; shift ;;
        *) echo "unknown coinbase-start arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    if [[ "$FORCE_RESTART" == "1" ]]; then
      kill_coinbase_spot_loops
      sleep 1
    fi

    if coinbase_spot_running; then
      PID="$(coinbase_spot_process_lines | awk 'NR==1{print $1}')"
      LATEST_LOG="$(ls -1t "$PROJECT_ROOT"/logs/coinbase_live_*.log 2>/dev/null | head -n 1)"
      echo "coinbase_loop already running pid=$PID"
      [[ -n "$LATEST_LOG" ]] && echo "$LATEST_LOG"
      exit 0
    fi

    "$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true

    LOG="$PROJECT_ROOT/logs/coinbase_live_$(date -u +%Y%m%d_%H%M%S).log"
    COINBASE_CMD=(
      "$PY" "$PROJECT_ROOT/scripts/run_shadow_training_loop.py"
      --broker coinbase
      --symbols "${COINBASE_WATCH_SYMBOLS:-BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD}"
      --context-symbols "${COINBASE_CONTEXT_SYMBOLS:-BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD}"
      --interval-seconds "${COINBASE_WATCH_INTERVAL_SECONDS:-20}"
      --max-iterations 0
    )
    if [[ "$COINBASE_SIMULATE" == "1" ]]; then
      COINBASE_CMD+=(--simulate)
    fi

    if [[ "$PAPER_MODE" == "1" ]]; then
      paper_trade_lock_env
      echo "coinbase_paper=enabled top_n=$PAPER_TOP_N min_acc=$PAPER_MIN_ACC profiles=$PAPER_PROFILES"
      TOP_BOT_PAPER_TRADING_ENABLED=1       TOP_BOT_PAPER_TRADING_TOP_N="$PAPER_TOP_N"       TOP_BOT_PAPER_TRADING_MIN_ACC="$PAPER_MIN_ACC"       TOP_BOT_PAPER_TRADING_PROFILES="$PAPER_PROFILES"       PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"       PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"       PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS="${PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS:-0}"       ADAPTIVE_INTERVAL_ENABLED="${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-1}"       PYTHONUNBUFFERED=1       nohup "${COINBASE_CMD[@]}" > "$LOG" 2>&1 & disown
    else
      PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS="${PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS:-0}"       ADAPTIVE_INTERVAL_ENABLED="${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-1}"       PYTHONUNBUFFERED=1       nohup "${COINBASE_CMD[@]}" > "$LOG" 2>&1 & disown
    fi

    sleep 2
    if coinbase_spot_running; then
      echo "$LOG"
      echo "coinbase_loop_started simulate=$COINBASE_SIMULATE paper_mode=$PAPER_MODE"
      OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --require-coinbase --json >/dev/null 2>&1 || true
    else
      echo "coinbase_loop failed_to_start"
      tail -n 60 "$LOG" || true
      exit 1
    fi
    ;;
  coinbase-futures-start)
    FORCE_RESTART=0
    COINBASE_SIMULATE="${COINBASE_FUTURES_SIMULATE:-0}"
    PAPER_MODE=1
    FUTURES_PROFILE="${COINBASE_FUTURES_PROFILE:-crypto_futures}"
    PAPER_TOP_N="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}"
    PAPER_MIN_ACC="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.56}"
    PAPER_PROFILES="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-$FUTURES_PROFILE}"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --force-restart) FORCE_RESTART=1 ;;
        --paper) PAPER_MODE=1; COINBASE_SIMULATE=0 ;;
        --simulate) COINBASE_SIMULATE=1 ;;
        --live-data|--no-simulate) COINBASE_SIMULATE=0 ;;
        --top-n) PAPER_TOP_N="${2:-$PAPER_TOP_N}"; shift ;;
        --min-acc) PAPER_MIN_ACC="${2:-$PAPER_MIN_ACC}"; shift ;;
        --profiles) PAPER_PROFILES="${2:-$PAPER_PROFILES}"; shift ;;
        *) echo "unknown coinbase-futures-start arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    if [[ "$FORCE_RESTART" == "1" ]]; then
      pkill -f "scripts/run_shadow_training_loop.py --broker coinbase --profile $FUTURES_PROFILE" || true
      sleep 1
    fi

    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase --profile $FUTURES_PROFILE" | grep -v grep >/dev/null 2>&1; then
      PID="$(ps -axo pid,command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase --profile $FUTURES_PROFILE" | grep -v grep | awk 'NR==1{print $1}')"
      LATEST_LOG="$(ls -1t "$PROJECT_ROOT"/logs/coinbase_futures_live_*.log 2>/dev/null | head -n 1)"
      echo "coinbase_futures_loop already running pid=$PID profile=$FUTURES_PROFILE"
      [[ -n "$LATEST_LOG" ]] && echo "$LATEST_LOG"
      exit 0
    fi

    "$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true

    LOG="$PROJECT_ROOT/logs/coinbase_futures_live_$(date -u +%Y%m%d_%H%M%S).log"
    COINBASE_CMD=(
      "$PY" "$PROJECT_ROOT/scripts/run_shadow_training_loop.py"
      --broker coinbase
      --profile "$FUTURES_PROFILE"
      --domain crypto
      --symbols "${COINBASE_FUTURES_WATCH_SYMBOLS:-BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LINK-USD,DOGE-USD}"
      --context-symbols "${COINBASE_FUTURES_CONTEXT_SYMBOLS:-BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD}"
      --interval-seconds "${COINBASE_FUTURES_WATCH_INTERVAL_SECONDS:-20}"
      --max-iterations 0
    )
    if [[ "$COINBASE_SIMULATE" == "1" ]]; then
      COINBASE_CMD+=(--simulate)
    fi

    if [[ "$PAPER_MODE" == "1" ]]; then
      paper_trade_lock_env
      echo "coinbase_futures_paper=enabled profile=$FUTURES_PROFILE top_n=$PAPER_TOP_N min_acc=$PAPER_MIN_ACC profiles=$PAPER_PROFILES"
      SHADOW_PROFILE="$FUTURES_PROFILE"       SHADOW_DOMAIN=crypto       SHADOW_THRESHOLD_SHIFT="${COINBASE_FUTURES_THRESHOLD_SHIFT:-0.02}"       SIZING_MAX_NOTIONAL_PCT="${COINBASE_FUTURES_MAX_NOTIONAL_PCT:-0.03}"       PORTFOLIO_BASE_BUDGET="${COINBASE_FUTURES_BASE_BUDGET:-0.50}"       CROSS_SYMBOL_MAX_LONG="${COINBASE_FUTURES_MAX_LONG:-4}"       CROSS_SYMBOL_MAX_SHORT="${COINBASE_FUTURES_MAX_SHORT:-4}"       RISK_MAX_DAILY_LOSS_PROXY="${COINBASE_FUTURES_MAX_DAILY_LOSS_PROXY:-0.03}"       LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"       LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"       LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"       LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"       TOP_BOT_PAPER_TRADING_ENABLED=1       TOP_BOT_PAPER_TRADING_TOP_N="$PAPER_TOP_N"       TOP_BOT_PAPER_TRADING_MIN_ACC="$PAPER_MIN_ACC"       TOP_BOT_PAPER_TRADING_PROFILES="$PAPER_PROFILES"       PAPER_BROKER_BRIDGE_ENABLED="${PAPER_BROKER_BRIDGE_ENABLED:-1}"       PAPER_BROKER_BRIDGE_MODE="${PAPER_BROKER_BRIDGE_MODE:-jsonl}"       PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS="${PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS:-0}"       ADAPTIVE_INTERVAL_ENABLED="${COINBASE_FUTURES_ADAPTIVE_INTERVAL_ENABLED:-${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-1}}"       PYTHONUNBUFFERED=1       nohup "${COINBASE_CMD[@]}" > "$LOG" 2>&1 & disown
    else
      SHADOW_PROFILE="$FUTURES_PROFILE"       SHADOW_DOMAIN=crypto       SHADOW_THRESHOLD_SHIFT="${COINBASE_FUTURES_THRESHOLD_SHIFT:-0.02}"       SIZING_MAX_NOTIONAL_PCT="${COINBASE_FUTURES_MAX_NOTIONAL_PCT:-0.03}"       PORTFOLIO_BASE_BUDGET="${COINBASE_FUTURES_BASE_BUDGET:-0.50}"       CROSS_SYMBOL_MAX_LONG="${COINBASE_FUTURES_MAX_LONG:-4}"       CROSS_SYMBOL_MAX_SHORT="${COINBASE_FUTURES_MAX_SHORT:-4}"       RISK_MAX_DAILY_LOSS_PROXY="${COINBASE_FUTURES_MAX_DAILY_LOSS_PROXY:-0.03}"       LOG_SUB_BOT_DECISIONS="${LOG_SUB_BOT_DECISIONS:-1}"       LOG_MASTER_VARIANT_DECISIONS="${LOG_MASTER_VARIANT_DECISIONS:-1}"       LOG_GRAND_MASTER_DECISIONS="${LOG_GRAND_MASTER_DECISIONS:-1}"       LOG_OPTIONS_MASTER_DECISIONS="${LOG_OPTIONS_MASTER_DECISIONS:-1}"       PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS="${PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS:-0}"       ADAPTIVE_INTERVAL_ENABLED="${COINBASE_FUTURES_ADAPTIVE_INTERVAL_ENABLED:-${COINBASE_ADAPTIVE_INTERVAL_ENABLED:-1}}"       PYTHONUNBUFFERED=1       nohup "${COINBASE_CMD[@]}" > "$LOG" 2>&1 & disown
    fi

    sleep 2
    if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase --profile $FUTURES_PROFILE" | grep -v grep >/dev/null 2>&1; then
      echo "$LOG"
      echo "coinbase_futures_loop_started profile=$FUTURES_PROFILE simulate=$COINBASE_SIMULATE paper_mode=$PAPER_MODE"
      OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --require-coinbase-futures --json >/dev/null 2>&1 || true
    else
      OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --require-coinbase-futures --json >/dev/null 2>&1 || true
      sleep 2
      if ps -axo command | grep -F "scripts/run_shadow_training_loop.py --broker coinbase --profile $FUTURES_PROFILE" | grep -v grep >/dev/null 2>&1; then
        echo "$LOG"
        echo "coinbase_futures_loop_started profile=$FUTURES_PROFILE simulate=$COINBASE_SIMULATE paper_mode=$PAPER_MODE via=watchdog"
      else
        echo "coinbase_futures_loop failed_to_start profile=$FUTURES_PROFILE"
        tail -n 60 "$LOG" || true
        exit 1
      fi
    fi
    ;;
  coinbase-futures-stop)
    FUTURES_PROFILE="${COINBASE_FUTURES_PROFILE:-crypto_futures}"
    pkill -f "scripts/run_shadow_training_loop.py --broker coinbase --profile $FUTURES_PROFILE" || true
    echo "coinbase futures loop stopped profile=$FUTURES_PROFILE"
    ;;
  fx-start)
    FORCE_RESTART=0
    FX_SIMULATE="${FX_START_SIMULATE:-0}"
    PAPER_MODE=1
    DIRECT_EXECUTION=0
    FX_SYMBOL_SET="${FX_SYMBOLS:-UUP,FXE,FXY,FXB,FXC,FXA,CYB,EUO,YCS,UDN}"
    FX_CONTEXT_SET="${FX_CONTEXT_SYMBOLS:-SPY,QQQ,TLT,GLD,UUP,FXE,FXY,FXB,FXC,FXA}"
    FX_INTERVAL="${FX_SHADOW_INTERVAL:-45}"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --force-restart) FORCE_RESTART=1 ;;
        --paper) PAPER_MODE=1; FX_SIMULATE=0 ;;
        --simulate) FX_SIMULATE=1 ;;
        --live-data|--no-simulate) FX_SIMULATE=0 ;;
        --symbols) FX_SYMBOL_SET="${2:-$FX_SYMBOL_SET}"; shift ;;
        --context-symbols) FX_CONTEXT_SET="${2:-$FX_CONTEXT_SET}"; shift ;;
        --interval-seconds) FX_INTERVAL="${2:-$FX_INTERVAL}"; shift ;;
        --direct-execution) DIRECT_EXECUTION=1 ;;
        *) echo "unknown fx-start arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    if [[ "$DIRECT_EXECUTION" == "1" ]]; then
      if [[ "${SCHWAB_FOREX_API_VERIFIED:-0}" != "1" || "${FX_DIRECT_EXECUTION_ENABLED:-0}" != "1" ]]; then
        echo "fx-start direct execution blocked: Schwab forex API support is not officially verified for this stack" >&2
        exit 2
      fi
      echo "fx-start direct execution blocked: no direct Schwab forex execution implementation is enabled in this repo yet" >&2
      exit 2
    fi

    if [[ "$PAPER_MODE" != "1" ]]; then
      echo "fx-start only supports paper-only proxy mode" >&2
      exit 2
    fi

    if [[ "$FORCE_RESTART" == "1" ]]; then
      kill_fx_loops
      sleep 1
    fi

    if fx_running; then
      PID="$(fx_process_lines | awk 'NR==1{print $1}')"
      LATEST_LOG="$(ls -1t "$PROJECT_ROOT"/logs/fx_live_*.log 2>/dev/null | head -n 1)"
      echo "fx_loop already running pid=$PID"
      [[ -n "$LATEST_LOG" ]] && echo "$LATEST_LOG"
      exit 0
    fi

    "$PY" "$PROJECT_ROOT/scripts/ops/lock_watchdog.py" --apply --json >/dev/null 2>&1 || true

    LOG="$PROJECT_ROOT/logs/fx_live_$(date -u +%Y%m%d_%H%M%S).log"
    FX_CMD=(
      "$PY" "$PROJECT_ROOT/scripts/run_fx_shadow.py"
      --broker schwab
      --symbols "$FX_SYMBOL_SET"
      --context-symbols "$FX_CONTEXT_SET"
      --interval-seconds "$FX_INTERVAL"
      --max-iterations "${FX_SHADOW_MAX_ITERS:-0}"
    )
    if [[ "$FX_SIMULATE" == "1" ]]; then
      FX_CMD+=(--simulate)
    fi

    echo "fx_paper=enabled symbols=$FX_SYMBOL_SET context=$FX_CONTEXT_SET interval=$FX_INTERVAL"
    paper_trade_lock_env
    MARKET_DATA_ONLY=1 \
    ALLOW_ORDER_EXECUTION=0 \
    FX_DIRECT_EXECUTION_ENABLED=0 \
    SCHWAB_FOREX_API_VERIFIED="${SCHWAB_FOREX_API_VERIFIED:-0}" \
    nohup "${FX_CMD[@]}" > "$LOG" 2>&1 & disown

    sleep 2
    if fx_running; then
      echo "$LOG"
      echo "fx_loop_started simulate=$FX_SIMULATE paper_mode=$PAPER_MODE"
      OPS_WATCHDOG_REFRESH_REPORTS=0 "$PY" "$PROJECT_ROOT/scripts/ops/process_watchdog.py" --json >/dev/null 2>&1 || true
    else
      echo "fx_loop failed_to_start"
      tail -n 60 "$LOG" || true
      exit 1
    fi
    ;;
  fx-stop)
    kill_fx_loops
    echo "fx loop stopped"
    ;;
  coinbase-stop)
    kill_coinbase_spot_loops
    echo "coinbase loop stopped"
    ;;
  feed-refresh|livefeed-refresh|live-feed-refresh)
    SOURCE="all"
    ENFORCE_ALL_SOURCE=0
    if [[ "$cmd" == "livefeed-refresh" || "$cmd" == "live-feed-refresh" ]]; then
      ENFORCE_ALL_SOURCE=1
    fi
    SCHWAB_PAPER=1
    COINBASE_PAPER=1
    DRY_RUN=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        -h|--help)
          if [[ "$ENFORCE_ALL_SOURCE" == "1" ]]; then
            echo "Usage: opsctl.sh $cmd [--dry-run] [--paper|--schwab-paper] [--coinbase-paper]"
          else
            echo "Usage: opsctl.sh $cmd [--dry-run] [--source schwab|coinbase|fx|all] [--paper|--schwab-paper] [--coinbase-paper]"
          fi
          echo "Refresh live feed loops; dry-run validates the route without touching processes."
          exit 0
          ;;
        --dry-run) DRY_RUN=1 ;;
        --source)
          if [[ "$ENFORCE_ALL_SOURCE" == "1" ]]; then
            echo "$cmd refreshes all live feeds and does not accept --source" >&2
            exit 2
          fi
          SOURCE="${2:-all}"
          shift
          ;;
        --paper|--schwab-paper) SCHWAB_PAPER=1 ;;
        --coinbase-paper) COINBASE_PAPER=1 ;;
        *) echo "unknown feed-refresh arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    if [[ "$SOURCE" != "all" && "$SOURCE" != "schwab" && "$SOURCE" != "coinbase" && "$SOURCE" != "fx" ]]; then
      echo "--source must be all, schwab, coinbase, or fx" >&2
      exit 2
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "feed_refresh_dry_run=1"
      echo "cmd=$cmd"
      echo "source=$SOURCE"
      echo "schwab_paper=$SCHWAB_PAPER"
      echo "coinbase_paper=$COINBASE_PAPER"
      exit 0
    fi

    abort_loop_refresh_if_safety_flags_active "${cmd//-/_}"

    if [[ "$SOURCE" == "schwab" || "$SOURCE" == "all" ]]; then
      "$PROJECT_ROOT/scripts/ops/opsctl.sh" dividend-drip-sync --json >/dev/null 2>&1 || true
      kill_schwab_live_loops
      sleep 1
      start_schwab_live_loops "$SCHWAB_PAPER"
      "$PROJECT_ROOT/scripts/ops/opsctl.sh" schwab-futures-start --paper --force-restart --live-data
    fi

    if [[ "$SOURCE" == "coinbase" || "$SOURCE" == "all" ]]; then
      if [[ "$COINBASE_PAPER" == "1" ]]; then
        "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-start --paper --force-restart --live-data
        "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-futures-start --paper --force-restart --live-data
      else
        "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-start --force-restart --live-data
        "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-futures-start --force-restart --live-data
      fi
    fi

    if [[ "$SOURCE" == "fx" || "$SOURCE" == "all" ]]; then
      "$PROJECT_ROOT/scripts/ops/opsctl.sh" fx-market-sync --json >/dev/null 2>&1 || true
      "$PROJECT_ROOT/scripts/ops/opsctl.sh" fx-start --paper --force-restart --live-data
    fi

    if [[ "${LIVEFEED_REFRESH_MARKET_CORRELATION_SYNC:-0}" == "1" ]]; then
      "$PROJECT_ROOT/scripts/ops/opsctl.sh" market-correlation-sync \
        --lookback-days "${LIVEFEED_REFRESH_MARKET_CORRELATION_LOOKBACK_DAYS:-1}" \
        --bucket-seconds "${LIVEFEED_REFRESH_MARKET_CORRELATION_BUCKET_SECONDS:-300}" \
        --min-points "${LIVEFEED_REFRESH_MARKET_CORRELATION_MIN_POINTS:-3}" \
        --timeout-seconds "${LIVEFEED_REFRESH_MARKET_CORRELATION_TIMEOUT_SECONDS:-90}" \
        --json >/dev/null 2>&1 || true
    elif [[ "${LIVEFEED_REFRESH_MARKET_CORRELATION_ASYNC:-1}" == "1" ]]; then
      if market_correlation_running; then
        MARKET_CORR_PIDS="$(market_correlation_process_lines | awk '{print $1}' | paste -sd, -)"
        echo "market_correlation_sync_skipped reason=already_running pids=${MARKET_CORR_PIDS:-unknown}"
      else
        MARKET_CORR_LOG="$PROJECT_ROOT/logs/market_correlation_livefeed_$(date -u +%Y%m%d_%H%M%S).log"
        nohup "$PROJECT_ROOT/scripts/ops/opsctl.sh" market-correlation-sync \
          --lookback-days "${LIVEFEED_REFRESH_MARKET_CORRELATION_LOOKBACK_DAYS:-1}" \
          --bucket-seconds "${LIVEFEED_REFRESH_MARKET_CORRELATION_BUCKET_SECONDS:-300}" \
          --min-points "${LIVEFEED_REFRESH_MARKET_CORRELATION_MIN_POINTS:-3}" \
          --timeout-seconds "${LIVEFEED_REFRESH_MARKET_CORRELATION_TIMEOUT_SECONDS:-90}" \
          --json > "$MARKET_CORR_LOG" 2>&1 & disown
        echo "market_correlation_sync_started_async log=$MARKET_CORR_LOG"
      fi
    else
      echo "market_correlation_sync_skipped async=0 sync=0"
    fi
    echo "livefeed_refresh_completed source=$SOURCE schwab_paper=$SCHWAB_PAPER coinbase_paper=$COINBASE_PAPER"
    exit 0
    ;;
  storage-switch-local|storage-safe-eject)
    DO_REFRESH=1
    DO_EJECT=0
    DRY_RUN=0
    if [[ "$cmd" == "storage-safe-eject" ]]; then
      DO_EJECT=1
    fi

    while [[ $# -gt 0 ]]; do
      case "$1" in
        -h|--help)
          echo "Usage: opsctl.sh $cmd [--dry-run] [--no-refresh] [--eject|--no-eject]"
          echo "Switch storage routing to local fallback; dry-run prints the orchestrator intent."
          exit 0
          ;;
        --dry-run) DRY_RUN=1 ;;
        --no-refresh) DO_REFRESH=0 ;;
        --eject) DO_EJECT=1 ;;
        --no-eject) DO_EJECT=0 ;;
        *) echo "unknown $cmd arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    ORCH_ARGS=(--target-mode local)
    if [[ "$DO_REFRESH" != "1" ]]; then
      ORCH_ARGS+=(--no-restart)
    fi
    if [[ "$DO_EJECT" == "1" ]]; then
      ORCH_ARGS+=(--eject)
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "storage_switch_dry_run=1"
      echo "cmd=$cmd"
      echo "target_mode=local"
      echo "refresh=$DO_REFRESH"
      echo "eject=$DO_EJECT"
      echo "orchestrator_args=${ORCH_ARGS[*]}"
      exit 0
    fi

    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_switch_orchestrator.py" "${ORCH_ARGS[@]}"
    ;;
  storage-switch-external)
    DO_REFRESH=1
    DRY_RUN=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        -h|--help)
          echo "Usage: opsctl.sh storage-switch-external [--dry-run] [--no-refresh]"
          echo "Switch storage routing to the external volume; dry-run prints the orchestrator intent."
          exit 0
          ;;
        --dry-run) DRY_RUN=1 ;;
        --no-refresh) DO_REFRESH=0 ;;
        *) echo "unknown storage-switch-external arg: $1" >&2; exit 2 ;;
      esac
      shift
    done

    ORCH_ARGS=(--target-mode external)
    if [[ "$DO_REFRESH" != "1" ]]; then
      ORCH_ARGS+=(--no-restart)
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "storage_switch_dry_run=1"
      echo "cmd=$cmd"
      echo "target_mode=external"
      echo "refresh=$DO_REFRESH"
      echo "orchestrator_args=${ORCH_ARGS[*]}"
      exit 0
    fi

    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_switch_orchestrator.py" "${ORCH_ARGS[@]}"
    ;;
  storage-prune-standby|storage-standby-prune)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_standby_prune.py" "$@"
    ;;
  storage-transition-coordinator|storage-transition-bots)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_transition_coordinator.py" "$@"
    ;;
  storage-disaster-recovery|storage-recovery-bot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_disaster_recovery.py" "$@"
    ;;
  feed)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" "$@"
    ;;
  schwab-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source schwab "$@"
    ;;
  coinbase-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source coinbase "$@"
    ;;
  main-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source main "$@"
    ;;
  futures-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source futures "$@"
    ;;
  schwab-futures-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source schwab_futures "$@"
    ;;
  coinbase-futures-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source coinbase_futures "$@"
    ;;
  fx-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source fx "$@"
    ;;
  infra-tail)
    exec "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" --source infra "$@"
    ;;
  timeline-report)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/project_timeline_report.py" "$@"
    ;;
  crash-report)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/crash_report_digest.py" "$@"
    ;;
  training-report)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/training_report.py" "$@"
    ;;
  report-pdfs)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/sendout_pdf_refresh.py" "$@"
    ;;
  system-summary|executive-summary)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/system_summary_report.py" "$@"
    ;;
  system-summary-autopilot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/system_summary_autopilot.py" "$@"
    ;;
  system-explainers|system-explainer-docs)
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
      echo "Usage: opsctl.sh $cmd"
      echo "Generate system explainer markdown artifacts."
      exit 0
    fi
    exec "$PY" "$PROJECT_ROOT/scripts/ops/system_explainer_docs.py" "$@"
    ;;
  model-card)
    exec "$PY" "$PROJECT_ROOT/scripts/export_model_card.py" "$@"
    ;;
  explainability)
    exec "$PY" "$PROJECT_ROOT/scripts/export_bot_explainability.py" "$@"
    ;;
  strategy-attribution)
    exec "$PY" "$PROJECT_ROOT/scripts/strategy_attribution_report.py" "$@"
    ;;
  strategy-inventory|strategy-inventory-report)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/strategy_inventory_report.py" "$@"
    ;;
  strategy-research)
    exec "$PY" "$PROJECT_ROOT/scripts/strategy_research_lane.py" "$@"
    ;;
  derived-state)
    exec "$PY" "$PROJECT_ROOT/scripts/derived_state_snapshot.py" "$@"
    ;;
  cold-lane-refresh)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/cold_lane_refresh.py" "$@"
    ;;
  ops-coordinator)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/ops_coordinator.py" "$@"
    ;;
  storage-maintenance)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/storage_maintenance_lane.py" "$@"
    ;;
  paper-calibration)
    exec "$PY" "$PROJECT_ROOT/scripts/paper_execution_calibration_report.py" "$@"
    ;;
  paper-performance)
    exec "$PY" "$PROJECT_ROOT/scripts/paper_performance_report.py" "$@"
    ;;
  sentiment-report)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/sentiment_report.py" "$@"
    ;;
  post-trade-analysis)
    exec "$PY" "$PROJECT_ROOT/scripts/post_trade_analysis.py" "$@"
    ;;
  report-quality-guard|reporter-quality|reporter-infrabot|report-infrabot)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/report_quality_guard.py" "$@"
    ;;
  macro-bulletin)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/live_macro_bulletin.py" "$@"
    ;;
  macro-replay)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/live_macro_auto_watch.py" --once --replay-full-video "$@"
    ;;
  macro-media-ingest)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/live_macro_media_ingest.py" "$@"
    ;;
  macro-auto-start)
    FORCE_RESTART=0
    RUN_ONCE=0
    DRY_RUN=0
    PASS_ARGS=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        -h|--help)
          echo "Usage: opsctl.sh macro-auto-start [--dry-run] [--once] [--force-restart] [live_macro_auto_watch options]"
          echo "Start the macro auto watcher; dry-run validates the route without starting or stopping processes."
          exit 0
          ;;
        --dry-run) DRY_RUN=1 ;;
        --force-restart) FORCE_RESTART=1 ;;
        --once) RUN_ONCE=1; PASS_ARGS+=("$1") ;;
        *) PASS_ARGS+=("$1") ;;
      esac
      shift
    done

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "macro_auto_start_dry_run=1"
      echo "force_restart=$FORCE_RESTART"
      echo "run_once=$RUN_ONCE"
      echo "args=${PASS_ARGS[*]}"
      exit 0
    fi

    if [[ "$FORCE_RESTART" == "1" ]]; then
      pkill -f "scripts/ops/live_macro_auto_watch.py" || true
      rm -f "$PROJECT_ROOT/governance/health/macro_auto_watch.pid"
      sleep 1
    fi

    if [[ "$RUN_ONCE" == "1" ]]; then
      exec "$PY" "$PROJECT_ROOT/scripts/ops/live_macro_auto_watch.py" "${PASS_ARGS[@]}"
    fi

    if ps -axo command | grep -F "scripts/ops/live_macro_auto_watch.py" | grep -v grep >/dev/null 2>&1; then
      PID="$(ps -axo pid,command | grep -F "scripts/ops/live_macro_auto_watch.py" | grep -v grep | awk 'NR==1{print $1}')"
      LATEST_LOG="$(ls -1t "$PROJECT_ROOT"/logs/macro_auto_watch_*.log 2>/dev/null | head -n 1)"
      echo "macro_auto_watch already running pid=$PID"
      [[ -n "$LATEST_LOG" ]] && echo "$LATEST_LOG"
      exit 0
    fi

    LOG="$PROJECT_ROOT/logs/macro_auto_watch_$(date -u +%Y%m%d_%H%M%S).log"
    PYTHONUNBUFFERED=1 nohup "$PY" "$PROJECT_ROOT/scripts/ops/live_macro_auto_watch.py" "${PASS_ARGS[@]}" > "$LOG" 2>&1 & disown
    sleep 2
    if ps -axo command | grep -F "scripts/ops/live_macro_auto_watch.py" | grep -v grep >/dev/null 2>&1; then
      echo "$LOG"
      echo "macro_auto_watch_started"
    else
      echo "macro_auto_watch failed_to_start log=$LOG" >&2
      tail -n 60 "$LOG" || true
      exit 1
    fi
    ;;
  macro-auto-stop)
    pkill -f "scripts/ops/live_macro_auto_watch.py" || true
    rm -f "$PROJECT_ROOT/governance/health/macro_auto_watch.pid"
    echo "macro_auto_watch stopped"
    ;;
  macro-auto-status)
    STATUS_PATH="$PROJECT_ROOT/governance/health/macro_auto_watch_status.json"
    PID_PATH="$PROJECT_ROOT/governance/health/macro_auto_watch.pid"
    EXTERNAL_STATUS_PATH="/Volumes/BOT_LOGS/schwab_trading_bot/governance/health/macro_auto_watch_status.json"
    EXTERNAL_PID_PATH="/Volumes/BOT_LOGS/schwab_trading_bot/governance/health/macro_auto_watch.pid"
    if [[ -f "$STATUS_PATH" ]]; then
      cat "$STATUS_PATH"
    elif [[ -f "$EXTERNAL_STATUS_PATH" ]]; then
      cat "$EXTERNAL_STATUS_PATH"
    else
      echo "{\"ok\":false,\"reason\":\"status_missing\",\"status_path\":\"$STATUS_PATH\"}"
    fi
    if [[ -f "$PID_PATH" ]] && ps -p "$(cat "$PID_PATH")" >/dev/null 2>&1; then
      echo
      echo "pid=$(cat "$PID_PATH")"
    elif [[ -f "$EXTERNAL_PID_PATH" ]] && ps -p "$(cat "$EXTERNAL_PID_PATH")" >/dev/null 2>&1; then
      echo
      echo "pid=$(cat "$EXTERNAL_PID_PATH")"
    elif ps -axo pid,command | grep -F "scripts/ops/live_macro_auto_watch.py" | grep -v grep >/dev/null 2>&1; then
      echo
      echo "pid=$(ps -axo pid,command | grep -F "scripts/ops/live_macro_auto_watch.py" | grep -v grep | awk 'NR==1{print $1}')"
    fi
    ;;
  regime-validate)
    exec "$PY" "$PROJECT_ROOT/scripts/regime_segmented_validate.py" "$@"
    ;;
  retrain-force-full)
    load_runtime_profile live
    RETRAIN_TRIGGER_SOURCE="opsctl_retrain_force_full" \
    RETRAIN_TRIGGER_LABEL="opsctl:retrain-force-full" \
    RETRAIN_TRIGGER_CONTEXT="manual_force_full" \
    RETRAIN_AFTER_HOURS_ONLY=0 \
    RETRAIN_REQUIRE_DATA_QUALITY_FLOOR=0 \
    RETRAIN_REQUIRE_ARTIFACT_FRESHNESS=0 \
    RETRAIN_REQUIRE_SAMPLE_QUOTAS=0 \
    RETRAIN_REQUIRE_FULL_SNAPSHOT_SYNC=0 \
    RETRAIN_REFRESH_PROMOTION_ARTIFACTS=0 \
    RETRAIN_ALLOW_PRECHECK_FAILURES=1 \
    RETRAIN_THERMAL_GUARD=0 \
    RETRAIN_RETIRE_PERSISTENT_LOSERS=0 \
    MLX_METAL_JIT="${MLX_METAL_JIT:-0}" \
    exec "$TRAINING_PY" "$PROJECT_ROOT/scripts/weekly_retrain.py" --continue-on-error "$@"
    ;;
  retrain-force-targeted)
    if [[ $# -eq 0 ]]; then
      echo "retrain-force-targeted requires selector args such as --include-bot-ids or --regime-focus" >&2
      exit 2
    fi
    load_runtime_profile live
    RETRAIN_TRIGGER_SOURCE="opsctl_retrain_force_targeted" \
    RETRAIN_TRIGGER_LABEL="opsctl:retrain-force-targeted" \
    RETRAIN_TRIGGER_CONTEXT="manual_force_targeted" \
    RETRAIN_AFTER_HOURS_ONLY=0 \
    RETRAIN_REQUIRE_DATA_QUALITY_FLOOR=0 \
    RETRAIN_REQUIRE_ARTIFACT_FRESHNESS=0 \
    RETRAIN_REQUIRE_SAMPLE_QUOTAS=0 \
    RETRAIN_REQUIRE_FULL_SNAPSHOT_SYNC=0 \
    RETRAIN_REFRESH_PROMOTION_ARTIFACTS=0 \
    RETRAIN_ALLOW_PRECHECK_FAILURES=1 \
    RETRAIN_THERMAL_GUARD=0 \
    ENABLE_TRADE_BEHAVIOR_RETRAIN=0 \
    RETRAIN_DISTILLATION_STUDENT_EXTRA_PASS=0 \
    RETRAIN_NEW_BOT_EXTRA_PASS=0 \
    RETRAIN_RETIRE_PERSISTENT_LOSERS=0 \
    MLX_METAL_JIT="${MLX_METAL_JIT:-0}" \
    exec "$TRAINING_PY" "$PROJECT_ROOT/scripts/weekly_retrain.py" --continue-on-error "$@"
    ;;
  timeline-install-autoupdate)
    exec "$PROJECT_ROOT/scripts/install_project_timeline_autoupdate_launchd.sh" "$@"
    ;;
  token-refresh)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/premarket_token_guard.py" "$@"
    ;;
  chrome-headless-guard|chrome-guard|chrome-pdf-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/chrome_headless_guard.py" "$@"
    ;;
  one-numbers-regression-guard|one-numbers-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/one_numbers_regression_guard.py" "$@"
    ;;
  point-in-time-event-store|pit-event-store|event-store)
    exec "$PY" "$PROJECT_ROOT/scripts/point_in_time_event_store.py" "$@"
    ;;
  replay-hash-registry|replay-hash-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/replay_hash_registry_guard.py" "$@"
    ;;
  golden-replay-regression|golden-replay-guard)
    exec "$PY" "$PROJECT_ROOT/scripts/golden_replay_regression_guard.py" "$@"
    ;;
  cost-telemetry)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/cost_telemetry.py" "$@"
    ;;
  cross-host-parity|parity-report|parity-proof)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/cross_host_parity_report.py" "$@"
    ;;
  experiment-ledger|immutable-experiment-ledger)
    exec "$PY" "$PROJECT_ROOT/scripts/experiment_tracker.py" "$@"
    ;;
  token-refresh-interactive)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/schwab_auth_refresh.py" "$@"
    ;;
  token-install-autorefresh)
    exec "$PROJECT_ROOT/scripts/install_premarket_token_guard_launchd.sh" "$@"
    ;;
  notify-watch)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/mac_notification_watch.py" "$@"
    ;;
  notify-start)
    exec "$PROJECT_ROOT/scripts/install_mac_notification_watch_launchd.sh" "$@"
    ;;
  notify-stop)
    LABEL="com.dankingsley.mac_notification_watch"
    UID_NUM="$(id -u)"
    PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
    launchctl bootout "gui/$UID_NUM" "$PLIST_PATH" >/dev/null 2>&1 || true
    launchctl disable "gui/$UID_NUM/$LABEL" >/dev/null 2>&1 || true
    pkill -f "scripts/ops/mac_notification_watch.py" || true
    rm -f "$PROJECT_ROOT/governance/health/mac_notification_watch.pid"
    echo "notification_watch stopped"
    ;;
  notify-test)
    exec "$PY" "$PROJECT_ROOT/scripts/ops/mac_notification_watch.py" --test "$@"
    ;;
  help|*)
    cat <<'EOF'
opsctl commands:
  start [--profile sim|live] [--force-restart] [--no-coinbase] [--simulate] [--paper|--schwab-paper] [--coinbase-paper] [--coinbase-live-data] [--disable-circuit-breakers] [--run-all-sleeves]
  start-sim [--force-restart] [--no-coinbase] [--disable-circuit-breakers] [--run-all-sleeves]
  start-live [--force-restart] [--no-coinbase] [--paper|--schwab-paper] [--coinbase-paper] [--coinbase-live-data] [--disable-circuit-breakers] [--run-all-sleeves]
  stop
  status
  restart-sanity [--json] [--start-after] [--start-mode start|start-sim|start-live] [--force-restart]
  post-restart-settle|post-restart-settlement [--apply] [--max-renice-processes N] [--json]
  retrain
  retrain-force-full [extra weekly_retrain args...]
  retrain-force-targeted --include-bot-ids CSV [extra weekly_retrain args...]
  retrain-orchestrate [--bypass-market-guard] [--json]
  scorecard [--lookback-hours 24] [--json]
  sql-sync
  tradingeconomics-sync [--countries CSV] [--market-symbols CSV] [--lookahead-days N] [--news-limit N] [--json]
  macro-context-sync [--json]
  sec-edgar-sync [--symbols CSV] [--timeout N] [--pause-seconds N] [--json]
  extended-quant-sync [--symbols CSV] [--timeout N] [--json]
  quant-model-control [--no-render-pdf] [--json]
  pricing-grad [--spot N --strike N --expiry-days N --volatility N] [--json]
  gpu-mc-sim [--paths N] [--json]
  kalman-parallel [--symbols CSV] [--json]
  options-flow-sync [--symbols CSV] [--timeout-seconds N] [--json]
  tastytrade-sync [legacy alias for options-flow-sync]
  crypto-market-sync [--symbols CSV] [--timeout N] [--json]
  market-correlation-sync [--lookback-days N] [--bucket-seconds N] [--min-points N] [--timeout-seconds N] [--json]
  fx-market-sync [--timeout N] [--json]
  dividend-drip-sync [--lookback-days N] [--recent-window-days N] [--json]
  showcase-refresh
  system-explainers
  macro-crosscheck [--json]
  source-verification [--json]
  collector-contracts [--json]
  sleeve-strategy-coverage [--json]
  mlx-audit [--json]
  mlx-library-upgrade [--apply] [--json]
  mlx-audio-audit [--json]
  mlx-intelligence-router|mlx-compute-brain|mlx-utilization [--apply] [--json]
  library-utilization-router|library-router|non-mlx-library-router [--apply] [--json]
  onnx-audit [--json]
  pytorch-audit [--json]
  pytorch-replay-canary [--json]
  sql-audit [--json]
  training-registry-audit [--json]
  training-label-audit [--json]
  training-quality [--json]
  feature-store [--json]
  multiple-testing-guard [--json]
  decay-monitor [--json]
  security-audit
  secret-scan [--staged]
  schema-migration [--json]
  ingestion-storage-control [--json]
  ingestion-storage-governor [status|apply] [--json]
  external-backlog-drain [--apply] [--follow-through] [--poll-seconds N] [--wait-timeout-seconds N] [--force-live-window] [--json]
  backpressure-drainer-fleet [--apply] [--force-live-window] [--ttl-seconds N] [--json]
  external-backlog-retry-bot [--apply] [--poll-seconds N] [--wait-timeout-seconds N] [--command-timeout-seconds N] [--json]
  writer-cycle-coordinator [--apply] [--skip-drain] [--skip-maintenance] [--maintenance-force] [--maintenance-vacuum] [--poll-seconds N] [--wait-timeout-seconds N] [--command-timeout-seconds N] [--json]
  retention-debt-sheriff [--apply] [--force] [--poll-seconds N] [--wait-timeout-seconds N] [--command-timeout-seconds N] [--json]
  backpressure-slo-bot [--apply] [--command-timeout-seconds N] [--json]
  backlog-quarantine [--apply] [--max-move-files N] [--json]
  ingestion-priority-queue [--top-n N] [--mark-retry REL] [--ack REL] [--json]
  content-store [--path REL_OR_ABS] [--json]
  split-brain-reconcile [--force-failback-if-hashes-match] [--json]
  storage-resilience [--json]
  storage-tier-policy [--top-n N] [--hot-budget-gb N] [--cold-candidate-min-mb N] [--json]
  runtime-training-snapshot [--lookback-days N] [--reuse-if-fresh-minutes N] [--json]
  training-runtime-control [--fresh-minutes N] [--limit N] [--json]
  training-requalification [--include-bot-ids CSV] [--apply-repair] [--write-queue] [--json]
  coverage-seed [--write-queue] [--json]
  coverage-gap-closer [--apply-stage] [--launch] [--json]
  regime-control [--json]
  supportability-control [--limit N] [--json]
  teacher-quality [--json]
  bot-quality-autopilot [--apply] [--timeout-sec N] [--mentor-limit N] [--json]
  commands-hygiene [--apply] [--json]
  command-validity|commands-verify|command-audit [--apply] [--timeout-sec N] [--json]
  codex-project-guard|codex-guard [--staged] [--json]
  system-cleanliness-autopilot [--apply] [--timeout-sec N] [--json]
  system-cleanliness-infrabot [--apply] [--timeout-sec N] [--json]
  system-drift-guard|drift-guard|drift-mesh [--json]
  system-drift-autopilot|drift-autopilot|drift-mesh-autopilot [--apply] [--max-steps N] [--json]
  dashboard-refresh|runtime-artifact-refresh|runtime-contract-refresh [--json]
  dashboard [--skip-refresh] [--max-rows N] [--json]
  options-flow-export-hygiene [--apply] [--json]
  options-flow-efficiency [--apply] [--timeout-sec N] [--json]
  bot-stack-report [--top N] [--render-pdf] [--allow-gui-pdf-renderer] [--print]
  core-bot-catalog|bot-catalog [--md-out PATH] [--json-out PATH] [--json]
  core-bot-materialize|materialize-core-bots [--overwrite-generated] [--json]
  core-bot-materialization-guard|core-bot-file-guard [--json]
  core-bot-materialization-infrabot|core-bot-file-infrabot [--apply] [--overwrite-generated] [--json]
  core-bot-tier-organizer|organize-core-bot-tiers [--json]
  storage-backpressure-autopilot [--apply] [--poll-seconds N] [--wait-timeout-seconds N] [--command-timeout-seconds N] [--json]
  storage-pressure-clearance [--apply] [--force-clear-stale-gate] [--checkpoint-mode passive|restart|truncate] [--json]
  storage-reconnect-regression-guard [--skip-launchd] [--skip-swift-parse] [--json]
  storage-reconnect-infrabot [--apply] [--timeout-sec N] [--json]
  stateful-storage-regression-guard [--apply] [--json]
  data-collection-storage-guard [--apply] [--cleanup-duplicates] [--json]
  data-collection-observation-rollup [--apply] [--days N] [--bootstrap-tail-lines N] [--json]
  infrastructure-autofix [--apply] [--timeout-sec N] [--json]
  master-infra-supervisor|master-infrastructure-supervisor|infra-supervisor [--apply] [--timeout-sec N] [--json]
  coinbase-api-health|coinbase-health [--symbol SYMBOL] [--snapshot] [--json]
  global-halt-status|halt-status [--json]
  global-halt-refresh|global-halt-clear-blockers [--json]
  global-halt-auto-clear|halt-auto-clear [--json]
  clear-all-halts|clear-global-halts [--json]
  operator-control|operator-stop-status [--json]
  operator-release [--json]
  live-runtime-separation [--live-fresh-minutes N] [--json]
  rolling-restart [--max-session-age-minutes N] [--swap-restart-gb N] [--json]
  schwab-auth-supervisor [--apply] [--json]
  auth-lease [--min-lease-seconds N] [--critical-lease-seconds N] [--json]
  incident-timeline [--files-per-pattern N] [--rows-per-file N] [--recent-limit N] [--json]
  promotion-autopilot [--json]
  autonomy-control [--json]
  runtime-throttle|throttle-control|throttle-bot [--apply] [--max-renice-processes N] [--json]
  mode-switchboard|mode-switchboard-mission-control [--json]
  decision-provenance|decision-provenance-cards [--limit N] [--json]
  grade-regression-guard|regression-guard [--json]
  grade-regression-autopilot|grade-regression-bot|regression-upgrade [--apply] [--timeout-sec N] [--storage-max-cycles N] [--json]
  section-grade-guard|section-floor-guard|grade-floor-guard [--json]
  section-grade-autopilot|section-floor-autopilot|grade-floor-autopilot [--apply] [--json]
  blackstart-recovery [--max-session-age-minutes N] [--json]
  sleeve-isolation [--max-quarantine-events N] [--json]
  artifact-freshness-slo [--json]
  runtime-snapshot-cache [--fresh-minutes N] [--stale-minutes N] [--json]
  remote-alert-control [--hours N] [--ack-event NAME] [--ack-all-critical] [--json]
  storage-quota-guard [--json]
  release-freeze [--activate-days N | --clear-window] [--reason TEXT] [--json]
  legacy-bot-harmonizer|legacy-v107-harmonizer [--apply] [--json]
  roster-expansion [--apply-registry] [--json]
  roster-resilience [--json]
  chaos-drills [--record-drill NAME] [--note TEXT] [--json]
  calibration-control [--apply] [--json]
  portfolio-allocator [--intents-file PATH] [--json]
  portfolio-capacity-curves [--json]
  risk-service [--json]
  execution-lab [--json]
  operator-cockpit [--json]
  daily-verify-remediation [--apply] [--json]
  memory-efficiency [status|apply] [--json]
  swap-pressure-governor|swap-pressure [status|apply] [--json]
  creative-cotenant-guard|creative-cotenant|cotenant-guard [status|apply] [--json]
  platform-control-plane [--max-rows N] [--json]
  intelligence-capability-expansion|capability-expansion [--apply] [--json]
  platform-intelligence [--max-rows N] [--json]
  expansion-capacity [--wave-size N] [--json]
  advanced-intelligence-mesh|intelligence-mesh [--apply] [--json]
  cognitive-control-plane|cognitive-control [--apply] [--json]
  recursive-research-foundry|research-foundry [--apply] [--json]
  coordination-intelligence|strategy-conflict-resolver [--apply] [--json]
  adaptive-intelligence-kernel|intelligence-kernel [--apply] [--json]
  mlx-intelligence-router|mlx-compute-brain|mlx-utilization [--apply] [--json]
  library-utilization-router|library-router|non-mlx-library-router [--apply] [--json]
  big-platform-brain|platform-brain|system-self-model|self-model|self-awareness [--json]
  self-awareness-infrabots|system-self-awareness [--apply] [--json]
  alpha-intelligence-evolution|alpha-advancement [--apply] [--json]
  intelligence-layer-advancement|intelligence-layer-v2 [--apply] [--json]
  new-bot-admission|new-bot-admission-guard [--json]
  bot-founder-dna|founder-dna [--apply-registry] [--json]
  cross-host-parity [--json]
  experiment-ledger [--name TEXT] [--event-type TEXT] [--json]
  stale-sweeper [--stale-stage-sections all|logs,governance,exports] [--json]
  data-retention [--apply] [--stale-stage] [--stale-purge] [--json]
  stale-reaper [--stale-purge-days DAYS] [--stale-purge-low-value-days DAYS] [--max-delete-gb GB] [--json]
  model-lifecycle [--json]
  access-portable
  access-native
  access-status
  runtime-backend-switch [portable_auto|mlx|pytorch|onnx|tensorflow|jax] [--json]
  runtime-backend-status [--json]
  runtime-backend-native [--json]
  apple-profile [status|apply] [--tier air_safe|pro_balanced|max_throughput] [--json]
  storage-switch-local [--no-refresh]
  storage-switch-external [--no-refresh]
  storage-prune-standby [--apply] [--include-curated-standby] [--min-route-soak-hours N] [--relative-path PATH] [--json]
  storage-transition-coordinator [--transition-mode local|external] [--apply] [--json]
  storage-disaster-recovery|storage-recovery-bot [--apply] [--json]
  storage-safe-eject [--no-refresh] [--no-eject]
  sql-maint|sqlite-maint [--vacuum] [--json]
  health
  py314-canary|py314-ready [--refresh-deps] [--skip-install] [--json]
  doctor
  coinbase-start [paper default] [--paper] [--force-restart] [--live-data|--simulate] [--top-n N] [--min-acc X] [--profiles default]
  schwab-futures-start [paper default] [--paper] [--force-restart] [--live-data|--simulate] [--top-n N] [--min-acc X] [--profiles schwab_futures]
  schwab-futures-stop
  coinbase-stop
  coinbase-futures-start [paper default] [--paper] [--force-restart] [--live-data|--simulate] [--top-n N] [--min-acc X] [--profiles crypto_futures]
  coinbase-futures-stop
  fx-start [paper only] [--paper] [--force-restart] [--live-data|--simulate] [--symbols CSV] [--context-symbols CSV] [--interval-seconds N]
  fx-stop
  paper-trade-lock-infrabot [--apply] [--json]
  feed-refresh [paper default] [--dry-run] [--source schwab|coinbase|fx|all] [--paper|--schwab-paper] [--coinbase-paper]
  livefeed-refresh|live-feed-refresh [paper default] [--dry-run] [--paper|--schwab-paper] [--coinbase-paper]
  feed [--source schwab|coinbase|fx|futures|schwab_futures|coinbase_futures|main|infra|all] [--symbol SYMBOL] [--lines 40] [--raw] [--color|--no-color] [--include-decisions|--heavy]
  phone-feed [--host 127.0.0.1|0.0.0.0] [--port 8787] [--source all] [--lines 80] [--include-decisions] [--token TOKEN]
  schwab-tail [--symbol SYMBOL] [--lines 40]
  coinbase-tail [--symbol SYMBOL] [--lines 40]
  main-tail [--symbol SYMBOL] [--lines 40]
  futures-tail [--symbol SYMBOL] [--lines 40]
  schwab-futures-tail [--symbol SYMBOL] [--lines 40]
  coinbase-futures-tail [--symbol SYMBOL] [--lines 40]
  fx-tail [--symbol SYMBOL] [--lines 40]
  infra-tail [--lines 40] [--heavy]
  timeline-report [--auto] [--json]
  incident-review-packet|incident-packet [--render-pdf|--no-render-pdf] [--json]
  incident-report [--recent-limit N] [--surface-limit N] [--json-only] [--allow-gui-pdf-renderer] [--json]
  crash-report [--lookback-days N] [--recent-limit N] [--json]
  training-report [--render-pdf] [--allow-gui-pdf-renderer] [--json]
  report-pdfs [--only SLUG] [--json]
  system-summary [--refresh-supporting-artifacts] [--render-pdf] [--allow-gui-pdf-renderer] [--json]
  system-summary-autopilot [--step-timeout-seconds N] [--json]
  one-numbers-regression-guard [--apply] [--json]
  point-in-time-event-store [--limit N] [--json]
  replay-hash-registry [--json]
  golden-replay-regression [--json]
  model-card [--json]
  explainability [--limit N] [--bot-ids CSV] [--json]
  strategy-attribution [--day YYYYMMDD] [--json]
  strategy-inventory [--no-render-pdf] [--json]
  strategy-research [--day YYYYMMDD] [--max-rows N] [--skip-sandbox] [--max-age-minutes N] [--sandbox-max-age-minutes N] [--json]
  derived-state [--json]
  cold-lane-refresh [--day YYYYMMDD] [--strategy-max-age-minutes N] [--sandbox-max-age-minutes N] [--force] [--json]
  ops-coordinator [--day YYYYMMDD] [--max-rows N] [--strategy-max-age-minutes N] [--sandbox-max-age-minutes N] [--watchdog-refresh-max-age-seconds N] [--json]
  storage-maintenance [--force] [--vacuum] [--json]
  paper-calibration [--hours N] [--json]
  paper-performance [--day YYYYMMDD] [--week-days N] [--json]
  sentiment-report [--day YYYYMMDD] [--lookback-days N] [--allow-gui-pdf-renderer] [--json]
  post-trade-analysis [--day YYYYMMDD] [--hours N] [--json]
  report-quality-guard|reporter-quality|reporter-infrabot [--repair] [--json]
  macro-bulletin [--template powell|fed|policy_testimony|legal_policy|earnings_call|ceo_interview|analyst_day|generic] [--headline TEXT] [--summary TEXT] [--content TEXT] [--url URL] [--stance auto|hawkish|dovish|neutral|mixed] [--impact low|medium|high|critical] [--expires-hours N] [--status] [--clear] [--json]
  macro-auto-start (--youtube-url URL | --youtube-channel-url URL | --channel-preset apple_ir|nvidia_ir|tesla_ir|cnbc_ceo|bloomberg_ceo|cspan_legal|cspan_policy|cspan_general|fed_policy|treasury_policy|schwab_education|schwab_network) [--template powell|fed|policy_testimony|legal_policy|earnings_call|ceo_interview|analyst_day|generic] [--speaker NAME] [--source NAME] [--symbols CSV] [--poll-seconds N] [--lookback-seconds N] [--expires-hours N] [--correlate-with-schwab-calendar] [--trigger-media-ingest-on-live] [--trigger-media-ingest-before-minutes N] [--media-ingest-cookies-from-browser chrome|safari] [--media-ingest-retain-policy all|actionable_only] [--media-ingest-publish-bulletin] [--replay-full-video-on-stream-end] [--once] [--force-restart] [--json]
  macro-replay --youtube-url URL [--template powell|fed|policy_testimony|legal_policy|earnings_call|ceo_interview|analyst_day|generic] [--speaker NAME] [--source NAME] [--symbols CSV] [--replay-window-seconds N] [--expires-hours N] [--json]
  macro-media-ingest --youtube-url URL [--template powell|fed|policy_testimony|legal_policy|earnings_call|ceo_interview|analyst_day|generic] [--speaker NAME] [--source NAME] [--symbols CSV] [--language en] [--audio-format mp3] [--asr-backend auto|mlx_whisper] [--asr-model MODEL] [--cookies-from-browser chrome|safari] [--wait-for-live-seconds N] [--retry-interval-seconds N] [--retain-policy all|actionable_only] [--publish-bulletin] [--force-redownload] [--json]
  macro-auto-stop
  macro-auto-status
  regime-validate [--out-file PATH]
  timeline-install-autoupdate
  token-refresh [--always-auth] [--json]
  token-refresh-interactive [--force] [--callback-timeout-seconds N] [--requested-browser BROWSER] [--prompt-before-browser] [--skip-account-probe] [--json]
  token-install-autorefresh
  notify-watch [--poll-seconds N] [--enable-imessage] [--imessage-recipient DEST] [--imessage-min-severity info|warn|critical] [--imessage-event-allowlist CSV]
  notify-start [--poll-seconds N] [--enable-imessage] [--imessage-recipient DEST] [--imessage-min-severity info|warn|critical] [--imessage-event-allowlist CSV]
  notify-stop
  notify-test [--enable-imessage] [--imessage-recipient DEST] [--imessage-min-severity info|warn|critical] [--imessage-event-allowlist CSV]
EOF
    ;;
esac
