#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$PROJECT_ROOT/.venv312/bin/python"
DAY_UTC="$(date -u +%Y%m%d)"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

cd "$PROJECT_ROOT"

[[ -f "$PROJECT_ROOT/scripts/load_ops_thresholds_env.sh" ]] && source "$PROJECT_ROOT/scripts/load_ops_thresholds_env.sh"

export MARKET_DATA_ONLY="${MARKET_DATA_ONLY:-1}"
export ALLOW_ORDER_EXECUTION="${ALLOW_ORDER_EXECUTION:-0}"

export SHADOW_SYMBOLS_CORE="${SHADOW_SYMBOLS_CORE:-SPY,QQQ,AAPL,MSFT,NVDA,DIA,IWM,MDY}"
export SHADOW_SYMBOLS_VOLATILE="${SHADOW_SYMBOLS_VOLATILE:-SOXL,SOXS,MSTR,SMCI,COIN,TSLA,UVXY,VIXY}"
export SHADOW_SYMBOLS_DEFENSIVE="${SHADOW_SYMBOLS_DEFENSIVE:-TLT,GLD,XLV,XLU,XLP,HYG,LQD,UUP,XLE,XLF,XLI,XLK,XLY,IEF,SHY,TIP,TLH,JNK}"
export SHADOW_SYMBOLS_COMMOD_FX_INTL="${SHADOW_SYMBOLS_COMMOD_FX_INTL:-DBC,UNG,CORN,SLV,USO,FXE,FXY,EFA,EEM,EWJ,FXI}"

existing_launchers=$(ps -axo command | grep "scripts/run_parallel_shadows.py" | grep -v grep | wc -l | tr -d ' ')
if [[ "$existing_launchers" -gt 0 && "$FORCE" -ne 1 ]]; then
  echo "start_session blocked: existing run_parallel_shadows.py detected ($existing_launchers). Use --force if intentional."
  exit 1
fi

"$PY" "$PROJECT_ROOT/scripts/capture_run_config.py"
"$PY" "$PROJECT_ROOT/scripts/version_data_features.py"
"$PY" "$PROJECT_ROOT/scripts/shadow_preflight.py" --broker schwab --simulate
"$PY" "$PROJECT_ROOT/scripts/resource_guard.py"
"$PY" "$PROJECT_ROOT/scripts/restore_state_probe.py" || true

"$PY" "$PROJECT_ROOT/scripts/link_jsonl_to_sql.py" --mode sqlite
"$PY" "$PROJECT_ROOT/scripts/build_one_numbers_report.py" --day "$DAY_UTC"
"$PY" "$PROJECT_ROOT/scripts/daily_auto_verify.py" --day "$DAY_UTC" || true

if [[ -f "$HOME/Library/LaunchAgents/com.dankingsley.shadow_watchdog.plist" ]]; then
  launchctl unload "$HOME/Library/LaunchAgents/com.dankingsley.shadow_watchdog.plist" >/dev/null 2>&1 || true
  launchctl load "$HOME/Library/LaunchAgents/com.dankingsley.shadow_watchdog.plist"
else
  "$PROJECT_ROOT/scripts/install_shadow_watchdog_launchd.sh"
fi

sleep 2
"$PY" "$PROJECT_ROOT/scripts/shadow_watchdog.py" --once --watch-coinbase --simulate-schwab --interval-seconds 30
"$PY" "$PROJECT_ROOT/scripts/session_ready_check.py" || true
"$PY" "$PROJECT_ROOT/scripts/event_bus_relay.py" --day "$DAY_UTC" || true

"$PY" "$PROJECT_ROOT/scripts/experiment_tracker.py" --name "runtime_session" --status "started" --notes "start_session" || true

echo "session_start_complete"
