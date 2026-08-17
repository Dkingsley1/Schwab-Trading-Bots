#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export BOT_RUNTIME_LANE="${BOT_RUNTIME_LANE:-${BOT_SHADOW_RUNTIME_LANE:-canary314}}"
export BOT_PYTHON_VERSION="${BOT_PYTHON_VERSION:-3.14.5}"
export BOT_TRAINING_RUNTIME_LANE="${BOT_TRAINING_RUNTIME_LANE:-canary314}"
export BOT_TRAINING_PYTHON_VERSION="${BOT_TRAINING_PYTHON_VERSION:-3.14.5}"
export PY314_RUNTIME_FLIP_APPROVED="${PY314_RUNTIME_FLIP_APPROVED:-1}"
export PY314_RETIRE_312_ANCHOR="${PY314_RETIRE_312_ANCHOR:-1}"
unset __PYVENV_LAUNCHER__
RUNNER_SCRIPT="$PROJECT_ROOT/scripts/ops/run_shadow_watchdog_launchd.sh"
PLIST_PATH="$HOME/Library/LaunchAgents/com.dankingsley.shadow_watchdog.plist"
LABEL="com.dankingsley.shadow_watchdog"
UID_NUM="$(id -u)"
LOG_DIR="${SHADOW_WATCHDOG_LAUNCHD_LOG_DIR:-$HOME/Library/Logs/schwab_trading_bot/launchd_watchdog}"
OUT_LOG="$LOG_DIR/shadow_watchdog.out.log"
ERR_LOG="$LOG_DIR/shadow_watchdog.err.log"
BOOT_LOG="$LOG_DIR/shadow_watchdog.boot.log"
RUNTIME_PROFILE="${BOT_RUNTIME_PROFILE:-live}"
MARKET_OPEN_HOUR="${MARKET_SESSION_START_HOUR:-4}"
PAPER_TRADE_LOCK_PATH="$PROJECT_ROOT/governance/health/PAPER_TRADE_LOCK.flag"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
chmod +x "$RUNNER_SCRIPT"
mkdir -p "$(dirname "$PAPER_TRADE_LOCK_PATH")"
printf 'enabled_at_utc=%s\npolicy=live_data_paper_trade_only\nmanaged_by=install_shadow_watchdog_launchd\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$PAPER_TRADE_LOCK_PATH"

source "$PROJECT_ROOT/scripts/ops/runtime_python.sh"
if [[ -f "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/scripts/ops/load_runtime_env.sh" "$RUNTIME_PROFILE" --quiet
fi
PYTHON_BIN="$(resolve_runtime_python)"
PYTHON_PROGRAM="$(readlink "$PYTHON_BIN" 2>/dev/null || true)"
if [[ -z "$PYTHON_PROGRAM" ]]; then
  PYTHON_PROGRAM="$PYTHON_BIN"
fi

xml_escape() {
  local s="${1:-}"
  s="${s//&/&amp;}"
  s="${s//</&lt;}"
  s="${s//>/&gt;}"
  printf '%s' "$s"
}

plist_arg() {
  printf '    <string>%s</string>\n' "$(xml_escape "$1")"
}

json_argv() {
  "$PYTHON_BIN" - "$@" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:]))
PY
}

paper_watchdog_args() {
  local mode_var="$1"
  local top_n="$2"
  local min_acc="$3"
  local profiles="$4"
  local -a args=(--live-data --top-n "$top_n" --min-acc "$min_acc" --profiles "$profiles")
  if [[ "$mode_var" == "1" ]]; then
    args=(--paper "${args[@]}")
  fi
  printf '%s\n' "${args[@]}"
}

credentials_ready_for_watchdog() {
  local key="${SCHWAB_API_KEY:-}"
  local secret="${SCHWAB_SECRET:-}"
  case "$key" in
    ""|"YOUR_KEY_HERE"|"YOUR_REAL_KEY"|"<real_key>") return 1 ;;
  esac
  case "$secret" in
    ""|"YOUR_SECRET_HERE"|"YOUR_REAL_SECRET"|"<real_secret>") return 1 ;;
  esac
  return 0
}

SCHWAB_FUTURES_PROFILE="${SCHWAB_FUTURES_PROFILE:-schwab_futures}"
SCHWAB_FUTURES_TOP_N="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}"
SCHWAB_FUTURES_MIN_ACC="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.53}"
SCHWAB_FUTURES_PROFILES="${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-$SCHWAB_FUTURES_PROFILE}"
COINBASE_TOP_N="${COINBASE_TOP_BOT_PAPER_TRADING_TOP_N:-${TOP_BOT_PAPER_TRADING_TOP_N:-5}}"
COINBASE_MIN_ACC="${COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC:-${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.58}}"
COINBASE_PROFILES="${COINBASE_TOP_BOT_PAPER_TRADING_PROFILES:-default}"
COINBASE_FUTURES_PROFILE="${COINBASE_FUTURES_PROFILE:-crypto_futures}"
COINBASE_FUTURES_TOP_N="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}"
COINBASE_FUTURES_MIN_ACC="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.56}"
COINBASE_FUTURES_PROFILES="${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-$COINBASE_FUTURES_PROFILE}"

SCHWAB_START_CMD="$(json_argv "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_all_sleeves.py" --with-aggressive-modes)"
AGGRESSIVE_START_CMD="$(json_argv "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_parallel_aggressive_modes.py")"
DIVIDEND_START_CMD="$(json_argv "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_dividend_shadow.py" --interval-seconds 60)"
DIVIDEND_CAPTURE_START_CMD="$(json_argv "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_dividend_capture_shadow.py" --interval-seconds 60)"
BOND_START_CMD="$(json_argv "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_bond_shadow.py" --interval-seconds 120)"
FX_START_CMD="$(json_argv "$PROJECT_ROOT/scripts/ops/opsctl.sh" fx-start --paper --live-data)"
SCHWAB_FUTURES_START_CMD="$(json_argv "$PROJECT_ROOT/scripts/ops/opsctl.sh" schwab-futures-start $(paper_watchdog_args "${SCHWAB_FUTURES_WATCHDOG_PAPER_MODE:-1}" "$SCHWAB_FUTURES_TOP_N" "$SCHWAB_FUTURES_MIN_ACC" "$SCHWAB_FUTURES_PROFILES"))"
COINBASE_START_CMD="$(json_argv "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-start $(paper_watchdog_args "${COINBASE_WATCHDOG_PAPER_MODE:-1}" "$COINBASE_TOP_N" "$COINBASE_MIN_ACC" "$COINBASE_PROFILES"))"
COINBASE_FUTURES_START_CMD="$(json_argv "$PROJECT_ROOT/scripts/ops/opsctl.sh" coinbase-futures-start $(paper_watchdog_args "${COINBASE_FUTURES_WATCHDOG_PAPER_MODE:-1}" "$COINBASE_FUTURES_TOP_N" "$COINBASE_FUTURES_MIN_ACC" "$COINBASE_FUTURES_PROFILES"))"

WATCHDOG_ARGS=(
  "$PYTHON_PROGRAM"
  "$PROJECT_ROOT/scripts/shadow_watchdog.py"
  --watch-schwab-futures
  --watch-coinbase
  --watch-coinbase-futures
  --interval-seconds "${SHADOW_WATCHDOG_INTERVAL_SECONDS:-20}"
  --max-restarts-per-window "${SHADOW_WATCHDOG_MAX_RESTARTS_PER_WINDOW:-12}"
  --restart-window-seconds "${SHADOW_WATCHDOG_RESTART_WINDOW_SECONDS:-3600}"
  --schwab-heartbeat-stale-seconds "${SHADOW_WATCHDOG_SCHWAB_HEARTBEAT_STALE_SECONDS:-180}"
  --coinbase-heartbeat-stale-seconds "${SHADOW_WATCHDOG_COINBASE_HEARTBEAT_STALE_SECONDS:-210}"
  --schwab-start-cmd "$SCHWAB_START_CMD"
  --schwab-futures-start-cmd "$SCHWAB_FUTURES_START_CMD"
  --aggressive-modes-start-cmd "$AGGRESSIVE_START_CMD"
  --dividend-start-cmd "$DIVIDEND_START_CMD"
  --dividend-capture-start-cmd "$DIVIDEND_CAPTURE_START_CMD"
  --bond-start-cmd "$BOND_START_CMD"
  --fx-start-cmd "$FX_START_CMD"
  --coinbase-start-cmd "$COINBASE_START_CMD"
  --coinbase-futures-start-cmd "$COINBASE_FUTURES_START_CMD"
  --no-event-log
)

if ! credentials_ready_for_watchdog; then
  WATCHDOG_ARGS=( "${WATCHDOG_ARGS[@]:0:4}" --schwab-futures-optional "${WATCHDOG_ARGS[@]:4}" )
fi

if [[ "${SHADOW_WATCHDOG_DIRECT_CHILD_SLEEVES:-0}" == "1" ]]; then
  WATCHDOG_ARGS=( "${WATCHDOG_ARGS[@]:0:4}" --watch-aggressive-modes --watch-dividend --watch-bond --watch-fx "${WATCHDOG_ARGS[@]:4}" )
  if [[ "${DIVIDEND_CAPTURE_SHADOW_ENABLED:-1}" == "1" ]]; then
    WATCHDOG_ARGS=( "${WATCHDOG_ARGS[@]:0:8}" --watch-dividend-capture "${WATCHDOG_ARGS[@]:8}" )
  fi
fi

PROGRAM_ARGUMENTS_XML=""
for arg in "${WATCHDOG_ARGS[@]}"; do
  PROGRAM_ARGUMENTS_XML+="$(plist_arg "$arg")"$'\n'
done

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>

  <key>Program</key>
  <string>/bin/zsh</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$(xml_escape "$RUNNER_SCRIPT")</string>
  </array>

  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
    <key>BOT_RUNTIME_PROFILE</key><string>$RUNTIME_PROFILE</string>
    <key>BOT_RUNTIME_LANE</key><string>${BOT_RUNTIME_LANE:-${BOT_SHADOW_RUNTIME_LANE:-canary314}}</string>
    <key>BOT_PYTHON_VERSION</key><string>${BOT_PYTHON_VERSION:-3.14.5}</string>
    <key>BOT_TRAINING_RUNTIME_LANE</key><string>${BOT_TRAINING_RUNTIME_LANE:-canary314}</string>
    <key>BOT_TRAINING_PYTHON_VERSION</key><string>${BOT_TRAINING_PYTHON_VERSION:-3.14.5}</string>
    <key>PY314_RUNTIME_FLIP_APPROVED</key><string>${PY314_RUNTIME_FLIP_APPROVED:-1}</string>
    <key>PY314_RETIRE_312_ANCHOR</key><string>${PY314_RETIRE_312_ANCHOR:-1}</string>
    <key>MARKET_SESSION_START_HOUR</key><string>$MARKET_OPEN_HOUR</string>
    <key>MARKET_DATA_ONLY</key><string>1</string>
    <key>ALLOW_ORDER_EXECUTION</key><string>0</string>
    <key>TOP_BOT_PAPER_TRADING_TOP_N</key><string>${TOP_BOT_PAPER_TRADING_TOP_N:-5}</string>
    <key>TOP_BOT_PAPER_TRADING_MIN_ACC</key><string>${TOP_BOT_PAPER_TRADING_MIN_ACC:-0.55}</string>
    <key>TOP_BOT_PAPER_TRADING_PROFILES</key><string>${TOP_BOT_PAPER_TRADING_PROFILES:-default,conservative,aggressive,intraday_aggressive,swing_aggressive,dividend,bond,fx}</string>
    <key>DIVIDEND_CAPTURE_SHADOW_ENABLED</key><string>${DIVIDEND_CAPTURE_SHADOW_ENABLED:-1}</string>
    <key>TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED</key><string>${TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED:-1}</string>
    <key>TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N</key><string>${TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N:-2}</string>
    <key>TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC</key><string>${TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC:-0.55}</string>
    <key>TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES</key><string>${TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES:-default,aggressive,intraday_aggressive,swing_aggressive,options_on_futures,options_on_futures_aggressive}</string>
    <key>SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N</key><string>${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}</string>
    <key>SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC</key><string>${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.53}</string>
    <key>SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES</key><string>${SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-schwab_futures}</string>
    <key>COINBASE_TOP_BOT_PAPER_TRADING_TOP_N</key><string>${COINBASE_TOP_BOT_PAPER_TRADING_TOP_N:-5}</string>
    <key>COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC</key><string>${COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.58}</string>
    <key>COINBASE_TOP_BOT_PAPER_TRADING_PROFILES</key><string>${COINBASE_TOP_BOT_PAPER_TRADING_PROFILES:-default}</string>
    <key>COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N</key><string>${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N:-10}</string>
    <key>COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC</key><string>${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC:-0.56}</string>
    <key>COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES</key><string>${COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES:-crypto_futures}</string>
    <key>SCHWAB_FUTURES_WATCHDOG_PAPER_MODE</key><string>${SCHWAB_FUTURES_WATCHDOG_PAPER_MODE:-1}</string>
    <key>COINBASE_WATCHDOG_PAPER_MODE</key><string>${COINBASE_WATCHDOG_PAPER_MODE:-1}</string>
    <key>COINBASE_FUTURES_WATCHDOG_PAPER_MODE</key><string>${COINBASE_FUTURES_WATCHDOG_PAPER_MODE:-1}</string>
    <key>SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT</key><string>${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT:-1}</string>
    <key>SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_MIN_AGE_SECONDS</key><string>${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_MIN_AGE_SECONDS:-60}</string>
    <key>SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_ALLOWED_REASONS</key><string>${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_ALLOWED_REASONS:-incident_auto_halt,global_risk_killswitch,repeated_hard_gates,softguard_api_circuit_opened}</string>
    <key>SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_REQUIRE_PAPER_ONLY</key><string>${SHADOW_WATCHDOG_AUTO_CLEAR_GLOBAL_HALT_REQUIRE_PAPER_ONLY:-1}</string>
    <key>SHADOW_WATCHDOG_ALLOW_SCHWAB_STANDBY_HEARTBEATS</key><string>${SHADOW_WATCHDOG_ALLOW_SCHWAB_STANDBY_HEARTBEATS:-1}</string>
    <key>SHADOW_WATCHDOG_BOOT_LOG</key><string>$BOOT_LOG</string>
  </dict>

  <key>WorkingDirectory</key>
  <string>$PROJECT_ROOT</string>

  <key>RunAtLoad</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>${SHADOW_WATCHDOG_LAUNCHD_THROTTLE_INTERVAL_SECONDS:-30}</integer>
  <key>KeepAlive</key>
  <dict>
    <key>Crashed</key>
    <true/>
    <key>SuccessfulExit</key>
    <false/>
  </dict>

  <key>StandardOutPath</key>
  <string>$OUT_LOG</string>
  <key>StandardErrorPath</key>
  <string>$ERR_LOG</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl bootout "gui/$UID_NUM" "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl enable "gui/$UID_NUM/$LABEL" || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_PATH"
launchctl kickstart -k "gui/$UID_NUM/$LABEL" || true

echo "Installed and loaded: $PLIST_PATH"
echo "Logs: $OUT_LOG and $ERR_LOG"
