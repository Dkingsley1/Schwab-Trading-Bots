#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HEALTH_DIR="$PROJECT_ROOT/governance/health"
LOG_DIR="$PROJECT_ROOT/governance/local_livefeed_logs"
OUTPUT_LOG="$LOG_DIR/livefeed_heavy_guarded.out.log"
STATE_FILE="$HEALTH_DIR/live_feed_heavy_guarded_latest.json"

SOURCE="main"
LINES="80"
CHECK_INTERVAL_SECONDS="20"
WAIT_SECONDS="60"
TAIL_LINES="120"
HEAVY_TTL_SECONDS="${LIVE_FEED_GUARDED_HEAVY_TTL_SECONDS:-900}"
RUN_ONCE="0"
CHECK_ONLY="0"
COLOR_ARGS=(--color --red-actions)

usage() {
  cat <<'EOF'
Usage: scripts/ops/live_feed_heavy_guarded.sh [--source main|all|infra|schwab|coinbase|fx|futures|schwab_futures|coinbase_futures] [--lines N] [--check-interval-seconds N] [--wait-seconds N] [--tail-lines N] [--ttl-seconds N] [--once] [--check-only] [--no-color]

Runs a bounded heavy livefeed while runtime/storage/Mac fluidity safeguards are clear.
When safeguards say to wait, the heavy child is stopped and this command waits
until the guard clears. Terminal only follows one processed heavy-output file.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="${2:-main}"
      shift 2
      ;;
    --lines)
      LINES="${2:-80}"
      shift 2
      ;;
    --check-interval-seconds)
      CHECK_INTERVAL_SECONDS="${2:-20}"
      shift 2
      ;;
    --wait-seconds)
      WAIT_SECONDS="${2:-60}"
      shift 2
      ;;
    --tail-lines)
      TAIL_LINES="${2:-120}"
      shift 2
      ;;
    --ttl-seconds|--heavy-ttl-seconds)
      HEAVY_TTL_SECONDS="${2:-900}"
      shift 2
      ;;
    --once)
      RUN_ONCE="1"
      shift
      ;;
    --check-only)
      CHECK_ONLY="1"
      shift
      ;;
    --no-color|--no-highlight)
      COLOR_ARGS=(--no-color)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown guarded heavy livefeed arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$SOURCE" in
  schwab|coinbase|fx|futures|schwab_futures|coinbase_futures|main|infra|all)
    ;;
  *)
    echo "--source must be schwab, coinbase, fx, futures, schwab_futures, coinbase_futures, main, infra, or all" >&2
    exit 2
    ;;
esac

for numeric_value in "$LINES" "$CHECK_INTERVAL_SECONDS" "$WAIT_SECONDS" "$TAIL_LINES" "$HEAVY_TTL_SECONDS"; do
  if ! [[ "$numeric_value" =~ ^[0-9]+$ ]]; then
    echo "numeric options must be positive integers" >&2
    exit 2
  fi
done

mkdir -p "$HEALTH_DIR" "$LOG_DIR"

PYTHON_BIN="$PROJECT_ROOT/.venv314/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3 not found; cannot evaluate heavy livefeed safeguards" >&2
  exit 2
fi

HEAVY_PID=""
VIEWER_PID=""

write_state() {
  local state_status="$1"
  local allowed="$2"
  local reason="$3"
  local heavy_pid="${4:-}"
  "$PYTHON_BIN" - "$STATE_FILE" "$state_status" "$allowed" "$reason" "$SOURCE" "$heavy_pid" "$OUTPUT_LOG" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
status = sys.argv[2]
allowed = sys.argv[3].lower() == "true"
reason = sys.argv[4]
source = sys.argv[5]
heavy_pid = sys.argv[6]
output_log = sys.argv[7]

payload = {
    "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "schema_version": 1,
    "status": status,
    "allowed": allowed,
    "reason": reason,
    "source": source,
    "heavy_pid": int(heavy_pid) if str(heavy_pid).isdigit() else None,
    "output_log": output_log,
    "contract": "guarded_operator_heavy_livefeed",
}
path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY
}

guard_status() {
  "$PYTHON_BIN" - "$PROJECT_ROOT" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
health = root / "governance" / "health"


def load(name: str) -> dict:
    try:
        payload = json.loads((health / name).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def as_int(value, default=0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def lower(value) -> str:
    return str(value or "").strip().lower()


def age_seconds(payload: dict) -> float:
    stamp = str(payload.get("timestamp_utc") or "")
    if not stamp:
        return 999999.0
    try:
        if stamp.endswith("Z"):
            stamp = stamp[:-1] + "+00:00"
        return (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()
    except Exception:
        return 999999.0


runtime = load("runtime_throttle_control_latest.json")
ingestion = load("ingestion_backpressure_latest.json")
pressure = load("pressure_relief_control_latest.json")

reasons: list[str] = []
warnings: list[str] = []

runtime_age = age_seconds(runtime)
if runtime_age > 600:
    reasons.append(f"runtime_throttle_stale:{int(runtime_age)}s")

runtime_status = lower(runtime.get("overall_status"))
throttle_profile = lower(runtime.get("throttle_profile"))
compute = lower(runtime.get("compute_pressure_level"))
memory = lower(runtime.get("memory_pressure_level"))
host = as_float(runtime.get("host_saturation_score"), 0.0)
if runtime_status in {"blocked", "critical", "saturated", "protect"}:
    reasons.append(f"runtime_status={runtime_status}")
if throttle_profile in {"protect_live", "protect", "survival"}:
    reasons.append(f"throttle_profile={throttle_profile}")
if compute in {"high", "critical", "blocked"}:
    reasons.append(f"compute={compute}")
if memory in {"high", "critical", "blocked"}:
    reasons.append(f"memory={memory}")
if host >= 68:
    reasons.append(f"host_saturation={host:.1f}")

mac = runtime.get("mac_fluidity_contract") if isinstance(runtime.get("mac_fluidity_contract"), dict) else {}
fluidity_status = lower(mac.get("overall_status"))
fluidity_band = lower(mac.get("fluidity_band"))
fluidity_score = as_float(mac.get("fluidity_score"), 100.0)
if fluidity_status in {"blocked", "critical", "protect"}:
    reasons.append(f"mac_fluidity_status={fluidity_status}")
if fluidity_band in {"protect", "pause", "saturated"}:
    reasons.append(f"mac_fluidity_band={fluidity_band}")
if fluidity_score < 75:
    reasons.append(f"mac_fluidity_score={fluidity_score:.1f}")

storage = runtime.get("runtime_snapshot", {}).get("storage_pressure", {})
if not isinstance(storage, dict):
    storage = {}
storage_index = as_float(storage.get("pressure_index"), as_float((pressure.get("storage_pressure") or {}).get("pressure_index"), 0.0))
storage_severity = lower(storage.get("severity") or (pressure.get("storage_pressure") or {}).get("severity"))
total_pending = as_int(storage.get("total_pending_lines"), as_int(ingestion.get("pending_lines_total"), 0))
pending_threshold = as_int(storage.get("pending_lines_threshold"), as_int(ingestion.get("pending_lines_threshold"), 15000))
oldest_age = as_float(storage.get("oldest_pending_age_seconds"), as_float(ingestion.get("oldest_pending_age_seconds_total"), 0.0))
oldest_threshold = as_float(storage.get("oldest_age_threshold_seconds"), as_float(ingestion.get("oldest_age_threshold_seconds"), 240.0))
if storage_severity in {"critical", "blocked"} and storage_index >= 0.85:
    reasons.append(f"storage={storage_severity}:{storage_index:.3f}")
if pending_threshold > 0 and total_pending > pending_threshold:
    reasons.append(f"pending_lines={total_pending}>{pending_threshold}")
if oldest_threshold > 0 and oldest_age > oldest_threshold:
    reasons.append(f"pending_age={oldest_age:.1f}>{oldest_threshold:.1f}")
if bool(ingestion.get("overload")):
    reasons.append("ingestion_overload=true")

pressure_age = age_seconds(pressure)
pressure_tier = lower(pressure.get("tier") or pressure.get("overall_status"))
if pressure_age > 900:
    warnings.append(f"pressure_relief_stale:{int(pressure_age)}s")
if pressure_tier in {"survival", "critical", "protect"}:
    reasons.append(f"pressure_tier={pressure_tier}")

support = pressure.get("support_maintenance_stabilization") if isinstance(pressure.get("support_maintenance_stabilization"), dict) else {}
if bool(support.get("support_jobs_hot")):
    reasons.append("support_jobs_hot=true")
if bool(support.get("system_cotenant_hot")) and host >= 50:
    reasons.append("system_cotenant_hot=true")

allowed = not reasons
summary = "clear" if allowed else ";".join(reasons)
warning_text = ",".join(warnings)
print(json.dumps({
    "allowed": allowed,
    "summary": summary,
    "warnings": warning_text,
    "runtime_status": runtime_status,
    "throttle_profile": throttle_profile,
    "compute": compute,
    "memory": memory,
    "host_saturation_score": host,
    "mac_fluidity_band": fluidity_band,
    "storage_pressure_index": storage_index,
    "storage_total_pending_lines": total_pending,
    "storage_pending_threshold": pending_threshold,
    "oldest_pending_age_seconds": oldest_age,
    "oldest_age_threshold_seconds": oldest_threshold,
    "policy": "operator_heavy_livefeed_allowed_until_runtime_storage_or_mac_fluidity_guard_says_wait",
}, sort_keys=True))
PY
}

cleanup() {
  if [[ -n "${VIEWER_PID:-}" ]]; then
    kill "$VIEWER_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${HEAVY_PID:-}" ]]; then
    kill "$HEAVY_PID" >/dev/null 2>&1 || true
    sleep 1
    kill -TERM -- "-$HEAVY_PID" >/dev/null 2>&1 || true
  fi
  write_state "stopped" "false" "operator_closed" ""
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM HUP

guard="$(guard_status)"
if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "$guard"
  allowed="$("$PYTHON_BIN" -c 'import json,sys; print(str(json.loads(sys.stdin.read()).get("allowed", False)).lower())' <<< "$guard")"
  reason="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.stdin.read()).get("summary", ""))' <<< "$guard")"
  write_state "check" "$allowed" "$reason" ""
  [[ "$allowed" == "true" ]]
  exit $?
fi

while true; do
  guard="$(guard_status)"
  allowed="$("$PYTHON_BIN" -c 'import json,sys; print(str(json.loads(sys.stdin.read()).get("allowed", False)).lower())' <<< "$guard")"
  reason="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.stdin.read()).get("summary", ""))' <<< "$guard")"
  if [[ "$allowed" != "true" ]]; then
    echo "live_feed_heavy_guard_wait reason=$reason next_check_seconds=$WAIT_SECONDS"
    write_state "waiting" "false" "$reason" ""
    sleep "$WAIT_SECONDS"
    continue
  fi

  : > "$OUTPUT_LOG"
  echo "live_feed_heavy_guard_start source=$SOURCE ttl_seconds=$HEAVY_TTL_SECONDS check_interval_seconds=$CHECK_INTERVAL_SECONDS output_log=$OUTPUT_LOG"
  env \
    LIVE_FEED_DECISION_FILE_MODE=latest_only \
    LIVE_FEED_DECISION_MAX_AGE_HOURS=12 \
    LIVE_FEED_INCLUDE_COINBASE_WATCHDOG_LOG=0 \
    LIVE_FEED_INCLUDE_WATCHDOG_LOG_DEFAULT=0 \
    LIVE_FEED_HEAVY_TTL_ENABLED=1 \
    LIVE_FEED_HEAVY_TTL_SECONDS="$HEAVY_TTL_SECONDS" \
    LIVE_FEED_KEEPALIVE_SECONDS=20 \
  "$PROJECT_ROOT/scripts/ops/live_feed_tail.sh" \
      --source "$SOURCE" \
      --lines "$LINES" \
      --heavy \
      --memory-aware \
      --important-only \
      "${COLOR_ARGS[@]}" \
      --heavy-ttl-seconds "$HEAVY_TTL_SECONDS" \
      > "$OUTPUT_LOG" 2>&1 &
  HEAVY_PID=$!
  HEAVY_STARTED_EPOCH="$(date +%s)"
  renice -n 16 -p "$HEAVY_PID" >/dev/null 2>&1 || true
  write_state "running" "true" "clear" "$HEAVY_PID"

  tail -n "$TAIL_LINES" -F "$OUTPUT_LOG" &
  VIEWER_PID=$!

  while kill -0 "$HEAVY_PID" >/dev/null 2>&1; do
    sleep "$CHECK_INTERVAL_SECONDS"
    if [[ "$HEAVY_TTL_SECONDS" -gt 0 ]]; then
      now_epoch="$(date +%s)"
      if (( now_epoch - HEAVY_STARTED_EPOCH >= HEAVY_TTL_SECONDS )); then
        echo "live_feed_heavy_guard_cycle ttl_seconds=$HEAVY_TTL_SECONDS action=refreshing_heavy"
        write_state "cycling" "true" "ttl_refresh" "$HEAVY_PID"
        kill "$HEAVY_PID" >/dev/null 2>&1 || true
        sleep 1
        kill -TERM -- "-$HEAVY_PID" >/dev/null 2>&1 || true
        break
      fi
    fi
    guard="$(guard_status)"
    allowed="$("$PYTHON_BIN" -c 'import json,sys; print(str(json.loads(sys.stdin.read()).get("allowed", False)).lower())' <<< "$guard")"
    reason="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.stdin.read()).get("summary", ""))' <<< "$guard")"
    if [[ "$allowed" != "true" ]]; then
      echo "live_feed_heavy_guard_wait reason=$reason action=stopping_heavy next_check_seconds=$WAIT_SECONDS"
      write_state "waiting" "false" "$reason" "$HEAVY_PID"
      kill "$HEAVY_PID" >/dev/null 2>&1 || true
      sleep 1
      kill -TERM -- "-$HEAVY_PID" >/dev/null 2>&1 || true
      break
    fi
    write_state "running" "true" "clear" "$HEAVY_PID"
  done

  if [[ -n "${VIEWER_PID:-}" ]]; then
    kill "$VIEWER_PID" >/dev/null 2>&1 || true
    VIEWER_PID=""
  fi
  wait "$HEAVY_PID" >/dev/null 2>&1 || true
  HEAVY_PID=""

  if [[ "$RUN_ONCE" == "1" ]]; then
    write_state "stopped" "true" "once_completed" ""
    exit 0
  fi
  sleep "$WAIT_SECONDS"
done
