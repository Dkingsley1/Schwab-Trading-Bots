#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$PROJECT_ROOT/governance/ops_thresholds.json"

if [[ ! -f "$CFG" ]]; then
  exit 0
fi

while IFS= read -r line; do
  eval "export $line"
done < <(/usr/bin/python3 - "$CFG" <<'PY'
import json
import sys

cfg = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
rg = cfg.get('resource_guard', {})
hg = cfg.get('health_gates', {})
sm = cfg.get('safe_mode', {})
sr = cfg.get('session_ready', {})

pairs = {
  'RESOURCE_GUARD_MAX_LOAD_PER_CORE': rg.get('max_load_per_core', 1.8),
  'RESOURCE_GUARD_MIN_DISK_GB': rg.get('min_disk_gb', 20.0),
  'RESOURCE_GUARD_MIN_MEMORY_FREE_PCT': rg.get('min_memory_free_pct', 10.0),
  'RESOURCE_GUARD_MAX_EDITING_CPU': rg.get('max_editing_cpu', 180.0),
  'HEALTH_GATE_STALE_WINDOW_LIMIT': hg.get('stale_window_limit', 0),
  'HEALTH_GATE_BLOCKED_RATE_LIMIT': hg.get('blocked_rate_limit', 0.30),
  'HEALTH_GATE_WATCHDOG_RESTARTS_LIMIT': hg.get('watchdog_restarts_limit', 3),
  'SAFE_MODE_TRIP_STREAK_REQUIRED': sm.get('trip_streak_required', 3),
  'SAFE_MODE_CLEAR_STREAK_REQUIRED': sm.get('cooldown_clear_streak_required', 2),
  'SESSION_READY_MIN_DISK_GB': sr.get('min_disk_gb', 15.0),
  'SESSION_READY_HEARTBEAT_MAX_AGE_SEC': sr.get('heartbeat_max_age_sec', 300),
}
for k, v in pairs.items():
  print(f"{k}={v}")
PY
)
