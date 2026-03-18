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
ks = cfg.get('global_killswitch', {})
ah = cfg.get('auto_heavy', {})
pg = cfg.get('promotion_gates', {})
pgp = pg.get('promotion_gate', {}) if isinstance(pg.get('promotion_gate', {}), dict) else {}
gg = pg.get('graduation_gate', {}) if isinstance(pg.get('graduation_gate', {}), dict) else {}

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
  'GLOBAL_KILL_BLOCKED_RATE_MAX': ks.get('blocked_rate_max', 0.45),
  'GLOBAL_KILL_ABS_PNL_PROXY_MAX': ks.get('abs_pnl_proxy_max', 0.03),
  'GLOBAL_KILL_STALE_WINDOWS_MAX': ks.get('stale_windows_max', 2),
  'GLOBAL_KILL_WATCHDOG_RESTARTS_MAX': ks.get('watchdog_restarts_max', 5),
  'AUTO_HEAVY_THRESHOLD_CPU': ah.get('threshold_cpu', 120),
  'AUTO_HEAVY_ON_STREAK_REQ': ah.get('on_streak_required', 3),
  'AUTO_HEAVY_OFF_STREAK_REQ': ah.get('off_streak_required', 8),
  'AUTO_HEAVY_MEMORY_PRESSURE_TRIP': ah.get('memory_pressure_trip', 1),
  'PROMOTION_GATE_MIN_FORWARD_MEAN': pgp.get('min_forward_mean', 0.53),
  'PROMOTION_GATE_MIN_DELTA': pgp.get('min_delta', -0.01),
  'PROMOTION_GATE_MIN_TRADING_QUALITY_SCORE': pgp.get('min_trading_quality_score', 0.52),
  'PROMOTION_GATE_MAX_OVERFIT_GAP': pgp.get('max_overfit_gap', 0.10),
  'PROMOTION_GATE_MAX_FAIL_SHARE': pgp.get('max_fail_share', 0.25),
  'PROMOTION_GATE_MAX_SEVERE_OVERFIT_SHARE': pgp.get('max_severe_overfit_share', 0.10),
  'PROMOTION_GATE_MIN_RUNS': pgp.get('min_runs_per_bot', 12),
  'PROMOTION_GATE_MIN_CONSIDERED': pgp.get('min_considered_bots', 4),
  'PROMOTION_GATE_REQUIRE_ACTIVE_REGISTRY': int(bool(pgp.get('require_active_registry', True))),
  'PROMOTION_GATE_INCLUDE_INFRASTRUCTURE': int(bool(pgp.get('include_infrastructure', False))),
  'GRADUATION_MIN_RUNS': gg.get('min_runs', 24),
  'GRADUATION_MIN_FORWARD_MEAN': gg.get('min_forward_mean', 0.52),
  'GRADUATION_MIN_DELTA': gg.get('min_delta', -0.02),
  'GRADUATION_MIN_MATURE_BOTS': gg.get('min_mature_bots', 10),
  'GRADUATION_MIN_MATURE_PASS_RATE': gg.get('min_mature_pass_rate', 0.30),
  'GRADUATION_MAX_IMMATURE_ACTIVE': gg.get('max_immature_active', 0),
}
for k, v in pairs.items():
  print(f"{k}={v}")
PY
)
