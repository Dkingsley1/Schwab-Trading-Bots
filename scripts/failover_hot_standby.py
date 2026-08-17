import argparse
import glob
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python

os.environ.setdefault("BOT_RUNTIME_LANE", os.getenv("BOT_SHADOW_RUNTIME_LANE", "canary314"))
RUNTIME_PY = resolve_runtime_python(PROJECT_ROOT)
LOG_PATH = PROJECT_ROOT / 'governance' / 'watchdog' / 'failover_events.jsonl'
FALLBACK_LOG_PATH = Path('/tmp/failover_events.jsonl')
DEFAULT_HEARTBEAT_GLOB = str(PROJECT_ROOT / 'governance' / 'health' / 'shadow_loop_conservative_equities_schwab_*.json')
SWAP_OVERRIDE_PATH = PROJECT_ROOT / 'config' / '.env.swap_pressure_override'
MEMORY_OVERRIDE_PATH = PROJECT_ROOT / 'config' / '.env.memory_efficiency_override'



def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()



def _append(row: dict) -> str:
    encoded = json.dumps(row, ensure_ascii=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open('a', encoding='utf-8') as f:
            f.write(encoded + '\n')
        return str(LOG_PATH)
    except Exception:
        FALLBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FALLBACK_LOG_PATH.open('a', encoding='utf-8') as f:
            f.write(encoded + '\n')
        return str(FALLBACK_LOG_PATH)


def _parse_env_override_file(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding='utf-8')
    except Exception:
        return {}

    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, raw_value = line.split('=', 1)
        key = key.strip()
        if not key:
            continue
        try:
            parsed = shlex.split(raw_value.strip(), comments=False, posix=True)
            value = parsed[0] if parsed else ''
        except Exception:
            value = raw_value.strip().strip('"').strip("'")
        out[key] = value
    return out


def _swap_research_pause_state() -> dict:
    env = dict(os.environ)
    env.update(_parse_env_override_file(MEMORY_OVERRIDE_PATH))
    env.update(_parse_env_override_file(SWAP_OVERRIDE_PATH))
    tier = str(env.get('SWAP_PRESSURE_TIER', '')).strip().lower()
    active = (
        str(env.get('SWAP_PRESSURE_HEAVY_RESEARCH_PAUSED', '0')).strip() == '1'
        or str(env.get('TRAINING_RUNTIME_PAUSED_FOR_SWAP', '0')).strip() == '1'
        or str(env.get('AUTO_RETRAIN_PAUSED_FOR_SWAP', '0')).strip() == '1'
        or tier in {'pause_research', 'survival'}
    )
    return {
        'active': bool(active),
        'tier': tier,
        'swap_used_gb': str(env.get('SWAP_PRESSURE_SWAP_USED_GB', '')).strip(),
        'source': str(SWAP_OVERRIDE_PATH),
    }



def _proc_alive(match: str, exclude_matches: tuple[str, ...] = ()) -> bool:
    p = subprocess.run(['ps', '-ax', '-o', 'command='], capture_output=True, text=True, check=False)
    excludes = tuple(item for item in exclude_matches if item)
    return any(
        match in line and not any(ex in line for ex in excludes)
        for line in (p.stdout or '').splitlines()
    )


def _default_standby_cmd() -> str:
    return f"{PROJECT_ROOT}/scripts/ops/opsctl.sh feed-refresh --source schwab --paper"


def _simulate_disallowed(cmd: str, allow_simulate: bool) -> bool:
    return (not allow_simulate) and "--simulate" in shlex.split(cmd)



def _heartbeat_age_sec(path_or_glob: str) -> float:
    raw = (path_or_glob or '').strip()
    if not raw:
        return 1e9

    if any(ch in raw for ch in '*?[]'):
        newest = 0.0
        found = False
        for fp in glob.glob(raw):
            try:
                ts = Path(fp).stat().st_mtime
            except Exception:
                continue
            if (not found) or ts > newest:
                newest = ts
                found = True
        if not found:
            return 1e9
        return max(time.time() - newest, 0.0)

    path = Path(raw)
    if not path.exists():
        return 1e9
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        ts = str(payload.get('timestamp_utc', '')).replace('Z', '+00:00')
        dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
        return max((datetime.now(timezone.utc) - dt).total_seconds(), 0.0)
    except Exception:
        return max(time.time() - path.stat().st_mtime, 0.0)



def _start_cmd(cmd: str) -> bool:
    try:
        subprocess.Popen(shlex.split(cmd), cwd=str(PROJECT_ROOT))
        return True
    except Exception:
        return False


def _build_failover_event(
    *,
    primary_alive: bool,
    live_parent_alive: bool,
    heartbeat_age_sec: float,
    max_heartbeat_age_sec: float,
    swap_pause: dict,
    standby_cmd: str,
    allow_simulate: bool,
    start_cmd=_start_cmd,
) -> dict:
    stale = heartbeat_age_sec > max_heartbeat_age_sec
    event = {
        'timestamp_utc': _now_iso(),
        'primary_alive': bool(primary_alive),
        'live_parent_alive': bool(live_parent_alive),
        'heartbeat_age_sec': heartbeat_age_sec,
        'stale': bool(stale),
        'action': 'none',
        'swap_pause_active': bool(swap_pause.get('active', False)),
        'simulate_standby_allowed': bool(allow_simulate),
    }

    if live_parent_alive:
        if stale:
            event['action'] = 'live_parent_active_primary_stale'
            event['standby_ok'] = False
            event['standby_cmd'] = standby_cmd
            event['standby_skip_reason'] = 'live_parent_alive'
        elif not primary_alive:
            event['action'] = 'live_parent_active_primary_missing'
            event['standby_ok'] = False
            event['standby_cmd'] = standby_cmd
            event['standby_skip_reason'] = 'live_parent_alive'
        else:
            event['action'] = 'live_parent_healthy'
        return event

    if (not primary_alive) or stale:
        if swap_pause.get('active', False):
            event['action'] = 'standby_start_skipped_swap_pause'
            event['standby_ok'] = False
            event['standby_cmd'] = standby_cmd
            event['swap_pause'] = swap_pause
        elif _simulate_disallowed(standby_cmd, bool(allow_simulate)):
            event['action'] = 'standby_start_skipped_simulate_disallowed'
            event['standby_ok'] = False
            event['standby_cmd'] = standby_cmd
        else:
            ok = start_cmd(standby_cmd)
            event['action'] = 'standby_start_attempt'
            event['standby_ok'] = ok
            event['standby_cmd'] = standby_cmd
    return event


def _event_signature(event: dict) -> tuple:
    return (
        event.get('action'),
        bool(event.get('primary_alive', False)),
        bool(event.get('live_parent_alive', False)),
        bool(event.get('stale', False)),
        bool(event.get('swap_pause_active', False)),
        event.get('standby_ok'),
        event.get('standby_skip_reason', ''),
    )



def main() -> int:
    parser = argparse.ArgumentParser(description='Hot-standby failover monitor for shadow runtime.')
    parser.add_argument('--primary-match', default='scripts/run_parallel_shadows.py')
    parser.add_argument('--primary-heartbeat', default=DEFAULT_HEARTBEAT_GLOB)
    parser.add_argument('--max-heartbeat-age-sec', type=float, default=150.0)
    parser.add_argument('--standby-start-cmd', default='')
    parser.add_argument(
        '--live-parent-match',
        default=os.getenv('FAILOVER_LIVE_PARENT_MATCH', 'scripts/run_all_sleeves.py'),
        help='A live-data parent process that should suppress fallback standby starts while healthy.',
    )
    parser.add_argument(
        '--allow-simulate-standby',
        action='store_true',
        default=os.getenv('FAILOVER_ALLOW_SIMULATE_STANDBY', '0').strip().lower() in {'1', 'true', 'yes', 'on'},
        help='Permit --simulate standby commands. Default is off so failover preserves live-data sleeves.',
    )
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval-seconds', type=int, default=20)
    parser.add_argument(
        '--log-unchanged-every-seconds',
        type=int,
        default=int(os.getenv('FAILOVER_LOG_UNCHANGED_EVERY_SECONDS', '300')),
        help='Debounce repeated unchanged failover states while still logging transitions.',
    )
    args = parser.parse_args()

    standby_cmd = args.standby_start_cmd.strip() or _default_standby_cmd()
    last_logged_signature: tuple | None = None
    last_logged_ts = 0.0

    while True:
        primary_excludes = () if args.allow_simulate_standby else ('--simulate',)
        alive = _proc_alive(args.primary_match, primary_excludes)
        live_parent_alive = _proc_alive(args.live_parent_match, ('--simulate',)) if args.live_parent_match else False
        hb_age = _heartbeat_age_sec(args.primary_heartbeat)
        swap_pause = _swap_research_pause_state()
        event = _build_failover_event(
            primary_alive=alive,
            live_parent_alive=live_parent_alive,
            heartbeat_age_sec=hb_age,
            max_heartbeat_age_sec=args.max_heartbeat_age_sec,
            swap_pause=swap_pause,
            standby_cmd=standby_cmd,
            allow_simulate=bool(args.allow_simulate_standby),
        )

        signature = _event_signature(event)
        now_ts = time.time()
        log_unchanged_every = max(int(args.log_unchanged_every_seconds), 0)
        should_log = (
            args.once
            or signature != last_logged_signature
            or log_unchanged_every == 0
            or (now_ts - last_logged_ts) >= log_unchanged_every
        )
        if should_log:
            log_file = _append(event)
            event['log_file'] = log_file
            print(json.dumps(event, ensure_ascii=True))
            last_logged_signature = signature
            last_logged_ts = now_ts

        if args.once:
            return 0
        time.sleep(max(args.interval_seconds, 5))


if __name__ == '__main__':
    raise SystemExit(main())
