import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = PROJECT_ROOT / '.venv312' / 'bin' / 'python'
DEFAULT_STATE_PATH = PROJECT_ROOT / 'governance' / 'health' / 'process_watchdog_state.json'
DEFAULT_OUT_PATH = PROJECT_ROOT / 'governance' / 'health' / 'process_watchdog_latest.json'
FALLBACK_STATE_PATH = Path('/tmp/process_watchdog_state.json')
FALLBACK_OUT_PATH = Path('/tmp/process_watchdog_latest.json')
DEFAULT_STORAGE_MOUNT_GUARD_PATH = PROJECT_ROOT / 'governance' / 'health' / 'storage_mount_guard_latest.json'
FALLBACK_STORAGE_MOUNT_GUARD_PATH = Path('/tmp/storage_mount_guard_latest.json')
SNAPSHOT_SCRIPT = PROJECT_ROOT / 'scripts' / 'collect_debug_snapshot.sh'
ALERT_ROUTER = PROJECT_ROOT / 'scripts' / 'pager_alert_router.py'


def _env_flag(name: str, default: str = '0') -> bool:
    return os.getenv(name, default).strip().lower() in {'1', 'true', 'yes', 'on'}


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in (raw or '').split(',') if x.strip()]


def _proc_running(pattern: str) -> int:
    p = subprocess.run(['ps', '-axo', 'command'], capture_output=True, text=True, check=False)
    out = p.stdout or ''
    return sum(1 for line in out.splitlines() if pattern in line)


def _spawn(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, 'a', encoding='utf-8')
    p = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    return int(p.pid)


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout or '').strip(), (p.stderr or '').strip()


def _latest_heartbeat_age_seconds(glob_pattern: str) -> float:
    if not glob_pattern:
        return 1e12
    newest: float = 0.0
    matched = False
    for fp in glob.glob(glob_pattern):
        try:
            ts = Path(fp).stat().st_mtime
            if (not matched) or ts > newest:
                newest = ts
                matched = True
        except Exception:
            continue
    if not matched:
        return 1e12
    return max(time.time() - newest, 0.0)


def _load_state(path: Path, fallback: Path) -> Dict[str, Any]:
    for candidate in (path, fallback):
        try:
            return json.loads(candidate.read_text(encoding='utf-8'))
        except Exception:
            continue
    return {'events': []}


def _save_state(path: Path, fallback: Path, state: Dict[str, Any]) -> Path:
    encoded = json.dumps(state, ensure_ascii=True, indent=2)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding='utf-8')
        return path
    except Exception:
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(encoded, encoding='utf-8')
        return fallback


def _write_payload(path: Path, fallback: Path, payload: Dict[str, Any]) -> Path:
    encoded = json.dumps(payload, ensure_ascii=True, indent=2)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding='utf-8')
        return path
    except Exception:
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(encoded, encoding='utf-8')
        return fallback


def _within_budget(events: List[Dict[str, Any]], name: str, max_per_hour: int) -> bool:
    cutoff = time.time() - 3600
    recent = [e for e in events if e.get('name') == name and float(e.get('ts_epoch', 0)) >= cutoff]
    return len(recent) < max(max_per_hour, 1)


def _file_age_seconds(path: Path) -> float:
    try:
        return max(time.time() - path.stat().st_mtime, 0.0)
    except Exception:
        return 1e12


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _refresh_runtime_reports(max_age_seconds: int) -> Dict[str, Any]:
    day = datetime.now(timezone.utc).strftime('%Y%m%d')
    one_numbers = PROJECT_ROOT / 'exports' / 'one_numbers' / 'one_numbers_summary.json'
    one_numbers_health = PROJECT_ROOT / 'governance' / 'health' / 'one_numbers_latest.json'
    daily_summary = PROJECT_ROOT / 'exports' / 'sql_reports' / f'daily_runtime_summary_{day}.json'
    daily_summary_health = PROJECT_ROOT / 'governance' / 'health' / 'daily_runtime_summary_latest.json'

    out: Dict[str, Any] = {
        'one_numbers': {
            'path': str(one_numbers),
            'refreshed': False,
            'age_seconds_before': round(_file_age_seconds(one_numbers), 2),
            'rc': 0,
            'error': '',
            'synced_health': False,
        },
        'daily_runtime_summary': {
            'path': str(daily_summary),
            'refreshed': False,
            'age_seconds_before': round(_file_age_seconds(daily_summary), 2),
            'rc': 0,
            'error': '',
            'synced_health': False,
        },
    }

    if _file_age_seconds(one_numbers) > float(max_age_seconds):
        if _proc_running('scripts/build_one_numbers_report.py') > 0:
            out['one_numbers']['error'] = 'refresh_already_running'
        else:
            rc, _stdout, err = _run([str(PY), str(PROJECT_ROOT / 'scripts' / 'build_one_numbers_report.py'), '--day', day])
            out['one_numbers']['refreshed'] = rc == 0
            out['one_numbers']['rc'] = int(rc)
            out['one_numbers']['error'] = err[-500:] if err else ''

    out['one_numbers']['synced_health'] = _copy_if_exists(one_numbers, one_numbers_health)
    out['one_numbers']['age_seconds_after'] = round(_file_age_seconds(one_numbers), 2)

    if _file_age_seconds(daily_summary) > float(max_age_seconds):
        rc, stdout, err = _run([str(PY), str(PROJECT_ROOT / 'scripts' / 'daily_runtime_summary.py'), '--day', day, '--json'])
        out['daily_runtime_summary']['refreshed'] = rc == 0
        out['daily_runtime_summary']['rc'] = int(rc)
        out['daily_runtime_summary']['error'] = err[-500:] if err else ''
        if rc == 0 and stdout:
            try:
                payload = json.loads(stdout)
                daily_summary.parent.mkdir(parents=True, exist_ok=True)
                daily_summary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
            except Exception as exc:
                out['daily_runtime_summary']['error'] = f'parse_or_write_failed:{exc}'

    out['daily_runtime_summary']['synced_health'] = _copy_if_exists(daily_summary, daily_summary_health)
    out['daily_runtime_summary']['age_seconds_after'] = round(_file_age_seconds(daily_summary), 2)
    return out


def _all_sleeves_start_ready(broker: str, simulate: bool) -> Tuple[bool, str]:
    missing = []
    if not _split_csv(os.getenv('SHADOW_SYMBOLS_CORE', '')):
        missing.append('SHADOW_SYMBOLS_CORE')
    if not _split_csv(os.getenv('SHADOW_SYMBOLS_VOLATILE', '')):
        missing.append('SHADOW_SYMBOLS_VOLATILE')
    if not _split_csv(os.getenv('SHADOW_SYMBOLS_DEFENSIVE', '')) and not _split_csv(os.getenv('SHADOW_SYMBOLS_COMMOD_FX_INTL', '')):
        missing.append('SHADOW_SYMBOLS_DEFENSIVE_or_SHADOW_SYMBOLS_COMMOD_FX_INTL')
    if missing:
        return False, 'missing_symbol_env:' + ','.join(missing)

    if broker == 'schwab' and (not simulate):
        key = os.getenv('SCHWAB_API_KEY', '').strip()
        secret = os.getenv('SCHWAB_SECRET', '').strip()
        invalid = {'', 'YOUR_KEY_HERE', 'YOUR_SECRET_HERE', 'YOUR_REAL_KEY', 'YOUR_REAL_SECRET', '<real_key>', '<real_secret>'}
        if key in invalid or secret in invalid:
            return False, 'missing_schwab_credentials'

    return True, 'ready'


def _build_all_sleeves_target(heartbeat_max_age_seconds: int) -> Dict[str, Any]:
    broker = os.getenv('DATA_BROKER', 'schwab').strip().lower()
    if broker not in {'schwab', 'coinbase'}:
        broker = 'schwab'

    simulate = _env_flag('OPS_WATCHDOG_ALL_SLEEVES_SIMULATE', '0')
    with_aggressive = _env_flag('OPS_WATCHDOG_ALL_SLEEVES_WITH_AGGRESSIVE', '1')

    cmd: List[str] = [str(PY), str(PROJECT_ROOT / 'scripts' / 'run_all_sleeves.py')]
    if with_aggressive:
        cmd.append('--with-aggressive-modes')
    cmd.extend(['--broker', broker])
    if simulate:
        cmd.append('--simulate')

    arg_env = [
        ('--symbols-core', 'SHADOW_SYMBOLS_CORE'),
        ('--symbols-volatile', 'SHADOW_SYMBOLS_VOLATILE'),
        ('--symbols-defensive', 'SHADOW_SYMBOLS_DEFENSIVE'),
        ('--dividend-symbols', 'DIVIDEND_SYMBOLS'),
        ('--bond-symbols', 'BOND_SYMBOLS'),
    ]
    for arg, env_name in arg_env:
        val = os.getenv(env_name, '').strip()
        if val:
            cmd.extend([arg, val])

    return {
        'name': 'all_sleeves',
        'pattern': 'scripts/run_all_sleeves.py',
        'alt_patterns': [
            'scripts/run_parallel_shadows.py',
            'scripts/run_dividend_shadow.py',
            'scripts/run_bond_shadow.py',
            'scripts/run_parallel_aggressive_modes.py',
        ],
        'cmd': cmd,
        'log': PROJECT_ROOT / 'logs' / 'watchdog_all_sleeves.log',
        'broker': broker,
        'simulate': simulate,
        'heartbeat_glob': str(PROJECT_ROOT / 'governance' / 'health' / 'shadow_loop_*_equities_schwab_*.json'),
        'heartbeat_max_age_seconds': max(int(heartbeat_max_age_seconds), 60),
    }


def _probe_host(hostport: str, timeout_seconds: float) -> Dict[str, Any]:
    raw = (hostport or '').strip()
    if not raw:
        return {'hostport': '', 'ok': False, 'error': 'empty'}

    host = raw
    port = 443
    if ':' in raw:
        h, p = raw.rsplit(':', 1)
        host = h.strip()
        try:
            port = int(p.strip())
        except Exception:
            port = 443
    if not host:
        return {'hostport': raw, 'ok': False, 'error': 'empty_host'}

    try:
        with socket.create_connection((host, port), timeout=max(float(timeout_seconds), 0.2)):
            return {'hostport': f'{host}:{port}', 'ok': True}
    except Exception as exc:
        return {'hostport': f'{host}:{port}', 'ok': False, 'error': f'{type(exc).__name__}:{exc}'}


def _storage_failback_sync_cmd() -> List[str]:
    return [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'storage_failback_sync.py'), '--json']


def _resolve_external_storage_paths() -> Tuple[Path, Path]:
    mount_root = Path(os.getenv('BOT_LOGS_EXTERNAL_MOUNT', '/Volumes/BOT_LOGS')).expanduser()
    configured_root = os.getenv('BOT_LOGS_EXTERNAL_PROJECT_ROOT', '').strip()
    if configured_root:
        return mount_root, Path(configured_root).expanduser()

    project_dir = os.getenv('BOT_LOGS_EXTERNAL_PROJECT_DIR', 'schwab_trading_bot').strip() or 'schwab_trading_bot'
    return mount_root, (mount_root / project_dir)


def _probe_storage_mount() -> Dict[str, Any]:
    mount_root, external_root = _resolve_external_storage_paths()
    mount_present = bool(mount_root.exists() and mount_root.is_dir())
    external_root_exists = bool(external_root.exists() and external_root.is_dir())
    external_root_writable = bool(external_root_exists and os.access(external_root, os.W_OK))
    external_available = bool(mount_present and external_root_writable)
    return {
        'mount_root': str(mount_root),
        'external_root': str(external_root),
        'mount_present': mount_present,
        'external_root_exists': external_root_exists,
        'external_root_writable': external_root_writable,
        'external_available': external_available,
    }


def _evaluate_storage_mount_transition(previous_mount_present: Any, mount_present_now: bool) -> Dict[str, Any]:
    if previous_mount_present is None:
        if mount_present_now:
            return {}
        return {'from': 'unknown', 'to': False}

    prev = bool(previous_mount_present)
    if prev == mount_present_now:
        return {}
    return {'from': prev, 'to': mount_present_now}


def _kickstart_labels(labels: List[str]) -> List[Dict[str, Any]]:
    uid = os.getuid()
    actions: List[Dict[str, Any]] = []
    for label in labels:
        full = f'gui/{uid}/{label}'
        rc, out, err = _run(['launchctl', 'kickstart', '-k', full])
        actions.append({'label': label, 'rc': int(rc), 'stdout': out[-200:], 'stderr': err[-200:]})
    return actions


def _alert(severity: str, event: str, message: str, suppress_seconds: int = 600, force: bool = False) -> Dict[str, Any]:
    if not ALERT_ROUTER.exists() or not PY.exists():
        return {'attempted': False, 'reason': 'alert_router_missing'}

    cmd = [
        str(PY),
        str(ALERT_ROUTER),
        '--severity',
        severity,
        '--event',
        event,
        '--message',
        message,
        '--suppress-seconds',
        str(max(int(suppress_seconds), 0)),
    ]
    if force:
        cmd.append('--force')
    rc, out, err = _run(cmd)
    return {
        'attempted': True,
        'rc': int(rc),
        'stdout': out[-500:],
        'stderr': err[-500:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Watchdog: restart key loops with bounded backoff.')
    parser.add_argument('--max-restarts-per-hour', type=int, default=int(os.getenv('OPS_WATCHDOG_MAX_RESTARTS_PER_HOUR', '6')))
    parser.add_argument('--require-all-sleeves', action='store_true', default=os.getenv('OPS_WATCHDOG_REQUIRE_ALL_SLEEVES', '1') == '1')
    parser.add_argument('--require-coinbase', action='store_true', default=os.getenv('OPS_WATCHDOG_REQUIRE_COINBASE', '1') == '1')
    parser.add_argument('--refresh-reports', action='store_true', default=os.getenv('OPS_WATCHDOG_REFRESH_REPORTS', '1') == '1')
    parser.add_argument('--refresh-max-age-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_REFRESH_MAX_AGE_SECONDS', '7200')))
    parser.add_argument('--all-sleeves-heartbeat-stale-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_ALL_SLEEVES_HEARTBEAT_STALE_SECONDS', '360')))
    parser.add_argument('--coinbase-heartbeat-stale-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_COINBASE_HEARTBEAT_STALE_SECONDS', '420')))
    parser.add_argument('--network-guard', action='store_true', default=_env_flag('OPS_WATCHDOG_NETWORK_GUARD', '1'))
    parser.add_argument('--network-hosts', default=os.getenv('OPS_WATCHDOG_NETWORK_HOSTS', 'api.schwabapi.com:443,api.exchange.coinbase.com:443'))
    parser.add_argument('--network-timeout-seconds', type=float, default=float(os.getenv('OPS_WATCHDOG_NETWORK_TIMEOUT_SECONDS', '2.5')))
    parser.add_argument('--network-fail-threshold', type=int, default=int(os.getenv('OPS_WATCHDOG_NETWORK_FAIL_THRESHOLD', '3')))
    parser.add_argument('--restart-storm-threshold', type=int, default=int(os.getenv('OPS_WATCHDOG_RESTART_STORM_THRESHOLD', '4')))
    parser.add_argument('--restart-storm-window-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_RESTART_STORM_WINDOW_SECONDS', '3600')))
    parser.add_argument('--alert-suppress-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_ALERT_SUPPRESS_SECONDS', '600')))
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    state = _load_state(DEFAULT_STATE_PATH, FALLBACK_STATE_PATH)
    events = state.get('events') if isinstance(state.get('events'), list) else []

    maintenance: List[Dict[str, Any]] = []
    storage_mode = str(state.get('storage_mode', '') or '')
    storage_mode_transition: Dict[str, Any] = {}
    storage_mount_prev_raw = state.get('storage_mount_present', None)
    storage_mount_prev = None if storage_mount_prev_raw is None else bool(storage_mount_prev_raw)
    storage_mount_transition: Dict[str, Any] = {}
    storage_mount_guard: Dict[str, Any] = {}

    for name, cmd in [
        ('lock_watchdog', [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'lock_watchdog.py'), '--apply', '--json']),
        ('storage_failback_sync', _storage_failback_sync_cmd()),
        ('canary_auto_tuner', [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'canary_auto_tuner.py'), '--json']),
    ]:
        rc, out, err = _run(cmd)
        row: Dict[str, Any] = {
            'name': name,
            'ok': rc == 0,
            'rc': int(rc),
            'stdout_tail': '\n'.join((out or '').splitlines()[-6:]),
            'stderr_tail': '\n'.join((err or '').splitlines()[-6:]),
        }

        if name == 'storage_failback_sync':
            try:
                payload = json.loads(out) if out else {}
            except Exception:
                payload = {}
            row['payload'] = payload
            mode_now = str(payload.get('mode', '') or '')
            if mode_now and mode_now != storage_mode:
                storage_mode_transition = {
                    'from': storage_mode or 'unknown',
                    'to': mode_now,
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                }
                if mode_now == 'local_fallback':
                    storage_mode_transition['alert'] = _alert(
                        'warn',
                        'storage_fallback_activated',
                        'External BOT_LOGS unavailable. Switched to local fallback storage.',
                        suppress_seconds=max(args.alert_suppress_seconds, 60),
                    )
                elif storage_mode == 'local_fallback' and mode_now == 'external':
                    storage_mode_transition['alert'] = _alert(
                        'info',
                        'storage_external_restored',
                        'External BOT_LOGS restored. Storage routing back on external root.',
                        suppress_seconds=max(args.alert_suppress_seconds, 60),
                    )
                storage_mode = mode_now
            elif mode_now:
                storage_mode = mode_now

        maintenance.append(row)

    storage_mount_guard = _probe_storage_mount()
    storage_mount_present = bool(storage_mount_guard.get('mount_present', False))
    storage_mount_guard['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    storage_mount_guard['storage_mode'] = storage_mode or 'unknown'
    storage_mount_guard['previous_mount_present'] = storage_mount_prev

    mount_transition_base = _evaluate_storage_mount_transition(storage_mount_prev, storage_mount_present)
    if mount_transition_base:
        storage_mount_transition = {
            **mount_transition_base,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'mount_root': storage_mount_guard.get('mount_root'),
            'external_root': storage_mount_guard.get('external_root'),
        }

        if not storage_mount_present:
            storage_mount_transition['alert'] = _alert(
                'critical',
                'storage_external_mount_missing',
                f"External BOT_LOGS mount missing at {storage_mount_guard.get('mount_root')}.",
                suppress_seconds=max(args.alert_suppress_seconds, 60),
            )
        else:
            recovery: Dict[str, Any] = {'attempted': False}
            if storage_mode != 'external':
                rc, out, err = _run(_storage_failback_sync_cmd())
                try:
                    recovery_payload = json.loads(out) if out else {}
                except Exception:
                    recovery_payload = {}
                recovery = {
                    'attempted': True,
                    'ok': rc == 0,
                    'rc': int(rc),
                    'payload': recovery_payload,
                    'stdout_tail': '\n'.join((out or '').splitlines()[-6:]),
                    'stderr_tail': '\n'.join((err or '').splitlines()[-6:]),
                }
                maintenance.append(
                    {
                        'name': 'storage_failback_sync_recovery',
                        'ok': rc == 0,
                        'rc': int(rc),
                        'payload': recovery_payload,
                        'stdout_tail': recovery['stdout_tail'],
                        'stderr_tail': recovery['stderr_tail'],
                    }
                )
                mode_now = str(recovery_payload.get('mode', '') or '')
                if mode_now:
                    storage_mode = mode_now
            storage_mount_transition['recovery'] = recovery
            if not storage_mode_transition:
                storage_mount_transition['alert'] = _alert(
                    'info',
                    'storage_external_mount_restored',
                    f"External BOT_LOGS mount restored at {storage_mount_guard.get('mount_root')}.",
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                )

    refresh_payload: Dict[str, Any] = {}
    if args.refresh_reports:
        refresh_payload = _refresh_runtime_reports(max_age_seconds=max(int(args.refresh_max_age_seconds), 60))

    network_payload: Dict[str, Any] = {'enabled': bool(args.network_guard), 'results': []}
    network_outage_active = False
    network_fail_streak_prev = int(state.get('network_fail_streak', 0) or 0)
    network_fail_streak_now = network_fail_streak_prev

    if args.network_guard:
        hosts = _split_csv(args.network_hosts)
        results = [_probe_host(h, timeout_seconds=float(args.network_timeout_seconds)) for h in hosts]
        any_ok = any(r.get('ok') for r in results) if results else True

        if any_ok:
            network_fail_streak_now = 0
            if network_fail_streak_prev >= max(int(args.network_fail_threshold), 1):
                kickstarts = _kickstart_labels([
                    'com.dankingsley.shadow_watchdog',
                    'com.dankingsley.all_sleeves',
                ])
                recovered_alert = _alert(
                    'warn',
                    'network_recovered_restart',
                    f'Network recovered after fail_streak={network_fail_streak_prev}; kickstarted core services.',
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                )
                network_payload['recovery_actions'] = {
                    'kickstarts': kickstarts,
                    'alert': recovered_alert,
                }
        else:
            network_fail_streak_now = network_fail_streak_prev + 1
            if network_fail_streak_now == max(int(args.network_fail_threshold), 1):
                network_payload['degraded_alert'] = _alert(
                    'critical',
                    'network_outage_detected',
                    f'Network probe failed for {network_fail_streak_now} consecutive checks.',
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                )

        network_outage_active = network_fail_streak_now >= max(int(args.network_fail_threshold), 1) and (not any_ok)
        network_payload.update(
            {
                'hosts': hosts,
                'results': results,
                'any_ok': bool(any_ok),
                'fail_streak_prev': int(network_fail_streak_prev),
                'fail_streak_now': int(network_fail_streak_now),
                'outage_active': bool(network_outage_active),
            }
        )

    targets: List[Dict[str, Any]] = [
        {
            'name': 'sql_link_writer',
            'pattern': 'scripts/ops/sql_link_writer_service.py',
            'cmd': [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'sql_link_writer_service.py')],
            'log': PROJECT_ROOT / 'logs' / 'watchdog_sql_link_writer.log',
            'alt_patterns': [],
            'heartbeat_glob': '',
            'heartbeat_max_age_seconds': 0,
        },
    ]

    if args.require_all_sleeves:
        targets.append(_build_all_sleeves_target(heartbeat_max_age_seconds=args.all_sleeves_heartbeat_stale_seconds))

    if args.require_coinbase:
        coinbase_cmd: List[str] = [
            str(PY),
            str(PROJECT_ROOT / 'scripts' / 'run_shadow_training_loop.py'),
            '--broker',
            'coinbase',
            '--symbols',
            os.getenv('COINBASE_WATCH_SYMBOLS', 'BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD'),
            '--interval-seconds',
            os.getenv('COINBASE_WATCH_INTERVAL_SECONDS', '20'),
            '--max-iterations',
            '0',
        ]
        if _env_flag('OPS_WATCHDOG_COINBASE_SIMULATE', '0'):
            coinbase_cmd.append('--simulate')
        targets.append(
            {
                'name': 'coinbase_loop',
                'pattern': 'scripts/run_shadow_training_loop.py --broker coinbase',
                'cmd': coinbase_cmd,
                'log': PROJECT_ROOT / 'logs' / 'watchdog_coinbase_loop.log',
                'alt_patterns': [],
                'heartbeat_glob': str(PROJECT_ROOT / 'governance' / 'health' / 'shadow_loop_*_crypto_coinbase_*.json'),
                'heartbeat_max_age_seconds': max(int(args.coinbase_heartbeat_stale_seconds), 60),
            }
        )

    restarts: List[Dict[str, Any]] = []
    status: List[Dict[str, Any]] = []
    alerts: List[Dict[str, Any]] = []

    for t in targets:
        running = _proc_running(t['pattern'])
        alt_running = sum(_proc_running(p) for p in t.get('alt_patterns', []) if p)
        heartbeat_glob = str(t.get('heartbeat_glob', '') or '')
        heartbeat_required = bool(heartbeat_glob)
        heartbeat_age = _latest_heartbeat_age_seconds(heartbeat_glob) if heartbeat_required else 0.0
        heartbeat_max_age = float(t.get('heartbeat_max_age_seconds', 0) or 0.0)
        heartbeat_ok = (not heartbeat_required) or (heartbeat_age <= heartbeat_max_age)

        row: Dict[str, Any] = {
            'name': t['name'],
            'running': int(running),
            'heartbeat_ok': bool(heartbeat_ok),
        }
        if alt_running > 0:
            row['alt_running'] = int(alt_running)
        if heartbeat_required:
            row['heartbeat_age_seconds'] = round(float(heartbeat_age), 2)
            row['heartbeat_max_age_seconds'] = float(heartbeat_max_age)

        process_live = (running > 0) or (alt_running > 0)
        if process_live and heartbeat_ok:
            status.append(row)
            continue

        if t['name'] == 'all_sleeves':
            ready, reason = _all_sleeves_start_ready(str(t.get('broker', 'schwab')), bool(t.get('simulate', False)))
            if not ready:
                row['restart_skipped'] = 'startup_not_ready'
                row['reason'] = reason
                status.append(row)
                continue

        if network_outage_active and t['name'] in {'all_sleeves', 'coinbase_loop'}:
            row['restart_skipped'] = 'network_outage_active'
            status.append(row)
            continue

        if not _within_budget(events, t['name'], args.max_restarts_per_hour):
            row['restart_skipped'] = 'budget_exhausted'
            status.append(row)
            alerts.append(
                {
                    'name': t['name'],
                    'type': 'budget_exhausted',
                    'alert': _alert(
                        'critical',
                        'watchdog_restart_budget_exhausted',
                        f"Restart budget exhausted for {t['name']}.",
                        suppress_seconds=max(args.alert_suppress_seconds, 60),
                    ),
                }
            )
            continue

        pid = _spawn(t['cmd'], t['log'])
        ts = datetime.now(timezone.utc).isoformat()
        evt = {
            'name': t['name'],
            'event': 'restart',
            'pid': pid,
            'timestamp_utc': ts,
            'ts_epoch': time.time(),
            'reason': 'process_missing' if not process_live else 'heartbeat_stale',
        }
        events.append(evt)
        restarts.append(evt)
        row['restarted_pid'] = pid
        row['restart_reason'] = evt['reason']
        status.append(row)

    if SNAPSHOT_SCRIPT.exists() and restarts:
        subprocess.run([str(SNAPSHOT_SCRIPT)], cwd=str(PROJECT_ROOT), check=False)

    restart_window_seconds = max(int(args.restart_storm_window_seconds), 60)
    cutoff = time.time() - restart_window_seconds
    by_name: Dict[str, int] = {}
    for e in events:
        if e.get('event') != 'restart':
            continue
        if float(e.get('ts_epoch', 0)) < cutoff:
            continue
        name = str(e.get('name', 'unknown'))
        by_name[name] = by_name.get(name, 0) + 1

    restart_storms = [
        {'name': name, 'count': count, 'window_seconds': restart_window_seconds}
        for name, count in sorted(by_name.items())
        if count >= max(int(args.restart_storm_threshold), 1)
    ]
    for storm in restart_storms:
        alerts.append(
            {
                'name': storm['name'],
                'type': 'restart_storm',
                'count': storm['count'],
                'alert': _alert(
                    'critical',
                    'watchdog_restart_storm',
                    f"Restart storm: {storm['name']} restarted {storm['count']} times in {restart_window_seconds}s.",
                    suppress_seconds=max(args.alert_suppress_seconds, 60),
                ),
            }
        )

    events = sorted(events, key=lambda x: float(x.get('ts_epoch', 0)))[-800:]
    state = {
        'events': events,
        'network_fail_streak': int(network_fail_streak_now),
        'storage_mode': storage_mode,
        'storage_mount_present': bool(storage_mount_present),
        'storage_external_available': bool(storage_mount_guard.get('external_available', False)),
        'storage_mount_root': str(storage_mount_guard.get('mount_root', '')),
        'storage_external_root': str(storage_mount_guard.get('external_root', '')),
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    state_written = _save_state(DEFAULT_STATE_PATH, FALLBACK_STATE_PATH, state)

    storage_mount_guard_written = _write_payload(
        DEFAULT_STORAGE_MOUNT_GUARD_PATH,
        FALLBACK_STORAGE_MOUNT_GUARD_PATH,
        {
            **storage_mount_guard,
            'storage_mode_transition': storage_mode_transition,
            'storage_mount_transition': storage_mount_transition,
        },
    )

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'status': status,
        'restarts': restarts,
        'restart_storms': restart_storms,
        'alerts': alerts,
        'max_restarts_per_hour': int(args.max_restarts_per_hour),
        'maintenance': maintenance,
        'refresh_reports': refresh_payload,
        'network': network_payload,
        'storage_mode_transition': storage_mode_transition,
        'storage_mount_transition': storage_mount_transition,
        'storage_mount_guard': {
            **storage_mount_guard,
            'out_file': str(storage_mount_guard_written),
        },
        'state_file': str(state_written),
    }
    out_written = _write_payload(DEFAULT_OUT_PATH, FALLBACK_OUT_PATH, payload)
    payload['out_file'] = str(out_written)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"process_watchdog restarts={len(restarts)} out={out_written}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
