import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = PROJECT_ROOT / '.venv312' / 'bin' / 'python'
STATE_PATH = PROJECT_ROOT / 'governance' / 'health' / 'process_watchdog_state.json'
OUT_PATH = PROJECT_ROOT / 'governance' / 'health' / 'process_watchdog_latest.json'
SNAPSHOT_SCRIPT = PROJECT_ROOT / 'scripts' / 'collect_debug_snapshot.sh'


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


def _load_state() -> Dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {'events': []}


def _save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding='utf-8')


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
        if key in {'', 'YOUR_KEY_HERE'} or secret in {'', 'YOUR_SECRET_HERE'}:
            return False, 'missing_schwab_credentials'

    return True, 'ready'


def _build_all_sleeves_target() -> Dict[str, Any]:
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Watchdog: restart key loops with bounded backoff.')
    parser.add_argument('--max-restarts-per-hour', type=int, default=int(os.getenv('OPS_WATCHDOG_MAX_RESTARTS_PER_HOUR', '6')))
    parser.add_argument('--require-all-sleeves', action='store_true', default=os.getenv('OPS_WATCHDOG_REQUIRE_ALL_SLEEVES', '1') == '1')
    parser.add_argument('--require-coinbase', action='store_true', default=os.getenv('OPS_WATCHDOG_REQUIRE_COINBASE', '1') == '1')
    parser.add_argument('--refresh-reports', action='store_true', default=os.getenv('OPS_WATCHDOG_REFRESH_REPORTS', '1') == '1')
    parser.add_argument('--refresh-max-age-seconds', type=int, default=int(os.getenv('OPS_WATCHDOG_REFRESH_MAX_AGE_SECONDS', '7200')))
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    state = _load_state()
    events = state.get('events') if isinstance(state.get('events'), list) else []

    maintenance: List[Dict[str, Any]] = []
    for name, cmd in [
        ('lock_watchdog', [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'lock_watchdog.py'), '--apply', '--json']),
        ('storage_failback_sync', [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'storage_failback_sync.py'), '--json']),
        ('canary_auto_tuner', [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'canary_auto_tuner.py'), '--json']),
    ]:
        rc, out, err = _run(cmd)
        maintenance.append(
            {
                'name': name,
                'ok': rc == 0,
                'rc': int(rc),
                'stdout_tail': '\n'.join((out or '').splitlines()[-6:]),
                'stderr_tail': '\n'.join((err or '').splitlines()[-6:]),
            }
        )

    refresh_payload: Dict[str, Any] = {}
    if args.refresh_reports:
        refresh_payload = _refresh_runtime_reports(max_age_seconds=max(int(args.refresh_max_age_seconds), 60))

    targets: List[Dict[str, Any]] = [
        {
            'name': 'sql_link_writer',
            'pattern': 'scripts/ops/sql_link_writer_service.py',
            'cmd': [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'sql_link_writer_service.py')],
            'log': PROJECT_ROOT / 'logs' / 'watchdog_sql_link_writer.log',
            'alt_patterns': [],
        },
    ]

    if args.require_all_sleeves:
        targets.append(_build_all_sleeves_target())

    if args.require_coinbase:
        coinbase_cmd: List[str] = [
            str(PY), str(PROJECT_ROOT / 'scripts' / 'run_shadow_training_loop.py'),
            '--broker', 'coinbase',
            '--symbols', os.getenv('COINBASE_WATCH_SYMBOLS', 'BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD'),
            '--interval-seconds', os.getenv('COINBASE_WATCH_INTERVAL_SECONDS', '20'),
            '--max-iterations', '0',
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
            }
        )

    restarts: List[Dict[str, Any]] = []
    status: List[Dict[str, Any]] = []

    for t in targets:
        running = _proc_running(t['pattern'])
        alt_running = sum(_proc_running(p) for p in t.get('alt_patterns', []) if p)
        row: Dict[str, Any] = {'name': t['name'], 'running': int(running)}
        if alt_running > 0:
            row['alt_running'] = int(alt_running)

        if running > 0 or alt_running > 0:
            status.append(row)
            continue

        if t['name'] == 'all_sleeves':
            ready, reason = _all_sleeves_start_ready(str(t.get('broker', 'schwab')), bool(t.get('simulate', False)))
            if not ready:
                row['restart_skipped'] = 'startup_not_ready'
                row['reason'] = reason
                status.append(row)
                continue

        if not _within_budget(events, t['name'], args.max_restarts_per_hour):
            row['restart_skipped'] = 'budget_exhausted'
            status.append(row)
            continue

        pid = _spawn(t['cmd'], t['log'])
        ts = datetime.now(timezone.utc).isoformat()
        evt = {'name': t['name'], 'event': 'restart', 'pid': pid, 'timestamp_utc': ts, 'ts_epoch': time.time()}
        events.append(evt)
        restarts.append(evt)
        row['restarted_pid'] = pid
        status.append(row)

    if SNAPSHOT_SCRIPT.exists() and restarts:
        subprocess.run([str(SNAPSHOT_SCRIPT)], cwd=str(PROJECT_ROOT), check=False)

    events = sorted(events, key=lambda x: float(x.get('ts_epoch', 0)))[-400:]
    state = {'events': events, 'updated_at_utc': datetime.now(timezone.utc).isoformat()}
    _save_state(state)

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'status': status,
        'restarts': restarts,
        'max_restarts_per_hour': int(args.max_restarts_per_hour),
        'maintenance': maintenance,
        'refresh_reports': refresh_payload,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"process_watchdog restarts={len(restarts)}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
