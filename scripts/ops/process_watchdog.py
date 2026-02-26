import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = PROJECT_ROOT / '.venv312' / 'bin' / 'python'
STATE_PATH = PROJECT_ROOT / 'governance' / 'health' / 'process_watchdog_state.json'
OUT_PATH = PROJECT_ROOT / 'governance' / 'health' / 'process_watchdog_latest.json'
SNAPSHOT_SCRIPT = PROJECT_ROOT / 'scripts' / 'collect_debug_snapshot.sh'


def _proc_running(pattern: str) -> int:
    p = subprocess.run(['ps', '-axo', 'command'], capture_output=True, text=True, check=False)
    out = p.stdout or ''
    return sum(1 for line in out.splitlines() if pattern in line)


def _spawn(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, 'a', encoding='utf-8')
    p = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    return int(p.pid)


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {'events': []}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding='utf-8')


def _within_budget(events: list[dict], name: str, max_per_hour: int) -> bool:
    cutoff = time.time() - 3600
    recent = [e for e in events if e.get('name') == name and float(e.get('ts_epoch', 0)) >= cutoff]
    return len(recent) < max(max_per_hour, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description='Watchdog: restart key loops with bounded backoff.')
    parser.add_argument('--max-restarts-per-hour', type=int, default=int(os.getenv('OPS_WATCHDOG_MAX_RESTARTS_PER_HOUR', '6')))
    parser.add_argument('--require-coinbase', action='store_true', default=os.getenv('OPS_WATCHDOG_REQUIRE_COINBASE', '1') == '1')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    state = _load_state()
    events = state.get('events') if isinstance(state.get('events'), list) else []

    targets = [
        {
            'name': 'all_sleeves',
            'pattern': 'scripts/run_all_sleeves.py --with-aggressive-modes',
            'cmd': [str(PY), str(PROJECT_ROOT / 'scripts' / 'run_all_sleeves.py'), '--with-aggressive-modes'],
            'log': PROJECT_ROOT / 'logs' / 'watchdog_all_sleeves.log',
        },
        {
            'name': 'sql_link_writer',
            'pattern': 'scripts/ops/sql_link_writer_service.py',
            'cmd': [str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'sql_link_writer_service.py')],
            'log': PROJECT_ROOT / 'logs' / 'watchdog_sql_link_writer.log',
        },
    ]

    if args.require_coinbase:
        targets.append(
            {
                'name': 'coinbase_loop',
                'pattern': 'scripts/run_shadow_training_loop.py --broker coinbase',
                'cmd': [
                    str(PY), str(PROJECT_ROOT / 'scripts' / 'run_shadow_training_loop.py'),
                    '--broker', 'coinbase',
                    '--symbols', os.getenv('COINBASE_WATCH_SYMBOLS', 'BTC-USD,ETH-USD,SOL-USD,AVAX-USD,LTC-USD,LINK-USD,DOGE-USD'),
                    '--interval-seconds', os.getenv('COINBASE_WATCH_INTERVAL_SECONDS', '20'),
                    '--max-iterations', '0',
                    '--simulate',
                ],
                'log': PROJECT_ROOT / 'logs' / 'watchdog_coinbase_loop.log',
            }
        )

    restarts = []
    status = []

    # Keep lock hygiene first.
    subprocess.run([str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'lock_watchdog.py'), '--apply', '--json'], cwd=str(PROJECT_ROOT), check=False)
    subprocess.run([str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'storage_failback_sync.py'), '--json'], cwd=str(PROJECT_ROOT), check=False)
    subprocess.run([str(PY), str(PROJECT_ROOT / 'scripts' / 'ops' / 'canary_auto_tuner.py'), '--json'], cwd=str(PROJECT_ROOT), check=False)

    for t in targets:
        running = _proc_running(t['pattern'])
        row = {'name': t['name'], 'running': int(running)}
        if running > 0:
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

    # keep last 400 events
    events = sorted(events, key=lambda x: float(x.get('ts_epoch', 0)))[-400:]
    state = {'events': events, 'updated_at_utc': datetime.now(timezone.utc).isoformat()}
    _save_state(state)

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'status': status,
        'restarts': restarts,
        'max_restarts_per_hour': int(args.max_restarts_per_hour),
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
