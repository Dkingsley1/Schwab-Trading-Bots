import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = PROJECT_ROOT / '.venv312' / 'bin' / 'python'
LINK_SCRIPT = PROJECT_ROOT / 'scripts' / 'link_jsonl_to_sql.py'


def _run_link(timeout_s: int, lock_retries: int, retry_delay_s: float) -> tuple[int, str, str]:
    cmd = [
        str(PY),
        str(LINK_SCRIPT),
        '--mode', 'sqlite',
        '--sqlite-timeout-seconds', str(max(timeout_s, 30)),
        '--sqlite-lock-retries', str(max(lock_retries, 0)),
        '--sqlite-lock-retry-delay-seconds', str(max(retry_delay_s, 0.1)),
    ]
    # Some launchd/watchdog contexts can leave stdin as an invalid descriptor.
    # Force a stable stdin handle so child Python does not fail at init_sys_streams.
    p = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return p.returncode, (p.stdout or '').strip(), (p.stderr or '').strip()


def main() -> int:
    parser = argparse.ArgumentParser(description='Single-writer SQLite linker service with lock arbitration.')
    parser.add_argument('--interval-seconds', type=int, default=int(os.getenv('SQL_LINK_SERVICE_INTERVAL_SECONDS', '120')))
    parser.add_argument('--sqlite-timeout-seconds', type=int, default=int(os.getenv('SQL_LINK_SERVICE_SQLITE_TIMEOUT', '300')))
    parser.add_argument('--sqlite-lock-retries', type=int, default=int(os.getenv('SQL_LINK_SERVICE_LOCK_RETRIES', '200')))
    parser.add_argument('--sqlite-lock-retry-delay-seconds', type=float, default=float(os.getenv('SQL_LINK_SERVICE_LOCK_RETRY_DELAY_SECONDS', '0.5')))
    parser.add_argument('--lock-path', default=str(PROJECT_ROOT / 'governance' / 'locks' / 'jsonl_sql_writer.lock'))
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    lock_path = Path(args.lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    fh = open(lock_path, 'a+', encoding='utf-8')
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        owner = fh.read().strip()
        msg = {'ok': False, 'reason': 'writer_lock_busy', 'lock_path': str(lock_path), 'owner': owner}
        print(json.dumps(msg, ensure_ascii=True) if args.json else f"sql_link_writer_service busy owner={owner or 'unknown'}")
        return 0

    fh.seek(0)
    fh.truncate(0)
    fh.write(f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()} cmd=sql_link_writer_service")
    fh.flush()

    out_path = PROJECT_ROOT / 'governance' / 'health' / 'sql_link_service_latest.json'

    while True:
        ts = datetime.now(timezone.utc).isoformat()
        rc, out, err = _run_link(
            timeout_s=int(args.sqlite_timeout_seconds),
            lock_retries=int(args.sqlite_lock_retries),
            retry_delay_s=float(args.sqlite_lock_retry_delay_seconds),
        )
        payload = {
            'timestamp_utc': ts,
            'ok': rc == 0,
            'rc': int(rc),
            'lock_path': str(lock_path),
            'stdout_tail': '\n'.join(out.splitlines()[-20:]),
            'stderr_tail': '\n'.join(err.splitlines()[-20:]),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"sql_link_writer_service rc={rc} ok={rc == 0} ts={ts}")

        if args.once:
            break
        time.sleep(max(int(args.interval_seconds), 10))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
