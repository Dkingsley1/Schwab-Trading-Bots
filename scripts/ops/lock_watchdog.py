import argparse
import glob
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / 'governance' / 'health' / 'lock_watchdog_latest.json'

PID_RE = re.compile(r'pid=(\d+)')


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _extract_pid(text: str) -> int | None:
    m = PID_RE.search(text or '')
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _lock_candidates() -> list[Path]:
    rows: list[Path] = []
    rows.extend(Path(PROJECT_ROOT / 'governance').glob('*.lock'))
    rows.extend(Path(PROJECT_ROOT / 'governance' / 'locks').glob('*.lock'))
    uniq = {str(p.resolve(strict=False)): p for p in rows if p.exists() and p.is_file()}
    return [uniq[k] for k in sorted(uniq.keys())]


def main() -> int:
    parser = argparse.ArgumentParser(description='Detect and optionally clear stale lock files.')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    stale: list[dict] = []
    healthy: list[dict] = []

    for path in _lock_candidates():
        text = ''
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            text = ''

        pid = _extract_pid(text)
        if pid is None:
            stale.append({'lock_path': str(path), 'reason': 'missing_pid', 'pid': None})
            continue

        if _pid_alive(pid):
            healthy.append({'lock_path': str(path), 'pid': pid})
        else:
            stale.append({'lock_path': str(path), 'reason': 'owner_pid_not_running', 'pid': pid})

    removed: list[str] = []
    if args.apply:
        for row in stale:
            lp = Path(row['lock_path'])
            try:
                lp.unlink(missing_ok=True)
                removed.append(str(lp))
            except Exception:
                continue

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'healthy_locks': healthy,
        'stale_locks': stale,
        'apply': bool(args.apply),
        'removed': removed,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"lock_watchdog stale={len(stale)} removed={len(removed)} healthy={len(healthy)}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
