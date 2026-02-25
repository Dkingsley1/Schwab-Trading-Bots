import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_ts(raw):
    if not raw:
        return None
    s = str(raw).replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description='Execution queue stress bot.')
    parser.add_argument('--hours', type=int, default=4)
    parser.add_argument('--max-queue-depth', type=int, default=2000)
    parser.add_argument('--max-queue-breach-rate', type=float, default=0.25)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=max(args.hours, 1))
    rows = 0
    breaches = 0
    max_depth_seen = 0

    for p in (PROJECT_ROOT / 'governance').glob('shadow*/master_control_*.jsonl'):
        try:
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ts = _parse_ts(row.get('timestamp_utc'))
                    if ts is None or ts < since:
                        continue
                    rows += 1
                    portfolio = row.get('portfolio', {}) if isinstance(row.get('portfolio', {}), dict) else {}
                    depth = int(portfolio.get('queue_depth', 0) or 0)
                    if depth > max_depth_seen:
                        max_depth_seen = depth
                    if depth > int(args.max_queue_depth):
                        breaches += 1
        except Exception:
            continue

    breach_rate = breaches / float(max(rows, 1))
    ok = breach_rate <= float(args.max_queue_breach_rate)

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ok': bool(ok),
        'window_hours': int(args.hours),
        'samples': int(rows),
        'max_queue_depth_seen': int(max_depth_seen),
        'queue_depth_breaches': int(breaches),
        'queue_breach_rate': round(breach_rate, 6),
        'max_queue_depth': int(args.max_queue_depth),
        'max_queue_breach_rate': float(args.max_queue_breach_rate),
    }

    out = PROJECT_ROOT / 'governance' / 'health' / 'execution_queue_stress_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print('execution_queue_stress_ok=' + str(payload['ok']).lower() + f" breach_rate={payload['queue_breach_rate']}")

    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
