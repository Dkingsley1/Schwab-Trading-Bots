import argparse
import json
from collections import Counter
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
    parser = argparse.ArgumentParser(description='Quarantine pressure bot.')
    parser.add_argument('--hours', type=int, default=6)
    parser.add_argument('--max-quarantine-events', type=int, default=120)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=max(args.hours, 1))
    event_count = 0
    by_symbol = Counter()
    reasons = Counter()

    for p in (PROJECT_ROOT / 'governance').glob('shadow*/snapshot_debug_*.jsonl'):
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
                    reason = str(row.get('reason', ''))
                    if reason not in {'quarantine_skip', 'stale_price_quarantine'}:
                        continue
                    event_count += 1
                    sym = str(row.get('symbol', 'UNKNOWN'))
                    by_symbol[sym] += 1
                    reasons[reason] += 1
        except Exception:
            continue

    ok = event_count <= int(args.max_quarantine_events)
    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ok': bool(ok),
        'window_hours': int(args.hours),
        'quarantine_events': int(event_count),
        'max_quarantine_events': int(args.max_quarantine_events),
        'reason_breakdown': dict(reasons),
        'top_symbols': [{'symbol': k, 'count': v} for k, v in by_symbol.most_common(20)],
    }

    out = PROJECT_ROOT / 'governance' / 'health' / 'quarantine_pressure_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print('quarantine_pressure_ok=' + str(payload['ok']).lower() + f" events={event_count}")

    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
