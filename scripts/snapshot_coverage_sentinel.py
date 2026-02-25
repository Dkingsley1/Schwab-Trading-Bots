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


def _latest_heartbeat_symbols_total() -> int:
    hb_dir = PROJECT_ROOT / 'governance' / 'health'
    best_ts = None
    best_total = 0
    for p in hb_dir.glob('shadow_loop_*.json'):
        try:
            row = json.loads(p.read_text(encoding='utf-8'))
            ts = _parse_ts(row.get('timestamp_utc'))
            if ts is None:
                continue
            if best_ts is None or ts > best_ts:
                best_ts = ts
                best_total = int(row.get('symbols_total', 0) or 0)
        except Exception:
            continue
    return max(best_total, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description='Snapshot coverage sentinel.')
    parser.add_argument('--hours', type=int, default=2)
    parser.add_argument('--min-coverage-ratio', type=float, default=0.75)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=max(args.hours, 1))
    expected_symbols = _latest_heartbeat_symbols_total()

    total_rows = 0
    snapshot_rows = 0
    unique_snapshot_ids = set()

    for p in (PROJECT_ROOT / 'governance').glob('shadow*/master_control_*.jsonl'):
        try:
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    ts = _parse_ts(row.get('timestamp_utc'))
                    if ts is None or ts < since:
                        continue
                    total_rows += 1
                    sid = row.get('snapshot_id')
                    if sid:
                        snapshot_rows += 1
                        unique_snapshot_ids.add(str(sid))
        except Exception:
            continue

    unique_count = len(unique_snapshot_ids)
    expected_floor = max(expected_symbols, 1)
    coverage_ratio = unique_count / float(expected_floor)
    ok = (total_rows > 0) and (coverage_ratio >= float(args.min_coverage_ratio))

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ok': bool(ok),
        'window_hours': int(args.hours),
        'expected_symbols_floor': expected_floor,
        'rows_scanned': total_rows,
        'rows_with_snapshot_id': snapshot_rows,
        'unique_snapshot_ids': unique_count,
        'coverage_ratio': round(coverage_ratio, 6),
        'min_coverage_ratio': float(args.min_coverage_ratio),
    }

    out = PROJECT_ROOT / 'governance' / 'health' / 'snapshot_coverage_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print('snapshot_coverage_ok=' + str(payload['ok']).lower() + f" ratio={payload['coverage_ratio']}")

    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
