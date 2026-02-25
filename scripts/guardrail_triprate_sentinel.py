import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_REASONS = (
    'feature_freshness_guard',
    'master_latency_slo_timeout',
    'event_lock_paused',
    'circuit_open_skip',
)


def _parse_ts(raw):
    if not raw:
        return None
    s = str(raw).replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description='Guardrail trip-rate sentinel.')
    parser.add_argument('--hours', type=int, default=6)
    parser.add_argument('--max-trip-rate', type=float, default=0.40)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=max(args.hours, 1))
    counters = Counter()
    total = 0

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
                    total += 1
                    reasons = row.get('recommendations')
                    _ = reasons
                    m_reasons = row.get('master_outputs', {})
                    _ = m_reasons
                    action_reasons = row.get('options_plan', {})
                    _ = action_reasons
                    # main source: decision reasons captured in governance row via string fields is limited,
                    # so inspect stringified row for guardrail reason tags.
                    blob = json.dumps(row, ensure_ascii=True)
                    for r in TARGET_REASONS:
                        if r in blob:
                            counters[r] += 1
        except Exception:
            continue

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
                    if reason in {'event_lock_paused', 'circuit_open_skip'}:
                        counters[reason] += 1
        except Exception:
            continue

    total_trips = sum(counters.values())
    trip_rate = (total_trips / float(max(total, 1)))
    ok = trip_rate <= float(args.max_trip_rate)

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ok': bool(ok),
        'window_hours': int(args.hours),
        'samples': int(total),
        'trip_count': int(total_trips),
        'trip_rate': round(trip_rate, 6),
        'max_trip_rate': float(args.max_trip_rate),
        'trip_breakdown': dict(counters),
    }

    out = PROJECT_ROOT / 'governance' / 'health' / 'guardrail_triprate_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print('guardrail_triprate_ok=' + str(payload['ok']).lower() + f" rate={payload['trip_rate']}")

    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
