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
    parser = argparse.ArgumentParser(description='Data source divergence bot.')
    parser.add_argument('--hours', type=int, default=2)
    parser.add_argument('--max-relative-spread', type=float, default=0.03)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=max(args.hours, 1))
    # key: (symbol, minute_bucket) -> list of prices across sleeves/providers
    buckets = {}

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
                    sym = str(row.get('symbol', '') or '')
                    if not sym:
                        continue
                    market = row.get('market', {}) if isinstance(row.get('market', {}), dict) else {}
                    px = float(market.get('last_price', 0.0) or 0.0)
                    if px <= 0:
                        continue
                    minute = ts.replace(second=0, microsecond=0).isoformat()
                    k = (sym, minute)
                    buckets.setdefault(k, []).append(px)
        except Exception:
            continue

    worst_rel = 0.0
    compared = 0
    offenders = []
    for (sym, minute), prices in buckets.items():
        if len(prices) < 2:
            continue
        mn = min(prices)
        mx = max(prices)
        rel = (mx - mn) / max(mn, 1e-8)
        compared += 1
        if rel > worst_rel:
            worst_rel = rel
        if rel > float(args.max_relative_spread):
            offenders.append({'symbol': sym, 'minute': minute, 'rel_spread': round(rel, 6), 'n': len(prices)})

    ok = len(offenders) == 0
    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ok': bool(ok),
        'window_hours': int(args.hours),
        'compared_buckets': int(compared),
        'worst_relative_spread': round(worst_rel, 6),
        'max_relative_spread': float(args.max_relative_spread),
        'offenders': offenders[:50],
    }

    out = PROJECT_ROOT / 'governance' / 'health' / 'data_source_divergence_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print('data_source_divergence_ok=' + str(payload['ok']).lower() + f" worst={payload['worst_relative_spread']}")

    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
