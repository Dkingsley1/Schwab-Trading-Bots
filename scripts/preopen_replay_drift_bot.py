import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _median(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    m = n // 2
    if n % 2 == 1:
        return float(s[m])
    return (float(s[m - 1]) + float(s[m])) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description='Pre-open replay drift bot.')
    parser.add_argument('--max-row-drift', type=float, default=float(os.getenv('PREOPEN_REPLAY_DRIFT_MAX_ROW', '1.20')))
    parser.add_argument('--max-stale-drift', type=float, default=float(os.getenv('PREOPEN_REPLAY_DRIFT_MAX_STALE', '1.00')))
    parser.add_argument('--min-history-points', type=int, default=int(os.getenv('PREOPEN_REPLAY_DRIFT_MIN_HISTORY', '3')))
    parser.add_argument('--strict-exit', action='store_true', default=os.getenv('PREOPEN_REPLAY_DRIFT_STRICT_EXIT', '0').strip() == '1')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    latest = _load(PROJECT_ROOT / 'governance' / 'health' / 'replay_preopen_sanity_latest.json')
    if not latest:
        payload = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'ok': False,
            'reason': 'missing_replay_preopen_sanity_latest',
        }
        out = PROJECT_ROOT / 'governance' / 'health' / 'preopen_replay_drift_latest.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=True))
        return 2

    history_path = PROJECT_ROOT / 'governance' / 'health' / 'preopen_replay_drift_history.jsonl'
    history = []
    if history_path.exists():
        with history_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                history.append(row)

    cur_dec_rows = float(latest.get('decision', {}).get('rows', 0) or 0)
    cur_gov_rows = float(latest.get('governance', {}).get('rows', 0) or 0)
    cur_dec_stale = float(latest.get('decision', {}).get('stale_windows', 0) or 0)
    cur_gov_stale = float(latest.get('governance', {}).get('stale_windows', 0) or 0)

    base_dec_rows = _median([float(x.get('decision_rows', 0) or 0) for x in history[-7:]])
    base_gov_rows = _median([float(x.get('governance_rows', 0) or 0) for x in history[-7:]])
    base_dec_stale = _median([float(x.get('decision_stale', 0) or 0) for x in history[-7:]])
    base_gov_stale = _median([float(x.get('governance_stale', 0) or 0) for x in history[-7:]])

    dec_row_drift = abs(cur_dec_rows - base_dec_rows) / max(base_dec_rows, 1.0)
    gov_row_drift = abs(cur_gov_rows - base_gov_rows) / max(base_gov_rows, 1.0)
    dec_stale_drift = abs(cur_dec_stale - base_dec_stale) / max(base_dec_stale + 1.0, 1.0)
    gov_stale_drift = abs(cur_gov_stale - base_gov_stale) / max(base_gov_stale + 1.0, 1.0)

    history_points = len(history[-7:])
    warmup = history_points < max(int(args.min_history_points), 1)
    ok = warmup or (
        (dec_row_drift <= float(args.max_row_drift)) and
        (gov_row_drift <= float(args.max_row_drift)) and
        (dec_stale_drift <= float(args.max_stale_drift)) and
        (gov_stale_drift <= float(args.max_stale_drift))
    )

    severity = 'ok' if ok else 'warn'
    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'ok': bool(ok),
        'severity': severity,
        'current': {
            'decision_rows': cur_dec_rows,
            'governance_rows': cur_gov_rows,
            'decision_stale': cur_dec_stale,
            'governance_stale': cur_gov_stale,
        },
        'baseline_median_7': {
            'decision_rows': base_dec_rows,
            'governance_rows': base_gov_rows,
            'decision_stale': base_dec_stale,
            'governance_stale': base_gov_stale,
        },
        'drift': {
            'decision_rows': round(dec_row_drift, 6),
            'governance_rows': round(gov_row_drift, 6),
            'decision_stale': round(dec_stale_drift, 6),
            'governance_stale': round(gov_stale_drift, 6),
        },
        'history_points': int(history_points),
        'warmup_mode': bool(warmup),
        'thresholds': {
            'max_row_drift': float(args.max_row_drift),
            'max_stale_drift': float(args.max_stale_drift),
        },
    }

    out = PROJECT_ROOT / 'governance' / 'health' / 'preopen_replay_drift_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    hist_row = {
        'timestamp_utc': payload['timestamp_utc'],
        'decision_rows': cur_dec_rows,
        'governance_rows': cur_gov_rows,
        'decision_stale': cur_dec_stale,
        'governance_stale': cur_gov_stale,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(hist_row, ensure_ascii=True) + '\n')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print('preopen_replay_drift_ok=' + str(payload['ok']).lower() + f" drift={payload['drift']}")

    if ok:
        return 0
    return 2 if args.strict_exit else 0


if __name__ == '__main__':
    raise SystemExit(main())
