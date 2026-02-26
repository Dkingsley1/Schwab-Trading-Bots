import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _one_numbers_data_quality() -> float:
    summary = _read_json(PROJECT_ROOT / 'exports' / 'one_numbers' / 'one_numbers_summary.json')
    if summary:
        try:
            return float(summary.get('data_quality_score', 0.0) or 0.0)
        except Exception:
            pass

    latest_csv = PROJECT_ROOT / 'exports' / 'one_numbers' / 'latest.csv'
    if latest_csv.exists():
        try:
            with latest_csv.open('r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if str(row.get('metric', '')) == 'data_quality_score':
                        return float(row.get('value', 0.0) or 0.0)
        except Exception:
            pass
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description='Auto-tune canary weight from fail-share and data-quality trends.')
    parser.add_argument('--apply-env', action='store_true', default=True)
    parser.add_argument('--min-weight', type=float, default=float(os.getenv('CANARY_TUNER_MIN_WEIGHT', '0.04')))
    parser.add_argument('--max-weight', type=float, default=float(os.getenv('CANARY_TUNER_MAX_WEIGHT', '0.12')))
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    readiness = _read_json(PROJECT_ROOT / 'governance' / 'walk_forward' / 'promotion_readiness_latest.json')
    fail_share = float(readiness.get('fail_share', 1.0) or 1.0)
    dq = _one_numbers_data_quality()

    if dq < 75 or fail_share > 0.90:
        target = 0.04
        reason = 'defensive_floor'
    elif dq < 82 or fail_share > 0.75:
        target = 0.06
        reason = 'high_risk'
    elif dq < 88 or fail_share > 0.55:
        target = 0.08
        reason = 'moderate_risk'
    elif fail_share > 0.35:
        target = 0.10
        reason = 'cautious_expand'
    else:
        target = 0.12
        reason = 'healthy_expand'

    target = max(min(float(target), float(args.max_weight)), float(args.min_weight))

    env_file = PROJECT_ROOT / 'governance' / 'health' / 'canary_rollout.env'
    if args.apply_env:
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(f'CANARY_MAX_WEIGHT={target:.4f}\n', encoding='utf-8')

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'fail_share': round(fail_share, 6),
        'data_quality_score': round(dq, 4),
        'target_canary_max_weight': round(target, 6),
        'reason': reason,
        'env_file': str(env_file),
        'applied': bool(args.apply_env),
    }

    out = PROJECT_ROOT / 'governance' / 'health' / 'canary_auto_tuner_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"canary_auto_tuner weight={target:.4f} reason={reason} fail_share={fail_share:.4f} dq={dq:.1f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
