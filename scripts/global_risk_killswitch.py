import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _first_non_empty(paths: List[Path]) -> Tuple[dict, str]:
    for p in paths:
        payload = _load(p)
        if payload:
            return payload, str(p)
    return {}, ''


def main() -> int:
    parser = argparse.ArgumentParser(description='Account-level global risk kill-switch.')
    parser.add_argument('--max-blocked-rate', type=float, default=float(os.getenv('GLOBAL_KILL_BLOCKED_RATE_MAX', '0.45')))
    parser.add_argument('--max-abs-pnl-proxy', type=float, default=float(os.getenv('GLOBAL_KILL_ABS_PNL_PROXY_MAX', '0.03')))
    parser.add_argument('--max-stale-windows', type=int, default=int(os.getenv('GLOBAL_KILL_STALE_WINDOWS_MAX', '2')))
    parser.add_argument('--max-watchdog-restarts', type=int, default=int(os.getenv('GLOBAL_KILL_WATCHDOG_RESTARTS_MAX', '5')))
    parser.add_argument('--auto-clear', action='store_true')
    args = parser.parse_args()

    one, one_src = _first_non_empty(
        [
            PROJECT_ROOT / 'governance' / 'health' / 'one_numbers_latest.json',
            PROJECT_ROOT / 'exports' / 'one_numbers' / 'one_numbers_summary.json',
            PROJECT_ROOT / 'exports' / 'one_numbers' / 'latest' / 'one_numbers_summary.json',
        ]
    )
    health = _load(PROJECT_ROOT / 'governance' / 'health' / 'health_gates_latest.json')

    blocked_rate = float(one.get('combined_blocked_rate', 0.0) or 0.0)
    pnl_proxy = float(one.get('combined_pnl_proxy', one.get('crypto_pnl_proxy', 0.0) or 0.0) or 0.0)
    stale = int(one.get('decision_stale_windows_4h', one.get('decision_stale_windows', 0) or 0) or 0)
    restarts = int(one.get('watchdog_restarts', 0) or 0)

    reasons = []
    if blocked_rate > args.max_blocked_rate:
        reasons.append(f'blocked_rate>{args.max_blocked_rate}')
    if abs(pnl_proxy) > args.max_abs_pnl_proxy:
        reasons.append(f'abs_pnl_proxy>{args.max_abs_pnl_proxy}')
    if stale > args.max_stale_windows:
        reasons.append(f'stale_windows>{args.max_stale_windows}')
    if restarts > args.max_watchdog_restarts:
        reasons.append(f'watchdog_restarts>{args.max_watchdog_restarts}')
    if bool(health.get('hard_gate_triggered', False)):
        reasons.append('health_hard_gate_triggered')

    halt_flag = PROJECT_ROOT / 'governance' / 'health' / 'GLOBAL_TRADING_HALT.flag'
    now = datetime.now(timezone.utc).isoformat()
    action = 'none'

    if reasons:
        halt_flag.parent.mkdir(parents=True, exist_ok=True)
        halt_flag.write_text(
            json.dumps({'timestamp_utc': now, 'reason': 'global_risk_killswitch', 'details': reasons}, ensure_ascii=True),
            encoding='utf-8',
        )
        action = 'halt_set'
    elif args.auto_clear and halt_flag.exists():
        halt_flag.unlink()
        action = 'halt_cleared'

    payload = {
        'timestamp_utc': now,
        'action': action,
        'halt': halt_flag.exists(),
        'source_files': {
            'one_numbers': one_src,
            'health_gates': str(PROJECT_ROOT / 'governance' / 'health' / 'health_gates_latest.json'),
        },
        'reasons': reasons,
        'metrics': {
            'blocked_rate': blocked_rate,
            'pnl_proxy': pnl_proxy,
            'stale_windows': stale,
            'watchdog_restarts': restarts,
        },
    }

    out = PROJECT_ROOT / 'governance' / 'health' / 'global_killswitch_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    events = PROJECT_ROOT / 'governance' / 'watchdog' / 'global_killswitch_events.jsonl'
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=True) + '\n')

    print(json.dumps(payload, ensure_ascii=True))
    return 2 if reasons else 0


if __name__ == '__main__':
    raise SystemExit(main())
