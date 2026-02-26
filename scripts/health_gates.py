import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _first_non_empty_json(paths: List[Path]) -> Tuple[dict, str]:
    for p in paths:
        payload = _load_json(p)
        if payload:
            return payload, str(p)
    return {}, ''


def _latest_match(root: Path, pattern: str) -> Path:
    try:
        files = [p for p in root.glob(pattern) if p.is_file()]
    except Exception:
        return Path('')
    if not files:
        return Path('')
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description='Compute single health score and hard gate flags.')
    parser.add_argument('--project-root', default=str(PROJECT_ROOT))
    parser.add_argument('--stale-window-limit', type=int, default=int(os.getenv('HEALTH_GATE_STALE_WINDOW_LIMIT', '0')))
    parser.add_argument('--blocked-rate-limit', type=float, default=float(os.getenv('HEALTH_GATE_BLOCKED_RATE_LIMIT', '0.30')))
    parser.add_argument('--watchdog-restarts-limit', type=int, default=int(os.getenv('HEALTH_GATE_WATCHDOG_RESTARTS_LIMIT', '3')))
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    day = datetime.now(timezone.utc).strftime('%Y%m%d')

    one_numbers_paths = [
        project_root / 'governance' / 'health' / 'one_numbers_latest.json',
        project_root / 'exports' / 'one_numbers' / 'one_numbers_summary.json',
        project_root / 'exports' / 'one_numbers' / 'latest' / 'one_numbers_summary.json',
    ]
    one_numbers, one_numbers_source = _first_non_empty_json(one_numbers_paths)

    daily_summary_paths = [
        project_root / 'governance' / 'health' / 'daily_runtime_summary_latest.json',
        project_root / 'exports' / 'sql_reports' / 'daily_runtime_summary_latest.json',
        project_root / 'exports' / 'sql_reports' / f'daily_runtime_summary_{day}.json',
    ]
    daily_summary, daily_summary_source = _first_non_empty_json(daily_summary_paths)

    if not daily_summary:
        latest_daily = _latest_match(project_root / 'exports' / 'sql_reports', 'daily_runtime_summary_*.json')
        if latest_daily:
            daily_summary = _load_json(latest_daily)
            if daily_summary:
                daily_summary_source = str(latest_daily)

    blocked_rate = float(one_numbers.get('combined_blocked_rate', 0.0) or 0.0)
    stale_windows = int(one_numbers.get('decision_stale_windows_4h', 0) or one_numbers.get('decision_stale_windows', 0) or 0)
    watchdog_restarts = int((daily_summary.get('watchdog', {}) or {}).get('restarts', one_numbers.get('watchdog_restarts', 0) or 0))

    gate_stale = stale_windows > args.stale_window_limit
    gate_blocked = blocked_rate > args.blocked_rate_limit
    gate_restarts = watchdog_restarts > args.watchdog_restarts_limit

    score = 100.0
    score -= min(blocked_rate * 100.0 * 0.35, 35.0)
    score -= min(stale_windows * 8.0, 32.0)
    score -= min(watchdog_restarts * 7.0, 21.0)
    score = max(score, 0.0)

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'data_quality_score': round(score, 2),
        'source_files': {
            'one_numbers': one_numbers_source,
            'daily_runtime_summary': daily_summary_source,
        },
        'inputs': {
            'blocked_rate': blocked_rate,
            'stale_windows': stale_windows,
            'watchdog_restarts': watchdog_restarts,
        },
        'hard_gates': {
            'stale_windows': gate_stale,
            'blocked_rate': gate_blocked,
            'watchdog_restart_spike': gate_restarts,
        },
        'hard_gate_triggered': bool(gate_stale or gate_blocked or gate_restarts),
    }

    out = project_root / 'governance' / 'health' / 'health_gates_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"health_score={payload['data_quality_score']:.2f} hard_gate_triggered={payload['hard_gate_triggered']} "
            f"stale_windows={stale_windows} blocked_rate={blocked_rate:.4f} watchdog_restarts={watchdog_restarts}"
        )

    return 2 if payload['hard_gate_triggered'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
