import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_ts(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    s = str(raw).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def _iter_paths(base: Path, pattern: str, profile: str, domain: str) -> Iterable[Path]:
    if not base.exists():
        return []
    out: List[Path] = []
    for p in sorted(base.glob(pattern)):
        rel = p.as_posix().lower()
        if profile and (f"shadow_{profile.lower()}" not in rel):
            continue
        if domain and (f"_{domain.lower()}" not in rel):
            continue
        out.append(p)
    return out


def _scan_jsonl_rows(paths: Iterable[Path], since_ts: datetime) -> Dict[str, Any]:
    rows = 0
    timestamps: List[datetime] = []
    files = 0
    for p in paths:
        files += 1
        try:
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    ts = _parse_ts(obj.get('timestamp_utc'))
                    if ts is None:
                        continue
                    if ts < since_ts:
                        continue
                    rows += 1
                    timestamps.append(ts)
        except Exception:
            continue

    timestamps.sort()
    return {
        'rows': rows,
        'files': files,
        'timestamps': timestamps,
    }


def _stale_windows(stamps: List[datetime], stale_seconds: int) -> int:
    if len(stamps) < 2:
        return 0
    out = 0
    for i in range(1, len(stamps)):
        if (stamps[i] - stamps[i - 1]).total_seconds() > stale_seconds:
            out += 1
    return out


def _summary_candidate_paths(day_utc: str, previous_day_utc: str) -> List[Path]:
    sql_root = PROJECT_ROOT / 'exports' / 'sql_reports'
    return [
        sql_root / f'daily_runtime_summary_{day_utc}.json',
        sql_root / f'daily_runtime_summary_{previous_day_utc}.json',
        sql_root / 'daily_runtime_summary_latest.json',
        PROJECT_ROOT / 'governance' / 'health' / 'daily_runtime_summary_latest.json',
    ]


def _summary_metrics(now: datetime, *, profile: str, domain: str) -> Optional[Dict[str, Any]]:
    if profile or domain:
        return None

    day_utc = now.strftime("%Y%m%d")
    previous_day_utc = (now - timedelta(days=1)).strftime("%Y%m%d")
    best: Optional[Dict[str, Any]] = None
    best_rows = -1
    for path in _summary_candidate_paths(day_utc, previous_day_utc):
        payload = _load_json(path)
        if not payload:
            continue
        payload_day = str(payload.get('day', '')).strip()
        if payload_day not in {'', day_utc, previous_day_utc}:
            continue
        decision = payload.get('decision', {}) if isinstance(payload.get('decision'), dict) else {}
        governance = payload.get('governance', {}) if isinstance(payload.get('governance'), dict) else {}
        candidate = {
            'source': 'daily_runtime_summary',
            'path': str(path),
            'decision': {
                'rows': int(decision.get('rows', 0) or 0),
                'files': len(decision.get('files', []) or []),
                'stale_windows': int(decision.get('stale_windows', 0) or 0),
            },
            'governance': {
                'rows': int(governance.get('rows', 0) or 0),
                'files': len(governance.get('files', []) or []),
                'stale_windows': int(governance.get('stale_windows', 0) or 0),
            },
        }
        total_rows = int(candidate['decision']['rows']) + int(candidate['governance']['rows'])
        if total_rows > best_rows:
            best = candidate
            best_rows = total_rows
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description='24h replay pre-open sanity check over decision/governance logs.')
    parser.add_argument('--hours', type=int, default=24)
    parser.add_argument('--stale-seconds', type=int, default=900)
    parser.add_argument('--min-decision-rows', type=int, default=200)
    parser.add_argument('--min-governance-rows', type=int, default=100)
    parser.add_argument('--max-decision-stale-windows', type=int, default=int(os.getenv('REPLAY_PREOPEN_MAX_DECISION_STALE_WINDOWS', '12')))
    parser.add_argument('--max-governance-stale-windows', type=int, default=int(os.getenv('REPLAY_PREOPEN_MAX_GOVERNANCE_STALE_WINDOWS', '12')))
    parser.add_argument('--strict-exit', action='store_true', default=os.getenv('REPLAY_PREOPEN_STRICT_EXIT', '0').strip() == '1')
    parser.add_argument('--profile', default='')
    parser.add_argument('--domain', default='')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since_ts = now - timedelta(hours=max(args.hours, 1))

    metrics = _summary_metrics(now, profile=args.profile, domain=args.domain)
    if metrics is not None:
        decision = {
            'rows': int(metrics['decision']['rows']),
            'files': int(metrics['decision']['files']),
            'timestamps': [],
        }
        governance = {
            'rows': int(metrics['governance']['rows']),
            'files': int(metrics['governance']['files']),
            'timestamps': [],
        }
        decision_stale = int(metrics['decision']['stale_windows'])
        governance_stale = int(metrics['governance']['stale_windows'])
        data_source = str(metrics.get('source') or 'daily_runtime_summary')
        data_source_path = str(metrics.get('path') or '')
    else:
        decision_paths = _iter_paths(
            PROJECT_ROOT / 'decision_explanations',
            'shadow*/decision_explanations_*.jsonl',
            profile=args.profile,
            domain=args.domain,
        )
        governance_paths = _iter_paths(
            PROJECT_ROOT / 'governance',
            'shadow*/master_control_*.jsonl',
            profile=args.profile,
            domain=args.domain,
        )

        decision = _scan_jsonl_rows(decision_paths, since_ts)
        governance = _scan_jsonl_rows(governance_paths, since_ts)

        decision_stale = _stale_windows(decision['timestamps'], max(args.stale_seconds, 1))
        governance_stale = _stale_windows(governance['timestamps'], max(args.stale_seconds, 1))
        data_source = 'jsonl_scan'
        data_source_path = ''

    failed: List[str] = []
    if decision['rows'] < max(args.min_decision_rows, 1):
        failed.append('decision_rows_low')
    if governance['rows'] < max(args.min_governance_rows, 1):
        failed.append('governance_rows_low')
    if decision_stale > max(args.max_decision_stale_windows, 0):
        failed.append('decision_stale_windows_high')
    if governance_stale > max(args.max_governance_stale_windows, 0):
        failed.append('governance_stale_windows_high')

    severity = 'ok' if len(failed) == 0 else 'warn'
    payload = {
        'timestamp_utc': now.isoformat(),
        'ok': len(failed) == 0,
        'severity': severity,
        'failed_checks': failed,
        'window_hours': int(args.hours),
        'since_utc': since_ts.isoformat(),
        'filters': {
            'profile': args.profile or 'all',
            'domain': args.domain or 'all',
        },
        'data_source': data_source,
        'data_source_path': data_source_path,
        'decision': {
            'rows': int(decision['rows']),
            'files_scanned': int(decision['files']),
            'stale_windows': int(decision_stale),
        },
        'governance': {
            'rows': int(governance['rows']),
            'files_scanned': int(governance['files']),
            'stale_windows': int(governance_stale),
        },
        'thresholds': {
            'stale_seconds': int(args.stale_seconds),
            'min_decision_rows': int(args.min_decision_rows),
            'min_governance_rows': int(args.min_governance_rows),
            'max_decision_stale_windows': int(args.max_decision_stale_windows),
            'max_governance_stale_windows': int(args.max_governance_stale_windows),
        },
    }

    out = PROJECT_ROOT / 'governance' / 'health' / 'replay_preopen_sanity_latest.json'
    compat = PROJECT_ROOT / 'governance' / 'health' / 'preopen_replay_sanity_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=True, indent=2)
    out.write_text(encoded, encoding='utf-8')
    compat.write_text(encoded, encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            'replay_preopen_sanity_ok=' + str(payload['ok']).lower()
            + f" decision_rows={payload['decision']['rows']} governance_rows={payload['governance']['rows']}"
            + f" decision_stale={payload['decision']['stale_windows']} governance_stale={payload['governance']['stale_windows']}"
        )

    if payload['ok']:
        return 0
    return 2 if args.strict_exit else 0


if __name__ == '__main__':
    raise SystemExit(main())
