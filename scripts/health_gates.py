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


def _parse_iso_utc(raw: object) -> datetime | None:
    text = str(raw or '').strip()
    if not text:
        return None
    text = text.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _payload_timestamp(payload: dict, path: Path) -> float:
    for key in ('timestamp_utc', 'updated_at_utc', 'updated_at', 'created_at', 'ended_utc', 'started_utc'):
        ts = _parse_iso_utc(payload.get(key))
        if ts is not None:
            return float(ts.timestamp())
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _freshest_non_empty_json(paths: List[Path]) -> Tuple[dict, str]:
    candidates: List[Tuple[float, dict, str]] = []
    for p in paths:
        payload = _load_json(p)
        if payload:
            candidates.append((_payload_timestamp(payload, p), payload, str(p)))
    if not candidates:
        return {}, ''
    candidates.sort(key=lambda row: row[0])
    _, payload, source = candidates[-1]
    return payload, source


def _latest_match(root: Path, pattern: str) -> Path:
    try:
        files = [p for p in root.glob(pattern) if p.is_file()]
    except Exception:
        return Path('')
    if not files:
        return Path('')
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def _as_bool(raw: str, default: bool = True) -> bool:
    text = str(raw or '').strip().lower()
    if not text:
        return bool(default)
    return text in {'1', 'true', 'yes', 'on'}


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _effective_blocked_rate(data_blocked_rate: float, risk_blocked_rate: float, *, risk_weight: float = 0.25) -> float:
    return min(max(float(data_blocked_rate), 0.0) + (max(float(risk_blocked_rate), 0.0) * max(float(risk_weight), 0.0)), 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description='Compute single health score and hard gate flags.')
    parser.add_argument('--project-root', default=str(PROJECT_ROOT))
    parser.add_argument('--stale-window-limit', type=int, default=int(os.getenv('HEALTH_GATE_STALE_WINDOW_LIMIT', '0')))
    parser.add_argument('--blocked-rate-limit', type=float, default=float(os.getenv('HEALTH_GATE_BLOCKED_RATE_LIMIT', '0.30')))
    parser.add_argument('--watchdog-restarts-limit', type=int, default=int(os.getenv('HEALTH_GATE_WATCHDOG_RESTARTS_LIMIT', '3')))
    parser.add_argument('--ingestion-pending-lines-limit', type=int, default=int(os.getenv('HEALTH_GATE_INGEST_PENDING_LINES_LIMIT', '20000')))
    parser.add_argument('--ingestion-oldest-age-seconds-limit', type=int, default=int(os.getenv('HEALTH_GATE_INGEST_OLDEST_AGE_SECONDS_LIMIT', '600')))
    parser.add_argument('--ingestion-invalid-lines-limit', type=int, default=int(os.getenv('HEALTH_GATE_INGEST_INVALID_LINES_LIMIT', '10')))
    parser.add_argument(
        '--ingestion-backpressure-overload-fails',
        action='store_true',
        default=_as_bool(os.getenv('HEALTH_GATE_INGEST_BACKPRESSURE_OVERLOAD_FAILS', '1')),
    )
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    day = datetime.now(timezone.utc).strftime('%Y%m%d')

    one_numbers_paths = [
        project_root / 'governance' / 'health' / 'one_numbers_latest.json',
        project_root / 'exports' / 'one_numbers' / 'one_numbers_summary.json',
        project_root / 'exports' / 'one_numbers' / 'latest' / 'one_numbers_summary.json',
    ]
    one_numbers, one_numbers_source = _freshest_non_empty_json(one_numbers_paths)

    daily_summary_paths = [
        project_root / 'governance' / 'health' / 'daily_runtime_summary_latest.json',
        project_root / 'exports' / 'sql_reports' / 'daily_runtime_summary_latest.json',
        project_root / 'exports' / 'sql_reports' / f'daily_runtime_summary_{day}.json',
    ]
    daily_summary, daily_summary_source = _freshest_non_empty_json(daily_summary_paths)

    if not daily_summary:
        latest_daily = _latest_match(project_root / 'exports' / 'sql_reports', 'daily_runtime_summary_*.json')
        if latest_daily:
            daily_summary = _load_json(latest_daily)
            if daily_summary:
                daily_summary_source = str(latest_daily)

    ingestion_health_paths = [
        project_root / 'governance' / 'health' / 'jsonl_sql_ingestion_health_latest.json',
        project_root / 'governance' / 'health' / 'jsonl_sql_ingestion_health_trading_latest.json',
        project_root / 'governance' / 'health' / 'jsonl_sql_ingestion_health_data_latest.json',
        project_root / 'governance' / 'health' / 'jsonl_sql_ingestion_health_governance_latest.json',
    ]
    ingestion_health, ingestion_health_source = _freshest_non_empty_json(ingestion_health_paths)

    backpressure_paths = [
        project_root / 'governance' / 'health' / 'ingestion_backpressure_latest.json',
    ]
    backpressure, backpressure_source = _freshest_non_empty_json(backpressure_paths)

    combined_blocked_rate = float(one_numbers.get('combined_blocked_rate', 0.0) or 0.0)
    data_blocked_raw = one_numbers.get('data_blocked_rate')
    risk_blocked_raw = one_numbers.get('risk_blocked_rate')
    if data_blocked_raw is None and risk_blocked_raw is None:
        data_blocked_rate = combined_blocked_rate
        risk_blocked_rate = 0.0
    else:
        data_blocked_rate = float(data_blocked_raw or 0.0)
        risk_blocked_rate = float(risk_blocked_raw or 0.0)
    blocked_rate = _effective_blocked_rate(data_blocked_rate, risk_blocked_rate)
    stale_windows = int(one_numbers.get('decision_stale_windows_4h', 0) or one_numbers.get('decision_stale_windows', 0) or 0)
    watchdog_restarts = int((daily_summary.get('watchdog', {}) or {}).get('restarts', one_numbers.get('watchdog_restarts', 0) or 0))

    sqlite_ingest = ingestion_health.get('sqlite', {}) if isinstance(ingestion_health.get('sqlite', {}), dict) else {}
    ingest_pending_lines = _to_int(sqlite_ingest.get('pending_lines'), 0)
    ingest_oldest_age_s = _to_float(sqlite_ingest.get('oldest_uningested_age_seconds'), 0.0)
    ingest_invalid_lines = _to_int(sqlite_ingest.get('invalid'), 0)
    ingest_p95_latency_s = _to_float(
        (((ingestion_health.get('latency_slo', {}) or {}).get('sqlite', {}) or {}).get('all', {}) or {}).get('p95_seconds'),
        0.0,
    )

    backpressure_overload = bool(backpressure.get('overload', False))
    backpressure_pending_lines = _to_int(backpressure.get('pending_lines'), 0)
    backpressure_oldest_age_s = _to_float(backpressure.get('oldest_pending_age_seconds'), 0.0)

    gate_stale = stale_windows > args.stale_window_limit
    gate_blocked = blocked_rate > args.blocked_rate_limit
    gate_restarts = watchdog_restarts > args.watchdog_restarts_limit

    gate_ingest_pending = ingest_pending_lines > int(args.ingestion_pending_lines_limit)
    gate_ingest_oldest_age = ingest_oldest_age_s > float(args.ingestion_oldest_age_seconds_limit)
    gate_ingest_invalid = ingest_invalid_lines > int(args.ingestion_invalid_lines_limit)
    gate_backpressure_overload = bool(backpressure_overload and args.ingestion_backpressure_overload_fails)

    score = 100.0
    score -= min(blocked_rate * 100.0 * 0.35, 35.0)
    score -= min(stale_windows * 8.0, 32.0)
    score -= min(watchdog_restarts * 7.0, 21.0)
    score -= min((ingest_pending_lines / 1000.0) * 0.8, 8.0)
    score -= min((ingest_oldest_age_s / 60.0) * 0.7, 7.0)
    score -= min(max(ingest_invalid_lines, 0) * 0.25, 5.0)
    if backpressure_overload:
        score -= 4.0
    score = max(score, 0.0)

    hard_gate_triggered = bool(
        gate_stale
        or gate_blocked
        or gate_restarts
        or gate_ingest_pending
        or gate_ingest_oldest_age
        or gate_ingest_invalid
        or gate_backpressure_overload
    )

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'data_quality_score': round(score, 2),
        'source_files': {
            'one_numbers': one_numbers_source,
            'daily_runtime_summary': daily_summary_source,
            'jsonl_sql_ingestion_health': ingestion_health_source,
            'ingestion_backpressure': backpressure_source,
        },
        'inputs': {
            'blocked_rate': blocked_rate,
            'combined_blocked_rate': combined_blocked_rate,
            'data_blocked_rate': data_blocked_rate,
            'risk_blocked_rate': risk_blocked_rate,
            'blocked_rate_risk_weight': 0.25,
            'stale_windows': stale_windows,
            'watchdog_restarts': watchdog_restarts,
            'ingest_pending_lines': ingest_pending_lines,
            'ingest_oldest_uningested_age_seconds': ingest_oldest_age_s,
            'ingest_invalid_lines': ingest_invalid_lines,
            'ingest_p95_latency_seconds': ingest_p95_latency_s,
            'backpressure_overload': backpressure_overload,
            'backpressure_pending_lines': backpressure_pending_lines,
            'backpressure_oldest_pending_age_seconds': backpressure_oldest_age_s,
        },
        'hard_gates': {
            'stale_windows': gate_stale,
            'blocked_rate': gate_blocked,
            'watchdog_restart_spike': gate_restarts,
            'ingestion_pending_lines': gate_ingest_pending,
            'ingestion_oldest_age': gate_ingest_oldest_age,
            'ingestion_invalid_lines': gate_ingest_invalid,
            'ingestion_backpressure_overload': gate_backpressure_overload,
        },
        'thresholds': {
            'stale_window_limit': int(args.stale_window_limit),
            'blocked_rate_limit': float(args.blocked_rate_limit),
            'watchdog_restarts_limit': int(args.watchdog_restarts_limit),
            'ingestion_pending_lines_limit': int(args.ingestion_pending_lines_limit),
            'ingestion_oldest_age_seconds_limit': int(args.ingestion_oldest_age_seconds_limit),
            'ingestion_invalid_lines_limit': int(args.ingestion_invalid_lines_limit),
            'ingestion_backpressure_overload_fails': bool(args.ingestion_backpressure_overload_fails),
        },
        'hard_gate_triggered': hard_gate_triggered,
    }

    out = project_root / 'governance' / 'health' / 'health_gates_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"health_score={payload['data_quality_score']:.2f} hard_gate_triggered={payload['hard_gate_triggered']} "
            f"stale_windows={stale_windows} blocked_rate={blocked_rate:.4f} watchdog_restarts={watchdog_restarts} "
            f"ingest_pending_lines={ingest_pending_lines} ingest_oldest_age_s={ingest_oldest_age_s:.1f} "
            f"ingest_invalid_lines={ingest_invalid_lines} backpressure_overload={backpressure_overload}"
        )

    return 2 if payload['hard_gate_triggered'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
