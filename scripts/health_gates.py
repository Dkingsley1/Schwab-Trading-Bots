import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_CONTROL_BACKPRESSURE_OVERRIDE_MAX_AGE_SECONDS = 1800.0


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


def _path_size_gb(path: Path) -> float:
    try:
        return float(path.stat().st_size) / (1024.0 ** 3)
    except Exception:
        return 0.0


def _sqlite_live_size_gb(path: Path) -> float:
    try:
        logical_bytes = float(path.stat().st_size)
    except Exception:
        return 0.0

    try:
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=1)
        try:
            page_size_row = conn.execute('PRAGMA page_size').fetchone()
            page_count_row = conn.execute('PRAGMA page_count').fetchone()
            freelist_row = conn.execute('PRAGMA freelist_count').fetchone()
        finally:
            conn.close()
        page_size = int(page_size_row[0] if page_size_row and page_size_row[0] is not None else 0)
        page_count = int(page_count_row[0] if page_count_row and page_count_row[0] is not None else 0)
        freelist_count = int(freelist_row[0] if freelist_row and freelist_row[0] is not None else 0)
        live_page_bytes = max(page_count - freelist_count, 0) * max(page_size, 0)
        if live_page_bytes > 0:
            logical_bytes = min(logical_bytes, float(live_page_bytes))
    except Exception:
        pass
    return logical_bytes / (1024.0 ** 3)


def _priority_shard_live_db_size_gb(project_root: Path, shard_name: str) -> float | None:
    safe_name = str(shard_name or '').strip().lower().replace('-', '_')
    if not safe_name:
        return None
    db_path = project_root / 'data' / 'sql_link_shards' / f'jsonl_link_{safe_name}.sqlite3'
    if not db_path.exists():
        return None
    size_gb = _sqlite_live_size_gb(db_path)
    return float(size_gb) if size_gb > 0.0 else 0.0


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


def _parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw or '').split(',') if part.strip()]


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


def _storage_control_backpressure_override(storage_control: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(storage_control, dict) or not storage_control:
        return {"active": False}
    timestamp = _parse_iso_utc(storage_control.get("timestamp_utc"))
    if timestamp is None:
        return {"active": False, "reason": "missing_storage_control_timestamp"}
    age_seconds = max((datetime.now(timezone.utc) - timestamp).total_seconds(), 0.0)
    backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    effective = backpressure.get("effective_raw_live") if isinstance(backpressure.get("effective_raw_live"), dict) else {}
    data_integrity = storage_control.get("data_integrity") if isinstance(storage_control.get("data_integrity"), dict) else {}
    steady_state = storage_control.get("steady_state") if isinstance(storage_control.get("steady_state"), dict) else {}
    targets = steady_state.get("targets") if isinstance(steady_state.get("targets"), dict) else {}
    source = str(backpressure.get("effective_raw_live_source") or effective.get("source") or "").strip()
    overlay_adjusted = bool(backpressure.get("overlay_adjusted", False))
    overlay_clear = bool(backpressure.get("overlay_pressure_clear", False) or source == "fresh_empty_sql_ingestion_overlay")
    storage_ready = bool(
        str(storage_control.get("overall_status") or "").strip().lower() == "ready"
        and str(storage_control.get("severity") or "").strip().lower() == "stable"
    )
    data_clean = bool(
        _to_int(data_integrity.get("sql_overlay_invalid_lines"), 0) <= 0
        and _to_int(data_integrity.get("sql_overlay_oversize_payloads"), 0) <= 0
        and _to_int(data_integrity.get("sql_overlay_ops_write_failures"), 0) <= 0
    )
    total_pending = _to_int(effective.get("total_pending_lines"), _to_int(backpressure.get("total_pending_lines"), 0))
    core_pending = _to_int(effective.get("core_pending_lines"), _to_int(backpressure.get("core_pending_lines"), total_pending))
    oldest_age = _to_float(effective.get("oldest_pending_age_seconds"), _to_float(backpressure.get("oldest_pending_age_seconds"), 0.0))
    total_target = _to_int(targets.get("total_pending_lines"), _to_int(backpressure.get("total_pending_lines_threshold"), 15000)) or 15000
    core_target = _to_int(targets.get("core_pending_lines"), _to_int(backpressure.get("pending_lines_threshold"), 5000)) or 5000
    oldest_target = _to_float(targets.get("oldest_pending_age_seconds"), 600.0) or 600.0
    queue_clear = bool(total_pending <= total_target and core_pending <= core_target)
    age_reconciled = bool(
        effective.get("age_reconciled_from_stale_locator", False)
        or effective.get("oldest_age_reconciled", False)
        or "fresh_empty_sql" in source
    )
    age_clear = bool(oldest_age <= oldest_target and (oldest_age > 0.0 or age_reconciled or overlay_clear))
    authoritative_clear = bool(storage_ready and overlay_adjusted and overlay_clear)
    effective_queue_clear = bool(queue_clear and age_clear and age_reconciled)
    if not (
        (authoritative_clear or effective_queue_clear)
        and data_clean
        and age_seconds <= STORAGE_CONTROL_BACKPRESSURE_OVERRIDE_MAX_AGE_SECONDS
    ):
        return {
            "active": False,
            "reason": "storage_control_not_authoritative_clear",
            "storage_ready": storage_ready,
            "overlay_adjusted": overlay_adjusted,
            "overlay_clear": overlay_clear,
            "queue_clear": queue_clear,
            "age_clear": age_clear,
            "age_reconciled": age_reconciled,
            "data_clean": data_clean,
            "age_seconds": round(age_seconds, 3),
            "pending_lines": int(core_pending),
            "pending_lines_total": int(total_pending),
            "oldest_pending_age_seconds": round(oldest_age, 3),
        }
    return {
        "active": True,
        "source": source or "ingestion_storage_control_effective_raw_live",
        "age_seconds": round(age_seconds, 3),
        "pending_lines": int(core_pending),
        "pending_lines_total": int(total_pending),
        "oldest_pending_age_seconds": round(oldest_age, 3),
        "overload": False,
        "line_pressure": False,
        "file_pressure": False,
        "age_pressure": False,
        "storage_ready": storage_ready,
        "overlay_adjusted": overlay_adjusted,
        "overlay_clear": overlay_clear,
        "queue_clear": queue_clear,
        "age_clear": age_clear,
        "age_reconciled": age_reconciled,
        "reason": "fresh_sql_overlay_clear" if authoritative_clear else "fresh_storage_control_queue_clear",
    }


def _effective_blocked_rate(data_blocked_rate: float, risk_blocked_rate: float, *, risk_weight: float = 0.25) -> float:
    return min(max(float(data_blocked_rate), 0.0) + (max(float(risk_blocked_rate), 0.0) * max(float(risk_weight), 0.0)), 1.0)


def _priority_shard_tier(shard_name: str) -> str:
    text = str(shard_name or '').strip().lower()
    if any(token in text for token in ('shadow', 'attribution', 'explanation', 'analytics', 'counterfactual', 'research')):
        return 'supporting'
    return 'critical'


def _priority_shard_recommended_action(row: dict) -> str:
    if bool(row.get('storage_breached')) and bool(row.get('latency_breached')):
        return 'force_retention_and_throttle'
    if bool(row.get('storage_breached')):
        return 'force_retention'
    if bool(row.get('latency_breached')):
        return 'drain_backlog'
    if int(row.get('invalid_lines', 0) or 0) > 0:
        return 'quarantine_invalid_records'
    if int(row.get('pending_lines', 0) or 0) > 0:
        return 'monitor_backlog'
    return 'healthy'


def _priority_shard_summary(
    project_root: Path,
    shard_names: List[str],
    sql_link_service: dict,
    *,
    p95_seconds_limit: float,
    breach_ratio_limit: float,
) -> List[dict]:
    hot_retention_rows = sql_link_service.get('shard_hot_retention', []) if isinstance(sql_link_service.get('shard_hot_retention'), list) else []
    retention_by_shard = {
        str(row.get('shard') or '').strip(): row
        for row in hot_retention_rows
        if isinstance(row, dict) and str(row.get('shard') or '').strip()
    }

    rows: List[dict] = []
    for shard_name in shard_names:
        health_path = project_root / 'governance' / 'health' / f'jsonl_sql_ingestion_health_{shard_name}_latest.json'
        health_payload = _load_json(health_path)
        sqlite = health_payload.get('sqlite', {}) if isinstance(health_payload.get('sqlite'), dict) else {}
        latency = (((health_payload.get('latency_slo', {}) or {}).get('sqlite', {}) or {}).get('all', {}) or {})
        retention = retention_by_shard.get(shard_name, {})
        live_db_size_gb = _priority_shard_live_db_size_gb(project_root, shard_name)
        db_size_gb = (
            round(float(live_db_size_gb), 3)
            if live_db_size_gb is not None
            else _to_float(retention.get('db_size_gb_after', retention.get('db_size_gb_before', 0.0)), 0.0)
        )
        max_db_gb = _to_float(retention.get('max_db_gb'), 0.0)
        size_over_max = bool(max_db_gb > 0.0 and db_size_gb >= max_db_gb)
        p95_seconds = _to_float(latency.get('p95_seconds'), 0.0)
        breach_ratio = _to_float(latency.get('slo_breach_ratio_gt_300s'), 0.0)
        latency_breached = bool(
            p95_seconds > float(p95_seconds_limit)
            or breach_ratio > float(breach_ratio_limit)
        )
        retention_debt_gb = round(max(db_size_gb - max_db_gb, 0.0), 3) if max_db_gb > 0.0 else 0.0
        row = {
            'shard': shard_name,
            'tier': _priority_shard_tier(shard_name),
            'source_file': str(health_path) if health_payload else '',
            'health_present': bool(health_payload),
            'pending_lines': _to_int(sqlite.get('pending_lines'), 0),
            'oldest_uningested_age_seconds': _to_float(sqlite.get('oldest_uningested_age_seconds'), 0.0),
            'invalid_lines': _to_int(sqlite.get('invalid'), 0),
            'p95_seconds': p95_seconds,
            'slo_breach_ratio_gt_300s': breach_ratio,
            'latency_breached': latency_breached,
            'latency_limit_multiplier': round((p95_seconds / p95_seconds_limit), 3) if p95_seconds_limit > 0.0 else 0.0,
            'db_size_gb': db_size_gb,
            'max_db_gb': max_db_gb,
            'size_over_max': size_over_max,
            'storage_breached': size_over_max,
            'retention_debt_gb': retention_debt_gb,
            'size_over_max_ratio': round((db_size_gb / max_db_gb), 3) if max_db_gb > 0.0 else 0.0,
            'retention_trigger_reasons': list(retention.get('trigger_reasons') or []) if isinstance(retention, dict) else [],
            'retention_skipped_reason': str(retention.get('skipped_reason') or '') if isinstance(retention, dict) else '',
        }
        row['recommended_action'] = _priority_shard_recommended_action(row)
        rows.append(row)
    return rows


def _recommended_operating_mode(
    *,
    hard_gate_triggered: bool,
    gate_sql_progress_stall: bool,
    gate_sql_wal_pressure: bool,
    gate_priority_shard_storage: bool,
    gate_priority_shard_latency: bool,
    gate_backpressure_overload: bool,
    oversized_priority_shard_count: int,
    critical_priority_failures: List[str],
) -> str:
    if gate_sql_progress_stall or gate_sql_wal_pressure or oversized_priority_shard_count >= 2:
        return 'maintenance_only'
    if gate_priority_shard_storage or gate_backpressure_overload or critical_priority_failures:
        return 'shadow_only'
    if hard_gate_triggered or gate_priority_shard_latency:
        return 'live_cautious'
    return 'live_full'


def _recommendations(
    *,
    hard_gate_triggered: bool,
    gate_ingest_invalid: bool,
    gate_sql_progress_stall: bool,
    gate_sql_wal_pressure: bool,
    gate_backpressure_overload: bool,
    priority_shard_latency_failures: List[str],
    priority_shard_storage_failures: List[str],
    collector_required_failures: List[str],
) -> List[str]:
    notes: List[str] = []
    if priority_shard_storage_failures:
        notes.append('force_priority_shard_retention')
    if priority_shard_latency_failures:
        notes.append('throttle_shadow_and_research_ingestion')
    if gate_backpressure_overload:
        notes.append('pause_new_shadow_writers_until_backlog_recovers')
    if gate_sql_progress_stall or gate_sql_wal_pressure:
        notes.append('checkpoint_primary_sqlite_wal_and_review_writer_lock')
    if gate_ingest_invalid:
        notes.append('quarantine_invalid_jsonl_records')
    if collector_required_failures:
        notes.append('repair_required_collectors_before_resuming_full_load')
    if not notes and not hard_gate_triggered:
        notes.append('continue_live_full_operation')
    return notes


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
    parser.add_argument(
        '--priority-shards',
        default=os.getenv('HEALTH_GATE_PRIORITY_SHARDS', 'crypto_explanations,explanations,crypto_shadow_attribution,shadow_attribution'),
        help='Comma-separated shard names to treat as first-class ingestion/storage gates.',
    )
    parser.add_argument(
        '--priority-shard-p95-seconds-limit',
        type=float,
        default=float(os.getenv('HEALTH_GATE_PRIORITY_SHARD_P95_SECONDS_LIMIT', '300')),
    )
    parser.add_argument(
        '--priority-shard-breach-ratio-limit',
        type=float,
        default=float(os.getenv('HEALTH_GATE_PRIORITY_SHARD_BREACH_RATIO_LIMIT', '0.05')),
    )
    parser.add_argument(
        '--priority-shard-storage-over-max-fails',
        action='store_true',
        default=_as_bool(os.getenv('HEALTH_GATE_PRIORITY_SHARD_STORAGE_OVER_MAX_FAILS', '1')),
    )
    parser.add_argument(
        '--priority-shard-latency-fails',
        action='store_true',
        default=_as_bool(os.getenv('HEALTH_GATE_PRIORITY_SHARD_LATENCY_FAILS', '1')),
    )
    parser.add_argument(
        '--collector-contract-failures-fail',
        action='store_true',
        default=_as_bool(os.getenv('HEALTH_GATE_COLLECTOR_CONTRACT_FAILURES_FAIL', '1')),
    )
    parser.add_argument(
        '--sql-progress-idle-seconds-limit',
        type=int,
        default=int(os.getenv('HEALTH_GATE_SQL_PROGRESS_IDLE_SECONDS_LIMIT', '5400')),
    )
    parser.add_argument(
        '--sql-wal-size-gb-limit',
        type=float,
        default=float(os.getenv('HEALTH_GATE_SQL_WAL_SIZE_GB_LIMIT', '24')),
    )
    parser.add_argument(
        '--sql-progress-stall-fails',
        action='store_true',
        default=_as_bool(os.getenv('HEALTH_GATE_SQL_PROGRESS_STALL_FAILS', '1')),
    )
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    health_root = project_root / 'governance' / 'health'
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
        health_root / 'ingestion_backpressure_latest.json',
    ]
    backpressure, backpressure_source = _freshest_non_empty_json(backpressure_paths)
    storage_control_path = health_root / 'ingestion_storage_control_latest.json'
    storage_control = _load_json(storage_control_path)
    backpressure_override = _storage_control_backpressure_override(storage_control)
    if bool(backpressure_override.get("active", False)):
        raw_backpressure = dict(backpressure)
        backpressure = {
            **backpressure,
            "overload": bool(backpressure_override.get("overload", False)),
            "line_pressure": bool(backpressure_override.get("line_pressure", False)),
            "file_pressure": bool(backpressure_override.get("file_pressure", False)),
            "age_pressure": bool(backpressure_override.get("age_pressure", False)),
            "pending_lines": _to_int(backpressure_override.get("pending_lines"), 0),
            "pending_lines_total": _to_int(backpressure_override.get("pending_lines_total"), 0),
            "oldest_pending_age_seconds": _to_float(backpressure_override.get("oldest_pending_age_seconds"), 0.0),
            "storage_control_override": backpressure_override,
            "raw_backpressure_estimate": {
                "pending_lines": _to_int(raw_backpressure.get("pending_lines"), 0),
                "pending_lines_total": _to_int(raw_backpressure.get("pending_lines_total"), 0),
                "oldest_pending_age_seconds": _to_float(raw_backpressure.get("oldest_pending_age_seconds"), 0.0),
                "overload": bool(raw_backpressure.get("overload", False)),
            },
        }
        backpressure_source = f"{backpressure_source} + {storage_control_path}"
    sql_link_service_path = health_root / 'sql_link_service_latest.json'
    sql_link_service = _load_json(sql_link_service_path)
    priority_shards = _priority_shard_summary(
        project_root,
        _parse_csv(args.priority_shards),
        sql_link_service,
        p95_seconds_limit=float(args.priority_shard_p95_seconds_limit),
        breach_ratio_limit=float(args.priority_shard_breach_ratio_limit),
    )
    collector_contracts_path = project_root / 'governance' / 'health' / 'collector_contracts_latest.json'
    collector_contracts = _load_json(collector_contracts_path)
    storage_tier_policy_path = project_root / 'governance' / 'health' / 'storage_tier_policy_latest.json'
    storage_tier_policy = _load_json(storage_tier_policy_path)
    sql_progress_path = project_root / 'governance' / 'health' / 'sql_link_service_progress_latest.json'
    sql_progress = _load_json(sql_progress_path)
    primary_db_path = Path(str(sql_progress.get('primary_db') or sql_link_service.get('primary_db') or '')).expanduser()
    sql_progress_age_seconds = 0.0
    sql_progress_timestamp = _parse_iso_utc(sql_progress.get('timestamp_utc'))
    if sql_progress_timestamp is not None:
        sql_progress_age_seconds = max((datetime.now(timezone.utc) - sql_progress_timestamp).total_seconds(), 0.0)
    sql_wal_size_gb_live = _path_size_gb(Path(str(primary_db_path) + '-wal')) if str(primary_db_path) else 0.0

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
    collector_required_failures = [
        str(name).strip()
        for name in (collector_contracts.get('required_failures') or [])
        if str(name).strip()
    ]
    collector_soft_failures = [
        str(name).strip()
        for name in (collector_contracts.get('soft_failures') or [])
        if str(name).strip()
    ]
    priority_shard_latency_failures = [row['shard'] for row in priority_shards if bool(row.get('latency_breached'))]
    priority_shard_storage_failures = [row['shard'] for row in priority_shards if bool(row.get('storage_breached'))]
    critical_priority_latency_failures = [
        row['shard']
        for row in priority_shards
        if row.get('tier') == 'critical' and bool(row.get('latency_breached'))
    ]
    critical_priority_storage_failures = [
        row['shard']
        for row in priority_shards
        if row.get('tier') == 'critical' and bool(row.get('storage_breached'))
    ]
    critical_priority_failures = [
        row['shard']
        for row in priority_shards
        if row.get('tier') == 'critical' and (bool(row.get('latency_breached')) or bool(row.get('storage_breached')))
    ]
    oversized_priority_shard_count = sum(
        1 for row in priority_shards if row.get('tier') == 'critical' and bool(row.get('storage_breached'))
    )
    retention_debt_gb = round(sum(float(row.get('retention_debt_gb', 0.0) or 0.0) for row in priority_shards), 3)
    worst_priority_latency_multiplier = max(
        (float(row.get('latency_limit_multiplier', 0.0) or 0.0) for row in priority_shards),
        default=0.0,
    )

    gate_stale = stale_windows > args.stale_window_limit
    gate_blocked = blocked_rate > args.blocked_rate_limit
    gate_restarts = watchdog_restarts > args.watchdog_restarts_limit

    gate_ingest_pending = ingest_pending_lines > int(args.ingestion_pending_lines_limit)
    gate_ingest_oldest_age = ingest_oldest_age_s > float(args.ingestion_oldest_age_seconds_limit)
    gate_ingest_invalid = ingest_invalid_lines > int(args.ingestion_invalid_lines_limit)
    severe_backpressure_overload = bool(
        backpressure_overload
        and (
            backpressure_pending_lines > int(args.ingestion_pending_lines_limit)
            or backpressure_oldest_age_s > float(args.ingestion_oldest_age_seconds_limit)
            or bool(critical_priority_failures)
        )
    )
    gate_backpressure_overload = bool(severe_backpressure_overload and args.ingestion_backpressure_overload_fails)
    gate_priority_shard_latency = bool(critical_priority_latency_failures and args.priority_shard_latency_fails)
    gate_priority_shard_storage = bool(critical_priority_storage_failures and args.priority_shard_storage_over_max_fails)
    gate_collector_contracts = bool(collector_required_failures and args.collector_contract_failures_fail)
    gate_sql_progress_stall = bool(
        args.sql_progress_stall_fails
        and bool(sql_progress.get('running', False))
        and sql_progress_age_seconds > float(args.sql_progress_idle_seconds_limit)
    )
    gate_sql_wal_pressure = bool(sql_wal_size_gb_live > float(args.sql_wal_size_gb_limit))

    score = 100.0
    score -= min(blocked_rate * 100.0 * 0.35, 35.0)
    score -= min(stale_windows * 8.0, 32.0)
    score -= min(watchdog_restarts * 7.0, 21.0)
    score -= min((ingest_pending_lines / 1000.0) * 0.8, 8.0)
    score -= min((ingest_oldest_age_s / 60.0) * 0.7, 7.0)
    score -= min(max(ingest_invalid_lines, 0) * 0.25, 5.0)
    if backpressure_overload:
        score -= 4.0
    score -= min(len(priority_shard_latency_failures) * 4.0, 12.0)
    score -= min(len(priority_shard_storage_failures) * 3.0, 9.0)
    score -= min(len(collector_required_failures) * 5.0, 10.0)
    score -= min(len(collector_soft_failures) * 1.5, 6.0)
    if gate_sql_progress_stall:
        score -= 7.0
    if gate_sql_wal_pressure:
        score -= min((sql_wal_size_gb_live / max(float(args.sql_wal_size_gb_limit), 1.0)) * 4.0, 8.0)
    score = max(score, 0.0)

    hard_gate_triggered = bool(
        gate_stale
        or gate_blocked
        or gate_restarts
        or gate_ingest_pending
        or gate_ingest_oldest_age
        or gate_ingest_invalid
        or gate_backpressure_overload
        or gate_priority_shard_latency
        or gate_priority_shard_storage
        or gate_collector_contracts
        or gate_sql_progress_stall
        or gate_sql_wal_pressure
    )
    recommended_operating_mode = _recommended_operating_mode(
        hard_gate_triggered=hard_gate_triggered,
        gate_sql_progress_stall=gate_sql_progress_stall,
        gate_sql_wal_pressure=gate_sql_wal_pressure,
        gate_priority_shard_storage=gate_priority_shard_storage,
        gate_priority_shard_latency=gate_priority_shard_latency,
        gate_backpressure_overload=gate_backpressure_overload,
        oversized_priority_shard_count=oversized_priority_shard_count,
        critical_priority_failures=critical_priority_failures,
    )
    recommendations = _recommendations(
        hard_gate_triggered=hard_gate_triggered,
        gate_ingest_invalid=gate_ingest_invalid,
        gate_sql_progress_stall=gate_sql_progress_stall,
        gate_sql_wal_pressure=gate_sql_wal_pressure,
        gate_backpressure_overload=gate_backpressure_overload,
        priority_shard_latency_failures=priority_shard_latency_failures,
        priority_shard_storage_failures=priority_shard_storage_failures,
        collector_required_failures=collector_required_failures,
    )

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'data_quality_score': round(score, 2),
        'source_files': {
            'one_numbers': one_numbers_source,
            'daily_runtime_summary': daily_summary_source,
            'jsonl_sql_ingestion_health': ingestion_health_source,
            'ingestion_backpressure': backpressure_source,
            'sql_link_service': str(sql_link_service_path) if sql_link_service else '',
            'collector_contracts': str(collector_contracts_path) if collector_contracts else '',
            'storage_tier_policy': str(storage_tier_policy_path) if storage_tier_policy else '',
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
            'backpressure_overload_severe': severe_backpressure_overload,
            'backpressure_pending_lines': backpressure_pending_lines,
            'backpressure_oldest_pending_age_seconds': backpressure_oldest_age_s,
            'priority_shard_latency_failures': priority_shard_latency_failures,
            'priority_shard_storage_failures': priority_shard_storage_failures,
            'critical_priority_shard_latency_failures': critical_priority_latency_failures,
            'critical_priority_shard_storage_failures': critical_priority_storage_failures,
            'critical_priority_failures': critical_priority_failures,
            'collector_required_failures': collector_required_failures,
            'collector_soft_failures': collector_soft_failures,
            'sql_progress_status': str(sql_progress.get('status') or ''),
            'sql_progress_step': str(sql_progress.get('current_step') or ''),
            'sql_progress_age_seconds': round(sql_progress_age_seconds, 3),
            'sql_wal_size_gb_live': round(sql_wal_size_gb_live, 3),
            'backpressure_storage_control_override': backpressure_override,
        },
        'priority_shards': priority_shards,
        'recommended_operating_mode': recommended_operating_mode,
        'recommendations': recommendations,
        'storage_pressure': {
            'oversized_priority_shard_count': oversized_priority_shard_count,
            'retention_debt_gb': retention_debt_gb,
            'worst_priority_latency_multiplier': round(worst_priority_latency_multiplier, 3),
        },
        'ingestion_pressure': {
            'severe_backpressure_overload': severe_backpressure_overload,
            'priority_shard_latency_failures': priority_shard_latency_failures,
            'priority_shard_storage_failures': priority_shard_storage_failures,
            'critical_priority_shard_latency_failures': critical_priority_latency_failures,
            'critical_priority_shard_storage_failures': critical_priority_storage_failures,
            'critical_priority_failures': critical_priority_failures,
        },
        'hard_gates': {
            'stale_windows': gate_stale,
            'blocked_rate': gate_blocked,
            'watchdog_restart_spike': gate_restarts,
            'ingestion_pending_lines': gate_ingest_pending,
            'ingestion_oldest_age': gate_ingest_oldest_age,
            'ingestion_invalid_lines': gate_ingest_invalid,
            'ingestion_backpressure_overload': gate_backpressure_overload,
            'priority_shard_latency': gate_priority_shard_latency,
            'priority_shard_storage': gate_priority_shard_storage,
            'collector_contracts': gate_collector_contracts,
            'sql_progress_stall': gate_sql_progress_stall,
            'sql_wal_pressure': gate_sql_wal_pressure,
        },
        'thresholds': {
            'stale_window_limit': int(args.stale_window_limit),
            'blocked_rate_limit': float(args.blocked_rate_limit),
            'watchdog_restarts_limit': int(args.watchdog_restarts_limit),
            'ingestion_pending_lines_limit': int(args.ingestion_pending_lines_limit),
            'ingestion_oldest_age_seconds_limit': int(args.ingestion_oldest_age_seconds_limit),
            'ingestion_invalid_lines_limit': int(args.ingestion_invalid_lines_limit),
            'ingestion_backpressure_overload_fails': bool(args.ingestion_backpressure_overload_fails),
            'priority_shards': _parse_csv(args.priority_shards),
            'priority_shard_p95_seconds_limit': float(args.priority_shard_p95_seconds_limit),
            'priority_shard_breach_ratio_limit': float(args.priority_shard_breach_ratio_limit),
            'priority_shard_storage_over_max_fails': bool(args.priority_shard_storage_over_max_fails),
            'priority_shard_latency_fails': bool(args.priority_shard_latency_fails),
            'collector_contract_failures_fail': bool(args.collector_contract_failures_fail),
            'sql_progress_idle_seconds_limit': int(args.sql_progress_idle_seconds_limit),
            'sql_wal_size_gb_limit': float(args.sql_wal_size_gb_limit),
            'sql_progress_stall_fails': bool(args.sql_progress_stall_fails),
        },
        'hard_gate_triggered': hard_gate_triggered,
    }

    out = project_root / 'governance' / 'health' / 'health_gates_latest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
    stability_out = project_root / 'governance' / 'health' / 'system_stability_latest.json'
    stability_payload = {
        'timestamp_utc': payload['timestamp_utc'],
        'safe_operating_envelope': not hard_gate_triggered,
        'hard_gate_triggered': hard_gate_triggered,
        'recommended_operating_mode': recommended_operating_mode,
        'recommendations': recommendations,
        'critical_issues': [name for name, triggered in payload['hard_gates'].items() if bool(triggered)],
        'priority_shards': priority_shards,
        'collector_contracts': {
            'required_failures': collector_required_failures,
            'soft_failures': collector_soft_failures,
        },
        'storage_pressure': {
            'oversized_shards': priority_shard_storage_failures,
            'retention_debt_gb': retention_debt_gb,
            'worst_size_over_max_ratio': max((float(row.get('size_over_max_ratio', 0.0) or 0.0) for row in priority_shards), default=0.0),
            'tier_policy': storage_tier_policy.get('pressure', {}) if isinstance(storage_tier_policy, dict) else {},
        },
        'ingestion_pressure': {
            'backpressure_overload': backpressure_overload,
            'severe_backpressure_overload': severe_backpressure_overload,
            'pending_lines': backpressure_pending_lines,
            'oldest_pending_age_seconds': backpressure_oldest_age_s,
            'priority_shard_latency_failures': priority_shard_latency_failures,
            'priority_shard_storage_failures': priority_shard_storage_failures,
        },
        'sql_pressure': {
            'status': str(sql_progress.get('status') or ''),
            'current_step': str(sql_progress.get('current_step') or ''),
            'progress_age_seconds': round(sql_progress_age_seconds, 3),
            'wal_size_gb_live': round(sql_wal_size_gb_live, 3),
            'progress_stalled': gate_sql_progress_stall,
            'wal_pressure': gate_sql_wal_pressure,
        },
    }
    stability_out.write_text(json.dumps(stability_payload, ensure_ascii=True, indent=2), encoding='utf-8')

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
