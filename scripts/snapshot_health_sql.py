import hashlib
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


SNAPSHOT_FILE_MAP = {
    "snapshot_coverage": "snapshot_coverage_latest.json",
    "replay_preopen_sanity": "replay_preopen_sanity_latest.json",
    "preopen_replay_drift": "preopen_replay_drift_latest.json",
    "data_source_divergence": "data_source_divergence_latest.json",
    "guardrail_triprate": "guardrail_triprate_latest.json",
    "execution_queue_stress": "execution_queue_stress_latest.json",
}


def _safe_load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else default
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_sqlite_path(project_root: Path) -> Path:
    return project_root / "data" / "jsonl_link.sqlite3"


def _sqlite_has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return bool(row)


def _ensure_snapshot_health_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshot_health_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_key TEXT NOT NULL,
            timestamp_utc TEXT,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            source_path TEXT,
            ingested_at TEXT NOT NULL,
            UNIQUE(metric_key, payload_sha1)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_snapshot_health_metric_ts ON snapshot_health_records(metric_key, timestamp_utc)"
    )


def load_snapshot_health_payloads_from_files(project_root: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    health_root = project_root / "governance" / "health"
    payloads: Dict[str, Dict[str, Any]] = {}
    paths: Dict[str, str] = {}
    for key, filename in SNAPSHOT_FILE_MAP.items():
        p = health_root / filename
        payloads[key] = _safe_load_json(p, default={})
        paths[key] = str(p)
    return payloads, paths


def sync_snapshot_health_to_sqlite(
    *,
    project_root: Path,
    sqlite_path: Optional[Path] = None,
    payloads: Optional[Dict[str, Dict[str, Any]]] = None,
    source_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    db_path = sqlite_path if sqlite_path is not None else _default_sqlite_path(project_root)
    db_path = Path(db_path).expanduser().resolve()

    if payloads is None:
        payloads, source_paths = load_snapshot_health_payloads_from_files(project_root)
    if source_paths is None:
        source_paths = {}

    rows = []
    for metric_key in SNAPSHOT_FILE_MAP:
        obj = payloads.get(metric_key) or {}
        if not isinstance(obj, dict) or not obj:
            continue
        payload_json = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
        payload_sha1 = hashlib.sha1(payload_json.encode("utf-8")).hexdigest()
        timestamp_utc = str(obj.get("timestamp_utc") or "")
        rows.append(
            (
                metric_key,
                timestamp_utc,
                payload_sha1,
                payload_json,
                str(source_paths.get(metric_key, "")),
                _now_utc_iso(),
            )
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _ensure_snapshot_health_schema(conn)

        inserted = 0
        if rows:
            before = conn.total_changes
            conn.executemany(
                """
                INSERT OR IGNORE INTO snapshot_health_records
                (metric_key, timestamp_utc, payload_sha1, payload_json, source_path, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            inserted = max(conn.total_changes - before, 0)
            conn.commit()

        total = conn.execute("SELECT COUNT(*) FROM snapshot_health_records").fetchone()
        return {
            "db_path": str(db_path),
            "rows_attempted": int(len(rows)),
            "rows_inserted": int(inserted),
            "table_rows": int(total[0]) if total else 0,
        }
    finally:
        conn.close()


def load_snapshot_health_payloads_from_sqlite(sqlite_path: Path) -> Dict[str, Dict[str, Any]]:
    db_path = Path(sqlite_path).expanduser().resolve()
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    try:
        if not _sqlite_has_table(conn, "snapshot_health_records"):
            return {}

        rows = conn.execute(
            """
            SELECT t.metric_key, t.payload_json
            FROM snapshot_health_records t
            JOIN (
                SELECT metric_key, MAX(id) AS max_id
                FROM snapshot_health_records
                GROUP BY metric_key
            ) latest
                ON latest.metric_key = t.metric_key
               AND latest.max_id = t.id
            """
        ).fetchall()

        out: Dict[str, Dict[str, Any]] = {}
        for metric_key, payload_json in rows:
            try:
                obj = json.loads(payload_json)
            except Exception:
                continue
            if isinstance(obj, dict):
                out[str(metric_key)] = obj
        return out
    finally:
        conn.close()


def compute_snapshot_health_context(payloads: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    coverage = payloads.get("snapshot_coverage") or {}
    replay = payloads.get("replay_preopen_sanity") or {}
    drift = payloads.get("preopen_replay_drift") or {}
    divergence = payloads.get("data_source_divergence") or {}
    triprate = payloads.get("guardrail_triprate") or {}
    queue_stress = payloads.get("execution_queue_stress") or {}

    coverage_ratio = _to_float(coverage.get("coverage_ratio"), 0.0)
    coverage_log_ratio = _clamp01(math.log1p(max(coverage_ratio, 0.0)) / 6.0)
    rows_scanned = max(_to_float(coverage.get("rows_scanned"), 0.0), 1.0)
    rows_with_sid = _to_float(coverage.get("rows_with_snapshot_id"), 0.0)
    coverage_fill_ratio = _clamp01(rows_with_sid / rows_scanned)

    replay_decision_stale = _to_float((replay.get("decision") or {}).get("stale_windows"), 0.0)
    replay_governance_stale = _to_float((replay.get("governance") or {}).get("stale_windows"), 0.0)
    replay_max_decision_stale = max(_to_float((replay.get("thresholds") or {}).get("max_decision_stale_windows"), 12.0), 1.0)
    replay_max_governance_stale = max(_to_float((replay.get("thresholds") or {}).get("max_governance_stale_windows"), 12.0), 1.0)
    replay_stale_ratio = _clamp01((replay_decision_stale + replay_governance_stale) / (replay_max_decision_stale + replay_max_governance_stale))

    drift_obj = drift.get("drift") or {}
    thresholds_obj = drift.get("thresholds") or {}
    row_drift = max(abs(_to_float(drift_obj.get("decision_rows"), 0.0)), abs(_to_float(drift_obj.get("governance_rows"), 0.0)))
    stale_drift = max(abs(_to_float(drift_obj.get("decision_stale"), 0.0)), abs(_to_float(drift_obj.get("governance_stale"), 0.0)))
    max_row_drift = max(_to_float(thresholds_obj.get("max_row_drift"), 1.2), 1e-6)
    max_stale_drift = max(_to_float(thresholds_obj.get("max_stale_drift"), 1.0), 1e-6)
    replay_drift_ratio = _clamp01((0.6 * (row_drift / max_row_drift)) + (0.4 * (stale_drift / max_stale_drift)))

    worst_spread = _to_float(divergence.get("worst_relative_spread"), 0.0)
    max_spread = max(_to_float(divergence.get("max_relative_spread"), 0.03), 1e-6)
    divergence_ratio = _clamp01(worst_spread / max_spread)

    trip_rate = _to_float(triprate.get("trip_rate"), 0.0)
    max_trip_rate = max(_to_float(triprate.get("max_trip_rate"), 0.4), 1e-6)
    triprate_ratio = _clamp01(trip_rate / max_trip_rate)

    depth_seen = _to_float(queue_stress.get("max_queue_depth_seen"), 0.0)
    depth_max = max(_to_float(queue_stress.get("max_queue_depth"), 2000.0), 1.0)
    depth_ratio = _clamp01(depth_seen / depth_max)
    breach_rate = _to_float(queue_stress.get("queue_breach_rate"), 0.0)
    breach_rate_max = max(_to_float(queue_stress.get("max_queue_breach_rate"), 0.25), 1e-6)
    breach_ratio = _clamp01(breach_rate / breach_rate_max)
    queue_pressure_ratio = _clamp01(max(depth_ratio, breach_ratio))

    canary_weight_cap_norm = _clamp01(_to_float(os.getenv("CANARY_MAX_WEIGHT", "0.08"), 0.08) / 0.20)

    context = {
        "snapshot_cov_ok": 1.0 if bool(coverage.get("ok", False)) else 0.0,
        "snapshot_cov_log_ratio": coverage_log_ratio,
        "snapshot_cov_fill_ratio": coverage_fill_ratio,
        "snapshot_replay_ok": 1.0 if bool(replay.get("ok", False)) else 0.0,
        "snapshot_replay_stale_ratio": replay_stale_ratio,
        "snapshot_replay_drift_ratio": replay_drift_ratio,
        "snapshot_divergence_ratio": divergence_ratio,
        "snapshot_triprate_ratio": triprate_ratio,
        "snapshot_queue_pressure_ratio": queue_pressure_ratio,
        "canary_weight_cap_norm": canary_weight_cap_norm,
    }

    meta = {
        "coverage_ts": coverage.get("timestamp_utc"),
        "replay_ts": replay.get("timestamp_utc"),
        "drift_ts": drift.get("timestamp_utc"),
        "divergence_ts": divergence.get("timestamp_utc"),
        "triprate_ts": triprate.get("timestamp_utc"),
        "queue_stress_ts": queue_stress.get("timestamp_utc"),
    }
    return context, meta


def load_snapshot_context(
    *,
    project_root: Path,
    sqlite_path: Optional[Path] = None,
    prefer_sql: bool = True,
    persist_files_to_sql: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    file_payloads, file_paths = load_snapshot_health_payloads_from_files(project_root)

    sync_meta: Dict[str, Any] = {}
    db_path = sqlite_path if sqlite_path is not None else _default_sqlite_path(project_root)
    db_path = Path(db_path).expanduser().resolve()

    if persist_files_to_sql:
        try:
            sync_meta = sync_snapshot_health_to_sqlite(
                project_root=project_root,
                sqlite_path=db_path,
                payloads=file_payloads,
                source_paths=file_paths,
            )
        except Exception as exc:
            sync_meta = {"error": str(exc), "db_path": str(db_path)}

    sql_payloads: Dict[str, Dict[str, Any]] = {}
    try:
        sql_payloads = load_snapshot_health_payloads_from_sqlite(db_path)
    except Exception:
        sql_payloads = {}

    merged: Dict[str, Dict[str, Any]] = {}
    selected_source: Dict[str, str] = {}
    for key in SNAPSHOT_FILE_MAP:
        file_obj = file_payloads.get(key) or {}
        sql_obj = sql_payloads.get(key) or {}
        if prefer_sql:
            if sql_obj:
                merged[key] = sql_obj
                selected_source[key] = "sql"
            elif file_obj:
                merged[key] = file_obj
                selected_source[key] = "file"
            else:
                merged[key] = {}
                selected_source[key] = "none"
        else:
            if file_obj:
                merged[key] = file_obj
                selected_source[key] = "file"
            elif sql_obj:
                merged[key] = sql_obj
                selected_source[key] = "sql"
            else:
                merged[key] = {}
                selected_source[key] = "none"

    context, calc_meta = compute_snapshot_health_context(merged)
    source_values = set(selected_source.values())
    if source_values == {"sql"}:
        source_mode = "sql"
    elif source_values == {"file"}:
        source_mode = "file"
    elif source_values == {"none"}:
        source_mode = "none"
    else:
        source_mode = "mixed"

    meta = dict(calc_meta)
    meta.update(
        {
            "source_mode": source_mode,
            "prefer_sql": bool(prefer_sql),
            "sqlite_path": str(db_path),
            "selected_source": selected_source,
            "file_paths": file_paths,
            "sql_sync": sync_meta,
        }
    )
    return context, meta
