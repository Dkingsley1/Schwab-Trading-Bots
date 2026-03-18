import gzip
import hashlib
import json
import math
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


SNAPSHOT_DRILL_KEY = "state_snapshot_drill"
SNAPSHOT_DRILL_LATEST_REL = Path("exports") / "state_snapshot_drills" / "latest.json"

SNAPSHOT_FILE_MAP = {
    "snapshot_coverage": "snapshot_coverage_latest.json",
    "replay_preopen_sanity": "replay_preopen_sanity_latest.json",
    "preopen_replay_drift": "preopen_replay_drift_latest.json",
    "data_source_divergence": "data_source_divergence_latest.json",
    "guardrail_triprate": "guardrail_triprate_latest.json",
    "execution_queue_stress": "execution_queue_stress_latest.json",
}

DEBUG_SNAPSHOT_ROOT_REL = Path("exports") / "debug_snapshots"
DEBUG_SNAPSHOT_DIR_RE = re.compile(r"^\d{8}_\d{6}$")

RAW_DEBUG_CONTEXT_KEYS = [
    "snapshot_raw_sql_ingest_ratio",
    "snapshot_raw_count_norm",
    "snapshot_raw_file_count_norm",
    "snapshot_raw_bytes_norm",
    "snapshot_raw_json_ratio",
    "snapshot_raw_event_file_ratio",
    "snapshot_raw_lock_file_ratio",
    "snapshot_raw_recency_norm",
]


def _metric_keys() -> list[str]:
    return list(SNAPSHOT_FILE_MAP.keys()) + [SNAPSHOT_DRILL_KEY]


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


def _parse_ts_utc(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    s = str(raw).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _default_sqlite_path(project_root: Path) -> Path:
    return project_root / "data" / "snapshot_context.sqlite3"


def _sqlite_has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return bool(row)


def _connect_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=120.0)
    try:
        conn.execute("PRAGMA busy_timeout=120000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    return conn


def _default_raw_debug_context() -> Dict[str, float]:
    return {k: 0.0 for k in RAW_DEBUG_CONTEXT_KEYS}


def _parse_snapshot_id_utc(snapshot_id: str) -> Optional[datetime]:
    s = str(snapshot_id or "").strip()
    if not DEBUG_SNAPSHOT_DIR_RE.match(s):
        return None
    try:
        dt = datetime.strptime(s, "%Y%m%d_%H%M%S")
    except Exception:
        return None
    return dt.replace(tzinfo=timezone.utc)


def _iter_debug_snapshot_dirs(project_root: Path) -> list[Path]:
    root = project_root / DEBUG_SNAPSHOT_ROOT_REL
    if not root.exists():
        return []

    out: list[Path] = []
    for child in root.iterdir():
        if child.name == "latest":
            continue
        if not child.is_dir():
            continue
        if not DEBUG_SNAPSHOT_DIR_RE.match(child.name):
            continue
        out.append(child)

    out.sort(key=lambda p: p.name)
    return out


def _snapshot_files(snapshot_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in snapshot_dir.rglob("*"):
        if p.is_file() and not p.is_symlink():
            files.append(p)
    files.sort(key=lambda p: str(p))
    return files


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


def _ensure_debug_snapshot_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS debug_snapshot_file_blobs (
            payload_sha1 TEXT PRIMARY KEY,
            raw_size_bytes INTEGER NOT NULL,
            stored_size_bytes INTEGER NOT NULL,
            payload_gzip BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS debug_snapshot_raw_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL,
            snapshot_rel TEXT NOT NULL,
            file_rel TEXT NOT NULL,
            source_path TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            raw_size_bytes INTEGER NOT NULL,
            mtime_utc TEXT,
            captured_ts_utc TEXT,
            ingested_at TEXT NOT NULL,
            UNIQUE(snapshot_id, file_rel, payload_sha1)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_debug_snapshot_records_snapshot ON debug_snapshot_raw_records(snapshot_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_debug_snapshot_records_captured ON debug_snapshot_raw_records(captured_ts_utc)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_debug_snapshot_records_sha1 ON debug_snapshot_raw_records(payload_sha1)"
    )


def load_snapshot_health_payloads_from_files(project_root: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    health_root = project_root / "governance" / "health"
    payloads: Dict[str, Dict[str, Any]] = {}
    paths: Dict[str, str] = {}
    for key, filename in SNAPSHOT_FILE_MAP.items():
        p = health_root / filename
        payloads[key] = _safe_load_json(p, default={})
        paths[key] = str(p)

    drill_path = project_root / SNAPSHOT_DRILL_LATEST_REL
    payloads[SNAPSHOT_DRILL_KEY] = _safe_load_json(drill_path, default={})
    paths[SNAPSHOT_DRILL_KEY] = str(drill_path)
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
    for metric_key in _metric_keys():
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
    conn = _connect_sqlite(db_path)
    try:
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


def sync_raw_debug_snapshots_to_sqlite(
    *,
    project_root: Path,
    sqlite_path: Optional[Path] = None,
    snapshot_dirs: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    db_path = sqlite_path if sqlite_path is not None else _default_sqlite_path(project_root)
    db_path = Path(db_path).expanduser().resolve()

    dirs = [Path(p).resolve() for p in snapshot_dirs] if snapshot_dirs is not None else _iter_debug_snapshot_dirs(project_root)

    store_content = os.getenv("SNAPSHOT_RAW_SQL_STORE_CONTENT", "1").strip() == "1"
    fast_skip = os.getenv("SNAPSHOT_RAW_SYNC_FAST_SKIP", "1").strip() == "1"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect_sqlite(db_path)

    dirs_seen = 0
    files_seen = 0
    skipped_fast = 0
    record_rows_inserted = 0
    blob_rows_inserted = 0
    error_count = 0

    try:
        _ensure_debug_snapshot_schema(conn)

        existing_counts: Dict[str, int] = {}
        if dirs:
            ids = [p.name for p in dirs]
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"SELECT snapshot_id, COUNT(*) FROM debug_snapshot_raw_records WHERE snapshot_id IN ({placeholders}) GROUP BY snapshot_id",
                ids,
            ).fetchall()
            existing_counts = {str(k): int(v) for k, v in rows}

        for snap_dir in dirs:
            if (not snap_dir.exists()) or (not snap_dir.is_dir()):
                continue
            if not DEBUG_SNAPSHOT_DIR_RE.match(snap_dir.name):
                continue

            dirs_seen += 1
            snapshot_id = snap_dir.name
            files = _snapshot_files(snap_dir)
            files_seen += len(files)

            if fast_skip and len(files) > 0 and existing_counts.get(snapshot_id, 0) >= len(files):
                skipped_fast += 1
                continue

            captured_dt = _parse_snapshot_id_utc(snapshot_id)
            captured_ts_utc = captured_dt.isoformat() if captured_dt is not None else ""

            try:
                snapshot_rel = str(snap_dir.resolve().relative_to(project_root.resolve()))
            except Exception:
                snapshot_rel = str(snap_dir)

            for fp in files:
                try:
                    raw = fp.read_bytes()
                except Exception:
                    error_count += 1
                    continue

                payload_sha1 = hashlib.sha1(raw).hexdigest()
                raw_size_bytes = int(len(raw))
                try:
                    mtime_utc = datetime.fromtimestamp(fp.stat().st_mtime, tz=timezone.utc).isoformat()
                except Exception:
                    mtime_utc = ""

                try:
                    file_rel = str(fp.resolve().relative_to(snap_dir.resolve()))
                except Exception:
                    file_rel = str(fp.name)

                if store_content:
                    try:
                        payload_gzip = gzip.compress(raw, compresslevel=6)
                    except Exception:
                        error_count += 1
                        continue

                    cur_blob = conn.execute(
                        """
                        INSERT OR IGNORE INTO debug_snapshot_file_blobs
                        (payload_sha1, raw_size_bytes, stored_size_bytes, payload_gzip, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            payload_sha1,
                            raw_size_bytes,
                            int(len(payload_gzip)),
                            sqlite3.Binary(payload_gzip),
                            _now_utc_iso(),
                        ),
                    )
                    if isinstance(cur_blob.rowcount, int) and cur_blob.rowcount > 0:
                        blob_rows_inserted += int(cur_blob.rowcount)

                cur_rec = conn.execute(
                    """
                    INSERT OR IGNORE INTO debug_snapshot_raw_records
                    (snapshot_id, snapshot_rel, file_rel, source_path, payload_sha1, raw_size_bytes, mtime_utc, captured_ts_utc, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        snapshot_rel,
                        file_rel,
                        str(fp),
                        payload_sha1,
                        raw_size_bytes,
                        mtime_utc,
                        captured_ts_utc,
                        _now_utc_iso(),
                    ),
                )
                if isinstance(cur_rec.rowcount, int) and cur_rec.rowcount > 0:
                    record_rows_inserted += int(cur_rec.rowcount)

        conn.commit()

        rec_total = conn.execute("SELECT COUNT(*) FROM debug_snapshot_raw_records").fetchone()
        blob_total = conn.execute("SELECT COUNT(*) FROM debug_snapshot_file_blobs").fetchone()

        return {
            "db_path": str(db_path),
            "dirs_seen": int(dirs_seen),
            "files_seen": int(files_seen),
            "dirs_skipped_fast": int(skipped_fast),
            "record_rows_inserted": int(record_rows_inserted),
            "blob_rows_inserted": int(blob_rows_inserted),
            "record_table_rows": int(rec_total[0]) if rec_total else 0,
            "blob_table_rows": int(blob_total[0]) if blob_total else 0,
            "store_content": bool(store_content),
            "error_count": int(error_count),
        }
    finally:
        conn.close()


def debug_snapshot_ingest_coverage(
    *,
    project_root: Path,
    sqlite_path: Optional[Path] = None,
    snapshot_dirs: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    db_path = sqlite_path if sqlite_path is not None else _default_sqlite_path(project_root)
    db_path = Path(db_path).expanduser().resolve()

    dirs = [Path(p).resolve() for p in snapshot_dirs] if snapshot_dirs is not None else _iter_debug_snapshot_dirs(project_root)
    dirs = [d for d in dirs if d.exists() and d.is_dir() and DEBUG_SNAPSHOT_DIR_RE.match(d.name)]

    rows: list[Dict[str, Any]] = []
    if not dirs:
        return {
            "db_path": str(db_path),
            "snapshot_total": 0,
            "ready_total": 0,
            "coverage_ratio": 1.0,
            "all_ready": True,
            "rows": rows,
        }

    ingested_counts: Dict[str, int] = {}
    if db_path.exists():
        conn = _connect_sqlite(db_path)
        try:
            if _sqlite_has_table(conn, "debug_snapshot_raw_records"):
                ids = [d.name for d in dirs]
                placeholders = ",".join("?" for _ in ids)
                q_rows = conn.execute(
                    f"SELECT snapshot_id, COUNT(*) FROM debug_snapshot_raw_records WHERE snapshot_id IN ({placeholders}) GROUP BY snapshot_id",
                    ids,
                ).fetchall()
                ingested_counts = {str(k): int(v) for k, v in q_rows}
        finally:
            conn.close()

    ready_total = 0
    for d in dirs:
        expected_files = len(_snapshot_files(d))
        ingested_files = int(ingested_counts.get(d.name, 0))
        ready = (expected_files == 0) or (ingested_files >= expected_files)
        if ready:
            ready_total += 1
        rows.append(
            {
                "snapshot_id": d.name,
                "snapshot_path": str(d),
                "expected_files": int(expected_files),
                "ingested_files": int(ingested_files),
                "ready": bool(ready),
            }
        )

    total = len(rows)
    coverage_ratio = (float(ready_total) / float(total)) if total > 0 else 1.0
    return {
        "db_path": str(db_path),
        "snapshot_total": int(total),
        "ready_total": int(ready_total),
        "coverage_ratio": float(coverage_ratio),
        "all_ready": bool(ready_total == total),
        "rows": rows,
    }


def load_snapshot_health_payloads_from_sqlite(sqlite_path: Path) -> Dict[str, Dict[str, Any]]:
    db_path = Path(sqlite_path).expanduser().resolve()
    if not db_path.exists():
        return {}

    conn = _connect_sqlite(db_path)
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


def load_raw_debug_snapshot_context_from_sqlite(
    *,
    sqlite_path: Path,
    project_root: Optional[Path] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    context = _default_raw_debug_context()
    meta: Dict[str, Any] = {
        "lookback_hours": float(max(_to_float(os.getenv("SNAPSHOT_RAW_CONTEXT_LOOKBACK_HOURS", "168"), 168.0), 1.0)),
        "snapshot_count": 0,
        "file_count": 0,
        "byte_count": 0,
        "latest_ts": None,
        "ingest_coverage_ratio": 0.0,
    }

    db_path = Path(sqlite_path).expanduser().resolve()
    if not db_path.exists():
        meta["reason"] = "db_missing"
        return context, meta

    lookback_hours = float(max(_to_float(os.getenv("SNAPSHOT_RAW_CONTEXT_LOOKBACK_HOURS", "168"), 168.0), 1.0))
    meta["lookback_hours"] = lookback_hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    cutoff_iso = cutoff.isoformat()

    conn = _connect_sqlite(db_path)
    try:
        if not _sqlite_has_table(conn, "debug_snapshot_raw_records"):
            meta["reason"] = "table_missing"
            return context, meta

        row = conn.execute(
            """
            SELECT
                COUNT(DISTINCT snapshot_id) AS snapshot_count,
                COUNT(*) AS file_count,
                COALESCE(SUM(raw_size_bytes), 0) AS byte_count,
                COALESCE(SUM(CASE WHEN lower(file_rel) LIKE '%.json' OR lower(file_rel) LIKE '%.jsonl' OR lower(file_rel) LIKE '%.jsonl.gz' THEN 1 ELSE 0 END), 0) AS json_file_count,
                COALESCE(SUM(CASE WHEN lower(file_rel) LIKE '%events%' AND (lower(file_rel) LIKE '%.jsonl' OR lower(file_rel) LIKE '%.jsonl.gz') THEN 1 ELSE 0 END), 0) AS event_file_count,
                COALESCE(SUM(CASE WHEN lower(file_rel) LIKE '%lock%' THEN 1 ELSE 0 END), 0) AS lock_file_count,
                MAX(COALESCE(NULLIF(captured_ts_utc, ''), ingested_at)) AS latest_ts
            FROM debug_snapshot_raw_records
            WHERE COALESCE(NULLIF(captured_ts_utc, ''), ingested_at) >= ?
            """,
            (cutoff_iso,),
        ).fetchone()

        snapshot_count = int(row[0] or 0) if row else 0
        file_count = int(row[1] or 0) if row else 0
        byte_count = int(row[2] or 0) if row else 0
        json_file_count = int(row[3] or 0) if row else 0
        event_file_count = int(row[4] or 0) if row else 0
        lock_file_count = int(row[5] or 0) if row else 0
        latest_ts_raw = row[6] if row else None

        latest_ts = _parse_ts_utc(latest_ts_raw)
        if latest_ts is not None:
            age_hours = max((datetime.now(timezone.utc) - latest_ts).total_seconds() / 3600.0, 0.0)
            recency_norm = 1.0 - _clamp01(age_hours / 72.0)
            latest_ts_iso = latest_ts.isoformat()
        else:
            recency_norm = 0.0
            latest_ts_iso = None

        json_ratio = _clamp01(float(json_file_count) / max(float(file_count), 1.0))
        event_ratio = _clamp01(float(event_file_count) / max(float(file_count), 1.0))
        lock_ratio = _clamp01(float(lock_file_count) / max(float(file_count), 1.0))

        ingest_coverage_ratio = 0.0
        if project_root is not None:
            try:
                coverage = debug_snapshot_ingest_coverage(project_root=project_root, sqlite_path=db_path)
                ingest_coverage_ratio = _clamp01(_to_float(coverage.get("coverage_ratio"), 0.0))
                meta["ingest_coverage"] = {
                    "snapshot_total": int(coverage.get("snapshot_total", 0) or 0),
                    "ready_total": int(coverage.get("ready_total", 0) or 0),
                    "coverage_ratio": float(coverage.get("coverage_ratio", 0.0) or 0.0),
                }
            except Exception:
                ingest_coverage_ratio = 0.0

        context = {
            "snapshot_raw_sql_ingest_ratio": ingest_coverage_ratio,
            "snapshot_raw_count_norm": _clamp01(math.log1p(max(snapshot_count, 0)) / 6.0),
            "snapshot_raw_file_count_norm": _clamp01(math.log1p(max(file_count, 0)) / 10.0),
            "snapshot_raw_bytes_norm": _clamp01(math.log1p(max(byte_count, 0)) / 20.0),
            "snapshot_raw_json_ratio": json_ratio,
            "snapshot_raw_event_file_ratio": event_ratio,
            "snapshot_raw_lock_file_ratio": lock_ratio,
            "snapshot_raw_recency_norm": recency_norm,
        }

        meta.update(
            {
                "snapshot_count": int(snapshot_count),
                "file_count": int(file_count),
                "byte_count": int(byte_count),
                "latest_ts": latest_ts_iso,
                "cutoff_ts": cutoff_iso,
                "ingest_coverage_ratio": float(ingest_coverage_ratio),
            }
        )
        return context, meta
    finally:
        conn.close()


def compute_snapshot_health_context(
    payloads: Dict[str, Dict[str, Any]],
    *,
    raw_context: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    coverage = payloads.get("snapshot_coverage") or {}
    replay = payloads.get("replay_preopen_sanity") or {}
    drift = payloads.get("preopen_replay_drift") or {}
    divergence = payloads.get("data_source_divergence") or {}
    triprate = payloads.get("guardrail_triprate") or {}
    queue_stress = payloads.get("execution_queue_stress") or {}
    drill = payloads.get(SNAPSHOT_DRILL_KEY) or {}

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

    drill_files_checked = max(_to_float(drill.get("files_checked"), 0.0), 0.0)
    drill_missing_files = drill.get("missing_files") if isinstance(drill.get("missing_files"), list) else []
    drill_missing_count = float(len(drill_missing_files))
    drill_missing_ratio = _clamp01(drill_missing_count / max(drill_files_checked + drill_missing_count, 1.0))

    drill_rows = drill.get("rows") if isinstance(drill.get("rows"), list) else []
    drill_restore_total = float(len(drill_rows))
    drill_restore_ok = float(
        sum(1 for row in drill_rows if isinstance(row, dict) and bool(row.get("restore_ok", False)))
    )
    drill_restore_fail_ratio = _clamp01((drill_restore_total - drill_restore_ok) / max(drill_restore_total, 1.0))

    drill_ts = _parse_ts_utc(drill.get("timestamp_utc"))
    if drill_ts is not None:
        drill_age_hours = max((datetime.now(timezone.utc) - drill_ts).total_seconds() / 3600.0, 0.0)
        drill_recency_norm = 1.0 - _clamp01(drill_age_hours / 72.0)
    else:
        drill_recency_norm = 0.0

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
        "snapshot_drill_ok": 1.0 if bool(drill.get("ok", False)) else 0.0,
        "snapshot_drill_restore_fail_ratio": drill_restore_fail_ratio,
        "snapshot_drill_missing_ratio": drill_missing_ratio,
        "snapshot_drill_recency_norm": drill_recency_norm,
        "canary_weight_cap_norm": canary_weight_cap_norm,
    }

    for k, v in _default_raw_debug_context().items():
        context[k] = float(v)

    for k, v in (raw_context or {}).items():
        if k in context:
            context[k] = _clamp01(_to_float(v, 0.0))

    meta = {
        "coverage_ts": coverage.get("timestamp_utc"),
        "replay_ts": replay.get("timestamp_utc"),
        "drift_ts": drift.get("timestamp_utc"),
        "divergence_ts": divergence.get("timestamp_utc"),
        "triprate_ts": triprate.get("timestamp_utc"),
        "queue_stress_ts": queue_stress.get("timestamp_utc"),
        "state_snapshot_drill_ts": drill.get("timestamp_utc"),
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
    raw_sync_meta: Dict[str, Any] = {}
    raw_context_meta: Dict[str, Any] = {}

    db_path = sqlite_path if sqlite_path is not None else _default_sqlite_path(project_root)
    db_path = Path(db_path).expanduser().resolve()

    raw_context_enabled = os.getenv("SNAPSHOT_RAW_CONTEXT_ENABLED", "1").strip() == "1"
    raw_sync_enabled = os.getenv("SNAPSHOT_RAW_SYNC_ENABLED", "1").strip() == "1"

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

        if raw_context_enabled and raw_sync_enabled:
            try:
                raw_sync_meta = sync_raw_debug_snapshots_to_sqlite(
                    project_root=project_root,
                    sqlite_path=db_path,
                )
            except Exception as exc:
                raw_sync_meta = {"error": str(exc), "db_path": str(db_path)}

    sql_payloads: Dict[str, Dict[str, Any]] = {}
    try:
        sql_payloads = load_snapshot_health_payloads_from_sqlite(db_path)
    except Exception:
        sql_payloads = {}

    raw_context = _default_raw_debug_context()
    if raw_context_enabled:
        try:
            raw_context, raw_context_meta = load_raw_debug_snapshot_context_from_sqlite(
                sqlite_path=db_path,
                project_root=project_root,
            )
        except Exception as exc:
            raw_context = _default_raw_debug_context()
            raw_context_meta = {"error": str(exc), "sqlite_path": str(db_path)}

    merged: Dict[str, Dict[str, Any]] = {}
    selected_source: Dict[str, str] = {}
    for key in _metric_keys():
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

    context, calc_meta = compute_snapshot_health_context(merged, raw_context=raw_context)
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
            "raw_debug_sync": raw_sync_meta,
            "raw_debug_context": raw_context_meta,
            "raw_context_enabled": bool(raw_context_enabled),
        }
    )
    return context, meta
