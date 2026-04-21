#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from core.sqlite_runtime import (
    apply_sqlite_runtime_settings as _shared_apply_sqlite_runtime_settings,
    connect_sqlite as _shared_connect_sqlite,
    normalize_temp_store_mode as _shared_normalize_temp_store_mode,
    resolve_sqlite_runtime_settings as _shared_resolve_sqlite_runtime_settings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "governance" / "ops_data_plane.sqlite3"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _metadata_text(metadata: Mapping[str, Any] | None) -> str:
    return _json_text(dict(metadata or {}))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _truthy(raw: Any, default: bool = False) -> bool:
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def resolve_db_path(project_root: Path = PROJECT_ROOT, *, db_path: Path | str | None = None) -> Path:
    if db_path is not None:
        return Path(db_path).expanduser()
    configured = str(os.getenv("BOT_OPS_CONTROL_DB", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return project_root / DEFAULT_DB_PATH.relative_to(PROJECT_ROOT)


def normalize_entity_key(project_root: Path = PROJECT_ROOT, entity_key: Any = "", *, namespace: str = "") -> str:
    raw = str(entity_key or "").strip()
    if not raw:
        return str(namespace or "").strip()

    normalized = raw.replace("\\", "/").strip()
    candidate = Path(raw).expanduser()
    project_root_path = Path(project_root).expanduser().resolve(strict=False)
    if candidate.is_absolute():
        candidate_path = candidate.resolve(strict=False)
        try:
            normalized = candidate_path.relative_to(project_root_path).as_posix()
        except Exception:
            normalized = candidate_path.as_posix()
    elif "/" in normalized:
        normalized = Path(normalized.lstrip("./")).as_posix()

    namespace_text = str(namespace or "").strip().strip("/")
    if namespace_text:
        prefix = f"{namespace_text}/"
        if normalized != namespace_text and not normalized.startswith(prefix):
            normalized = f"{prefix}{normalized}"
    return normalized


def _env_value(primary: str, fallback: str = "", default: str = "") -> str:
    for name in (primary, fallback):
        if not name:
            continue
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
    return str(default or "")


def _normalize_temp_store_mode(raw: Any, default: str = "MEMORY") -> str:
    return _shared_normalize_temp_store_mode(raw, default)


def resolve_sqlite_runtime_settings(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return _shared_resolve_sqlite_runtime_settings(project_root)


def _apply_sqlite_runtime_settings(conn: sqlite3.Connection, settings: Mapping[str, Any]) -> None:
    _shared_apply_sqlite_runtime_settings(conn, settings)


def _commit_if_needed(conn: sqlite3.Connection, *, commit: bool) -> None:
    if commit:
        conn.commit()


def connect(project_root: Path = PROJECT_ROOT, *, db_path: Path | str | None = None, timeout_seconds: float = 30.0) -> sqlite3.Connection:
    path = resolve_db_path(project_root, db_path=db_path)
    conn = _shared_connect_sqlite(path, project_root=project_root, timeout_seconds=timeout_seconds)
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_watermarks (
            collector_key TEXT NOT NULL,
            source_name TEXT NOT NULL,
            entity_key TEXT NOT NULL DEFAULT '',
            watermark_type TEXT NOT NULL DEFAULT 'timestamp',
            watermark_value TEXT NOT NULL,
            etag TEXT NOT NULL DEFAULT '',
            payload_sha256 TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            updated_utc TEXT NOT NULL,
            PRIMARY KEY (collector_key, source_name, entity_key, watermark_type)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS collector_provenance_runs (
            run_uid TEXT PRIMARY KEY,
            collector_key TEXT NOT NULL,
            cache_key TEXT NOT NULL DEFAULT '',
            command_json TEXT NOT NULL DEFAULT '[]',
            command_fingerprint TEXT NOT NULL DEFAULT '',
            expect_paths_json TEXT NOT NULL DEFAULT '[]',
            fingerprint_files_json TEXT NOT NULL DEFAULT '[]',
            skipped INTEGER NOT NULL DEFAULT 0,
            rc INTEGER NOT NULL DEFAULT 0,
            started_utc TEXT NOT NULL,
            finished_utc TEXT NOT NULL,
            duration_seconds REAL NOT NULL DEFAULT 0,
            payload_sha256 TEXT NOT NULL DEFAULT '',
            stdout_tail TEXT NOT NULL DEFAULT '',
            stderr_tail TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_dead_letters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lane TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL DEFAULT 0,
            offset_bytes INTEGER NOT NULL DEFAULT 0,
            error_class TEXT NOT NULL,
            error_message TEXT NOT NULL,
            raw_payload TEXT NOT NULL DEFAULT '',
            payload_sha256 TEXT NOT NULL DEFAULT '',
            run_id TEXT NOT NULL DEFAULT '',
            iter_id TEXT NOT NULL DEFAULT '',
            recorded_utc TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_drift_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lane TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL DEFAULT 0,
            observed_schema_version INTEGER NOT NULL DEFAULT 0,
            expected_schema_version INTEGER NOT NULL DEFAULT 0,
            drift_kind TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL DEFAULT '',
            run_id TEXT NOT NULL DEFAULT '',
            iter_id TEXT NOT NULL DEFAULT '',
            recorded_utc TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS storage_route_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_root TEXT NOT NULL,
            mode TEXT NOT NULL,
            active_root TEXT NOT NULL,
            switched_links_json TEXT NOT NULL DEFAULT '[]',
            passthrough_paths_json TEXT NOT NULL DEFAULT '[]',
            autosync_copied_files INTEGER NOT NULL DEFAULT 0,
            autosync_copy_errors INTEGER NOT NULL DEFAULT 0,
            autosync_pruned_files INTEGER NOT NULL DEFAULT 0,
            split_brain_conflicts INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            recorded_utc TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_access_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_family TEXT NOT NULL,
            source_name TEXT NOT NULL DEFAULT '',
            shard_name TEXT NOT NULL DEFAULT '',
            consumer TEXT NOT NULL DEFAULT '',
            query_hash TEXT NOT NULL DEFAULT '',
            rows_scanned INTEGER NOT NULL DEFAULT 0,
            rows_returned INTEGER NOT NULL DEFAULT 0,
            duration_ms REAL NOT NULL DEFAULT 0,
            recorded_utc TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shard_heat_state (
            shard_name TEXT PRIMARY KEY,
            query_count INTEGER NOT NULL DEFAULT 0,
            rows_scanned_total INTEGER NOT NULL DEFAULT 0,
            rows_returned_total INTEGER NOT NULL DEFAULT 0,
            duration_ms_total REAL NOT NULL DEFAULT 0,
            last_heat_score REAL NOT NULL DEFAULT 0,
            promotion_candidate INTEGER NOT NULL DEFAULT 0,
            last_query_family TEXT NOT NULL DEFAULT '',
            last_access_utc TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS canonical_reconciliation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            canonical_source TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            divergence_score REAL NOT NULL DEFAULT 0,
            canonical_payload_json TEXT NOT NULL DEFAULT '{}',
            provider_votes_json TEXT NOT NULL DEFAULT '{}',
            recorded_utc TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS materialized_stream_daily (
            day_utc TEXT NOT NULL,
            stream TEXT NOT NULL,
            record_count INTEGER NOT NULL DEFAULT 0,
            distinct_sources INTEGER NOT NULL DEFAULT 0,
            min_schema_version INTEGER NOT NULL DEFAULT 0,
            max_schema_version INTEGER NOT NULL DEFAULT 0,
            last_ingested_at TEXT NOT NULL DEFAULT '',
            refreshed_utc TEXT NOT NULL,
            PRIMARY KEY (day_utc, stream)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS materialized_symbol_daily (
            day_utc TEXT NOT NULL,
            symbol TEXT NOT NULL,
            record_count INTEGER NOT NULL DEFAULT 0,
            buy_count INTEGER NOT NULL DEFAULT 0,
            sell_count INTEGER NOT NULL DEFAULT 0,
            hold_count INTEGER NOT NULL DEFAULT 0,
            last_ingested_at TEXT NOT NULL DEFAULT '',
            refreshed_utc TEXT NOT NULL,
            PRIMARY KEY (day_utc, symbol)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_source_watermarks_updated
        ON source_watermarks(collector_key, updated_utc DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_collector_provenance_runs_lookup
        ON collector_provenance_runs(collector_key, finished_utc DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ingest_dead_letters_lookup
        ON ingest_dead_letters(source_rel, recorded_utc DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_schema_drift_events_lookup
        ON schema_drift_events(source_rel, recorded_utc DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_storage_route_events_recorded
        ON storage_route_events(recorded_utc DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_query_access_events_lookup
        ON query_access_events(shard_name, recorded_utc DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_canonical_reconciliation_lookup
        ON canonical_reconciliation_events(domain, entity_key, recorded_utc DESC)
        """
    )
    conn.commit()


def record_watermark(
    conn: sqlite3.Connection,
    *,
    collector_key: str,
    source_name: str,
    watermark_value: str,
    entity_key: str = "",
    watermark_type: str = "timestamp",
    etag: str = "",
    payload_sha256: str = "",
    metadata: Mapping[str, Any] | None = None,
    commit: bool = True,
) -> None:
    conn.execute(
        """
        INSERT INTO source_watermarks(
            collector_key, source_name, entity_key, watermark_type, watermark_value,
            etag, payload_sha256, metadata_json, updated_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(collector_key, source_name, entity_key, watermark_type) DO UPDATE SET
            watermark_value=excluded.watermark_value,
            etag=excluded.etag,
            payload_sha256=excluded.payload_sha256,
            metadata_json=excluded.metadata_json,
            updated_utc=excluded.updated_utc
        """,
        (
            str(collector_key or "").strip(),
            str(source_name or "").strip(),
            str(entity_key or "").strip(),
            str(watermark_type or "timestamp").strip(),
            str(watermark_value or "").strip(),
            str(etag or "").strip(),
            str(payload_sha256 or "").strip(),
            _metadata_text(metadata),
            _now_utc(),
        ),
    )
    _commit_if_needed(conn, commit=commit)


def record_collector_run(
    conn: sqlite3.Connection,
    *,
    collector_key: str,
    command: list[str],
    cache_key: str = "",
    expect_paths: list[str] | None = None,
    fingerprint_files: list[str] | None = None,
    command_fingerprint: str = "",
    skipped: bool = False,
    rc: int = 0,
    started_utc: str = "",
    finished_utc: str = "",
    stdout_tail: str = "",
    stderr_tail: str = "",
    payload_sha256: str = "",
    metadata: Mapping[str, Any] | None = None,
    commit: bool = True,
) -> str:
    start = str(started_utc or _now_utc())
    finish = str(finished_utc or _now_utc())
    try:
        started_ts = datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
        finished_ts = datetime.fromisoformat(finish.replace("Z", "+00:00")).timestamp()
        duration_seconds = max(finished_ts - started_ts, 0.0)
    except Exception:
        duration_seconds = 0.0
    run_uid = str((metadata or {}).get("run_uid") or uuid.uuid4().hex)
    conn.execute(
        """
        INSERT OR REPLACE INTO collector_provenance_runs(
            run_uid, collector_key, cache_key, command_json, command_fingerprint, expect_paths_json,
            fingerprint_files_json, skipped, rc, started_utc, finished_utc, duration_seconds,
            payload_sha256, stdout_tail, stderr_tail, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_uid,
            str(collector_key or "").strip(),
            str(cache_key or "").strip(),
            _json_text(list(command or [])),
            str(command_fingerprint or "").strip(),
            _json_text(list(expect_paths or [])),
            _json_text(list(fingerprint_files or [])),
            int(bool(skipped)),
            int(rc),
            start,
            finish,
            round(float(duration_seconds), 6),
            str(payload_sha256 or "").strip(),
            str(stdout_tail or ""),
            str(stderr_tail or ""),
            _metadata_text(metadata),
        ),
    )
    _commit_if_needed(conn, commit=commit)
    return run_uid


def record_dead_letter(
    conn: sqlite3.Connection,
    *,
    lane: str,
    source_rel: str,
    line_no: int,
    offset_bytes: int,
    error_class: str,
    error_message: str,
    raw_payload: str,
    run_id: str = "",
    iter_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    commit: bool = True,
) -> None:
    payload_sha256 = _sha256_text(str(raw_payload or "")) if str(raw_payload or "") else ""
    conn.execute(
        """
        INSERT INTO ingest_dead_letters(
            lane, source_rel, line_no, offset_bytes, error_class, error_message,
            raw_payload, payload_sha256, run_id, iter_id, recorded_utc, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(lane or "").strip(),
            str(source_rel or "").strip(),
            int(line_no),
            int(offset_bytes),
            str(error_class or "").strip(),
            str(error_message or "").strip(),
            str(raw_payload or ""),
            payload_sha256,
            str(run_id or "").strip(),
            str(iter_id or "").strip(),
            _now_utc(),
            _metadata_text(metadata),
        ),
    )
    _commit_if_needed(conn, commit=commit)


def record_schema_drift(
    conn: sqlite3.Connection,
    *,
    lane: str,
    source_rel: str,
    line_no: int,
    observed_schema_version: int,
    expected_schema_version: int,
    drift_kind: str,
    payload_json: str,
    run_id: str = "",
    iter_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    commit: bool = True,
) -> None:
    payload_sha256 = _sha256_text(str(payload_json or "")) if str(payload_json or "") else ""
    conn.execute(
        """
        INSERT INTO schema_drift_events(
            lane, source_rel, line_no, observed_schema_version, expected_schema_version,
            drift_kind, payload_sha256, payload_json, run_id, iter_id, recorded_utc, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(lane or "").strip(),
            str(source_rel or "").strip(),
            int(line_no),
            int(observed_schema_version),
            int(expected_schema_version),
            str(drift_kind or "").strip(),
            payload_sha256,
            str(payload_json or ""),
            str(run_id or "").strip(),
            str(iter_id or "").strip(),
            _now_utc(),
            _metadata_text(metadata),
        ),
    )
    _commit_if_needed(conn, commit=commit)


def record_storage_route_event(
    conn: sqlite3.Connection,
    *,
    project_root: str | Path,
    mode: str,
    active_root: str | Path,
    switched_links: list[str] | tuple[str, ...],
    passthrough_paths: list[str] | tuple[str, ...],
    autosync_copied_files: int = 0,
    autosync_copy_errors: int = 0,
    autosync_pruned_files: int = 0,
    split_brain_conflicts: int = 0,
    metadata: Mapping[str, Any] | None = None,
    commit: bool = True,
) -> None:
    conn.execute(
        """
        INSERT INTO storage_route_events(
            project_root, mode, active_root, switched_links_json, passthrough_paths_json,
            autosync_copied_files, autosync_copy_errors, autosync_pruned_files,
            split_brain_conflicts, metadata_json, recorded_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(project_root),
            str(mode or "").strip(),
            str(active_root),
            _json_text(list(switched_links or [])),
            _json_text(list(passthrough_paths or [])),
            int(autosync_copied_files),
            int(autosync_copy_errors),
            int(autosync_pruned_files),
            int(split_brain_conflicts),
            _metadata_text(metadata),
            _now_utc(),
        ),
    )
    _commit_if_needed(conn, commit=commit)


def record_query_access(
    conn: sqlite3.Connection,
    *,
    query_family: str,
    shard_name: str,
    consumer: str,
    query_text: str,
    rows_scanned: int,
    rows_returned: int,
    duration_ms: float,
    source_name: str = "",
    metadata: Mapping[str, Any] | None = None,
    commit: bool = True,
) -> float:
    query_hash = _sha256_text(str(query_text or "").strip()) if str(query_text or "").strip() else ""
    recorded_utc = _now_utc()
    rows_scanned_i = max(int(rows_scanned), 0)
    rows_returned_i = max(int(rows_returned), 0)
    duration_ms_f = max(float(duration_ms), 0.0)
    conn.execute(
        """
        INSERT INTO query_access_events(
            query_family, source_name, shard_name, consumer, query_hash, rows_scanned,
            rows_returned, duration_ms, recorded_utc, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(query_family or "").strip(),
            str(source_name or "").strip(),
            str(shard_name or "").strip(),
            str(consumer or "").strip(),
            query_hash,
            rows_scanned_i,
            rows_returned_i,
            round(duration_ms_f, 6),
            recorded_utc,
            _metadata_text(metadata),
        ),
    )
    heat_score = round(
        min(
            (rows_scanned_i / 5000.0)
            + (duration_ms_f / 750.0)
            + (0.35 if rows_returned_i <= 100 else 0.0),
            10.0,
        ),
        6,
    )
    promotion_candidate = int(bool(heat_score >= 2.5 or rows_scanned_i >= 10000 or duration_ms_f >= 1500.0))
    conn.execute(
        """
        INSERT INTO shard_heat_state(
            shard_name, query_count, rows_scanned_total, rows_returned_total, duration_ms_total,
            last_heat_score, promotion_candidate, last_query_family, last_access_utc, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(shard_name) DO UPDATE SET
            query_count=shard_heat_state.query_count + excluded.query_count,
            rows_scanned_total=shard_heat_state.rows_scanned_total + excluded.rows_scanned_total,
            rows_returned_total=shard_heat_state.rows_returned_total + excluded.rows_returned_total,
            duration_ms_total=ROUND(shard_heat_state.duration_ms_total + excluded.duration_ms_total, 6),
            last_heat_score=excluded.last_heat_score,
            promotion_candidate=CASE
                WHEN excluded.promotion_candidate != 0 OR shard_heat_state.promotion_candidate != 0 THEN 1
                ELSE 0
            END,
            last_query_family=excluded.last_query_family,
            last_access_utc=excluded.last_access_utc,
            metadata_json=excluded.metadata_json
        """,
        (
            str(shard_name or "").strip(),
            1,
            rows_scanned_i,
            rows_returned_i,
            round(duration_ms_f, 6),
            heat_score,
            promotion_candidate,
            str(query_family or "").strip(),
            recorded_utc,
            _metadata_text(metadata),
        ),
    )
    _commit_if_needed(conn, commit=commit)
    return heat_score


def load_shard_heat_map(project_root: Path = PROJECT_ROOT, *, db_path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    path = resolve_db_path(project_root, db_path=db_path)
    if not path.exists():
        return {}
    with connect(project_root, db_path=path) as conn:
        rows = conn.execute(
            """
            SELECT shard_name, query_count, rows_scanned_total, rows_returned_total, duration_ms_total,
                   last_heat_score, promotion_candidate, last_query_family, last_access_utc
            FROM shard_heat_state
            """
        ).fetchall()
    return {
        str(row[0]): {
            "query_count": int(row[1] or 0),
            "rows_scanned_total": int(row[2] or 0),
            "rows_returned_total": int(row[3] or 0),
            "duration_ms_total": float(row[4] or 0.0),
            "last_heat_score": float(row[5] or 0.0),
            "promotion_candidate": bool(row[6]),
            "last_query_family": str(row[7] or ""),
            "last_access_utc": str(row[8] or ""),
        }
        for row in rows
    }


def record_canonical_reconciliation(
    conn: sqlite3.Connection,
    *,
    domain: str,
    entity_key: str,
    canonical_source: str,
    confidence: float,
    divergence_score: float,
    canonical_payload: Mapping[str, Any],
    provider_votes: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    commit: bool = True,
) -> None:
    conn.execute(
        """
        INSERT INTO canonical_reconciliation_events(
            domain, entity_key, canonical_source, confidence, divergence_score,
            canonical_payload_json, provider_votes_json, recorded_utc, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(domain or "").strip(),
            str(entity_key or "").strip(),
            str(canonical_source or "").strip(),
            round(max(float(confidence), 0.0), 6),
            round(max(float(divergence_score), 0.0), 6),
            _json_text(dict(canonical_payload or {})),
            _json_text(dict(provider_votes or {})),
            _now_utc(),
            _metadata_text(metadata),
        ),
    )
    _commit_if_needed(conn, commit=commit)


def latest_collector_run(project_root: Path = PROJECT_ROOT, *, collector_key: str, db_path: Path | str | None = None) -> dict[str, Any]:
    path = resolve_db_path(project_root, db_path=db_path)
    if not path.exists():
        return {}
    with connect(project_root, db_path=path) as conn:
        row = conn.execute(
            """
            SELECT run_uid, cache_key, command_fingerprint, skipped, rc, started_utc, finished_utc,
                   duration_seconds, payload_sha256, stdout_tail, stderr_tail, metadata_json
            FROM collector_provenance_runs
            WHERE collector_key=?
            ORDER BY finished_utc DESC
            LIMIT 1
            """,
            (str(collector_key or "").strip(),),
        ).fetchone()
    if row is None:
        return {}
    metadata = {}
    try:
        metadata = json.loads(str(row[11] or "{}"))
    except Exception:
        metadata = {}
    return {
        "run_uid": str(row[0] or ""),
        "cache_key": str(row[1] or ""),
        "command_fingerprint": str(row[2] or ""),
        "skipped": bool(row[3]),
        "rc": int(row[4] or 0),
        "started_utc": str(row[5] or ""),
        "finished_utc": str(row[6] or ""),
        "duration_seconds": float(row[7] or 0.0),
        "payload_sha256": str(row[8] or ""),
        "stdout_tail": str(row[9] or ""),
        "stderr_tail": str(row[10] or ""),
        "metadata": metadata,
    }


def collector_error_budget(project_root: Path = PROJECT_ROOT, *, collector_key: str, window_hours: float = 24.0, db_path: Path | str | None = None) -> dict[str, Any]:
    path = resolve_db_path(project_root, db_path=db_path)
    if not path.exists():
        return {
            "run_count": 0,
            "failed_runs": 0,
            "skipped_runs": 0,
            "success_rate": 1.0,
            "error_budget_remaining": 1.0,
        }
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max(float(window_hours), 1.0))).isoformat()
    with connect(project_root, db_path=path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*),
                SUM(CASE WHEN rc != 0 AND skipped = 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN skipped = 1 THEN 1 ELSE 0 END)
            FROM collector_provenance_runs
            WHERE collector_key=? AND finished_utc>=?
            """,
            (str(collector_key or "").strip(), cutoff),
        ).fetchone()
        if row and int(row[0] or 0) <= 0:
            row = conn.execute(
                """
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN rc != 0 AND skipped = 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN skipped = 1 THEN 1 ELSE 0 END)
                FROM collector_provenance_runs
                WHERE collector_key=?
                """,
                (str(collector_key or "").strip(),),
            ).fetchone()
    run_count = int(row[0] or 0) if row else 0
    failed_runs = int(row[1] or 0) if row else 0
    skipped_runs = int(row[2] or 0) if row else 0
    success_rate = 1.0 if run_count <= 0 else max(1.0 - (failed_runs / max(run_count, 1)), 0.0)
    error_budget_remaining = max(1.0 - (failed_runs / max(run_count, 1)), 0.0) if run_count > 0 else 1.0
    return {
        "run_count": run_count,
        "failed_runs": failed_runs,
        "skipped_runs": skipped_runs,
        "success_rate": round(success_rate, 6),
        "error_budget_remaining": round(error_budget_remaining, 6),
    }


def emit_materialized_summaries(
    conn: sqlite3.Connection,
    *,
    source_db_path: Path | str,
    lookback_days: int = 7,
) -> dict[str, Any]:
    source_path = Path(source_db_path).expanduser()
    refreshed_utc = _now_utc()
    day_floor = (datetime.now(timezone.utc) - timedelta(days=max(int(lookback_days), 1))).strftime("%Y-%m-%d")
    attached = False
    source_record_count = 0
    conn.execute("ATTACH DATABASE ? AS sourcedb", (str(source_path),))
    attached = True
    try:
        source_cols = {
            str(row[1])
            for row in conn.execute("PRAGMA sourcedb.table_info(jsonl_records)").fetchall()
            if isinstance(row, (list, tuple)) and len(row) > 1
        }
        day_expr = "COALESCE(source_day_utc, substr(COALESCE(json_extract(payload_json, '$.timestamp_utc'), ingested_at), 1, 10))"
        if "source_day_utc" not in source_cols:
            day_expr = "substr(COALESCE(json_extract(payload_json, '$.timestamp_utc'), ingested_at), 1, 10)"
        stream_expr = "COALESCE(source_stream, CASE " \
            "WHEN source_rel LIKE 'decision_explanations/%' THEN 'decision_explanations' " \
            "WHEN source_rel LIKE 'decisions/%' THEN 'decisions' " \
            "WHEN source_rel LIKE 'governance/events/%' THEN 'governance_events' " \
            "WHEN source_rel LIKE 'governance/watchdog/%' THEN 'governance_watchdog' " \
            "WHEN source_rel LIKE 'governance/%' THEN 'governance' " \
            "WHEN source_rel LIKE 'exports/trade_logs/%' THEN 'trade_logs' " \
            "WHEN source_rel LIKE 'exports/paper_broker_bridge/%' THEN 'paper_broker_bridge' " \
            "WHEN source_rel LIKE 'data/%' THEN 'data' " \
            "ELSE 'other' END)"
        if "source_stream" not in source_cols:
            stream_expr = "CASE " \
                "WHEN source_rel LIKE 'decision_explanations/%' THEN 'decision_explanations' " \
                "WHEN source_rel LIKE 'decisions/%' THEN 'decisions' " \
                "WHEN source_rel LIKE 'governance/events/%' THEN 'governance_events' " \
                "WHEN source_rel LIKE 'governance/watchdog/%' THEN 'governance_watchdog' " \
                "WHEN source_rel LIKE 'governance/%' THEN 'governance' " \
                "WHEN source_rel LIKE 'exports/trade_logs/%' THEN 'trade_logs' " \
                "WHEN source_rel LIKE 'exports/paper_broker_bridge/%' THEN 'paper_broker_bridge' " \
                "WHEN source_rel LIKE 'data/%' THEN 'data' " \
                "ELSE 'other' END"
        source_record_count = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM sourcedb.jsonl_records
                WHERE {day_expr} >= ?
                """,
                (day_floor,),
            ).fetchone()[0]
            or 0
        )
        conn.execute("DELETE FROM materialized_stream_daily WHERE day_utc>=?", (day_floor,))
        conn.execute(
            f"""
            INSERT OR REPLACE INTO materialized_stream_daily(
                day_utc, stream, record_count, distinct_sources, min_schema_version,
                max_schema_version, last_ingested_at, refreshed_utc
            )
            SELECT
                {day_expr} AS day_utc,
                {stream_expr} AS stream,
                COUNT(*) AS record_count,
                COUNT(DISTINCT source_rel) AS distinct_sources,
                MIN(COALESCE(log_schema_version, 0)) AS min_schema_version,
                MAX(COALESCE(log_schema_version, 0)) AS max_schema_version,
                MAX(ingested_at) AS last_ingested_at,
                ? AS refreshed_utc
            FROM sourcedb.jsonl_records
            WHERE {day_expr} >= ?
            GROUP BY 1, 2
            """,
            (refreshed_utc, day_floor),
        )
        conn.execute("DELETE FROM materialized_symbol_daily WHERE day_utc>=?", (day_floor,))
        conn.execute(
            f"""
            INSERT OR REPLACE INTO materialized_symbol_daily(
                day_utc, symbol, record_count, buy_count, sell_count, hold_count, last_ingested_at, refreshed_utc
            )
            SELECT
                {day_expr} AS day_utc,
                UPPER(TRIM(COALESCE(json_extract(payload_json, '$.symbol'), ''))) AS symbol,
                COUNT(*) AS record_count,
                SUM(CASE WHEN UPPER(TRIM(COALESCE(json_extract(payload_json, '$.action'), ''))) = 'BUY' THEN 1 ELSE 0 END) AS buy_count,
                SUM(CASE WHEN UPPER(TRIM(COALESCE(json_extract(payload_json, '$.action'), ''))) = 'SELL' THEN 1 ELSE 0 END) AS sell_count,
                SUM(CASE WHEN UPPER(TRIM(COALESCE(json_extract(payload_json, '$.action'), ''))) = 'HOLD' THEN 1 ELSE 0 END) AS hold_count,
                MAX(ingested_at) AS last_ingested_at,
                ? AS refreshed_utc
            FROM sourcedb.jsonl_records
            WHERE {day_expr} >= ?
              AND LENGTH(TRIM(COALESCE(json_extract(payload_json, '$.symbol'), ''))) > 0
            GROUP BY 1, 2
            """,
            (refreshed_utc, day_floor),
        )
        stream_count = int(conn.execute("SELECT COUNT(*) FROM materialized_stream_daily WHERE day_utc>=?", (day_floor,)).fetchone()[0] or 0)
        symbol_count = int(conn.execute("SELECT COUNT(*) FROM materialized_symbol_daily WHERE day_utc>=?", (day_floor,)).fetchone()[0] or 0)
        conn.commit()
    finally:
        if attached:
            conn.execute("DETACH DATABASE sourcedb")
    return {
        "refreshed_utc": refreshed_utc,
        "day_floor": day_floor,
        "source_record_count": source_record_count,
        "stream_summary_rows": stream_count,
        "symbol_summary_rows": symbol_count,
    }


def file_sha256(path: Path | str) -> str:
    target = Path(path).expanduser()
    if not target.exists() or not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def time_query(fn) -> tuple[Any, float]:
    started = time.perf_counter()
    out = fn()
    duration_ms = (time.perf_counter() - started) * 1000.0
    return out, duration_ms
