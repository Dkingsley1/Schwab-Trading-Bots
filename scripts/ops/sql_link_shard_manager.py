import argparse
import fcntl
import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = PROJECT_ROOT / ".venv312" / "bin" / "python"
LINK_SCRIPT = PROJECT_ROOT / "scripts" / "link_jsonl_to_sql.py"
HOT_RETENTION_SCRIPT = PROJECT_ROOT / "scripts" / "sql_hot_retention.py"
QUEUE_RETENTION_SCRIPT = PROJECT_ROOT / "scripts" / "sql_queue_retention.py"
SQLITE_MAINTENANCE_SCRIPT = PROJECT_ROOT / "scripts" / "sqlite_performance_maintenance.py"
PRIMARY_DB_PATH = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"
QUEUE_DB_PATH = Path(
    str(
        os.getenv(
            "SQL_LINK_SERVICE_QUEUE_DB",
            os.getenv(
                "BOT_CHANNEL_QUEUE_DB",
                str(PROJECT_ROOT / "local_fallback_storage" / "data" / "bot_channel_queue.sqlite3"),
            ),
        )
    )
).expanduser()
SHARD_DB_ROOT = PROJECT_ROOT / "data" / "sql_link_shards"
SHARD_STATE_ROOT = PROJECT_ROOT / "governance" / "sql_link_shards"
HEALTH_ROOT = PROJECT_ROOT / "governance" / "health"
EVENT_ROOT = PROJECT_ROOT / "governance" / "events"
LATEST_HEALTH = HEALTH_ROOT / "sql_link_service_latest.json"
PROGRESS_HEALTH = HEALTH_ROOT / "sql_link_service_progress_latest.json"
MAINTENANCE_STATE_PATH = HEALTH_ROOT / "sql_link_service_maintenance_state.json"
INTEGRITY_MARKER_ROOT = HEALTH_ROOT / "sql_link_integrity"

JSONL_COLUMNS = [
    "source_file",
    "source_rel",
    "line_no",
    "ingested_at",
    "payload_sha1",
    "payload_json",
    "run_id",
    "iter_id",
    "decision_id",
    "parent_decision_id",
    "log_schema_version",
]
JSON_FILE_COLUMNS = [
    "source_file",
    "source_rel",
    "stream",
    "modified_at",
    "ingested_at",
    "payload_sha1",
    "payload_json",
    "payload_size_bytes",
    "log_schema_version",
]
DEFAULT_SHARD_DEFS = {
    "health_fast": {
        "path_contains": (
            "governance/health/data_ingress_latest_,"
            "governance/health/ingestion_backpressure_latest.json,"
            "governance/health/data_source_divergence_latest.json,"
            "governance/health/data_source_divergence_bond_latest.json,"
            "governance/health/data_source_divergence_non_bond_latest.json,"
            "governance/health/jsonl_sql_ingestion_health_,"
            "governance/health/sql_link_service_,"
            "governance/health/paper_performance_latest.json,"
            "governance/health/one_numbers_latest.json,"
            "governance/health/daily_runtime_summary_latest.json"
        ),
        "skip_json_files": False,
        "max_files": 12,
    },
    "crypto_governance": {
        "include_streams": "governance_events,governance_watchdog,governance",
        "path_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": False,
        "max_files": 12,
    },
    "crypto_trading_fast": {
        "include_streams": "paper_broker_bridge,top_level_trade_links",
        "path_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 8,
    },
    "crypto_explanations": {
        "include_streams": "decision_explanations",
        "path_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 8,
    },
    "crypto_shadow_attribution": {
        "include_streams": "governance",
        "path_contains": "shadow_pnl_attribution_,shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 8,
    },
    "crypto_trading": {
        "include_streams": "decisions,trade_logs",
        "path_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 10,
    },
    "governance": {
        "include_streams": "governance_events,governance_watchdog,governance",
        "path_not_contains": "shadow_pnl_attribution_,shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": False,
    },
    "trading_fast": {
        "include_streams": "paper_broker_bridge,top_level_trade_links",
        "path_not_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 8,
    },
    "explanations": {
        "include_streams": "decision_explanations",
        "path_not_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 8,
    },
    "shadow_attribution": {
        "include_streams": "governance",
        "path_contains": "shadow_pnl_attribution_",
        "path_not_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 8,
    },
    "aggressive_trading": {
        "include_streams": "decisions,trade_logs",
        "path_contains": "shadow_aggressive_,shadow_intraday_aggressive_,shadow_swing_aggressive_",
        "path_not_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab",
        "skip_json_files": True,
        "max_files": 10,
    },
    "trading": {
        "include_streams": "decisions,trade_logs",
        "path_not_contains": "shadow_crypto/,shadow_crypto_futures_crypto/,default_crypto_coinbase,crypto_futures_crypto_coinbase,default_crypto_schwab,crypto_futures_crypto_schwab,shadow_aggressive_,shadow_intraday_aggressive_,shadow_swing_aggressive_",
        "skip_json_files": True,
    },
    "data": {
        "include_streams": "data",
        "skip_json_files": True,
    },
}
ARCHIVE_MAINTENANCE_GLOBS = ("*.compact.sqlite3", "*.precompact.bak.sqlite3")
LEGACY_DEFAULT_SHARDS = "trading,governance,data"
PRE_FAST_DEFAULT_SHARDS = "crypto_governance,crypto_trading,governance,trading,data"
PRE_BACKLOG_SPLIT_DEFAULT_SHARDS = "health_fast,crypto_trading_fast,trading_fast,crypto_governance,crypto_trading,governance,trading,data"
CURRENT_DEFAULT_SHARDS = "health_fast,crypto_trading_fast,trading_fast,crypto_explanations,explanations,crypto_shadow_attribution,shadow_attribution,crypto_governance,crypto_trading,governance,aggressive_trading,trading,data"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_size_gb(path: Path) -> float:
    try:
        return float(path.stat().st_size) / (1024.0 ** 3)
    except Exception:
        return 0.0


def _wal_size_gb(path: Path) -> float:
    return _db_size_gb(Path(f"{path}-wal"))


def _parse_json_output(text: str) -> dict:
    raw = str(text or "").strip()
    if not raw:
        return {}
    lines = [line for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _busy_progress_summary() -> dict:
    payload = _load_json(PROGRESS_HEALTH)
    if not isinstance(payload, dict):
        return {}
    summary = {
        "timestamp_utc": str(payload.get("timestamp_utc") or ""),
        "status": str(payload.get("status") or ""),
        "current_step": str(payload.get("current_step") or ""),
        "completed_shard_count": _as_int(payload.get("completed_shard_count"), 0),
        "completed_merge_count": _as_int(payload.get("completed_merge_count"), 0),
        "merged_rows_this_cycle": _as_int(payload.get("merged_rows_this_cycle"), 0),
        "primary_db": str(payload.get("primary_db") or ""),
    }
    return {k: v for k, v in summary.items() if v not in ("", 0)}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_service_progress(
    *,
    cycle_started_utc: str,
    current_step: str,
    lock_path: Path,
    primary_db: Path,
    shards: list[dict[str, object]] | None = None,
    shard_results: list[dict[str, object]] | None = None,
    merge_results: list[dict[str, object]] | None = None,
    merged_rows_this_cycle: int = 0,
    running: bool = True,
    ok: bool = True,
    note: str = "",
) -> None:
    payload = {
        "timestamp_utc": _now_utc(),
        "status": ("running" if running else ("ok" if ok else "error")),
        "ok": bool(ok),
        "running": bool(running),
        "cycle_started_utc": str(cycle_started_utc or ""),
        "current_step": str(current_step or ""),
        "lock_path": str(lock_path),
        "primary_db": str(primary_db),
        "maintenance_state_path": str(MAINTENANCE_STATE_PATH),
        "shards": shard_results if isinstance(shard_results, list) else [],
        "merge_results": merge_results if isinstance(merge_results, list) else [],
        "planned_shards": [str((row or {}).get("name", "")) for row in (shards or []) if str((row or {}).get("name", "")).strip()],
        "completed_shard_count": len(shard_results or []),
        "completed_merge_count": len(merge_results or []),
        "merged_rows_this_cycle": int(merged_rows_this_cycle),
        "note": str(note or ""),
    }
    _write_json(PROGRESS_HEALTH, payload)


def _as_float(raw: object, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _as_int(raw: object, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


def _as_reason_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _load_maintenance_state(path: Path, *, db_size_gb: float, wal_size_gb: float) -> dict[str, object]:
    payload = _load_json(path)
    wal = payload.get("wal_checkpoint", {}) if isinstance(payload.get("wal_checkpoint"), dict) else {}
    hot = payload.get("hot_retention", {}) if isinstance(payload.get("hot_retention"), dict) else {}
    return {
        "timestamp_utc": str(payload.get("timestamp_utc") or ""),
        "wal_checkpoint": {
            "last_run_utc": str(wal.get("last_run_utc") or ""),
            "baseline_db_size_gb": _as_float(wal.get("baseline_db_size_gb"), db_size_gb),
            "baseline_wal_size_gb": _as_float(wal.get("baseline_wal_size_gb"), wal_size_gb),
            "rows_since_last_run": _as_int(wal.get("rows_since_last_run"), 0),
            "last_trigger_reasons": _as_reason_list(wal.get("last_trigger_reasons")),
        },
        "hot_retention": {
            "last_run_utc": str(hot.get("last_run_utc") or ""),
            "baseline_db_size_gb": _as_float(hot.get("baseline_db_size_gb"), db_size_gb),
            "baseline_wal_size_gb": _as_float(hot.get("baseline_wal_size_gb"), wal_size_gb),
            "rows_since_last_run": _as_int(hot.get("rows_since_last_run"), 0),
            "last_trigger_reasons": _as_reason_list(hot.get("last_trigger_reasons")),
        },
    }


def _merged_rows_inserted(merge_results: list[dict[str, object]]) -> int:
    total = 0
    for row in merge_results:
        if not isinstance(row, dict):
            continue
        total += _as_int(row.get("jsonl_rows_inserted"), 0)
        total += _as_int(row.get("json_file_rows_inserted"), 0)
    return max(total, 0)


def _wal_checkpoint_trigger_reasons(
    *,
    wal_size_gb: float,
    wal_threshold_gb: float,
    wal_growth_gb: float,
    wal_growth_trigger_gb: float,
    rows_since_last_run: int,
    row_trigger: int,
) -> list[str]:
    reasons: list[str] = []
    if wal_size_gb <= 0.0:
        return reasons
    if wal_threshold_gb > 0.0 and wal_size_gb >= wal_threshold_gb:
        reasons.append(f"wal_size_gb>={wal_threshold_gb:g}")
    if wal_growth_trigger_gb > 0.0 and wal_growth_gb >= wal_growth_trigger_gb:
        reasons.append(f"wal_growth_gb>={wal_growth_trigger_gb:g}")
    if row_trigger > 0 and rows_since_last_run >= row_trigger:
        reasons.append(f"rows_since_last_run>={row_trigger}")
    return reasons


def _hot_retention_trigger_reasons(
    *,
    db_size_gb: float,
    max_db_gb: float,
    db_growth_gb: float,
    growth_trigger_gb: float,
    rows_since_last_run: int,
    row_trigger: int,
    has_successful_run: bool,
) -> list[str]:
    reasons: list[str] = []
    if not has_successful_run:
        if max_db_gb > 0.0 and db_size_gb >= max_db_gb:
            reasons.append(f"bootstrap_db_size_gb>={max_db_gb:g}")
        return reasons
    if growth_trigger_gb <= 0.0 and row_trigger <= 0:
        if max_db_gb > 0.0 and db_size_gb >= max_db_gb:
            reasons.append(f"db_size_gb>={max_db_gb:g}")
        return reasons
    if growth_trigger_gb > 0.0 and db_growth_gb >= growth_trigger_gb:
        reasons.append(f"db_growth_gb>={growth_trigger_gb:g}")
    if row_trigger > 0 and rows_since_last_run >= row_trigger:
        reasons.append(f"rows_since_last_run>={row_trigger}")
    return reasons


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _normalized_shard_config(raw: str) -> str:
    cleaned = ",".join(_parse_csv(raw))
    if not cleaned or cleaned in {LEGACY_DEFAULT_SHARDS, PRE_FAST_DEFAULT_SHARDS, PRE_BACKLOG_SPLIT_DEFAULT_SHARDS}:
        return CURRENT_DEFAULT_SHARDS
    return cleaned


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _archive_maintenance_blockers(archive_root: str) -> list[str]:
    root = Path(str(archive_root or "")).expanduser()
    if not root.exists():
        return []
    blockers: list[str] = []
    for pattern in ARCHIVE_MAINTENANCE_GLOBS:
        for path in sorted(root.glob(pattern)):
            blockers.append(str(path))
    return blockers


def _integrity_marker_path(shard_name: str) -> Path:
    safe_name = str(shard_name or "unknown").strip().replace("/", "_") or "unknown"
    return INTEGRITY_MARKER_ROOT / f"{safe_name}.json"


def _load_integrity_marker(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_integrity_marker(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _should_run_deep_integrity_check(*, shard_name: str, sqlite_db: Path) -> tuple[bool, Path, dict[str, object]]:
    marker_path = _integrity_marker_path(shard_name)
    marker = _load_integrity_marker(marker_path)
    min_interval_seconds = max(int(os.getenv("SQL_LINK_SERVICE_DEEP_INTEGRITY_MIN_INTERVAL_SECONDS", "21600")), 300)
    max_inline_db_gb = max(float(os.getenv("SQL_LINK_SERVICE_DEEP_INTEGRITY_MAX_INLINE_DB_GB", "1.5")), 0.0)
    db_size_gb = _db_size_gb(sqlite_db)
    checked_at_epoch = float(marker.get("checked_at_epoch", 0.0) or 0.0)
    marker_ok = bool(marker.get("ok", False))
    if max_inline_db_gb > 0.0 and db_size_gb > max_inline_db_gb:
        return False, marker_path, marker
    if checked_at_epoch <= 0.0 or not marker_ok:
        return True, marker_path, marker
    age_seconds = max(time.time() - checked_at_epoch, 0.0)
    if age_seconds >= float(min_interval_seconds):
        return True, marker_path, marker
    return False, marker_path, marker


def _sqlite_integrity_status(path: Path, *, deep: bool) -> tuple[bool, str]:
    if not path.exists():
        return True, "missing"
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(path))
        if deep:
            row = conn.execute("PRAGMA quick_check").fetchone()
            status = str((row or ("unknown",))[0] or "unknown").strip() or "unknown"
            ok = status.lower() == "ok"
        else:
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
            status = "opened"
            ok = True
        return ok, status
    except sqlite3.DatabaseError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _quarantine_shard_artifacts(
    *,
    shard_name: str,
    sqlite_db: Path,
    state_file: Path,
    health_file: Path,
) -> dict[str, object]:
    deep_check, marker_path, marker = _should_run_deep_integrity_check(shard_name=shard_name, sqlite_db=sqlite_db)
    marker_age_seconds = max(time.time() - _as_float(marker.get("checked_at_epoch"), 0.0), 0.0)
    db_size_gb = _db_size_gb(sqlite_db)
    skip_probe_min_interval_seconds = max(
        int(os.getenv("SQL_LINK_SERVICE_OPEN_PROBE_MIN_INTERVAL_SECONDS", "900")),
        60,
    )
    skip_probe_min_db_gb = max(
        float(os.getenv("SQL_LINK_SERVICE_OPEN_PROBE_SKIP_MIN_DB_GB", "4.0")),
        0.0,
    )
    payload = {
        "triggered": False,
        "reason": "",
        "quarantine_root": "",
        "moved_paths": [],
        "health_file_reset": False,
        "integrity_probe_mode": "quick_check" if deep_check else "open_probe",
        "integrity_marker": str(marker_path),
    }
    if (
        not deep_check
        and bool(marker.get("ok", False))
        and db_size_gb >= float(skip_probe_min_db_gb)
        and marker_age_seconds < float(skip_probe_min_interval_seconds)
    ):
        payload.update(
            {
                "reason": "recent_ok_marker_skip",
                "integrity_probe_mode": "recent_marker_skip",
            }
        )
        return payload

    ok, detail = _sqlite_integrity_status(sqlite_db, deep=deep_check)
    payload["reason"] = str(detail)
    try:
        stat = sqlite_db.stat()
        _save_integrity_marker(
            marker_path,
            {
                "shard_name": str(shard_name),
                "checked_at": _now_utc(),
                "checked_at_epoch": float(time.time()),
                "db_size_bytes": int(stat.st_size),
                "db_mtime": float(stat.st_mtime),
                "ok": bool(ok),
                "detail": str(detail),
                "probe_mode": str(payload["integrity_probe_mode"]),
            },
        )
    except Exception:
        pass
    if ok:
        return payload

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_root = SHARD_DB_ROOT / "corrupt_quarantine" / f"{shard_name}_{stamp}"
    quarantine_root.mkdir(parents=True, exist_ok=True)

    moved_paths: list[str] = []
    for path in (sqlite_db, Path(f"{sqlite_db}-wal"), Path(f"{sqlite_db}-shm"), state_file):
        if not path.exists():
            continue
        dest = quarantine_root / path.name
        path.replace(dest)
        moved_paths.append(str(dest))

    if health_file.exists():
        try:
            health_file.unlink()
            payload["health_file_reset"] = True
        except Exception:
            payload["health_file_reset"] = False

    payload.update(
        {
            "triggered": True,
            "quarantine_root": str(quarantine_root),
            "moved_paths": moved_paths,
        }
    )
    return payload


def _shard_env(name: str, suffix: str) -> str:
    return f"SQL_LINK_SERVICE_SHARD_{name.upper()}_{suffix}"


def _table_exists(conn: sqlite3.Connection, db_alias: str, table: str) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {db_alias}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return bool(row)


def _ensure_primary_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jsonl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            run_id TEXT,
            iter_id TEXT,
            decision_id TEXT,
            parent_decision_id TEXT,
            log_schema_version INTEGER,
            UNIQUE(source_file, line_no)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS json_file_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            stream TEXT NOT NULL,
            modified_at TEXT,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_size_bytes INTEGER NOT NULL DEFAULT 0,
            log_schema_version INTEGER,
            UNIQUE(source_rel, payload_sha1)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shard_merge_state (
            shard_name TEXT PRIMARY KEY,
            last_jsonl_id INTEGER NOT NULL DEFAULT 0,
            last_json_file_id INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jsonl_records_source_rel_line ON jsonl_records(source_rel, line_no)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jsonl_records_ingested_at ON jsonl_records(ingested_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_json_file_records_source_rel ON json_file_records(source_rel)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_json_file_records_ingested_at ON json_file_records(ingested_at)")


def _read_shard_cursor(conn: sqlite3.Connection, shard_name: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT last_jsonl_id, last_json_file_id FROM shard_merge_state WHERE shard_name = ?",
        (shard_name,),
    ).fetchone()
    if not row:
        return 0, 0
    return int(row[0] or 0), int(row[1] or 0)


def _probe_shard_merge_state(
    *,
    shard_name: str,
    shard_db: Path,
    primary_db: Path,
    sqlite_timeout_seconds: int,
) -> dict[str, object]:
    result = {
        "shard": shard_name,
        "shard_db": str(shard_db),
        "primary_db": str(primary_db),
        "ok": True,
        "merge_required": False,
        "last_jsonl_id": 0,
        "last_json_file_id": 0,
        "max_jsonl_id": 0,
        "max_json_file_id": 0,
    }
    if not shard_db.exists():
        result["ok"] = False
        result["error"] = "shard_db_missing"
        return result

    conn = sqlite3.connect(str(primary_db), timeout=max(float(sqlite_timeout_seconds), 1.0))
    try:
        conn.execute(f"PRAGMA busy_timeout={int(max(float(sqlite_timeout_seconds), 1.0) * 1000)}")
        _ensure_primary_schema(conn)
        last_jsonl_id, last_json_file_id = _read_shard_cursor(conn, shard_name)
        result["last_jsonl_id"] = last_jsonl_id
        result["last_json_file_id"] = last_json_file_id
        conn.execute("ATTACH DATABASE ? AS sharddb", (str(shard_db),))
        if _table_exists(conn, "sharddb", "jsonl_records"):
            result["max_jsonl_id"] = int(
                conn.execute("SELECT COALESCE(MAX(id), 0) FROM sharddb.jsonl_records").fetchone()[0] or 0
            )
        if _table_exists(conn, "sharddb", "json_file_records"):
            result["max_json_file_id"] = int(
                conn.execute("SELECT COALESCE(MAX(id), 0) FROM sharddb.json_file_records").fetchone()[0] or 0
            )
        conn.execute("DETACH DATABASE sharddb")
        result["merge_required"] = bool(
            int(result["max_jsonl_id"]) > int(last_jsonl_id)
            or int(result["max_json_file_id"]) > int(last_json_file_id)
            or (int(result["max_jsonl_id"]) > 0 and int(last_jsonl_id) > int(result["max_jsonl_id"]))
            or (int(result["max_json_file_id"]) > 0 and int(last_json_file_id) > int(result["max_json_file_id"]))
        )
        return result
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
        return result
    finally:
        conn.close()


def _merge_shard_into_primary(
    *,
    shard_name: str,
    shard_db: Path,
    primary_db: Path,
    sqlite_timeout_seconds: int,
) -> dict[str, object]:
    result = {
        "shard": shard_name,
        "shard_db": str(shard_db),
        "primary_db": str(primary_db),
        "ok": True,
        "jsonl_rows_inserted": 0,
        "json_file_rows_inserted": 0,
        "last_jsonl_id": 0,
        "last_json_file_id": 0,
        "jsonl_cursor_reset": False,
        "json_file_cursor_reset": False,
    }
    if not shard_db.exists():
        result["ok"] = False
        result["error"] = "shard_db_missing"
        return result

    conn = sqlite3.connect(str(primary_db), timeout=max(float(sqlite_timeout_seconds), 1.0))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={int(max(float(sqlite_timeout_seconds), 1.0) * 1000)}")
        _ensure_primary_schema(conn)
        last_jsonl_id, last_json_file_id = _read_shard_cursor(conn, shard_name)
        conn.execute("ATTACH DATABASE ? AS sharddb", (str(shard_db),))

        if _table_exists(conn, "sharddb", "jsonl_records"):
            max_jsonl_id = int(
                conn.execute("SELECT COALESCE(MAX(id), 0) FROM sharddb.jsonl_records").fetchone()[0] or 0
            )
            if max_jsonl_id > 0 and last_jsonl_id > max_jsonl_id:
                last_jsonl_id = 0
                result["jsonl_cursor_reset"] = True
            col_list = ",".join(JSONL_COLUMNS)
            conn.execute(
                f"""
                INSERT OR IGNORE INTO main.jsonl_records ({col_list})
                SELECT {col_list}
                FROM sharddb.jsonl_records
                WHERE id > ?
                ORDER BY id
                """,
                (last_jsonl_id,),
            )
            result["jsonl_rows_inserted"] = int(conn.execute("SELECT changes()").fetchone()[0] or 0)
            last_jsonl_id = max_jsonl_id

        if _table_exists(conn, "sharddb", "json_file_records"):
            max_json_file_id = int(
                conn.execute("SELECT COALESCE(MAX(id), 0) FROM sharddb.json_file_records").fetchone()[0] or 0
            )
            if max_json_file_id > 0 and last_json_file_id > max_json_file_id:
                last_json_file_id = 0
                result["json_file_cursor_reset"] = True
            col_list = ",".join(JSON_FILE_COLUMNS)
            conn.execute(
                f"""
                INSERT OR IGNORE INTO main.json_file_records ({col_list})
                SELECT {col_list}
                FROM sharddb.json_file_records
                WHERE id > ?
                ORDER BY id
                """,
                (last_json_file_id,),
            )
            result["json_file_rows_inserted"] = int(conn.execute("SELECT changes()").fetchone()[0] or 0)
            last_json_file_id = max_json_file_id

        conn.execute(
            """
            INSERT INTO shard_merge_state (shard_name, last_jsonl_id, last_json_file_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(shard_name) DO UPDATE SET
                last_jsonl_id = excluded.last_jsonl_id,
                last_json_file_id = excluded.last_json_file_id,
                updated_at = excluded.updated_at
            """,
            (shard_name, last_jsonl_id, last_json_file_id, _now_utc()),
        )
        conn.commit()
        conn.execute("DETACH DATABASE sharddb")
        result["last_jsonl_id"] = last_jsonl_id
        result["last_json_file_id"] = last_json_file_id
        return result
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
        return result
    finally:
        conn.close()


def _run_hot_retention(
    *,
    db_path: Path,
    hot_days: int,
    batch_size: int,
    max_rows: int,
    archive_db: str,
    archive_root: str,
    archive_period: str,
    archive_retention_days: int,
    archive_prune_vacuum: bool,
    cold_export_root: str,
    cold_export_format: str,
    cold_export_batch_size: int,
    cold_export_compression: str,
    vacuum: bool,
) -> tuple[int, str, str]:
    cmd = [
        str(PY),
        str(HOT_RETENTION_SCRIPT),
        "--db",
        str(db_path),
        "--archive-db",
        str(archive_db),
        "--hot-days",
        str(max(hot_days, 1)),
        "--batch-size",
        str(max(batch_size, 1000)),
        "--max-rows",
        str(max(max_rows, 0)),
        "--archive-period",
        str(archive_period or "single"),
        "--archive-retention-days",
        str(max(archive_retention_days, 0)),
        "--json",
    ]
    if archive_prune_vacuum:
        cmd.append("--archive-prune-vacuum")
    if str(archive_root or "").strip():
        cmd.extend(["--archive-root", str(archive_root)])
    if str(cold_export_root or "").strip():
        cmd.extend(
            [
                "--cold-export-root",
                str(cold_export_root),
                "--cold-export-format",
                str(cold_export_format or "parquet"),
                "--cold-export-batch-size",
                str(max(cold_export_batch_size, 1000)),
                "--cold-export-compression",
                str(cold_export_compression or "zstd"),
            ]
        )
    if vacuum:
        cmd.append("--vacuum")
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _run_queue_retention(
    *,
    db_path: str,
    acked_days: int,
    batch_size: int,
    max_rows: int,
    cleanup_consumer_state_days: int,
    prune_orphans: bool,
    orphan_days: int,
    vacuum: bool,
) -> tuple[int, str, str]:
    cmd = [
        str(PY),
        str(QUEUE_RETENTION_SCRIPT),
        "--db",
        str(db_path),
        "--acked-days",
        str(max(acked_days, 1)),
        "--batch-size",
        str(max(batch_size, 1000)),
        "--max-rows",
        str(max(max_rows, 0)),
        "--cleanup-consumer-state-days",
        str(max(cleanup_consumer_state_days, 1)),
        "--json",
    ]
    if prune_orphans:
        cmd.extend(["--prune-orphans", "--orphan-days", str(max(orphan_days, 1))])
    if vacuum:
        cmd.append("--vacuum")
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _run_wal_checkpoint(
    *,
    db_path: Path,
    checkpoint_threshold_gb: float,
    truncate_max_gb: float,
    checkpoint_mode: str,
) -> tuple[int, str, str]:
    cmd = [
        str(PY),
        str(SQLITE_MAINTENANCE_SCRIPT),
        "--db",
        str(db_path),
        "--checkpoint-only",
        "--wal-checkpoint-threshold-gb",
        str(max(checkpoint_threshold_gb, 0.0)),
        "--wal-truncate-max-gb",
        str(max(truncate_max_gb, 0.0)),
        "--wal-checkpoint-mode",
        str(checkpoint_mode or "auto"),
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _build_shards(shard_names: list[str]) -> list[dict[str, object]]:
    day_utc = datetime.now(timezone.utc).strftime("%Y%m%d")
    specs: list[dict[str, object]] = []
    for name in shard_names:
        safe_name = str(name).strip().lower().replace("-", "_")
        if not safe_name:
            continue
        defaults = DEFAULT_SHARD_DEFS.get(safe_name, {})
        include_streams = os.getenv(_shard_env(safe_name, "INCLUDE_STREAMS"), defaults.get("include_streams", ""))
        exclude_streams = os.getenv(_shard_env(safe_name, "EXCLUDE_STREAMS"), defaults.get("exclude_streams", ""))
        path_contains = os.getenv(_shard_env(safe_name, "PATH_CONTAINS"), defaults.get("path_contains", ""))
        path_not_contains = os.getenv(_shard_env(safe_name, "PATH_NOT_CONTAINS"), defaults.get("path_not_contains", ""))
        skip_json_files = _env_flag(_shard_env(safe_name, "SKIP_JSON_FILES"), bool(defaults.get("skip_json_files", True)))
        max_files = max(_env_int(_shard_env(safe_name, "MAX_FILES"), int(defaults.get("max_files", 0) or 0)), 0)
        specs.append(
            {
                "name": safe_name,
                "include_streams": include_streams,
                "exclude_streams": exclude_streams,
                "path_contains": path_contains,
                "path_not_contains": path_not_contains,
                "skip_json_files": skip_json_files,
                "max_files": max_files,
                "sqlite_db": SHARD_DB_ROOT / f"jsonl_link_{safe_name}.sqlite3",
                "state_file": SHARD_STATE_ROOT / f"jsonl_sql_link_state_{safe_name}.json",
                "health_file": HEALTH_ROOT / f"jsonl_sql_ingestion_health_{safe_name}_latest.json",
                "journal_file": HEALTH_ROOT / f"jsonl_ingest_batch_journal_{safe_name}_latest.jsonl",
                "journal_events_file": EVENT_ROOT / f"jsonl_ingest_batches_{safe_name}_{day_utc}.jsonl",
                "invalid_log_file": EVENT_ROOT / f"jsonl_ingestion_invalid_{safe_name}_{day_utc}.jsonl",
            }
        )
    return specs


def _run_shard_links(
    *,
    shards: list[dict[str, object]],
    sqlite_timeout_seconds: int,
    sqlite_lock_retries: int,
    sqlite_lock_retry_delay_seconds: float,
    progress_callback=None,
) -> list[dict[str, object]]:
    recoveries: dict[str, dict[str, object]] = {}
    results: list[dict[str, object]] = []
    for shard in shards:
        recovery = _quarantine_shard_artifacts(
            shard_name=str(shard["name"]),
            sqlite_db=Path(str(shard["sqlite_db"])),
            state_file=Path(str(shard["state_file"])),
            health_file=Path(str(shard["health_file"])),
        )
        recoveries[str(shard["name"])] = recovery
        cmd = [
            str(PY),
            str(LINK_SCRIPT),
            "--project-root",
            str(PROJECT_ROOT),
            "--mode",
            "sqlite",
            "--sqlite-db",
            str(shard["sqlite_db"]),
            "--state-file",
            str(shard["state_file"]),
            "--health-file",
            str(shard["health_file"]),
            "--journal-file",
            str(shard["journal_file"]),
            "--journal-events-file",
            str(shard["journal_events_file"]),
            "--invalid-log-file",
            str(shard["invalid_log_file"]),
            "--sqlite-timeout-seconds",
            str(max(sqlite_timeout_seconds, 30)),
            "--sqlite-lock-retries",
            str(max(sqlite_lock_retries, 0)),
            "--sqlite-lock-retry-delay-seconds",
            str(max(sqlite_lock_retry_delay_seconds, 0.1)),
        ]
        for key, flag in [
            ("include_streams", "--include-streams"),
            ("exclude_streams", "--exclude-streams"),
            ("path_contains", "--path-contains"),
            ("path_not_contains", "--path-not-contains"),
        ]:
            raw = str(shard.get(key, "") or "").strip()
            if raw:
                cmd.extend([flag, raw])
        if int(shard.get("max_files", 0) or 0) > 0:
            cmd.extend(["--max-files", str(int(shard["max_files"]))])
        if bool(shard.get("skip_json_files")):
            cmd.append("--skip-json-files")
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        health = {}
        health_path = Path(str(shard["health_file"]))
        if health_path.exists():
            try:
                health = json.loads(health_path.read_text(encoding="utf-8"))
            except Exception:
                health = {}
        results.append(
            {
                "shard": str(shard["name"]),
                "sqlite_db": str(shard["sqlite_db"]),
                "state_file": str(shard["state_file"]),
                "health_file": str(shard["health_file"]),
                "filters": {
                    "include_streams": _parse_csv(str(shard.get("include_streams", "") or "")),
                    "exclude_streams": _parse_csv(str(shard.get("exclude_streams", "") or "")),
                    "path_contains": _parse_csv(str(shard.get("path_contains", "") or "")),
                    "path_not_contains": _parse_csv(str(shard.get("path_not_contains", "") or "")),
                    "max_files": int(shard.get("max_files", 0) or 0),
                    "skip_json_files": bool(shard.get("skip_json_files")),
                },
                "recovery": recoveries.get(str(shard["name"]), {}),
                "rc": int(proc.returncode),
                "stdout_tail": "\n".join((stdout or "").splitlines()[-20:]),
                "stderr_tail": "\n".join((stderr or "").splitlines()[-20:]),
                "health": health,
            }
        )
        if callable(progress_callback):
            try:
                progress_callback(list(results))
            except Exception:
                pass
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Sharded SQL linker manager with incremental merge back into the primary SQLite DB.")
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("SQL_LINK_SERVICE_INTERVAL_SECONDS", "45")))
    parser.add_argument("--sqlite-timeout-seconds", type=int, default=int(os.getenv("SQL_LINK_SERVICE_SQLITE_TIMEOUT", "300")))
    parser.add_argument("--sqlite-lock-retries", type=int, default=int(os.getenv("SQL_LINK_SERVICE_LOCK_RETRIES", "200")))
    parser.add_argument("--sqlite-lock-retry-delay-seconds", type=float, default=float(os.getenv("SQL_LINK_SERVICE_LOCK_RETRY_DELAY_SECONDS", "0.5")))
    parser.add_argument("--lock-path", default=str(PROJECT_ROOT / "governance" / "locks" / "jsonl_sql_writer.lock"))
    parser.add_argument("--primary-db", default=os.getenv("SQL_LINK_SERVICE_PRIMARY_DB", str(PRIMARY_DB_PATH)))
    parser.add_argument("--shards", default=os.getenv("SQL_LINK_SERVICE_SHARDS", ""))
    parser.add_argument("--auto-wal-checkpoint", action="store_true", default=os.getenv("SQL_LINK_SERVICE_AUTO_WAL_CHECKPOINT", "1") == "1")
    parser.add_argument("--wal-checkpoint-threshold-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB", "2")))
    parser.add_argument("--wal-checkpoint-trigger-growth-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB", "1.5")))
    parser.add_argument("--wal-checkpoint-trigger-rows", type=int, default=int(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_ROWS", "750000")))
    parser.add_argument("--wal-checkpoint-min-interval-seconds", type=int, default=int(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_MIN_INTERVAL_SECONDS", "900")))
    parser.add_argument("--wal-truncate-max-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_WAL_TRUNCATE_MAX_GB", "8")))
    parser.add_argument("--wal-checkpoint-mode", choices=("auto", "passive", "truncate", "restart"), default=os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_MODE", "auto"))
    parser.add_argument("--auto-hot-retention", action="store_true", default=os.getenv("SQL_LINK_SERVICE_AUTO_HOT_RETENTION", "1") == "1")
    parser.add_argument("--hot-retention-max-db-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_HOT_MAX_DB_GB", "25")))
    parser.add_argument("--hot-retention-trigger-growth-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_HOT_TRIGGER_GROWTH_GB", "12")))
    parser.add_argument("--hot-retention-trigger-rows", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_TRIGGER_ROWS", "2500000")))
    parser.add_argument("--hot-retention-hot-days", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_DAYS", "5")))
    parser.add_argument("--hot-retention-batch-size", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_BATCH_SIZE", "120000")))
    parser.add_argument("--hot-retention-max-rows", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_MAX_ROWS", "1000000")))
    parser.add_argument("--hot-retention-min-interval-seconds", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS", "180")))
    parser.add_argument("--hot-retention-vacuum-threshold-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_HOT_VACUUM_THRESHOLD_GB", "150")))
    parser.add_argument("--hot-retention-archive-db", default=os.getenv("SQL_LINK_SERVICE_HOT_ARCHIVE_DB", str(PROJECT_ROOT / "data" / "jsonl_link_archive.sqlite3")))
    parser.add_argument("--hot-retention-archive-root", default=os.getenv("SQL_LINK_SERVICE_HOT_ARCHIVE_ROOT", str(PROJECT_ROOT / "data" / "jsonl_link_archives")))
    parser.add_argument("--hot-retention-archive-period", choices=("single", "day", "month"), default=os.getenv("SQL_LINK_SERVICE_HOT_ARCHIVE_PERIOD", "day"))
    parser.add_argument("--hot-retention-archive-retention-days", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_ARCHIVE_RETENTION_DAYS", "365")))
    parser.add_argument("--hot-retention-archive-prune-vacuum", action="store_true", default=os.getenv("SQL_LINK_SERVICE_HOT_ARCHIVE_PRUNE_VACUUM", "1") == "1")
    parser.add_argument("--hot-retention-cold-export-root", default=os.getenv("SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_ROOT", ""))
    parser.add_argument("--hot-retention-cold-export-format", choices=("parquet",), default=os.getenv("SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_FORMAT", "parquet"))
    parser.add_argument("--hot-retention-cold-export-batch-size", type=int, default=int(os.getenv("SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_BATCH_SIZE", "50000")))
    parser.add_argument("--hot-retention-cold-export-compression", default=os.getenv("SQL_LINK_SERVICE_HOT_COLD_ARCHIVE_COMPRESSION", "zstd"))
    parser.add_argument("--auto-queue-retention", action="store_true", default=os.getenv("SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION", "1") == "1")
    parser.add_argument("--queue-retention-db", default=os.getenv("SQL_LINK_SERVICE_QUEUE_DB", str(QUEUE_DB_PATH)))
    parser.add_argument("--queue-retention-max-db-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_QUEUE_MAX_DB_GB", "10")))
    parser.add_argument("--queue-retention-acked-days", type=int, default=int(os.getenv("SQL_LINK_SERVICE_QUEUE_ACKED_DAYS", "7")))
    parser.add_argument("--queue-retention-batch-size", type=int, default=int(os.getenv("SQL_LINK_SERVICE_QUEUE_BATCH_SIZE", "80000")))
    parser.add_argument("--queue-retention-max-rows", type=int, default=int(os.getenv("SQL_LINK_SERVICE_QUEUE_MAX_ROWS", "240000")))
    parser.add_argument("--queue-retention-min-interval-seconds", type=int, default=int(os.getenv("SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS", "900")))
    parser.add_argument("--queue-retention-vacuum-threshold-gb", type=float, default=float(os.getenv("SQL_LINK_SERVICE_QUEUE_VACUUM_THRESHOLD_GB", "20")))
    parser.add_argument("--queue-retention-cleanup-consumer-state-days", type=int, default=int(os.getenv("SQL_LINK_SERVICE_QUEUE_CLEANUP_CONSUMER_STATE_DAYS", "30")))
    parser.add_argument("--queue-retention-prune-orphans", action="store_true", default=os.getenv("SQL_LINK_SERVICE_QUEUE_PRUNE_ORPHANS", "0") == "1")
    parser.add_argument("--queue-retention-orphan-days", type=int, default=int(os.getenv("SQL_LINK_SERVICE_QUEUE_ORPHAN_DAYS", "45")))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    shard_names = _parse_csv(_normalized_shard_config(args.shards))
    if not shard_names:
        msg = {"ok": False, "reason": "no_shards_configured"}
        print(json.dumps(msg, ensure_ascii=True) if args.json else "sql_link_shard_manager no shards configured")
        return 2

    lock_path = Path(args.lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        owner = fh.read().strip()
        msg = {"ok": False, "reason": "writer_lock_busy", "lock_path": str(lock_path), "owner": owner}
        progress = _busy_progress_summary()
        if progress:
            msg["service_progress"] = progress
        if args.json:
            print(json.dumps(msg, ensure_ascii=True))
        else:
            detail = ""
            if progress:
                detail = (
                    f" current_step={progress.get('current_step', 'unknown')}"
                    f" completed_shards={progress.get('completed_shard_count', 0)}"
                    f" completed_merges={progress.get('completed_merge_count', 0)}"
                    f" merged_rows={progress.get('merged_rows_this_cycle', 0)}"
                )
            print(f"sql_link_shard_manager busy owner={owner or 'unknown'}{detail}")
        return 0

    fh.seek(0)
    fh.truncate(0)
    fh.write(f"pid={os.getpid()} started={_now_utc()} cmd=sql_link_shard_manager")
    fh.flush()

    primary_db = Path(args.primary_db).resolve()
    primary_db.parent.mkdir(parents=True, exist_ok=True)
    SHARD_DB_ROOT.mkdir(parents=True, exist_ok=True)
    SHARD_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    HEALTH_ROOT.mkdir(parents=True, exist_ok=True)
    EVENT_ROOT.mkdir(parents=True, exist_ok=True)

    maintenance_state = _load_maintenance_state(
        MAINTENANCE_STATE_PATH,
        db_size_gb=_db_size_gb(primary_db),
        wal_size_gb=_wal_size_gb(primary_db),
    )

    last_wal_checkpoint_ts = 0.0
    last_hot_retention_ts = 0.0
    last_queue_retention_ts = 0.0

    while True:
        ts = _now_utc()
        cycle_ts = time.time()
        shards = _build_shards(shard_names)
        merge_results: list[dict[str, object]] = []
        _write_service_progress(
            cycle_started_utc=ts,
            current_step="shard_linking",
            lock_path=lock_path,
            primary_db=primary_db,
            shards=shards,
            shard_results=[],
            merge_results=merge_results,
            merged_rows_this_cycle=0,
            running=True,
            ok=True,
        )
        shard_results = _run_shard_links(
            shards=shards,
            sqlite_timeout_seconds=int(args.sqlite_timeout_seconds),
            sqlite_lock_retries=int(args.sqlite_lock_retries),
            sqlite_lock_retry_delay_seconds=float(args.sqlite_lock_retry_delay_seconds),
            progress_callback=lambda rows: _write_service_progress(
                cycle_started_utc=ts,
                current_step="shard_linking",
                lock_path=lock_path,
                primary_db=primary_db,
                shards=shards,
                shard_results=rows,
                merge_results=merge_results,
                merged_rows_this_cycle=0,
                running=True,
                ok=True,
            ),
        )

        _write_service_progress(
            cycle_started_utc=ts,
            current_step="merge_primary",
            lock_path=lock_path,
            primary_db=primary_db,
            shards=shards,
            shard_results=shard_results,
            merge_results=merge_results,
            merged_rows_this_cycle=0,
            running=True,
            ok=True,
        )
        for shard in shards:
            result = next((row for row in shard_results if row["shard"] == shard["name"]), None)
            if result and int(result.get("rc", 1)) == 0:
                merge_probe = _probe_shard_merge_state(
                    shard_name=str(shard["name"]),
                    shard_db=Path(str(shard["sqlite_db"])),
                    primary_db=primary_db,
                    sqlite_timeout_seconds=int(args.sqlite_timeout_seconds),
                )
                if bool(merge_probe.get("ok", False)) and not bool(merge_probe.get("merge_required", True)):
                    merge_results.append(
                        {
                            "shard": str(shard["name"]),
                            "shard_db": str(shard["sqlite_db"]),
                            "primary_db": str(primary_db),
                            "ok": True,
                            "skipped": True,
                            "reason": "merge_up_to_date",
                            "jsonl_rows_inserted": 0,
                            "json_file_rows_inserted": 0,
                            "last_jsonl_id": int(merge_probe.get("last_jsonl_id", 0) or 0),
                            "last_json_file_id": int(merge_probe.get("last_json_file_id", 0) or 0),
                            "max_jsonl_id": int(merge_probe.get("max_jsonl_id", 0) or 0),
                            "max_json_file_id": int(merge_probe.get("max_json_file_id", 0) or 0),
                        }
                    )
                else:
                    merge_results.append(
                        _merge_shard_into_primary(
                            shard_name=str(shard["name"]),
                            shard_db=Path(str(shard["sqlite_db"])),
                            primary_db=primary_db,
                            sqlite_timeout_seconds=int(args.sqlite_timeout_seconds),
                        )
                    )
                _write_service_progress(
                    cycle_started_utc=ts,
                    current_step="merge_primary",
                    lock_path=lock_path,
                    primary_db=primary_db,
                    shards=shards,
                    shard_results=shard_results,
                    merge_results=merge_results,
                    merged_rows_this_cycle=_merged_rows_inserted(merge_results),
                    running=True,
                    ok=True,
                )

        overall_rc = 0 if all(int(row.get("rc", 1)) == 0 for row in shard_results) and all(bool(row.get("ok", False)) for row in merge_results) else 1
        merged_rows = _merged_rows_inserted(merge_results)
        for key in ("wal_checkpoint", "hot_retention"):
            bucket = maintenance_state.get(key, {})
            if isinstance(bucket, dict):
                bucket["rows_since_last_run"] = _as_int(bucket.get("rows_since_last_run"), 0) + int(merged_rows)

        wal_size = _wal_size_gb(primary_db)
        checkpoint_state = maintenance_state.get("wal_checkpoint", {}) if isinstance(maintenance_state.get("wal_checkpoint"), dict) else {}
        checkpoint_rows_since_last = _as_int(checkpoint_state.get("rows_since_last_run"), 0)
        checkpoint_wal_growth_gb = max(wal_size - _as_float(checkpoint_state.get("baseline_wal_size_gb"), wal_size), 0.0)
        checkpoint_trigger_reasons = _wal_checkpoint_trigger_reasons(
            wal_size_gb=wal_size,
            wal_threshold_gb=float(args.wal_checkpoint_threshold_gb),
            wal_growth_gb=checkpoint_wal_growth_gb,
            wal_growth_trigger_gb=float(args.wal_checkpoint_trigger_growth_gb),
            rows_since_last_run=checkpoint_rows_since_last,
            row_trigger=int(args.wal_checkpoint_trigger_rows),
        )
        wal_checkpoint = {
            "enabled": bool(args.auto_wal_checkpoint),
            "wal_size_gb_before": round(wal_size, 3),
            "threshold_gb": float(args.wal_checkpoint_threshold_gb),
            "trigger_growth_gb": float(args.wal_checkpoint_trigger_growth_gb),
            "trigger_rows": int(args.wal_checkpoint_trigger_rows),
            "truncate_max_gb": float(args.wal_truncate_max_gb),
            "mode": str(args.wal_checkpoint_mode),
            "rows_since_last_run": int(checkpoint_rows_since_last),
            "wal_growth_gb_since_last_run": round(checkpoint_wal_growth_gb, 3),
            "trigger_reasons": list(checkpoint_trigger_reasons),
            "ran": False,
            "rc": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "details": {},
            "skipped_reason": "",
        }
        if args.auto_wal_checkpoint and overall_rc == 0 and wal_size <= 0.0:
            wal_checkpoint["skipped_reason"] = "no_wal"
        elif args.auto_wal_checkpoint and overall_rc == 0 and checkpoint_trigger_reasons:
            since_last = cycle_ts - float(last_wal_checkpoint_ts)
            if since_last >= max(int(args.wal_checkpoint_min_interval_seconds), 60):
                rc, out, err = _run_wal_checkpoint(
                    db_path=primary_db,
                    checkpoint_threshold_gb=float(args.wal_checkpoint_threshold_gb),
                    truncate_max_gb=float(args.wal_truncate_max_gb),
                    checkpoint_mode=str(args.wal_checkpoint_mode),
                )
                wal_checkpoint.update(
                    {
                        "ran": True,
                        "rc": int(rc),
                        "stdout_tail": "\n".join(out.splitlines()[-12:]),
                        "stderr_tail": "\n".join(err.splitlines()[-12:]),
                        "details": _parse_json_output(out),
                    }
                )
                last_wal_checkpoint_ts = cycle_ts
                if int(rc) == 0:
                    checkpoint_state.update(
                        {
                            "last_run_utc": ts,
                            "baseline_db_size_gb": round(_db_size_gb(primary_db), 3),
                            "baseline_wal_size_gb": round(_wal_size_gb(primary_db), 3),
                            "rows_since_last_run": 0,
                            "last_trigger_reasons": list(checkpoint_trigger_reasons),
                        }
                    )
            else:
                wal_checkpoint["skipped_reason"] = f"min_interval_not_met:{int(since_last)}s"
        elif args.auto_wal_checkpoint:
            wal_checkpoint["skipped_reason"] = "link_failed" if overall_rc != 0 else "below_data_trigger"
        wal_checkpoint["wal_size_gb_after"] = round(_wal_size_gb(primary_db), 3)

        archive_blockers = _archive_maintenance_blockers(str(args.hot_retention_archive_root or ""))
        db_size = _db_size_gb(primary_db)
        hot_state = maintenance_state.get("hot_retention", {}) if isinstance(maintenance_state.get("hot_retention"), dict) else {}
        hot_rows_since_last = _as_int(hot_state.get("rows_since_last_run"), 0)
        hot_db_growth_gb = max(db_size - _as_float(hot_state.get("baseline_db_size_gb"), db_size), 0.0)
        hot_trigger_reasons = _hot_retention_trigger_reasons(
            db_size_gb=db_size,
            max_db_gb=float(args.hot_retention_max_db_gb),
            db_growth_gb=hot_db_growth_gb,
            growth_trigger_gb=float(args.hot_retention_trigger_growth_gb),
            rows_since_last_run=hot_rows_since_last,
            row_trigger=int(args.hot_retention_trigger_rows),
            has_successful_run=bool(str(hot_state.get("last_run_utc") or "").strip()),
        )
        hot_retention = {
            "enabled": bool(args.auto_hot_retention),
            "db_size_gb_before": round(db_size, 3),
            "max_db_gb": float(args.hot_retention_max_db_gb),
            "trigger_growth_gb": float(args.hot_retention_trigger_growth_gb),
            "trigger_rows": int(args.hot_retention_trigger_rows),
            "archive_db": str(args.hot_retention_archive_db),
            "archive_root": str(args.hot_retention_archive_root or ""),
            "archive_period": str(args.hot_retention_archive_period),
            "archive_retention_days": int(args.hot_retention_archive_retention_days),
            "archive_prune_vacuum": bool(args.hot_retention_archive_prune_vacuum),
            "cold_export_root": str(args.hot_retention_cold_export_root or ""),
            "cold_export_format": str(args.hot_retention_cold_export_format),
            "batch_size": int(args.hot_retention_batch_size),
            "max_rows": int(args.hot_retention_max_rows),
            "rows_since_last_run": int(hot_rows_since_last),
            "db_growth_gb_since_last_run": round(hot_db_growth_gb, 3),
            "trigger_reasons": list(hot_trigger_reasons),
            "ran": False,
            "rc": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "details": {},
            "skipped_reason": "",
            "maintenance_blockers": archive_blockers,
        }
        if archive_blockers:
            hot_retention["skipped_reason"] = "archive_maintenance_blocked"
        elif args.auto_hot_retention and overall_rc == 0 and hot_trigger_reasons:
            since_last = cycle_ts - float(last_hot_retention_ts)
            if since_last >= max(int(args.hot_retention_min_interval_seconds), 60):
                do_vacuum = db_size >= float(args.hot_retention_vacuum_threshold_gb)
                rc, out, err = _run_hot_retention(
                    db_path=primary_db,
                    hot_days=int(args.hot_retention_hot_days),
                    batch_size=int(args.hot_retention_batch_size),
                    max_rows=int(args.hot_retention_max_rows),
                    archive_db=str(args.hot_retention_archive_db),
                    archive_root=str(args.hot_retention_archive_root or ""),
                    archive_period=str(args.hot_retention_archive_period),
                    archive_retention_days=int(args.hot_retention_archive_retention_days),
                    archive_prune_vacuum=bool(args.hot_retention_archive_prune_vacuum),
                    cold_export_root=str(args.hot_retention_cold_export_root or ""),
                    cold_export_format=str(args.hot_retention_cold_export_format),
                    cold_export_batch_size=int(args.hot_retention_cold_export_batch_size),
                    cold_export_compression=str(args.hot_retention_cold_export_compression),
                    vacuum=do_vacuum,
                )
                hot_retention.update(
                    {
                        "ran": True,
                        "rc": int(rc),
                        "stdout_tail": "\n".join(out.splitlines()[-12:]),
                        "stderr_tail": "\n".join(err.splitlines()[-12:]),
                        "details": _parse_json_output(out),
                        "vacuum": bool(do_vacuum),
                    }
                )
                last_hot_retention_ts = cycle_ts
                if int(rc) == 0:
                    hot_state.update(
                        {
                            "last_run_utc": ts,
                            "baseline_db_size_gb": round(_db_size_gb(primary_db), 3),
                            "baseline_wal_size_gb": round(_wal_size_gb(primary_db), 3),
                            "rows_since_last_run": 0,
                            "last_trigger_reasons": list(hot_trigger_reasons),
                        }
                    )
            else:
                hot_retention["skipped_reason"] = f"min_interval_not_met:{int(since_last)}s"
        elif args.auto_hot_retention:
            hot_retention["skipped_reason"] = "below_data_trigger" if overall_rc == 0 else "link_failed"
        hot_retention["db_size_gb_after"] = round(_db_size_gb(primary_db), 3)

        queue_db_path = Path(str(args.queue_retention_db))
        queue_db_size = _db_size_gb(queue_db_path)
        queue_retention = {
            "enabled": bool(args.auto_queue_retention),
            "db_path": str(queue_db_path),
            "db_size_gb_before": round(queue_db_size, 3),
            "max_db_gb": float(args.queue_retention_max_db_gb),
            "acked_days": int(args.queue_retention_acked_days),
            "batch_size": int(args.queue_retention_batch_size),
            "max_rows": int(args.queue_retention_max_rows),
            "prune_orphans": bool(args.queue_retention_prune_orphans),
            "orphan_days": int(args.queue_retention_orphan_days),
            "cleanup_consumer_state_days": int(args.queue_retention_cleanup_consumer_state_days),
            "ran": False,
            "rc": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "details": {},
            "skipped_reason": "",
        }
        if args.auto_queue_retention and overall_rc == 0 and queue_db_path.exists() and queue_db_size >= float(args.queue_retention_max_db_gb):
            since_last = cycle_ts - float(last_queue_retention_ts)
            if since_last >= max(int(args.queue_retention_min_interval_seconds), 60):
                do_vacuum = queue_db_size >= float(args.queue_retention_vacuum_threshold_gb)
                rc, out, err = _run_queue_retention(
                    db_path=str(queue_db_path),
                    acked_days=int(args.queue_retention_acked_days),
                    batch_size=int(args.queue_retention_batch_size),
                    max_rows=int(args.queue_retention_max_rows),
                    cleanup_consumer_state_days=int(args.queue_retention_cleanup_consumer_state_days),
                    prune_orphans=bool(args.queue_retention_prune_orphans),
                    orphan_days=int(args.queue_retention_orphan_days),
                    vacuum=do_vacuum,
                )
                queue_retention.update(
                    {
                        "ran": True,
                        "rc": int(rc),
                        "stdout_tail": "\n".join(out.splitlines()[-12:]),
                        "stderr_tail": "\n".join(err.splitlines()[-12:]),
                        "details": _parse_json_output(out),
                        "vacuum": bool(do_vacuum),
                    }
                )
                last_queue_retention_ts = cycle_ts
            else:
                queue_retention["skipped_reason"] = f"min_interval_not_met:{int(since_last)}s"
        elif args.auto_queue_retention:
            if not queue_db_path.exists():
                queue_retention["skipped_reason"] = "db_missing"
            else:
                queue_retention["skipped_reason"] = "db_below_threshold" if overall_rc == 0 else "link_failed"
        queue_retention["db_size_gb_after"] = round(_db_size_gb(queue_db_path), 3)

        payload = {
            "timestamp_utc": ts,
            "ok": overall_rc == 0,
            "rc": int(overall_rc),
            "mode": "sharded_merge",
            "lock_path": str(lock_path),
            "primary_db": str(primary_db),
            "sqlite_db_size_gb": round(_db_size_gb(primary_db), 3),
            "sqlite_wal_size_gb": round(_wal_size_gb(primary_db), 3),
            "queue_db_size_gb": round(_db_size_gb(queue_db_path), 3),
            "maintenance_state_path": str(MAINTENANCE_STATE_PATH),
            "merged_rows_this_cycle": int(merged_rows),
            "shards": shard_results,
            "merge_results": merge_results,
            "wal_checkpoint": wal_checkpoint,
            "hot_retention": hot_retention,
            "queue_retention": queue_retention,
            "archive_maintenance_blockers": archive_blockers,
        }
        LATEST_HEALTH.parent.mkdir(parents=True, exist_ok=True)
        LATEST_HEALTH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        _write_service_progress(
            cycle_started_utc=ts,
            current_step="complete",
            lock_path=lock_path,
            primary_db=primary_db,
            shards=shards,
            shard_results=shard_results,
            merge_results=merge_results,
            merged_rows_this_cycle=int(merged_rows),
            running=False,
            ok=overall_rc == 0,
        )
        maintenance_state["timestamp_utc"] = ts
        _write_json(MAINTENANCE_STATE_PATH, maintenance_state)

        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(f"sql_link_shard_manager rc={overall_rc} ok={overall_rc == 0} ts={ts}")

        if args.once:
            break
        time.sleep(max(int(args.interval_seconds), 10))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
