import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "sqlite_maintenance_latest.json"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _sqlite_exec_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple = (),
    *,
    lock_retries: int,
    lock_retry_delay_seconds: float,
):
    attempt = 0
    while True:
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            is_locked = ("database is locked" in msg) or ("database table is locked" in msg)
            if (not is_locked) or attempt >= max(lock_retries, 0):
                raise
            sleep_s = min(max(lock_retry_delay_seconds, 0.01) * (2 ** attempt), 5.0)
            print(
                f"SQLite busy during maintenance; retrying in {sleep_s:.2f}s "
                f"(attempt {attempt + 1}/{max(lock_retries, 0)})"
            )
            time.sleep(sleep_s)
            attempt += 1


def _sqlite_sidecar_path(db_path: Path, suffix: str) -> Path:
    return Path(f"{db_path}{suffix}")


def _size_gb(path: Path) -> float:
    try:
        return float(path.stat().st_size) / (1024 ** 3)
    except Exception:
        return 0.0


def _normalize_checkpoint_mode(raw: str) -> str:
    mode = str(raw or "auto").strip().lower()
    if mode in {"auto", "passive", "truncate", "restart"}:
        return mode
    return "auto"


def _checkpoint_mode_for_wal(wal_size_gb: float, requested_mode: str, truncate_max_gb: float) -> str:
    mode = _normalize_checkpoint_mode(requested_mode)
    if wal_size_gb <= 0.0:
        return ""
    if mode != "auto":
        return mode
    return "truncate" if wal_size_gb <= max(float(truncate_max_gb), 0.0) else "passive"


def _emit(payload: dict, out_path: Path, as_json: bool) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if as_json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "sqlite_maintenance "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"vacuum_ran={int(bool(payload.get('vacuum_ran', False)))} "
            f"size_gb_after={payload.get('size_gb_after', 'n/a')}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply SQLite performance tuning and maintenance.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--vacuum", action="store_true")
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--wal-checkpoint-threshold-gb", type=float, default=float(os.getenv("SQLITE_WAL_CHECKPOINT_THRESHOLD_GB", "0.25")))
    parser.add_argument("--wal-truncate-max-gb", type=float, default=float(os.getenv("SQLITE_WAL_TRUNCATE_MAX_GB", "8")))
    parser.add_argument("--wal-checkpoint-mode", choices=("auto", "passive", "truncate", "restart"), default=_normalize_checkpoint_mode(os.getenv("SQLITE_WAL_CHECKPOINT_MODE", "auto")))
    parser.add_argument("--auto-vacuum-over-gb", type=float, default=float(os.getenv("SQLITE_AUTO_VACUUM_OVER_GB", "24")))
    parser.add_argument("--vacuum-min-interval-hours", type=float, default=float(os.getenv("SQLITE_VACUUM_MIN_INTERVAL_HOURS", "24")))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--sqlite-timeout-seconds", type=float, default=float(os.getenv("SQLITE_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--sqlite-lock-retries", type=int, default=int(os.getenv("SQLITE_LOCK_RETRIES", "8")))
    parser.add_argument("--sqlite-lock-retry-delay-seconds", type=float, default=float(os.getenv("SQLITE_LOCK_RETRY_DELAY_SECONDS", "0.25")))
    args = parser.parse_args()

    db_path = Path(args.db)
    out_path = Path(args.out_file)
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    if not db_path.exists():
        payload = {
            "timestamp_utc": timestamp_utc,
            "ok": False,
            "db_path": str(db_path),
            "error": f"db_missing:{db_path}",
            "vacuum_ran": False,
            "indexes_touched": 0,
            "jsonl_records_rows": 0,
            "size_gb_before": 0.0,
            "size_gb_after": 0.0,
            "auto_vacuum_over_gb": float(args.auto_vacuum_over_gb),
            "vacuum_min_interval_hours": float(args.vacuum_min_interval_hours),
        }
        return _emit(payload, out_path, args.json)

    conn = None
    created_indexes = 0
    total_rows = 0
    do_vacuum = False
    size_gb_before = db_path.stat().st_size / (1024 ** 3)
    wal_path = _sqlite_sidecar_path(db_path, "-wal")
    wal_size_gb_before = _size_gb(wal_path)

    payload = {
        "timestamp_utc": timestamp_utc,
        "ok": False,
        "db_path": str(db_path),
        "vacuum_ran": False,
        "indexes_touched": 0,
        "jsonl_records_rows": 0,
        "size_gb_before": round(size_gb_before, 3),
        "size_gb_after": round(size_gb_before, 3),
        "wal_size_gb_before": round(wal_size_gb_before, 3),
        "wal_size_gb_after": round(wal_size_gb_before, 3),
        "checkpoint_only": bool(args.checkpoint_only),
        "checkpoint_ran": False,
        "checkpoint_mode_requested": _normalize_checkpoint_mode(args.wal_checkpoint_mode),
        "checkpoint_mode_applied": "",
        "checkpoint_result": {},
        "checkpoint_skipped_reason": "",
        "wal_checkpoint_threshold_gb": float(args.wal_checkpoint_threshold_gb),
        "wal_truncate_max_gb": float(args.wal_truncate_max_gb),
        "auto_vacuum_over_gb": float(args.auto_vacuum_over_gb),
        "vacuum_min_interval_hours": float(args.vacuum_min_interval_hours),
    }

    try:
        conn = sqlite3.connect(str(db_path), timeout=max(float(args.sqlite_timeout_seconds), 1.0))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-20000")
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute(f"PRAGMA busy_timeout={int(max(float(args.sqlite_timeout_seconds), 1.0) * 1000)}")

        if (not args.checkpoint_only) and _table_exists(conn, "jsonl_records"):
            idx_sql = [
                "CREATE INDEX IF NOT EXISTS idx_jsonl_source_rel_ingested ON jsonl_records(source_rel, ingested_at)",
                "CREATE INDEX IF NOT EXISTS idx_jsonl_source_rel_line ON jsonl_records(source_rel, line_no)",
                "CREATE INDEX IF NOT EXISTS idx_jsonl_action_expr ON jsonl_records((json_extract(payload_json, '$.action')))",
                "CREATE INDEX IF NOT EXISTS idx_jsonl_symbol_expr ON jsonl_records((json_extract(payload_json, '$.symbol')))",
                "CREATE INDEX IF NOT EXISTS idx_jsonl_ts_expr ON jsonl_records((json_extract(payload_json, '$.timestamp_utc')))",
            ]
            for sql in idx_sql:
                _sqlite_exec_with_retry(
                    conn,
                    sql,
                    lock_retries=max(args.sqlite_lock_retries, 0),
                    lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                )
                created_indexes += 1

        if (not args.checkpoint_only) and _table_exists(conn, "json_file_records"):
            idx_sql = [
                "CREATE INDEX IF NOT EXISTS idx_json_file_source_rel_ingested ON json_file_records(source_rel, ingested_at)",
                "CREATE INDEX IF NOT EXISTS idx_json_file_stream_ingested ON json_file_records(stream, ingested_at)",
            ]
            for sql in idx_sql:
                _sqlite_exec_with_retry(
                    conn,
                    sql,
                    lock_retries=max(args.sqlite_lock_retries, 0),
                    lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                )
                created_indexes += 1

        _sqlite_exec_with_retry(
            conn,
            """
            CREATE TABLE IF NOT EXISTS db_maintenance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                db_path TEXT NOT NULL,
                vacuum_ran INTEGER NOT NULL,
                indexes_touched INTEGER NOT NULL,
                notes TEXT NOT NULL
            )
            """,
            lock_retries=max(args.sqlite_lock_retries, 0),
            lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
        )

        if not args.checkpoint_only:
            _sqlite_exec_with_retry(
                conn,
                "ANALYZE",
                lock_retries=max(args.sqlite_lock_retries, 0),
                lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
            )
            _sqlite_exec_with_retry(
                conn,
                "PRAGMA optimize",
                lock_retries=max(args.sqlite_lock_retries, 0),
                lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
            )

        checkpoint_threshold_gb = max(float(args.wal_checkpoint_threshold_gb), 0.0)
        checkpoint_mode_applied = ""
        if wal_size_gb_before <= 0.0:
            payload["checkpoint_skipped_reason"] = "no_wal"
        elif wal_size_gb_before < checkpoint_threshold_gb:
            payload["checkpoint_skipped_reason"] = "wal_below_threshold"
        else:
            checkpoint_mode_applied = _checkpoint_mode_for_wal(
                wal_size_gb=wal_size_gb_before,
                requested_mode=str(args.wal_checkpoint_mode),
                truncate_max_gb=float(args.wal_truncate_max_gb),
            )
            if checkpoint_mode_applied:
                row = _sqlite_exec_with_retry(
                    conn,
                    f"PRAGMA wal_checkpoint({checkpoint_mode_applied.upper()})",
                    lock_retries=max(args.sqlite_lock_retries, 0),
                    lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                ).fetchone()
                payload["checkpoint_ran"] = True
                payload["checkpoint_mode_applied"] = checkpoint_mode_applied
                payload["checkpoint_result"] = {
                    "busy": int(row[0] if row and len(row) > 0 else 0),
                    "log_frames": int(row[1] if row and len(row) > 1 else 0),
                    "checkpointed_frames": int(row[2] if row and len(row) > 2 else 0),
                }
            else:
                payload["checkpoint_skipped_reason"] = "checkpoint_mode_unresolved"

        last_vacuum_ts = None
        if not args.checkpoint_only:
            try:
                row = _sqlite_exec_with_retry(
                    conn,
                    "SELECT timestamp_utc FROM db_maintenance_events WHERE vacuum_ran=1 ORDER BY id DESC LIMIT 1",
                    lock_retries=max(args.sqlite_lock_retries, 0),
                    lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
                ).fetchone()
                if row and row[0]:
                    last_vacuum_ts = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                last_vacuum_ts = None

        do_vacuum = bool(args.vacuum and (not args.checkpoint_only))
        if (not args.checkpoint_only) and (not do_vacuum) and size_gb_before >= float(args.auto_vacuum_over_gb):
            if last_vacuum_ts is None:
                do_vacuum = True
            else:
                elapsed_h = (datetime.now(timezone.utc) - last_vacuum_ts).total_seconds() / 3600.0
                do_vacuum = elapsed_h >= float(args.vacuum_min_interval_hours)

        if do_vacuum:
            _sqlite_exec_with_retry(
                conn,
                "VACUUM",
                lock_retries=max(args.sqlite_lock_retries, 0),
                lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
            )

        if _table_exists(conn, "jsonl_records"):
            row = _sqlite_exec_with_retry(
                conn,
                "SELECT COUNT(*) FROM jsonl_records",
                lock_retries=max(args.sqlite_lock_retries, 0),
                lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
            ).fetchone()
            total_rows = int(row[0] if row else 0)

        _sqlite_exec_with_retry(
            conn,
            "INSERT INTO db_maintenance_events(timestamp_utc, db_path, vacuum_ran, indexes_touched, notes) VALUES (?, ?, ?, ?, ?)",
            (
                timestamp_utc,
                str(db_path),
                1 if do_vacuum else 0,
                created_indexes,
                (f"checkpoint_only:{payload['checkpoint_mode_applied'] or payload['checkpoint_skipped_reason']}" if args.checkpoint_only else "auto_maintenance"),
            ),
            lock_retries=max(args.sqlite_lock_retries, 0),
            lock_retry_delay_seconds=max(args.sqlite_lock_retry_delay_seconds, 0.01),
        )
        conn.commit()

        size_gb_after = db_path.stat().st_size / (1024 ** 3)
        wal_size_gb_after = _size_gb(wal_path)
        payload.update(
            {
                "ok": True,
                "vacuum_ran": bool(do_vacuum),
                "indexes_touched": int(created_indexes),
                "jsonl_records_rows": int(total_rows),
                "size_gb_before": round(size_gb_before, 3),
                "size_gb_after": round(size_gb_after, 3),
                "wal_size_gb_after": round(wal_size_gb_after, 3),
            }
        )
    except Exception as exc:
        size_gb_after = db_path.stat().st_size / (1024 ** 3) if db_path.exists() else 0.0
        wal_size_gb_after = _size_gb(wal_path)
        payload.update(
            {
                "ok": False,
                "error": str(exc),
                "vacuum_ran": bool(do_vacuum),
                "indexes_touched": int(created_indexes),
                "jsonl_records_rows": int(total_rows),
                "size_gb_after": round(size_gb_after, 3),
                "wal_size_gb_after": round(wal_size_gb_after, 3),
            }
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    return _emit(payload, out_path, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
