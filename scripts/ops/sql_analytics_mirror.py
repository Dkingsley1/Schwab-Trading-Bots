#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sqlite_runtime import connect_sqlite
from core.storage_router import inspect_storage_path
from scripts import ops_data_plane
from scripts.ops.long_runtime_common import run_bounded_process_group, write_payload


DEFAULT_SOURCE_DB = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"
DEFAULT_DUCKDB_PATH = PROJECT_ROOT / "data" / "analytics_mirror.duckdb"
DEFAULT_OUT = (
    PROJECT_ROOT / "governance" / "health" / "sql_analytics_mirror_latest.json"
)


def _local_route(path: Path, root: Path) -> Path:
    route = inspect_storage_path(path, boundary_root=root, allow_external=False)
    if route.get("status") not in {"present", "missing"}:
        raise ValueError(f"analytics_route_rejected:{route.get('status')}")
    return Path(str(route["resolved_path"]))


def _data_route(path: Path) -> Path:
    route = inspect_storage_path(path)
    if route.get("status") not in {"present", "missing"}:
        raise ValueError(f"analytics_route_rejected:{route.get('status')}")
    resolved = Path(str(route["resolved_path"]))
    if resolved.parts[:2] == ("/", "Volumes"):
        volume = Path(*resolved.parts[:3])
        if not volume.is_mount():
            raise ValueError("analytics_external_volume_unavailable")
    return resolved


def _check_headroom(root: Path) -> None:
    floor = max(float(os.getenv("BOT_LOCAL_STORAGE_PRESSURE_FREE_GB", "64")), 64.0)
    if not floor < float("inf"):
        raise ValueError("analytics_storage_threshold_invalid")
    # Reserve a bounded work allowance above the pressure floor, never below it.
    if shutil.disk_usage(root).free / 1024**3 < floor + 1.0:
        raise RuntimeError("analytics_storage_headroom_required")


def _deferred(reason: str) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": False,
        "overall_status": "deferred",
        "summary_refresh_ok": False,
        "summary_refresh_error": reason,
        "partial_refresh_not_accepted": True,
        "live_execution_authority": False,
    }


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _write_duckdb_mirror(*, ops_db_path: Path, duckdb_path: Path) -> dict[str, Any]:
    try:
        import duckdb
    except Exception as exc:
        return {
            "duckdb_available": False,
            "mirror_ready": False,
            "error": f"duckdb_import_failed:{exc}",
            "stream_summary_rows": 0,
            "symbol_summary_rows": 0,
        }

    try:
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        stream_rows: list[tuple[Any, ...]] = []
        symbol_rows: list[tuple[Any, ...]] = []
        with closing(connect_sqlite(ops_db_path, project_root=PROJECT_ROOT, timeout_seconds=30.0, query_only=True, readonly=True)) as sqlite_conn:
            # Both summaries must come from the same committed source snapshot.
            sqlite_conn.execute("BEGIN")
            stream_rows = sqlite_conn.execute(
                """
                SELECT day_utc, stream, record_count, distinct_sources, min_schema_version,
                       max_schema_version, last_ingested_at, refreshed_utc
                FROM materialized_stream_daily
                ORDER BY day_utc DESC, stream ASC
                """
            ).fetchall()
            symbol_rows = sqlite_conn.execute(
                """
                SELECT day_utc, symbol, record_count, buy_count, sell_count, hold_count,
                       last_ingested_at, refreshed_utc
                FROM materialized_symbol_daily
                ORDER BY day_utc DESC, symbol ASC
                """
            ).fetchall()

        conn = duckdb.connect(str(duckdb_path))
        try:
            conn.execute("BEGIN TRANSACTION")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS materialized_stream_daily (
                    day_utc VARCHAR,
                    stream VARCHAR,
                    record_count BIGINT,
                    distinct_sources BIGINT,
                    min_schema_version BIGINT,
                    max_schema_version BIGINT,
                    last_ingested_at VARCHAR,
                    refreshed_utc VARCHAR
                )
                """
            )
            conn.execute("DELETE FROM materialized_stream_daily")
            if stream_rows:
                conn.executemany(
                    "INSERT INTO materialized_stream_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    stream_rows,
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS materialized_symbol_daily (
                    day_utc VARCHAR,
                    symbol VARCHAR,
                    record_count BIGINT,
                    buy_count BIGINT,
                    sell_count BIGINT,
                    hold_count BIGINT,
                    last_ingested_at VARCHAR,
                    refreshed_utc VARCHAR
                )
                """
            )
            conn.execute("DELETE FROM materialized_symbol_daily")
            if symbol_rows:
                conn.executemany(
                    "INSERT INTO materialized_symbol_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    symbol_rows,
                )
            conn.execute("COMMIT")
        finally:
            # Closing an uncommitted connection rolls back both table replacements.
            conn.close()
    except Exception as exc:
        return {
            "duckdb_available": True,
            "mirror_ready": False,
            "error": f"duckdb_sync_failed:{type(exc).__name__}:{exc}",
            "stream_summary_rows": 0,
            "symbol_summary_rows": 0,
        }
    return {
        "duckdb_available": True,
        "mirror_ready": True,
        "error": "",
        "stream_summary_rows": len(stream_rows),
        "symbol_summary_rows": len(symbol_rows),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    source_db_path: Path = DEFAULT_SOURCE_DB,
    duckdb_path: Path = DEFAULT_DUCKDB_PATH,
    lookback_days: int = 7,
) -> dict[str, Any]:
    try:
        _local_route(project_root, project_root)
        source_db_path = _data_route(source_db_path)
        duckdb_path = _data_route(duckdb_path)
        ops_path = _data_route(ops_data_plane.resolve_db_path(project_root))
        for path in (source_db_path, ops_path, duckdb_path):
            for suffix in ("-wal", "-shm", "-journal", ".wal"):
                _data_route(Path(str(path) + suffix))
        _check_headroom(project_root)
        for path in (duckdb_path.parent, ops_path.parent):
            if path.exists():
                _check_headroom(path)
        if not source_db_path.is_file():
            return _deferred("analytics_source_missing")
        lock_path = _local_route(
            project_root / "governance/locks/sql_analytics_mirror.lock", project_root
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(
            os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "w"
        ) as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {**_deferred("analytics_mirror_already_running"), "busy": True}
            return _build_payload(
                project_root,
                source_db_path=source_db_path,
                duckdb_path=duckdb_path,
                lookback_days=lookback_days,
            )
    except (OSError, ValueError, RuntimeError) as exc:
        return _deferred(str(exc))


def _build_payload(
    project_root: Path,
    *,
    source_db_path: Path,
    duckdb_path: Path,
    lookback_days: int,
) -> dict[str, Any]:
    ops_db_path = ops_data_plane.resolve_db_path(project_root)
    summary_payload: dict[str, Any] = {
        "refreshed_utc": "",
        "day_floor": "",
        "source_record_count": 0,
        "stream_summary_rows": 0,
        "symbol_summary_rows": 0,
    }
    duration_ms = 0.0
    summary_refresh_ok = False
    summary_refresh_error = ""

    def admission_check():
        _check_headroom(project_root)
        for path in (ops_db_path.parent, duckdb_path.parent):
            _data_route(path)
            if path.exists():
                _check_headroom(path)

    try:
        with closing(
            ops_data_plane.connect(project_root, timeout_seconds=1.0, quick_check=False)
        ) as ops_conn:
            ops_conn.execute("PRAGMA cache_size=-8192")
            ops_conn.execute("PRAGMA mmap_size=0")
            summary_payload, duration_ms = ops_data_plane.time_query(
                lambda: ops_data_plane.emit_materialized_summaries(
                    ops_conn,
                    source_db_path=source_db_path,
                    lookback_days=max(int(lookback_days), 1),
                    admission_check=admission_check,
                )
            )
            ops_data_plane.record_query_access(
                ops_conn,
                query_family="materialized_summary_refresh",
                shard_name="primary_sqlite",
                consumer="sql_analytics_mirror",
                query_text="materialized_stream_daily + materialized_symbol_daily refresh",
                rows_scanned=max(
                    _safe_int(summary_payload.get("source_record_count"), 0), 1
                ),
                rows_returned=_safe_int(summary_payload.get("stream_summary_rows"), 0)
                + _safe_int(summary_payload.get("symbol_summary_rows"), 0),
                duration_ms=duration_ms,
                source_name=str(source_db_path),
            )
            ops_conn.commit()
        summary_refresh_ok = True
    except Exception as exc:
        summary_refresh_error = f"{type(exc).__name__}:{exc}"
        summary_payload["error"] = summary_refresh_error

    if summary_refresh_ok:
        try:
            admission_check()
        except (OSError, ValueError, RuntimeError) as exc:
            return _deferred(str(exc))
    duckdb_payload = (
        _write_duckdb_mirror(ops_db_path=ops_db_path, duckdb_path=duckdb_path)
        if summary_refresh_ok
        else {
            "duckdb_available": False,
            "mirror_ready": False,
            "error": "summary_refresh_failed",
            "stream_summary_rows": 0,
            "symbol_summary_rows": 0,
        }
    )
    duckdb_optional = not bool(duckdb_payload.get("duckdb_available", False))
    overall_ok = bool(
        summary_refresh_ok
        and (duckdb_optional or duckdb_payload.get("mirror_ready", False))
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": overall_ok,
        "overall_status": "ready" if overall_ok else "deferred",
        "resource_contract": {
            "source_query_seconds": 20,
            "source_rows": 250000,
            "projected_bytes": 32 * 1024**2,
            "cli_process_seconds": 35,
            "minimum_free_gib": 65,
            "raw_payload_sorting": False,
        },
        "source_db_path": str(source_db_path),
        "source_db_present": bool(source_db_path.exists()),
        "ops_db_path": str(ops_db_path),
        "duckdb_path": str(duckdb_path),
        "lookback_days": max(int(lookback_days), 1),
        "summary_refresh_ok": bool(summary_refresh_ok),
        "summary_refresh_error": summary_refresh_error,
        "materialized_summaries": summary_payload,
        "duckdb_mirror": duckdb_payload,
        "top_actions": [
            "point dashboards and training prep at materialized_stream_daily and materialized_symbol_daily before scanning raw jsonl payloads",
            "treat shard_heat_state and query_access_events as the signal source for future shard-promotion decisions",
        ],
        "analytics_read_contract": {
            "primary_hot_path": "sqlite_and_apsw",
            "analytical_read_offload": "duckdb_and_adbc",
            "transform_layer": "polars_and_arrow",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh SQLite materialized summaries and optional DuckDB analytics mirror."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--source-db", default=str(DEFAULT_SOURCE_DB))
    parser.add_argument("--duckdb-path", default=str(DEFAULT_DUCKDB_PATH))
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = Path(args.project_root).absolute()
    if args.worker:
        payload = build_payload(
            root,
            source_db_path=Path(args.source_db).expanduser(),
            duckdb_path=Path(args.duckdb_path).expanduser(),
            lookback_days=max(int(args.lookback_days), 1),
        )
    else:
        result = run_bounded_process_group(
            [
                sys.executable,
                str(Path(__file__).absolute()),
                "--worker",
                "--json",
                "--project-root",
                str(root),
                "--source-db",
                args.source_db,
                "--duckdb-path",
                args.duckdb_path,
                "--lookback-days",
                str(args.lookback_days),
            ],
            cwd=PROJECT_ROOT,
            timeout_seconds=35,
        )
        try:
            payload = json.loads(result["stdout"])
            if result["timed_out"] or not isinstance(payload, dict):
                raise ValueError("incomplete_worker")
        except (ValueError, TypeError):
            payload = _deferred(
                "analytics_worker_timeout"
                if result["timed_out"]
                else "analytics_worker_failed"
            )
            payload["timeout_cleanup"] = result.get("timeout_cleanup", {})
        if not payload.get("busy"):
            out_path = _local_route(Path(args.out_file).expanduser(), root)
            write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "sql_analytics_mirror "
            f"stream_rows={int((payload.get('materialized_summaries') or {}).get('stream_summary_rows', 0) or 0)} "
            f"symbol_rows={int((payload.get('materialized_summaries') or {}).get('symbol_summary_rows', 0) or 0)} "
            f"duckdb_ready={str(bool((payload.get('duckdb_mirror') or {}).get('mirror_ready', False))).lower()}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
