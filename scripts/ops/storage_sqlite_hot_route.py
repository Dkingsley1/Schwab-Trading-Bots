#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from core.storage_mounts import resolve_external_storage
    from scripts.ops import writer_cycle_coordinator as writer_state
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from core.storage_mounts import resolve_external_storage
    from scripts.ops import writer_cycle_coordinator as writer_state


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_sqlite_hot_route_latest.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _utc_now()).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _disk_free_bytes(path: Path) -> int | None:
    try:
        return int(os.statvfs(path).f_bavail * os.statvfs(path).f_frsize)
    except Exception:
        return None


def _size_bytes(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _sidecars(path: Path) -> list[Path]:
    return [Path(f"{path}{suffix}") for suffix in ("-wal", "-shm")]


def _table_exists(conn: sqlite3.Connection, table: str, *, schema: str = "main") -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str, *, schema: str = "main") -> list[tuple[str, str]]:
    return [
        (str(row[1]), str(row[2] or "TEXT"))
        for row in conn.execute(f"PRAGMA {schema}.table_info({table})").fetchall()
        if row and len(row) >= 3 and row[1]
    ]


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _arrow_type(sql_type: str):
    import pyarrow as pa

    raw = str(sql_type or "").upper()
    if "INT" in raw:
        return pa.int64()
    if any(token in raw for token in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return pa.float64()
    if "BLOB" in raw:
        return pa.binary()
    return pa.string()


def _connect(path: Path, *, readonly: bool = False, timeout_seconds: float = 60.0) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=max(float(timeout_seconds), 1.0))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=max(float(timeout_seconds), 1.0))
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={int(max(float(timeout_seconds), 1.0) * 1000)}")
    return conn


def _quick_check(path: Path, *, timeout_seconds: float) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "result": "missing"}
    try:
        with _connect(path, readonly=False, timeout_seconds=timeout_seconds) as conn:
            row = conn.execute("PRAGMA quick_check").fetchone()
            result = str(row[0] if row else "")
    except Exception as exc:
        return {"ok": False, "result": str(exc), "error_type": type(exc).__name__}
    return {"ok": result.lower() == "ok", "result": result}


def _source_counts(conn: sqlite3.Connection, cutoff: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for table in ("jsonl_records", "json_file_records"):
        if not _table_exists(conn, table):
            continue
        cols = [name for name, _ in _table_columns(conn, table)]
        total = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()[0] or 0)
        bucket: dict[str, Any] = {"total_rows": total}
        if "ingested_at" in cols:
            hot = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {_quote_ident(table)} WHERE ingested_at >= ?",
                    (cutoff,),
                ).fetchone()[0]
                or 0
            )
            bucket["hot_rows"] = hot
            bucket["cold_rows"] = max(total - hot, 0)
            row = conn.execute(
                f"SELECT MIN(ingested_at), MAX(ingested_at) FROM {_quote_ident(table)}"
            ).fetchone()
            bucket["min_ingested_at"] = str(row[0] or "") if row else ""
            bucket["max_ingested_at"] = str(row[1] or "") if row else ""
        out[table] = bucket
    return out


def _export_cold_table_to_parquet(
    src: sqlite3.Connection,
    *,
    table: str,
    cutoff: str,
    out_path: Path,
    batch_size: int,
    compression: str,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not _table_exists(src, table):
        return {"table": table, "rows_exported": 0, "output_path": str(out_path), "skipped_reason": "table_missing"}

    col_specs = _table_columns(src, table)
    col_names = [name for name, _ in col_specs]
    if "ingested_at" not in col_names:
        return {"table": table, "rows_exported": 0, "output_path": str(out_path), "skipped_reason": "no_ingested_at"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    quoted_cols = ", ".join(_quote_ident(name) for name in col_names)
    order_cols = ["ingested_at ASC"]
    if "id" in col_names:
        order_cols.append("id ASC")
    cursor = src.execute(
        f"SELECT {quoted_cols} FROM {_quote_ident(table)} WHERE ingested_at < ? ORDER BY {', '.join(order_cols)}",
        (cutoff,),
    )
    col_types = {name: _arrow_type(sql_type) for name, sql_type in col_specs}
    writer = None
    rows_exported = 0
    min_ingested_at = ""
    max_ingested_at = ""
    try:
        while True:
            rows = cursor.fetchmany(max(int(batch_size), 1000))
            if not rows:
                break
            arrays = {}
            for idx, name in enumerate(col_names):
                values = [row[idx] for row in rows]
                arrays[name] = pa.array(values, type=col_types[name])
            batch = pa.Table.from_pydict(arrays)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), batch.schema, compression=str(compression or "zstd"))
            writer.write_table(batch)
            rows_exported += len(rows)
            ts_idx = col_names.index("ingested_at")
            ingested = [str(row[ts_idx]) for row in rows if row[ts_idx]]
            if ingested:
                batch_min = min(ingested)
                batch_max = max(ingested)
                min_ingested_at = batch_min if not min_ingested_at or batch_min < min_ingested_at else min_ingested_at
                max_ingested_at = batch_max if not max_ingested_at or batch_max > max_ingested_at else max_ingested_at
    finally:
        if writer is not None:
            writer.close()

    if rows_exported == 0 and out_path.exists():
        try:
            out_path.unlink()
        except Exception:
            pass
    return {
        "table": table,
        "rows_exported": int(rows_exported),
        "output_path": str(out_path) if rows_exported > 0 else "",
        "size_bytes": _size_bytes(out_path) if rows_exported > 0 else 0,
        "min_ingested_at": min_ingested_at,
        "max_ingested_at": max_ingested_at,
    }


def _schema_rows(src: sqlite3.Connection, kind: str, *, schema: str = "main") -> list[sqlite3.Row]:
    if schema not in {"main", "src"}:
        raise ValueError(f"unsupported sqlite schema: {schema}")
    return list(
        src.execute(
            f"""
            SELECT type, name, tbl_name, sql
            FROM {schema}.sqlite_master
            WHERE type=?
              AND sql IS NOT NULL
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """,
            (kind,),
        ).fetchall()
    )


def _build_hot_db(
    *,
    source_db: Path,
    dest_tmp: Path,
    cutoff: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    if dest_tmp.exists():
        dest_tmp.unlink()
    for sidecar in _sidecars(dest_tmp):
        if sidecar.exists():
            sidecar.unlink()

    dest = _connect(dest_tmp, readonly=False, timeout_seconds=timeout_seconds)
    try:
        dest.execute("PRAGMA journal_mode=OFF")
        dest.execute("PRAGMA synchronous=OFF")
        dest.execute("PRAGMA temp_store=FILE")
        dest.execute("ATTACH DATABASE ? AS src", (str(source_db),))

        tables = []
        for row in _schema_rows(dest, "table", schema="src"):
            sql = str(row["sql"] or "").strip()
            name = str(row["name"] or "")
            if not sql or not name:
                continue
            dest.execute(sql)
            tables.append(name)

        copied: dict[str, int] = {}
        for table in tables:
            cols = [name for name, _ in _table_columns(dest, table, schema="src")]
            if not cols:
                continue
            quoted_cols = ", ".join(_quote_ident(name) for name in cols)
            has_ingested = "ingested_at" in cols
            if table in {"jsonl_records", "json_file_records"} and has_ingested:
                sql = (
                    f"INSERT OR IGNORE INTO main.{_quote_ident(table)} ({quoted_cols}) "
                    f"SELECT {quoted_cols} FROM src.{_quote_ident(table)} WHERE ingested_at >= ?"
                )
                dest.execute(sql, (cutoff,))
            else:
                sql = (
                    f"INSERT OR IGNORE INTO main.{_quote_ident(table)} ({quoted_cols}) "
                    f"SELECT {quoted_cols} FROM src.{_quote_ident(table)}"
                )
                dest.execute(sql)
            copied[table] = int(dest.execute(f"SELECT COUNT(*) FROM main.{_quote_ident(table)}").fetchone()[0] or 0)
            dest.commit()

        indexes = []
        for row in _schema_rows(dest, "index", schema="src"):
            sql = str(row["sql"] or "").strip()
            name = str(row["name"] or "")
            if not sql or not name:
                continue
            try:
                dest.execute(sql)
                indexes.append(name)
            except sqlite3.OperationalError as exc:
                if "already exists" not in str(exc).lower():
                    raise
        dest.commit()
        try:
            dest.execute("ANALYZE")
            dest.execute("PRAGMA optimize")
            dest.commit()
        except Exception:
            pass
        dest.execute("DETACH DATABASE src")
    finally:
        dest.close()

    quick = _quick_check(dest_tmp, timeout_seconds=timeout_seconds)
    return {
        "path": str(dest_tmp),
        "size_bytes": _size_bytes(dest_tmp),
        "copied_rows": copied,
        "indexes_created": indexes,
        "quick_check": quick,
    }


def _safe_replace_symlink(link: Path, target: Path) -> str:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"repo path exists and is not a symlink: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    return str(target)


def _switch_repo_links(project_root: Path, external_db: Path, relative_path: str) -> dict[str, Any]:
    repo_db = project_root / relative_path
    switched = {"primary": _safe_replace_symlink(repo_db, external_db), "sidecars": {}}
    for suffix in ("-wal", "-shm"):
        repo_sidecar = Path(f"{repo_db}{suffix}")
        external_sidecar = Path(f"{external_db}{suffix}")
        switched["sidecars"][suffix] = _safe_replace_symlink(repo_sidecar, external_sidecar)
    return switched


def _unlink_artifact(path: Path) -> dict[str, Any]:
    deleted = []
    deleted_bytes = 0
    errors = []
    for candidate in [path, *_sidecars(path)]:
        if not candidate.exists() and not candidate.is_symlink():
            continue
        try:
            deleted_bytes += _size_bytes(candidate)
            candidate.unlink()
            deleted.append(str(candidate))
        except Exception as exc:
            errors.append(f"{candidate}:{type(exc).__name__}:{exc}")
    return {"deleted_paths": deleted, "deleted_bytes": int(deleted_bytes), "errors": errors}


def build_payload(
    project_root: Path,
    *,
    relative_path: str,
    hot_hours: float,
    apply: bool,
    prune_local: bool,
    min_external_free_after_gb: float,
    cold_export_root: Path,
    batch_size: int,
    compression: str,
    require_writer_idle: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    relative_path = str(relative_path or "data/jsonl_link.sqlite3").replace("\\", "/").lstrip("./")
    source_db = project_root / "local_fallback_storage" / relative_path
    repo_db = project_root / relative_path
    external_root = resolve_external_storage().external_root
    external_db = external_root / relative_path
    run_id = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    temp_db = external_db.with_name(f"{external_db.name}.hot_route_{run_id}.tmp")
    cutoff_dt = _utc_now() - timedelta(hours=max(float(hot_hours), 1.0))
    cutoff = cutoff_dt.isoformat()
    writer_snapshot = writer_state.writer_state_snapshot(project_root)
    writer_idle = not bool(writer_snapshot.get("active", False)) and not bool(writer_snapshot.get("running", False))

    free_before = _disk_free_bytes(external_root)
    source_bytes = _size_bytes(source_db)
    min_external_free_after_bytes = int(max(float(min_external_free_after_gb), 0.0) * (1024**3))
    projected_free_after_full_tmp = None if free_before is None else int(free_before - source_bytes)

    payload: dict[str, Any] = {
        "timestamp_utc": _iso(),
        "schema_version": 1,
        "ok": False,
        "overall_status": "dry_run",
        "apply": bool(apply),
        "prune_local": bool(prune_local),
        "project_root": str(project_root),
        "relative_path": relative_path,
        "repo_db": str(repo_db),
        "source_db": str(source_db),
        "external_db": str(external_db),
        "temp_db": str(temp_db),
        "cold_export_root": str(cold_export_root),
        "hot_hours": float(hot_hours),
        "cutoff_utc": cutoff,
        "batch_size": int(batch_size),
        "compression": str(compression),
        "source_size_bytes": int(source_bytes),
        "external_free_bytes_before": free_before,
        "projected_external_free_after_full_tmp_bytes": projected_free_after_full_tmp,
        "min_external_free_after_bytes": min_external_free_after_bytes,
        "writer_idle_required": bool(require_writer_idle),
        "writer_idle": bool(writer_idle),
        "writer_state": writer_snapshot,
        "source_counts": {},
        "cold_exports": [],
        "hot_db": {},
        "switch": {},
        "local_prune": {},
        "blockers": [],
    }

    if not source_db.exists():
        payload["blockers"].append("source_db_missing")
    if not external_root.exists() or not os.access(external_root, os.W_OK):
        payload["blockers"].append("external_root_not_writable")
    if require_writer_idle and not writer_idle:
        payload["blockers"].append("writer_not_idle")
    if free_before is None:
        payload["blockers"].append("external_free_unknown")
    elif projected_free_after_full_tmp is not None and projected_free_after_full_tmp < min_external_free_after_bytes:
        payload["blockers"].append("external_free_after_full_tmp_below_guard")

    if source_db.exists():
        try:
            with _connect(source_db, readonly=True, timeout_seconds=timeout_seconds) as src:
                payload["source_counts"] = _source_counts(src, cutoff)
        except Exception as exc:
            payload["blockers"].append(f"source_count_error:{type(exc).__name__}:{exc}")

    if payload["blockers"]:
        payload["overall_status"] = "blocked"
        return payload
    if not apply:
        payload["ok"] = True
        payload["overall_status"] = "dry_run"
        return payload

    cold_export_root.mkdir(parents=True, exist_ok=True)
    external_db.parent.mkdir(parents=True, exist_ok=True)

    with _connect(source_db, readonly=True, timeout_seconds=timeout_seconds) as src:
        for table in ("jsonl_records", "json_file_records"):
            export_path = cold_export_root / f"{table}_cold_before_{cutoff_dt.strftime('%Y%m%dT%H%M%SZ')}_{run_id}.parquet"
            export = _export_cold_table_to_parquet(
                src,
                table=table,
                cutoff=cutoff,
                out_path=export_path,
                batch_size=max(int(batch_size), 1000),
                compression=compression,
            )
            payload["cold_exports"].append(export)

    hot_db = _build_hot_db(
        source_db=source_db,
        dest_tmp=temp_db,
        cutoff=cutoff,
        timeout_seconds=timeout_seconds,
    )
    payload["hot_db"] = hot_db
    if not bool((hot_db.get("quick_check") or {}).get("ok", False)):
        payload["overall_status"] = "hot_db_quick_check_failed"
        return payload

    source_counts = payload.get("source_counts") if isinstance(payload.get("source_counts"), dict) else {}
    copied_rows = hot_db.get("copied_rows") if isinstance(hot_db.get("copied_rows"), dict) else {}
    exported_by_table = {
        str(row.get("table")): int(row.get("rows_exported", 0) or 0)
        for row in payload["cold_exports"]
        if isinstance(row, dict)
    }
    coverage_errors = []
    for table in ("jsonl_records", "json_file_records"):
        bucket = source_counts.get(table) if isinstance(source_counts.get(table), dict) else {}
        if not bucket:
            continue
        expected = int(bucket.get("total_rows", 0) or 0)
        actual = int(copied_rows.get(table, 0) or 0) + int(exported_by_table.get(table, 0) or 0)
        if actual != expected:
            coverage_errors.append(f"{table}:expected={expected}:covered={actual}")
    payload["coverage_check"] = {"ok": not coverage_errors, "errors": coverage_errors}
    if coverage_errors:
        payload["overall_status"] = "coverage_check_failed"
        return payload

    if external_db.exists():
        backup_existing = external_db.with_name(f"{external_db.name}.pre_hot_route_{run_id}.bak")
        external_db.replace(backup_existing)
        payload["previous_external_backup"] = str(backup_existing)
    temp_db.replace(external_db)
    payload["external_size_bytes_after"] = _size_bytes(external_db)
    payload["switch"] = _switch_repo_links(project_root, external_db, relative_path)
    route_quick = _quick_check(repo_db, timeout_seconds=timeout_seconds)
    payload["route_quick_check"] = route_quick
    if not bool(route_quick.get("ok", False)):
        payload["overall_status"] = "route_quick_check_failed"
        return payload

    if prune_local:
        payload["local_prune"] = _unlink_artifact(source_db)
        if payload["local_prune"].get("errors"):
            payload["overall_status"] = "switched_prune_errors"
            return payload

    payload["external_free_bytes_after"] = _disk_free_bytes(external_root)
    payload["internal_source_exists_after"] = bool(source_db.exists())
    payload["ok"] = True
    payload["overall_status"] = "switched_pruned" if prune_local else "switched"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact external hot SQLite route and optional cold parquet archive for jsonl_link.sqlite3.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--relative-path", default="data/jsonl_link.sqlite3")
    parser.add_argument("--hot-hours", type=float, default=float(os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_HOURS", "18")))
    parser.add_argument("--cold-export-root", default=os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_COLD_ROOT", ""))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_BATCH_SIZE", "25000")))
    parser.add_argument("--compression", default=os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_COMPRESSION", "zstd"))
    parser.add_argument("--min-external-free-after-gb", type=float, default=float(os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_MIN_FREE_AFTER_GB", "40")))
    parser.add_argument("--sqlite-timeout-seconds", type=float, default=float(os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_TIMEOUT_SECONDS", "300")))
    parser.add_argument("--no-require-writer-idle", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--prune-local", action="store_true")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    external_root = resolve_external_storage().external_root
    cold_root = (
        Path(args.cold_export_root).expanduser()
        if str(args.cold_export_root or "").strip()
        else external_root / "data" / "cold_archives" / "jsonl_link_hot_route"
    )
    payload = build_payload(
        project_root,
        relative_path=str(args.relative_path),
        hot_hours=float(args.hot_hours),
        apply=bool(args.apply),
        prune_local=bool(args.prune_local),
        min_external_free_after_gb=float(args.min_external_free_after_gb),
        cold_export_root=cold_root,
        batch_size=max(int(args.batch_size), 1000),
        compression=str(args.compression or "zstd"),
        require_writer_idle=not bool(args.no_require_writer_idle),
        timeout_seconds=max(float(args.sqlite_timeout_seconds), 1.0),
    )
    out_file = Path(args.out_file).expanduser()
    _write_json(out_file, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_sqlite_hot_route "
            f"status={payload.get('overall_status')} "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"hot_hours={payload.get('hot_hours')}"
        )
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
