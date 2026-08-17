#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from core.runtime_maintenance import (
        engage_maintenance_hold,
        maintenance_hold_snapshot,
        release_maintenance_hold,
    )
    from core.storage_mounts import resolve_external_storage
    from scripts.ops import writer_cycle_coordinator as writer_state
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from core.runtime_maintenance import (
        engage_maintenance_hold,
        maintenance_hold_snapshot,
        release_maintenance_hold,
    )
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
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "resolved_path": str(path.resolve()),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        return False
    return True


def _discard_partial_datasets(transaction: dict[str, Any], cold_export_root: Path) -> list[str]:
    deleted: list[str] = []
    for raw in transaction.get("partial_datasets", []) if isinstance(transaction.get("partial_datasets"), list) else []:
        candidate = Path(str(raw or "")).expanduser()
        if (
            candidate.name.startswith(".")
            and candidate.name.endswith(".partial_dataset")
            and _path_is_within(candidate, cold_export_root)
            and candidate.is_dir()
        ):
            shutil.rmtree(candidate, ignore_errors=True)
            if not candidate.exists():
                deleted.append(str(candidate))
    return deleted


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
        with closing(_connect(path, readonly=False, timeout_seconds=timeout_seconds)) as conn:
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


def _estimated_hot_db_bytes(conn: sqlite3.Connection, cutoff: str, source_bytes: int) -> int:
    payload_bytes = 0
    hot_rows = 0
    total_rows = 0
    for table in ("jsonl_records", "json_file_records"):
        if not _table_exists(conn, table):
            continue
        columns = [name for name, _ in _table_columns(conn, table)]
        if "ingested_at" not in columns:
            continue
        total_rows += int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()[0] or 0)
        hot_rows += int(
            conn.execute(
                f"SELECT COUNT(*) FROM {_quote_ident(table)} WHERE ingested_at >= ?",
                (cutoff,),
            ).fetchone()[0]
            or 0
        )
        payload_columns = [name for name in columns if name in {"payload_json", "payload_sha1", "source_file", "source_rel"}]
        if payload_columns:
            expression = " + ".join(f"COALESCE(LENGTH({_quote_ident(name)}), 0)" for name in payload_columns)
            payload_bytes += int(
                conn.execute(
                    f"SELECT COALESCE(SUM({expression}), 0) FROM {_quote_ident(table)} WHERE ingested_at >= ?",
                    (cutoff,),
                ).fetchone()[0]
                or 0
            )
    ratio_estimate = int(max(int(source_bytes), 0) * (float(hot_rows) / float(max(total_rows, 1))) * 1.5)
    payload_estimate = int(max(payload_bytes, 0) * 2.25)
    return max(ratio_estimate, payload_estimate, 64 * 1024**2)


def _export_cold_table_to_parquet(
    src: sqlite3.Connection,
    *,
    table: str,
    cutoff: str,
    out_path: Path,
    batch_size: int,
    compression: str,
    free_guard_root: Path | None = None,
    min_free_after_bytes: int = 0,
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
    max_batch_rows = max(int(os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_MAX_EXPORT_BATCH_ROWS", "2000")), 100)
    effective_batch_size = min(max(int(batch_size), 100), max_batch_rows)
    memory_pool = pa.system_memory_pool()
    writer = None
    rows_exported = 0
    batches_written = 0
    min_ingested_at = ""
    max_ingested_at = ""
    failed = False
    try:
        while True:
            rows = cursor.fetchmany(effective_batch_size)
            if not rows:
                break
            arrays = {}
            for idx, name in enumerate(col_names):
                values = [row[idx] for row in rows]
                arrays[name] = pa.array(values, type=col_types[name], memory_pool=memory_pool)
            batch = pa.Table.from_pydict(arrays)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), batch.schema, compression=str(compression or "zstd"))
            writer.write_table(batch)
            rows_exported += len(rows)
            batches_written += 1
            if free_guard_root is not None:
                free_bytes = _disk_free_bytes(free_guard_root)
                if free_bytes is None or free_bytes < max(int(min_free_after_bytes), 0):
                    raise RuntimeError("cold_export_free_space_guard")
            ts_idx = col_names.index("ingested_at")
            ingested = [str(row[ts_idx]) for row in rows if row[ts_idx]]
            if ingested:
                batch_min = min(ingested)
                batch_max = max(ingested)
                min_ingested_at = batch_min if not min_ingested_at or batch_min < min_ingested_at else min_ingested_at
                max_ingested_at = batch_max if not max_ingested_at or batch_max > max_ingested_at else max_ingested_at
            del batch, arrays, rows, values
            if batches_written % 8 == 0:
                gc.collect()
                memory_pool.release_unused()
    except Exception:
        failed = True
        raise
    finally:
        if writer is not None:
            writer.close()
        if failed and out_path.exists():
            out_path.unlink()

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
        "effective_batch_size": int(effective_batch_size),
        "batches_written": int(batches_written),
        "min_ingested_at": min_ingested_at,
        "max_ingested_at": max_ingested_at,
    }


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _export_cold_table_to_parquet_duckdb(
    source_db: Path,
    *,
    table: str,
    cutoff: str,
    out_path: Path,
    compression: str,
    free_guard_root: Path,
    min_free_after_bytes: int,
) -> dict[str, Any]:
    import duckdb
    import pyarrow.parquet as pq

    with closing(_connect(source_db, readonly=True)) as sqlite_conn:
        if not _table_exists(sqlite_conn, table):
            return {"table": table, "rows_exported": 0, "output_path": "", "skipped_reason": "table_missing"}
        columns = [name for name, _ in _table_columns(sqlite_conn, table)]
        if "ingested_at" not in columns:
            return {"table": table, "rows_exported": 0, "output_path": "", "skipped_reason": "no_ingested_at"}
        if "id" in columns:
            bounds = sqlite_conn.execute(
                f"SELECT MIN(id), MAX(id) FROM {_quote_ident(table)} WHERE ingested_at < ?",
                (cutoff,),
            ).fetchone()
        else:
            bounds = None
    if "id" not in columns:
        raise RuntimeError(f"duckdb_partition_key_missing:{table}:id")
    min_id = int(bounds[0]) if bounds and bounds[0] is not None else None
    max_id = int(bounds[1]) if bounds and bounds[1] is not None else None
    if min_id is None or max_id is None:
        return {
            "table": table,
            "rows_exported": 0,
            "output_path": "",
            "size_bytes": 0,
            "engine": "duckdb_partitioned",
        }

    free_before = _disk_free_bytes(free_guard_root)
    if free_before is None or free_before < max(int(min_free_after_bytes), 0):
        raise RuntimeError("cold_export_free_space_guard")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_root = out_path.with_name(f".{out_path.name}.partial_dataset")
    spill_root = out_path.parent / f".duckdb_spill_{out_path.stem}"
    if out_path.exists() or partial_root.exists() or spill_root.exists():
        raise RuntimeError("duckdb_export_target_already_exists")
    partial_root.mkdir(parents=True)
    spill_root.mkdir(parents=True, exist_ok=True)
    memory_limit = str(os.getenv("BOT_LOGS_SQLITE_DUCKDB_EXPORT_MEMORY_LIMIT", "1024MB") or "1024MB")
    threads = max(int(os.getenv("BOT_LOGS_SQLITE_DUCKDB_EXPORT_THREADS", "1")), 1)
    row_group_size = max(int(os.getenv("BOT_LOGS_SQLITE_DUCKDB_EXPORT_ROW_GROUP_SIZE", "2048")), 2048)
    row_group_size = max((row_group_size // 2048) * 2048, 2048)
    id_span = max(int(os.getenv("BOT_LOGS_SQLITE_DUCKDB_EXPORT_ID_SPAN", "25000")), 2048)
    min_id_span = max(int(os.getenv("BOT_LOGS_SQLITE_DUCKDB_EXPORT_MIN_ID_SPAN", "2048")), 1)
    codec = str(compression or "zstd").strip().upper()
    if codec not in {"ZSTD", "SNAPPY", "GZIP", "LZ4", "UNCOMPRESSED"}:
        raise ValueError(f"unsupported parquet compression: {compression}")
    rows_exported = 0
    part_paths: list[Path] = []
    adaptive_splits = 0

    def export_range(start_id: int, end_id: int) -> None:
        nonlocal rows_exported, adaptive_splits
        part_path = partial_root / f"part_{start_id:012d}_{end_id - 1:012d}.parquet"
        conn = duckdb.connect(":memory:")
        conn_closed = False
        try:
            conn.execute(f"SET memory_limit={_sql_literal(memory_limit)}")
            conn.execute(f"SET threads={threads}")
            conn.execute("SET preserve_insertion_order=false")
            conn.execute(f"SET temp_directory={_sql_literal(spill_root)}")
            conn.execute(f"ATTACH {_sql_literal(source_db)} AS sqlite_source (TYPE sqlite, READ_ONLY)")
            conn.execute(
                f"COPY (SELECT * FROM sqlite_source.{_quote_ident(table)} "
                f"WHERE ingested_at < {_sql_literal(cutoff)} AND id >= {int(start_id)} AND id < {int(end_id)}) "
                f"TO {_sql_literal(part_path)} (FORMAT PARQUET, COMPRESSION {codec}, ROW_GROUP_SIZE {row_group_size})"
            )
        except Exception as exc:
            part_path.unlink(missing_ok=True)
            if "out of memory" in str(exc).lower() and end_id - start_id > min_id_span:
                adaptive_splits += 1
                conn.close()
                conn_closed = True
                gc.collect()
                midpoint = start_id + max((end_id - start_id) // 2, 1)
                export_range(start_id, midpoint)
                export_range(midpoint, end_id)
                return
            raise
        finally:
            if not conn_closed:
                conn.close()

        metadata = pq.ParquetFile(part_path).metadata
        part_rows = int(metadata.num_rows if metadata is not None else 0)
        if part_rows <= 0:
            part_path.unlink(missing_ok=True)
            return
        with part_path.open("rb") as handle:
            os.fsync(handle.fileno())
        rows_exported += part_rows
        part_paths.append(part_path)
        free_now = _disk_free_bytes(free_guard_root)
        if free_now is None or free_now < max(int(min_free_after_bytes), 0):
            raise RuntimeError("cold_export_free_space_guard")

    try:
        range_start = min_id
        while range_start <= max_id:
            range_end = min(range_start + id_span, max_id + 1)
            export_range(range_start, range_end)
            range_start = range_end
    except Exception:
        shutil.rmtree(partial_root, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(spill_root, ignore_errors=True)

    free_after = _disk_free_bytes(free_guard_root)
    if free_after is None or free_after < max(int(min_free_after_bytes), 0):
        shutil.rmtree(partial_root, ignore_errors=True)
        raise RuntimeError("cold_export_free_space_guard")
    if rows_exported <= 0:
        shutil.rmtree(partial_root, ignore_errors=True)
        return {
            "table": table,
            "rows_exported": 0,
            "output_path": "",
            "size_bytes": 0,
            "engine": "duckdb_partitioned",
            "memory_limit": memory_limit,
            "threads": threads,
            "row_group_size": row_group_size,
        }
    total_size = sum(_size_bytes(path) for path in part_paths)
    partial_root.replace(out_path)
    return {
        "table": table,
        "rows_exported": rows_exported,
        "output_path": str(out_path),
        "size_bytes": int(total_size),
        "engine": "duckdb_partitioned",
        "memory_limit": memory_limit,
        "threads": threads,
        "row_group_size": row_group_size,
        "id_span": id_span,
        "min_id_span": min_id_span,
        "min_id": min_id,
        "max_id": max_id,
        "part_count": len(part_paths),
        "adaptive_split_count": adaptive_splits,
        "part_files_sample": [path.name for path in part_paths[:20]],
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair_final_manifest_paths(manifest_path: Path, *, apply: bool) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    dataset_root = path.parent
    payload = _load_json(path)
    parts = payload.get("parts") if isinstance(payload.get("parts"), list) else []
    result: dict[str, Any] = {
        "ok": False,
        "apply": bool(apply),
        "manifest_path": str(path),
        "dataset_root": str(dataset_root),
        "part_count": len(parts),
        "validated_part_count": 0,
        "changed_path_count": 0,
        "errors": [],
    }
    if not path.is_file() or not payload:
        result["errors"].append("manifest_missing_or_invalid")
        return result
    if not parts:
        result["errors"].append("manifest_parts_missing")
        return result

    repaired_parts: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(parts):
        if not isinstance(raw, dict):
            result["errors"].append(f"part_{index}:invalid_receipt")
            continue
        receipt = dict(raw)
        rows = int(receipt.get("rows_exported", 0) or 0)
        if rows <= 0:
            repaired_parts.append(receipt)
            continue
        original = Path(str(receipt.get("output_path") or ""))
        if not original.name:
            result["errors"].append(f"part_{index}:output_path_missing")
            continue
        canonical = dataset_root / original.name
        canonical_text = str(canonical)
        if canonical_text in seen_paths:
            result["errors"].append(f"part_{index}:duplicate_output_path:{canonical.name}")
            continue
        seen_paths.add(canonical_text)
        if not canonical.is_file():
            result["errors"].append(f"part_{index}:canonical_part_missing:{canonical.name}")
            continue
        expected_size = int(receipt.get("size_bytes", 0) or 0)
        actual_size = _size_bytes(canonical)
        if expected_size <= 0 or actual_size != expected_size:
            result["errors"].append(
                f"part_{index}:size_mismatch:{canonical.name}:expected={expected_size}:actual={actual_size}"
            )
            continue
        expected_sha = str(receipt.get("sha256") or "").strip().lower()
        actual_sha = _sha256(canonical)
        if not expected_sha or actual_sha != expected_sha:
            result["errors"].append(f"part_{index}:sha256_mismatch:{canonical.name}")
            continue
        if str(receipt.get("output_path") or "") != canonical_text:
            result["changed_path_count"] += 1
        receipt["output_path"] = canonical_text
        repaired_parts.append(receipt)
        result["validated_part_count"] += 1

    if result["errors"] or len(repaired_parts) != len(parts):
        return result
    repaired = dict(payload)
    repaired["parts"] = repaired_parts
    if isinstance(repaired.get("range_receipts"), list):
        by_range = {
            (int(row.get("start_id", 0) or 0), int(row.get("end_id", 0) or 0)): row
            for row in repaired_parts
            if isinstance(row, dict)
        }
        repaired["range_receipts"] = [
            dict(by_range.get((int(row.get("start_id", 0) or 0), int(row.get("end_id", 0) or 0)), row))
            if isinstance(row, dict)
            else row
            for row in repaired["range_receipts"]
        ]
    repaired["status"] = str(repaired.get("status") or "complete")
    repaired["dataset_root"] = str(dataset_root)
    repaired["part_count"] = len(parts)
    repaired["updated_at_utc"] = _iso()
    repaired["path_repair"] = {
        "schema_version": 1,
        "repaired_at_utc": _iso(),
        "validated_part_count": int(result["validated_part_count"]),
        "validation": "canonical_path_size_and_sha256",
        "legacy_manifest_schema_preserved": int(payload.get("schema_version", 0) or 0) < 2,
    }
    if apply:
        _write_json(path, repaired)
    result["ok"] = True
    result["overall_status"] = "repaired" if apply and result["changed_path_count"] else "verified"
    result["legacy_schema_preserved"] = int(payload.get("schema_version", 0) or 0) < 2
    return result


def _export_arrow_id_range(
    source_db: Path,
    *,
    table: str,
    cutoff: str,
    start_id: int,
    end_id: int,
    out_path: Path,
    compression: str,
    batch_size: int,
) -> dict[str, Any]:
    import resource

    import pyarrow as pa
    import pyarrow.parquet as pq

    codec = str(compression or "zstd").strip().lower()
    if codec not in {"zstd", "snappy", "gzip", "lz4", "none"}:
        raise ValueError(f"unsupported parquet compression: {compression}")
    with closing(_connect(source_db, readonly=True, timeout_seconds=300)) as conn:
        columns = _table_columns(conn, table)
        names = [name for name, _ in columns]
        if "id" not in names or "ingested_at" not in names:
            raise RuntimeError(f"isolated_arrow_required_columns_missing:{table}")
        quoted_columns = ", ".join(_quote_ident(name) for name in names)
        cursor = conn.execute(
            f"SELECT {quoted_columns} FROM {_quote_ident(table)} "
            "WHERE id >= ? AND id < ? AND ingested_at < ?",
            (int(start_id), int(end_id), cutoff),
        )
        memory_pool = pa.system_memory_pool()
        max_batch = max(int(os.getenv("BOT_LOGS_SQLITE_ARROW_WORKER_BATCH_ROWS", "100")), 100)
        effective_batch = min(max(int(batch_size), 100), max_batch)
        writer = None
        rows_exported = 0
        batches_written = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.unlink(missing_ok=True)
        try:
            while True:
                rows = cursor.fetchmany(effective_batch)
                if not rows:
                    break
                arrays = []
                for index, (_, sql_type) in enumerate(columns):
                    values = [row[index] for row in rows]
                    arrays.append(pa.array(values, type=_arrow_type(sql_type), memory_pool=memory_pool))
                table_batch = pa.Table.from_arrays(arrays, names=names)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_path), table_batch.schema, compression=None if codec == "none" else codec)
                writer.write_table(table_batch, row_group_size=len(rows))
                rows_exported += len(rows)
                batches_written += 1
                del table_batch, arrays, rows, values
                gc.collect()
                memory_pool.release_unused()
        except Exception:
            out_path.unlink(missing_ok=True)
            raise
        finally:
            if writer is not None:
                writer.close()
    if rows_exported <= 0:
        out_path.unlink(missing_ok=True)
        sha256 = ""
        size_bytes = 0
    else:
        with out_path.open("rb") as handle:
            os.fsync(handle.fileno())
        size_bytes = _size_bytes(out_path)
        sha256 = _sha256(out_path)
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "ok": True,
        "table": table,
        "start_id": int(start_id),
        "end_id": int(end_id),
        "rows_exported": int(rows_exported),
        "output_path": str(out_path) if rows_exported > 0 else "",
        "size_bytes": int(size_bytes),
        "sha256": sha256,
        "effective_batch_size": int(effective_batch),
        "batches_written": int(batches_written),
        "max_rss_bytes": max_rss,
    }


def _export_cold_table_to_parquet_isolated(
    source_db: Path,
    *,
    table: str,
    cutoff: str,
    out_path: Path,
    compression: str,
    free_guard_root: Path,
    min_free_after_bytes: int,
) -> dict[str, Any]:
    with closing(_connect(source_db, readonly=True, timeout_seconds=300)) as conn:
        if not _table_exists(conn, table):
            return {"table": table, "rows_exported": 0, "output_path": "", "skipped_reason": "table_missing"}
        columns = [name for name, _ in _table_columns(conn, table)]
        if "id" not in columns or "ingested_at" not in columns:
            raise RuntimeError(f"isolated_arrow_required_columns_missing:{table}")
        bounds = conn.execute(
            f"SELECT MIN(id), MAX(id) FROM {_quote_ident(table)} WHERE ingested_at < ?",
            (cutoff,),
        ).fetchone()
    min_id = int(bounds[0]) if bounds and bounds[0] is not None else None
    max_id = int(bounds[1]) if bounds and bounds[1] is not None else None
    if min_id is None or max_id is None:
        return {"table": table, "rows_exported": 0, "output_path": "", "size_bytes": 0, "engine": "isolated_pyarrow"}

    free_before = _disk_free_bytes(free_guard_root)
    if free_before is None or free_before < max(int(min_free_after_bytes), 0):
        raise RuntimeError("cold_export_free_space_guard")
    partial_root = out_path.with_name(f".{out_path.name}.partial_dataset")
    id_span = max(int(os.getenv("BOT_LOGS_SQLITE_ARROW_WORKER_ID_SPAN", "50000")), 1000)
    min_id_span = max(int(os.getenv("BOT_LOGS_SQLITE_ARROW_WORKER_MIN_ID_SPAN", "1000")), 100)
    batch_size = max(int(os.getenv("BOT_LOGS_SQLITE_ARROW_WORKER_BATCH_ROWS", "100")), 100)
    timeout_seconds = max(int(os.getenv("BOT_LOGS_SQLITE_ARROW_WORKER_TIMEOUT_SECONDS", "1800")), 60)
    source_signature = _source_signature(source_db)
    checkpoint_path = partial_root / "_checkpoint.json"

    def canonical_receipt(raw: dict[str, Any], root: Path) -> dict[str, Any]:
        receipt = dict(raw)
        start = int(receipt.get("start_id", 0) or 0)
        end = int(receipt.get("end_id", 0) or 0)
        rows = int(receipt.get("rows_exported", 0) or 0)
        receipt["start_id"] = start
        receipt["end_id"] = end
        receipt["rows_exported"] = rows
        receipt["output_path"] = (
            str(root / f"part_{start:012d}_{end - 1:012d}.parquet") if rows > 0 else ""
        )
        return receipt

    def metadata_matches(payload: dict[str, Any]) -> bool:
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        return bool(
            int(payload.get("schema_version", 0) or 0) >= 2
            and str(payload.get("engine") or "") == "isolated_pyarrow"
            and str(payload.get("table") or "") == table
            and str(payload.get("cutoff_utc") or "") == cutoff
            and int(payload.get("min_id", -1) or -1) == min_id
            and int(payload.get("max_id", -1) or -1) == max_id
            and payload.get("source_signature") == source_signature
            and int(config.get("id_span", 0) or 0) == id_span
            and int(config.get("min_id_span", 0) or 0) == min_id_span
            and int(config.get("batch_size", 0) or 0) == batch_size
            and str(config.get("compression") or "") == str(compression)
        )

    def validated_receipts(payload: dict[str, Any], root: Path) -> list[dict[str, Any]]:
        raw_receipts = payload.get("range_receipts")
        if not isinstance(raw_receipts, list):
            raw_receipts = payload.get("parts") if isinstance(payload.get("parts"), list) else []
        valid: list[dict[str, Any]] = []
        previous_end = min_id
        for raw in sorted(
            (row for row in raw_receipts if isinstance(row, dict)),
            key=lambda row: (int(row.get("start_id", 0) or 0), int(row.get("end_id", 0) or 0)),
        ):
            receipt = canonical_receipt(raw, root)
            start = int(receipt["start_id"])
            end = int(receipt["end_id"])
            rows = int(receipt["rows_exported"])
            if start < min_id or end > max_id + 1 or end <= start or start < previous_end:
                continue
            part_path = Path(str(receipt.get("output_path") or "")) if rows > 0 else None
            if rows > 0:
                if part_path is None or not part_path.is_file():
                    continue
                if _size_bytes(part_path) != int(receipt.get("size_bytes", -1) or -1):
                    part_path.unlink(missing_ok=True)
                    continue
                expected_hash = str(receipt.get("sha256") or "")
                if not expected_hash or _sha256(part_path) != expected_hash:
                    part_path.unlink(missing_ok=True)
                    continue
            valid.append(receipt)
            previous_end = end
        return valid

    def uncovered_ranges(start_id: int, end_id: int, rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
        cursor = int(start_id)
        gaps: list[tuple[int, int]] = []
        for receipt in rows:
            receipt_start = int(receipt.get("start_id", 0) or 0)
            receipt_end = int(receipt.get("end_id", 0) or 0)
            if receipt_end <= cursor or receipt_start >= end_id:
                continue
            if receipt_start > cursor:
                gaps.append((cursor, min(receipt_start, end_id)))
            cursor = max(cursor, receipt_end)
            if cursor >= end_id:
                break
        if cursor < end_id:
            gaps.append((cursor, end_id))
        return [(start, end) for start, end in gaps if end > start]

    def manifest_payload(
        receipts: list[dict[str, Any]],
        *,
        status: str,
        root: Path,
        error: str = "",
    ) -> dict[str, Any]:
        canonical = [canonical_receipt(row, root) for row in receipts]
        parts = [row for row in canonical if int(row.get("rows_exported", 0) or 0) > 0]
        payload = {
            "schema_version": 2,
            "created_at_utc": _iso(),
            "updated_at_utc": _iso(),
            "status": status,
            "engine": "isolated_pyarrow",
            "source_db": str(source_db),
            "source_signature": source_signature,
            "table": table,
            "cutoff_utc": cutoff,
            "min_id": min_id,
            "max_id": max_id,
            "rows_exported": sum(int(row.get("rows_exported", 0) or 0) for row in canonical),
            "part_count": len(parts),
            "range_count": len(canonical),
            "config": {
                "id_span": id_span,
                "min_id_span": min_id_span,
                "batch_size": batch_size,
                "compression": str(compression),
            },
            "range_receipts": canonical,
            "parts": parts,
        }
        if error:
            payload["last_error"] = error
        return payload

    if out_path.exists():
        completed_manifest = _load_json(out_path / "_manifest.json")
        if metadata_matches(completed_manifest):
            completed_receipts = validated_receipts(completed_manifest, out_path)
            if not uncovered_ranges(min_id, max_id + 1, completed_receipts):
                rows_exported = sum(int(row.get("rows_exported", 0) or 0) for row in completed_receipts)
                return {
                    "table": table,
                    "rows_exported": rows_exported,
                    "output_path": str(out_path),
                    "size_bytes": sum(int(row.get("size_bytes", 0) or 0) for row in completed_receipts),
                    "engine": "isolated_pyarrow",
                    "id_span": id_span,
                    "min_id_span": min_id_span,
                    "batch_size": batch_size,
                    "min_id": min_id,
                    "max_id": max_id,
                    "part_count": sum(1 for row in completed_receipts if int(row.get("rows_exported", 0) or 0) > 0),
                    "range_count": len(completed_receipts),
                    "resumed_range_count": len(completed_receipts),
                    "reused_completed_dataset": True,
                    "adaptive_split_count": 0,
                    "max_worker_rss_bytes": max(
                        (int(row.get("max_rss_bytes", 0) or 0) for row in completed_receipts), default=0
                    ),
                    "manifest_path": str(out_path / "_manifest.json"),
                    "free_bytes_before": free_before,
                    "free_bytes_after": _disk_free_bytes(free_guard_root),
                }
        raise RuntimeError("isolated_arrow_completed_target_conflict")

    receipts: list[dict[str, Any]] = []
    if partial_root.exists():
        checkpoint = _load_json(checkpoint_path)
        if metadata_matches(checkpoint):
            receipts = validated_receipts(checkpoint, partial_root)
        else:
            shutil.rmtree(partial_root, ignore_errors=True)
    partial_root.mkdir(parents=True, exist_ok=True)
    resumed_range_count = len(receipts)
    adaptive_splits = 0

    def write_checkpoint(*, status: str = "in_progress", error: str = "") -> None:
        _write_json(
            checkpoint_path,
            manifest_payload(receipts, status=status, root=partial_root, error=error),
        )

    write_checkpoint()

    def export_range(start_id: int, end_id: int) -> None:
        nonlocal adaptive_splits
        part_path = partial_root / f"part_{start_id:012d}_{end_id - 1:012d}.parquet"
        cmd = [
            str(PY),
            str(Path(__file__).resolve()),
            "--export-part-worker",
            "--worker-source-db",
            str(source_db),
            "--worker-table",
            table,
            "--worker-cutoff",
            cutoff,
            "--worker-start-id",
            str(int(start_id)),
            "--worker-end-id",
            str(int(end_id)),
            "--worker-out-path",
            str(part_path),
            "--worker-compression",
            str(compression),
            "--worker-batch-size",
            str(batch_size),
            "--json",
        ]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False, timeout=timeout_seconds)
        worker_payload: dict[str, Any] = {}
        for line in reversed((proc.stdout or "").splitlines()):
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict):
                worker_payload = parsed
                break
        if proc.returncode != 0 or not bool(worker_payload.get("ok", False)):
            part_path.unlink(missing_ok=True)
            if end_id - start_id > min_id_span:
                adaptive_splits += 1
                midpoint = start_id + max((end_id - start_id) // 2, 1)
                export_range(start_id, midpoint)
                export_range(midpoint, end_id)
                return
            error = str(worker_payload.get("error") or (proc.stderr or "").strip() or f"worker_rc={proc.returncode}")
            raise RuntimeError(f"isolated_arrow_worker_failed:{start_id}:{end_id}:{error}")
        receipts.append(canonical_receipt(worker_payload, partial_root))
        receipts.sort(key=lambda row: (int(row.get("start_id", 0) or 0), int(row.get("end_id", 0) or 0)))
        write_checkpoint()
        free_now = _disk_free_bytes(free_guard_root)
        if free_now is None or free_now < max(int(min_free_after_bytes), 0):
            raise RuntimeError("cold_export_free_space_guard")

    try:
        range_start = min_id
        while range_start <= max_id:
            range_end = min(range_start + id_span, max_id + 1)
            for gap_start, gap_end in uncovered_ranges(range_start, range_end, receipts):
                export_range(gap_start, gap_end)
            range_start = range_end
        remaining = uncovered_ranges(min_id, max_id + 1, receipts)
        if remaining:
            raise RuntimeError(f"isolated_arrow_resume_coverage_gap:{remaining[:5]}")
        rows_exported = sum(int(row.get("rows_exported", 0) or 0) for row in receipts)
        manifest = manifest_payload(receipts, status="complete", root=out_path)
        checkpoint_path.unlink(missing_ok=True)
        _write_json(partial_root / "_manifest.json", manifest)
        partial_root.replace(out_path)
    except BaseException as exc:
        try:
            write_checkpoint(status="interrupted", error=f"{type(exc).__name__}:{exc}")
        except Exception:
            pass
        raise

    total_size = sum(int(row.get("size_bytes", 0) or 0) for row in receipts)
    free_after = _disk_free_bytes(free_guard_root)
    return {
        "table": table,
        "rows_exported": rows_exported,
        "output_path": str(out_path),
        "size_bytes": total_size,
        "engine": "isolated_pyarrow",
        "id_span": id_span,
        "min_id_span": min_id_span,
        "batch_size": batch_size,
        "min_id": min_id,
        "max_id": max_id,
        "part_count": sum(1 for row in receipts if int(row.get("rows_exported", 0) or 0) > 0),
        "range_count": len(receipts),
        "resumed_range_count": resumed_range_count,
        "reused_completed_dataset": False,
        "adaptive_split_count": adaptive_splits,
        "max_worker_rss_bytes": max((int(row.get("max_rss_bytes", 0) or 0) for row in receipts), default=0),
        "manifest_path": str(out_path / "_manifest.json"),
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
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


def _move_sqlite_family(source: Path, destination: Path) -> list[dict[str, str]]:
    moved: list[dict[str, str]] = []
    for src, dst in zip([source, *_sidecars(source)], [destination, *_sidecars(destination)]):
        if not src.exists():
            continue
        src.replace(dst)
        moved.append({"source": str(src), "destination": str(dst)})
    return moved


def build_local_cache_payload(
    project_root: Path,
    *,
    relative_path: str,
    hot_hours: float,
    apply: bool,
    prune_old_cache: bool,
    min_local_free_after_gb: float,
    min_external_free_after_gb: float,
    cold_export_root: Path,
    batch_size: int,
    compression: str,
    require_writer_idle: bool,
    timeout_seconds: float,
    external_root: Path | None = None,
) -> dict[str, Any]:
    relative_path = str(relative_path or "data/jsonl_link.sqlite3").replace("\\", "/").lstrip("./")
    source_db = project_root / "local_fallback_storage" / relative_path
    repo_db = project_root / relative_path
    external_root = external_root or resolve_external_storage().external_root
    cold_export_root = cold_export_root.expanduser()
    transaction_path = project_root / "governance" / "health" / "storage_sqlite_hot_route_transaction.json"
    previous_transaction = _load_json(transaction_path) if apply else {}
    source_signature_before = _source_signature(source_db) if source_db.is_file() else {}
    transaction_resumed = False
    discarded_partial_datasets: list[str] = []
    now = _utc_now()
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    cutoff_dt = now - timedelta(hours=max(float(hot_hours), 1.0))
    cutoff = cutoff_dt.isoformat()
    if previous_transaction:
        compatible = bool(
            str(previous_transaction.get("mode") or "") == "rebuild_local_compatibility_cache"
            and str(previous_transaction.get("relative_path") or "") == relative_path
            and str(previous_transaction.get("cold_export_root") or "") == str(cold_export_root)
            and abs(float(previous_transaction.get("hot_hours", -1.0) or -1.0) - float(hot_hours)) < 1e-9
            and previous_transaction.get("source_signature") == source_signature_before
        )
        if compatible:
            try:
                run_id = str(previous_transaction["run_id"])
                cutoff = str(previous_transaction["cutoff_utc"])
                cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
                transaction_resumed = bool(run_id and cutoff)
            except Exception:
                transaction_resumed = False
        if not transaction_resumed:
            discarded_partial_datasets = _discard_partial_datasets(previous_transaction, cold_export_root)
            transaction_path.unlink(missing_ok=True)
    external_tmp = external_root / "data" / "maintenance_staging" / f"{source_db.name}.local_cache_{run_id}.tmp"
    local_staged = source_db.with_name(f".{source_db.name}.local_cache_{run_id}.tmp")
    old_cache = source_db.with_name(f"{source_db.name}.pre_local_cache_{run_id}.bak")
    writer_snapshot = writer_state.writer_state_snapshot(project_root)
    writer_idle = not bool(writer_snapshot.get("active", False))
    local_free_before = _disk_free_bytes(source_db.parent)
    external_free_before = _disk_free_bytes(external_root)
    min_local_free_after_bytes = int(max(float(min_local_free_after_gb), 0.0) * (1024**3))
    min_external_free_after_bytes = int(max(float(min_external_free_after_gb), 0.0) * (1024**3))
    source_bytes = _size_bytes(source_db)
    payload: dict[str, Any] = {
        "timestamp_utc": _iso(),
        "schema_version": 2,
        "mode": "rebuild_local_compatibility_cache",
        "ok": False,
        "overall_status": "dry_run",
        "apply": bool(apply),
        "prune_old_cache": bool(prune_old_cache),
        "project_root": str(project_root),
        "relative_path": relative_path,
        "repo_db": str(repo_db),
        "source_db": str(source_db),
        "external_staging_db": str(external_tmp),
        "local_staged_db": str(local_staged),
        "old_cache_db": str(old_cache),
        "cold_export_root": str(cold_export_root),
        "hot_hours": float(hot_hours),
        "cutoff_utc": cutoff,
        "source_size_bytes": int(source_bytes),
        "local_free_bytes_before": local_free_before,
        "external_free_bytes_before": external_free_before,
        "min_local_free_after_bytes": min_local_free_after_bytes,
        "min_external_free_after_bytes": min_external_free_after_bytes,
        "writer_idle_required": bool(require_writer_idle),
        "writer_idle": bool(writer_idle),
        "writer_state": writer_snapshot,
        "source_role": "compatibility_cache",
        "authority_policy": "cold rows are exported with exact row coverage before the verified derived cache is replaced",
        "cold_export_engine": str(os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_EXPORT_ENGINE", "isolated_pyarrow") or "isolated_pyarrow").strip().lower(),
        "source_counts": {},
        "estimated_hot_db_bytes": 0,
        "cold_exports": [],
        "hot_db": {},
        "coverage_check": {},
        "atomic_switch": {},
        "transaction": {
            "path": str(transaction_path),
            "run_id": run_id,
            "resumed": transaction_resumed,
            "discarded_partial_datasets": discarded_partial_datasets,
        },
        "blockers": [],
    }
    if not source_db.is_file():
        payload["blockers"].append("source_db_missing")
    if not external_root.is_dir() or not os.access(external_root, os.W_OK):
        payload["blockers"].append("external_root_not_writable")
    if require_writer_idle and not writer_idle:
        payload["blockers"].append("writer_not_idle")
    if external_free_before is None or external_free_before < min_external_free_after_bytes:
        payload["blockers"].append("external_free_below_guard")
    if source_db.is_file():
        try:
            with closing(_connect(source_db, readonly=True, timeout_seconds=timeout_seconds)) as src:
                payload["source_counts"] = _source_counts(src, cutoff)
                payload["estimated_hot_db_bytes"] = _estimated_hot_db_bytes(src, cutoff, source_bytes)
        except Exception as exc:
            payload["blockers"].append(f"source_inspection_error:{type(exc).__name__}:{exc}")
    estimated_hot_bytes = int(payload.get("estimated_hot_db_bytes", 0) or 0)
    if local_free_before is None or local_free_before - estimated_hot_bytes < min_local_free_after_bytes:
        payload["blockers"].append("local_free_after_staged_cache_below_guard")
    if payload["blockers"]:
        payload["overall_status"] = "blocked"
        return payload
    if not apply:
        payload["ok"] = True
        return payload

    external_tmp.parent.mkdir(parents=True, exist_ok=True)
    cold_export_root.mkdir(parents=True, exist_ok=True)
    export_paths = {
        table: cold_export_root / f"{table}_cold_before_{cutoff_dt.strftime('%Y%m%dT%H%M%SZ')}_{run_id}.parquet"
        for table in ("jsonl_records", "json_file_records")
    }
    try:
        if not transaction_resumed:
            with closing(_connect(source_db, readonly=False, timeout_seconds=timeout_seconds)) as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        source_signature_after_checkpoint = _source_signature(source_db)
        if transaction_resumed and source_signature_after_checkpoint != source_signature_before:
            raise RuntimeError("source_changed_while_resuming_archive_transaction")
        transaction = {
            "schema_version": 1,
            "status": "in_progress",
            "updated_at_utc": _iso(),
            "mode": "rebuild_local_compatibility_cache",
            "run_id": run_id,
            "relative_path": relative_path,
            "hot_hours": float(hot_hours),
            "cutoff_utc": cutoff,
            "source_db": str(source_db),
            "source_signature": source_signature_after_checkpoint,
            "cold_export_root": str(cold_export_root),
            "export_paths": {name: str(path) for name, path in export_paths.items()},
            "partial_datasets": [
                str(path.with_name(f".{path.name}.partial_dataset")) for path in export_paths.values()
            ],
            "completed_exports": [],
        }
        if transaction_resumed:
            transaction["completed_exports"] = list(previous_transaction.get("completed_exports", []))
        _write_json(transaction_path, transaction)

        export_engine = str(payload["cold_export_engine"])
        if export_engine not in {"isolated_pyarrow", "duckdb", "pyarrow"}:
            raise ValueError(f"unsupported cold export engine: {export_engine}")
        with closing(_connect(source_db, readonly=True, timeout_seconds=timeout_seconds)) as src:
            for table in ("jsonl_records", "json_file_records"):
                export_path = export_paths[table]
                if export_engine == "isolated_pyarrow":
                    export = _export_cold_table_to_parquet_isolated(
                        source_db,
                        table=table,
                        cutoff=cutoff,
                        out_path=export_path,
                        compression=compression,
                        free_guard_root=external_root,
                        min_free_after_bytes=min_external_free_after_bytes,
                    )
                elif export_engine == "duckdb":
                    export = _export_cold_table_to_parquet_duckdb(
                        source_db,
                        table=table,
                        cutoff=cutoff,
                        out_path=export_path,
                        compression=compression,
                        free_guard_root=external_root,
                        min_free_after_bytes=min_external_free_after_bytes,
                    )
                else:
                    export = _export_cold_table_to_parquet(
                        src,
                        table=table,
                        cutoff=cutoff,
                        out_path=export_path,
                        batch_size=max(int(batch_size), 1000),
                        compression=compression,
                        free_guard_root=external_root,
                        min_free_after_bytes=min_external_free_after_bytes,
                    )
                payload["cold_exports"].append(export)
                transaction["completed_exports"] = sorted(
                    {str(item) for item in [*transaction.get("completed_exports", []), table] if str(item)}
                )
                transaction["updated_at_utc"] = _iso()
                _write_json(transaction_path, transaction)

        hot_db = _build_hot_db(
            source_db=source_db,
            dest_tmp=external_tmp,
            cutoff=cutoff,
            timeout_seconds=timeout_seconds,
        )
        payload["hot_db"] = hot_db
        if not bool((hot_db.get("quick_check") or {}).get("ok", False)):
            payload["overall_status"] = "hot_db_quick_check_failed"
            return payload

        source_counts = payload["source_counts"]
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
            covered = int(copied_rows.get(table, 0) or 0) + int(exported_by_table.get(table, 0) or 0)
            if expected != covered:
                coverage_errors.append(f"{table}:expected={expected}:covered={covered}")
        payload["coverage_check"] = {"ok": not coverage_errors, "errors": coverage_errors}
        if coverage_errors:
            payload["overall_status"] = "coverage_check_failed"
            return payload

        staged_bytes = _size_bytes(external_tmp)
        current_local_free = _disk_free_bytes(source_db.parent)
        if current_local_free is None or current_local_free - staged_bytes < min_local_free_after_bytes:
            payload["overall_status"] = "local_free_before_copy_below_guard"
            payload["blockers"].append("local_free_before_copy_below_guard")
            return payload
        shutil.copy2(external_tmp, local_staged)
        with local_staged.open("rb") as handle:
            os.fsync(handle.fileno())
        staged_quick = _quick_check(local_staged, timeout_seconds=timeout_seconds)
        payload["local_staged_quick_check"] = staged_quick
        if not bool(staged_quick.get("ok", False)):
            payload["overall_status"] = "local_staged_quick_check_failed"
            return payload

        moved_old = _move_sqlite_family(source_db, old_cache)
        try:
            local_staged.replace(source_db)
            route_quick = _quick_check(repo_db, timeout_seconds=timeout_seconds)
            payload["route_quick_check"] = route_quick
            if not bool(route_quick.get("ok", False)):
                raise RuntimeError(f"replacement_route_quick_check_failed:{route_quick.get('result')}")
        except Exception:
            failed_replacement = source_db.with_name(f"{source_db.name}.failed_local_cache_{run_id}")
            if source_db.exists():
                source_db.replace(failed_replacement)
            _move_sqlite_family(old_cache, source_db)
            raise
        payload["atomic_switch"] = {
            "old_cache_moves": moved_old,
            "new_cache_path": str(source_db),
            "route_path": str(repo_db),
            "route_resolved": str(repo_db.resolve(strict=False)),
        }
        if prune_old_cache:
            payload["old_cache_prune"] = _unlink_artifact(old_cache)
            if payload["old_cache_prune"].get("errors"):
                payload["overall_status"] = "switched_prune_errors"
                return payload
        payload["new_cache_size_bytes"] = _size_bytes(source_db)
        payload["reclaimed_bytes"] = max(source_bytes - _size_bytes(source_db), 0) if prune_old_cache else 0
        payload["local_free_bytes_after"] = _disk_free_bytes(source_db.parent)
        payload["external_free_bytes_after"] = _disk_free_bytes(external_root)
        payload["ok"] = True
        payload["overall_status"] = "rebuilt_pruned" if prune_old_cache else "rebuilt_backup_retained"
        payload["transaction"]["status"] = "complete"
        transaction_path.unlink(missing_ok=True)
        return payload
    except Exception as exc:
        payload["overall_status"] = "failed"
        payload["blockers"].append(f"rebuild_error:{type(exc).__name__}:{exc}")
        transaction = _load_json(transaction_path)
        if transaction:
            transaction["status"] = "interrupted"
            transaction["updated_at_utc"] = _iso()
            transaction["last_error"] = f"{type(exc).__name__}:{exc}"
            _write_json(transaction_path, transaction)
        payload["transaction"]["status"] = "interrupted"
        return payload
    finally:
        for temporary in (local_staged, external_tmp):
            if temporary.exists():
                temporary.unlink()


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
    writer_idle = not bool(writer_snapshot.get("active", False))

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
            with closing(_connect(source_db, readonly=True, timeout_seconds=timeout_seconds)) as src:
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

    with closing(_connect(source_db, readonly=True, timeout_seconds=timeout_seconds)) as src:
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


def _wait_for_writer_idle(
    project_root: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        snapshot = writer_state.writer_state_snapshot(project_root)
        if not bool(snapshot.get("active", False)):
            return {
                "ok": True,
                "timed_out": False,
                "attempts": attempts,
                "waited_seconds": round(time.monotonic() - started, 3),
                "final_state": snapshot,
            }
        waited = time.monotonic() - started
        if waited >= max(float(timeout_seconds), 0.0):
            return {
                "ok": False,
                "timed_out": True,
                "attempts": attempts,
                "waited_seconds": round(waited, 3),
                "final_state": snapshot,
            }
        time.sleep(max(float(poll_seconds), 0.1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact external hot SQLite route and optional cold parquet archive for jsonl_link.sqlite3.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--relative-path", default="data/jsonl_link.sqlite3")
    parser.add_argument("--hot-hours", type=float, default=float(os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_HOURS", "18")))
    parser.add_argument("--cold-export-root", default=os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_COLD_ROOT", ""))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_BATCH_SIZE", "25000")))
    parser.add_argument("--compression", default=os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_COMPRESSION", "zstd"))
    parser.add_argument("--min-external-free-after-gb", type=float, default=float(os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_MIN_FREE_AFTER_GB", "40")))
    parser.add_argument("--min-local-free-after-gb", type=float, default=float(os.getenv("BOT_LOGS_SQLITE_LOCAL_CACHE_MIN_FREE_AFTER_GB", "32")))
    parser.add_argument("--sqlite-timeout-seconds", type=float, default=float(os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_TIMEOUT_SECONDS", "300")))
    parser.add_argument("--no-require-writer-idle", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--prune-local", action="store_true")
    parser.add_argument("--rebuild-local-cache", action="store_true")
    parser.add_argument("--prune-old-cache", action="store_true")
    parser.add_argument(
        "--no-coordinate-maintenance-hold",
        action="store_true",
        help="Do not automatically engage a runtime maintenance hold and wait for the SQLite writer handoff.",
    )
    parser.add_argument(
        "--writer-drain-timeout-seconds",
        type=float,
        default=float(os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_WRITER_DRAIN_TIMEOUT_SECONDS", "900")),
    )
    parser.add_argument(
        "--writer-drain-poll-seconds",
        type=float,
        default=float(os.getenv("BOT_LOGS_SQLITE_HOT_ROUTE_WRITER_DRAIN_POLL_SECONDS", "5")),
    )
    parser.add_argument("--export-part-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-source-db", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-table", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-cutoff", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-start-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-end-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-compression", default="zstd", help=argparse.SUPPRESS)
    parser.add_argument("--worker-batch-size", type=int, default=100, help=argparse.SUPPRESS)
    parser.add_argument(
        "--repair-final-manifest",
        action="append",
        default=[],
        help="Verify a finalized parquet manifest and atomically repair legacy part paths to its dataset directory.",
    )
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.export_part_worker:
        try:
            payload = _export_arrow_id_range(
                Path(args.worker_source_db).expanduser(),
                table=str(args.worker_table),
                cutoff=str(args.worker_cutoff),
                start_id=int(args.worker_start_id),
                end_id=int(args.worker_end_id),
                out_path=Path(args.worker_out_path).expanduser(),
                compression=str(args.worker_compression),
                batch_size=max(int(args.worker_batch_size), 100),
            )
            rc = 0
        except Exception as exc:
            payload = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}
            rc = 1
        print(json.dumps(payload, ensure_ascii=True))
        return rc

    if args.repair_final_manifest:
        repairs = [
            repair_final_manifest_paths(Path(raw), apply=bool(args.apply))
            for raw in args.repair_final_manifest
        ]
        payload = {
            "ok": all(bool(row.get("ok", False)) for row in repairs),
            "overall_status": "repaired" if args.apply else "verified",
            "apply": bool(args.apply),
            "repairs": repairs,
        }
        out_file = Path(args.out_file).expanduser()
        _write_json(out_file, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print(
                "storage_sqlite_hot_route "
                f"status={payload.get('overall_status')} "
                f"ok={int(bool(payload.get('ok', False)))} "
                f"manifests={len(repairs)}"
            )
        return 0 if bool(payload.get("ok", False)) else 1

    project_root = Path(args.project_root).expanduser().resolve()
    external_root = resolve_external_storage().external_root
    cold_root = (
        Path(args.cold_export_root).expanduser()
        if str(args.cold_export_root or "").strip()
        else external_root / "data" / "cold_archives" / "jsonl_link_hot_route"
    )
    if args.rebuild_local_cache:
        coordination_enabled = bool(args.apply and not args.no_coordinate_maintenance_hold)
        coordination: dict[str, Any] = {
            "enabled": coordination_enabled,
            "owned_hold": False,
            "hold_before": {},
            "writer_handoff": {},
            "hold_release": {},
        }
        owned_token = ""
        if coordination_enabled:
            hold_before = maintenance_hold_snapshot(project_root)
            coordination["hold_before"] = hold_before
            if bool(hold_before.get("active", False)) and not bool(hold_before.get("valid", False)):
                payload = {
                    "ok": False,
                    "overall_status": "blocked",
                    "blockers": ["runtime_maintenance_hold_invalid"],
                    "maintenance_coordination": coordination,
                }
            else:
                if bool(hold_before.get("active", False)):
                    coordination["hold_engaged"] = hold_before
                else:
                    engaged = engage_maintenance_hold(
                        project_root,
                        reason="transactional_local_compatibility_cache_rebuild",
                        owner="storage_sqlite_hot_route",
                        ttl_seconds=max(int(float(args.writer_drain_timeout_seconds)) + 12 * 60 * 60, 3600),
                    )
                    coordination["hold_engaged"] = engaged
                    coordination["owned_hold"] = True
                    owned_token = str(engaged.get("token") or "")
                try:
                    handoff = _wait_for_writer_idle(
                        project_root,
                        timeout_seconds=max(float(args.writer_drain_timeout_seconds), 0.0),
                        poll_seconds=max(float(args.writer_drain_poll_seconds), 0.1),
                    )
                    coordination["writer_handoff"] = handoff
                    if not bool(handoff.get("ok", False)):
                        payload = {
                            "ok": False,
                            "overall_status": "blocked",
                            "blockers": ["writer_handoff_timeout"],
                        }
                    else:
                        payload = build_local_cache_payload(
                            project_root,
                            relative_path=str(args.relative_path),
                            hot_hours=float(args.hot_hours),
                            apply=True,
                            prune_old_cache=bool(args.prune_old_cache),
                            min_local_free_after_gb=float(args.min_local_free_after_gb),
                            min_external_free_after_gb=float(args.min_external_free_after_gb),
                            cold_export_root=cold_root,
                            batch_size=max(int(args.batch_size), 1000),
                            compression=str(args.compression or "zstd"),
                            require_writer_idle=not bool(args.no_require_writer_idle),
                            timeout_seconds=max(float(args.sqlite_timeout_seconds), 1.0),
                        )
                finally:
                    if owned_token:
                        coordination["hold_release"] = release_maintenance_hold(
                            project_root,
                            expected_token=owned_token,
                        )
                payload["maintenance_coordination"] = coordination
        else:
            payload = build_local_cache_payload(
                project_root,
                relative_path=str(args.relative_path),
                hot_hours=float(args.hot_hours),
                apply=bool(args.apply),
                prune_old_cache=bool(args.prune_old_cache),
                min_local_free_after_gb=float(args.min_local_free_after_gb),
                min_external_free_after_gb=float(args.min_external_free_after_gb),
                cold_export_root=cold_root,
                batch_size=max(int(args.batch_size), 1000),
                compression=str(args.compression or "zstd"),
                require_writer_idle=not bool(args.no_require_writer_idle),
                timeout_seconds=max(float(args.sqlite_timeout_seconds), 1.0),
            )
    else:
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
