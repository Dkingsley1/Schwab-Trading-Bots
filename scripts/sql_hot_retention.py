import argparse
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_FILE_RE = re.compile(r"^jsonl_link_archive_(\d{4})_(\d{2})\.sqlite3$")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return bool(row)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _ensure_archive_schema(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    if _table_exists(dst, "jsonl_records"):
        return

    src_create = src.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='jsonl_records'"
    ).fetchone()
    if src_create and src_create[0]:
        dst.execute(src_create[0])
    dst.execute("CREATE INDEX IF NOT EXISTS idx_jsonl_archive_ingested_at ON jsonl_records(ingested_at)")
    dst.execute("CREATE INDEX IF NOT EXISTS idx_jsonl_archive_source_rel_line ON jsonl_records(source_rel, line_no)")
    dst.commit()


def _archive_key(ingested_at: str, period: str) -> str:
    raw = str(ingested_at or "")
    if period == "month" and len(raw) >= 7:
        return raw[:7].replace("-", "_")
    return "single"


def _archive_path_for_key(*, archive_db: Path, archive_root: Path | None, key: str) -> Path:
    if key == "single" or archive_root is None:
        return archive_db
    archive_root.mkdir(parents=True, exist_ok=True)
    return archive_root / f"jsonl_link_archive_{key}.sqlite3"


def _archive_db_candidates(*, archive_db: Path, archive_root: Path | None) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for candidate in [archive_db]:
        if candidate.exists():
            key = str(candidate.resolve())
            if key not in seen:
                seen.add(key)
                out.append(candidate)
    if archive_root is not None and archive_root.exists():
        for candidate in sorted(archive_root.glob("jsonl_link_archive_*.sqlite3")):
            if not candidate.is_file():
                continue
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
    return out


def _archive_month_fully_before_cutoff(path: Path, cutoff_dt: datetime) -> bool:
    match = ARCHIVE_FILE_RE.match(path.name)
    if match is None:
        return False
    year = int(match.group(1))
    month = int(match.group(2))
    if month == 12:
        next_month = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_month = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return next_month <= cutoff_dt


def _count_archive_rows(path: Path) -> int:
    if not path.exists():
        return 0
    conn = _connect(path)
    try:
        if not _table_exists(conn, "jsonl_records"):
            return 0
        row = conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()
        return int(row[0] if row and row[0] is not None else 0)
    finally:
        conn.close()


def _unlink_sqlite_artifact(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            if candidate.exists():
                candidate.unlink()
        except OSError:
            continue


def _sqlite_column_specs(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    return [
        (str(row[1]), str(row[2] or "TEXT"))
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        if row and len(row) >= 3 and row[1]
    ]


def _export_output_path(path: Path, *, cold_export_root: Path, cold_export_format: str) -> Path:
    ext = str(cold_export_format or "parquet").strip().lower()
    return cold_export_root / f"{path.stem}.{ext}"


def _export_sqlite_archive_to_parquet(
    path: Path,
    *,
    out_path: Path,
    batch_size: int,
    compression: str,
) -> dict[str, object]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(f"pyarrow_unavailable:{exc}") from exc

    def _arrow_type(sql_type: str):
        raw = str(sql_type or "").upper()
        if "INT" in raw:
            return pa.int64()
        if any(tok in raw for tok in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
            return pa.float64()
        if "BLOB" in raw:
            return pa.binary()
        return pa.string()

    conn = sqlite3.connect(str(path))
    writer = None
    rows_exported = 0
    min_ingested_at = ""
    max_ingested_at = ""
    try:
        if not _table_exists(conn, "jsonl_records"):
            return {
                "output_path": str(out_path),
                "rows_exported": 0,
                "min_ingested_at": "",
                "max_ingested_at": "",
            }

        col_specs = _sqlite_column_specs(conn, "jsonl_records")
        if not col_specs:
            return {
                "output_path": str(out_path),
                "rows_exported": 0,
                "min_ingested_at": "",
                "max_ingested_at": "",
            }

        col_names = [name for name, _ in col_specs]
        col_types = {name: _arrow_type(sql_type) for name, sql_type in col_specs}
        order_cols = []
        if "ingested_at" in col_names:
            order_cols.append("ingested_at ASC")
        if "id" in col_names:
            order_cols.append("id ASC")
        order_clause = f" ORDER BY {', '.join(order_cols)}" if order_cols else ""
        cursor = conn.execute(f"SELECT {','.join(col_names)} FROM jsonl_records{order_clause}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            rows = cursor.fetchmany(max(int(batch_size), 1000))
            if not rows:
                break
            batch_cols = {
                name: pa.array([row[idx] for row in rows], type=col_types[name])
                for idx, name in enumerate(col_names)
            }
            table = pa.Table.from_pydict(batch_cols)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema, compression=(compression or "zstd"))
            writer.write_table(table)
            rows_exported += len(rows)
            if "ingested_at" in col_names:
                ts_idx = col_names.index("ingested_at")
                ingested = [str(row[ts_idx]) for row in rows if row[ts_idx]]
                if ingested:
                    batch_min = min(ingested)
                    batch_max = max(ingested)
                    min_ingested_at = batch_min if (not min_ingested_at or batch_min < min_ingested_at) else min_ingested_at
                    max_ingested_at = batch_max if (not max_ingested_at or batch_max > max_ingested_at) else max_ingested_at
    finally:
        if writer is not None:
            writer.close()
        conn.close()

    return {
        "output_path": str(out_path),
        "rows_exported": int(rows_exported),
        "min_ingested_at": min_ingested_at,
        "max_ingested_at": max_ingested_at,
    }


def _prune_archive_storage(
    *,
    archive_db: Path,
    archive_root: Path | None,
    archive_retention_days: int,
    archive_prune_vacuum: bool,
    cold_export_root: Path | None,
    cold_export_format: str,
    cold_export_batch_size: int,
    cold_export_compression: str,
) -> dict[str, object]:
    retention_days = max(int(archive_retention_days), 0)
    if retention_days <= 0:
        return {
            "enabled": False,
            "retention_days": 0,
            "cutoff_utc": "",
            "pruned_rows": 0,
            "rows_pruned_by_db": {},
            "deleted_archive_files": [],
            "vacuumed_archive_dbs": [],
        }

    cutoff_dt = _now_utc() - timedelta(days=retention_days)
    cutoff = cutoff_dt.isoformat()
    pruned_rows = 0
    rows_pruned_by_db: dict[str, int] = {}
    deleted_archive_files: list[str] = []
    vacuumed_archive_dbs: list[str] = []
    cold_export = {
        "enabled": bool(cold_export_root),
        "root": str(cold_export_root) if cold_export_root else "",
        "format": str(cold_export_format or "parquet"),
        "compression": str(cold_export_compression or "zstd"),
        "exported_files": [],
        "rows_exported_by_db": {},
        "output_files": {},
        "errors": {},
    }

    for path in _archive_db_candidates(archive_db=archive_db, archive_root=archive_root):
        if archive_root is not None and path.parent == archive_root and _archive_month_fully_before_cutoff(path, cutoff_dt):
            row_count = _count_archive_rows(path)
            if cold_export_root is not None:
                export_target = _export_output_path(path, cold_export_root=cold_export_root, cold_export_format=cold_export_format)
                try:
                    export_summary = _export_sqlite_archive_to_parquet(
                        path,
                        out_path=export_target,
                        batch_size=max(int(cold_export_batch_size), 1000),
                        compression=str(cold_export_compression or "zstd"),
                    )
                    cold_export["exported_files"].append(str(path))
                    cold_export["rows_exported_by_db"][str(path)] = int(export_summary.get("rows_exported", 0) or 0)
                    cold_export["output_files"][str(path)] = str(export_summary.get("output_path") or export_target)
                except Exception as exc:
                    cold_export["errors"][str(path)] = str(exc)
                    continue
            _unlink_sqlite_artifact(path)
            deleted_archive_files.append(str(path))
            pruned_rows += int(row_count)
            if row_count > 0:
                rows_pruned_by_db[str(path)] = int(row_count)
            continue

        conn = _connect(path)
        remaining_rows = 0
        removed_rows = 0
        try:
            if not _table_exists(conn, "jsonl_records"):
                continue
            before_row = conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()
            before = int(before_row[0] if before_row and before_row[0] is not None else 0)
            conn.execute("DELETE FROM jsonl_records WHERE ingested_at < ?", (cutoff,))
            conn.commit()
            row = conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()
            remaining_rows = int(row[0] if row and row[0] is not None else 0)
            removed_rows = max(int(before) - int(remaining_rows), 0)
            if removed_rows > 0:
                pruned_rows += int(removed_rows)
                rows_pruned_by_db[str(path)] = int(removed_rows)
                if archive_prune_vacuum and remaining_rows > 0:
                    conn.execute("VACUUM")
                    vacuumed_archive_dbs.append(str(path))
        finally:
            conn.close()

        if remaining_rows == 0 and path.exists():
            _unlink_sqlite_artifact(path)
            deleted_archive_files.append(str(path))

    return {
        "enabled": True,
        "retention_days": retention_days,
        "cutoff_utc": cutoff,
        "pruned_rows": int(pruned_rows),
        "rows_pruned_by_db": rows_pruned_by_db,
        "deleted_archive_files": sorted(set(deleted_archive_files)),
        "vacuumed_archive_dbs": sorted(set(vacuumed_archive_dbs)),
        "cold_archive_export": cold_export,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep SQLite hot window small and archive older rows.")
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "jsonl_link.sqlite3"))
    parser.add_argument("--archive-db", default=str(PROJECT_ROOT / "data" / "jsonl_link_archive.sqlite3"))
    parser.add_argument("--archive-root", default="")
    parser.add_argument("--archive-period", choices=("single", "month"), default="single")
    parser.add_argument("--archive-retention-days", type=int, default=0, help="Prune archived rows/files older than this many days (0 = disabled).")
    parser.add_argument("--archive-prune-vacuum", action="store_true", help="Vacuum archive DBs after row-level pruning when rows remain.")
    parser.add_argument("--cold-export-root", default="", help="Optional root for compressed cold archive exports before old monthly archive files are deleted.")
    parser.add_argument("--cold-export-format", choices=("parquet",), default="parquet")
    parser.add_argument("--cold-export-batch-size", type=int, default=50000)
    parser.add_argument("--cold-export-compression", default="zstd")
    parser.add_argument("--hot-days", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--max-rows", type=int, default=0, help="Maximum rows to move in one pass (0 = unlimited).")
    parser.add_argument("--vacuum", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"sql_hot_retention_skip db_missing={db_path}")
        return 0

    archive_db = Path(args.archive_db)
    archive_db.parent.mkdir(parents=True, exist_ok=True)
    archive_root = Path(args.archive_root).expanduser().resolve() if str(args.archive_root).strip() else None
    cold_export_root = Path(args.cold_export_root).expanduser().resolve() if str(args.cold_export_root).strip() else None

    cutoff = (_now_utc() - timedelta(days=max(args.hot_days, 1))).isoformat()

    src = _connect(db_path)
    src.row_factory = sqlite3.Row
    if not _table_exists(src, "jsonl_records"):
        print("sql_hot_retention_skip table_missing=jsonl_records")
        src.close()
        return 0

    cols = [r[1] for r in src.execute("PRAGMA table_info(jsonl_records)").fetchall()]
    col_list = ",".join(cols)
    qmarks = ",".join(["?"] * len(cols))

    archive_conns: dict[Path, sqlite3.Connection] = {}
    archive_rows_by_db: dict[str, int] = {}
    total_moved = 0
    max_rows = max(int(args.max_rows), 0)

    try:
        while True:
            limit = max(int(args.batch_size), 1000)
            if max_rows > 0:
                remaining_budget = max_rows - total_moved
                if remaining_budget <= 0:
                    break
                limit = min(limit, remaining_budget)

            rows = src.execute(
                """
                SELECT *
                FROM jsonl_records
                WHERE ingested_at < ?
                ORDER BY ingested_at ASC, id ASC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
            if not rows:
                break

            grouped: dict[Path, list[sqlite3.Row]] = {}
            for row in rows:
                key = _archive_key(str(row["ingested_at"] or ""), str(args.archive_period))
                archive_path = _archive_path_for_key(
                    archive_db=archive_db,
                    archive_root=archive_root,
                    key=key,
                )
                grouped.setdefault(archive_path, []).append(row)

            for archive_path, group_rows in grouped.items():
                conn = archive_conns.get(archive_path)
                if conn is None:
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    conn = _connect(archive_path)
                    _ensure_archive_schema(src, conn)
                    archive_conns[archive_path] = conn

                payload = [tuple(r[c] for c in cols) for r in group_rows]
                conn.executemany(
                    f"INSERT OR IGNORE INTO jsonl_records ({col_list}) VALUES ({qmarks})",
                    payload,
                )
                conn.commit()
                archive_rows_by_db[str(archive_path)] = archive_rows_by_db.get(str(archive_path), 0) + len(group_rows)

            ids = [int(r["id"]) for r in rows]
            id_marks = ",".join("?" for _ in ids)
            src.execute(f"DELETE FROM jsonl_records WHERE id IN ({id_marks})", ids)
            src.commit()
            total_moved += len(rows)

        if args.vacuum and total_moved > 0:
            src.execute("VACUUM")

        remaining = src.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()[0]
    finally:
        src.close()
        for conn in archive_conns.values():
            conn.close()

    archive_pruning = _prune_archive_storage(
        archive_db=archive_db,
        archive_root=archive_root,
        archive_retention_days=int(args.archive_retention_days),
        archive_prune_vacuum=bool(args.archive_prune_vacuum),
        cold_export_root=cold_export_root,
        cold_export_format=str(args.cold_export_format),
        cold_export_batch_size=int(args.cold_export_batch_size),
        cold_export_compression=str(args.cold_export_compression),
    )

    out = {
        "timestamp_utc": _now_utc().isoformat(),
        "db": str(db_path),
        "archive_db": str(archive_db),
        "archive_root": str(archive_root) if archive_root else "",
        "archive_period": str(args.archive_period),
        "hot_days": int(args.hot_days),
        "batch_size": int(args.batch_size),
        "max_rows": int(max_rows),
        "moved_rows": int(total_moved),
        "remaining_rows": int(remaining),
        "archive_dbs_touched": sorted(archive_rows_by_db.keys()),
        "archive_rows_by_db": archive_rows_by_db,
        "cutoff_utc": cutoff,
        "archive_pruning": archive_pruning,
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(
            "sql_hot_retention_ok moved_rows={moved} remaining={rem} archives={archives} hot_days={days} archive_pruned={pruned} archive_files_deleted={deleted}".format(
                moved=out["moved_rows"],
                rem=out["remaining_rows"],
                archives=len(out["archive_dbs_touched"]),
                days=out["hot_days"],
                pruned=int((out.get("archive_pruning") or {}).get("pruned_rows", 0) or 0),
                deleted=len((out.get("archive_pruning") or {}).get("deleted_archive_files", []) or []),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
