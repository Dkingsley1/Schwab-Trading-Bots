import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return bool(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep SQLite hot window small and archive older rows.")
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "jsonl_link.sqlite3"))
    parser.add_argument("--archive-db", default=str(PROJECT_ROOT / "data" / "jsonl_link_archive.sqlite3"))
    parser.add_argument("--hot-days", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--vacuum", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"sql_hot_retention_skip db_missing={db_path}")
        return 0

    archive_path = Path(args.archive_db)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    cutoff = (_now_utc() - timedelta(days=max(args.hot_days, 1))).isoformat()

    src = sqlite3.connect(str(db_path))
    src.row_factory = sqlite3.Row
    dst = sqlite3.connect(str(archive_path))

    if not _table_exists(src, "jsonl_records"):
        print("sql_hot_retention_skip table_missing=jsonl_records")
        src.close()
        dst.close()
        return 0

    src_create = src.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='jsonl_records'"
    ).fetchone()
    if src_create and src_create[0]:
        dst.execute(src_create[0])

    cols = [r[1] for r in src.execute("PRAGMA table_info(jsonl_records)").fetchall()]
    col_list = ",".join(cols)
    qmarks = ",".join(["?"] * len(cols))

    total_moved = 0
    while True:
        rows = src.execute(
            """
            SELECT *
            FROM jsonl_records
            WHERE ingested_at < ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (cutoff, max(args.batch_size, 1000)),
        ).fetchall()
        if not rows:
            break

        payload = [tuple(r[c] for c in cols) for r in rows]
        ids = [int(r["id"]) for r in rows]

        dst.executemany(
            f"INSERT OR IGNORE INTO jsonl_records ({col_list}) VALUES ({qmarks})",
            payload,
        )

        id_marks = ",".join("?" for _ in ids)
        src.execute(f"DELETE FROM jsonl_records WHERE id IN ({id_marks})", ids)

        src.commit()
        dst.commit()
        total_moved += len(rows)

    if args.vacuum and total_moved > 0:
        src.execute("VACUUM")

    remaining = src.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()[0]
    archived = dst.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()[0]

    src.close()
    dst.close()

    out = {
        "timestamp_utc": _now_utc().isoformat(),
        "db": str(db_path),
        "archive_db": str(archive_path),
        "hot_days": int(args.hot_days),
        "moved_rows": int(total_moved),
        "remaining_rows": int(remaining),
        "archive_rows": int(archived),
        "cutoff_utc": cutoff,
    }

    if args.json:
        import json

        print(json.dumps(out, ensure_ascii=True))
    else:
        print(
            "sql_hot_retention_ok moved_rows={moved} remaining={rem} archived={arc} hot_days={days}".format(
                moved=out["moved_rows"], rem=out["remaining_rows"], arc=out["archive_rows"], days=out["hot_days"]
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
