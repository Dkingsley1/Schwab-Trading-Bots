import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply SQLite performance tuning and maintenance.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--vacuum", action="store_true")
    parser.add_argument("--auto-vacuum-over-gb", type=float, default=float(os.getenv("SQLITE_AUTO_VACUUM_OVER_GB", "24")))
    parser.add_argument("--vacuum-min-interval-hours", type=float, default=float(os.getenv("SQLITE_VACUUM_MIN_INTERVAL_HOURS", "24")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"SQLite DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-20000")
    conn.execute("PRAGMA mmap_size=268435456")

    created_indexes = 0
    if _table_exists(conn, "jsonl_records"):
        idx_sql = [
            "CREATE INDEX IF NOT EXISTS idx_jsonl_source_rel_ingested ON jsonl_records(source_rel, ingested_at)",
            "CREATE INDEX IF NOT EXISTS idx_jsonl_action_expr ON jsonl_records((json_extract(payload_json, '$.action')))",
            "CREATE INDEX IF NOT EXISTS idx_jsonl_symbol_expr ON jsonl_records((json_extract(payload_json, '$.symbol')))",
            "CREATE INDEX IF NOT EXISTS idx_jsonl_ts_expr ON jsonl_records((json_extract(payload_json, '$.timestamp_utc')))",
        ]
        for sql in idx_sql:
            conn.execute(sql)
            created_indexes += 1

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS db_maintenance_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL,
            db_path TEXT NOT NULL,
            vacuum_ran INTEGER NOT NULL,
            indexes_touched INTEGER NOT NULL,
            notes TEXT NOT NULL
        )
        """
    )

    conn.execute("ANALYZE")
    conn.execute("PRAGMA optimize")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    size_gb_before = db_path.stat().st_size / (1024 ** 3)

    last_vacuum_ts = None
    try:
        row = conn.execute(
            "SELECT timestamp_utc FROM db_maintenance_events WHERE vacuum_ran=1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row and row[0]:
            last_vacuum_ts = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        last_vacuum_ts = None

    do_vacuum = bool(args.vacuum)
    if (not do_vacuum) and size_gb_before >= float(args.auto_vacuum_over_gb):
        if last_vacuum_ts is None:
            do_vacuum = True
        else:
            elapsed_h = (datetime.now(timezone.utc) - last_vacuum_ts).total_seconds() / 3600.0
            do_vacuum = elapsed_h >= float(args.vacuum_min_interval_hours)

    if do_vacuum:
        conn.execute("VACUUM")

    total_rows = 0
    if _table_exists(conn, "jsonl_records"):
        row = conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()
        total_rows = int(row[0] if row else 0)

    size_gb_after = db_path.stat().st_size / (1024 ** 3)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "vacuum_ran": bool(do_vacuum),
        "indexes_touched": created_indexes,
        "jsonl_records_rows": total_rows,
        "size_gb_before": round(size_gb_before, 3),
        "size_gb_after": round(size_gb_after, 3),
        "auto_vacuum_over_gb": float(args.auto_vacuum_over_gb),
        "vacuum_min_interval_hours": float(args.vacuum_min_interval_hours),
    }

    conn.execute(
        "INSERT INTO db_maintenance_events(timestamp_utc, db_path, vacuum_ran, indexes_touched, notes) VALUES (?, ?, ?, ?, ?)",
        (payload["timestamp_utc"], payload["db_path"], 1 if do_vacuum else 0, created_indexes, "auto_maintenance"),
    )
    conn.commit()
    conn.close()

    out = PROJECT_ROOT / "governance" / "health" / "sqlite_maintenance_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"sqlite_maintenance_ok=1 vacuum_ran={payload['vacuum_ran']} indexes_touched={created_indexes} "
            f"jsonl_records_rows={total_rows} size_gb={payload['size_gb_after']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
