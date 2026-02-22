import argparse
import json
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
        CREATE TABLE IF NOT EXISTS sqlite_maintenance_events (
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
    if args.vacuum:
        conn.execute("VACUUM")

    total_rows = 0
    if _table_exists(conn, "jsonl_records"):
        row = conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()
        total_rows = int(row[0] if row else 0)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "vacuum_ran": bool(args.vacuum),
        "indexes_touched": created_indexes,
        "jsonl_records_rows": total_rows,
    }

    conn.execute(
        "INSERT INTO sqlite_maintenance_events(timestamp_utc, db_path, vacuum_ran, indexes_touched, notes) VALUES (?, ?, ?, ?, ?)",
        (payload["timestamp_utc"], payload["db_path"], 1 if args.vacuum else 0, created_indexes, "auto_maintenance"),
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
            f"jsonl_records_rows={total_rows}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
