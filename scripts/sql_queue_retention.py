import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _db_size_gb(path: Path) -> float:
    try:
        return float(path.stat().st_size) / (1024.0 ** 3)
    except Exception:
        return 0.0


def _delete_ids(conn: sqlite3.Connection, table: str, ids: list[int]) -> int:
    if not ids:
        return 0
    marks = ",".join("?" for _ in ids)
    conn.execute(f"DELETE FROM {table} WHERE id IN ({marks})", ids)
    return len(ids)


def _consumer_state_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM channel_consumer_state").fetchone()
    return int(row[0] if row and row[0] is not None else 0)


def _fetch_acked_batch(conn: sqlite3.Connection, *, cutoff: str, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        WITH ack_bounds AS (
            SELECT channel, MIN(last_id) AS acked_through
            FROM channel_consumer_state
            GROUP BY channel
            HAVING MIN(last_id) > 0
        )
        SELECT m.id, m.channel
        FROM channel_messages m
        JOIN ack_bounds a ON a.channel = m.channel
        WHERE m.id <= a.acked_through
          AND m.created_at < ?
        ORDER BY m.id ASC
        LIMIT ?
        """,
        (cutoff, limit),
    ).fetchall()


def _fetch_orphan_batch(conn: sqlite3.Connection, *, cutoff: str, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        WITH orphan_channels AS (
            SELECT m.channel
            FROM channel_messages m
            LEFT JOIN channel_consumer_state c ON c.channel = m.channel
            GROUP BY m.channel
            HAVING COUNT(c.consumer) = 0
               AND MAX(m.created_at) < ?
        )
        SELECT id, channel
        FROM channel_messages
        WHERE channel IN (SELECT channel FROM orphan_channels)
          AND created_at < ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (cutoff, cutoff, limit),
    ).fetchall()


def _fetch_unconsumed_batch(conn: sqlite3.Connection, *, cutoff: str, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, channel
        FROM channel_messages
        WHERE created_at < ?
        ORDER BY created_at ASC, id ASC
        LIMIT ?
        """,
        (cutoff, limit),
    ).fetchall()


def _cleanup_consumer_state(conn: sqlite3.Connection, *, cutoff: str, limit: int, dry_run: bool) -> int:
    rows = conn.execute(
        """
        SELECT cs.rowid
        FROM channel_consumer_state cs
        LEFT JOIN channel_messages m ON m.channel = cs.channel
        WHERE m.channel IS NULL
          AND cs.updated_at < ?
        ORDER BY cs.updated_at ASC
        LIMIT ?
        """,
        (cutoff, limit),
    ).fetchall()
    if dry_run or not rows:
        return len(rows)

    rowids = [int(row[0]) for row in rows]
    marks = ",".join("?" for _ in rowids)
    conn.execute(f"DELETE FROM channel_consumer_state WHERE rowid IN ({marks})", rowids)
    return len(rowids)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune safely acknowledged rows from bot_channel_queue SQLite.")
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "bot_channel_queue.sqlite3"))
    parser.add_argument("--acked-days", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--max-rows", type=int, default=0, help="Maximum rows to delete in one pass (0 = unlimited).")
    parser.add_argument("--cleanup-consumer-state-days", type=int, default=30)
    parser.add_argument("--prune-orphans", action="store_true", help="Also prune orphan channels with no consumer state.")
    parser.add_argument("--orphan-days", type=int, default=45)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vacuum", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"sql_queue_retention_skip db_missing={db_path}")
        return 0

    db_size_before = round(_db_size_gb(db_path), 3)

    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn, "channel_messages") or not _table_exists(conn, "channel_consumer_state"):
            print("sql_queue_retention_skip table_missing=channel_messages_or_channel_consumer_state")
            return 0

        conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_consumer_state_channel ON channel_consumer_state(channel, last_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_consumer_state_updated_at ON channel_consumer_state(updated_at)")

        consumer_state_rows = _consumer_state_count(conn)
        acked_cutoff = (_now_utc() - timedelta(days=max(int(args.acked_days), 1))).isoformat()
        consumer_state_cutoff = (
            _now_utc() - timedelta(days=max(int(args.cleanup_consumer_state_days), max(int(args.acked_days), 1)))
        ).isoformat()
        orphan_cutoff = (_now_utc() - timedelta(days=max(int(args.orphan_days), 1))).isoformat()
        batch_size = max(int(args.batch_size), 1000)
        max_rows = max(int(args.max_rows), 0)

        deleted_acked_rows = 0
        deleted_orphan_rows = 0
        deleted_rows_total = 0
        deleted_consumer_state_rows = 0
        channels_touched: set[str] = set()

        while True:
            limit = batch_size
            if max_rows > 0:
                remaining_budget = max_rows - deleted_rows_total
                if remaining_budget <= 0:
                    break
                limit = min(limit, remaining_budget)

            rows = _fetch_acked_batch(conn, cutoff=acked_cutoff, limit=limit)
            if not rows:
                break

            ids = [int(row["id"]) for row in rows]
            channels_touched.update(str(row["channel"] or "") for row in rows)
            if not args.dry_run:
                _delete_ids(conn, "channel_messages", ids)
                conn.commit()
            deleted_acked_rows += len(ids)
            deleted_rows_total += len(ids)

            if len(rows) < limit:
                break

        if args.prune_orphans:
            while True:
                limit = batch_size
                if max_rows > 0:
                    remaining_budget = max_rows - deleted_rows_total
                    if remaining_budget <= 0:
                        break
                    limit = min(limit, remaining_budget)

                if consumer_state_rows == 0:
                    rows = _fetch_unconsumed_batch(conn, cutoff=orphan_cutoff, limit=limit)
                else:
                    rows = _fetch_orphan_batch(conn, cutoff=orphan_cutoff, limit=limit)
                if not rows:
                    break

                ids = [int(row["id"]) for row in rows]
                channels_touched.update(str(row["channel"] or "") for row in rows)
                if not args.dry_run:
                    _delete_ids(conn, "channel_messages", ids)
                    conn.commit()
                deleted_orphan_rows += len(ids)
                deleted_rows_total += len(ids)

                if len(rows) < limit:
                    break

        cleanup_limit = batch_size if max_rows == 0 else min(batch_size, max(max_rows, 1))
        deleted_consumer_state_rows = _cleanup_consumer_state(
            conn,
            cutoff=consumer_state_cutoff,
            limit=cleanup_limit,
            dry_run=bool(args.dry_run),
        )
        if deleted_consumer_state_rows > 0 and not args.dry_run:
            conn.commit()

        if args.vacuum and deleted_rows_total > 0 and not args.dry_run:
            conn.execute("VACUUM")
    finally:
        conn.close()

    out = {
        "timestamp_utc": _now_utc().isoformat(),
        "db": str(db_path),
        "db_size_gb_before": db_size_before,
        "db_size_gb_after": round(_db_size_gb(db_path), 3),
        "consumer_state_rows": int(consumer_state_rows),
        "acked_days": int(args.acked_days),
        "acked_cutoff_utc": acked_cutoff,
        "cleanup_consumer_state_days": int(args.cleanup_consumer_state_days),
        "consumer_state_cutoff_utc": consumer_state_cutoff,
        "prune_orphans": bool(args.prune_orphans),
        "orphan_days": int(args.orphan_days),
        "orphan_cutoff_utc": orphan_cutoff,
        "batch_size": int(batch_size),
        "max_rows": int(max_rows),
        "dry_run": bool(args.dry_run),
        "vacuum": bool(args.vacuum and not args.dry_run and deleted_rows_total > 0),
        "deleted_acked_rows": int(deleted_acked_rows),
        "deleted_orphan_rows": int(deleted_orphan_rows),
        "deleted_consumer_state_rows": int(deleted_consumer_state_rows),
        "deleted_rows_total": int(deleted_rows_total),
        "channels_touched": sorted(ch for ch in channels_touched if ch),
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(
            "sql_queue_retention_ok deleted_rows={deleted} deleted_consumer_states={states} channels={channels}".format(
                deleted=out["deleted_rows_total"],
                states=out["deleted_consumer_state_rows"],
                channels=len(out["channels_touched"]),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
