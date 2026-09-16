import sqlite3
from datetime import datetime, timedelta, timezone

from scripts import sql_queue_retention as src


def test_queue_retention_uses_shorter_window_under_capacity_pressure() -> None:
    hours, mode = src._effective_acked_retention_hours(
        db_size_gb=12.0,
        acked_days=7,
        acked_hours=-1.0,
        pressure_db_gb=8.0,
        pressure_acked_hours=6.0,
    )

    assert hours == 6.0
    assert mode == "capacity_pressure"


def test_queue_retention_explicit_hours_take_precedence() -> None:
    hours, mode = src._effective_acked_retention_hours(
        db_size_gb=12.0,
        acked_days=7,
        acked_hours=0.25,
        pressure_db_gb=8.0,
        pressure_acked_hours=6.0,
    )

    assert hours == 0.25
    assert mode == "explicit_hours"


def test_full_vacuum_enables_future_incremental_reclamation(tmp_path) -> None:
    db_path = tmp_path / "queue.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE rows(value TEXT)")
        conn.executemany("INSERT INTO rows(value) VALUES (?)", [("x" * 1000,)] * 1000)
        conn.commit()
        assert int(conn.execute("PRAGMA auto_vacuum").fetchone()[0]) == 0

        src._full_vacuum_with_incremental_mode(conn)

        assert int(conn.execute("PRAGMA auto_vacuum").fetchone()[0]) == 2
    finally:
        conn.close()


def test_processing_claim_retention_preserves_unresolved_and_live_message_claims(
    tmp_path,
) -> None:
    db_path = tmp_path / "queue.sqlite3"
    conn = sqlite3.connect(db_path)
    old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    try:
        conn.execute(
            "CREATE TABLE channel_messages (id INTEGER PRIMARY KEY, channel TEXT, message_id TEXT)"
        )
        conn.execute(
            """
            CREATE TABLE channel_processing_claims (
                consumer TEXT, channel TEXT, message_id TEXT, state TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO channel_messages VALUES (1, 'execution_intent', 'still-live')"
        )
        conn.executemany(
            "INSERT INTO channel_processing_claims VALUES (?, ?, ?, ?, ?)",
            [
                ("paper", "execution_intent", "terminal-old", "completed", old),
                ("paper", "execution_intent", "ambiguous-old", "outcome_ambiguous", old),
                ("paper", "execution_intent", "processing-old", "processing", old),
                ("paper", "execution_intent", "still-live", "completed", old),
            ],
        )
        conn.commit()

        deleted = src._cleanup_processing_claims(
            conn,
            cutoff=datetime.now(timezone.utc).isoformat(),
            limit=100,
            dry_run=False,
        )
        conn.commit()

        remaining = {
            str(row[0])
            for row in conn.execute(
                "SELECT message_id FROM channel_processing_claims"
            ).fetchall()
        }
        assert deleted == 1
        assert remaining == {"ambiguous-old", "processing-old", "still-live"}
    finally:
        conn.close()
