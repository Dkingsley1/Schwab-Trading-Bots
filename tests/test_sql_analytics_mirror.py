import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import sql_analytics_mirror as src


def test_sql_analytics_mirror_builds_materialized_summaries_and_heat(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    source_db = project_root / "data" / "jsonl_link.sqlite3"
    source_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source_db)) as conn:
        conn.execute("""
            CREATE TABLE jsonl_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                source_rel TEXT NOT NULL,
                line_no INTEGER NOT NULL,
                ingested_at TEXT NOT NULL,
                payload_sha1 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                run_id TEXT,
                iter_id TEXT,
                decision_id TEXT,
                parent_decision_id TEXT,
                log_schema_version INTEGER
            )
            """)
        conn.execute(
            """
            INSERT INTO jsonl_records(
                source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json,
                run_id, iter_id, decision_id, parent_decision_id, log_schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(project_root / "decisions" / "a.jsonl"),
                "decisions/a.jsonl",
                1,
                timestamp_utc,
                "sha1",
                json.dumps(
                    {"timestamp_utc": timestamp_utc, "symbol": "SPY", "action": "BUY"}
                ),
                "run-1",
                "iter-1",
                "d-1",
                "p-1",
                2,
            ),
        )
        conn.commit()

    payload = src.build_payload(
        project_root,
        source_db_path=source_db,
        duckdb_path=project_root / "data" / "analytics_mirror.duckdb",
        lookback_days=30,
    )

    assert payload["summary_refresh_ok"] is True
    assert payload["source_db_present"] is True
    assert payload["materialized_summaries"]["source_record_count"] == 1
    assert payload["materialized_summaries"]["stream_summary_rows"] >= 1
    assert payload["materialized_summaries"]["symbol_summary_rows"] >= 1
    ops_db = project_root / "governance" / "ops_data_plane.sqlite3"
    with sqlite3.connect(str(ops_db)) as conn:
        heat = conn.execute(
            "SELECT query_count, rows_scanned_total FROM shard_heat_state WHERE shard_name='primary_sqlite'"
        ).fetchone()

    assert heat is not None
    assert int(heat[0]) >= 1
    assert int(heat[1]) >= 1


def _seed_summaries(path: Path, generation: str, *, invalid_symbol_count=False):
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS materialized_stream_daily (day_utc TEXT, stream TEXT, record_count INTEGER, distinct_sources INTEGER, min_schema_version INTEGER, max_schema_version INTEGER, last_ingested_at TEXT, refreshed_utc TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS materialized_symbol_daily (day_utc TEXT, symbol TEXT, record_count INTEGER, buy_count INTEGER, sell_count INTEGER, hold_count INTEGER, last_ingested_at TEXT, refreshed_utc TEXT)"
        )
        conn.execute("DELETE FROM materialized_stream_daily")
        conn.execute("DELETE FROM materialized_symbol_daily")
        conn.execute(
            "INSERT INTO materialized_stream_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("2026-09-14", generation, 1, 1, 1, 1, generation, generation),
        )
        conn.execute(
            "INSERT INTO materialized_symbol_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "2026-09-14",
                generation,
                "invalid" if invalid_symbol_count else 1,
                1,
                0,
                0,
                generation,
                generation,
            ),
        )


def _generations(conn):
    return (
        conn.execute("SELECT stream FROM materialized_stream_daily").fetchall(),
        conn.execute("SELECT symbol FROM materialized_symbol_daily").fetchall(),
    )


def test_failed_second_table_load_preserves_entire_previous_mirror(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    source = tmp_path / "source.sqlite3"
    mirror = tmp_path / "mirror.duckdb"
    _seed_summaries(source, "old")
    assert src._write_duckdb_mirror(ops_db_path=source, duckdb_path=mirror)[
        "mirror_ready"
    ]
    _seed_summaries(source, "new", invalid_symbol_count=True)
    failed = src._write_duckdb_mirror(ops_db_path=source, duckdb_path=mirror)
    assert failed["mirror_ready"] is False
    assert "duckdb_sync_failed" in failed["error"]
    with duckdb.connect(str(mirror)) as conn:
        assert _generations(conn) == ([("old",)], [("old",)])
    _seed_summaries(source, "new")
    assert src._write_duckdb_mirror(ops_db_path=source, duckdb_path=mirror)[
        "mirror_ready"
    ]
    with duckdb.connect(str(mirror)) as conn:
        assert _generations(conn) == ([("new",)], [("new",)])


def test_readers_cannot_see_half_published_mirror(tmp_path, monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    source = tmp_path / "source.sqlite3"
    mirror = tmp_path / "mirror.duckdb"
    _seed_summaries(source, "old")
    assert src._write_duckdb_mirror(ops_db_path=source, duckdb_path=mirror)[
        "mirror_ready"
    ]
    _seed_summaries(source, "new")
    real_connect = duckdb.connect
    observed = []

    class ObservedConnection:
        def __init__(self, *args, **kwargs):
            self.conn = real_connect(*args, **kwargs)

        def execute(self, *args, **kwargs):
            return self.conn.execute(*args, **kwargs)

        def executemany(self, query, rows):
            if "materialized_symbol_daily" in query:
                with real_connect(str(mirror)) as reader:
                    observed.append(_generations(reader))
            return self.conn.executemany(query, rows)

        def close(self):
            self.conn.close()

    monkeypatch.setattr(duckdb, "connect", ObservedConnection)
    assert src._write_duckdb_mirror(ops_db_path=source, duckdb_path=mirror)[
        "mirror_ready"
    ]
    assert observed == [([("old",)], [("old",)])]
    with real_connect(str(mirror)) as conn:
        assert _generations(conn) == ([("new",)], [("new",)])


def test_source_refresh_between_reads_cannot_mix_generations(tmp_path, monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    source = tmp_path / "source.sqlite3"
    mirror = tmp_path / "mirror.duckdb"
    _seed_summaries(source, "old")
    real_connect = src.connect_sqlite
    closed = []

    class ObservedSource:
        def __init__(self, *args, **kwargs):
            self.conn = real_connect(*args, **kwargs)

        def execute(self, query):
            if "FROM materialized_symbol_daily" in query:
                _seed_summaries(source, "new")
            return self.conn.execute(query)

        def close(self):
            self.conn.close()
            closed.append(True)

    monkeypatch.setattr(src, "connect_sqlite", ObservedSource)
    assert src._write_duckdb_mirror(ops_db_path=source, duckdb_path=mirror)[
        "mirror_ready"
    ]
    assert closed == [True]
    with duckdb.connect(str(mirror)) as conn:
        assert _generations(conn) == ([("old",)], [("old",)])


def test_failed_first_publication_rolls_back_schema_and_allows_retry(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    source = tmp_path / "source.sqlite3"
    mirror = tmp_path / "mirror.duckdb"
    _seed_summaries(source, "new", invalid_symbol_count=True)
    assert not src._write_duckdb_mirror(ops_db_path=source, duckdb_path=mirror)[
        "mirror_ready"
    ]
    with duckdb.connect(str(mirror)) as conn:
        assert conn.execute("SHOW TABLES").fetchall() == []
    _seed_summaries(source, "new")
    assert src._write_duckdb_mirror(ops_db_path=source, duckdb_path=mirror)[
        "mirror_ready"
    ]
