import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import sql_analytics_mirror as src


def test_sql_analytics_mirror_builds_materialized_summaries_and_heat(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    source_db = project_root / "data" / "jsonl_link.sqlite3"
    source_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source_db)) as conn:
        conn.execute(
            """
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
            """
        )
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
                json.dumps({"timestamp_utc": timestamp_utc, "symbol": "SPY", "action": "BUY"}),
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
