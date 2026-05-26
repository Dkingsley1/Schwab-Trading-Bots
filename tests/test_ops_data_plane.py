import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


from scripts import ops_data_plane as src


def test_ops_data_plane_records_and_reads_core_control_tables(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    with src.connect(project_root) as conn:
        src.record_watermark(
            conn,
            collector_key="demo_collector",
            source_name="demo_source",
            entity_key="EURUSD",
            watermark_type="cursor",
            watermark_value="2026-04-16T12:00:00+00:00",
            metadata={"rows": 12},
        )
        src.record_collector_run(
            conn,
            collector_key="demo_collector",
            cache_key="demo_collector",
            command=["python", "demo.py"],
            command_fingerprint="fp",
            skipped=False,
            rc=0,
            started_utc="2026-04-16T12:00:00+00:00",
            finished_utc="2026-04-16T12:00:05+00:00",
            metadata={"ok": True},
        )
        src.record_dead_letter(
            conn,
            lane="sqlite",
            source_rel="decisions/demo.jsonl",
            line_no=9,
            offset_bytes=123,
            error_class="JSONDecodeError",
            error_message="bad",
            raw_payload="{bad json}",
        )
        src.record_schema_drift(
            conn,
            lane="sqlite",
            source_rel="decisions/demo.jsonl",
            line_no=10,
            observed_schema_version=1,
            expected_schema_version=2,
            drift_kind="schema_version_mismatch",
            payload_json='{"symbol":"SPY"}',
        )
        heat_score = src.record_query_access(
            conn,
            query_family="summary_refresh",
            shard_name="primary_sqlite",
            consumer="test",
            query_text="select * from jsonl_records",
            rows_scanned=15000,
            rows_returned=250,
            duration_ms=1800.0,
        )
        src.record_storage_route_event(
            conn,
            project_root=project_root,
            mode="external",
            active_root=project_root / "external",
            switched_links=["logs"],
            passthrough_paths=[],
        )
        src.record_canonical_reconciliation(
            conn,
            domain="fx_pair",
            entity_key="EURUSD",
            canonical_source="twelve_data",
            confidence=0.91,
            divergence_score=0.002,
            canonical_payload={"pair": "EURUSD", "canonical_value": 1.082},
            provider_votes={"twelve_data": {"value": 1.082}},
        )

    assert heat_score > 0.0
    latest = src.latest_collector_run(project_root, collector_key="demo_collector")
    budget = src.collector_error_budget(project_root, collector_key="demo_collector")
    heat_map = src.load_shard_heat_map(project_root)

    assert latest["rc"] == 0
    assert budget["run_count"] == 1
    assert heat_map["primary_sqlite"]["promotion_candidate"] is True


def test_record_query_access_accumulates_atomically_with_batched_commits(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    with src.connect(project_root) as conn:
        src.record_query_access(
            conn,
            query_family="summary_refresh",
            shard_name="primary_sqlite",
            consumer="test",
            query_text="select 1",
            rows_scanned=10,
            rows_returned=1,
            duration_ms=5.0,
            commit=False,
        )
        src.record_query_access(
            conn,
            query_family="summary_refresh",
            shard_name="primary_sqlite",
            consumer="test",
            query_text="select 2",
            rows_scanned=25,
            rows_returned=2,
            duration_ms=7.5,
            commit=False,
        )
        conn.commit()

    heat_map = src.load_shard_heat_map(project_root)
    assert heat_map["primary_sqlite"]["query_count"] == 2
    assert heat_map["primary_sqlite"]["rows_scanned_total"] == 35
    assert heat_map["primary_sqlite"]["rows_returned_total"] == 3


def test_load_shard_heat_map_fails_open_for_corrupt_ops_plane(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    ops_db = project_root / "governance" / "ops_data_plane.sqlite3"
    ops_db.parent.mkdir(parents=True, exist_ok=True)
    ops_db.write_bytes(b"not a sqlite database")

    assert src.load_shard_heat_map(project_root) == {}


def test_connect_quarantines_corrupt_ops_plane_and_recreates_schema(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    ops_db = project_root / "governance" / "ops_data_plane.sqlite3"
    ops_db.parent.mkdir(parents=True, exist_ok=True)
    ops_db.write_bytes(b"not a sqlite database")

    with src.connect(project_root) as conn:
        src.record_dead_letter(
            conn,
            lane="sqlite",
            source_rel="decisions/demo.jsonl",
            line_no=1,
            offset_bytes=10,
            error_class="OversizePayload",
            error_message="payload too large",
            raw_payload="{}",
        )
        count = conn.execute("SELECT COUNT(*) FROM ingest_dead_letters").fetchone()[0]

    quarantined = list(ops_db.parent.glob("ops_data_plane.sqlite3.corrupt_*"))
    assert quarantined
    assert int(count) == 1


def test_connect_quarantines_corrupt_symlink_target_without_replacing_link(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    ops_db = project_root / "governance" / "ops_data_plane.sqlite3"
    external = tmp_path / "external" / "ops_data_plane.sqlite3"
    external.parent.mkdir(parents=True, exist_ok=True)
    ops_db.parent.mkdir(parents=True, exist_ok=True)
    external.write_bytes(b"not a sqlite database")
    ops_db.symlink_to(external)

    with src.connect(project_root) as conn:
        count = conn.execute("SELECT COUNT(*) FROM ingest_dead_letters").fetchone()[0]

    quarantined = list(external.parent.glob("ops_data_plane.sqlite3.corrupt_*"))
    assert ops_db.is_symlink()
    assert quarantined
    assert int(count) == 0


def test_normalize_entity_key_relativizes_project_paths(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    artifact = project_root / "governance" / "health" / "collector.json"
    normalized = src.normalize_entity_key(project_root, artifact)
    namespaced = src.normalize_entity_key(project_root, "fx_market_context", namespace="source")

    assert normalized == "governance/health/collector.json"
    assert namespaced == "source/fx_market_context"


def test_resolve_sqlite_runtime_settings_downshift_ops_plane_under_pressure(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health_root = project_root / "governance" / "health"
    health_root.mkdir(parents=True, exist_ok=True)
    (health_root / "resource_guard_latest.json").write_text(
        json.dumps(
            {
                "memory_pressure_state": "red",
                "memory_pressure_kind": "throttled",
                "memory_free_pct": 8.0,
                "swap_used_gb": 21.5,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    settings = src.resolve_sqlite_runtime_settings(project_root)

    assert settings["pressure_level"] == "red"
    assert settings["temp_store_mode"] == "FILE"
    assert settings["cache_size_kb"] == 2048
    assert settings["mmap_size_mb"] == 8


def test_emit_materialized_summaries_rolls_up_stream_and_symbol_daily(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source_db = project_root / "data" / "jsonl_link.sqlite3"
    source_db.parent.mkdir(parents=True, exist_ok=True)
    day_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
        conn.executemany(
            """
            INSERT INTO jsonl_records(
                source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json,
                run_id, iter_id, decision_id, parent_decision_id, log_schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                    (
                        str(project_root / "decisions" / "a.jsonl"),
                        "decisions/a.jsonl",
                        1,
                        f"{day_utc}T12:00:00+00:00",
                        "sha1",
                        json.dumps({"timestamp_utc": f"{day_utc}T12:00:00+00:00", "symbol": "SPY", "action": "BUY"}),
                        "run-1",
                        "iter-1",
                    "d-1",
                    "p-1",
                    2,
                ),
                    (
                        str(project_root / "decisions" / "a.jsonl"),
                        "decisions/a.jsonl",
                        2,
                        f"{day_utc}T12:10:00+00:00",
                        "sha2",
                        json.dumps({"timestamp_utc": f"{day_utc}T12:10:00+00:00", "symbol": "SPY", "action": "SELL"}),
                        "run-1",
                        "iter-1",
                    "d-2",
                    "p-1",
                    2,
                ),
                    (
                        str(project_root / "governance" / "events" / "g.jsonl"),
                        "governance/events/g.jsonl",
                        1,
                        f"{day_utc}T12:20:00+00:00",
                        "sha3",
                        json.dumps({"timestamp_utc": f"{day_utc}T12:20:00+00:00", "symbol": "QQQ", "action": "HOLD"}),
                        "run-2",
                    "iter-2",
                    "d-3",
                    "p-2",
                    2,
                ),
            ],
        )
        conn.commit()

    with src.connect(project_root) as conn:
        payload = src.emit_materialized_summaries(conn, source_db_path=source_db, lookback_days=30)
        stream_rows = conn.execute(
            "SELECT stream, record_count FROM materialized_stream_daily ORDER BY stream ASC"
        ).fetchall()
        symbol_rows = conn.execute(
            "SELECT symbol, buy_count, sell_count, hold_count FROM materialized_symbol_daily ORDER BY symbol ASC"
        ).fetchall()

    assert payload["source_record_count"] == 3
    assert payload["stream_summary_rows"] >= 2
    assert ("decisions", 2) in stream_rows
    assert ("governance_events", 1) in stream_rows
    assert ("QQQ", 0, 0, 1) in symbol_rows
    assert ("SPY", 1, 1, 0) in symbol_rows
