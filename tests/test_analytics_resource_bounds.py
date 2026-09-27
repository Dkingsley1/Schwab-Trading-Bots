import fcntl
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import ops_data_plane as plane
from scripts.ops import sql_analytics_mirror as mirror


def source_db(root, count=3):
    path = root / "data/source.sqlite3"
    path.parent.mkdir(parents=True)
    today = datetime.now(timezone.utc).date().isoformat()
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE jsonl_records (source_rel TEXT, log_schema_version INTEGER, ingested_at TEXT, payload_json TEXT)"
        )
        conn.executemany(
            "INSERT INTO jsonl_records VALUES (?, ?, ?, ?)",
            [
                (
                    "decisions/test.jsonl",
                    2,
                    today + "T12:00:00Z",
                    json.dumps({"symbol": "O", "action": "BUY", "large": "x" * 100000}),
                )
                for _ in range(count)
            ],
        )
    return path


def seed_old(conn):
    day = datetime.now(timezone.utc).date().isoformat()
    conn.execute(
        "INSERT INTO materialized_stream_daily VALUES (?, 'old', 9, 1, 1, 1, 'old', 'old')",
        (day,),
    )
    conn.execute(
        "INSERT INTO materialized_symbol_daily VALUES (?, 'OLD', 9, 9, 0, 0, 'old', 'old')",
        (day,),
    )
    conn.commit()


def assert_old(conn):
    assert conn.execute(
        "SELECT stream, record_count FROM materialized_stream_daily"
    ).fetchall() == [("old", 9)]
    assert conn.execute(
        "SELECT symbol, record_count FROM materialized_symbol_daily"
    ).fetchall() == [("OLD", 9)]


def test_streams_compact_fields_without_raw_payload_sorters(tmp_path):
    source = source_db(tmp_path)
    with plane.connect(tmp_path) as conn:
        queries = []
        conn.set_trace_callback(queries.append)
        result = plane.emit_materialized_summaries(conn, source_db_path=source)
        assert result["source_record_count"] == 3
        assert result["projected_bytes"] < 1000
        assert result["bounded_streaming"]
        assert not any("GROUP BY" in q or "ORDER BY" in q for q in queries)
        assert conn.execute(
            "SELECT record_count, distinct_sources FROM materialized_stream_daily"
        ).fetchall() == [(3, 1)]
        assert conn.execute(
            "SELECT buy_count FROM materialized_symbol_daily"
        ).fetchone() == (3,)


@pytest.mark.parametrize("limits", [{"max_rows": 2}, {"max_projected_bytes": 1}])
def test_partial_scans_preserve_both_prior_summaries(tmp_path, limits):
    source = source_db(tmp_path)
    with plane.connect(tmp_path) as conn:
        seed_old(conn)
        with pytest.raises(RuntimeError, match="budget_exceeded"):
            plane.emit_materialized_summaries(conn, source_db_path=source, **limits)
        assert_old(conn)
        assert not any(
            row[1] == "sourcedb" for row in conn.execute("PRAGMA database_list")
        )


def test_second_summary_failure_rolls_back_first_summary(tmp_path):
    source = source_db(tmp_path)
    with plane.connect(tmp_path) as conn:
        seed_old(conn)
        conn.execute(
            "CREATE TRIGGER fail_second BEFORE INSERT ON materialized_symbol_daily BEGIN SELECT RAISE(ABORT, 'test_failure'); END"
        )
        with pytest.raises(sqlite3.DatabaseError, match="test_failure"):
            plane.emit_materialized_summaries(conn, source_db_path=source)
        assert_old(conn)


def test_deadline_rolls_back_and_releases_source(tmp_path, monkeypatch):
    source = source_db(tmp_path)
    with plane.connect(tmp_path) as conn:
        seed_old(conn)
        clock = iter([0, 0, 0, 30, 30, 30])
        monkeypatch.setattr(plane.time, "monotonic", lambda: next(clock, 30))
        with pytest.raises(RuntimeError, match="deadline_exceeded"):
            plane.emit_materialized_summaries(conn, source_db_path=source)
        assert_old(conn)
        assert not conn.in_transaction


def test_storage_pressure_rejects_before_opening_databases(tmp_path, monkeypatch):
    source = source_db(tmp_path)
    monkeypatch.setattr(
        mirror.shutil, "disk_usage", lambda path: SimpleNamespace(free=64 * 1024**3)
    )
    monkeypatch.setattr(
        mirror.ops_data_plane,
        "connect",
        lambda *a, **k: pytest.fail("database opened under pressure"),
    )
    result = mirror.build_payload(
        tmp_path, source_db_path=source, duckdb_path=tmp_path / "mirror.duckdb"
    )
    assert not result["ok"]
    assert result["summary_refresh_error"] == "analytics_storage_headroom_required"


def test_headroom_loss_mid_scan_preserves_prior_summaries(tmp_path, monkeypatch):
    source = source_db(tmp_path)
    with plane.connect(tmp_path) as conn:
        seed_old(conn)
        clock = iter(range(100))
        monkeypatch.setattr(plane.time, "monotonic", lambda: next(clock))
        calls = []

        def admission():
            calls.append(True)
            if len(calls) >= 2:
                raise RuntimeError("analytics_storage_headroom_required")

        with pytest.raises(RuntimeError, match="headroom_required"):
            plane.emit_materialized_summaries(
                conn, source_db_path=source, admission_check=admission
            )
        assert_old(conn)


def test_busy_mirror_does_not_open_source(tmp_path, monkeypatch):
    source = source_db(tmp_path)
    monkeypatch.setattr(mirror, "_check_headroom", lambda root: None)
    lock = tmp_path / "governance/locks/sql_analytics_mirror.lock"
    lock.parent.mkdir(parents=True)
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = mirror.build_payload(
            tmp_path, source_db_path=source, duckdb_path=tmp_path / "mirror.duckdb"
        )
    assert result["busy"]
    assert not (tmp_path / "governance/ops_data_plane.sqlite3").exists()


def test_protected_source_rejected_before_database_or_disk_access(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        mirror,
        "_check_headroom",
        lambda root: pytest.fail("unsafe route reached admission"),
    )
    result = mirror.build_payload(
        tmp_path,
        source_db_path=Path("/Volumes/VIDEO/forbidden.sqlite3"),
        duckdb_path=tmp_path / "mirror.duckdb",
    )
    assert "route_rejected" in result["summary_refresh_error"]


def test_cli_timeout_does_not_use_stale_success(tmp_path, monkeypatch, capsys):
    receipt = tmp_path / "report.json"
    receipt.write_text('{"ok": true}')
    seen = []

    def timed_out(cmd, **kwargs):
        seen.append(kwargs)
        return {
            "stdout": '{"ok": true}',
            "timed_out": True,
            "timeout_cleanup": {"reaped": True},
        }

    monkeypatch.setattr(mirror, "run_bounded_process_group", timed_out)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mirror",
            "--project-root",
            str(tmp_path),
            "--out-file",
            str(receipt),
            "--json",
        ],
    )
    assert mirror.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"]
    assert result["summary_refresh_error"] == "analytics_worker_timeout"
    assert result["timeout_cleanup"]["reaped"]
    assert seen[0]["timeout_seconds"] == 35
    assert not json.loads(receipt.read_text())["ok"]


@pytest.mark.parametrize("mounted", [True, False])
def test_configured_external_cache_requires_mounted_volume(monkeypatch, mounted):
    path = Path("/Volumes/BOT_LOGS/schwab_trading_bot/data/analytics_mirror.duckdb")
    monkeypatch.setattr(
        mirror,
        "inspect_storage_path",
        lambda p: {"status": "present", "resolved_path": str(path)},
    )
    monkeypatch.setattr(Path, "is_mount", lambda p: mounted)
    if mounted:
        assert mirror._data_route(path) == path
    else:
        with pytest.raises(ValueError, match="external_volume_unavailable"):
            mirror._data_route(path)


def test_missing_source_never_creates_empty_database(tmp_path, monkeypatch):
    monkeypatch.setattr(mirror, "_check_headroom", lambda root: None)
    source = tmp_path / "missing.sqlite3"
    result = mirror.build_payload(
        tmp_path, source_db_path=source, duckdb_path=tmp_path / "mirror.duckdb"
    )
    assert result["summary_refresh_error"] == "analytics_source_missing"
    assert not source.exists()


def test_source_read_failure_prevents_duckdb_publication(tmp_path, monkeypatch):
    source = source_db(tmp_path)
    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE jsonl_records SET payload_json='invalid json'")
    monkeypatch.setattr(mirror, "_check_headroom", lambda root: None)
    monkeypatch.setattr(
        mirror,
        "_write_duckdb_mirror",
        lambda **kwargs: pytest.fail("partial summaries were mirrored"),
    )
    result = mirror.build_payload(
        tmp_path, source_db_path=source, duckdb_path=tmp_path / "mirror.duckdb"
    )
    assert not result["summary_refresh_ok"]
    assert not result["ok"]
