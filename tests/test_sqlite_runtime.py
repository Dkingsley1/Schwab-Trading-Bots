import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core import sqlite_runtime as src


@pytest.fixture(autouse=True)
def clean_sqlite_environment(monkeypatch):
    import os

    for key in os.environ:
        if key.startswith(("BOT_OPS_SQLITE_", "SQLITE_")):
            monkeypatch.delenv(key)


def resource(root, **overrides):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "memory_pressure_state": "green",
        "memory_pressure_kind": "normal",
        "swap_used_gb": 0,
        "memory_free_pct": 60,
        "input_evidence_ready": True,
        "measurement_refreshed": True,
        **overrides,
    }
    health = root / "governance/health"
    health.mkdir(parents=True, exist_ok=True)
    (health / "resource_guard_latest.json").write_text(json.dumps(payload))


@pytest.mark.parametrize(
    "overrides",
    [
        {"timestamp_utc": "2000-01-01T00:00:00+00:00"},
        {"source_timestamp_utc": "2000-01-01T00:00:00+00:00"},
        {"timestamp_utc": "2099-01-01T00:00:00+00:00"},
        {"timestamp_utc": "2026-09-12T00:00:00"},
        {"input_evidence_ready": False},
        {"measurement_refreshed": False},
        {"swap_used_gb": float("nan")},
        {"swap_used_gb": -1},
        {"memory_free_pct": float("inf")},
        {"memory_free_pct": True},
    ],
)
def test_unusable_evidence_cannot_enable_aggressive_settings(
    tmp_path, monkeypatch, overrides
):
    resource(tmp_path, **overrides)
    monkeypatch.setenv("SQLITE_CACHE_SIZE_KB", "65536")
    monkeypatch.setenv("SQLITE_TEMP_STORE_MODE", "MEMORY")
    monkeypatch.setenv("SQLITE_ALLOW_MMAP", "1")
    monkeypatch.setenv("SQLITE_MMAP_SIZE_MB", "256")
    settings = src.resolve_sqlite_runtime_settings(tmp_path)
    assert settings["pressure_level"] == "red"
    assert settings["resource_evidence_ready"] is False
    assert settings["temp_store_mode"] == "FILE"
    assert settings["cache_size_kb"] == 2048
    assert settings["mmap_size_bytes"] == 0


def test_missing_evidence_is_restrictive(tmp_path):
    assert src.resolve_sqlite_runtime_settings(tmp_path)["pressure_level"] == "red"


@pytest.mark.parametrize("age,expected", [(0, "red"), (121, "green")])
def test_adaptive_safety_floor_is_applied_only_while_fresh(tmp_path, age, expected):
    resource(tmp_path)
    (tmp_path / "governance/health/runtime_throttle_control_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": (
                    datetime.now(timezone.utc) - timedelta(seconds=age)
                ).isoformat(),
                "input_evidence_ready": False,
                "adaptive_safety_limits": {
                    "active": True,
                    "minimum_memory_pressure_level": "high",
                },
            }
        )
    )
    assert src.resolve_sqlite_runtime_settings(tmp_path)["pressure_level"] == expected


def test_readonly_sees_committed_wal_and_escapes_uri_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_WAL_AUTOCHECKPOINT_PAGES", "0")
    path = tmp_path / "special ?#% database.sqlite3"
    writer = src.connect_sqlite(path, project_root=tmp_path)
    try:
        writer.execute("CREATE TABLE evidence (id INTEGER)")
        writer.execute("INSERT INTO evidence VALUES (7)")
        writer.commit()
        assert Path(f"{path}-wal").stat().st_size > 0
        assert writer.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 0
        assert writer.execute("PRAGMA synchronous").fetchone()[0] == 1
        reader = src.connect_sqlite(
            path, project_root=tmp_path, readonly=True, query_only=True
        )
        try:
            assert reader.execute("SELECT id FROM evidence").fetchall() == [(7,)]
            with pytest.raises(sqlite3.OperationalError):
                reader.execute("INSERT INTO evidence VALUES (8)")
            assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            reader.close()
    finally:
        writer.close()


def test_caller_lock_budget_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_OPS_SQLITE_BUSY_TIMEOUT_MS", "900000")
    path = tmp_path / "db.sqlite3"
    writer = src.connect_sqlite(path, project_root=tmp_path)
    contender = src.connect_sqlite(path, project_root=tmp_path, timeout_seconds=0.04)
    try:
        writer.execute("CREATE TABLE evidence (id INTEGER)")
        writer.execute("BEGIN IMMEDIATE")
        assert contender.execute("PRAGMA busy_timeout").fetchone()[0] == 40
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            contender.execute("INSERT INTO evidence VALUES (1)")
        assert time.monotonic() - started < 0.75
    finally:
        writer.close()
        contender.close()


def test_zero_timeout_and_explicit_full_durability_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_OPS_SQLITE_BUSY_TIMEOUT_MS", "0")
    conn = src.connect_sqlite(
        tmp_path / "db",
        project_root=tmp_path,
        extra_pragmas=["PRAGMA synchronous=FULL"],
    )
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 0
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2
    finally:
        conn.close()


def test_setup_failure_closes_connection(tmp_path, monkeypatch):
    conn = sqlite3.connect(":memory:")
    monkeypatch.setattr(src.sqlite3, "connect", lambda *a, **k: conn)
    with pytest.raises(sqlite3.OperationalError):
        src.connect_sqlite(
            tmp_path / "db", project_root=tmp_path, extra_pragmas=["invalid SQL"]
        )
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")


def test_in_memory_connection_stays_in_memory(tmp_path):
    conn = src.connect_sqlite(":memory:", project_root=tmp_path)
    try:
        assert conn.execute("PRAGMA database_list").fetchall() == [(0, "main", "")]
        conn.execute("CREATE TABLE evidence (id INTEGER)")
    finally:
        conn.close()


@pytest.mark.parametrize("target", ["direct", "alias", "wal"])
def test_protected_database_family_rejected_before_metadata(
    tmp_path, monkeypatch, target
):
    path = tmp_path / "db"
    if target == "direct":
        path = Path("/Volumes/VIDEO/db")
    elif target == "alias":
        path.symlink_to("/Volumes/VIDEO/db")
    else:
        Path(f"{path}-wal").symlink_to("/Volumes/VIDEO/db-wal")
    original = Path.lstat

    def checked_lstat(candidate, *args, **kwargs):
        assert not str(candidate).casefold().startswith("/volumes/video")
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", checked_lstat)
    with pytest.raises(PermissionError, match="protected_path"):
        src.connect_sqlite(path, project_root=tmp_path)
    summary = src.sqlite_integrity_summary(path, project_root=tmp_path)
    assert summary["ok"] is False
    assert "protected_path" in summary["quick_check"]


def test_integrity_check_reports_success_missing_and_expired_budget(tmp_path):
    path = tmp_path / "db"
    assert (
        src.sqlite_integrity_summary(path, project_root=tmp_path)["quick_check"]
        == "missing"
    )
    conn = src.connect_sqlite(path, project_root=tmp_path)
    conn.execute("CREATE TABLE evidence (id INTEGER)")
    conn.close()
    assert src.sqlite_integrity_summary(path, project_root=tmp_path)["ok"] is True
    expired = src.sqlite_integrity_summary(
        path, project_root=tmp_path, timeout_seconds=0
    )
    assert expired["quick_check"] == "timeout"
    assert expired["timed_out"] is True
    assert expired["ok"] is False


def test_integrity_vm_work_is_interrupted_and_handle_closed(tmp_path, monkeypatch):
    path = tmp_path / "db"
    path.touch()
    real = sqlite3.connect(":memory:")

    class SlowCheck:
        def set_progress_handler(self, callback, steps):
            real.set_progress_handler(callback, steps)

        def execute(self, sql):
            return real.execute(
                "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<1000000000) SELECT sum(x) FROM n"
            )

        def close(self):
            real.close()

    monkeypatch.setattr(src, "connect_sqlite", lambda *a, **k: SlowCheck())
    result = src.sqlite_integrity_summary(
        path, project_root=tmp_path, timeout_seconds=0.03
    )
    assert result["timed_out"] is True
    assert result["quick_check"] == "timeout"
    assert result["elapsed_seconds"] < 0.75
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        real.execute("SELECT 1")
