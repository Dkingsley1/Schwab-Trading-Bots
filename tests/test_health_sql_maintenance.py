import sqlite3

from core.runtime_maintenance import engage_maintenance_hold
from scripts import health_gates, session_ready_check


def test_shard_size_falls_back_to_file_size_without_database_open(
    tmp_path, monkeypatch
):
    path = tmp_path / "data/sql_link_shards/jsonl_link_explanations.sqlite3"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unopened test database")
    engage_maintenance_hold(tmp_path, reason="storage_handoff")
    monkeypatch.delenv("SQL_LINK_SERVICE_MAINTENANCE_HOLD_TOKEN", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("health reader opened a held database")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    assert (
        health_gates._priority_shard_live_db_size_gb(tmp_path, "explanations")
        == path.stat().st_size / 1024**3
    )
    assert not list(path.parent.glob("*-shm"))


def test_session_probe_defers_without_directory_or_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(session_ready_check, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(session_ready_check, "DB_PATH", tmp_path / "new/db.sqlite3")
    hold = engage_maintenance_hold(tmp_path, reason="storage_handoff")
    monkeypatch.setenv("SQL_LINK_SERVICE_MAINTENANCE_HOLD_TOKEN", hold["token"])
    assert session_ready_check._sql_writable() is False
    assert not (tmp_path / "new").exists()
