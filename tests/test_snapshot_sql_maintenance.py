import json
import sqlite3
from pathlib import Path

import pytest

from core.runtime_maintenance import engage_maintenance_hold
from scripts import snapshot_health_sql as owner


@pytest.mark.parametrize("sync", [owner.sync_snapshot_health_to_sqlite, owner.sync_raw_debug_snapshots_to_sqlite])
def test_maintenance_blocks_before_mkdir_or_sql_open(tmp_path, monkeypatch, sync):
    engage_maintenance_hold(tmp_path, reason="storage handoff")
    monkeypatch.setattr(owner, "connect_sqlite", lambda *a, **k: pytest.fail("DB opened under hold"))
    with pytest.raises(RuntimeError, match="snapshot_sql_deferred"):
        sync(project_root=tmp_path)
    assert not (tmp_path / "data").exists()


def test_context_can_observe_files_without_writing_database_under_hold(tmp_path, monkeypatch):
    engage_maintenance_hold(tmp_path, reason="storage handoff")
    health = tmp_path / "governance/health/snapshot_coverage_latest.json"
    health.write_text(json.dumps({"ok": True, "coverage_ratio": 1.0}))
    monkeypatch.setattr(owner, "connect_sqlite", lambda *a, **k: pytest.fail("DB opened under hold"))
    _, meta = owner.load_snapshot_context(project_root=tmp_path)
    assert meta["selected_source"]["snapshot_coverage"] == "file"
    assert "snapshot_sql_deferred" in meta["sql_sync"]["error"]
    coverage = owner.debug_snapshot_ingest_coverage(project_root=tmp_path)
    assert coverage["overall_status"] == "deferred" and not coverage["all_ready"]
    assert not (tmp_path / "data").exists()


def test_reader_is_read_only_and_uses_real_project_root(tmp_path, monkeypatch):
    actual = tmp_path / "standby/snapshot.sqlite3"
    actual.parent.mkdir()
    db = sqlite3.connect(actual)
    owner._ensure_snapshot_health_schema(db)
    db.commit()
    db.close()
    logical = tmp_path / "data/snapshot_context.sqlite3"
    logical.parent.mkdir()
    logical.symlink_to(actual)
    calls = []
    real = owner.connect_sqlite
    def connect(path, **kwargs):
        calls.append((path, kwargs))
        return real(path, **kwargs)
    monkeypatch.setattr(owner, "connect_sqlite", connect)
    assert owner.load_snapshot_health_payloads_from_sqlite(logical, project_root=tmp_path) == {}
    assert calls[0][0] == logical
    assert calls[0][1]["project_root"] == tmp_path
    assert calls[0][1]["readonly"] is True


def test_hold_arriving_before_commit_rolls_back_health_rows(tmp_path, monkeypatch):
    calls = 0
    real = owner._require_database_io
    def check(root):
        nonlocal calls
        calls += 1
        if calls == 3:
            engage_maintenance_hold(root, reason="arrived during write")
        real(root)
    monkeypatch.setattr(owner, "_require_database_io", check)
    with pytest.raises(RuntimeError, match="snapshot_sql_deferred"):
        owner.sync_snapshot_health_to_sqlite(project_root=tmp_path, payloads={"snapshot_coverage": {"ok": True}})
    db = sqlite3.connect(tmp_path / "data/snapshot_context.sqlite3")
    assert db.execute("select count(*) from snapshot_health_records").fetchone()[0] == 0
    db.close()
