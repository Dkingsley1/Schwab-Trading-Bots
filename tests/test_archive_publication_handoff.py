import time
import sys

import pytest

from core.runtime_maintenance import engage_maintenance_hold, maintenance_hold_snapshot
from scripts.ops import cold_archive_compactor as owner


def setup_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNTIME_MAINTENANCE_HOLD_PATH", str(tmp_path / "hold.flag"))
    monkeypatch.setattr(
        owner, "configured_sql_writer_lock_path", lambda root: root / "writer.lock"
    )
    monkeypatch.setattr(
        owner, "wait_for_writer_handoff", lambda *a, **kw: {"ready": True}
    )


def test_publication_exclusively_locks_then_releases_on_exception(
    tmp_path, monkeypatch
):
    setup_owner(tmp_path, monkeypatch)
    contract = {}
    with pytest.raises(RuntimeError, match="test_failure"):
        with owner._filesystem_publication_guard(
            tmp_path,
            deadline=time.monotonic() + 1200,
            timeout_seconds=30,
            poll_seconds=0.1,
        ) as (deadline, contract):
            assert deadline - time.monotonic() <= 60
            assert maintenance_hold_snapshot(tmp_path)["active"]
            assert owner._acquire_lock(tmp_path / "writer.lock")[0] is None
            raise RuntimeError("test_failure")
    assert not maintenance_hold_snapshot(tmp_path)["active"]
    assert contract["released"] is True
    handle, _ = owner._acquire_lock(tmp_path / "writer.lock")
    assert handle is not None
    handle.close()


def test_publication_preserves_an_existing_hold(tmp_path, monkeypatch):
    setup_owner(tmp_path, monkeypatch)
    existing = engage_maintenance_hold(tmp_path, reason="operator", owner="operator")
    with pytest.raises(RuntimeError, match="existing_maintenance_hold"):
        with owner._filesystem_publication_guard(
            tmp_path,
            deadline=time.monotonic() + 1200,
            timeout_seconds=0,
            poll_seconds=0.1,
        ):
            pytest.fail("must not publish")
    assert maintenance_hold_snapshot(tmp_path)["token"] == existing["token"]


def test_publication_timeout_releases_only_its_hold(tmp_path, monkeypatch):
    setup_owner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        owner, "wait_for_writer_handoff", lambda *a, **kw: {"ready": False}
    )
    with pytest.raises(RuntimeError, match="writer_handoff_timeout"):
        with owner._filesystem_publication_guard(
            tmp_path,
            deadline=time.monotonic() + 1200,
            timeout_seconds=0,
            poll_seconds=0.1,
        ):
            pytest.fail("must not publish")
    assert not maintenance_hold_snapshot(tmp_path)["active"]


def test_expired_publication_has_no_side_effects(tmp_path, monkeypatch):
    setup_owner(tmp_path, monkeypatch)
    with pytest.raises(TimeoutError):
        with owner._filesystem_publication_guard(
            tmp_path, deadline=time.monotonic() - 1, timeout_seconds=0, poll_seconds=0.1
        ):
            pytest.fail("must not publish")
    assert not (tmp_path / "hold.flag").exists()


def test_native_cli_keeps_writer_free_during_copy_and_locks_only_at_publication(
    tmp_path, monkeypatch
):
    setup_owner(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(owner, "writer_state_snapshot", lambda root: {"active": True})
    monkeypatch.setattr(
        owner.cold_sqlite_filesystem_compaction,
        "require_compressor",
        lambda name: "tool",
    )
    from scripts.ops import cold_archive_catalog

    monkeypatch.setattr(
        cold_archive_catalog, "build_catalog", lambda *a, **kw: {"ok": True}
    )
    archive = tmp_path / "archives"
    archive.mkdir()
    observed = []

    def build(**kwargs):
        assert not maintenance_hold_snapshot(tmp_path)["active"]
        handle, _ = owner._acquire_lock(tmp_path / "writer.lock")
        assert handle is not None, "copy phase must not own SQL writer lock"
        handle.close()
        with kwargs["publication_guard"](time.monotonic() + 300) as (_, contract):
            assert maintenance_hold_snapshot(tmp_path)["active"]
            assert owner._acquire_lock(tmp_path / "writer.lock")[0] is None
        assert contract["released"]
        observed.append(True)
        return {"ok": True, "overall_status": "ready"}

    monkeypatch.setattr(owner.cold_sqlite_filesystem_compaction, "build_payload", build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cold_archive_compactor.py",
            "--apply",
            "--archive-root",
            str(archive),
            "--filesystem-compress-sqlite",
            str(archive / "cold.sqlite3"),
            "--filesystem-compressor",
            "ditto",
            "--coordinate-writer-handoff",
            "--filesystem-timeout-seconds",
            "300",
            "--maintenance-hold-ttl-seconds",
            "360",
            "--out-file",
            str(tmp_path / "result.json"),
            "--lock-path",
            str(tmp_path / "owner.lock"),
            "--json",
        ],
    )
    assert owner.main() == 0
    assert observed == [True]
    assert not maintenance_hold_snapshot(tmp_path)["active"]
