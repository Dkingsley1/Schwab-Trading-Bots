from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.ops import storage_sqlite_local_failover as failover


def _make_db(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        conn.executemany("INSERT INTO observations(value) VALUES (?)", [(value,) for value in values])
        conn.commit()


def _route_database(project_root: Path, source_root: Path, relative_path: str) -> None:
    repo = project_root / relative_path
    source = source_root / relative_path
    repo.parent.mkdir(parents=True, exist_ok=True)
    repo.symlink_to(source)
    for suffix in ("-wal", "-shm"):
        Path(f"{repo}{suffix}").symlink_to(Path(f"{source}{suffix}"))


def _idle_guards(monkeypatch) -> None:
    monkeypatch.setattr(
        failover.writer_state,
        "writer_state_snapshot",
        lambda _project_root: {"active": False, "running": False, "status": "idle"},
    )
    monkeypatch.setattr(failover, "_runtime_processes", lambda: [])
    monkeypatch.setattr(
        failover,
        "maintenance_hold_snapshot",
        lambda _project_root: {"active": True, "reason": "test_maintenance"},
    )


def test_dry_run_requires_no_route_change(tmp_path: Path, monkeypatch) -> None:
    _idle_guards(monkeypatch)
    project_root = tmp_path / "project"
    source_root = tmp_path / "external"
    target_root = project_root / "local_fallback_storage"
    target_root.mkdir(parents=True)
    relative_path = "data/example.sqlite3"
    _make_db(source_root / relative_path, ["a", "b"])
    _route_database(project_root, source_root, relative_path)

    payload = failover.build_payload(
        project_root,
        source_root=source_root,
        target_root=target_root,
        relative_paths=[relative_path],
        apply=False,
        min_free_after_bytes=0,
        page_batch=4,
        timeout_seconds=10,
        lock_path=tmp_path / "failover.lock",
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "dry_run_ready"
    assert (project_root / relative_path).resolve() == (source_root / relative_path).resolve()
    assert not (target_root / relative_path).exists()


def test_apply_stages_all_databases_then_switches_routes(tmp_path: Path, monkeypatch) -> None:
    _idle_guards(monkeypatch)
    project_root = tmp_path / "project"
    source_root = tmp_path / "external"
    target_root = project_root / "local_fallback_storage"
    target_root.mkdir(parents=True)
    relative_paths = ["data/one.sqlite3", "data/two.sqlite3"]
    for index, rel in enumerate(relative_paths):
        _make_db(source_root / rel, [f"value-{index}", "shared"])
        _route_database(project_root, source_root, rel)
    _make_db(target_root / relative_paths[0], ["old-local"])

    payload = failover.build_payload(
        project_root,
        source_root=source_root,
        target_root=target_root,
        relative_paths=relative_paths,
        apply=True,
        min_free_after_bytes=0,
        page_batch=4,
        timeout_seconds=10,
        lock_path=tmp_path / "failover.lock",
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "switched_local_verified"
    assert payload["route_switch"]["ok"] is True
    assert (source_root / relative_paths[0]).exists()
    for rel in relative_paths:
        repo = project_root / rel
        target = target_root / rel
        assert repo.resolve() == target.resolve()
        assert Path(f"{repo}-wal").is_symlink()
        assert Path(f"{repo}-shm").is_symlink()
        with sqlite3.connect(target) as conn:
            count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        assert count == 2
    preserved = payload["entries"][0]["commit"]["preserved_existing"]
    assert len(preserved) == 1
    assert Path(preserved[0]).exists()


def test_stage_failure_leaves_every_route_on_source(tmp_path: Path, monkeypatch) -> None:
    _idle_guards(monkeypatch)
    project_root = tmp_path / "project"
    source_root = tmp_path / "external"
    target_root = project_root / "local_fallback_storage"
    target_root.mkdir(parents=True)
    relative_paths = ["data/good.sqlite3", "data/bad.sqlite3"]
    _make_db(source_root / relative_paths[0], ["good"])
    bad = source_root / relative_paths[1]
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not-a-sqlite-database")
    for rel in relative_paths:
        _route_database(project_root, source_root, rel)

    payload = failover.build_payload(
        project_root,
        source_root=source_root,
        target_root=target_root,
        relative_paths=relative_paths,
        apply=True,
        min_free_after_bytes=0,
        page_batch=4,
        timeout_seconds=10,
        lock_path=tmp_path / "failover.lock",
    )

    assert payload["ok"] is False
    assert payload["overall_status"] == "stage_failed"
    for rel in relative_paths:
        assert (project_root / rel).resolve(strict=False) == (source_root / rel).resolve(strict=False)
        assert not (target_root / rel).exists()


def test_preflight_blocks_when_runtime_is_active(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        failover.writer_state,
        "writer_state_snapshot",
        lambda _project_root: {"active": False, "running": False},
    )
    monkeypatch.setattr(
        failover,
        "_runtime_processes",
        lambda: [{"pid": 123, "command": "scripts/run_all_sleeves.py --broker schwab"}],
    )
    monkeypatch.setattr(
        failover,
        "maintenance_hold_snapshot",
        lambda _project_root: {"active": True, "reason": "test_maintenance"},
    )
    project_root = tmp_path / "project"
    source_root = tmp_path / "external"
    target_root = project_root / "local_fallback_storage"
    target_root.mkdir(parents=True)
    relative_path = "data/example.sqlite3"
    _make_db(source_root / relative_path, ["a"])
    _route_database(project_root, source_root, relative_path)

    payload = failover.build_payload(
        project_root,
        source_root=source_root,
        target_root=target_root,
        relative_paths=[relative_path],
        apply=True,
        min_free_after_bytes=0,
        page_batch=4,
        timeout_seconds=10,
        lock_path=tmp_path / "failover.lock",
    )

    assert payload["ok"] is False
    assert payload["overall_status"] == "blocked"
    assert "runtime_processes_active" in payload["blockers"]
    assert (project_root / relative_path).resolve() == (source_root / relative_path).resolve()


def test_preflight_blocks_without_runtime_maintenance_hold(tmp_path: Path, monkeypatch) -> None:
    _idle_guards(monkeypatch)
    monkeypatch.setattr(
        failover,
        "maintenance_hold_snapshot",
        lambda _project_root: {"active": False, "reason": ""},
    )
    project_root = tmp_path / "project"
    source_root = tmp_path / "external"
    target_root = project_root / "local_fallback_storage"
    target_root.mkdir(parents=True)
    relative_path = "data/example.sqlite3"
    _make_db(source_root / relative_path, ["a"])
    _route_database(project_root, source_root, relative_path)

    payload = failover.build_payload(
        project_root,
        source_root=source_root,
        target_root=target_root,
        relative_paths=[relative_path],
        apply=True,
        min_free_after_bytes=0,
        page_batch=4,
        timeout_seconds=10,
        lock_path=tmp_path / "failover.lock",
    )

    assert payload["ok"] is False
    assert payload["overall_status"] == "blocked"
    assert "runtime_maintenance_hold_inactive" in payload["blockers"]
    assert (project_root / relative_path).resolve() == (source_root / relative_path).resolve()


def test_orphaned_writer_progress_does_not_block_quiesced_failover() -> None:
    snapshot = {
        "active": False,
        "running": True,
        "progress_orphaned": True,
        "writer_lock_held": False,
        "writer_owner_pid_live": False,
        "child_writer_active": False,
        "active_child_writer_count": 0,
    }

    assert failover._writer_is_active(snapshot) is False


def test_writer_lock_blocks_failover_even_when_progress_is_orphaned() -> None:
    snapshot = {
        "active": False,
        "running": True,
        "progress_orphaned": True,
        "writer_lock_held": True,
        "writer_owner_pid_live": False,
        "child_writer_active": False,
        "active_child_writer_count": 0,
    }

    assert failover._writer_is_active(snapshot) is True


def test_stale_running_metadata_without_physical_owner_does_not_block_failover() -> None:
    snapshot = {
        "active": False,
        "active_source": "idle",
        "running": True,
        "progress_orphaned": False,
        "progress_age_minutes": 35.0,
        "writer_lock_held": False,
        "writer_owner_pid_live": False,
        "child_writer_active": False,
        "active_child_writer_count": 0,
    }

    assert failover._writer_is_active(snapshot) is False


def test_identity_match_ignores_sqlite_schema_cookie_churn() -> None:
    source = {
        "page_count": 10,
        "page_size": 4096,
        "freelist_count": 0,
        "schema_version": 15,
        "user_version": 0,
        "application_id": 0,
        "schema_object_count": 2,
        "schema_sha256": "abc",
    }
    target = {**source, "schema_version": 1}

    ok, mismatches = failover._identity_matches(source, target)

    assert ok is True
    assert mismatches == []


def test_empty_wal_creation_does_not_look_like_source_mutation(tmp_path: Path) -> None:
    database = tmp_path / "example.sqlite3"
    database.write_bytes(b"database")
    before = failover._source_signature(database)
    Path(f"{database}-wal").touch()
    after = failover._source_signature(database)

    assert before == after


def test_activate_existing_local_reuses_prior_verified_copy(tmp_path: Path, monkeypatch) -> None:
    _idle_guards(monkeypatch)
    project_root = tmp_path / "project"
    source_root = tmp_path / "external"
    target_root = project_root / "local_fallback_storage"
    target_root.mkdir(parents=True)
    relative_path = "data/example.sqlite3"
    _make_db(source_root / relative_path, ["a", "b", "c"])
    _route_database(project_root, source_root, relative_path)
    proof_path = tmp_path / "prior_failover.json"

    first = failover.build_payload(
        project_root,
        source_root=source_root,
        target_root=target_root,
        relative_paths=[relative_path],
        apply=True,
        min_free_after_bytes=0,
        page_batch=4,
        timeout_seconds=10,
        lock_path=tmp_path / "first.lock",
        prior_proof_path=proof_path,
    )
    proof_path.write_text(json.dumps(first), encoding="utf-8")
    target = target_root / relative_path
    target_inode = target.stat().st_ino
    repo = project_root / relative_path
    for suffix in ("", "-wal", "-shm"):
        link = Path(f"{repo}{suffix}")
        link.unlink()
        link.symlink_to(Path(f"{source_root / relative_path}{suffix}"))

    activated = failover.build_payload(
        project_root,
        source_root=source_root,
        target_root=target_root,
        relative_paths=[relative_path],
        apply=True,
        min_free_after_bytes=0,
        page_batch=4,
        timeout_seconds=10,
        activate_existing_local=True,
        prior_proof_path=proof_path,
        lock_path=tmp_path / "activation.lock",
    )

    assert activated["ok"] is True
    assert activated["overall_status"] == "activated_existing_local_verified"
    assert activated["entries"][0]["stage"]["prior_proof_ok"] is True
    assert activated["entries"][0]["commit"]["existing_target_reused"] is True
    assert target.stat().st_ino == target_inode
    assert repo.resolve() == target.resolve()
    assert (project_root / "config" / ".env.storage_override").read_text(encoding="utf-8").endswith(
        "BOT_LOGS_PREFER_EXTERNAL=0\n"
    )


def test_activate_existing_local_is_idempotent_when_route_is_already_local(tmp_path: Path, monkeypatch) -> None:
    _idle_guards(monkeypatch)
    project_root = tmp_path / "project"
    source_root = tmp_path / "external"
    target_root = project_root / "local_fallback_storage"
    target_root.mkdir(parents=True)
    relative_path = "data/example.sqlite3"
    _make_db(source_root / relative_path, ["a"])
    _make_db(target_root / relative_path, ["a"])
    _route_database(project_root, target_root, relative_path)

    payload = failover.build_payload(
        project_root,
        source_root=source_root,
        target_root=target_root,
        relative_paths=[relative_path],
        apply=False,
        min_free_after_bytes=0,
        page_batch=4,
        timeout_seconds=10,
        activate_existing_local=True,
        require_prior_activation_proof=False,
        lock_path=tmp_path / "activation.lock",
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "activation_dry_run_ready"
    assert payload["required_stage_bytes"] == 0
    assert payload["stage_headroom_bytes"] == 0
