import json
import os
import time
from pathlib import Path

import pytest

from scripts import build_runtime_training_snapshot as src


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    rows = tmp_path / "rows.jsonl"
    directory = tmp_path / ".rows.generations"
    directory.mkdir()
    old = directory / ("." + "a" * 32 + ".jsonl.building")
    old.write_bytes(b"unpublished scratch\n")
    now = time.time() + 7200
    monkeypatch.setattr(src.shutil, "which", lambda name: "/usr/sbin/lsof")
    monkeypatch.setattr(
        src,
        "run_bounded_process_group",
        lambda *a, **kw: {"rc": 1, "stdout": "", "stderr": ""},
    )
    return rows, old, now


def test_only_old_unpublished_regular_single_link_scratch_is_removed(scratch):
    rows, old, now = scratch
    directory = old.parent
    protected = []
    for name in ("b" * 32 + ".jsonl", "c" * 32 + ".jsonl.gz", "foreign.building"):
        path = directory / name
        path.write_bytes(b"preserve")
        protected.append(path)
    recent = directory / ("." + "d" * 32 + ".jsonl.building")
    recent.write_bytes(b"active")
    os.utime(recent, (now, now))
    linked = directory / ("." + "e" * 32 + ".jsonl.building")
    os.link(protected[0], linked)
    symlink = directory / ("." + "f" * 32 + ".jsonl.building")
    symlink.symlink_to(protected[0])
    result = src._cleanup_abandoned_snapshot_builds(
        rows, project_root=rows.parent, apply=True, now=now
    )
    assert result["ok"] and result["removed_count"] == 1
    assert not old.exists()
    assert all(p.exists() for p in [*protected, recent, linked, symlink])
    assert not result["published_snapshots_modified"]


def test_preview_and_manifest_reference_never_remove_scratch(scratch):
    rows, old, now = scratch
    preview = src._cleanup_abandoned_snapshot_builds(
        rows, project_root=rows.parent, now=now
    )
    assert preview["candidate_count"] == 1 and old.exists()
    kept = src._cleanup_abandoned_snapshot_builds(
        rows, project_root=rows.parent, apply=True, protected_paths=(old,), now=now
    )
    assert kept["removed_count"] == 0 and old.exists()


@pytest.mark.parametrize(
    "probe",
    [
        {"rc": 0, "stdout": "123", "stderr": ""},
        {"rc": 124, "stdout": "", "stderr": ""},
        {"rc": 1, "stdout": "", "stderr": "denied"},
    ],
)
def test_open_or_unknown_handle_probe_retains_every_candidate(
    scratch, monkeypatch, probe
):
    rows, old, now = scratch
    monkeypatch.setattr(src, "run_bounded_process_group", lambda *a, **kw: probe)
    result = src._cleanup_abandoned_snapshot_builds(
        rows, project_root=rows.parent, apply=True, now=now
    )
    assert not result["ok"] and result["removed_count"] == 0 and old.exists()


def test_changed_identity_after_idle_probe_is_not_deleted(scratch, monkeypatch):
    rows, old, now = scratch

    def probe(*a, **kw):
        old.write_bytes(b"changed")
        return {"rc": 1, "stdout": "", "stderr": ""}

    monkeypatch.setattr(src, "run_bounded_process_group", probe)
    result = src._cleanup_abandoned_snapshot_builds(
        rows, project_root=rows.parent, apply=True, now=now
    )
    assert not result["ok"] and old.read_bytes() == b"changed"


def test_generation_directory_redirect_is_rejected(tmp_path):
    (tmp_path / ".rows.generations").symlink_to("/Volumes/VIDEO/private")
    result = src._cleanup_abandoned_snapshot_builds(
        tmp_path / "rows.jsonl", project_root=tmp_path, apply=True
    )
    assert not result["ok"] and result["removed_count"] == 0


@pytest.mark.parametrize("cleanup_only", [True, False])
def test_owner_cleans_before_rebuild_and_cleanup_only_never_rebuilds(
    scratch, monkeypatch, cleanup_only
):
    rows, old, now = scratch
    root = rows.parent
    health = root / "health.json"
    health.write_text(
        json.dumps(
            {"rows_path": str(root / "published.jsonl"), "timestamp_utc": "original"}
        )
    )
    monkeypatch.setattr(src.time, "time", lambda: now)
    argv = [
        str(Path(src.__file__)),
        "--bounded-worker",
        "--project-root",
        str(root),
        "--rows-path",
        str(rows),
        "--health-path",
        str(health),
        "--lock-path",
        str(root / "snapshot.lock"),
        "--json",
    ]
    if cleanup_only:
        argv += ["--cleanup-abandoned-builds", "--apply-cleanup"]
    monkeypatch.setattr(src.sys, "argv", argv)
    builds = []

    def build(*args):
        assert not old.exists()
        builds.append(True)
        return 0

    monkeypatch.setattr(src, "_build_locked_snapshot", build)
    assert src.main() == 0
    assert bool(builds) is not cleanup_only
    assert not old.exists()
    assert json.loads(health.read_text())["timestamp_utc"] == "original"
    receipt = root / "governance/storage_recovery/snapshot_scratch_cleanup_latest.json"
    saved = json.loads(receipt.read_text())
    assert saved["last_reclamation"]["removed_count"] == 1
    assert src.main() == 0
    assert json.loads(receipt.read_text())["last_reclamation"] == saved["last_reclamation"]
    if cleanup_only:
        before_preview = receipt.read_bytes()
        monkeypatch.setattr(src.sys, "argv", [part for part in argv if part != "--apply-cleanup"])
        assert src.main() == 0 and receipt.read_bytes() == before_preview


def test_cleanup_cannot_enter_while_snapshot_writer_holds_lock(scratch, monkeypatch):
    rows, old, now = scratch
    root = rows.parent
    lock = root / "snapshot.lock"
    handle, _ = src._acquire_single_flight_lock(
        lock, project_root=root, health_path=root / "health.json", rows_path=rows
    )
    monkeypatch.setattr(
        src.sys,
        "argv",
        [
            str(Path(src.__file__)),
            "--bounded-worker",
            "--project-root",
            str(root),
            "--rows-path",
            str(rows),
            "--health-path",
            str(root / "health.json"),
            "--lock-path",
            str(lock),
            "--cleanup-abandoned-builds",
            "--apply-cleanup",
            "--json",
        ],
    )
    try:
        assert src.main() == 0 and old.exists()
    finally:
        handle.close()
