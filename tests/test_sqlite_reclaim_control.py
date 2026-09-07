import fcntl
import json
import sqlite3
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts.ops import sqlite_reclaim_control as reclaim

SPACE = {
    "physical_gb": 58.0,
    "live_gb": 12.0,
    "reclaimable_gb": 46.0,
    "reclaimable_ratio": 46 / 58,
}


def test_scheduled_cycle_runs_both_steps_inside_one_guard(tmp_path):
    repo = Path(reclaim.__file__).resolve().parents[2]
    ops = tmp_path / "scripts/ops"
    ops.mkdir(parents=True)
    launcher = ops / "sqlite_maintenance_launchd.sh"
    launcher.write_text(
        (repo / "scripts/ops/sqlite_maintenance_launchd.sh").read_text()
    )
    guard = ops / "run_guarded_maintenance.sh"
    guard.write_text(
        '#!/bin/sh\necho guard >> "$RECLAIM_TEST_TRACE"\nshift\nexec "$@"\n'
    )
    guard.chmod(0o755)
    python = tmp_path / ".venv314/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\necho "$*" >> "$RECLAIM_TEST_TRACE"\n')
    python.chmod(0o755)
    trace = tmp_path / "trace"
    env = dict(
        os.environ, RECLAIM_TEST_TRACE=str(trace), SQLITE_LAUNCHD_ALLOW_AUTO_VACUUM="0"
    )
    subprocess.run(["/bin/zsh", str(launcher)], env=env, check=True)
    lines = trace.read_text().splitlines()
    assert lines[0] == "guard"
    assert len(lines) == 3
    assert "sqlite_reclaim_control.py --apply --json" in lines[1]
    assert "sqlite_performance_maintenance.py" in lines[2]
    assert "--no-auto-vacuum" in lines[2]


def test_freelist_measurement_distinguishes_live_from_physical_bytes(tmp_path):
    path = tmp_path / "evidence.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY, payload BLOB)")
    conn.execute("INSERT INTO evidence VALUES (1, zeroblob(1048576))")
    conn.commit()
    conn.execute("DELETE FROM evidence")
    conn.commit()
    conn.close()
    space = reclaim.database_space(path)
    assert space["reclaimable_gb"] > 0
    assert space["live_gb"] < space["physical_gb"]


@pytest.mark.parametrize(
    "internal,scratch,memory,expected",
    [
        (64, 183, True, []),
        (32, 183, True, ["insufficient_database_volume_reserve"]),
        (64, 130, True, ["insufficient_scratch_reserve"]),
        (64, 183, False, ["memory_pressure"]),
    ],
)
def test_reclaim_respects_memory_and_both_volume_reserves(
    internal, scratch, memory, expected
):
    assert (
        reclaim.reclaim_blockers(
            SPACE,
            internal_free_gb=internal,
            scratch_free_gb=scratch,
            same_filesystem=False,
            memory_ready=memory,
        )
        == expected
    )


def test_small_free_page_tail_does_not_trigger_heavy_compaction():
    space = dict(SPACE, reclaimable_gb=0.5, reclaimable_ratio=0.01)
    assert "below_material_reclaim_threshold" in reclaim.reclaim_blockers(
        space,
        internal_free_gb=200,
        scratch_free_gb=200,
        same_filesystem=True,
        memory_ready=True,
    )


def test_noop_does_not_request_maintenance_hold(tmp_path, monkeypatch):
    monkeypatch.setattr(
        reclaim, "database_space", lambda path: dict(SPACE, reclaimable_gb=0)
    )
    monkeypatch.setattr(
        reclaim,
        "_coordinate_priority_retention_handoff",
        lambda *args, **kw: pytest.fail("unexpected hold"),
    )
    result = reclaim.build_payload(
        tmp_path, tmp_path / "db", tmp_path / "scratch", apply=True
    )
    assert result["overall_status"] == "nothing_to_do"


def prepared(tmp_path, monkeypatch):
    (tmp_path / "governance/locks").mkdir(parents=True)
    db = tmp_path / "db.sqlite3"
    db.touch()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(reclaim, "database_space", lambda path: dict(SPACE))
    monkeypatch.setattr(
        reclaim.shutil, "disk_usage", lambda path: SimpleNamespace(free=200 * 2**30)
    )
    monkeypatch.setattr(
        reclaim.maintenance,
        "resolve_runtime_settings",
        lambda root: {"auto_vacuum_allowed": True},
    )
    return db, scratch


def test_busy_maintenance_lock_defers_without_starting_vacuum(tmp_path, monkeypatch):
    db, scratch = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(
        reclaim.subprocess, "run", lambda *args, **kw: pytest.fail("unexpected vacuum")
    )
    with (tmp_path / "governance/locks/storage_maintenance.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = reclaim.build_payload(tmp_path, db, scratch, apply=True)
    assert result["overall_status"] == "deferred"
    assert result["blockers"] == ["maintenance_or_writer_lock_busy"]


def test_failed_vacuum_releases_only_owned_hold(tmp_path, monkeypatch):
    db, scratch = prepared(tmp_path, monkeypatch)
    handoff = {"ready": True, "owned_hold": True, "token": "fixture-token"}
    released = []
    monkeypatch.setattr(
        reclaim, "_coordinate_priority_retention_handoff", lambda *args, **kw: handoff
    )
    monkeypatch.setattr(
        reclaim,
        "_release_priority_retention_handoff",
        lambda root, hold: released.append(hold) or {"released": True},
    )

    def failed(command, **kwargs):
        assert "--vacuum" in command
        assert kwargs["env"][reclaim.MAINTENANCE_HOLD_TOKEN_ENV] == "fixture-token"
        return SimpleNamespace(
            returncode=2, stdout=json.dumps({"vacuum_ran": False}), stderr="failed"
        )

    monkeypatch.setattr(reclaim.subprocess, "run", failed)
    result = reclaim.build_payload(tmp_path, db, scratch, apply=True)
    assert result["ok"] is False
    assert result["overall_status"] == "error"
    assert released == [handoff]


def test_writer_handoff_timeout_defers_and_releases_owned_hold(tmp_path, monkeypatch):
    db, scratch = prepared(tmp_path, monkeypatch)
    released = []
    monkeypatch.setattr(
        reclaim,
        "_coordinate_priority_retention_handoff",
        lambda *args, **kw: {"ready": False, "reason": "writer_handoff_timeout"},
    )
    monkeypatch.setattr(
        reclaim,
        "_release_priority_retention_handoff",
        lambda root, hold: released.append(hold) or {"released": True},
    )
    result = reclaim.build_payload(tmp_path, db, scratch, apply=True)
    assert result["blockers"] == ["writer_handoff_timeout"]
    assert len(released) == 1
