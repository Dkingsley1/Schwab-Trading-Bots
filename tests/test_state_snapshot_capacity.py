from contextlib import closing
import gzip
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import background_work_budget as budget
from scripts.ops import state_snapshot_capacity as snapshot
from scripts import daily_state_snapshot_drill as drill


class Guard:
    def check(self):
        pass


def database(path):
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE rows(id INTEGER PRIMARY KEY, payload BLOB)")
        db.executemany(
            "INSERT INTO rows VALUES(?,?)", [(i, b"x" * 10000) for i in range(100)]
        )
        db.commit()
        db.execute("DELETE FROM rows WHERE id>3")
        db.commit()


def test_plan_uses_occupied_pages_and_enforced_output_cap(tmp_path):
    source = tmp_path / "source.sqlite3"
    database(source)
    plan = snapshot.plan_target(source, 2 * 1024**2)
    assert plan["freelist_bytes"] > plan["occupied_bytes"]
    assert plan["output_limit_bytes"] <= 2 * 1024**2
    with pytest.raises(ValueError, match="occupied_database"):
        snapshot.plan_target(source, 1)


def test_compact_snapshot_reads_wal_and_preserves_live_source(tmp_path):
    source, target = tmp_path / "source.sqlite3", tmp_path / "snapshot.sqlite3"
    database(source)
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO rows VALUES(100,?)", (b"committed-wal",))
        writer.commit()
        plan = snapshot.plan_target(source, 4 * 1024**2)
        snapshot.compact_snapshot(source, target, plan, Guard())
        assert source.stat().st_size > target.stat().st_size
        with closing(sqlite3.connect(target)) as restored:
            assert (
                restored.execute("SELECT * FROM rows ORDER BY id").fetchall()
                == writer.execute("SELECT * FROM rows ORDER BY id").fetchall()
            )
        assert writer.execute("SELECT COUNT(*) FROM rows").fetchone()[0] == 5


@pytest.mark.skipif(sys.platform != "darwin", reason="APFS clone requires macOS")
def test_real_clone_is_distinct_inode_and_write_isolated(tmp_path):
    source, target = tmp_path / "source", tmp_path / "restore"
    source.write_bytes(b"original bytes")
    snapshot.clone_restore(source, target)
    assert source.stat().st_ino != target.stat().st_ino
    assert snapshot.verify_snapshot(source, target, Guard(), False)[0]
    target.write_bytes(b"changed")
    assert source.read_bytes() == b"original bytes"


def test_clone_failure_never_falls_back_to_physical_copy(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "restore"
    source.write_bytes(b"original")
    monkeypatch.setattr(snapshot.sys, "platform", "unsupported")
    with pytest.raises(RuntimeError, match="clone_unavailable"):
        snapshot.clone_restore(source, target)
    assert not target.exists()


def test_combined_capacity_uses_each_target_size(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_LOCAL_STORAGE_TARGET_FREE_GB", "125")
    plans = [{"output_limit_bytes": 100}, {"output_limit_bytes": 200}]
    plan = snapshot.capacity(tmp_path, tmp_path, plans, False)
    assert plan["allocation_budget_bytes"] == 600
    assert plan["reserve_bytes"] == 125 * snapshot.GIB
    assert plan["reserve_scope"] == "local_configured_reserve"
    clone = snapshot.capacity(tmp_path, tmp_path, plans, True)
    assert clone["allocation_budget_bytes"] == 300 + 2 * 1024**2


def test_external_reserve_is_separate_and_never_below_64(tmp_path, monkeypatch):
    class Device:
        def __init__(self, n):
            self.st_dev = n

        def stat(self):
            return self

    monkeypatch.setattr(
        snapshot, "existing_parent", lambda p: Device(1 if p == "local" else 2)
    )
    monkeypatch.setenv("BOT_LOCAL_STORAGE_TARGET_FREE_GB", "125")
    monkeypatch.delenv("SNAPSHOT_DRILL_EXTERNAL_RESERVE_GB", raising=False)
    assert snapshot.reserve_bytes("local", "external") == (
        64 * snapshot.GIB,
        "separate_archive_volume_reserve",
    )
    monkeypatch.setenv("SNAPSHOT_DRILL_EXTERNAL_RESERVE_GB", "63")
    with pytest.raises(ValueError):
        snapshot.reserve_bytes("local", "external")


def test_corrupt_restore_never_gets_credit(tmp_path):
    source, target = tmp_path / "source", tmp_path / "restore"
    source.write_bytes(b"abc")
    target.write_bytes(b"abd")
    with pytest.raises(RuntimeError, match="hash_mismatch"):
        snapshot.verify_snapshot(source, target, Guard(), False)


def test_background_policy_uses_unprivileged_pacing_on_denial(monkeypatch):
    calls = []

    def deny(n):
        calls.append(n)
        raise PermissionError("denied")

    monkeypatch.setattr(budget.os, "nice", deny)
    assert budget.background_policy()["mode"] == "cooperative"
    assert calls == [15]


def test_pacing_repays_cpu_debt_without_privileged_calls():
    clock = [0.0, 0.0]
    sleeps = []

    def sleep(amount):
        sleeps.append(amount)
        clock[0] += amount

    p = budget.WorkBudget(wall=lambda: clock[0], cpu=lambda: clock[1], sleep=sleep)
    clock[:] = [1.0, 1.0]
    for _ in range(12):
        p.tick()
    assert sum(sleeps) == pytest.approx(3)
    assert p.snapshot()["process_cpu_seconds"] / p.snapshot()["elapsed_seconds"] == 0.25


def test_compact_cli_full_content_verification(tmp_path, monkeypatch):
    source = tmp_path / "source.sqlite3"
    database(source)
    monkeypatch.setattr(drill, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(snapshot.CopyGuard, "check", lambda self: None)
    original = snapshot.capacity

    def admitted(*args):
        result = original(*args)
        return {**result, "sufficient": True}

    monkeypatch.setattr(snapshot, "capacity", admitted)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drill",
            "--out-root",
            str(out),
            "--publish-latest",
            str(out / "latest.json"),
            "--targets",
            str(source),
            "--compact-sqlite",
            "--clone-restore",
            "--json",
        ],
    )
    assert drill.main() == 0
    result = json.loads((out / "latest.json").read_text())
    assert result["files_restore_verified"] == 1
    assert result["rows"][0]["sqlite_integrity_verified"]
    assert (
        result["rows"][0]["restore_storage_mode"] == "apfs_copy_on_write_shared_extents"
    )
    assert not result["full_platform_restore_verified"]


def test_protected_route_rejected_before_any_target_probe():
    with pytest.raises(ValueError, match="snapshot_route_unavailable"):
        snapshot.plan_target(Path("/Volumes/VIDEO/not-a-target"), 100)


def test_gzip_seal_verifies_all_decoded_bytes_before_releasing_probes(tmp_path):
    live, source, restored = (
        tmp_path / name for name in ("live", "snapshot", "restored")
    )
    data = b"retained history" * 10000
    for path in (live, source, restored):
        path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    proof = snapshot.seal_compressed_snapshot(source, restored, sha, Guard(), 1024**2)
    archive = Path(proof["archive_path"])
    assert gzip.decompress(archive.read_bytes()) == data
    assert proof["decoded_bytes"] == len(data)
    assert proof["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert json.loads(archive.with_suffix(".gz.receipt.json").read_text()) == proof
    assert live.read_bytes() == data
    assert not source.exists() and not restored.exists()


@pytest.mark.parametrize("mismatch", [False, True])
def test_failed_gzip_seal_preserves_both_probe_copies(tmp_path, mismatch):
    source, restored = tmp_path / "snapshot", tmp_path / "restored"
    data = os.urandom(10000)
    source.write_bytes(data)
    restored.write_bytes(data)
    with pytest.raises(RuntimeError, match="verification_failed|archive_budget"):
        snapshot.seal_compressed_snapshot(
            source,
            restored,
            "wrong" if mismatch else hashlib.sha256(data).hexdigest(),
            Guard(),
            1024**2 if mismatch else 10,
        )
    assert source.read_bytes() == restored.read_bytes() == data
    assert not (tmp_path / "snapshot.gz.receipt.json").exists()


def test_capacity_includes_compression_scratch(tmp_path):
    plan = snapshot.capacity(
        tmp_path,
        tmp_path,
        [{"output_limit_bytes": 100, "archive_limit_bytes": 50}],
        True,
    )
    assert plan["allocation_budget_bytes"] == 150 + 1024**2


@pytest.mark.parametrize("blocker", ["memory", "deadline", "reserve", "hold"])
def test_copy_guard_enforces_resource_bounds(tmp_path, monkeypatch, blocker):
    from scripts.ops import support_maintenance_gate

    monkeypatch.setattr(
        support_maintenance_gate,
        "support_maintenance_freeze_contract",
        lambda *a: {"active": False},
    )
    monkeypatch.setattr(
        snapshot.resource, "getrusage", lambda *a: SimpleNamespace(ru_maxrss=0)
    )
    monkeypatch.setattr(
        snapshot.shutil, "disk_usage", lambda *a: SimpleNamespace(free=100)
    )
    guard = snapshot.CopyGuard(tmp_path, tmp_path, 1, 60)
    if blocker == "memory":
        monkeypatch.setattr(
            snapshot.resource,
            "getrusage",
            lambda *a: SimpleNamespace(ru_maxrss=256 * 1024**2),
        )
    elif blocker == "deadline":
        guard.deadline = 0
    elif blocker == "reserve":
        guard.reserve = 101
    else:
        (tmp_path / "RUNTIME_MAINTENANCE_HOLD.flag").touch()
    with pytest.raises((RuntimeError, TimeoutError)):
        guard.check()


def test_support_pause_preserves_previous_restore_evidence(tmp_path, monkeypatch):
    from scripts.ops import support_maintenance_gate

    latest = tmp_path / "out/latest.json"
    latest.parent.mkdir()
    latest.write_text('{"previous": "restore proof"}')
    monkeypatch.setattr(drill, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        support_maintenance_gate,
        "support_maintenance_freeze_contract",
        lambda *a: {
            "active": True,
            "reason": "support_maintenance_frozen_for_mac_fluidity",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drill",
            "--compact-sqlite",
            "--out-root",
            str(latest.parent),
            "--publish-latest",
            str(latest),
            "--json",
        ],
    )
    assert drill.main() == 2
    assert json.loads(latest.read_text()) == {"previous": "restore proof"}
    assert json.loads(
        (
            tmp_path / "governance/health/state_snapshot_drill_attempt_latest.json"
        ).read_text()
    )["previous_restore_evidence_unchanged"]


@pytest.mark.parametrize(
    "change",
    [
        {"files_restore_verified": 4},
        {"files_checked": True},
        {"files_checked": "5"},
        {"accepted_metadata_only_large_files": 1},
        {"missing_files": ["db"]},
        {"published_latest_write_verified": False},
    ],
)
def test_incomplete_or_metadata_only_restore_cannot_admit_consumers(change):
    good = {
        "ok": True,
        "files_checked": 5,
        "files_restore_verified": 5,
        "latest_write_verified": True,
        "published_latest_write_verified": True,
    }
    assert snapshot.complete_restore_evidence(good)
    assert not snapshot.complete_restore_evidence({**good, **change})


def test_approved_recovery_yields_without_copy_work_until_resources_stabilize(
    tmp_path, monkeypatch
):
    from scripts.ops import approved_storage_recovery

    clock = [0.0]
    monkeypatch.setattr(snapshot.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        snapshot.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(
        snapshot.shutil, "disk_usage", lambda *args: SimpleNamespace(free=100)
    )
    monkeypatch.setattr(
        approved_storage_recovery, "resources_admitted", lambda *args: clock[0] >= 3
    )
    guard = snapshot.CopyGuard(tmp_path, tmp_path, 1, 60, operator_approved=True)
    guard._wait_for_recovery_resources()
    assert clock[0] == 13
    assert guard.admission_wait_seconds == 13


def test_approved_recovery_wait_cannot_bypass_deadline_or_operator_hold(
    tmp_path, monkeypatch
):
    from scripts.ops import approved_storage_recovery

    clock = [0.0]
    monkeypatch.setattr(snapshot.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        snapshot.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(
        snapshot.shutil, "disk_usage", lambda *args: SimpleNamespace(free=100)
    )
    monkeypatch.setattr(
        approved_storage_recovery, "resources_admitted", lambda *args: False
    )
    guard = snapshot.CopyGuard(tmp_path, tmp_path, 1, 10, operator_approved=True)
    with pytest.raises(RuntimeError, match="hard_resource_admission_withdrawn"):
        guard._wait_for_recovery_resources()
    assert clock[0] == 10
    guard.deadline = 100
    (tmp_path / "OPERATOR_STOP.flag").touch()
    with pytest.raises(RuntimeError, match="maintenance_or_operator_hold"):
        guard._wait_for_recovery_resources()
    assert clock[0] == 10
