from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from compression import zstd

from scripts.ops import cold_evidence_compactor as src

REAL_IDLE = src.idle


class Guard:
    reserve = 16 * src.GIB

    def check(self):
        pass

    def progress(self, *args):
        pass


@pytest.fixture
def source(tmp_path, monkeypatch):
    p = tmp_path / src.ROOT_REL / "old_20260101.jsonl.local_fallback.2"
    p.parent.mkdir(parents=True)
    p.write_bytes(b'{"historical_evidence":true}\n' * 50000)
    os.utime(p, (1600000000, 1600000000))
    monkeypatch.setattr(src, "idle", lambda path: None)
    monkeypatch.setattr(
        src.shutil, "disk_usage", lambda root: SimpleNamespace(free=100 * src.GIB)
    )
    return p


def test_verified_round_trip_and_receipt_order(tmp_path, source):
    original = source.read_bytes()
    result = src.compact_one(tmp_path, source, src.identity(source), Guard())
    assert not source.exists()
    with zstd.open(Path(str(source) + ".zst"), "rb") as f:
        assert f.read() == original
    assert result["sha256_uncompressed"] == hashlib.sha256(original).hexdigest()
    rows = [
        json.loads(line)
        for line in (
            tmp_path / "governance/storage_recovery/cold_evidence_compression.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert [r["event"] for r in rows] == [
        "verified_before_release",
        "original_replaced",
    ]
    assert not list(source.parent.glob(".cold_compact_*"))


def test_divergent_existing_archive_preserves_both(tmp_path, source):
    target = Path(str(source) + ".zst")
    target.write_bytes(zstd.compress(b"different source"))
    before = target.read_bytes()
    with pytest.raises(RuntimeError, match="mismatch"):
        src.compact_one(tmp_path, source, src.identity(source), Guard())
    assert source.exists() and target.read_bytes() == before


def test_existing_verified_archive_resumes(tmp_path, source):
    target = Path(str(source) + ".zst")
    target.write_bytes(zstd.compress(source.read_bytes()))
    before = target.read_bytes()
    src.compact_one(tmp_path, source, src.identity(source), Guard())
    assert not source.exists() and target.read_bytes() == before


def test_failed_receipt_retains_original_and_verified_archive(
    tmp_path, source, monkeypatch
):
    def fail(*args):
        raise OSError("receipt storage failure")

    monkeypatch.setattr(src, "receipt", fail)
    with pytest.raises(OSError):
        src.compact_one(tmp_path, source, src.identity(source), Guard())
    assert source.exists() and Path(str(source) + ".zst").exists()


def test_changed_source_preserved(tmp_path, source, monkeypatch):
    digest = src.digest_stream

    def mutate(*args):
        result = digest(*args)
        with source.open("ab") as f:
            f.write(b"new evidence")
        return result

    monkeypatch.setattr(src, "digest_stream", mutate)
    with pytest.raises(RuntimeError, match="source_changed"):
        src.compact_one(tmp_path, source, src.identity(source), Guard())
    assert source.exists() and source.read_bytes().endswith(b"new evidence")


def test_interruption_removes_only_owned_temporary(tmp_path, source):
    class Interrupted(Guard):
        calls = 0

        def check(self):
            self.calls += 1
            if self.calls > 1:
                raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        src.compact_one(tmp_path, source, src.identity(source), Interrupted())
    assert source.exists() and not Path(str(source) + ".zst").exists()
    assert not list(source.parent.glob(".cold_compact_*"))


def test_scratch_reserve_cannot_be_consumed(tmp_path, source, monkeypatch):
    monkeypatch.setattr(
        src.shutil, "disk_usage", lambda root: SimpleNamespace(free=16 * src.GIB)
    )
    with pytest.raises(RuntimeError, match="scratch_reserve"):
        src.compact_one(tmp_path, source, src.identity(source), Guard())
    assert source.exists()


def test_open_source_preserved(tmp_path, source, monkeypatch):
    monkeypatch.setattr(
        src.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="123\n", stderr=""),
    )
    # Exercise the real idle probe rather than the fixture's no-op.
    monkeypatch.setattr(src, "idle", REAL_IDLE)
    with pytest.raises(RuntimeError, match="source_open"):
        src.compact_one(tmp_path, source, src.identity(source), Guard())
    assert source.exists()


def test_inventory_rejects_database_recent_symlink_and_small(tmp_path, source):
    for name in ("old.sqlite3", "recent.jsonl", "tiny.log", "old.jsonl.gz"):
        p = source.parent / name
        p.write_bytes(b"x" * (1 if name == "tiny.log" else 1048576))
        if name != "recent.jsonl":
            os.utime(p, (1600000000, 1600000000))
    (source.parent / "alias.jsonl").symlink_to(source)
    assert [p for p, _ in src.inventory(tmp_path)] == [source]


def test_database_content_is_rejected_even_with_log_filename(tmp_path, source):
    source.write_bytes(b"SQLite format 3\x00" + b"x" * 1048576)
    os.utime(source, (1600000000, 1600000000))
    assert src.inventory(tmp_path) == []
    with pytest.raises(RuntimeError, match="database_content"):
        src.compact_one(tmp_path, source, src.identity(source), Guard())
    assert source.exists()


def test_protected_alias_rejected_before_target_metadata(tmp_path):
    root = tmp_path / src.ROOT_REL
    root.parent.mkdir(parents=True)
    root.symlink_to("/Volumes/VIDEO/never-inspect", target_is_directory=True)
    with pytest.raises(RuntimeError, match="protected"):
        src.inventory(tmp_path)


def test_storage_lock_contention_is_a_deferral(tmp_path, source, monkeypatch):
    monkeypatch.setattr(
        src.fcntl, "flock", lambda *a: (_ for _ in ()).throw(BlockingIOError())
    )
    result = src.build_payload(tmp_path, apply=True, target_free_gb=125)
    assert result["reason"] == "storage_maintenance_lock_busy"
    assert source.exists()


def test_dry_run_and_sufficient_headroom_do_not_compress(tmp_path, source, monkeypatch):
    monkeypatch.setattr(
        src, "compact_one", lambda *a: pytest.fail("unexpected compression")
    )
    assert (
        src.build_payload(tmp_path, apply=False, target_free_gb=125)["candidate_count"]
        == 1
    )
    assert (
        src.build_payload(tmp_path, apply=True, target_free_gb=64)["reason"]
        == "headroom_sufficient"
    )


def test_small_files_bootstrap_space_before_large_files(tmp_path, source, monkeypatch):
    small = source.parent / "small.jsonl"
    small.write_bytes(b"test")
    small_id, large_id = (1, 1, src.GIB // 2, 0, 0), (1, 2, 2 * src.GIB, 0, 0)
    monkeypatch.setattr(
        src, "inventory", lambda root: [(source, large_id), (small, small_id)]
    )
    free = [17 * src.GIB]
    monkeypatch.setattr(
        src.shutil, "disk_usage", lambda root: SimpleNamespace(free=free[0])
    )
    monkeypatch.setattr(src, "Guard", lambda *a: Guard())
    monkeypatch.setattr(src, "background_policy", lambda: {})
    seen = []

    def compact(root, path, info, guard):
        seen.append(path)
        free[0] += 2 * src.GIB
        return {"saved_bytes": 2 * src.GIB}

    monkeypatch.setattr(src, "compact_one", compact)
    result = src.build_payload(tmp_path, apply=True, target_free_gb=20)
    assert seen == [small, source]
    assert result["overall_status"] == "ready"


def ready_observations():
    now = datetime.now(timezone.utc).isoformat()
    return (
        {
            "timestamp_utc": now,
            "input_evidence_ready": True,
            "storage_recovery_memory_observation": {
                "timestamp_utc": now,
                "input_evidence_ready": True,
            },
            "memory_snapshot": {
                "memory_free_pct": 92,
                "swap_used_gb": 7,
                "memory_pressure_state": "green",
            },
        },
        {
            "timestamp_utc": now,
            "runtime_snapshot": {
                "thermal": {
                    "measurement_available": True,
                    "thermal_warning_active": False,
                    "performance_warning_active": False,
                    "cpu_power_warning_active": False,
                }
            },
            "mac_fluidity_contract": {
                "measurements": {
                    "foreground_app_cpu_percent": 20,
                    "macos_system_cpu_percent": 20,
                }
            },
        },
    )


@pytest.mark.parametrize(
    "failure", ["stale", "thermal", "foreground", "memory", "hold"]
)
def test_resource_admission_fails_closed(tmp_path, monkeypatch, failure):
    memory, runtime = ready_observations()
    monkeypatch.setattr(
        src, "load_json", lambda p: memory if "memory_efficiency" in p.name else runtime
    )
    monkeypatch.setattr(
        src, "maintenance_hold_snapshot", lambda r: {"active": failure == "hold"}
    )
    monkeypatch.setattr(src.os, "getloadavg", lambda: (0, 0, 0))
    if failure == "stale":
        runtime["timestamp_utc"] = (
            datetime.now(timezone.utc) - timedelta(hours=1)
        ).isoformat()
    elif failure == "thermal":
        runtime["runtime_snapshot"]["thermal"]["thermal_warning_active"] = True
    elif failure == "foreground":
        runtime["mac_fluidity_contract"]["measurements"][
            "foreground_app_cpu_percent"
        ] = 100
    elif failure == "memory":
        memory["memory_snapshot"]["memory_pressure_state"] = "red"
    assert not src.admission(tmp_path)[0]


def test_current_green_resource_observation_admitted(tmp_path, monkeypatch):
    memory, runtime = ready_observations()
    monkeypatch.setattr(
        src, "load_json", lambda p: memory if "memory_efficiency" in p.name else runtime
    )
    monkeypatch.setattr(src, "maintenance_hold_snapshot", lambda r: {"active": False})
    monkeypatch.setattr(src.os, "getloadavg", lambda: (0, 0, 0))
    assert src.admission(tmp_path)[0]
    memory["storage_recovery_memory_observation"][
        "timestamp_utc"
    ] = "2020-01-01T00:00:00+00:00"
    assert not src.admission(tmp_path)[0]


def _relaxed_recovery_observations(tmp_path, monkeypatch):
    memory, runtime = ready_observations()
    memory["creative_session"] = {"active": False, "cooldown_active": False}
    runtime.update(
        source_timestamp_utc=runtime["timestamp_utc"],
        input_evidence_ready=True,
        runtime_measurement_evidence={"ready": True},
        protective_hold=False,
        host_saturation_score=55,
    )
    runtime["mac_fluidity_contract"].update(
        support_pause_recommended=False, fluidity_band="guarded_smooth"
    )
    runtime["mac_fluidity_contract"]["measurements"].update(
        foreground_app_cpu_percent=110, macos_system_cpu_percent=160
    )
    monkeypatch.setattr(
        src, "load_json", lambda p: memory if "memory_efficiency" in p.name else runtime
    )
    monkeypatch.setattr(src, "maintenance_hold_snapshot", lambda r: {"active": False})
    monkeypatch.setattr(src.os, "cpu_count", lambda: 10)
    monkeypatch.setattr(src.os, "getloadavg", lambda: (8, 8, 8))
    monkeypatch.setattr(
        src.shutil, "disk_usage", lambda r: SimpleNamespace(free=60 * src.GIB)
    )
    return memory, runtime


def test_fresh_workload_lease_allows_paced_recovery_under_aggregate_protect_band(
    tmp_path, monkeypatch
):
    _, runtime = _relaxed_recovery_observations(tmp_path, monkeypatch)
    runtime["adaptive_safety_limits"] = {"active": False}
    runtime["mac_fluidity_contract"]["fluidity_band"] = "protect"
    runtime["workload_admission"] = {
        "schema_version": 1,
        "input_evidence_ready": True,
        "timestamp_utc": runtime["timestamp_utc"],
        "source_timestamp_utc": runtime["source_timestamp_utc"],
        "cpu_capacity_percent": 1000,
        "workloads": {"storage_recovery": {"admitted": True}},
    }
    assert src.admission(tmp_path) == (
        True,
        "bounded_workload_storage_recovery_admitted",
    )
    runtime["workload_admission"]["source_timestamp_utc"] = "2000-01-01T00:00:00+00:00"
    assert not src.admission(tmp_path)[0]


def test_disk_recovery_admits_paced_worker_with_moderate_multicore_competition(
    tmp_path, monkeypatch
):
    _relaxed_recovery_observations(tmp_path, monkeypatch)
    assert src.admission(tmp_path) == (
        True,
        "bounded_disk_recovery_relaxed_cpu_admitted",
    )
    assert src.Guard(tmp_path, 10).pace.fraction == 0.25


@pytest.mark.parametrize(
    "field,value",
    [
        ("foreground_app_cpu_percent", 150),
        ("macos_system_cpu_percent", 200),
        ("foreground_app_cpu_percent", None),
        ("foreground_app_cpu_percent", "110"),
        ("foreground_app_cpu_percent", True),
        ("macos_system_cpu_percent", float("nan")),
        ("macos_system_cpu_percent", float("inf")),
        ("macos_system_cpu_percent", -1),
        ("macos_system_cpu_percent", 10**400),
    ],
)
def test_relaxed_cpu_limits_remain_typed_and_bounded(
    tmp_path, monkeypatch, field, value
):
    _, runtime = _relaxed_recovery_observations(tmp_path, monkeypatch)
    runtime["mac_fluidity_contract"]["measurements"][field] = value
    assert src.admission(tmp_path) == (False, "foreground_or_system_pressure")


def test_relaxed_combined_cpu_and_host_load_limits(tmp_path, monkeypatch):
    _, runtime = _relaxed_recovery_observations(tmp_path, monkeypatch)
    runtime["mac_fluidity_contract"]["measurements"].update(
        foreground_app_cpu_percent=140, macos_system_cpu_percent=160
    )
    assert src.admission(tmp_path) == (False, "combined_foreground_system_pressure")
    runtime["mac_fluidity_contract"]["measurements"]["foreground_app_cpu_percent"] = 110
    monkeypatch.setattr(src.os, "getloadavg", lambda: (9, 8.51, 8))
    assert src.admission(tmp_path) == (False, "host_load_above_recovery_budget")


@pytest.mark.parametrize(
    "failure",
    [
        "disk_recovered",
        "emergency_disk",
        "small_host",
        "stale_source",
        "future_source",
        "missing_source",
        "input_unknown",
        "sensor_unknown",
        "protective_hold",
        "creative_active",
        "creative_cooldown",
        "creative_unknown",
        "support_pause",
        "fluidity_unknown",
        "saturation",
        "saturation_unknown",
        "memory",
        "thermal",
        "hold",
    ],
)
def test_cpu_relief_cannot_override_missing_evidence_or_protection(
    tmp_path, monkeypatch, failure
):
    memory, runtime = _relaxed_recovery_observations(tmp_path, monkeypatch)
    if failure in {"disk_recovered", "emergency_disk"}:
        free = 125 if failure == "disk_recovered" else 15
        monkeypatch.setattr(
            src.shutil, "disk_usage", lambda r: SimpleNamespace(free=free * src.GIB)
        )
    elif failure == "small_host":
        monkeypatch.setattr(src.os, "cpu_count", lambda: 4)
    elif failure in {"stale_source", "future_source"}:
        delta = -91 if failure == "stale_source" else 10
        runtime["source_timestamp_utc"] = (
            datetime.now(timezone.utc) + timedelta(seconds=delta)
        ).isoformat()
    elif failure == "missing_source":
        runtime.pop("source_timestamp_utc")
    elif failure == "input_unknown":
        runtime["input_evidence_ready"] = None
    elif failure == "sensor_unknown":
        runtime["runtime_measurement_evidence"]["ready"] = "true"
    elif failure == "protective_hold":
        runtime["protective_hold"] = True
    elif failure == "creative_active":
        memory["creative_session"]["active"] = True
    elif failure == "creative_cooldown":
        memory["creative_session"]["cooldown_active"] = True
    elif failure == "creative_unknown":
        memory["creative_session"] = {}
    elif failure == "support_pause":
        runtime["mac_fluidity_contract"]["support_pause_recommended"] = True
    elif failure == "fluidity_unknown":
        runtime["mac_fluidity_contract"]["fluidity_band"] = "unknown"
    elif failure in {"saturation", "saturation_unknown"}:
        runtime["host_saturation_score"] = 71 if failure == "saturation" else None
    elif failure == "memory":
        memory["memory_snapshot"]["memory_pressure_state"] = "red"
    elif failure == "thermal":
        runtime["runtime_snapshot"]["thermal"]["performance_warning_active"] = True
    elif failure == "hold":
        monkeypatch.setattr(
            src, "maintenance_hold_snapshot", lambda r: {"active": True}
        )
    assert src.admission(tmp_path)[0] is False


def test_deferred_resources_are_not_storage_readiness(tmp_path, source, monkeypatch):
    class DeferredGuard(Guard):
        def check(self):
            raise src.Deferred("thermal_observation_not_clear")

    monkeypatch.setattr(src, "Guard", lambda *args: DeferredGuard())
    monkeypatch.setattr(src, "background_policy", lambda: {})
    result = src.build_payload(tmp_path, apply=True, target_free_gb=125)
    assert result["ok"] and result["assessment_complete"]
    assert not result["target_ready"] and result["overall_status"] == "deferred"
    assert result["completed_count"] == 0 and result["saved_bytes"] == 0
    assert source.exists()


@pytest.mark.parametrize("field", ["memory", "runtime", "thermal", "measurements"])
def test_malformed_resource_observations_defer(tmp_path, monkeypatch, field):
    memory, runtime = ready_observations()
    monkeypatch.setattr(src, "maintenance_hold_snapshot", lambda r: {"active": False})
    if field == "memory":
        memory["storage_recovery_memory_observation"] = None
    elif field == "runtime":
        runtime["runtime_snapshot"] = []
    elif field == "thermal":
        runtime["runtime_snapshot"]["thermal"] = None
    else:
        runtime["mac_fluidity_contract"]["measurements"] = None
    monkeypatch.setattr(
        src, "load_json", lambda p: memory if "memory_efficiency" in p.name else runtime
    )
    assert not src.admission(tmp_path)[0]


@pytest.mark.parametrize("replacement", [False, True])
def test_lost_storage_lock_anchor_stops_work(tmp_path, monkeypatch, replacement):
    from types import SimpleNamespace

    monkeypatch.setattr(
        src.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=0)
    )
    lock = tmp_path / "storage.lock"
    lock.touch()
    guard = src.Guard(tmp_path, 10)
    guard.lock_anchor = (lock, src.identity(lock)[:2])
    monkeypatch.setattr(guard.pace, "tick", lambda: None)
    lock.rename(tmp_path / "retained-original.lock")
    if replacement:
        lock.touch()
    with pytest.raises(src.Deferred, match="storage_maintenance_lock_anchor"):
        guard.check()
