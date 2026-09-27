import argparse
import fcntl
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.ops import maintenance_slot_guard as guard


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for key, path in (
        ("PROJECT_ROOT", tmp_path),
        ("RUNTIME_ROOT", tmp_path / "runtime"),
        ("LOCK_ROOT", tmp_path / "runtime/locks"),
        ("STATE_ROOT", tmp_path / "runtime/state"),
        ("HEALTH_PATH", tmp_path / "health.json"),
        ("RUNTIME_THROTTLE_HEALTH_PATH", tmp_path / "runtime.json"),
    ):
        monkeypatch.setattr(guard, key, path)
    monkeypatch.setattr(guard, "_load_macro_status", lambda: {})
    monkeypatch.setattr(guard, "_process_running", lambda needles: False)
    monkeypatch.setattr(
        guard, "maintenance_hold_snapshot", lambda root: {"active": False}
    )
    monkeypatch.setattr(guard.os, "cpu_count", lambda: 10)
    monkeypatch.setattr(guard.os, "getloadavg", lambda: (7, 7, 7))
    monkeypatch.setattr(
        guard.shutil, "disk_usage", lambda root: SimpleNamespace(free=100 * 1024**3)
    )
    # Do not enumerate unrelated host processes in tests; the real child owns this group.
    monkeypatch.setattr(guard, "_process_tree_targets", lambda pid: ({pid}, {pid}))
    return tmp_path


def args(slot="infrastructure_autofix", **changes):
    values = dict(
        slot=slot,
        execute=True,
        adaptive=True,
        max_load_ratio=0.72,
        max_five_min_load_ratio=0.62,
        max_one_min_load=0,
        min_interval_seconds=None,
        stale_seconds=1800,
        protect_macro_before_minutes=180,
        protect_macro_after_minutes=75,
        allow_during_macro_event=False,
        defer_while_sql_link_active=True,
        quiet_windows_enabled=False,
        defer_outside_quiet_window=False,
        quiet_start_hour=21,
        quiet_end_hour=6,
        smooth_gate_enabled=False,
        smooth_gate_max_saturation_score=68,
        smooth_gate_exempt_slots="",
        skip_exit_code=75,
        json=True,
        runtime_limit=5,
        terminate_grace=0.2,
        command=[sys.executable, "-c", "pass"],
    )
    values.update(changes)
    return argparse.Namespace(**values)


def publish(workload="maintenance", *, age=0, allowed=True):
    now = (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat()
    lease = dict(
        schema_version=1,
        timestamp_utc=now,
        source_timestamp_utc=now,
        input_evidence_ready=True,
        workloads={workload: {"admitted": allowed}},
    )
    payload = dict(
        source_timestamp_utc=now,
        input_evidence_ready=True,
        protective_hold=False,
        adaptive_safety_limits={"active": False},
        memory_pressure_level="normal",
        mac_fluidity_contract={"support_pause_recommended": False},
        workload_admission=lease,
    )
    guard.RUNTIME_THROTTLE_HEALTH_PATH.write_text(json.dumps(payload))


def test_load_failure_is_not_zero_pressure(sandbox, monkeypatch):
    monkeypatch.setattr(
        guard.os, "getloadavg", lambda: (_ for _ in ()).throw(OSError("unavailable"))
    )
    publish()
    assert guard._begin(args()) == 75
    payload = json.loads(guard.HEALTH_PATH.read_text())
    assert "load_evidence_unavailable" in payload["reasons"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_load_never_admits(sandbox, monkeypatch, value):
    monkeypatch.setattr(guard.os, "getloadavg", lambda: (value, 1, 1))
    assert guard._host_pressure(0.85, 0.85, None)[0]


def test_extra_headroom_requires_supervised_fresh_lease(sandbox):
    publish(age=91)
    assert guard._begin(args()) == 75
    publish()
    assert guard._begin(args(execute=False)) == 75
    assert guard._begin(args(adaptive=False)) == 75
    assert guard._begin(args()) == 0
    assert guard._end(args()) == 0


def test_only_observer_can_relax_quiet_hours(sandbox, monkeypatch):
    monkeypatch.setattr(guard, "_in_quiet_window", lambda *a: (False, {}))
    publish()
    assert (
        guard._begin(args(quiet_windows_enabled=True, defer_outside_quiet_window=True))
        == 75
    )
    publish("observer")
    assert (
        guard._begin(
            args(
                "infrastructure_observe",
                quiet_windows_enabled=True,
                defer_outside_quiet_window=True,
            )
        )
        == 0
    )


def test_explicit_interval_and_absolute_cpu_cap_stay_authoritative(sandbox):
    publish()
    guard.STATE_ROOT.mkdir(parents=True)
    guard._write_json(
        guard._state_path("infrastructure_autofix"),
        {"last_end_epoch": guard.time.time() - 400, "last_duration_seconds": 60},
    )
    assert guard._begin(args(min_interval_seconds=1800)) == 75
    assert guard._begin(args(max_one_min_load=6)) == 75
    assert guard._begin(args()) == 0
    receipt = json.loads(guard.HEALTH_PATH.read_text())
    assert receipt["cooldown"]["min_interval_seconds"] == 300


def test_live_or_unknown_lock_never_expires(sandbox, monkeypatch):
    path = sandbox / "old.lock"
    path.mkdir()
    monkeypatch.setattr(guard, "_lock_age_seconds", lambda path: 99999)
    assert not guard._reap_abandoned_lock(path, stale_seconds=60)
    guard._write_json(path / "owner.json", {"pid": os.getpid()})
    assert not guard._reap_abandoned_lock(path, stale_seconds=60)
    assert path.exists()


def test_kernel_lease_rejects_overlapping_execution(sandbox):
    guard.LOCK_ROOT.mkdir(parents=True)
    with (guard.LOCK_ROOT / "maintenance_bundle.flock").open("a+") as locked:
        fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert guard._execute(args()) == 75


def test_bounded_lease_wait_acquires_without_replacing_lock(sandbox):
    publish()
    guard.LOCK_ROOT.mkdir(parents=True)
    path = guard.LOCK_ROOT / "maintenance_bundle.flock"
    with path.open("a+") as locked:
        inode = path.stat().st_ino
        fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
        release = threading.Timer(0.05, fcntl.flock, (locked, fcntl.LOCK_UN))
        release.start()
        try:
            assert guard._execute(args(lease_wait_seconds=1)) == 0
        finally:
            release.join()
        assert path.stat().st_ino == inode
    receipt = json.loads(guard.HEALTH_PATH.read_text())
    assert receipt["lease_wait_seconds"] >= 0.05


def test_bounded_lease_wait_timeout_never_runs_admission(sandbox, monkeypatch):
    guard.LOCK_ROOT.mkdir(parents=True)
    monkeypatch.setattr(guard, "_begin", lambda args: pytest.fail("lease not owned"))
    with (guard.LOCK_ROOT / "maintenance_bundle.flock").open("a+") as locked:
        fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        assert guard._execute(args(lease_wait_seconds=0.05)) == 75
        assert time.monotonic() - started < 1


def test_lease_wait_rechecks_new_hold_before_child(sandbox, monkeypatch):
    publish()
    guard.LOCK_ROOT.mkdir(parents=True)
    hold = {"active": False}
    monkeypatch.setattr(guard, "maintenance_hold_snapshot", lambda root: hold)
    marker = sandbox / "must_not_run"
    with (guard.LOCK_ROOT / "maintenance_bundle.flock").open("a+") as locked:
        fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def release_with_new_hold():
            hold["active"] = True
            fcntl.flock(locked, fcntl.LOCK_UN)

        release = threading.Timer(0.05, release_with_new_hold)
        release.start()
        try:
            assert (
                guard._execute(
                    args(
                        lease_wait_seconds=1,
                        command=[
                            sys.executable,
                            "-c",
                            f"open({str(marker)!r}, 'w').close()",
                        ],
                    )
                )
                == 75
            )
        finally:
            release.join()
    assert not marker.exists()
    assert (
        "runtime_maintenance_hold"
        in json.loads(guard.HEALTH_PATH.read_text())["reasons"]
    )


@pytest.mark.parametrize("seconds", [-1, 121, float("inf"), float("nan")])
def test_lease_wait_rejects_unbounded_values(sandbox, seconds):
    with pytest.raises(ValueError, match="lease_wait_seconds"):
        guard._execute(args(lease_wait_seconds=seconds))


def test_real_child_completion_lease_owner_and_cleanup(sandbox):
    publish()
    owner_path = guard.LOCK_ROOT / "maintenance_bundle.lock/owner.json"
    code = f"import json; p=json.load(open({str(owner_path)!r})); assert p['pid']=={os.getpid()}"
    assert guard._execute(args(command=[sys.executable, "-c", code])) == 0
    receipt = json.loads(guard.HEALTH_PATH.read_text())
    assert receipt["result"] == "completed"
    assert receipt["adaptive_admission"]["active"]
    assert not (guard.LOCK_ROOT / "maintenance_bundle.lock").exists()
    assert (guard.LOCK_ROOT / "maintenance_bundle.flock").exists()


def test_real_child_timeout_is_not_completion(sandbox):
    publish()
    assert (
        guard._execute(
            args(
                runtime_limit=1,
                command=[sys.executable, "-c", "import time; time.sleep(60)"],
            )
        )
        == 124
    )
    receipt = json.loads(guard.HEALTH_PATH.read_text())
    assert receipt["result"] == "deadline_reached"
    assert not receipt["allowed"]
    assert not (guard.LOCK_ROOT / "maintenance_bundle.lock").exists()


@pytest.mark.parametrize("cause", ["expired", "disk", "load", "hold", "operator"])
def test_live_admission_revocation(sandbox, monkeypatch, cause):
    publish()
    config = args()
    config.adaptive_admission = guard._adaptive_slot_policy(config)
    assert guard._adaptive_continue(config)
    if cause == "expired":
        publish(age=91)
    elif cause == "disk":
        monkeypatch.setattr(
            guard.shutil, "disk_usage", lambda root: SimpleNamespace(free=63 * 1024**3)
        )
    elif cause == "load":
        monkeypatch.setattr(guard.os, "getloadavg", lambda: (10, 10, 10))
    elif cause == "hold":
        monkeypatch.setattr(
            guard, "maintenance_hold_snapshot", lambda root: {"active": True}
        )
    else:
        (sandbox / "OPERATOR_STOP.flag").touch()
    assert not guard._adaptive_continue(config)


def test_end_cannot_remove_another_slots_lock(sandbox):
    publish()
    assert guard._begin(args()) == 0
    assert guard._end(args("different_slot")) == 75
    assert (guard.LOCK_ROOT / "maintenance_bundle.lock").exists()


def test_observer_cannot_run_apply_command(sandbox):
    publish("observer")
    with pytest.raises(ValueError, match="fixed_assessment"):
        guard._execute(args("infrastructure_observe", command=["true", "--apply"]))


def test_observation_pool_does_not_wait_behind_unrelated_heavy_job(sandbox):
    publish("observer")
    heavy = guard.LOCK_ROOT / "maintenance_bundle.lock"
    heavy.mkdir(parents=True)
    guard._write_json(heavy / "owner.json", {"slot": "sqlite_maintenance", "pid": os.getpid()})
    config = args("infrastructure_observe")
    assert guard._begin(config) == 0
    assert (guard.LOCK_ROOT / "observer_bundle.lock").exists()
    assert guard._health_path(config.slot).name == "maintenance_observer_latest.json"
    assert guard._end(config) == 0
    assert heavy.exists()
