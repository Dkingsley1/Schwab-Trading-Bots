from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path

import pytest

from core.operations_master import ACTIONS, GROUPS, RESPONSIBILITIES, build_directions
from scripts.ops.operations_master import controls, dispatch, local


def ready_controls():
    return dict(
        fresh=True,
        operator_hold=False,
        storage_pressure=False,
        maintenance_admitted=True,
        storage_recovery_admitted=True,
    )


def checks():
    return [
        dict(name=name, status="ready")
        for _, _, _, names, _ in GROUPS
        for name in names
    ]


def row(plan, group):
    return next(r for r in plan["directives"] if r["subgroup"] == group)


def write(root, name, value):
    path = local(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_responsibilities_ownership_and_no_authority():
    plan = build_directions(checks(), ready_controls())
    assert len(RESPONSIBILITIES) == 14
    assert plan["status"] == "ready"
    assert not plan["unmapped_checks"]
    assert len({r["subgroup"] for r in plan["directives"]}) == 8
    assert all(r["directive"] == "observe" for r in plan["directives"])
    assert all(not value for value in plan["authority"].values())
    assert not plan["escalations"]
    assert len({r["success_requires"] for r in plan["directives"]}) == 8
    owned = [name for _, _, _, names, _ in GROUPS for name in names]
    assert len(owned) == len(set(owned))


def test_storage_recovery_not_deadlocked_on_maintenance_admission():
    c = ready_controls() | dict(storage_pressure=True, maintenance_admitted=False)
    plan = build_directions([], c)
    assert row(plan, "storage")["directive"] == "delegate"
    assert row(plan, "ingestion")["directive"] == "defer"
    assert row(plan, "evidence")["directive"] == "defer"
    assert row(plan, "commands")["directive"] == "delegate"


def test_holds_unknown_controls_and_provider_cooldown():
    plan = build_directions([], {"operator_hold": True})
    assert plan["status"] == "held"
    assert not any(r["command"] for r in plan["directives"])
    plan = build_directions(checks(), ready_controls() | {"provider_cooldown": True})
    assert plan["status"] == "needs_attention"
    assert plan["escalations"][0]["needs"] == ["provider_cooldown_active"]
    assert row(plan, "provider")["directive"] == "owner_followup"
    assert row(plan, "provider")["command"] is None
    assert (
        "unmapped"
        in build_directions([{"name": "unmapped"}], ready_controls())["unmapped_checks"]
    )


def test_fresh_controls_and_independent_storage_lease(tmp_path):
    now = datetime.now(timezone.utc)
    lease = dict(
        schema_version=1,
        input_evidence_ready=True,
        timestamp_utc=now.isoformat(),
        source_timestamp_utc=now.isoformat(),
        workloads={"storage_recovery": {"admitted": True}},
    )
    write(
        tmp_path,
        "runtime_throttle_control_latest.json",
        dict(timestamp_utc=now.isoformat(), workload_admission=lease),
    )
    write(
        tmp_path,
        "local_storage_reserve_guard_latest.json",
        dict(
            timestamp_utc=now.isoformat(),
            local_storage_reserve={
                "timestamp_utc": now.isoformat(),
                "pressure_active": True,
                "disk": {"known": True},
            },
        ),
    )
    c = controls(tmp_path, now=now)
    assert c["fresh"] and c["storage_recovery_admitted"]
    assert not c["maintenance_admitted"]
    assert c["storage_pressure"]
    assert not controls(tmp_path, now=now + timedelta(seconds=181))["fresh"]
    assert not controls(tmp_path, now=now - timedelta(seconds=1))["fresh"]
    storage = local(tmp_path, "local_storage_reserve_guard_latest.json")
    payload = json.loads(storage.read_text())
    payload["local_storage_reserve"]["timestamp_utc"] = (
        now - timedelta(seconds=181)
    ).isoformat()
    write(tmp_path, storage.name, payload)
    assert not controls(tmp_path, now=now)["fresh"]


def test_protected_control_route_not_followed(tmp_path, monkeypatch):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    (health / "RUNTIME_MAINTENANCE_HOLD.flag").symlink_to("/Volumes/VIDEO/no-touch")
    original = Path.lstat

    def guarded(path, *a, **kw):
        assert not str(path).startswith("/Volumes/VIDEO")
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, "lstat", guarded)
    assert controls(tmp_path)["operator_hold"]


def test_dispatch_cap_cooldown_and_persist_before_child(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.ops.operations_master.controls", lambda *a, **kw: ready_controls()
    )
    plan = build_directions([], ready_controls())
    calls = []

    def runner(cmd, **kw):
        persisted = json.loads(
            local(tmp_path, "operations_master_dispatch_state.json").read_text()
        )
        assert persisted["last_attempt_epoch"]
        record = persisted["recent_dispatch_history"][-1]
        assert record["phase"] == "started" and record["issues_before"]
        assert record["command"] == cmd[1:]
        calls.append(cmd)
        assert kw["timeout_sec"] <= 100
        return dict(cmd=cmd, rc=2, timed_out=False)

    attempts, deferred = dispatch(tmp_path, plan, runner=runner, timeout_sec=900)
    assert len(attempts) == 2
    assert all(not r["completion_credit"] for r in attempts)
    assert any(r["reason"] == "shared_dispatch_budget" for r in deferred)
    repeated, deferred = dispatch(tmp_path, plan, runner=runner, timeout_sec=900)
    assert not any(r["subgroup"] in {a["subgroup"] for a in attempts} for r in repeated)
    assert any(r["reason"] == "owner_cooldown_or_invalid_clock" for r in deferred)


def test_dispatch_rereads_holds_and_distrusts_commands(tmp_path, monkeypatch):
    plan = build_directions([], ready_controls())
    monkeypatch.setattr(
        "scripts.ops.operations_master.controls",
        lambda *a, **kw: {"operator_hold": True},
    )
    calls = []
    attempts, deferred = dispatch(
        tmp_path, plan, runner=lambda *a, **kw: calls.append(a)
    )
    assert not attempts and not calls
    assert any(r["reason"] == "current_owner_admission_not_ready" for r in deferred)
    row(plan, "runtime")["command"] = ["supervised-broker-test", "submit"]
    assert any(
        r["reason"] == "command_not_allowlisted"
        for r in dispatch(tmp_path, plan, runner=lambda *a, **kw: None)[1]
    )


def test_dispatch_lock_and_corrupt_state_fail_closed(tmp_path):
    plan = build_directions([], ready_controls())
    lock = local(tmp_path, "operations_master.lock")
    lock.parent.mkdir(parents=True)
    with lock.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert dispatch(tmp_path, plan, runner=lambda *a, **kw: pytest.fail("busy"))[
            1
        ] == [{"reason": "operations_master_busy"}]
    local(tmp_path, "operations_master_dispatch_state.json").write_text("{bad")
    attempts, deferred = dispatch(
        tmp_path, plan, runner=lambda *a, **kw: pytest.fail("corrupt")
    )
    assert not attempts
    assert deferred[-1]["reason"] == "unsafe_or_invalid_operations_dispatch_state"


def test_failed_child_consumes_cooldown(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.ops.operations_master.controls", lambda *a, **kw: ready_controls()
    )
    plan = build_directions(
        checks(), ready_controls() | {"maintenance_admitted": False}
    )

    def failed(*a, **kw):
        raise OSError("failed")

    attempts, _ = dispatch(tmp_path, plan, runner=failed)
    assert len(attempts) == 1 and attempts[0]["rc"] == 127
    assert not dispatch(tmp_path, plan, runner=failed)[0]


def test_dispatch_reserves_full_child_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.ops.operations_master.controls", lambda *a, **kw: ready_controls()
    )
    plan = build_directions(
        checks(), ready_controls() | {"maintenance_admitted": False}
    )
    attempts, deferred = dispatch(
        tmp_path, plan, timeout_sec=44, runner=lambda *a, **kw: pytest.fail("budget")
    )
    assert not attempts
    assert deferred == [{"subgroup": "runtime", "reason": "shared_dispatch_budget"}]


@pytest.mark.parametrize("retry_clock", ["bad", float("nan"), float("inf"), 1e20])
def test_invalid_retry_clock_never_launches(tmp_path, monkeypatch, retry_clock):
    monkeypatch.setattr(
        "scripts.ops.operations_master.controls", lambda *a, **kw: ready_controls()
    )
    write(
        tmp_path,
        "operations_master_dispatch_state.json",
        {"last_attempt_epoch": {"runtime": retry_clock}},
    )
    plan = build_directions(
        checks(), ready_controls() | {"maintenance_admitted": False}
    )
    attempts, deferred = dispatch(
        tmp_path, plan, runner=lambda *a, **kw: pytest.fail("invalid clock")
    )
    assert not attempts and deferred


def test_history_stays_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.ops.operations_master.controls", lambda *a, **kw: ready_controls()
    )
    write(
        tmp_path,
        "operations_master_dispatch_state.json",
        {"recent_dispatch_history": [{"phase": "old"}] * 64},
    )
    plan = build_directions(
        checks(), ready_controls() | {"maintenance_admitted": False}
    )
    dispatch(tmp_path, plan, runner=lambda *a, **kw: {"rc": 0})
    state = json.loads(
        local(tmp_path, "operations_master_dispatch_state.json").read_text()
    )
    assert len(state["recent_dispatch_history"]) == 64
    assert state["recent_dispatch_history"][-1]["phase"] == "finished"
    assert not state["recent_dispatch_history"][-1]["completion_credit"]
