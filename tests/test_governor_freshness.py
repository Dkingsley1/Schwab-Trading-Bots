import fcntl
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import resource_guard
from scripts.ops import governor_refresh as refresh
from scripts.ops import memory_pressure_intelligence as intelligence
from scripts.ops import runtime_throttle_control as runtime
from scripts.ops import support_maintenance_gate as gate
from scripts.ops import training_runtime_control as training
from scripts.ops import whole_system_governor as whole
from scripts.ops.long_runtime_common import (
    evidence_freshness,
    governor_observation_contract,
    write_payload,
)


def stamp(seconds=0):
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


@pytest.fixture(autouse=True)
def clean_pause_environment(monkeypatch):
    for key in (
        "OPS_SUPPORT_MAINTENANCE_FREEZE",
        "MAC_FLUIDITY_SUPPORT_PAUSE",
        "SUPPORT_MAINTENANCE_CONCURRENCY",
        "RUNTIME_GOVERNOR_LEASE_TIMESTAMP_UTC",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("source", [None, "", "bad", 4000, -60])
def test_rewriting_a_derived_report_cannot_refresh_its_source(source):
    source = stamp(source) if isinstance(source, int) else source
    payload = {"timestamp_utc": stamp(), "source_timestamp_utc": source}
    assert evidence_freshness(payload, max_age_minutes=2)["fresh"] is False


def test_nested_source_ttl_is_enforced_after_parent_is_republished():
    now = datetime.now(timezone.utc)
    observation = governor_observation_contract(
        {
            "memory": ({"timestamp_utc": now.isoformat()}, 120),
            "storage": ({"timestamp_utc": now.isoformat()}, 900),
        },
        now=now,
    )
    payload = {"timestamp_utc": (now + timedelta(minutes=4)).isoformat(), **observation}
    assert not evidence_freshness(
        payload, max_age_minutes=15, now=now + timedelta(minutes=4)
    )["fresh"]


def test_touching_old_or_missing_evidence_never_refreshes_whole_governor(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("{}")
    os.utime(path, None)
    assert whole._payload_time(path, {"ok": True}, datetime.now(timezone.utc)) == (
        "",
        None,
    )
    assert whole._payload_time(
        path, {"timestamp_utc": stamp(-60)}, datetime.now(timezone.utc)
    ) == ("", None)
    _, age = whole._payload_time(
        path, {"timestamp_utc": stamp(7200)}, datetime.now(timezone.utc)
    )
    assert age >= 120


def test_whole_refresh_does_not_apply_or_change_registry(tmp_path, monkeypatch, capsys):
    registry = tmp_path / "master_bot_registry.json"
    registry.write_text('{"sub_bots": []}')
    before = registry.read_bytes()
    monkeypatch.setattr(
        whole, "apply_governor", lambda *_: pytest.fail("apply is forbidden")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "whole_system_governor.py",
            "--project-root",
            str(tmp_path),
            "--refresh",
            "--json",
        ],
    )
    assert whole.main() == 0
    assert registry.read_bytes() == before
    assert (tmp_path / "governance/health/whole_system_governor_latest.json").is_file()


def test_new_applied_release_supersedes_inherited_governor_pause(tmp_path, monkeypatch):
    old = stamp(60)
    override = tmp_path / "config/.env.runtime_resource_guard_override"
    override.parent.mkdir()
    override.write_text(
        f"RUNTIME_GOVERNOR_LEASE_TIMESTAMP_UTC={old}\nOPS_SUPPORT_MAINTENANCE_FREEZE=1\n"
    )
    monkeypatch.setenv("OPS_SUPPORT_MAINTENANCE_FREEZE", "1")
    monkeypatch.setenv("RUNTIME_GOVERNOR_LEASE_TIMESTAMP_UTC", old)
    write_payload(
        tmp_path / "governance/health/runtime_throttle_control_latest.json",
        {
            "timestamp_utc": stamp(),
            "mac_fluidity_contract": {"support_pause_recommended": False},
            "apply_result": {
                "applied": True,
                "support_maintenance_pause": {"pause_requested": False},
            },
        },
    )
    result = gate.support_maintenance_freeze_contract(tmp_path, "test")
    assert result["release_verified"] and not result["active"]


def test_explicit_operator_pause_is_not_auto_cleared(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_SUPPORT_MAINTENANCE_FREEZE", "1")
    write_payload(
        tmp_path / "governance/health/runtime_throttle_control_latest.json",
        {
            "timestamp_utc": stamp(),
            "apply_result": {
                "applied": True,
                "support_maintenance_pause": {"pause_requested": False},
            },
        },
    )
    assert gate.support_maintenance_freeze_contract(tmp_path, "test")["operator_freeze"]
    assert gate.support_maintenance_freeze_contract(tmp_path, "test")["active"]


@pytest.mark.parametrize("timestamp", [4000, -60, "bad", None])
def test_invalid_pause_reports_request_observation_not_fabricated_health(
    tmp_path, timestamp
):
    timestamp = stamp(timestamp) if isinstance(timestamp, int) else timestamp
    path = tmp_path / "governance/health/runtime_throttle_control_latest.json"
    write_payload(
        path,
        {
            "timestamp_utc": timestamp,
            "mac_fluidity_contract": {"support_pause_recommended": True},
        },
    )
    os.utime(path, None)
    contract = gate.support_maintenance_freeze_contract(tmp_path, "test")
    assert contract["active"] and contract["refresh_required"]
    assert not contract["fresh_runtime"]


def test_resource_sensor_still_observes_real_red_pressure_during_pause(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        resource_guard,
        "support_maintenance_freeze_contract",
        lambda *_: {"active": True, "reason": "support_pause"},
    )
    monkeypatch.setattr(
        resource_guard,
        "build_snapshot",
        lambda *_: {
            "timestamp_utc": stamp(),
            "memory_available_pct": 90,
            "memory_free_pct": 90,
            "swap_used_gb": 2,
            "pages_throttled": 3,
            "local_disk_free_gb": 100,
            "disk_free_gb": 100,
            "load1_per_core": 0.1,
            "editing_app_cpu_sum": 0,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resource_guard.py",
            "--project-root",
            str(tmp_path),
            "--profile",
            "refresh",
            "--json",
        ],
    )
    assert resource_guard.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["measurement_refreshed"] is True
    assert payload["support_maintenance_frozen"] is True
    assert payload["memory_pressure_state"] == "red"
    assert not payload["resource_guard_ok"]
    assert payload["pages_throttled"] == 3


def test_frozen_report_retains_earliest_observation(tmp_path):
    path = tmp_path / "prior.json"
    source = stamp(3600)
    write_payload(path, {"timestamp_utc": stamp(), "source_timestamp_utc": source})
    frozen = gate.frozen_health_payload(path, {"active": True})
    assert frozen["source_timestamp_utc"] == source
    assert not evidence_freshness(frozen, max_age_minutes=2)["fresh"]


def test_repeated_memory_snapshot_does_not_accrue_recovery_credit():
    source = stamp(1)
    snapshot = {"compressed_pressure_gb": 2, "swap_used_gb": 1, "pages_throttled": 0}
    previous = {
        "timestamp_utc": stamp(),
        "source_timestamp_utc": source,
        "snapshot": snapshot,
        "reopen_gate": {"consecutive_memory_clear_samples": 1},
    }
    trend = intelligence._memory_trend(previous, snapshot, source)
    result = intelligence._reopen_gate({"status": "clear"}, trend, snapshot, {}, {})
    assert result["consecutive_memory_clear_samples"] == 1
    assert not result["safe_to_widen_p_core_workers"]


@pytest.mark.parametrize("old_timestamp", [-30, 900, "bad"])
def test_invalid_memory_history_does_not_count_as_clear_soak(old_timestamp):
    old_timestamp = (
        stamp(old_timestamp) if isinstance(old_timestamp, int) else old_timestamp
    )
    previous = {
        "timestamp_utc": old_timestamp,
        "snapshot": {"swap_used_gb": 1},
        "reopen_gate": {"consecutive_memory_clear_samples": 99},
    }
    trend = intelligence._memory_trend(previous, {"swap_used_gb": 1}, stamp())
    assert trend["previous_clear_samples"] == 0


@pytest.mark.parametrize("age", [None, -60, 3600])
def test_training_rejects_missing_future_and_stale_green_resource(age, tmp_path):
    payload = {"resource_guard_ok": True, "memory_pressure_state": "green"}
    if age is not None:
        payload["timestamp_utc"] = stamp(age)
    result = training._build_resource_guard_training_gate(tmp_path, payload)
    assert not result["training_ok"]
    assert "resource_guard_evidence_requires_refresh" in result["launch_blockers"]


def test_training_missing_host_governors_is_not_implicit_permission(tmp_path):
    result = training._build_host_training_headroom_gate(
        project_root=tmp_path, memory_intelligence={}, autonomic_governor={}
    )
    assert not result["safe_for_training"]
    assert result["batch_cap"] == 0


@pytest.mark.parametrize(
    "name",
    [
        "governor_refresh",
        "memory_efficiency_control",
        "memory_pressure_intelligence",
        "autonomic_resource_governor",
        "swap_pressure_governor",
    ],
)
def test_fast_observers_are_not_support_pause_targets(name):
    row = runtime._classify_process(f"python scripts/ops/{name}.py --json")
    assert row["category"] == "operator_observability"
    assert not row["throttle_candidate"]


def test_native_refresh_is_dependency_ordered_and_single_flight(tmp_path, monkeypatch):
    calls = []

    def run(root, step, deadline):
        calls.append(step[0])
        return {
            "owner": step[0],
            "status": "complete",
            "attempted": True,
            "started_utc": stamp(),
        }

    monkeypatch.setattr(refresh, "_run_step", run)
    result = refresh.run_cycle(tmp_path)
    assert result["ok"]
    assert calls[: len(refresh.FAST_STEPS)] == [step[0] for step in refresh.FAST_STEPS]
    before = (tmp_path / "governance/health/governor_refresh_latest.json").read_bytes()
    with (tmp_path / "governance/health/governor_refresh.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert refresh.run_cycle(tmp_path)["reason"] == "owner_running"
    assert (
        tmp_path / "governance/health/governor_refresh_latest.json"
    ).read_bytes() == before
    calls.clear()
    refresh.run_cycle(tmp_path)
    assert calls == [step[0] for step in refresh.FAST_STEPS]


def test_failed_sensor_stops_dependent_refreshes(tmp_path, monkeypatch):
    calls = []

    def fail(root, step, deadline):
        calls.append(step[0])
        return {
            "owner": step[0],
            "status": "failed",
            "attempted": True,
            "started_utc": stamp(),
        }

    monkeypatch.setattr(refresh, "_run_step", fail)
    result = refresh.run_cycle(tmp_path)
    assert not result["ok"] and not result["fast_control_complete"]
    assert calls == ["resource_guard", "runtime_throttle_control"]
    assert result["protective_fallback"]["status"] == "failed"


def test_deadline_deferral_is_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(
        refresh,
        "_run_step",
        lambda root, step, deadline: {
            "owner": step[0],
            "status": "deferred",
            "attempted": False,
            "reason": "cycle_deadline",
        },
    )
    result = refresh.run_cycle(tmp_path)
    assert not result["ok"]
    assert result["unfinished_owners"] == ["resource_guard"]


def test_optional_failure_remains_debt_during_cooldown(tmp_path, monkeypatch):
    def run(root, step, deadline):
        return {
            "owner": step[0],
            "status": "failed" if step[0] == "whole_system_governor" else "complete",
            "attempted": True,
            "started_utc": stamp(),
        }

    monkeypatch.setattr(refresh, "_run_step", run)
    assert not refresh.run_cycle(tmp_path)["ok"]
    second = refresh.run_cycle(tmp_path)
    assert second["fast_control_complete"] and not second["ok"]
    assert second["unfinished_owners"] == ["whole_system_governor"]


def test_step_timeout_and_output_reuse_are_not_success(tmp_path, monkeypatch):
    write_payload(
        tmp_path / "governance/health/resource_guard_latest.json",
        {"timestamp_utc": stamp()},
    )
    seen = []

    def run(cmd, **kwargs):
        seen.append(kwargs)
        return {"rc": 124, "timed_out": True, "timeout_cleanup": {"reaped": True}}

    monkeypatch.setattr(refresh, "run_bounded_process_group", run)
    result = refresh._run_step(
        tmp_path, refresh.FAST_STEPS[0], refresh.time.monotonic() + 5
    )
    assert result["status"] == "failed" and result["timed_out"]
    assert seen[0]["timeout_seconds"] <= 3
    assert seen[0]["env"]["ALLOW_ORDER_EXECUTION"] == "0"
    assert seen[0]["env"]["MARKET_DATA_ONLY"] == "1"


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "unapplied",
        "unverified",
        "wrong_profile",
        "stale",
        "missing_input",
        "timeout",
        "old_publication",
        "crash",
    ],
)
def test_blocked_memory_decision_completes_only_after_verified_application(
    tmp_path, monkeypatch, failure
):
    def run(cmd, **kwargs):
        payload = {
            "timestamp_utc": stamp(300) if failure == "old_publication" else stamp(),
            "source_timestamp_utc": stamp(300) if failure == "stale" else stamp(),
            "input_evidence_ready": failure != "missing_input",
            "action": "apply",
            "overall_status": "blocked",
            "ok": False,
            "recommended_profile": "constrained",
            "apply_result": {
                "applied": failure != "unapplied",
                "override_verified": failure != "unverified",
                "profile": "air_safe" if failure == "wrong_profile" else "constrained",
            },
        }
        write_payload(
            tmp_path / "governance/health/memory_efficiency_control_latest.json",
            payload,
        )
        return {
            "rc": 1 if failure == "crash" else 2,
            "timed_out": failure == "timeout",
            "timeout_cleanup": {"reaped": True},
        }

    monkeypatch.setattr(refresh, "run_bounded_process_group", run)
    step = next(
        step for step in refresh.FAST_STEPS if step[0] == "memory_efficiency_control"
    )
    result = refresh._run_step(tmp_path, step, refresh.time.monotonic() + 10)
    assert result["status"] == ("complete" if failure is None else "failed")
    assert result["reported_status"] == "blocked"


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "timeout",
        "old_publication",
        "future",
        "missing_surfaces",
        "crash",
        "unknown_status",
    ],
)
@pytest.mark.parametrize("verdict", ["ready", "degraded", "blocked"])
def test_regression_observation_completes_without_clearing_reported_blockers(
    tmp_path, monkeypatch, failure, verdict
):
    def run(cmd, **kwargs):
        write_payload(
            tmp_path / "governance/health/grade_regression_guard_latest.json",
            {
                "timestamp_utc": (
                    stamp(300)
                    if failure == "old_publication"
                    else stamp(-60) if failure == "future" else stamp()
                ),
                "ok": verdict == "ready",
                "overall_status": (
                    "unknown" if failure == "unknown_status" else verdict
                ),
                "surfaces": (
                    []
                    if failure == "missing_surfaces"
                    else [{"surface": "storage_control", "state": verdict}]
                ),
            },
        )
        return {
            "rc": 1 if failure == "crash" else 2 if verdict == "blocked" else 0,
            "timed_out": failure == "timeout",
            "timeout_cleanup": {"reaped": True},
        }

    monkeypatch.setattr(refresh, "run_bounded_process_group", run)
    step = next(
        step for step in refresh.SLOW_STEPS if step[0] == "grade_regression_guard"
    )
    assert step[2] == ("--json",)
    assert step[3:] == (5, 300)
    result = refresh._run_step(tmp_path, step, refresh.time.monotonic() + 10)
    assert result["status"] == ("complete" if failure is None else "failed")
    assert not result["applied_control_verified"]
    if failure is None:
        assert result["regression_assessment_observed"]
        assert result["reported_status"] == verdict


def test_failed_atomic_publication_preserves_existing_override(tmp_path, monkeypatch):
    from scripts.ops import long_runtime_common as common

    path = tmp_path / "override"
    path.write_text("PAUSE=1\n")
    path.chmod(0o600)

    def fail(*_):
        raise OSError("replace failed")

    monkeypatch.setattr(common.os, "replace", fail)
    with pytest.raises(OSError):
        common.write_text_atomic(path, "PAUSE=0\n")
    assert path.read_text() == "PAUSE=1\n"
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [path]


def test_owner_launch_failure_is_terminal_evidence(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("cannot launch")

    monkeypatch.setattr(refresh, "run_bounded_process_group", fail)
    result = refresh.run_cycle(tmp_path)
    assert not result["ok"]
    assert result["steps"][0]["reason"] == "owner_launch_failed"
