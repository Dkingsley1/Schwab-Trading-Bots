from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.ops import soak_self_healing_control as src


def lease(root, *, age=0, admitted=True, reasons=None):
    now = datetime.now(timezone.utc)
    src.write_payload(
        root / "governance/health/runtime_throttle_control_latest.json",
        {
            "workload_admission": {
                "schema_version": 1,
                "timestamp_utc": now.isoformat(),
                "source_timestamp_utc": (now - timedelta(seconds=age)).isoformat(),
                "input_evidence_ready": True,
                "workloads": {
                    "storage_recovery": {"admitted": admitted, "reasons": reasons or []}
                },
            }
        },
    )


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    calls = []
    clock = [0]
    monkeypatch.setattr(src.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(src.os, "cpu_count", lambda: 10)
    monkeypatch.setattr(src.os, "getloadavg", lambda: (7, 7, 7))
    monkeypatch.setattr(
        src.shutil, "disk_usage", lambda _: SimpleNamespace(free=60 * 1024**3)
    )
    monkeypatch.setattr(src, "maintenance_hold_snapshot", lambda _: {"active": False})
    monkeypatch.setattr(
        src,
        "_configure_cold_archive_env",
        lambda *a, **k: pytest.fail("leased recovery must remain local"),
    )
    lease(tmp_path)

    def runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return {
            "rc": 0,
            "ok": True,
            "timed_out": False,
            "parsed": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "ok": True,
                "memory_snapshot": {
                    "memory_pressure_state": "green",
                    "memory_free_pct": 70,
                    "swap_used_gb": 1,
                },
                "removed_allocated_bytes": 0,
                "saved_bytes": 0,
                "summary": {"estimated_reduction_bytes": 0},
            },
        }

    monkeypatch.setattr(src, "_run_command", runner)
    return calls, clock, runner


def test_leased_recovery_caps_batch_and_process_budgets(tmp_path, recovery):
    calls, _, _ = recovery
    payload = src.build_storage_recovery_payload(
        tmp_path, apply=True, rebuild_reserve=True
    )
    assert payload["shared_deadline_seconds"] == 240
    assert payload["load_admission"]["lane"] == "leased_compression"
    assert not payload["ok"] and not payload["live_execution_authority"]
    for cmd, kwargs in calls:
        assert kwargs["timeout_sec"] <= 65
        if "--seconds" in cmd:
            assert cmd[cmd.index("--seconds") + 1] == "60"
            assert cmd[cmd.index("--max-files") + 1] == "4"
            assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
    effect = payload["recovery_effectiveness"]
    assert effect["pressure_shortfall_gb"] == 4
    assert effect["trigger_shortfall_gb"] == 65
    assert effect["capacity_shortfall_gb"] == 75
    assert effect["outcome"] == "no_measured_progress_in_completed_pass"
    assert len(effect["owner_retry_not_before_utc"]) == 3


@pytest.mark.parametrize("age", [91, -10])
def test_stale_or_future_lease_explains_deferral(tmp_path, recovery, age):
    calls, _, _ = recovery
    lease(tmp_path, age=age)
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert not calls
    assert payload["reason"] == "storage_recovery_lease_not_ready"
    assert payload["load_admission"]["lease_reasons"] == [
        "storage_recovery_lease_missing_stale_or_invalid"
    ]
    assert payload["recovery_effectiveness"]["outcome"] == "deferred"


def test_denied_lease_preserves_specific_resource_reason(tmp_path, recovery):
    lease(tmp_path, admitted=False, reasons=["unknown_cpu_percent_pressure"])
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert payload["load_admission"]["lease_reasons"] == [
        "unknown_cpu_percent_pressure"
    ]


@pytest.mark.parametrize(
    "change", ["expired_lease", "denied_lease", "load_spike", "hold"]
)
def test_admission_is_rechecked_between_steps(tmp_path, monkeypatch, recovery, change):
    calls, _, runner = recovery

    def changing(cmd, **kwargs):
        result = runner(cmd, **kwargs)
        if "runtime-training-snapshot" in cmd:
            if change == "expired_lease":
                lease(tmp_path, age=100)
                monkeypatch.setattr(src.os, "getloadavg", lambda: (1, 1, 1))
            elif change == "denied_lease":
                lease(tmp_path, admitted=False)
            elif change == "load_spike":
                monkeypatch.setattr(src.os, "getloadavg", lambda: (9, 7, 7))
            else:
                monkeypatch.setattr(
                    src, "maintenance_hold_snapshot", lambda _: {"active": True}
                )
        return result

    monkeypatch.setattr(src, "_run_command", changing)
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert not any("--seconds" in cmd for cmd, _ in calls)
    assert payload["recovery_effectiveness"]["outcome"] == "deferred"
    assert "local-storage-reserve-guard" in calls[-1][0]


def test_full_lane_cannot_inherit_extra_load_authority_mid_pass(
    tmp_path, monkeypatch, recovery
):
    calls, _, runner = recovery
    monkeypatch.setattr(src.os, "getloadavg", lambda: (1, 1, 1))
    monkeypatch.setattr(src, "_configure_cold_archive_env", lambda *a, **k: {})

    def changing(cmd, **kwargs):
        result = runner(cmd, **kwargs)
        if "runtime-training-snapshot" in cmd:
            monkeypatch.setattr(src.os, "getloadavg", lambda: (7, 7, 7))
        return result

    monkeypatch.setattr(src, "_run_command", changing)
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert payload["reason"] == "host_load_above_full_recovery_budget"
    assert not any("cold-evidence-compactor" in cmd for cmd, _ in calls)


def test_memory_rechecked_after_sixty_seconds(tmp_path, monkeypatch, recovery):
    calls, clock, runner = recovery
    memory_calls = [0]

    def changing(cmd, **kwargs):
        result = runner(cmd, **kwargs)
        if any("memory_efficiency_control.py" in part for part in cmd):
            memory_calls[0] += 1
            if memory_calls[0] == 2:
                result["parsed"]["memory_snapshot"]["memory_pressure_state"] = "red"
        if "cold-evidence-compactor" in cmd:
            clock[0] += 61
        return result

    monkeypatch.setattr(src, "_run_command", changing)
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert memory_calls[0] == 2
    assert payload["reason"] == "memory_admission_not_ready"
    assert not any("governance-lifecycle-compactor" in cmd for cmd, _ in calls)


def test_short_remaining_window_defers_complete_compression(
    tmp_path, monkeypatch, recovery
):
    calls, clock, runner = recovery

    def changing(cmd, **kwargs):
        result = runner(cmd, **kwargs)
        if "runtime-training-snapshot" in cmd:
            clock[0] = 180
        return result

    monkeypatch.setattr(src, "_run_command", changing)
    payload = src.build_storage_recovery_payload(tmp_path, apply=True)
    assert not any("--seconds" in cmd for cmd, _ in calls)
    deferred = [step for step in payload["steps"] if step.get("deferred")]
    assert len(deferred) == 2
    assert all(
        step["reason"] == "insufficient_complete_compression_window"
        for step in deferred
    )
    assert payload["recovery_effectiveness"]["outcome"] == "deferred"


@pytest.mark.parametrize(
    "loads", [(9, 1, 1), (1, 9, 1), (float("nan"), 1, 1), (-1, 1, 1)]
)
def test_load_checks_both_recent_windows(tmp_path, recovery, monkeypatch, loads):
    monkeypatch.setattr(src.os, "getloadavg", lambda: loads)
    assert not src._storage_load_admission(tmp_path)["admitted"]


def test_reserved_runtime_report_is_not_read(tmp_path, recovery, monkeypatch):
    path = tmp_path / "governance/health/runtime_throttle_control_latest.json"
    path.unlink()
    path.symlink_to("/Volumes/VIDEO/report.json")
    monkeypatch.setattr(
        src, "load_json", lambda _: pytest.fail("protected report read")
    )
    assert not src._storage_load_admission(tmp_path)["admitted"]


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("progress", "progress_capacity_unmet"),
        ("failed", "repair_failed"),
        ("unknown", "reclamation_not_measured"),
    ],
)
def test_outcome_does_not_invent_recovery(mode, expected):
    step = {"name": "repair", "executed": True, "ok": mode != "failed"}
    if mode == "progress":
        step["storage_recovery_progress"] = {
            "made_progress": True,
            "measurement_present": True,
        }
    result = src._storage_recovery_outcome(
        [step],
        state={},
        free_after=60,
        pressure_threshold=64,
        trigger=125,
        target=135,
        admitted=True,
        reason="bounded_storage_recovery",
    )
    assert result["outcome"] == expected
    assert result["capacity_shortfall_gb"] == 75


def test_retry_preserves_later_failure_circuit():
    now = datetime.now(timezone.utc)
    circuit = (now + timedelta(hours=1)).isoformat()
    cooldown = (now + timedelta(minutes=5)).isoformat()
    result = src._storage_recovery_outcome(
        [{"name": "repair", "executed": False}],
        state={
            "steps": {
                "repair": {"cooldown_until_utc": cooldown, "circuit_until_utc": circuit}
            }
        },
        free_after=60,
        pressure_threshold=64,
        trigger=125,
        target=135,
        admitted=True,
        reason="bounded_storage_recovery",
    )
    assert result["owner_retry_not_before_utc"]["repair"] == circuit


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("storage_maintenance_lock_busy", "deferred"),
        ("insufficient_scratch_for_remaining_files", "deferred"),
        ("no_eligible_cold_logs", "no_measured_progress_in_completed_pass"),
        ("eligible_files_exhausted", "no_measured_progress_in_completed_pass"),
    ],
)
def test_owner_deferrals_are_not_reported_as_empty_completed_passes(reason, expected):
    result = {
        "ok": True,
        "parsed": {"overall_status": "deferred", "reason": reason, "saved_bytes": 0},
    }
    progress = src._storage_recovery_progress(
        "local_disk_cold_evidence_compaction", result
    )
    payload = src._storage_recovery_outcome(
        [
            {
                "name": "compaction",
                "executed": True,
                "ok": True,
                "storage_recovery_progress": progress,
            }
        ],
        state={},
        free_after=60,
        pressure_threshold=64,
        trigger=125,
        target=135,
        admitted=True,
        reason="bounded_storage_recovery",
    )
    assert payload["outcome"] == expected
