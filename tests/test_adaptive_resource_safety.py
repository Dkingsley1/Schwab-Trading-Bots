import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from core.adaptive_resource_safety import adaptive_headroom_guard
from scripts import resource_guard as resource
from scripts.ops import governor_refresh as refresh
from scripts.ops import runtime_throttle_control as runtime
from scripts.ops import memory_pressure_intelligence as memory
from scripts.ops.long_runtime_common import write_payload


def observation(now, **values):
    return {
        "timestamp_utc": now.isoformat(),
        "source_timestamp_utc": now.isoformat(),
        "input_evidence_ready": True,
        "memory_free_pct": 90.0,
        "memory_available_pct": 90.0,
        "local_disk_free_gb": 100.0,
        "disk_free_gb": 100.0,
        "load1_per_core": 0.3,
        "swap_used_gb": 0.0,
        "compressor_gb": 0.2,
        "compressed_store_gb": 1.0,
        "pages_throttled": 0,
        "memory_pressure_state": "green",
        "memory_pressure_kind": "none",
        "memory_pressure_thresholds": {
            "yellow_available_pct": 50.0,
            "yellow_free_pct": 8.0,
            "yellow_local_disk_gb": 32.0,
        },
        **values,
    }


def test_falling_headroom_tightens_before_existing_warning_limit():
    now = datetime.now(timezone.utc)
    prior = adaptive_headroom_guard(
        observation(now, memory_available_pct=70), {}, now=now
    )
    now += timedelta(seconds=20)
    result = adaptive_headroom_guard(
        observation(now, memory_available_pct=60), prior, now=now
    )
    assert result["state"] == "guarded"
    assert result["minimum_memory_pressure_level"] == "elevated"
    assert result["metrics"]["memory_available_pct"]["projected_40_seconds"] == 40
    assert result["metrics"]["memory_available_pct"]["warning_floor"] == 50
    assert result["authority"] == "tighten_existing_resource_controls_only"


def test_disk_reserve_is_predicted_without_inspecting_any_path():
    now = datetime.now(timezone.utc)
    prior = adaptive_headroom_guard(
        observation(now, local_disk_free_gb=55), {}, now=now
    )
    now += timedelta(seconds=20)
    result = adaptive_headroom_guard(
        observation(now, local_disk_free_gb=40), prior, now=now
    )
    assert result["active"]
    assert result["reasons"] == ["local_disk_free_gb:falling_headroom"]


@pytest.mark.parametrize("seconds", [0, 1, 121, -1])
def test_invalid_rate_windows_cannot_invent_a_headroom_forecast(seconds):
    now = datetime.now(timezone.utc)
    prior = adaptive_headroom_guard(
        observation(now, memory_available_pct=70), {}, now=now
    )
    current = now + timedelta(seconds=seconds)
    result = adaptive_headroom_guard(
        observation(current, memory_available_pct=60), prior, now=current
    )
    assert not result["active"]
    assert not result["metrics"]["memory_available_pct"]["forecast_crossing"]


def test_recovery_requires_a_full_minute_of_fresh_clear_observations():
    now = datetime.now(timezone.utc)
    prior = adaptive_headroom_guard(
        observation(now, memory_available_pct=40), {}, now=now
    )
    assert prior["active"]
    expected = ["recovering", "recovering", "recovering", "clear"]
    for seconds, state in zip((20, 40, 60, 80), expected):
        sample_time = now + timedelta(seconds=seconds)
        prior = adaptive_headroom_guard(
            observation(sample_time), prior, now=sample_time
        )
        assert prior["state"] == state


def test_rewriting_the_same_observation_cannot_expire_a_protective_hold():
    now = datetime.now(timezone.utc)
    prior = adaptive_headroom_guard(
        observation(now, memory_available_pct=40), {}, now=now
    )
    clear_time = now + timedelta(seconds=20)
    prior = adaptive_headroom_guard(observation(clear_time), prior, now=clear_time)
    result = adaptive_headroom_guard(
        observation(clear_time), prior, now=clear_time + timedelta(seconds=90)
    )
    assert result["state"] == "recovering"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("memory_available_pct", float("nan")),
        ("memory_free_pct", float("inf")),
        ("memory_free_pct", 101),
        ("local_disk_free_gb", -1),
        ("memory_free_pct", True),
    ],
)
def test_invalid_numbers_never_grant_headroom(field, bad):
    now = datetime.now(timezone.utc)
    current = observation(now, **{field: bad})
    result = adaptive_headroom_guard(current, {}, now=now)
    assert not result["input_evidence_ready"]
    assert result["minimum_memory_pressure_level"] == "high"
    assert not resource._measurement_evidence(current)["ready"]


@pytest.mark.parametrize("offset", [-121, 1])
def test_stale_and_future_sources_require_protection(offset):
    now = datetime.now(timezone.utc)
    result = adaptive_headroom_guard(
        observation(now + timedelta(seconds=offset)), {}, now=now
    )
    assert result["state"] == "unavailable"
    assert result["active"]


def test_failed_sensor_recovery_does_not_immediately_reopen_capacity():
    now = datetime.now(timezone.utc)
    prior = adaptive_headroom_guard(
        observation(now, input_evidence_ready=False), {}, now=now
    )
    now += timedelta(seconds=20)
    result = adaptive_headroom_guard(observation(now), prior, now=now)
    assert result["state"] == "recovering"
    assert result["minimum_memory_pressure_level"] == "elevated"


@pytest.mark.parametrize(
    "probe", ["_parse_memory_pressure", "_parse_swap_usage", "_parse_vm_stat"]
)
def test_resource_probes_are_bounded_and_failed_output_is_not_evidence(
    monkeypatch, probe
):
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            returncode=1, stdout="System-wide memory free percentage: 99%"
        )

    monkeypatch.setattr(resource.subprocess, "run", run)
    assert getattr(resource, probe)() == {}
    assert calls[0]["timeout"] == 1


@pytest.mark.parametrize(
    "probe,text",
    [
        ("_parse_memory_pressure", "System-wide memory free percentage: nan%"),
        ("_parse_swap_usage", "total = 10G used = nanG free = 10G"),
        (
            "_parse_vm_stat",
            "Pages throttled: -1.\nPages occupied by compressor: inf.\n",
        ),
    ],
)
def test_malformed_probe_counters_are_never_published_as_numbers(
    monkeypatch, probe, text
):
    monkeypatch.setattr(
        resource.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=text),
    )
    assert getattr(resource, probe)() == {}


def test_partial_hardware_sample_blocks_every_admission_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(
        resource,
        "_storage_disk_snapshot",
        lambda root: {"disk_free_gb": 100, "local_disk_free_gb": 100},
    )
    monkeypatch.setattr(resource, "_parse_memory_pressure", lambda: {})
    monkeypatch.setattr(resource, "_parse_swap_usage", lambda: {"swap_used_gb": 0})
    monkeypatch.setattr(
        resource,
        "_parse_vm_stat",
        lambda: {"pages_throttled": 0, "compressor_gb": 0, "compressed_store_gb": 0},
    )
    monkeypatch.setattr(resource, "_creative_apps_snapshot", lambda: {})
    monkeypatch.setattr(resource, "_co_running_apps_snapshot", lambda: {})
    snapshot = resource.build_snapshot(tmp_path)
    assert not snapshot["input_evidence_ready"]
    assert resource._memory_pressure_state(snapshot)[0] == "red"
    assert not resource.evaluate(
        snapshot,
        max_load_per_core=100,
        min_disk_gb=0,
        min_memory_free_pct=0,
        max_editing_cpu=1000,
    )[0]
    assert not resource.evaluate_optional_job(snapshot)[0]
    assert not resource.evaluate_refresh_job(snapshot)[0]


def test_runtime_probe_timeout_and_empty_sensor_output_fail_closed(monkeypatch):
    calls = []

    def timeout(cmd, **kwargs):
        calls.append(kwargs)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(runtime.subprocess, "run", timeout)
    assert runtime._run_capture(["vm_stat"]) == ""
    monkeypatch.setattr(runtime, "_run_capture", lambda cmd: "")
    snapshot = runtime.collect_runtime_snapshot()
    assert not snapshot["measurement_evidence"]["ready"]
    assert set(snapshot["measurement_evidence"]["missing_sensors"]) == {
        "thermal",
        "vm_stat",
        "process_cpu",
    }
    assert calls[0]["timeout"] == 1


def test_protective_fallback_avoids_repeating_failed_hardware_probes(monkeypatch):
    commands = []
    monkeypatch.setattr(
        runtime, "_run_capture", lambda command: commands.append(command) or ""
    )
    result = runtime._protective_runtime_snapshot()
    assert commands == [["ps", "-axo", "pid,ni,pcpu,pmem,etime,command"]]
    assert not result["measurement_evidence"]["ready"]
    assert result["measurement_evidence"]["mode"] == "protective_process_census_only"


@pytest.mark.parametrize("force,failed_sensor", [(True, False), (False, True)])
def test_failed_control_or_sensor_forces_protective_runtime_budget(
    tmp_path, force, failed_sensor
):
    now = datetime.now(timezone.utc)
    write_payload(
        tmp_path / "governance/health/resource_guard_latest.json", observation(now)
    )
    snapshot = {
        "cpu_count": 10,
        "load_averages": {},
        "thermal": {},
        "vm_stat": {},
        "top_processes": [],
        "category_cpu": {},
        "measurement_evidence": {"ready": not failed_sensor},
    }
    result = runtime.build_payload(
        tmp_path, runtime_snapshot=snapshot, protective_hold=force
    )
    assert not result["input_evidence_ready"]
    assert result["observed_memory_pressure_level"] == "normal"
    assert result["memory_pressure_level"] == "high"
    assert result["throttle_profile"] == "protect_live"
    assert result["adaptive_safety_limits"]["state"] == "protective_hold"
    assert (
        result["mac_fluidity_contract"]["env_overrides"][
            "TRAINING_RUNTIME_MAX_PARALLEL"
        ]
        == "0"
    )


def test_forecast_reaches_the_runtime_limits_without_altering_raw_pressure(tmp_path):
    now = datetime.now(timezone.utc)
    previous_time = now - timedelta(seconds=20)
    prior = adaptive_headroom_guard(
        observation(previous_time, memory_available_pct=70), {}, now=previous_time
    )
    health = tmp_path / "governance/health"
    write_payload(
        health / "runtime_throttle_control_latest.json",
        {"adaptive_safety_limits": prior},
    )
    write_payload(
        health / "resource_guard_latest.json", observation(now, memory_available_pct=60)
    )
    result = runtime.build_payload(tmp_path, runtime_snapshot={"cpu_count": 10})
    assert result["observed_memory_pressure_level"] == "normal"
    assert result["memory_pressure_level"] == "elevated"
    assert result["adaptive_safety_limits"]["active"]


def test_allocation_relief_cannot_cancel_adaptive_headroom_safety(tmp_path):
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance/health"
    write_payload(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "memory_pressure_level": "elevated",
            "adaptive_safety_limits": {"active": True},
        },
    )
    write_payload(
        health / "memory_efficiency_control_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "memory_snapshot": {
                "memory_free_pct": 95,
                "swap_used_gb": 8,
                "compressor_gb": 0.2,
                "compressed_store_gb": 20,
            },
        },
    )
    write_payload(
        health / "swap_pressure_governor_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "swap_pressure": {
                "tier": "normal",
                "memory_pressure_state": "green",
                "memory_pressure_kind": "normal",
                "swap_used_gb": 8,
            },
        },
    )
    result = memory.build_payload(tmp_path)
    assert result["snapshot"]["adaptive_headroom_guard_active"]
    assert result["snapshot"]["pressure_level"] == "elevated"
    assert not result["reopen_gate"]["safe_to_widen_p_core_workers"]
    assert result["reopen_gate"]["training_batch_cap"] == 0


def test_thermal_warning_propagates_a_hold_and_requires_recovery(tmp_path):
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance/health"
    write_payload(health / "resource_guard_latest.json", observation(now))
    result = runtime.build_payload(
        tmp_path,
        runtime_snapshot={"cpu_count": 10, "thermal": {"thermal_warning_active": True}},
    )
    assert result["observed_memory_pressure_level"] == "normal"
    assert result["memory_pressure_level"] == "high"
    assert result["adaptive_safety_limits"]["active"]
    later = now + timedelta(seconds=20)
    recovering = adaptive_headroom_guard(
        observation(later), result["adaptive_safety_limits"], now=later
    )
    assert recovering["state"] == "recovering"


def test_failed_dependency_runs_only_the_protective_fallback(tmp_path, monkeypatch):
    calls = []
    deadlines = []

    def run(root, step, deadline):
        calls.append(step)
        deadlines.append(deadline)
        return {
            "owner": step[0],
            "status": "complete" if "--protective-hold" in step[2] else "failed",
            "attempted": True,
        }

    monkeypatch.setattr(refresh, "_run_step", run)
    result = refresh.run_cycle(tmp_path)
    assert calls == [refresh.FAST_STEPS[0], refresh.FAIL_SAFE_STEP]
    assert not result["fast_control_complete"] and not result["ok"]
    assert result["protective_fallback"]["status"] == "complete"
    assert deadlines[1] - deadlines[0] == refresh.FAIL_SAFE_RESERVE_SECONDS


@pytest.mark.parametrize("applied", [True, False])
def test_fallback_receipt_requires_an_applied_protective_decision(
    tmp_path, monkeypatch, applied
):
    def run(cmd, **kwargs):
        write_payload(
            tmp_path / "governance/health/runtime_throttle_control_latest.json",
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "protective_hold": True,
                "apply_result": {"applied": applied},
                "input_evidence_ready": False,
            },
        )
        assert "--protective-hold" in cmd
        assert kwargs["env"]["ALLOW_ORDER_EXECUTION"] == "0"
        return {"rc": 2, "timed_out": False, "timeout_cleanup": {"reaped": True}}

    monkeypatch.setattr(refresh, "run_bounded_process_group", run)
    result = refresh._run_step(
        tmp_path, refresh.FAIL_SAFE_STEP, refresh.time.monotonic() + 8
    )
    assert (result["status"] == "complete") is applied


def test_applied_pressure_decision_is_not_mistaken_for_a_failed_refresh(
    tmp_path, monkeypatch
):
    def run(cmd, **kwargs):
        write_payload(
            tmp_path / "governance/health/runtime_throttle_control_latest.json",
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "overall_status": "blocked",
                "apply_result": {"applied": True},
            },
        )
        return {"rc": 2, "timed_out": False, "timeout_cleanup": {"reaped": True}}

    monkeypatch.setattr(refresh, "run_bounded_process_group", run)
    result = refresh._run_step(
        tmp_path, refresh.FAST_STEPS[2], refresh.time.monotonic() + 8
    )
    assert result["status"] == "complete"
    assert result["reported_status"] == "blocked"
