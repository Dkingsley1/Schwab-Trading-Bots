from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from core.workload_admission import POLICIES, build_admission, current_lease

NOW = datetime(2026, 9, 14, 15, tzinfo=timezone.utc)


def observations(now=NOW):
    source = now.isoformat()
    resource = dict(
        source_timestamp_utc=source,
        input_evidence_ready=True,
        cpu_count=10,
        memory_pressure_state="green",
        memory_free_pct=60,
        swap_used_gb=2,
        local_disk_free_gb=100,
        pages_throttled=0,
    )
    runtime = dict(
        source_timestamp_utc=source,
        input_evidence_ready=True,
        protective_hold=False,
        adaptive_safety_limits={"active": False},
        runtime_measurement_evidence={"ready": True},
        memory_pressure_level="normal",
        host_saturation_score=45,
        mac_fluidity_contract={
            "support_pause_recommended": False,
            "fluidity_band": "guarded_smooth",
        },
        runtime_snapshot={
            "cpu_count": 10,
            "load_averages": {"one_minute": 7, "five_minutes": 7},
            "thermal": dict(
                measurement_available=True,
                thermal_warning_active=False,
                performance_warning_active=False,
                cpu_power_warning_active=False,
            ),
        },
        host_pressure_attribution=dict(
            bot_owned_cpu_percent=160,
            external_cpu_percent=140,
            foreground_app_cpu_percent=60,
            macos_system_cpu_percent=70,
            throttle_candidate_support_cpu_percent=80,
            protected_live_or_macro_cpu_percent=70,
            unknown_cpu_percent=10,
        ),
    )
    memory = dict(
        source_timestamp_utc=source,
        input_evidence_ready=True,
        creative_session={"active": False, "cooldown_active": False},
    )
    return resource, runtime, memory


def settled(now=NOW):
    prior = None
    for seconds in (0, 30, 60):
        at = now - timedelta(seconds=60 - seconds)
        prior = build_admission(*observations(at), prior, now=at)
    return prior


def test_workload_costs_dwell_and_actual_source_time():
    first = build_admission(*observations(), now=NOW)
    assert current_lease(first, "observer", now=NOW)
    assert current_lease(first, "storage_recovery", now=NOW)
    assert not current_lease(first, "maintenance", now=NOW)
    reused = build_admission(*observations(), first, now=NOW + timedelta(seconds=60))
    assert reused["workloads"]["maintenance"]["clear_seconds"] == 0
    assert not current_lease(reused, "maintenance", now=NOW + timedelta(seconds=60))
    ready = settled()
    assert current_lease(ready, "maintenance", now=NOW)
    assert current_lease(ready, "training_canary", now=NOW)
    assert not current_lease(ready, "bulk", now=NOW)
    assert ready["workloads"]["training_canary"]["max_workers"] == 1


@pytest.mark.parametrize(
    "bad", [None, True, "60", float("nan"), float("inf"), -1, 10**400]
)
@pytest.mark.parametrize(
    "key", ["memory_free_pct", "swap_used_gb", "local_disk_free_gb"]
)
def test_invalid_metrics_never_grant_extra_admission(key, bad):
    resource, runtime, memory = observations()
    resource[key] = bad
    result = build_admission(resource, runtime, memory, settled(), now=NOW)
    assert not any(row["admitted"] for row in result["workloads"].values())


@pytest.mark.parametrize(
    "change",
    [
        "resource_stale",
        "runtime_future",
        "memory_missing_time",
        "sensor_unknown",
        "protective_hold",
        "adaptive_hold",
        "thermal",
        "power_warning",
        "memory_yellow",
        "runtime_memory_high",
        "creative",
        "creative_cooldown",
        "four_cores",
        "cpu_mismatch",
        "unknown_cpu",
        "foreground_cpu",
        "system_cpu",
        "support_cpu",
        "protected_cpu",
        "aggregate_cpu",
        "pages_throttled",
    ],
)
def test_hard_pressure_revokes_every_lease(change):
    resource, runtime, memory = observations()
    if change == "resource_stale":
        resource["source_timestamp_utc"] = (NOW - timedelta(seconds=91)).isoformat()
    elif change == "runtime_future":
        runtime["source_timestamp_utc"] = (NOW + timedelta(seconds=1)).isoformat()
    elif change == "memory_missing_time":
        memory.pop("source_timestamp_utc")
    elif change == "sensor_unknown":
        runtime["runtime_measurement_evidence"]["ready"] = None
    elif change == "protective_hold":
        runtime["protective_hold"] = True
    elif change == "adaptive_hold":
        runtime["adaptive_safety_limits"]["active"] = True
    elif change in ("thermal", "power_warning"):
        runtime["runtime_snapshot"]["thermal"][
            (
                "thermal_warning_active"
                if change == "thermal"
                else "cpu_power_warning_active"
            )
        ] = True
    elif change == "memory_yellow":
        resource["memory_pressure_state"] = "yellow"
    elif change == "runtime_memory_high":
        runtime["memory_pressure_level"] = "high"
    elif change in ("creative", "creative_cooldown"):
        memory["creative_session"][
            "active" if change == "creative" else "cooldown_active"
        ] = True
    elif change == "four_cores":
        resource["cpu_count"] = runtime["runtime_snapshot"]["cpu_count"] = 4
    elif change == "cpu_mismatch":
        resource["cpu_count"] = 8
    elif change == "pages_throttled":
        resource["pages_throttled"] = 1
    else:
        key = {
            "unknown_cpu": "unknown_cpu_percent",
            "foreground_cpu": "foreground_app_cpu_percent",
            "system_cpu": "macos_system_cpu_percent",
            "support_cpu": "throttle_candidate_support_cpu_percent",
            "protected_cpu": "protected_live_or_macro_cpu_percent",
            "aggregate_cpu": "bot_owned_cpu_percent",
        }[change]
        runtime["host_pressure_attribution"][key] = 900
    result = build_admission(resource, runtime, memory, settled(), now=NOW)
    assert all(
        not row["admitted"] and row["clear_since_source_timestamp_utc"] is None
        for row in result["workloads"].values()
    )


def test_disk_pressure_allows_only_capacity_probe_observation_and_recovery():
    resource, runtime, memory = observations()
    resource["local_disk_free_gb"] = 60
    result = build_admission(resource, runtime, memory, settled(), now=NOW)
    assert current_lease(result, "capacity_probe", now=NOW)
    assert current_lease(result, "observer", now=NOW)
    assert current_lease(result, "storage_recovery", now=NOW)
    assert not current_lease(result, "maintenance", now=NOW)
    assert not current_lease(result, "training_canary", now=NOW)
    resource["local_disk_free_gb"] = 15.99
    result = build_admission(resource, runtime, memory, result, now=NOW)
    assert not any(row["admitted"] for row in result["workloads"].values())


def test_workload_load_boundaries_and_rollback():
    resource, runtime, memory = observations()
    runtime["runtime_snapshot"]["load_averages"]["five_minutes"] = 8.5
    result = build_admission(resource, runtime, memory, settled(), now=NOW)
    assert current_lease(result, "maintenance", now=NOW)
    runtime["runtime_snapshot"]["load_averages"]["one_minute"] = 8.51
    result = build_admission(resource, runtime, memory, result, now=NOW)
    assert not current_lease(result, "maintenance", now=NOW)
    assert current_lease(result, "observer", now=NOW)
    runtime["runtime_snapshot"]["load_averages"]["one_minute"] = 7
    result = build_admission(resource, runtime, memory, result, now=NOW)
    assert not current_lease(result, "maintenance", now=NOW)


def test_gaps_expiry_and_unknown_workloads_fail_closed():
    ready = settled()
    for name in POLICIES:
        assert not current_lease(ready, name, now=NOW + timedelta(seconds=91))
    assert not current_lease(ready, "force", now=NOW)
    at = NOW + timedelta(seconds=91)
    result = build_admission(*observations(at), ready, now=at)
    assert not current_lease(result, "maintenance", now=at)
    altered = deepcopy(ready)
    altered["source_timestamp_utc"] = (NOW + timedelta(seconds=1)).isoformat()
    assert not current_lease(altered, "observer", now=NOW)


def test_aggregate_protect_band_does_not_starve_observation_or_recovery():
    resource, runtime, memory = observations()
    runtime["mac_fluidity_contract"]["fluidity_band"] = "protect"
    result = build_admission(resource, runtime, memory, settled(), now=NOW)
    assert current_lease(result, "observer", now=NOW)
    assert current_lease(result, "storage_recovery", now=NOW)
    assert not current_lease(result, "maintenance", now=NOW)
    assert not current_lease(result, "training_canary", now=NOW)
    runtime["mac_fluidity_contract"]["support_pause_recommended"] = True
    result = build_admission(resource, runtime, memory, result, now=NOW)
    assert not any(row["admitted"] for row in result["workloads"].values())


def test_mixed_cpu_activity_is_not_automatically_host_saturation():
    from scripts.ops import autonomic_resource_governor as governor

    now = datetime.now(timezone.utc)
    _, runtime, _ = observations(now)
    runtime.update(
        overall_status="degraded",
        compute_pressure_level="elevated",
        throttle_profile="soft_cap",
        workload_admission=settled(now),
    )
    runtime["host_pressure_attribution"].update(
        system_cotenant_hot=True,
        protected_work_hot=True,
        support_jobs_hot=True,
        protected_pressure_dominant=True,
        external_pressure_dominant=True,
        dominant_bucket="bot_owned",
    )
    result = governor._runtime_pressure_attribution_policy(runtime)
    assert result["mode"] == "workload_capacity_advisory"
    assert result["training_allowed"]
    assert result["training_batch_cap"] == 1
    assert not result["p_core_widen_allowed"]
    assert result["system_cotenant_hot"]  # Raw activity evidence remains visible.
    runtime["workload_admission"]["workloads"]["training_canary"]["admitted"] = False
    assert not governor._runtime_pressure_attribution_policy(runtime)[
        "training_allowed"
    ]
    runtime["protective_hold"] = True
    assert (
        governor._runtime_pressure_attribution_policy(runtime)["mode"]
        != "workload_capacity_advisory"
    )


def test_capacity_advisory_cannot_expand_into_a_large_training_batch():
    from scripts.ops import autonomic_resource_governor as governor

    runtime = dict(
        overall_status="ready",
        memory_pressure_level="normal",
        compute_pressure_level="normal",
        host_saturation_score=45,
    )
    storage = dict(
        green=True,
        total_pending_lines=0,
        target_pending_lines=15000,
        oldest_pending_age_seconds=0,
        severity="ready",
        green_gate={"line_green": True, "age_green": True, "overlay_green": True},
    )
    memory = dict(
        safe_for_training=True,
        small_canary_training_safe=True,
        small_batch_training_safe=True,
        batch10_training_safe=True,
        batch20_training_safe=True,
        batch30_training_safe=True,
    )
    policy = dict(
        training_allowed=True,
        training_batch_cap=1,
        mode="workload_capacity_advisory",
        attribution={},
    )
    inputs = [
        storage,
        runtime,
        {"user_active": False},
        {"overall_status": "ready"},
        {
            "writer_state_before": {
                "current_step": "complete",
                "status": "ok",
                "running": False,
            }
        },
        {"status": "improving"},
        {"consecutive_green_samples": 10, "trend_regressing": False},
        memory,
        policy,
    ]
    result = governor._training_reentry_gate(*inputs)
    assert result["allowed"]
    assert result["max_parallel_trainings"] == 1
    assert result["profile"] == "coverage_micro_canary"
    storage.update(
        green=False, total_pending_lines=50000, oldest_pending_age_seconds=5000
    )
    storage["green_gate"] = {
        "line_green": False,
        "age_green": False,
        "overlay_green": False,
    }
    assert not governor._training_reentry_gate(*inputs)["allowed"]


@pytest.mark.parametrize(
    "name",
    [
        "ingestion_storage_control",
        "soak_reliability_sentinel",
        "local_storage_reserve_guard",
    ],
)
def test_native_observers_are_counted_as_bot_observation_not_unknown_external(name):
    from scripts.ops import runtime_throttle_control as throttle

    row = throttle._classify_process(f"python /project/scripts/ops/{name}.py --json")
    assert row["category"] == "operator_observability"
    assert not row["throttle_candidate"]
