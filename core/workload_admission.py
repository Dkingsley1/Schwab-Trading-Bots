"""Fresh, workload-specific resource leases; never execution or storage authority."""

from __future__ import annotations

import math
from datetime import datetime, timezone

POLICIES = {
    "capacity_probe": dict(
        load=0.85, saturation=55, memory=35, disk=16, dwell=60, seconds=60, workers=1
    ),
    "observer": dict(
        load=1.0, saturation=75, memory=25, disk=16, dwell=0, seconds=60, workers=1
    ),
    "storage_recovery": dict(
        load=0.85, saturation=70, memory=25, disk=16, dwell=0, seconds=240, workers=1
    ),
    "maintenance": dict(
        load=0.85, saturation=65, memory=30, disk=64, dwell=60, seconds=180, workers=1
    ),
    "training_canary": dict(
        load=0.85, saturation=55, memory=35, disk=64, dwell=60, seconds=180, workers=1
    ),
    "bulk": dict(
        load=0.62, saturation=45, memory=40, disk=125, dwell=180, seconds=300, workers=1
    ),
}
SLOT_WORKLOADS = {
    "infrastructure_observe": "observer",
    "infrastructure_autofix": "maintenance",
    "system_drift_autopilot": "maintenance",
    "section_grade_autopilot": "maintenance",
    "grade_regression_autopilot": "maintenance",
}
MAX_SOURCE_AGE = 90


def mapping(value):
    return value if isinstance(value, dict) else {}


def number(value):
    if type(value) not in (int, float):
        return None
    try:
        return float(value) if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def stamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except (TypeError, ValueError):
        return None


def fresh(value, now):
    parsed = stamp(value)
    return parsed is not None and 0 <= (now - parsed).total_seconds() <= MAX_SOURCE_AGE


def current_lease(payload, workload, *, now=None):
    now = now or datetime.now(timezone.utc)
    payload = mapping(payload)
    row = mapping(mapping(payload.get("workloads")).get(workload))
    return bool(
        workload in POLICIES
        and type(payload.get("schema_version")) is int
        and payload.get("schema_version") == 1
        and payload.get("input_evidence_ready") is True
        and fresh(payload.get("timestamp_utc"), now)
        and fresh(payload.get("source_timestamp_utc"), now)
        and row.get("admitted") is True
    )


def build_admission(resource, runtime, memory, previous=None, *, now=None):
    now = now or datetime.now(timezone.utc)
    resource, runtime, memory, previous = map(
        mapping, (resource, runtime, memory, previous)
    )
    snapshot = mapping(runtime.get("runtime_snapshot"))
    thermal = mapping(snapshot.get("thermal"))
    creative = mapping(memory.get("creative_session"))
    common = []
    source_times = [
        resource.get("source_timestamp_utc"),
        runtime.get("source_timestamp_utc"),
        memory.get("source_timestamp_utc"),
    ]
    if not all(fresh(value, now) for value in source_times) or not all(
        value.get("input_evidence_ready") is True
        for value in (resource, runtime, memory)
    ):
        common.append("source_evidence_not_ready")
    if mapping(runtime.get("runtime_measurement_evidence")).get("ready") is not True:
        common.append("runtime_sensors_not_ready")
    if (
        runtime.get("protective_hold") is not False
        or mapping(runtime.get("adaptive_safety_limits")).get("active") is not False
    ):
        common.append("protective_hold")
    if thermal.get("measurement_available") is not True or any(
        thermal.get(key) is not False
        for key in (
            "thermal_warning_active",
            "performance_warning_active",
            "cpu_power_warning_active",
        )
    ):
        common.append("thermal_not_clear")
    if (
        creative.get("active") is not False
        or creative.get("cooldown_active") is not False
    ):
        common.append("creative_session_or_cooldown")
    fluidity = mapping(runtime.get("mac_fluidity_contract"))
    if fluidity.get("support_pause_recommended") is not False or fluidity.get(
        "fluidity_band"
    ) not in ("smooth", "guarded_smooth", "protect", "strained"):
        common.append("foreground_responsiveness_not_clear")
    if (
        resource.get("memory_pressure_state") not in ("green", "normal")
        or runtime.get("memory_pressure_level") != "normal"
    ):
        common.append("memory_pressure")
    cpu_count = resource.get("cpu_count")
    if (
        type(cpu_count) is not int
        or cpu_count < 8
        or cpu_count != snapshot.get("cpu_count")
    ):
        common.append("cpu_capacity_not_ready")
        cpu_count = 1
    loads = mapping(snapshot.get("load_averages"))
    metrics = {
        "load1": number(loads.get("one_minute")),
        "load5": number(loads.get("five_minutes")),
        "saturation": number(runtime.get("host_saturation_score")),
        "memory": number(resource.get("memory_free_pct")),
        "swap": number(resource.get("swap_used_gb")),
        "disk": number(resource.get("local_disk_free_gb")),
        "throttled_pages": number(resource.get("pages_throttled")),
    }
    if any(value is None for value in metrics.values()):
        common.append("invalid_metrics")
    elif (
        metrics["memory"] > 100
        or metrics["saturation"] > 100
        or metrics["swap"] > 8
        or metrics["throttled_pages"] != 0
    ):
        common.append("memory_or_sensor_limit")
    attribution = mapping(runtime.get("host_pressure_attribution"))
    cpu = {
        key: number(attribution.get(key))
        for key in (
            "bot_owned_cpu_percent",
            "external_cpu_percent",
            "foreground_app_cpu_percent",
            "macos_system_cpu_percent",
            "throttle_candidate_support_cpu_percent",
            "protected_live_or_macro_cpu_percent",
            "unknown_cpu_percent",
        )
    }
    if any(value is None for value in cpu.values()):
        common.append("cpu_attribution_not_ready")
    else:
        # Percentages count one core as 100, not the entire machine.
        capacity = cpu_count * 100
        if cpu["bot_owned_cpu_percent"] + cpu["external_cpu_percent"] > capacity * 0.65:
            common.append("aggregate_cpu_pressure")
        for key, fraction in (
            ("foreground_app_cpu_percent", 0.20),
            ("macos_system_cpu_percent", 0.25),
            ("throttle_candidate_support_cpu_percent", 0.15),
            ("protected_live_or_macro_cpu_percent", 0.20),
            ("unknown_cpu_percent", 0.10),
        ):
            if cpu[key] > capacity * fraction:
                common.append(key + "_pressure")
    source = min((stamp(value) for value in source_times if stamp(value)), default=None)
    prior_source = stamp(previous.get("source_timestamp_utc"))
    continuity = bool(
        source
        and prior_source
        and previous.get("input_evidence_ready") is True
        and 0 <= (source - prior_source).total_seconds() <= MAX_SOURCE_AGE
    )
    workloads = {}
    for name, policy in POLICIES.items():
        reasons = list(common)
        if name in ("maintenance", "training_canary", "bulk") and fluidity.get(
            "fluidity_band"
        ) in ("protect", "strained"):
            reasons.append("responsiveness_guard")
        if not common:
            if max(metrics["load1"], metrics["load5"]) / cpu_count > policy["load"]:
                reasons.append("host_load")
            if metrics["saturation"] > policy["saturation"]:
                reasons.append("host_saturation")
            if metrics["memory"] < policy["memory"]:
                reasons.append("memory_headroom")
            if metrics["disk"] < policy["disk"]:
                reasons.append("disk_reserve")
        prior = mapping(mapping(previous.get("workloads")).get(name))
        clear_since = (
            stamp(prior.get("clear_since_source_timestamp_utc")) if continuity else None
        )
        if clear_since and not clear_since <= prior_source <= source:
            clear_since = None
        if reasons:
            clear_since = None
        elif clear_since is None:
            clear_since = source
        clear_seconds = (
            (source - clear_since).total_seconds() if source and clear_since else 0
        )
        if not reasons and clear_seconds < policy["dwell"]:
            reasons.append("independent_clear_observations_needed")
        workloads[name] = {
            "admitted": not reasons,
            "reasons": reasons,
            "stage": (
                "held"
                if clear_since is None
                else "recovering" if reasons else "bounded"
            ),
            "clear_since_source_timestamp_utc": (
                clear_since.isoformat() if clear_since else None
            ),
            "clear_seconds": clear_seconds,
            "max_runtime_seconds": policy["seconds"],
            "max_workers": policy["workers"],
            "minimum_free_gib": policy["disk"],
            "max_load_per_cpu": policy["load"],
        }
    return {
        "schema_version": 1,
        "timestamp_utc": now.isoformat(),
        "source_timestamp_utc": source.isoformat() if source else None,
        "input_evidence_ready": not common,
        "source_max_age_seconds": MAX_SOURCE_AGE,
        "metrics": metrics,
        "cpu_capacity_percent": cpu_count * 100,
        "workloads": workloads,
        "authority": "bounded_resource_admission_only_owner_storage_quality_and_execution_gates_still_required",
    }
