#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_plumbing_control_latest.json"
RAW_LIVE_MAX_CORE_LINES = 5_000
RAW_LIVE_MAX_TOTAL_LINES = 15_000
RAW_LIVE_MAX_AGE_SECONDS = 15 * 60
BOUNDED_WRITE_FAILURE_LIMIT = 12
ROUTE_READY_STATES = {"ready", "verified", "curated_ready", "active_passthrough", "active_local_ready"}
STORAGE_PRESSURE_TARGET = 0.25
STORAGE_PRESSURE_ADVISORY_CEILING = 0.50
STORAGE_PRESSURE_HARD_CEILING = 1.0
EXTERNAL_STORAGE_TARGET_FREE_GB = 64.0
EXTERNAL_STORAGE_MIN_FREE_GB = 32.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    text = str(payload.get("overall_status") or payload.get("status") or "").strip().lower()
    if text:
        return text
    if "ok" in payload:
        return "ready" if bool(payload.get("ok", False)) else "blocked"
    return default


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _execution_expected(global_halt: dict[str, Any]) -> bool:
    metrics = _as_dict(global_halt.get("metrics"))
    if "execution_expected" in metrics:
        return bool(metrics.get("execution_expected", False))
    return bool(_truthy_env("ALLOW_ORDER_EXECUTION", "0") and not _truthy_env("MARKET_DATA_ONLY", "1"))


def _runtime_soft_cap_paper_relief(runtime: dict[str, Any]) -> dict[str, Any]:
    soft_cap = _as_dict(runtime.get("soft_cap_advisory_reclassification"))
    measurements = _as_dict(soft_cap.get("measurements"))
    thresholds = _as_dict(soft_cap.get("thresholds"))
    paper_policy = _as_dict(runtime.get("paper_execution_policy"))
    reason = str(soft_cap.get("reason") or "").strip()
    active = bool(soft_cap.get("active", False))
    paper_allowed = bool(paper_policy.get("paper_execution_allowed", False))
    paper_paused = bool(paper_policy.get("pause_paper_execution", False))
    external_guarded = bool(measurements.get("external_high_compute_guarded", False))
    storage_overlay_guarded = bool(measurements.get("bounded_storage_overlay_guarded", False))
    paper_memory_guarded = bool(measurements.get("paper_ramp_memory_guarded", False))
    paper_hot = bool(measurements.get("paper_execution_hot", False))
    bot_owned_dominant = bool(measurements.get("bot_owned_pressure_dominant", False))
    host_saturation = _safe_float(runtime.get("host_saturation_score"), _safe_float(measurements.get("host_saturation_score"), 0.0))
    host_ceiling = _safe_float(thresholds.get("max_guarded_external_high_compute_host_saturation_score"), 75.0)
    accepted_reason = reason in {
        "external_high_compute_pressure_is_capacity_limited_advisory_not_bot_runtime_degradation",
        "external_high_compute_with_bounded_storage_overlay_is_capacity_limited_advisory",
    }
    ok = bool(
        active
        and str(soft_cap.get("to_status") or "").strip().lower() in {"ready", "advisory", "guarded_ready"}
        and accepted_reason
        and external_guarded
        and (storage_overlay_guarded or paper_memory_guarded)
        and paper_allowed
        and not paper_paused
        and not paper_hot
        and not bot_owned_dominant
        and host_saturation < host_ceiling
    )
    return {
        "ok": ok,
        "active": active,
        "reason": reason,
        "to_status": str(soft_cap.get("to_status") or ""),
        "paper_execution_allowed": paper_allowed,
        "paper_execution_paused": paper_paused,
        "external_high_compute_guarded": external_guarded,
        "bounded_storage_overlay_guarded": storage_overlay_guarded,
        "paper_ramp_memory_guarded": paper_memory_guarded,
        "paper_execution_hot": paper_hot,
        "bot_owned_pressure_dominant": bot_owned_dominant,
        "host_saturation_score": host_saturation,
        "host_saturation_ceiling": host_ceiling,
        "policy": "consume runtime-throttle external high-compute reclassification only for paper-only operation while the paper lane remains unpaused",
    }


def _raw_live(storage: dict[str, Any]) -> dict[str, Any]:
    backpressure = _as_dict(storage.get("backpressure"))
    effective = _as_dict(backpressure.get("effective_raw_live"))
    raw = effective or _as_dict(backpressure.get("raw_live")) or backpressure
    core = _safe_int(raw.get("core_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), 0))
    total = _safe_int(raw.get("total_pending_lines"), _safe_int(backpressure.get("total_pending_lines"), 0))
    oldest = _safe_float(raw.get("oldest_pending_age_seconds"), _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0))
    clear = bool(core <= RAW_LIVE_MAX_CORE_LINES and total <= RAW_LIVE_MAX_TOTAL_LINES and oldest <= RAW_LIVE_MAX_AGE_SECONDS)
    result = {
        "ok": clear,
        "status": "ready" if clear else "blocked",
        "core_pending_lines": core,
        "total_pending_lines": total,
        "oldest_pending_age_seconds": round(oldest, 3),
        "max_core_pending_lines": RAW_LIVE_MAX_CORE_LINES,
        "max_total_pending_lines": RAW_LIVE_MAX_TOTAL_LINES,
        "max_oldest_pending_age_seconds": RAW_LIVE_MAX_AGE_SECONDS,
        "expansion_headroom_lines": max(RAW_LIVE_MAX_TOTAL_LINES - total, 0),
    }
    if effective:
        result["source"] = str(backpressure.get("effective_raw_live_source") or effective.get("source") or "effective_raw_live")
        result["reconciled_from_raw_live"] = bool(effective.get("reconciled_from_raw_live", False))
        if isinstance(effective.get("raw_live_estimate"), dict):
            result["raw_live_estimate"] = effective["raw_live_estimate"]
    return result


def _overlay_relief(storage: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    backpressure = _as_dict(storage.get("backpressure"))
    overlay_adjusted = bool(backpressure.get("overlay_adjusted", False))
    overlay_pressure_clear = bool(backpressure.get("overlay_pressure_clear", False))
    raw_clear = bool(raw.get("ok", False))
    overlay_total = _safe_int(backpressure.get("total_pending_lines"), 0)
    raw_estimate = _as_dict(raw.get("raw_live_estimate"))
    raw_total = _safe_int(raw_estimate.get("total_pending_lines"), _safe_int(raw.get("total_pending_lines"), 0))
    overlay_delta = max(overlay_total - raw_total, 0)
    active = bool(overlay_adjusted and overlay_pressure_clear and raw_clear and overlay_total <= 12000)
    return {
        "active": active,
        "overlay_adjusted": overlay_adjusted,
        "overlay_pressure_clear": overlay_pressure_clear,
        "overlay_total_pending_lines": overlay_total,
        "raw_total_pending_lines": raw_total,
        "overlay_delta_pending_lines": overlay_delta,
        "max_overlay_total_pending_lines": 12000,
        "policy": "SQL-overlay pressure is advisory for paper when raw-live queue health is cool and source-attributed cleanup is bounded",
    }


def _managed_deferred_backlog_relief(storage: dict[str, Any]) -> dict[str, Any]:
    backpressure = _as_dict(storage.get("backpressure"))
    storage_section = _as_dict(storage.get("storage"))
    route = _as_dict(storage.get("external_route_verification"))
    integrity = _as_dict(storage.get("data_integrity"))
    writer = _as_dict(storage.get("writer_shedding"))
    core_pending = _safe_int(backpressure.get("core_pending_lines"), 0)
    support_pending = _safe_int(backpressure.get("support_pending_lines"), 0)
    deferred_pending = _safe_int(backpressure.get("deferred_pending_lines"), 0)
    total_pending = _safe_int(backpressure.get("total_pending_lines"), 0)
    pending_threshold = max(_safe_int(backpressure.get("pending_lines_threshold"), RAW_LIVE_MAX_TOTAL_LINES), 1)
    backlog_status = str(storage_section.get("backlog_drain_status") or storage.get("backlog_drain_status") or "").strip().lower()
    route_ready = str(route.get("verification_state") or "").strip().lower() in ROUTE_READY_STATES
    integrity_clean = all(
        _safe_int(integrity.get(key), 0) == 0
        for key in ("sql_invalid_lines", "sql_overlay_invalid_lines", "sql_overlay_oversize_payloads", "sql_overlay_ops_write_failures")
    )
    allowed_hard_breaches = {"deferred", "support", "support_telemetry"}
    hard_breaches = {str(item).strip().lower() for item in _as_list(writer.get("hard_breaches")) if str(item).strip()}
    elevated_breaches = {str(item).strip().lower() for item in _as_list(writer.get("elevated_breaches")) if str(item).strip()}
    writer_breaches_managed = bool(hard_breaches <= allowed_hard_breaches and (elevated_breaches - allowed_hard_breaches) <= {"core"})
    hot_path_ok = bool(core_pending <= RAW_LIVE_MAX_CORE_LINES and support_pending <= 12000)
    deferred_managed = bool(
        deferred_pending > 0
        and backlog_status in {"waiting_for_off_hours", "off_hours_scheduled", "market_hours_guard", "handoff_requested"}
    )
    active = bool(
        hot_path_ok
        and deferred_managed
        and route_ready
        and integrity_clean
        and writer_breaches_managed
        and total_pending >= pending_threshold
    )
    return {
        "active": active,
        "status": "managed_deferred_backlog_waiting_for_off_hours" if active else "inactive",
        "hot_path_ok": hot_path_ok,
        "deferred_backlog_managed": deferred_managed,
        "route_ready": route_ready,
        "integrity_clean": integrity_clean,
        "writer_breaches_managed": writer_breaches_managed,
        "backlog_drain_status": backlog_status,
        "core_pending_lines": core_pending,
        "support_pending_lines": support_pending,
        "deferred_pending_lines": deferred_pending,
        "total_pending_lines": total_pending,
        "policy": "paper/data plumbing can continue when the hot write path is clean and deferred risk backlog is held for off-hours drain; live-money readiness still consumes raw backlog",
    }


def _queue_backpressure_section(storage: dict[str, Any]) -> dict[str, Any]:
    raw = _raw_live(storage)
    overlay_relief = _overlay_relief(storage, raw)
    deferred_relief = _managed_deferred_backlog_relief(storage)
    severity = str(storage.get("severity") or storage.get("overall_status") or "").strip().lower()
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    pressure_advisory = bool(
        severity in {"stable", "ready", ""}
        and pressure_index >= STORAGE_PRESSURE_TARGET
        and pressure_index < STORAGE_PRESSURE_ADVISORY_CEILING
        and raw.get("ok", False)
    )
    pressure_hard = bool(
        (severity in {"blocked", "critical", "high"} and not bool(overlay_relief.get("active", False)) and not bool(deferred_relief.get("active", False)))
        or (
            pressure_index >= STORAGE_PRESSURE_ADVISORY_CEILING
            and not bool(overlay_relief.get("active", False))
            and not bool(deferred_relief.get("active", False))
        )
        or (pressure_index >= STORAGE_PRESSURE_HARD_CEILING and not bool(overlay_relief.get("active", False)) and not bool(deferred_relief.get("active", False)))
    )
    storage_ready = bool((severity in {"stable", "ready", ""} or bool(overlay_relief.get("active", False)) or bool(deferred_relief.get("active", False))) and not pressure_hard)
    ok = bool((raw.get("ok", False) or bool(deferred_relief.get("active", False))) and storage_ready)
    blockers = []
    if not raw.get("ok", False) and not bool(deferred_relief.get("active", False)):
        blockers.append("raw_live_backpressure")
    if pressure_hard:
        blockers.append(f"storage_pressure={severity or pressure_index}")
    return {
        "ok": ok,
        "status": "managed_deferred_backlog_advisory" if ok and bool(deferred_relief.get("active", False)) else ("ready" if ok and not pressure_advisory else ("storage_pressure_advisory" if ok else "blocked")),
        "blockers": blockers,
        "severity": severity,
        "pressure_index": round(pressure_index, 3),
        "pressure_target": STORAGE_PRESSURE_TARGET,
        "pressure_advisory_ceiling": STORAGE_PRESSURE_ADVISORY_CEILING,
        "pressure_hard_ceiling": STORAGE_PRESSURE_HARD_CEILING,
        "pressure_advisory": pressure_advisory,
        "pressure_hard": pressure_hard,
        "raw_live": raw,
        "overlay_relief": overlay_relief,
        "managed_deferred_backlog_relief": deferred_relief,
        "backpressure_quality_score": _safe_float(storage.get("backpressure_quality_score"), 0.0),
        "policy": "raw-live queue health is authoritative for paper admission; stable storage pressure between target and advisory ceiling is a paper advisory, not a hard block",
    }


def _writer_section(writer: dict[str, Any], cycle: dict[str, Any]) -> dict[str, Any]:
    writer_health = _as_dict(writer.get("writer_health"))
    topology = _as_dict(writer.get("process_topology"))
    state = str(writer_health.get("state") or "").strip().lower()
    active = bool(writer_health.get("active", False))
    stale_or_stalled = state in {"stalled", "stale_progress", "orphaned_progress"}
    duplicate = bool(topology.get("duplicate_sql_writer_processes", False))
    timed_out = _safe_int(writer_health.get("timed_out_shard_count"), 0)
    child_count = _safe_int(writer_health.get("active_child_writer_count"), 0)
    cycle_status = _status(cycle, default="")
    ok = bool(not stale_or_stalled and not duplicate and timed_out <= 0 and _status(writer) in {"ready", "advisory", "degraded"})
    return {
        "ok": ok,
        "status": "ready" if ok else "blocked",
        "state": state or "unknown",
        "active": active,
        "current_step": str(writer_health.get("current_step") or _as_dict(cycle.get("writer_state_before")).get("current_step") or ""),
        "progress_age_minutes": _safe_float(writer_health.get("progress_age_minutes"), 0.0),
        "cycle_age_minutes": _safe_float(writer_health.get("cycle_age_minutes"), 0.0),
        "completed_shard_count": _safe_int(writer_health.get("completed_shard_count"), 0),
        "planned_shard_count": _safe_int(writer_health.get("planned_shard_count"), 0),
        "active_child_writer_count": child_count,
        "duplicate_sql_writer_processes": duplicate,
        "timed_out_shard_count": timed_out,
        "cycle_status": cycle_status,
        "single_writer_enforced": bool(_as_dict(writer.get("safety_envelope")).get("single_writer_only", True)),
        "policy": "allow active shard linking as healthy plumbing when the single writer is progressing and no duplicate primary writer exists",
    }


def _storage_route_section(storage: dict[str, Any], drain: dict[str, Any], failback: dict[str, Any]) -> dict[str, Any]:
    route = _as_dict(storage.get("external_route_verification")) or _as_dict(failback.get("route_verification"))
    route_state = str(route.get("verification_state") or "").strip().lower()
    mismatches = _as_list(route.get("mismatches"))
    resilience = _as_dict(storage.get("storage_resilience"))
    split_brain = _safe_int(resilience.get("unresolved_split_brain_conflicts"), _safe_int(failback.get("split_brain_conflicts"), 0))
    storage_plane = _as_dict(storage.get("storage_plane_contract"))
    disk_contract = _as_dict(storage_plane.get("disk_contract"))
    external_disk = _as_dict(disk_contract.get("external_disk"))
    external_available = bool(external_disk.get("exists", False) or _as_dict(_as_dict(storage.get("storage_efficiency_contract")).get("metrics")).get("external_available", False))
    external_available_gb = _safe_float(disk_contract.get("external_available_gb"), _safe_float(external_disk.get("available_gb"), 0.0))
    drain_status = _status(drain, default="")
    blocked_reasons = [str(item).strip().lower() for item in _as_list(drain.get("blocked_reasons")) if str(item).strip()]
    route_ok = bool(route_state in ROUTE_READY_STATES and not mismatches and split_brain <= 0)
    external_drain_advisory = bool(drain_status == "blocked" and set(blocked_reasons).issubset({"external_storage_unavailable", "market_hours_guard"}))
    ok = bool(route_ok)
    return {
        "ok": ok,
        "status": "ready" if ok else "blocked",
        "route_state": route_state,
        "route_ok": route_ok,
        "ready_count": _safe_int(route.get("ready_count"), 0),
        "tracked_count": _safe_int(route.get("tracked_count"), 0),
        "coverage_ratio": _safe_float(route.get("coverage_ratio"), 0.0),
        "mismatch_count": len(mismatches),
        "unresolved_split_brain_conflicts": split_brain,
        "external_available": external_available,
        "external_available_gb": external_available_gb,
        "external_reserve": {
            "target_free_gb": EXTERNAL_STORAGE_TARGET_FREE_GB,
            "min_free_gb": EXTERNAL_STORAGE_MIN_FREE_GB,
            "available_gb": round(external_available_gb, 3),
            "advisory": bool(external_available and external_available_gb < EXTERNAL_STORAGE_TARGET_FREE_GB and external_available_gb >= EXTERNAL_STORAGE_MIN_FREE_GB),
            "hard": bool(external_available and external_available_gb < EXTERNAL_STORAGE_MIN_FREE_GB),
        },
        "external_backlog_status": drain_status,
        "external_backlog_blocked_reasons": blocked_reasons,
        "external_drain_advisory": external_drain_advisory,
        "policy": "route verification and split-brain state are hard; off-hours/external drain availability is advisory when raw-live backpressure is clean",
    }


def _watchdog_section(process: dict[str, Any]) -> dict[str, Any]:
    alerts = _as_list(process.get("alerts"))
    safety = _as_dict(process.get("safety_pause"))
    restart_isolation = _as_dict(process.get("restart_storm_isolation"))
    critical_alerts = []
    warnings = []
    for raw in alerts:
        row = _as_dict(raw)
        alert = _as_dict(row.get("alert"))
        severity = str(alert.get("severity") or row.get("severity") or "").strip().lower()
        if severity in {"critical", "fatal", "blocker"}:
            critical_alerts.append(row)
        else:
            warnings.append(row)
    ok = bool(not safety.get("active", False) and not critical_alerts and _safe_int(restart_isolation.get("execution_blocking_count"), 0) <= 0)
    return {
        "ok": ok,
        "status": "ready" if ok else "blocked",
        "process_status": _status(process),
        "safety_pause_active": bool(safety.get("active", False)),
        "critical_alert_count": len(critical_alerts),
        "warning_alert_count": len(warnings),
        "restart_storm_isolation": restart_isolation,
        "policy": "read-only collector repair can stay isolated, but critical watchdog alerts and safety pauses block plumbing clear",
    }


def _runtime_section(
    runtime: dict[str, Any],
    memory: dict[str, Any],
    global_halt: dict[str, Any],
    queue_section: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_status = _status(runtime)
    memory_status = _status(memory)
    compute = str(runtime.get("compute_pressure_level") or "").strip().lower()
    memory_pressure = str(runtime.get("memory_pressure_level") or "").strip().lower()
    host_saturation = _safe_float(runtime.get("host_saturation_score"), 0.0)
    execution_expected = _execution_expected(global_halt)
    queue_section = _as_dict(queue_section)
    managed_deferred_backlog = _as_dict(queue_section.get("managed_deferred_backlog_relief"))
    memory_snapshot = _as_dict(memory.get("memory_snapshot"))
    memory_actual_clear = bool(
        _safe_float(memory_snapshot.get("memory_free_pct"), 100.0) >= 12.0
        and _safe_float(memory_snapshot.get("swap_used_gb"), 0.0) < 12.0
        and _safe_float(memory_snapshot.get("compressed_store_gb"), 0.0) < 28.0
        and _safe_float(memory_snapshot.get("compressor_gb"), 0.0) < 16.0
    )
    strict_ok = bool(
        runtime_status in {"ready", "advisory", "guarded_ready"}
        and memory_status in {"ready", "advisory", "guarded_ready"}
        and memory_pressure not in {"high", "critical"}
    )
    soft_cap_relief = _runtime_soft_cap_paper_relief(runtime)
    compute_ok_for_paper = bool(
        compute not in {"high", "critical"}
        or (compute == "high" and soft_cap_relief.get("ok", False))
    )
    paper_only_advisory = bool(
        not strict_ok
        and not execution_expected
        and runtime_status in {"ready", "advisory", "guarded_ready", "degraded"}
        and memory_status in {"ready", "advisory", "guarded_ready", "needs_work", "degraded"}
        and compute_ok_for_paper
        and memory_pressure not in {"high", "critical"}
        and host_saturation < 62.0
    )
    managed_deferred_advisory = bool(
        not strict_ok
        and bool(managed_deferred_backlog.get("active", False))
        and not execution_expected
        and runtime_status in {"ready", "advisory", "guarded_ready", "degraded", "blocked"}
        and memory_status in {"ready", "advisory", "guarded_ready", "needs_work", "degraded", "blocked"}
        and compute not in {"high", "critical"}
        and memory_pressure not in {"high", "critical"}
        and memory_actual_clear
        and host_saturation < 75.0
    )
    ok = bool(strict_ok or paper_only_advisory or managed_deferred_advisory)
    return {
        "ok": ok,
        "status": "ready" if strict_ok else ("managed_deferred_backlog_advisory" if managed_deferred_advisory else ("advisory" if paper_only_advisory else "blocked")),
        "runtime_status": runtime_status,
        "memory_status": memory_status,
        "host_saturation_score": host_saturation,
        "compute_pressure_level": compute,
        "memory_pressure_level": memory_pressure,
        "recommended_memory_profile": str(memory.get("recommended_profile") or ""),
        "execution_expected": execution_expected,
        "paper_only_runtime_memory_relief": paper_only_advisory,
        "managed_deferred_runtime_memory_relief": managed_deferred_advisory,
        "managed_deferred_backlog_relief": managed_deferred_backlog,
        "memory_actual_clear": memory_actual_clear,
        "compute_pressure_advisory": compute in {"elevated", "high"} and ok,
        "runtime_soft_cap_paper_relief": soft_cap_relief,
        "collector_duty_cycle_recommendation": {
            "active": compute in {"elevated", "high"} and ok,
            "env": {
                "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
                "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.24" if compute == "elevated" else "0.18",
                "BOT_HEAVY_COLLECTOR_QOS": "background",
            },
            "policy": "smooth elevated compute by duty-cycling heavy read-only collectors before runtime becomes degraded",
        },
        "policy": "runtime and memory pressure may be advisory for paper only while memory is not high/critical and live execution remains off",
    }


def _data_plane_section(
    data_plane: dict[str, Any],
    queue: dict[str, Any],
    writer: dict[str, Any],
    global_halt: dict[str, Any],
    queue_section: dict[str, Any] | None = None,
) -> dict[str, Any]:
    write_failures = _safe_int(data_plane.get("write_failure_count"), 0)
    raw_write_failures = _safe_int(data_plane.get("raw_write_failure_count"), write_failures)
    snapshot_failures = _safe_int(data_plane.get("account_snapshot_failure_count"), 0)
    queue_depth = _safe_int(data_plane.get("queue_depth"), _safe_int(_as_dict(_as_dict(queue.get("lane_counts")).get("core")).get("pending_lines"), _safe_int(queue.get("queue_depth"), 0)))
    recovery_state = str(data_plane.get("recovery_state") or "").strip().lower()
    writer_health = _as_dict(writer.get("writer_health"))
    writer_active = bool(writer_health.get("active", False) or _as_dict(data_plane.get("writer_handoff_contract")).get("writer_service_active", False))
    write_evidence = _as_dict(data_plane.get("write_path_recovery_evidence"))
    current_storage_write_ready = bool(data_plane.get("current_storage_write_ready", False) or write_evidence.get("ready", False))
    recovered_by_storage = bool(data_plane.get("write_path_recovered_by_storage", False))
    execution_expected = _execution_expected(global_halt)
    managed_deferred_backlog = _as_dict(_as_dict(queue_section).get("managed_deferred_backlog_relief"))
    managed_deferred_active = bool(managed_deferred_backlog.get("active", False))
    bounded_write_recovery = bool(
        raw_write_failures > 0
        and write_failures <= BOUNDED_WRITE_FAILURE_LIMIT
        and snapshot_failures <= 0
        and (queue_depth < RAW_LIVE_MAX_TOTAL_LINES or managed_deferred_active)
        and (current_storage_write_ready or recovered_by_storage or managed_deferred_active)
        and not execution_expected
        and (writer_active or recovery_state in {"stable", "recovering_under_guard", "recovering"})
    )
    ok = bool(snapshot_failures <= 0 and (write_failures <= 0 or bounded_write_recovery) and (queue_depth < RAW_LIVE_MAX_TOTAL_LINES or managed_deferred_active))
    return {
        "ok": ok,
        "status": "ready" if ok and write_failures <= 0 else ("advisory" if ok else "blocked"),
        "recovery_state": recovery_state,
        "write_failure_count": write_failures,
        "raw_write_failure_count": raw_write_failures,
        "account_snapshot_failure_count": snapshot_failures,
        "queue_depth": queue_depth,
        "writer_active": writer_active,
        "current_storage_write_ready": current_storage_write_ready,
        "write_path_recovered_by_storage": recovered_by_storage,
        "bounded_write_recovery": bounded_write_recovery,
        "managed_deferred_backlog_relief": managed_deferred_backlog,
        "bounded_write_failure_limit": BOUNDED_WRITE_FAILURE_LIMIT,
        "policy": "old write incidents are advisory for paper only when storage truth is current, raw queues are cool, and live execution is off",
    }


def _execution_boundary_section(global_halt: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    execution_expected = _execution_expected(global_halt)
    live_plane = _as_dict(runtime.get("live_plane"))
    live_lane_running = bool(live_plane.get("live_lane_running", _as_dict(global_halt.get("metrics")).get("live_lane_running", False)))
    ok = bool(not execution_expected)
    return {
        "ok": ok,
        "status": "paper_only_locked" if ok else "blocked_live_expected",
        "execution_expected": execution_expected,
        "live_lane_running": live_lane_running,
        "allow_order_execution_env": _truthy_env("ALLOW_ORDER_EXECUTION", "0"),
        "market_data_only_env": _truthy_env("MARKET_DATA_ONLY", "1"),
        "policy": "plumbing control can prepare paper and data lanes, but it never authorizes live order execution",
    }


def _global_clear_relief(global_halt: dict[str, Any], data_plane_section: dict[str, Any], watchdog_section: dict[str, Any]) -> dict[str, Any]:
    clear_blockers = [str(item).strip() for item in _as_list(global_halt.get("clear_blockers")) if str(item).strip()]
    blocker_set = set(clear_blockers)
    write_requested = "write_path_recovery_pending" in blocker_set
    queue_requested = "queue_backpressure_active" in blocker_set
    restart_requested = "restart_storm_active" in blocker_set
    restart_iso = _as_dict(watchdog_section.get("restart_storm_isolation"))
    isolated_restart_storm = bool(
        restart_requested
        and bool(restart_iso.get("safe_to_clear_when_not_executing", False))
        and _safe_int(restart_iso.get("execution_blocking_count"), 0) <= 0
    )
    bounded_write_recovery = bool(write_requested and data_plane_section.get("bounded_write_recovery", False))
    managed_deferred_backlog = _as_dict(data_plane_section.get("managed_deferred_backlog_relief"))
    managed_queue_backpressure = bool(queue_requested and managed_deferred_backlog.get("active", False) and data_plane_section.get("bounded_write_recovery", False))
    allowed_blockers = {"write_path_recovery_pending", "restart_storm_active", "queue_backpressure_active"}
    active = bool(
        not bool(global_halt.get("halt", False))
        and blocker_set
        and blocker_set <= allowed_blockers
        and (not write_requested or bounded_write_recovery)
        and (not queue_requested or managed_queue_backpressure)
        and (not restart_requested or isolated_restart_storm)
    )
    if active and queue_requested and write_requested:
        status = "managed_deferred_backpressure_advisory"
    elif active and queue_requested:
        status = "queue_backpressure_deferred_advisory"
    elif active:
        status = "ready"
    else:
        status = "not_needed" if not blocker_set else "blocked"
    return {
        "active": active,
        "status": status,
        "clear_blockers": sorted(blocker_set),
        "advisory_clear_blockers": sorted(blocker_set) if active else [],
        "bounded_write_recovery": bounded_write_recovery,
        "managed_queue_backpressure": managed_queue_backpressure,
        "managed_deferred_backlog_relief": managed_deferred_backlog,
        "isolated_restart_storm": isolated_restart_storm,
        "policy": "global clear blockers become advisory for paper only when every named blocker has a bounded read-only relief contract",
    }


def _root_cause_packet(
    *,
    blockers: list[str],
    warnings: list[str],
    sections: dict[str, dict[str, Any]],
    clear_relief: dict[str, Any],
) -> dict[str, Any]:
    if blockers:
        primary = blockers[0]
        command_map = {
            "queue_backpressure_blocked": ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--quick-bounded", "--json"],
            "sql_writer_blocked": ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--apply", "--json"],
            "storage_route_blocked": ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
            "collector_watchdog_blocked": ["./scripts/ops/opsctl.sh", "process-watchdog", "--json"],
            "runtime_memory_blocked": ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
            "execution_boundary_blocked": ["./scripts/ops/opsctl.sh", "global-halt-status", "--json"],
            "data_plane_recovery_blocked": ["./scripts/ops/opsctl.sh", "data-plane-recovery", "--json"],
            "global_clear_blockers_unbounded": ["./scripts/ops/opsctl.sh", "global-halt-refresh", "--json"],
        }
        return {
            "status": "blocked",
            "primary": primary,
            "why": "hard plumbing blocker remains bounded neither by raw-live queue health nor by paper-only relief contract",
            "confidence": 0.91,
            "next_command": command_map.get(primary, ["./scripts/ops/opsctl.sh", "health-fast", "--json"]),
            "blockers": ordered_unique(blockers),
            "warnings": ordered_unique(warnings),
        }

    if warnings:
        return {
            "status": "advisory",
            "primary": warnings[0],
            "why": "paper/data plumbing is ready, but one or more bounded maintenance lanes should keep draining in the background",
            "confidence": 0.86,
            "next_command": ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--quick-bounded", "--json"],
            "blockers": [],
            "warnings": ordered_unique(warnings),
        }

    queue = _as_dict(sections.get("queue_backpressure"))
    return {
        "status": "ready",
        "primary": "all_plumbing_clear",
        "why": "raw-live queue, writer, storage route, watchdog, runtime/memory, execution boundary, and data-plane recovery all satisfy the paper-only plumbing contract",
        "confidence": 0.94,
        "next_command": ["./scripts/ops/opsctl.sh", "health-fast", "--json"],
        "raw_live": queue.get("raw_live", {}),
        "blockers": [],
        "warnings": [],
        "clear_relief_active": bool(clear_relief.get("active", False)),
    }


def _managed_advisories(
    warnings: list[str],
    *,
    sections: dict[str, dict[str, Any]],
    clear_relief: dict[str, Any],
) -> dict[str, Any]:
    queue = _as_dict(sections.get("queue_backpressure"))
    storage = _as_dict(sections.get("storage_route"))
    runtime = _as_dict(sections.get("runtime_memory"))
    data_plane = _as_dict(sections.get("data_plane_recovery"))
    raw_live = _as_dict(queue.get("raw_live"))
    overlay_relief = _as_dict(queue.get("overlay_relief"))
    deferred_relief = _as_dict(queue.get("managed_deferred_backlog_relief"))
    external_reserve = _as_dict(storage.get("external_reserve"))

    managed: list[str] = []
    contracts: dict[str, dict[str, Any]] = {}

    if "external_backlog_drain_advisory" in warnings:
        active = bool(
            storage.get("route_ok", False)
            and (raw_live.get("ok", False) or deferred_relief.get("active", False))
            and storage.get("external_drain_advisory", False)
            and not external_reserve.get("hard", False)
        )
        if active:
            managed.append("external_backlog_drain_advisory")
        contracts["external_backlog_drain_advisory"] = {
            "managed": active,
            "route_ok": bool(storage.get("route_ok", False)),
            "raw_live_ready": bool(raw_live.get("ok", False)),
            "managed_deferred_backlog": bool(deferred_relief.get("active", False)),
            "external_drain_advisory": bool(storage.get("external_drain_advisory", False)),
            "external_reserve_hard": bool(external_reserve.get("hard", False)),
            "policy": "background drain availability is non-scoring when route truth is ready and either raw-live is cool or deferred backlog is explicitly owned by the off-hours drain",
        }

    if "external_storage_reserve_advisory" in warnings:
        active = bool(storage.get("route_ok", False) and external_reserve.get("advisory", False) and not external_reserve.get("hard", False))
        if active:
            managed.append("external_storage_reserve_advisory")
        contracts["external_storage_reserve_advisory"] = {
            "managed": active,
            "route_ok": bool(storage.get("route_ok", False)),
            "available_gb": _safe_float(external_reserve.get("available_gb"), 0.0),
            "min_free_gb": _safe_float(external_reserve.get("min_free_gb"), 0.0),
            "target_free_gb": _safe_float(external_reserve.get("target_free_gb"), 0.0),
            "policy": "reserve below target is advisory-only while it remains above the hard floor",
        }

    if "compute_pressure_advisory" in warnings:
        active = bool(runtime.get("ok", False) and runtime.get("memory_pressure_level") not in {"high", "critical"})
        if active:
            managed.append("compute_pressure_advisory")
        contracts["compute_pressure_advisory"] = {
            "managed": active,
            "runtime_ready": bool(runtime.get("ok", False)),
            "memory_pressure_level": runtime.get("memory_pressure_level"),
            "policy": "compute pressure is non-scoring while runtime is ready and memory is not high",
        }

    if "runtime_memory_paper_advisory" in warnings:
        active = bool(
            runtime.get("ok", False)
            and runtime.get("paper_only_runtime_memory_relief", False)
            and runtime.get("memory_pressure_level") not in {"high", "critical"}
        )
        if active:
            managed.append("runtime_memory_paper_advisory")
        contracts["runtime_memory_paper_advisory"] = {
            "managed": active,
            "runtime_ready": bool(runtime.get("ok", False)),
            "paper_only_runtime_memory_relief": bool(runtime.get("paper_only_runtime_memory_relief", False)),
            "memory_pressure_level": runtime.get("memory_pressure_level"),
            "host_saturation_score": _safe_float(runtime.get("host_saturation_score"), 0.0),
            "policy": "elevated soft-cap runtime pressure is non-scoring for paper/data plumbing while live execution is locked",
        }

    if "storage_pressure_hysteresis_advisory" in warnings:
        active = bool(raw_live.get("ok", False) and queue.get("pressure_advisory", False))
        if active:
            managed.append("storage_pressure_hysteresis_advisory")
        contracts["storage_pressure_hysteresis_advisory"] = {
            "managed": active,
            "raw_live_ready": bool(raw_live.get("ok", False)),
            "pressure_advisory": bool(queue.get("pressure_advisory", False)),
            "policy": "hysteresis pressure is non-scoring when raw-live queues are cool",
        }

    if "sql_overlay_cleanup_advisory" in warnings:
        active = bool(raw_live.get("ok", False) and overlay_relief.get("active", False))
        if active:
            managed.append("sql_overlay_cleanup_advisory")
        contracts["sql_overlay_cleanup_advisory"] = {
            "managed": active,
            "raw_live_ready": bool(raw_live.get("ok", False)),
            "overlay_relief_active": bool(overlay_relief.get("active", False)),
            "policy": "SQL overlay cleanup is non-scoring when raw-live queues are cool and overlay total is bounded",
        }

    if "managed_deferred_backlog_advisory" in warnings:
        deferred_relief = _as_dict(queue.get("managed_deferred_backlog_relief"))
        active = bool(queue.get("ok", False) and deferred_relief.get("active", False))
        if active:
            managed.append("managed_deferred_backlog_advisory")
        contracts["managed_deferred_backlog_advisory"] = {
            "managed": active,
            "queue_ready": bool(queue.get("ok", False)),
            "deferred_relief_active": bool(deferred_relief.get("active", False)),
            "core_pending_lines": _safe_int(deferred_relief.get("core_pending_lines"), 0),
            "support_pending_lines": _safe_int(deferred_relief.get("support_pending_lines"), 0),
            "deferred_pending_lines": _safe_int(deferred_relief.get("deferred_pending_lines"), 0),
            "backlog_drain_status": str(deferred_relief.get("backlog_drain_status") or ""),
            "policy": "deferred backlog is non-scoring for paper/data plumbing while hot queues are clean and the off-hours drain owns the debt",
        }

    if "write_path_recovery_advisory" in warnings:
        active = bool(
            data_plane.get("ok", False)
            and data_plane.get("bounded_write_recovery", False)
            and (
                data_plane.get("current_storage_write_ready", False)
                or _as_dict(data_plane.get("managed_deferred_backlog_relief")).get("active", False)
            )
            and str(clear_relief.get("status") or "") in {"not_needed", "ready", "managed_deferred_backpressure_advisory", "queue_backpressure_deferred_advisory"}
        )
        if active:
            managed.append("write_path_recovery_advisory")
        contracts["write_path_recovery_advisory"] = {
            "managed": active,
            "data_plane_ready": bool(data_plane.get("ok", False)),
            "bounded_write_recovery": bool(data_plane.get("bounded_write_recovery", False)),
            "current_storage_write_ready": bool(data_plane.get("current_storage_write_ready", False)),
            "global_clear_relief_status": str(clear_relief.get("status") or ""),
            "policy": "historical write-path recovery is non-scoring once current storage truth and data-plane recovery are ready",
        }

    managed = ordered_unique(managed)
    unmanaged = [item for item in ordered_unique(warnings) if item not in set(managed)]
    return {
        "managed": managed,
        "unmanaged": unmanaged,
        "all_managed": not unmanaged,
        "contracts": contracts,
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    process = load_json(health_root / "process_watchdog_latest.json")
    storage = load_json(health_root / "ingestion_storage_control_latest.json")
    drain = load_json(health_root / "external_backlog_drain_latest.json")
    failback = load_json(health_root / "storage_failback_sync_latest.json")
    writer = load_json(health_root / "writer_process_intelligence_latest.json")
    cycle = load_json(health_root / "writer_cycle_coordinator_latest.json")
    data_plane = load_json(health_root / "data_plane_recovery_controller_latest.json")
    queue = load_json(health_root / "ingestion_priority_queue_latest.json")
    runtime = load_json(health_root / "runtime_throttle_control_latest.json")
    memory = load_json(health_root / "memory_efficiency_control_latest.json")
    global_halt = load_json(health_root / "global_halt_auto_clear_latest.json") or load_json(health_root / "global_killswitch_latest.json")

    queue_section = _queue_backpressure_section(storage)
    writer_section = _writer_section(writer, cycle)
    storage_section = _storage_route_section(storage, drain, failback)
    watchdog_section = _watchdog_section(process)
    runtime_section = _runtime_section(runtime, memory, global_halt, queue_section)
    boundary_section = _execution_boundary_section(global_halt, runtime)
    data_plane_section = _data_plane_section(data_plane, queue, writer, global_halt, queue_section)
    clear_relief = _global_clear_relief(global_halt, data_plane_section, watchdog_section)

    sections = {
        "queue_backpressure": queue_section,
        "sql_writer": writer_section,
        "storage_route": storage_section,
        "collector_watchdog": watchdog_section,
        "runtime_memory": runtime_section,
        "execution_boundary": boundary_section,
        "data_plane_recovery": data_plane_section,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    for name, section in sections.items():
        if bool(section.get("ok", False)):
            continue
        blockers.append(f"{name}_blocked")
    if clear_relief["status"] == "blocked":
        blockers.append("global_clear_blockers_unbounded")
    if storage_section.get("external_drain_advisory", False):
        warnings.append("external_backlog_drain_advisory")
    if _as_dict(storage_section.get("external_reserve")).get("advisory", False):
        warnings.append("external_storage_reserve_advisory")
    if runtime_section.get("compute_pressure_advisory", False):
        warnings.append("compute_pressure_advisory")
    if runtime_section.get("paper_only_runtime_memory_relief", False):
        warnings.append("runtime_memory_paper_advisory")
    if queue_section.get("pressure_advisory", False):
        warnings.append("storage_pressure_hysteresis_advisory")
    if _as_dict(queue_section.get("overlay_relief")).get("active", False):
        warnings.append("sql_overlay_cleanup_advisory")
    if _as_dict(queue_section.get("managed_deferred_backlog_relief")).get("active", False):
        warnings.append("managed_deferred_backlog_advisory")
    if data_plane_section.get("bounded_write_recovery", False):
        warnings.append("write_path_recovery_advisory")

    ok = not blockers
    advisory_management = _managed_advisories(
        ordered_unique(warnings),
        sections=sections,
        clear_relief=clear_relief,
    )
    score_warnings = _as_list(advisory_management.get("unmanaged"))
    score = max(0, 100 - 20 * len(blockers) - 3 * len(score_warnings))
    overall_status = "ready" if ok else "blocked"
    root_cause = _root_cause_packet(
        blockers=ordered_unique(blockers),
        warnings=ordered_unique(warnings),
        sections=sections,
        clear_relief=clear_relief,
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "plumbing_score": score,
        "blockers": ordered_unique(blockers),
        "warnings": ordered_unique(warnings),
        "managed_advisories": advisory_management,
        "score_contract": {
            "structural_score": score,
            "warning_penalty_count": len(score_warnings),
            "managed_warning_count": len(_as_list(advisory_management.get("managed"))),
            "policy": "bounded managed advisories remain visible but do not reduce the plumbing score when hard readiness is green",
        },
        "root_cause": root_cause,
        "next_best_command": " ".join(root_cause.get("next_command") or []),
        "sections": sections,
        "global_clear_relief": clear_relief,
        "paper_ramp_relief_contract": {
            "ok": ok,
            "bounded_write_recovery": bool(data_plane_section.get("bounded_write_recovery", False)),
            "managed_deferred_backlog": bool(_as_dict(queue_section.get("managed_deferred_backlog_relief")).get("active", False)),
            "advisory_clear_blockers": clear_relief.get("advisory_clear_blockers", []),
            "execution_boundary": boundary_section.get("status"),
            "raw_live_expansion_ready": bool(_as_dict(queue_section.get("raw_live")).get("ok", False)),
            "policy": "paper ramp may consume this only for paper-live-data while live execution remains locked",
        },
        "control_env_recommendations": {
            "SYSTEM_PLUMBING_READY": "1" if ok else "0",
            "SYSTEM_PLUMBING_SCORE": str(score),
            "PAPER_WRITE_PATH_RECOVERY_ADVISORY": "1" if data_plane_section.get("bounded_write_recovery", False) else "0",
            "PAPER_GLOBAL_CLEAR_RELIEF_ACTIVE": "1" if clear_relief.get("active", False) else "0",
            "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1" if _as_dict(runtime_section.get("collector_duty_cycle_recommendation")).get("active", False) else "0",
            "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": _as_dict(_as_dict(runtime_section.get("collector_duty_cycle_recommendation")).get("env")).get("BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO", "0.35"),
            "ALLOW_ORDER_EXECUTION": "0",
            "MARKET_DATA_ONLY": "1",
        },
        "recommended_commands": [
            list(root_cause.get("next_command") or ["./scripts/ops/opsctl.sh", "health-fast", "--json"]),
            ["./scripts/ops/opsctl.sh", "data-plane-recovery", "--json"],
            ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--json"],
            ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
            ["./scripts/ops/opsctl.sh", "global-halt-refresh", "--json"],
            ["./scripts/ops/opsctl.sh", "paper-400-ramp", "--apply", "--json"],
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a shared plumbing control contract across queues, storage, writer, data-plane, and paper/live boundaries.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_file = Path(args.out_file).expanduser()
    if not out_file.is_absolute():
        out_file = project_root / out_file
    payload = build_payload(project_root)
    write_payload(out_file, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_plumbing_control "
            f"overall_status={payload.get('overall_status')} "
            f"score={payload.get('plumbing_score')} "
            f"blockers={len(payload.get('blockers') or [])}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
