#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "autonomic_resource_governor_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.autonomic_resource_governor_override"
BACKLOG_GREEN_AGE_SECONDS = 900.0
BACKLOG_STALE_AGE_SECONDS = 240.0


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
    if not payload:
        return default
    for key in ("overall_status", "status"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    if payload.get("ok") is True:
        return "ready"
    if payload.get("ok") is False:
        return "blocked"
    return default


def _storage_metrics(storage: dict[str, Any]) -> dict[str, Any]:
    backpressure = _as_dict(storage.get("backpressure"))
    effective_raw_live = _as_dict(backpressure.get("effective_raw_live"))
    raw_live = effective_raw_live or _as_dict(backpressure.get("raw_live"))
    truth = _as_dict(storage.get("backlog_truth"))
    sql_overlay = _as_dict(truth.get("sql_overlay"))
    direct_sql_overlay = _as_dict(storage.get("sql_ingestion_pending_overlay"))
    storage_plane = _as_dict(storage.get("storage_plane_contract"))
    allowed_work = _as_dict(storage_plane.get("allowed_work"))
    p_core_contract = _as_dict(_as_dict(storage.get("backlog_relief_contract")).get("p_core_backlog_allocation_contract"))
    p_core_intelligence = _as_dict(p_core_contract.get("p_core_burst_intelligence"))
    stale_locator = _as_dict(storage.get("stale_pending_locator"))
    oldest_sources = _as_list(stale_locator.get("oldest_sources"))
    direct_overlay_total = _safe_int(direct_sql_overlay.get("total_pending_lines"), 0)
    direct_overlay_clear = bool(
        direct_sql_overlay.get("active", False)
        and direct_overlay_total == 0
        and _safe_int(direct_sql_overlay.get("stale_pending_lines"), 0) == 0
        and _safe_int(direct_sql_overlay.get("files_with_pending"), 0) == 0
        and not _as_list(direct_sql_overlay.get("top_pending_files"))
        and _safe_int(direct_sql_overlay.get("fresh_source_count"), 0) > 0
        and _safe_int(direct_sql_overlay.get("explicit_empty_source_count"), 0) > 0
        and _safe_float(direct_sql_overlay.get("oldest_pending_age_seconds"), 0.0) <= 120.0
    )
    if direct_overlay_clear:
        core = 0
        total = 0
        support = 0
        deferred = 0
        oldest_age = 0.0
    else:
        core = _safe_int(backpressure.get("core_pending_lines") or raw_live.get("core_pending_lines"), 0)
        total = _safe_int(backpressure.get("total_pending_lines") or raw_live.get("total_pending_lines"), 0)
        support = _safe_int(backpressure.get("support_pending_lines") or raw_live.get("support_pending_lines"), 0)
        deferred = _safe_int(backpressure.get("deferred_pending_lines") or raw_live.get("deferred_pending_lines"), 0)
        oldest_age = _safe_float(backpressure.get("oldest_pending_age_seconds") or raw_live.get("oldest_pending_age_seconds"), 0.0)
    overlay_pending = _safe_int(sql_overlay.get("total_pending_lines") or truth.get("overlay_pending_lines") or direct_overlay_total, 0)
    target = _safe_int(backpressure.get("pending_lines_threshold"), 5000) or 5000
    age_green = oldest_age <= BACKLOG_GREEN_AGE_SECONDS
    line_green = core <= target and total <= max(target, core)
    overlay_green = overlay_pending <= target
    storage_training_allowed = bool(allowed_work.get("training", False))
    backlog_green = bool(
        (line_green and age_green and overlay_green)
        or (
            storage_training_allowed
            and total <= max(target, 1)
            and oldest_age <= BACKLOG_GREEN_AGE_SECONDS
            and overlay_green
        )
    )
    if core > max(target * 5, 25000) or oldest_age > 86400 or total > 50000:
        severity = "critical"
    elif core > target or oldest_age > BACKLOG_GREEN_AGE_SECONDS or total > target * 2:
        severity = "degraded"
    elif core > max(target * 0.5, 2500) or total > target:
        severity = "watch"
    else:
        severity = "ready"
    return {
        "core_pending_lines": core,
        "total_pending_lines": total,
        "support_pending_lines": support,
        "deferred_pending_lines": deferred,
        "oldest_pending_age_seconds": round(oldest_age, 3),
        "overlay_pending_lines": overlay_pending,
        "target_pending_lines": target,
        "severity": severity,
        "green": backlog_green,
        "green_gate": {
            "status": "green" if backlog_green else "not_green",
            "line_green": line_green,
            "age_green": age_green,
            "overlay_green": overlay_green,
            "core_pending_target": target,
            "total_pending_target": max(target, core),
            "oldest_age_target_seconds": BACKLOG_GREEN_AGE_SECONDS,
            "oldest_age_stale_seconds": BACKLOG_STALE_AGE_SECONDS,
            "reason": "lines_and_age_clear" if backlog_green else "oldest_pending_age_or_overlay_still_needs_catch_up",
            "storage_training_allowed": storage_training_allowed,
            "direct_sql_overlay_clear": direct_overlay_clear,
        },
        "oldest_sources": oldest_sources[:6],
        "p_core_preprocess_worker_recommendation": _safe_int(p_core_contract.get("preprocess_worker_budget"), 0),
        "p_core_shard_writer_lanes_recommendation": _safe_int(p_core_contract.get("shard_link_writer_lanes"), 0),
        "p_core_burst_mode": str(p_core_intelligence.get("mode") or ""),
        "p_core_burst_reason": str(p_core_intelligence.get("reason") or ""),
    }


def _writer_active(writer: dict[str, Any]) -> bool:
    state = _as_dict(writer.get("writer_state_before")) or _as_dict(writer.get("writer_state_after_wait"))
    current_step = str(state.get("current_step") or "").strip().lower()
    status = str(state.get("status") or "").strip().lower()
    running = bool(state.get("running", False))
    if current_step == "complete" and status in {"ok", "complete", "idle"} and not running:
        return False
    return bool(state.get("active", False) or state.get("running", False))


def _writer_effectiveness(writer: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(writer.get("drain_effectiveness"))


def _writer_state(writer: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(writer.get("writer_state_before")) or _as_dict(writer.get("writer_state_after_wait"))


def _positive_writer_progress(writer: dict[str, Any]) -> bool:
    effect = _writer_effectiveness(writer)
    status = str(effect.get("status") or "").strip().lower()
    return bool(
        status in {"strong_progress", "progress"}
        or _safe_int(effect.get("merged_rows"), 0) > 0
        or _safe_int(effect.get("total_pending_delta"), 0) > 0
        or _safe_int(effect.get("core_pending_delta"), 0) > 0
    )


def _backlog_trend(previous: dict[str, Any], current: dict[str, Any], writer: dict[str, Any]) -> dict[str, Any]:
    prior = _as_dict(previous.get("storage_metrics"))
    current_ts = parse_iso_utc(current.get("timestamp_utc")) if "timestamp_utc" in current else None
    previous_ts = parse_iso_utc(previous.get("timestamp_utc")) if previous else None
    elapsed_seconds = None
    if current_ts is not None and previous_ts is not None:
        elapsed_seconds = max((current_ts - previous_ts).total_seconds(), 0.0)

    def delta(key: str) -> int:
        return _safe_int(prior.get(key), 0) - _safe_int(current.get(key), 0)

    core_delta = delta("core_pending_lines")
    total_delta = delta("total_pending_lines")
    overlay_delta = delta("overlay_pending_lines")
    age_delta = round(_safe_float(prior.get("oldest_pending_age_seconds"), 0.0) - _safe_float(current.get("oldest_pending_age_seconds"), 0.0), 3)
    pending_improving = bool(core_delta > 0 or total_delta > 0 or overlay_delta > 0 or _positive_writer_progress(writer))
    pending_regressing = bool(core_delta < 0 or total_delta < 0 or overlay_delta < 0)
    age_improving = age_delta > 0
    age_regressing = age_delta < -BACKLOG_STALE_AGE_SECONDS
    if pending_improving and not pending_regressing and not age_regressing:
        status = "improving"
    elif pending_improving and age_regressing:
        status = "lines_improving_age_still_aging"
    elif pending_regressing or age_regressing:
        status = "regressing"
    elif current.get("green"):
        status = "green_stable"
    else:
        status = "flat"
    rows_per_minute = None
    if elapsed_seconds and elapsed_seconds > 0:
        rows_per_minute = round(max(total_delta, core_delta, overlay_delta, 0) / (elapsed_seconds / 60.0), 3)
    return {
        "status": status,
        "has_previous_sample": bool(prior),
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
        "core_pending_delta": core_delta,
        "total_pending_delta": total_delta,
        "overlay_pending_delta": overlay_delta,
        "oldest_age_delta_seconds": age_delta,
        "pending_improving": pending_improving,
        "pending_regressing": pending_regressing,
        "age_improving": age_improving,
        "age_regressing": age_regressing,
        "rows_per_minute_estimate": rows_per_minute,
        "policy": "positive_deltas_mean_backlog_is_shrinking; age must also clear before reopening",
    }


def _stability_state(storage_metrics: dict[str, Any], runtime: dict[str, Any], writer: dict[str, Any], trend: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    prior = _as_dict(previous.get("stability_state"))
    backlog_green = bool(storage_metrics.get("green", False))
    runtime_status = _status(runtime)
    runtime_clear = runtime_status in {"ready", "advisory", "ok"} and str(runtime.get("memory_pressure_level") or "normal").lower() not in {"high", "red"}
    pressure_policy = _runtime_pressure_attribution_policy(runtime)
    runtime_pressure_clear = str(pressure_policy.get("mode") or "").strip().lower() in {"clear", "operator_foreground_advisory"}
    trend_regressing = str(trend.get("status") or "") == "regressing"
    writer_idle = not _writer_active(writer)
    green_samples = (_safe_int(prior.get("consecutive_green_samples"), 0) + 1) if backlog_green and not trend_regressing else 0
    runtime_clear_samples = (_safe_int(prior.get("consecutive_runtime_clear_samples"), 0) + 1) if runtime_clear else 0
    runtime_pressure_clear_samples = (
        _safe_int(prior.get("consecutive_runtime_pressure_clear_samples", prior.get("consecutive_runtime_clear_samples")), 0) + 1
    ) if runtime_clear and runtime_pressure_clear else 0
    idle_samples = (_safe_int(prior.get("consecutive_writer_idle_samples"), 0) + 1) if writer_idle else 0
    improvement_samples = (_safe_int(prior.get("consecutive_improving_samples"), 0) + 1) if bool(trend.get("pending_improving")) and not trend_regressing else 0
    return {
        "consecutive_green_samples": green_samples,
        "consecutive_runtime_clear_samples": runtime_clear_samples,
        "consecutive_runtime_pressure_clear_samples": runtime_pressure_clear_samples,
        "consecutive_writer_idle_samples": idle_samples,
        "consecutive_improving_samples": improvement_samples,
        "collector_reopen_min_green_samples": 2,
        "training_reentry_min_green_samples": 3,
        "p_core_widen_min_green_samples": 1,
        "runtime_clear": runtime_clear,
        "runtime_pressure_clear": runtime_pressure_clear,
        "writer_idle": writer_idle,
        "trend_regressing": trend_regressing,
        "collector_reopen_ready": bool(green_samples >= 2 and runtime_clear and not trend_regressing),
        "training_reentry_ready": bool(green_samples >= 3 and runtime_clear and writer_idle and not trend_regressing),
        "p_core_widen_ready": bool(green_samples >= 1 and runtime_clear and runtime_pressure_clear and writer_idle and not trend_regressing),
    }


def _p_core_memory_pressure_controller(host: dict[str, Any], runtime: dict[str, Any], memory_intelligence: dict[str, Any]) -> dict[str, Any]:
    body = _as_dict(host.get("body_map"))
    memory = _as_dict(body.get("memory"))
    snapshot = _as_dict(memory.get("memory_snapshot"))
    p_feedback = _as_dict(runtime.get("p_core_runtime_feedback"))
    burst = _as_dict(p_feedback.get("p_core_burst_intelligence"))
    inputs = _as_dict(burst.get("inputs"))
    level = str(runtime.get("memory_pressure_level") or memory.get("pressure_level") or "normal").strip().lower()
    kind = str(snapshot.get("memory_pressure_kind") or inputs.get("memory_pressure_kind") or "none").strip().lower()
    swap_gb = max(_safe_float(memory.get("swap_used_gb"), 0.0), _safe_float(snapshot.get("swap_used_gb"), 0.0), _safe_float(inputs.get("swap_used_gb"), 0.0))
    compressed_store_gb = max(_safe_float(snapshot.get("compressed_store_gb"), 0.0), _safe_float(inputs.get("compressed_store_gb"), 0.0))
    compressor_gb = max(_safe_float(snapshot.get("compressor_gb"), 0.0), _safe_float(inputs.get("compressor_gb"), 0.0))
    compressed_pressure_gb = max(
        _safe_float(snapshot.get("compressed_pressure_gb"), 0.0),
        _safe_float(inputs.get("compressed_pressure_gb"), 0.0),
        compressor_gb if compressor_gb > 0.0 else compressed_store_gb,
    )
    pages_throttled = _safe_float(inputs.get("pages_throttled"), 0.0)
    burst_mode = str(burst.get("mode") or "").strip().lower()
    allocation_only_compression = bool(
        level in {"", "normal", "green", "none", "clear"}
        and kind in {"", "none", "normal", "clear"}
        and swap_gb < 3.0
        and pages_throttled <= 0.0
        and compressed_pressure_gb < 9.0
    )
    if level in {"red", "high", "critical"} or kind in {"red", "critical"} or pages_throttled > 0:
        cap = 2
        status = "hard_memory_relief"
        reason = "memory pressure is high or pages are throttled"
    elif burst_mode.startswith("memory_relief_2") or swap_gb >= 8.0:
        cap = 2
        status = "swap_relief"
        reason = "swap or runtime burst controller requested memory relief"
    elif (burst_mode.startswith("memory_relief_3") and not allocation_only_compression) or compressed_pressure_gb >= 14.0 or swap_gb >= 4.0:
        cap = 3
        status = "compression_relief"
        reason = "compressed memory or swap is elevated, so P-core preprocess width is capped"
    elif compressed_pressure_gb >= 10.0 or swap_gb >= 2.0:
        cap = 4
        status = "soft_memory_guard"
        reason = "unified-memory pressure is soft-elevated"
    else:
        cap = 6
        status = "clear"
        reason = "memory pressure is clear enough for normal P-core widening"
    intelligence_classification = _as_dict(memory_intelligence.get("classification"))
    intelligence_gate = _as_dict(memory_intelligence.get("reopen_gate"))
    intelligence_trend = _as_dict(memory_intelligence.get("trend"))
    multitasking_headroom = _as_dict(memory_intelligence.get("multitasking_headroom"))
    intelligence_cap = _safe_int(intelligence_classification.get("recommended_p_core_worker_cap"), 0)
    intelligence_status = str(intelligence_classification.get("status") or "").strip().lower()
    intelligence_reason = str(intelligence_classification.get("reason") or "").strip()
    if intelligence_cap:
        if intelligence_status == "clear" and intelligence_cap > cap:
            cap = intelligence_cap
            reason = intelligence_reason or "memory intelligence cleared the seventh P-core burst cap"
        else:
            cap = min(cap, intelligence_cap)
        if intelligence_status and intelligence_status != "clear":
            status = intelligence_status
            reason = intelligence_reason or reason
        elif intelligence_status == "clear" and status == "clear":
            reason = intelligence_reason or reason
    if (
        allocation_only_compression
        and burst_mode in {"guarded_backlog_probe_4", "protect_live_backlog_probe_4"}
        and cap < 4
    ):
        cap = 4
        status = "soft_memory_guard"
        reason = (
            "memory is allocation-heavy but active compressor/swap pressure is clear, "
            "so the backlog pump may borrow one reserved P-core"
        )
    return {
        "enabled": True,
        "status": status,
        "max_memory_safe_workers": cap,
        "memory_pressure_level": level,
        "memory_pressure_kind": kind,
        "swap_used_gb": round(swap_gb, 3),
        "compressed_store_gb": round(compressed_store_gb, 3),
        "compressor_gb": round(compressor_gb, 3),
        "compressed_pressure_gb": round(compressed_pressure_gb, 3),
        "allocation_only_compression": bool(allocation_only_compression),
        "pages_throttled": pages_throttled,
        "runtime_burst_mode": burst_mode,
        "intelligence_layer_status": intelligence_status or "missing",
        "intelligence_layer_overall_status": _status(memory_intelligence),
        "memory_trend_status": str(intelligence_trend.get("status") or "unknown"),
        "multitasking_headroom_level": str(multitasking_headroom.get("level") or "unknown"),
        "multitasking_collector_ratio_cap": _safe_float(multitasking_headroom.get("collector_ratio_cap"), 0.55),
        "safe_to_widen_p_core_workers": bool(intelligence_gate.get("safe_to_widen_p_core_workers", status == "clear")),
        "safe_for_training": bool(intelligence_gate.get("safe_for_training", status == "clear")),
        "small_canary_training_safe": bool(intelligence_gate.get("small_canary_training_safe", False)),
        "small_canary_max_parallel_trainings": _safe_int(intelligence_gate.get("small_canary_max_parallel_trainings"), 0),
        "small_canary_profile": str(intelligence_gate.get("small_canary_profile") or ""),
        "small_batch_training_safe": bool(intelligence_gate.get("small_batch_training_safe", False)),
        "small_batch_max_parallel_trainings": _safe_int(intelligence_gate.get("small_batch_max_parallel_trainings"), 0),
        "small_batch_profile": str(intelligence_gate.get("small_batch_profile") or ""),
        "batch10_training_safe": bool(intelligence_gate.get("batch10_training_safe", False)),
        "batch10_max_parallel_trainings": _safe_int(intelligence_gate.get("batch10_max_parallel_trainings"), 0),
        "batch10_profile": str(intelligence_gate.get("batch10_profile") or ""),
        "batch20_training_safe": bool(intelligence_gate.get("batch20_training_safe", False)),
        "batch20_max_parallel_trainings": _safe_int(intelligence_gate.get("batch20_max_parallel_trainings"), 0),
        "batch20_profile": str(intelligence_gate.get("batch20_profile") or ""),
        "batch20_execution_mode": str(intelligence_gate.get("batch20_execution_mode") or ""),
        "batch20_wave_size": _safe_int(intelligence_gate.get("batch20_wave_size"), 0),
        "batch20_requires_between_target_memory_recheck": bool(intelligence_gate.get("batch20_requires_between_target_memory_recheck", False)),
        "weekend_large_batch_window": bool(intelligence_gate.get("weekend_large_batch_window", False)),
        "weekend_soft_guard_batch20_wave_training_safe": bool(intelligence_gate.get("weekend_soft_guard_batch20_wave_training_safe", False)),
        "batch30_training_safe": bool(intelligence_gate.get("batch30_training_safe", False)),
        "batch30_max_parallel_trainings": _safe_int(intelligence_gate.get("batch30_max_parallel_trainings"), 0),
        "batch30_profile": str(intelligence_gate.get("batch30_profile") or ""),
        "batch30_execution_mode": str(intelligence_gate.get("batch30_execution_mode") or ""),
        "batch30_wave_size": _safe_int(intelligence_gate.get("batch30_wave_size"), 0),
        "batch30_requires_between_target_memory_recheck": bool(intelligence_gate.get("batch30_requires_between_target_memory_recheck", False)),
        "weekend_soft_guard_batch30_wave_training_safe": bool(intelligence_gate.get("weekend_soft_guard_batch30_wave_training_safe", False)),
        "training_batch_cap": _safe_int(intelligence_gate.get("training_batch_cap"), 4 if bool(intelligence_gate.get("safe_for_training", False)) else 0),
        "training_profile": str(intelligence_gate.get("training_profile") or ""),
        "multitasking_training_cap": _safe_int(intelligence_gate.get("multitasking_training_cap"), 30),
        "foreground_soft_guard_micro_canary_safe": bool(intelligence_gate.get("foreground_soft_guard_micro_canary_safe", False)),
        "foreground_soft_guard_small_canary_safe": bool(intelligence_gate.get("foreground_soft_guard_small_canary_safe", False)),
        "memory_clear_samples": _safe_int(intelligence_gate.get("consecutive_memory_clear_samples"), 0),
        "reason": reason,
        "policy": "cap_p_core_preprocess_workers_when_unified_memory_or_swap_pressure_rises_and_require_intelligence_soak",
    }


def _runtime_pressure_attribution_policy(runtime: dict[str, Any]) -> dict[str, Any]:
    attribution = _as_dict(runtime.get("host_pressure_attribution"))
    compute_pressure = str(runtime.get("compute_pressure_level") or runtime.get("host_pressure_level") or "").strip().lower()
    memory_pressure = str(runtime.get("memory_pressure_level") or "normal").strip().lower()
    throttle_profile = str(runtime.get("throttle_profile") or "").strip().lower()
    runtime_status = _status(runtime)
    host_saturation = _safe_float(runtime.get("host_saturation_score"), 0.0)
    runtime_hot = bool(
        status_rank(runtime_status) >= status_rank("degraded")
        or compute_pressure in {"high", "critical", "protect_live"}
        or throttle_profile in {"protect_live", "sustain"}
        or host_saturation >= 55.0
        or memory_pressure in {"high", "red"}
    )
    external_dominant = bool(attribution.get("external_pressure_dominant", False))
    system_hot = bool(attribution.get("system_cotenant_hot", False))
    support_hot = bool(attribution.get("support_jobs_hot", False))
    support_low_priority = bool(attribution.get("support_hot_low_priority", False))
    operator_observability_hot = bool(attribution.get("operator_observability_hot", False))
    protected_hot = bool(attribution.get("protected_work_hot", False))
    dominant_bucket = str(attribution.get("dominant_bucket") or "unknown").strip().lower() or "unknown"
    foreground_app_cpu = _safe_float(attribution.get("foreground_app_cpu_percent"), 0.0)
    system_cpu = _safe_float(attribution.get("macos_system_cpu_percent"), 0.0)
    bot_owned_dominant = bool(attribution.get("bot_owned_pressure_dominant", False)) or dominant_bucket == "bot_owned"
    support_dominant = bool(attribution.get("support_pressure_dominant", False)) or dominant_bucket == "throttle_candidate_support"
    operator_observability_dominant = (
        bool(attribution.get("operator_observability_pressure_dominant", False))
        or dominant_bucket == "operator_observability"
    )
    system_dominant = bool(attribution.get("macos_system_pressure_dominant", False)) or dominant_bucket == "macos_system"
    protected_dominant = bool(attribution.get("protected_pressure_dominant", False)) or dominant_bucket == "protected_live_or_macro"
    system_secondary_to_bot_owned = bool(attribution.get("system_secondary_to_bot_owned", False))
    support_trim_required = bool(attribution.get("support_trim_required", False)) or (support_hot and support_dominant)
    guarded_niced_support_advisory = bool(
        support_hot
        and support_low_priority
        and (support_dominant or bot_owned_dominant)
        and (not system_hot or system_secondary_to_bot_owned)
        and not protected_dominant
        and host_saturation <= 78.0
        and compute_pressure not in {"high", "critical", "protect_live"}
        and memory_pressure not in {"high", "red"}
        and throttle_profile != "protect_live"
    )
    foreground_system_guarded = bool(
        external_dominant
        and dominant_bucket == "foreground_apps"
        and system_hot
        and foreground_app_cpu >= max(system_cpu, 35.0)
        and not support_hot
        and not protected_hot
        and host_saturation <= 52.0
        and compute_pressure not in {"high", "critical", "protect_live"}
        and memory_pressure not in {"high", "red"}
        and throttle_profile != "protect_live"
    )
    low_pressure_external_advisory = bool(
        external_dominant
        and not system_hot
        and not support_hot
        and not protected_hot
        and host_saturation <= 50.0
        and compute_pressure not in {"high", "critical", "protect_live"}
        and memory_pressure not in {"high", "red"}
        and throttle_profile not in {"protect_live", "sustain"}
    )
    guarded_foreground_advisory = bool(
        external_dominant
        and dominant_bucket == "foreground_apps"
        and (not system_hot or foreground_system_guarded)
        and not support_hot
        and not protected_hot
        and host_saturation <= 70.0
        and compute_pressure not in {"critical", "protect_live"}
        and memory_pressure not in {"high", "red"}
        and throttle_profile != "protect_live"
    )
    low_pressure_support_advisory = bool(
        support_hot
        and not system_hot
        and not protected_hot
        and not external_dominant
        and host_saturation <= 50.0
        and compute_pressure not in {"high", "critical", "protect_live"}
        and memory_pressure not in {"high", "red"}
        and throttle_profile not in {"protect_live", "sustain"}
    )
    guarded_support_advisory = bool(
        support_hot
        and support_low_priority
        and not system_hot
        and not protected_hot
        and host_saturation <= 68.0
        and compute_pressure not in {"high", "critical", "protect_live"}
        and memory_pressure not in {"high", "red"}
        and throttle_profile != "protect_live"
    )
    guarded_operator_observability_advisory = bool(
        operator_observability_hot
        and not support_hot
        and not system_hot
        and not protected_hot
        and host_saturation <= 68.0
        and compute_pressure not in {"high", "critical", "protect_live"}
        and memory_pressure not in {"high", "red"}
        and throttle_profile != "protect_live"
    )
    low_pressure_system_advisory = bool(
        system_hot
        and not runtime_hot
        and not support_hot
        and not protected_hot
        and host_saturation <= 35.0
        and compute_pressure not in {"high", "critical", "protect_live"}
        and memory_pressure not in {"high", "red"}
        and throttle_profile not in {"protect_live", "sustain"}
    )
    guarded_protected_work_advisory = bool(
        protected_hot
        and not runtime_hot
        and not system_hot
        and not support_hot
        and host_saturation <= 50.0
        and compute_pressure not in {"high", "critical", "protect_live"}
        and memory_pressure not in {"high", "red"}
        and throttle_profile != "protect_live"
    )
    if not attribution:
        mode = "legacy_runtime_pressure" if runtime_hot else "clear"
        reason = "runtime pressure attribution is not published yet" if runtime_hot else "runtime pressure is clear"
    elif low_pressure_system_advisory:
        mode = "macos_system_advisory"
        reason = "macOS system activity is visible, but runtime saturation and memory pressure remain training-safe"
    elif guarded_protected_work_advisory:
        mode = "protected_work_guarded_advisory"
        reason = "protected live or macro work is warm, but host saturation and memory pressure remain bounded"
    elif guarded_niced_support_advisory:
        mode = "support_maintenance_niced_advisory"
        reason = "support maintenance is hot but already low-priority, bounded, and safer than blocking training entirely"
    elif support_trim_required and not protected_dominant:
        mode = "trim_support_maintenance"
        reason = "bot-owned support maintenance is the dominant throttleable pressure source"
    elif guarded_operator_observability_advisory and operator_observability_dominant:
        mode = "operator_observability_guarded_advisory"
        reason = "operator/Codex observability is hot but live, support, memory, and storage pressure are bounded"
    elif protected_dominant and protected_hot:
        mode = "protect_live_or_macro_hot"
        reason = "protected live, paper, or macro capture lanes are hot"
    elif low_pressure_external_advisory:
        mode = "operator_foreground_advisory"
        reason = "foreground/operator activity is visible but host saturation and memory pressure remain training-safe"
    elif guarded_foreground_advisory:
        mode = "operator_foreground_guarded_advisory"
        reason = "foreground/operator activity dominates but memory, storage, and protected bot lanes are clear"
    elif system_hot and not foreground_system_guarded and not system_secondary_to_bot_owned and (system_dominant or external_dominant or not bot_owned_dominant):
        mode = "macos_system_cooldown"
        reason = "macOS system services are consuming host headroom"
    elif low_pressure_support_advisory:
        mode = "support_maintenance_advisory"
        reason = "support maintenance is hot but host saturation, compute pressure, and memory pressure remain training-safe"
    elif guarded_support_advisory:
        mode = "support_maintenance_niced_advisory"
        reason = "support maintenance is hot but already low-priority and bounded by runtime pressure relief"
    elif guarded_operator_observability_advisory:
        mode = "operator_observability_guarded_advisory"
        reason = "operator/Codex observability is hot but live, support, memory, and storage pressure are bounded"
    elif external_dominant:
        mode = "external_cotenant_cooldown"
        reason = "foreground/user/operator co-tenants dominate host pressure"
    elif support_hot:
        mode = "trim_support_maintenance"
        reason = "support maintenance is the hottest throttleable bot-owned work"
    elif protected_hot:
        mode = "protect_live_or_macro_hot"
        reason = "protected live, paper, or macro capture lanes are hot"
    elif runtime_hot:
        mode = "runtime_soft_cap"
        reason = "runtime pressure is elevated without a dominant pressure bucket"
    else:
        mode = "clear"
        reason = "runtime pressure attribution is clear"
    if not attribution:
        p_core_widen_allowed = True
        collector_reopen_allowed = memory_pressure not in {"high", "red"}
        training_allowed = memory_pressure not in {"high", "red"}
    else:
        p_core_widen_allowed = bool(
            mode in {"clear", "operator_foreground_advisory", "macos_system_advisory", "protected_work_guarded_advisory"}
            or (mode in {"operator_foreground_guarded_advisory", "operator_observability_guarded_advisory"} and host_saturation <= 64.0)
            or (mode == "runtime_soft_cap" and compute_pressure not in {"high", "critical"})
        )
        collector_reopen_allowed = bool(mode in {"clear", "runtime_soft_cap", "operator_foreground_advisory", "operator_foreground_guarded_advisory", "operator_observability_guarded_advisory", "support_maintenance_advisory", "support_maintenance_niced_advisory", "macos_system_advisory", "protected_work_guarded_advisory"} and memory_pressure not in {"high", "red"})
        training_allowed = bool(mode in {"clear", "operator_foreground_advisory", "operator_foreground_guarded_advisory", "operator_observability_guarded_advisory", "support_maintenance_advisory", "support_maintenance_niced_advisory", "macos_system_advisory", "protected_work_guarded_advisory"} and memory_pressure not in {"high", "red"})
    recommended_command = ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"] if mode != "clear" else []
    return {
        "mode": mode,
        "runtime_hot": runtime_hot,
        "runtime_status": runtime_status,
        "compute_pressure_level": compute_pressure,
        "memory_pressure_level": memory_pressure,
        "host_saturation_score": round(host_saturation, 3),
        "dominant_bucket": dominant_bucket,
        "foreground_app_cpu_percent": round(foreground_app_cpu, 3),
        "macos_system_cpu_percent": round(system_cpu, 3),
        "external_pressure_dominant": external_dominant,
        "bot_owned_pressure_dominant": bot_owned_dominant,
        "support_pressure_dominant": support_dominant,
        "operator_observability_pressure_dominant": operator_observability_dominant,
        "macos_system_pressure_dominant": system_dominant,
        "protected_pressure_dominant": protected_dominant,
        "system_secondary_to_bot_owned": system_secondary_to_bot_owned,
        "support_trim_required": support_trim_required,
        "guarded_niced_support_advisory": guarded_niced_support_advisory,
        "foreground_system_guarded": foreground_system_guarded,
        "low_pressure_external_advisory": low_pressure_external_advisory,
        "low_pressure_system_advisory": low_pressure_system_advisory,
        "guarded_foreground_advisory": guarded_foreground_advisory,
        "low_pressure_support_advisory": low_pressure_support_advisory,
        "system_cotenant_hot": system_hot,
        "support_jobs_hot": support_hot,
        "support_hot_low_priority": support_low_priority,
        "guarded_support_advisory": guarded_support_advisory,
        "operator_observability_hot": operator_observability_hot,
        "guarded_operator_observability_advisory": guarded_operator_observability_advisory,
        "guarded_protected_work_advisory": guarded_protected_work_advisory,
        "protected_work_hot": protected_hot,
        "p_core_widen_allowed": p_core_widen_allowed,
        "collector_reopen_allowed": collector_reopen_allowed,
        "training_allowed": training_allowed,
        "collector_ratio_cap": 0.35 if low_pressure_external_advisory or low_pressure_system_advisory else 0.28 if guarded_foreground_advisory or guarded_operator_observability_advisory or guarded_support_advisory or low_pressure_support_advisory or guarded_protected_work_advisory or support_hot or protected_hot else 0.20 if system_hot or external_dominant else 0.55,
        "recommended_command": recommended_command,
        "reason": reason,
        "attribution": attribution,
        "policy": "runtime_pressure_source_must_clear_before_widening_collectors_or_training",
    }


def _p_core_ramp_controller(
    *,
    current: int,
    max_safe: int,
    selected: int,
    pressure_policy: dict[str, Any],
    memory_control: dict[str, Any],
    writer_is_active: bool,
    backlog_green: bool,
    progress_positive: bool,
    trend_regressing: bool,
    stability: dict[str, Any],
) -> dict[str, Any]:
    green_samples = _safe_int(stability.get("consecutive_green_samples"), 0)
    runtime_pressure_clear_samples = _safe_int(stability.get("consecutive_runtime_pressure_clear_samples"), 0)
    writer_idle_samples = _safe_int(stability.get("consecutive_writer_idle_samples"), 0)
    memory_safe = bool(memory_control.get("safe_to_widen_p_core_workers", True))
    pressure_allows = bool(pressure_policy.get("p_core_widen_allowed", True))
    blockers = ordered_unique(
        [
            "writer_active" if writer_is_active else "",
            "backlog_not_green" if not backlog_green else "",
            "runtime_pressure_not_clear" if not pressure_allows or runtime_pressure_clear_samples < 2 else "",
            "memory_widen_soak_needed" if not memory_safe else "",
            "backlog_trend_regressing" if trend_regressing else "",
            "positive_writer_progress_needed" if not progress_positive and current < max_safe else "",
        ]
    )
    if trend_regressing:
        mode = "rollback_one_worker"
        target = max(min(current, max_safe) - 1, 1)
        reason = "backlog trend regressed, so the ramp rolls back before reopening more work"
    elif writer_is_active:
        mode = "hold_active_writer"
        target = min(current, max_safe)
        reason = "writer is active; keep width stable until the current cycle finishes"
    elif blockers:
        mode = "hold_for_clear_soak"
        target = min(current, max_safe)
        reason = "P-core widening waits for backlog, memory, runtime pressure, and progress soak gates"
    elif current < max_safe:
        mode = "step_up_one_worker"
        target = min(current + 1, max_safe)
        reason = "green backlog, clear host attribution, and positive progress permit one-worker widening"
    else:
        mode = "steady_at_safe_limit"
        target = min(current, max_safe)
        reason = "P-core width is already at the safe host/memory limit"
    return {
        "enabled": True,
        "mode": mode,
        "current_workers": int(max(current, 1)),
        "selected_workers": int(max(selected, 1)),
        "next_target_workers": int(max(target, 1)),
        "max_safe_workers": int(max(max_safe, 1)),
        "green_samples": int(green_samples),
        "runtime_pressure_clear_samples": int(runtime_pressure_clear_samples),
        "writer_idle_samples": int(writer_idle_samples),
        "requires_runtime_pressure_clear_samples": 2,
        "requires_writer_idle": True,
        "rollback_enabled": True,
        "blockers": blockers,
        "host_pressure_attribution_gate": pressure_policy,
        "memory_pressure_controller": memory_control,
        "reason": reason,
        "policy": "step_one_p_core_worker_at_a_time_after_runtime_pressure_memory_and_backlog_clear_soak",
    }


def _p_core_widening_controller(
    host: dict[str, Any],
    runtime: dict[str, Any],
    benchmark: dict[str, Any],
    memory_intelligence: dict[str, Any],
    writer: dict[str, Any],
    storage_metrics: dict[str, Any],
    trend: dict[str, Any],
    stability: dict[str, Any],
) -> dict[str, Any]:
    body = _as_dict(host.get("body_map"))
    cpu = _as_dict(body.get("cpu_topology"))
    p_feedback = _as_dict(runtime.get("p_core_runtime_feedback"))
    p_burst = _as_dict(p_feedback.get("p_core_burst_intelligence"))
    host_primary = _safe_int(cpu.get("performance_core_count") or cpu.get("recommended_primary_compute_lanes"), 1)
    runtime_budget = _safe_int(p_feedback.get("preprocess_worker_budget") or p_burst.get("selected_workers"), 0)
    benchmark_limits = _as_dict(benchmark.get("self_tuned_limits"))
    memory_control = _p_core_memory_pressure_controller(host, runtime, memory_intelligence)
    user_reserve_target = max(
        _safe_int(
            os.getenv("AUTONOMIC_PCORE_USER_APP_RESERVE_TARGET"),
            _safe_int(os.getenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET"), 0),
        ),
        0,
    )
    max_safe = _safe_int(benchmark_limits.get("recommended_p_core_preprocess_workers"), min(max(host_primary - 2, 1), 6))
    runtime_burst_mode = str(p_burst.get("mode") or "").strip().lower()
    if runtime_burst_mode == "burst_7" and host_primary >= 8:
        max_safe = max(max_safe, min(runtime_budget or 7, 7))
    max_safe = min(max(max_safe, 1), max(host_primary - 1, 1), 8)
    max_safe = min(max_safe, _safe_int(memory_control.get("max_memory_safe_workers"), max_safe))
    full_budget_requested = str(os.getenv("BACKLOG_PCORE_USE_FULL_PERFORMANCE_CORE_BUDGET") or "").strip().lower() in {"1", "true", "yes", "on"}
    env_worker_target = max(
        _safe_int(os.getenv("BACKLOG_PCORE_ACCELERATOR_WORKERS"), 0),
        _safe_int(os.getenv("BACKLOG_PCORE_PREPROCESS_WORKERS"), 0),
        _safe_int(os.getenv("SQL_LINK_SERVICE_PREPROCESS_WORKERS"), 0),
        _safe_int(os.getenv("SQL_LINK_SERVICE_SHARD_WRITER_LANES"), 0),
    )
    if full_budget_requested and env_worker_target > 0 and str(memory_control.get("status") or "clear") not in {"hard_relief", "swap_relief"}:
        max_safe = min(max(max_safe, env_worker_target), max(host_primary - 1, 1), 8)
    user_reserve_worker_cap = 0
    elastic_reserve_loan_cap = 0
    user_reserve = _as_dict(p_burst.get("user_app_reserve"))
    elastic_reserve_loan_allowed = bool(
        user_reserve.get("elastic_loan_allowed", False)
        or runtime_burst_mode in {"guarded_backlog_probe_4", "protect_live_backlog_probe_4"}
    )
    if user_reserve_target > 0:
        user_reserve_worker_cap = max(host_primary - user_reserve_target, 1)
        if elastic_reserve_loan_allowed:
            elastic_reserve_loan_cap = max(host_primary - max(user_reserve_target - 1, 1), 1)
            max_safe = min(max_safe, max(user_reserve_worker_cap, elastic_reserve_loan_cap))
        else:
            max_safe = min(max_safe, user_reserve_worker_cap)
    current = runtime_budget or min(max(host_primary - 3, 1), max_safe)
    if full_budget_requested and env_worker_target > 0 and str(memory_control.get("status") or "clear") not in {"hard_relief", "swap_relief"}:
        current = max(current, min(env_worker_target, max_safe))
    current = min(max(current, 1), max_safe)
    efficiency_total = _safe_int(cpu.get("efficiency_core_count"), 0)
    writer_is_active = _writer_active(writer)
    progress_positive = _positive_writer_progress(writer) or bool(trend.get("pending_improving", False))
    runtime_status = _status(runtime)
    runtime_clear = runtime_status in {"ready", "advisory", "ok"} and str(runtime.get("memory_pressure_level") or "normal").lower() not in {"high", "red"}
    backlog_green = bool(storage_metrics.get("green", False))
    storage_worker_target = _safe_int(storage_metrics.get("p_core_preprocess_worker_recommendation"), 0)
    storage_worker_target = min(max(storage_worker_target, 0), max_safe)
    storage_burst_mode = str(storage_metrics.get("p_core_burst_mode") or "").strip().lower()
    trend_regressing = bool(stability.get("trend_regressing", False))
    memory_safe_to_widen = bool(memory_control.get("safe_to_widen_p_core_workers", True))
    stable_for_widen = bool(stability.get("p_core_widen_ready", False) and memory_safe_to_widen)
    pressure_policy = _runtime_pressure_attribution_policy(runtime)
    pressure_allows_widen = bool(pressure_policy.get("p_core_widen_allowed", True))
    prearmed_next = min(current + 1, max_safe) if stable_for_widen and progress_positive and current < max_safe else current
    if storage_worker_target > prearmed_next and progress_positive and not trend_regressing:
        prearmed_next = storage_worker_target
    memory_capped = str(memory_control.get("status") or "clear") != "clear"
    multitasking_level = str(memory_control.get("multitasking_headroom_level") or "unknown")
    memory_status = str(memory_control.get("status") or "clear")
    if _safe_int(memory_control.get("max_memory_safe_workers"), max_safe) <= 3 and memory_status not in {"foreground_headroom"}:
        e_spillover = 0
        e_mode = "sealed_memory_guard"
        e_reason = "memory pressure is warm, so background spillover is kept off E-cores"
    elif multitasking_level in {"realtime_creative", "interactive_developer"}:
        e_spillover = min(efficiency_total, 1)
        e_mode = "p_core_primary_foreground_reserve"
        e_reason = "foreground creative/developer apps keep protected P-core reserve; E-cores only get minimal spillover"
    elif multitasking_level in {"media_playback", "foreground_standard"}:
        e_spillover = min(efficiency_total, 1)
        e_mode = "minimal_foreground_spillover"
        e_reason = "foreground apps are active, so only one low-priority spillover worker is allowed"
    else:
        e_spillover = min(efficiency_total, 2)
        e_mode = "bounded_background_spillover"
        e_reason = "system is in background mode, so bounded support spillover is allowed"
    if writer_is_active:
        if storage_worker_target > current and progress_positive and not trend_regressing and storage_burst_mode.startswith(("protect_live_backlog_probe", "guarded_backlog_probe", "burst_", "daily_driver")):
            selected = storage_worker_target
            mode = "hold_active_writer_prearmed_storage_target"
            reason = "active writer owns the current cycle; storage/backlog control pre-arms the wider P-core pump for the next cycle"
        else:
            selected = current
            mode = "hold_active_writer_memory_cap" if memory_capped else "hold_active_writer"
            reason = "active writer owns the lane; memory-aware worker cap is pre-armed for the next cycle" if memory_capped else "active writer owns the lane; do not change worker width mid-cycle"
    elif memory_capped and current >= max_safe:
        selected = max_safe
        mode = "memory_pressure_cap"
        reason = str(memory_control.get("reason") or "memory pressure capped P-core workers")
    elif trend_regressing:
        selected = max(min(current, max_safe) - 1, 1) if current > 3 else min(current, max_safe)
        mode = "brake_regression"
        reason = "backlog trend regressed, so worker widening is braked before collectors or training reopen"
    elif not backlog_green:
        selected = current
        mode = "hold_until_backlog_age_green"
        reason = "oldest pending age or overlay pressure is still above green"
    elif not pressure_allows_widen:
        selected = min(current, max_safe)
        mode = "hold_pressure_attribution"
        reason = str(pressure_policy.get("reason") or "host pressure attribution is not clear enough for widening")
    elif not runtime_clear:
        selected = min(current, max_safe)
        mode = "hold_runtime_pressure"
        reason = "runtime pressure is not clear enough for widening"
    elif not memory_safe_to_widen:
        selected = min(current, max_safe)
        mode = "memory_soak"
        reason = "memory intelligence needs consecutive clear samples before P-core widening"
    elif not stable_for_widen:
        selected = min(current, max_safe)
        mode = "green_soak"
        reason = "backlog is green, but the controller needs a stable green sample before widening"
    elif progress_positive and current < max_safe:
        selected = min(current + 1, max_safe)
        mode = "step_up_one_worker"
        reason = "last writer pass made positive progress and backlog is green"
    else:
        selected = min(current, max_safe)
        mode = "steady"
        reason = "worker width is already at the safe limit or needs one more progress sample"
    ramp_controller = _p_core_ramp_controller(
        current=current,
        max_safe=max_safe,
        selected=selected,
        pressure_policy=pressure_policy,
        memory_control=memory_control,
        writer_is_active=writer_is_active,
        backlog_green=backlog_green,
        progress_positive=progress_positive,
        trend_regressing=trend_regressing,
        stability=stability,
    )
    if str(ramp_controller.get("mode") or "") == "rollback_one_worker":
        selected = min(selected, _safe_int(ramp_controller.get("next_target_workers"), selected))
    return {
        "primary_compute_lanes": host_primary,
        "runtime_preprocess_worker_budget": runtime_budget,
        "selected_p_core_preprocess_workers": max(selected, 1),
        "efficiency_core_spillover": e_spillover,
        "efficiency_core_total": efficiency_total,
        "policy": "performance_core_primary_single_writer_with_user_app_reserve",
        "p_core_allocation_contract": {
            "system_primary_workers": max(selected, 1),
            "primary_merge_writer_count": 1,
            "shard_link_writer_lanes": max(selected, 1),
            "max_shard_link_writer_lanes": max_safe,
            "user_app_reserved_p_cores": max(host_primary - max(selected, 1), 0),
            "user_app_reserve_target_p_cores": int(user_reserve_target),
            "user_reserve_worker_cap": int(user_reserve_worker_cap),
            "elastic_reserve_loan_allowed": bool(elastic_reserve_loan_allowed),
            "elastic_reserve_loan_worker_cap": int(elastic_reserve_loan_cap),
            "creative_app_reserved_p_cores": max(host_primary - max(selected, 1), 0) if multitasking_level == "realtime_creative" else 0,
            "foreground_app_policy": "logic_fcp_and_interactive_apps_keep_p_core_reserve_when_open",
            "spillover_policy": "efficiency_cores_are_support_spillover_not_primary_compute",
            "writer_lane_policy": "parallel_child_shard_writers_on_p_core_budget_single_serial_primary_merge",
        },
        "efficiency_core_pressure_guard": {
            "enabled": True,
            "mode": e_mode,
            "selected_spillover_workers": e_spillover,
            "total_efficiency_cores": efficiency_total,
            "multitasking_level": multitasking_level,
            "memory_status": memory_status,
            "reason": e_reason,
            "policy": "keep_e_cores_available_for_macos_foreground_and_audio_when_memory_or_multitasking_pressure_rises",
        },
        "p_core_widening_controller": {
            "enabled": True,
            "mode": mode,
            "current_workers": current,
            "selected_workers": max(selected, 1),
            "max_safe_workers": max_safe,
            "user_app_reserve_target_p_cores": int(user_reserve_target),
            "user_reserve_worker_cap": int(user_reserve_worker_cap),
            "elastic_reserve_loan_allowed": bool(elastic_reserve_loan_allowed),
            "elastic_reserve_loan_worker_cap": int(elastic_reserve_loan_cap),
            "full_p_core_budget_requested": bool(full_budget_requested),
            "backlog_green_required": True,
            "backlog_green": backlog_green,
            "positive_writer_progress_required": True,
            "positive_writer_progress": progress_positive,
            "writer_active": writer_is_active,
            "storage_requested_workers": int(storage_worker_target),
            "storage_burst_mode": storage_burst_mode,
            "trend_status": str(trend.get("status") or "unknown"),
            "green_samples": _safe_int(stability.get("consecutive_green_samples"), 0),
            "prearmed_next_workers_when_idle": max(prearmed_next, 1),
            "regression_brake_active": trend_regressing,
            "memory_pressure_controller": memory_control,
            "host_pressure_attribution_gate": pressure_policy,
            "ramp_controller": ramp_controller,
            "reason": reason,
        },
    }


def _user_context(computer: dict[str, Any], host: dict[str, Any]) -> dict[str, Any]:
    session = _as_dict(computer.get("session_context"))
    body = _as_dict(host.get("body_map"))
    foreground = _as_dict(body.get("foreground_apps_and_user_activity"))
    open_apps = ordered_unique([*[str(item) for item in _as_list(session.get("open_apps"))], *[str(item) for item in _as_list(foreground.get("open_apps"))]])
    creative_level = str(session.get("creative_level") or foreground.get("creative_level") or "none").lower()
    co_level = str(session.get("co_running_level") or foreground.get("co_running_level") or "none").lower()
    user_active = bool(open_apps or creative_level not in {"", "none"} or co_level not in {"", "none"})
    return {
        "user_active": user_active,
        "open_apps": open_apps,
        "creative_level": creative_level,
        "co_running_level": co_level,
        "policy": "foreground_apps_keep_headroom" if user_active else "system_can_use_safe_background_headroom",
    }


def _collector_reopening_controller(
    storage_metrics: dict[str, Any],
    runtime: dict[str, Any],
    user: dict[str, Any],
    trend: dict[str, Any],
    stability: dict[str, Any],
    memory_control: dict[str, Any],
    pressure_policy: dict[str, Any],
) -> dict[str, Any]:
    backlog_green = bool(storage_metrics.get("green", False))
    runtime_status = _status(runtime)
    user_active = bool(user.get("user_active", False))
    trend_regressing = bool(stability.get("trend_regressing", False))
    green_samples = _safe_int(stability.get("consecutive_green_samples"), 0)
    memory_cap = _safe_int(memory_control.get("max_memory_safe_workers"), 6)
    pressure_allows_reopen = bool(pressure_policy.get("collector_reopen_allowed", True))
    pressure_ratio_cap = _safe_float(pressure_policy.get("collector_ratio_cap"), 0.55)
    pressure_mode = str(pressure_policy.get("mode") or "").strip().lower()
    runtime_degraded_by_guarded_foreground = bool(
        runtime_status == "degraded"
        and pressure_mode in {
            "operator_foreground_guarded_advisory",
            "operator_observability_guarded_advisory",
            "operator_foreground_advisory",
            "support_maintenance_niced_advisory",
        }
        and pressure_allows_reopen
    )
    if memory_cap <= 3:
        ratio = 0.20 if user_active else 0.24
        stage = "memory_guard"
        rollback = True
        next_ratio = 0.28
        reason = "memory intelligence is preserving headroom while compression or swap cools"
    elif trend_regressing:
        ratio = 0.12 if user_active else 0.18
        stage = "rollback_regression"
        rollback = True
        next_ratio = 0.20 if user_active else 0.28
        reason = "backlog trend regressed, so collector reopening is rolled back"
    elif not backlog_green:
        ratio = 0.20 if user_active else 0.28
        stage = "protect_core"
        rollback = True
        next_ratio = 0.28 if user_active else 0.35
        reason = "backlog age is not green yet"
    elif green_samples < 2:
        ratio = 0.28
        stage = "green_soak"
        rollback = True
        next_ratio = 0.35
        reason = "backlog is green, but collector reopening waits for two consecutive green samples"
    elif not pressure_allows_reopen:
        ratio = min(0.20 if user_active else 0.28, pressure_ratio_cap)
        stage = "runtime_pressure_attribution_cooldown"
        rollback = True
        next_ratio = min(0.35, pressure_ratio_cap)
        reason = str(pressure_policy.get("reason") or "host pressure attribution is not clear enough to reopen collectors")
    elif runtime_status not in {"ready", "advisory", "ok"} and not runtime_degraded_by_guarded_foreground:
        ratio = 0.28
        stage = "runtime_cooldown"
        rollback = True
        next_ratio = 0.35
        reason = "runtime pressure needs one clear pass before reopening collectors"
    elif runtime_degraded_by_guarded_foreground:
        ratio = min(0.35 if user_active else 0.45, pressure_ratio_cap)
        stage = "foreground_attributed_reopen"
        rollback = True
        next_ratio = min(0.45 if user_active else 0.55, pressure_ratio_cap)
        reason = "runtime is degraded by foreground-attributed pressure, but memory/storage are clean enough for a bounded collector reopen"
    elif user_active:
        ratio = 0.35
        stage = "user_coexistent_reopen"
        rollback = True
        next_ratio = 0.55
        reason = "backlog is green, but foreground app headroom is preserved"
    else:
        ratio = 0.55
        stage = "normal_reopen"
        rollback = True
        next_ratio = 0.55
        reason = "backlog is green and host is available"
    return {
        "enabled": True,
        "stage": stage,
        "max_active_ratio": round(ratio, 3),
        "next_ratio_if_stable": round(next_ratio, 3),
        "rollback_if_oldest_age_exceeds_seconds": BACKLOG_GREEN_AGE_SECONDS,
        "rollback_if_core_pending_exceeds": storage_metrics.get("target_pending_lines"),
        "rollback_enabled": rollback,
        "green_samples": green_samples,
        "trend_status": str(trend.get("status") or "unknown"),
        "host_pressure_attribution_gate": pressure_policy,
        "reason": reason,
    }


def _training_reentry_gate(
    storage_metrics: dict[str, Any],
    runtime: dict[str, Any],
    user: dict[str, Any],
    mlx: dict[str, Any],
    writer: dict[str, Any],
    trend: dict[str, Any],
    stability: dict[str, Any],
    memory_control: dict[str, Any],
    pressure_policy: dict[str, Any],
) -> dict[str, Any]:
    backlog_green = bool(storage_metrics.get("green", False))
    runtime_status = _status(runtime)
    mlx_status = _status(mlx)
    user_active = bool(user.get("user_active", False))
    memory_pressure = str(runtime.get("memory_pressure_level") or "normal").lower()
    compute_pressure = str(runtime.get("compute_pressure_level") or runtime.get("host_pressure_level") or "").strip().lower()
    host_saturation = _safe_float(runtime.get("host_saturation_score"), 100.0)
    memory_safe_for_training = bool(memory_control.get("safe_for_training", memory_pressure not in {"high", "red"}))
    small_canary_memory_safe = bool(memory_control.get("small_canary_training_safe", False))
    small_batch_memory_safe = bool(memory_control.get("small_batch_training_safe", False))
    multitasking_training_cap = _safe_int(memory_control.get("multitasking_training_cap"), 30)
    user_active_micro_canary_allowed = bool(user_active and multitasking_training_cap >= 1 and small_canary_memory_safe)
    user_active_small_canary_allowed = bool(user_active and multitasking_training_cap >= 2 and small_batch_memory_safe)
    batch10_memory_safe = bool(memory_control.get("batch10_training_safe", False))
    batch20_memory_safe = bool(memory_control.get("batch20_training_safe", False))
    batch20_memory_guarded_waves = str(memory_control.get("batch20_execution_mode") or "") == "sequential_memory_guarded_waves"
    batch30_memory_safe = bool(memory_control.get("batch30_training_safe", False))
    batch30_memory_guarded_waves = str(memory_control.get("batch30_execution_mode") or "") == "sequential_memory_guarded_waves"
    weekend_large_batch_window = bool(memory_control.get("weekend_large_batch_window", False))
    user_active_large_batch_allowed = bool(
        user_active
        and weekend_large_batch_window
        and str(memory_control.get("multitasking_headroom_level") or "").strip().lower() == "media_playback"
        and multitasking_training_cap >= 10
        and (batch10_memory_safe or batch20_memory_safe or batch30_memory_safe)
    )
    host_pressure_allows_training = bool(pressure_policy.get("training_allowed", True))
    pressure_attribution = _as_dict(pressure_policy.get("attribution"))
    hot_support_processes = [
        _as_dict(item)
        for item in pressure_attribution.get("hot_support_processes", [])
        if isinstance(item, dict)
    ]
    support_hot_is_control_plane = bool(
        hot_support_processes
        and all(
            any(
                needle in str(item.get("command_excerpt") or "")
                for needle in ("creative_cotenant_guard.py", "swap_pressure_governor.py")
            )
            for item in hot_support_processes
        )
    )
    weekend_control_plane_pressure_allowed = bool(
        weekend_large_batch_window
        and support_hot_is_control_plane
        and memory_pressure not in {"high", "red"}
        and compute_pressure not in {"high", "critical", "protect_live"}
        and host_saturation <= 70.0
        and not bool(pressure_policy.get("protected_work_hot", False))
    )
    bounded_micro_pressure_modes = {
        "external_cotenant_cooldown",
        "macos_system_cooldown",
        "operator_foreground_guarded_advisory",
        "operator_observability_guarded_advisory",
        "runtime_soft_cap",
        "trim_support_maintenance",
    }
    paper_hot = bool(pressure_attribution.get("paper_execution_pressure_dominant", False))
    research_hot = bool(pressure_attribution.get("research_pressure_dominant", False))
    writer_idle = not _writer_active(writer)
    writer_state = _writer_state(writer)
    writer_progress_age = _safe_float(writer_state.get("progress_age_minutes"), 0.0)
    green_gate = _as_dict(storage_metrics.get("green_gate"))
    storage_total_pending = _safe_int(storage_metrics.get("total_pending_lines"), 0)
    storage_target_pending = max(_safe_int(storage_metrics.get("target_pending_lines"), 15_000), 1)
    storage_oldest_age = _safe_float(storage_metrics.get("oldest_pending_age_seconds"), 0.0)
    storage_severity = str(storage_metrics.get("severity") or "").strip().lower()
    writer_active_green_safe = bool(
        not writer_idle
        and backlog_green
        and _safe_int(storage_metrics.get("total_pending_lines"), 0) <= min(max(_safe_int(storage_metrics.get("target_pending_lines"), 15000) // 10, 250), 1500)
        and _safe_float(storage_metrics.get("oldest_pending_age_seconds"), 0.0) <= 120.0
        and writer_progress_age <= 5.0
    )
    micro_backlog_green = bool(
        backlog_green
        or (
            bool(green_gate.get("line_green", False))
            and bool(green_gate.get("age_green", False))
            and storage_total_pending <= storage_target_pending
            and storage_oldest_age <= 120.0
            and storage_severity in {"", "ready", "stable", "watch", "advisory"}
            and (writer_idle or writer_active_green_safe)
        )
    )
    green_samples = _safe_int(stability.get("consecutive_green_samples"), 0)
    trend_regressing = bool(stability.get("trend_regressing", False))
    runtime_clear = runtime_status in {"ready", "advisory", "ok"}
    runtime_micro_clear = bool(
        (runtime_status in {"ready", "advisory", "ok", "degraded"} and memory_pressure not in {"high", "red"})
        or (
            small_canary_memory_safe
            and not memory_safe_for_training
            and runtime_status == "blocked"
            and host_saturation <= 97.0
            and memory_pressure not in {"high", "red"}
            and not bool(pressure_policy.get("protected_work_hot", False))
            and (
                not bool(pressure_policy.get("support_jobs_hot", False))
                or support_hot_is_control_plane
            )
        )
    )
    micro_green_samples_ok = bool(
        green_samples >= 3
        or (green_samples >= 2 and (writer_idle or writer_active_green_safe))
        or (
            backlog_green
            and writer_idle
            and storage_total_pending <= min(max(storage_target_pending // 10, 250), 1500)
            and storage_oldest_age <= 120.0
        )
        or (not backlog_green and micro_backlog_green and writer_idle)
    )
    micro_host_pressure_allows_training = bool(
        host_pressure_allows_training
        or weekend_control_plane_pressure_allowed
        or (
            small_canary_memory_safe
            and not memory_safe_for_training
            and str(pressure_policy.get("mode") or "") in bounded_micro_pressure_modes
            and host_saturation <= (97.0 if support_hot_is_control_plane else 80.0)
            and memory_pressure not in {"high", "red"}
            and (
                not bool(pressure_policy.get("support_jobs_hot", False))
                or support_hot_is_control_plane
            )
            and not paper_hot
            and not research_hot
            and not bool(pressure_policy.get("protected_work_hot", False))
        )
    )
    batch20_runtime_wave_clear = bool(
        batch20_memory_guarded_waves
        and runtime_micro_clear
        and batch20_memory_safe
        and compute_pressure not in {"critical", "protect_live"}
        and host_saturation <= 70.0
    )
    batch30_runtime_wave_clear = bool(
        batch30_memory_guarded_waves
        and runtime_micro_clear
        and batch30_memory_safe
        and compute_pressure not in {"critical", "protect_live"}
        and host_saturation <= 70.0
    )
    writer_active_small_batch_safe = bool(
        writer_active_green_safe
        and (small_batch_memory_safe or batch10_memory_safe or batch20_memory_safe or batch30_memory_safe)
    )
    full_allowed = bool(
        backlog_green
        and green_samples >= 3
        and (runtime_clear or (writer_active_small_batch_safe and runtime_micro_clear))
        and mlx_status not in {"blocked", "missing"}
        and memory_pressure not in {"high", "red"}
        and memory_safe_for_training
        and (host_pressure_allows_training or weekend_control_plane_pressure_allowed)
        and (not user_active or user_active_large_batch_allowed)
        and (writer_idle or writer_active_small_batch_safe)
        and not trend_regressing
    )
    batch30_allowed = bool(
        backlog_green
        and green_samples >= (3 if weekend_large_batch_window else 4)
        and (runtime_clear or batch30_runtime_wave_clear)
        and mlx_status not in {"blocked", "missing"}
        and memory_pressure not in {"high", "red"}
        and batch30_memory_safe
        and (host_pressure_allows_training or weekend_control_plane_pressure_allowed)
        and (not user_active or user_active_large_batch_allowed)
        and (writer_idle or (batch30_memory_guarded_waves and writer_active_green_safe))
        and not trend_regressing
    )
    batch20_allowed = bool(
        not batch30_allowed
        and backlog_green
        and green_samples >= (3 if weekend_large_batch_window else 4)
        and (runtime_clear or batch20_runtime_wave_clear)
        and mlx_status not in {"blocked", "missing"}
        and memory_pressure not in {"high", "red"}
        and batch20_memory_safe
        and (host_pressure_allows_training or weekend_control_plane_pressure_allowed)
        and (not user_active or user_active_large_batch_allowed)
        and (writer_idle or (batch20_memory_guarded_waves and writer_active_green_safe))
        and not trend_regressing
    )
    batch10_allowed = bool(
        not (batch30_allowed or batch20_allowed)
        and backlog_green
        and green_samples >= 3
        and runtime_clear
        and mlx_status not in {"blocked", "missing"}
        and memory_pressure not in {"high", "red"}
        and batch10_memory_safe
        and (host_pressure_allows_training or weekend_control_plane_pressure_allowed)
        and (not user_active or user_active_large_batch_allowed)
        and writer_idle
        and not trend_regressing
    )
    micro_allowed = bool(
        not (batch30_allowed or batch20_allowed or batch10_allowed or full_allowed)
        and micro_backlog_green
        and micro_green_samples_ok
        and runtime_micro_clear
        and mlx_status not in {"blocked", "missing"}
        and small_canary_memory_safe
        and micro_host_pressure_allows_training
        and (not user_active or user_active_micro_canary_allowed)
        and (writer_idle or writer_active_green_safe)
        and not trend_regressing
    )
    small_allowed = bool(
        not (batch30_allowed or batch20_allowed or batch10_allowed or full_allowed)
        and backlog_green
        and green_samples >= 3
        and runtime_micro_clear
        and mlx_status not in {"blocked", "missing"}
        and small_batch_memory_safe
        and host_pressure_allows_training
        and (not user_active or user_active_small_canary_allowed)
        and (writer_idle or writer_active_green_safe)
        and not trend_regressing
    )
    allowed = bool(
        batch30_allowed or batch20_allowed or batch10_allowed or full_allowed or small_allowed or micro_allowed
    )
    profile = (
        "coverage_batch30_canary"
        if batch30_allowed
        else "coverage_batch20_canary"
        if batch20_allowed
        else "coverage_batch10_canary"
        if batch10_allowed
        else "coverage_canary"
        if full_allowed
        else "coverage_small_canary"
        if small_allowed
        else "coverage_micro_canary"
        if micro_allowed
        else "none"
    )
    mode = (
        "batch30_canary"
        if batch30_allowed
        else "batch20_canary"
        if batch20_allowed
        else "batch10_canary"
        if batch10_allowed
        else "small_targeted"
        if full_allowed
        else "small_canary"
        if small_allowed
        else "micro_canary"
        if micro_allowed
        else "paused"
    )
    max_parallel = 30 if batch30_allowed else 20 if batch20_allowed else 10 if batch10_allowed else 4 if full_allowed else 2 if small_allowed else 1 if micro_allowed else 0
    blockers = ordered_unique(
        [
            "backlog_age_not_green" if not backlog_green and not micro_backlog_green else "",
            "micro_backlog_overlay_not_green" if not backlog_green and not micro_backlog_green else "",
            "green_soak_samples_needed" if not micro_green_samples_ok else "",
            "runtime_not_clear" if not runtime_clear and not (micro_allowed or runtime_micro_clear) else "",
            "mlx_not_ready" if mlx_status in {"blocked", "missing"} else "",
            "memory_pressure_not_clear" if memory_pressure in {"high", "red"} else "",
            "host_pressure_attribution_not_clear" if not micro_host_pressure_allows_training else "",
            "memory_clear_soak_needed" if not memory_safe_for_training and not (batch30_memory_safe or batch20_memory_safe or batch10_memory_safe or small_batch_memory_safe or small_canary_memory_safe) else "",
            "foreground_user_apps_active" if user_active and not (user_active_micro_canary_allowed or user_active_large_batch_allowed) else "",
            "writer_still_active" if not writer_idle and not writer_active_green_safe else "",
            "backlog_trend_regressing" if trend_regressing else "",
        ]
    )
    return {
        "allowed": allowed,
        "mode": mode,
        "profile": profile,
        "max_parallel_trainings": max_parallel,
        "requires_backlog_green": True,
        "green_samples": green_samples,
        "micro_backlog_green": micro_backlog_green,
        "micro_green_samples_ok": micro_green_samples_ok,
        "micro_host_pressure_allows_training": micro_host_pressure_allows_training,
        "support_hot_is_control_plane": support_hot_is_control_plane,
        "weekend_control_plane_pressure_allowed": weekend_control_plane_pressure_allowed,
        "trend_status": str(trend.get("status") or "unknown"),
        "writer_idle_required": not writer_active_green_safe,
        "writer_active_green_safe": writer_active_green_safe,
        "writer_active_small_batch_safe": writer_active_small_batch_safe,
        "memory_small_canary_safe": small_canary_memory_safe,
        "memory_small_batch_safe": small_batch_memory_safe,
        "multitasking_training_cap": multitasking_training_cap,
        "user_active_micro_canary_allowed": user_active_micro_canary_allowed,
        "user_active_small_canary_allowed": user_active_small_canary_allowed,
        "user_active_large_batch_allowed": user_active_large_batch_allowed,
        "weekend_large_batch_window": weekend_large_batch_window,
        "memory_batch10_safe": batch10_memory_safe,
        "memory_batch20_safe": batch20_memory_safe,
        "memory_batch30_safe": batch30_memory_safe,
        "batch20_runtime_wave_clear": batch20_runtime_wave_clear,
        "batch20_execution_mode": str(memory_control.get("batch20_execution_mode") or ""),
        "batch20_wave_size": _safe_int(memory_control.get("batch20_wave_size"), 0),
        "batch20_requires_between_target_memory_recheck": bool(memory_control.get("batch20_requires_between_target_memory_recheck", False)),
        "batch30_runtime_wave_clear": batch30_runtime_wave_clear,
        "batch30_execution_mode": str(memory_control.get("batch30_execution_mode") or ""),
        "batch30_wave_size": _safe_int(memory_control.get("batch30_wave_size"), 0),
        "batch30_requires_between_target_memory_recheck": bool(memory_control.get("batch30_requires_between_target_memory_recheck", False)),
        "runtime_micro_clear": runtime_micro_clear,
        "host_pressure_attribution_gate": pressure_policy,
        "blockers": blockers,
        "recommended_command": ["./scripts/ops/opsctl.sh", "coverage-gap-closer", "--apply-stage", "--launch", "--retrain-profile", profile, "--json"] if allowed else [],
    }


def _watchdog_blocking_needs(watchdog_intelligence: dict[str, Any]) -> list[dict[str, Any]]:
    blocking: list[dict[str, Any]] = []
    exact_needs = [need for need in _as_list(watchdog_intelligence.get("exact_needs")) if isinstance(need, dict)]
    all_sleeves_intentionally_held = any(
        str(need.get("status") or "").strip().lower() == "intentional_hold"
        and str(need.get("target") or "").strip().lower() in {"all_sleeves", "launcher", "all_sleeves_launcher"}
        for need in exact_needs
    )
    for need in exact_needs:
        if not isinstance(need, dict):
            continue
        status = str(need.get("status") or "").strip().lower()
        if status == "intentional_hold":
            continue
        blocker = str(need.get("blocker") or "").strip().lower()
        source = str(need.get("source") or "").strip().lower()
        risk = str(need.get("risk_level") or need.get("severity") or "").strip().lower()
        severity = str(need.get("severity") or "").strip().lower()
        low_risk_startup = bool(
            source == "all_sleeves_launcher_readiness"
            and blocker == "startup_in_progress"
            and risk in {"", "info", "low"}
        )
        informational_low_risk = bool(severity in {"info", "advisory"} and risk in {"", "info", "low"})
        intentional_launcher_exit = bool(
            all_sleeves_intentionally_held
            and source == "all_sleeves_launcher_readiness"
            and blocker in {"exited", "not_running", "stopped"}
            and risk in {"", "info", "low", "medium"}
        )
        if low_risk_startup or informational_low_risk or intentional_launcher_exit:
            continue
        blocking.append(need)
    return blocking


def _watchdog_has_active_issues(watchdog_intelligence: dict[str, Any]) -> bool:
    if not watchdog_intelligence:
        return False
    active_count = _safe_int(watchdog_intelligence.get("active_issue_count"), 0)
    storm_count = _safe_int(watchdog_intelligence.get("restart_storm_count"), 0)
    alert_count = _safe_int(watchdog_intelligence.get("alert_count"), 0)
    status = _status(watchdog_intelligence)
    blocking_needs = _watchdog_blocking_needs(watchdog_intelligence)
    return bool(active_count > 0 or storm_count > 0 or alert_count > 0 or blocking_needs or status in {"blocked", "critical"})


def _watchdog_blocks_training(watchdog_intelligence: dict[str, Any]) -> bool:
    if not watchdog_intelligence:
        return False
    storm_count = _safe_int(watchdog_intelligence.get("restart_storm_count"), 0)
    alert_count = _safe_int(watchdog_intelligence.get("alert_count"), 0)
    status = _status(watchdog_intelligence)
    blocking_needs = _watchdog_blocking_needs(watchdog_intelligence)
    return bool(storm_count > 0 or alert_count > 0 or blocking_needs or status in {"blocked", "critical"})


def _watchdog_summary(watchdog_intelligence: dict[str, Any]) -> dict[str, Any]:
    if not watchdog_intelligence:
        return {
            "overall_status": "missing",
            "grade": "",
            "score": 0.0,
            "active_issue_count": 0,
            "restart_storm_count": 0,
            "alert_count": 0,
            "exact_needs": [],
        }
    return {
        "overall_status": _status(watchdog_intelligence),
        "grade": str(watchdog_intelligence.get("grade") or ""),
        "score": _safe_float(watchdog_intelligence.get("score"), 0.0),
        "active_issue_count": _safe_int(watchdog_intelligence.get("active_issue_count"), 0),
        "restart_storm_count": _safe_int(watchdog_intelligence.get("restart_storm_count"), 0),
        "alert_count": _safe_int(watchdog_intelligence.get("alert_count"), 0),
        "blocking_exact_need_count": len(_watchdog_blocking_needs(watchdog_intelligence)),
        "blocking_exact_needs": _watchdog_blocking_needs(watchdog_intelligence),
        "exact_needs": _as_list(watchdog_intelligence.get("exact_needs")),
    }


def _budgets(
    storage_metrics: dict[str, Any],
    runtime: dict[str, Any],
    mlx: dict[str, Any],
    lanes: dict[str, Any],
    user: dict[str, Any],
    writer: dict[str, Any],
    trend: dict[str, Any],
    stability: dict[str, Any],
    watchdog_intelligence: dict[str, Any],
) -> dict[str, Any]:
    storage_severity = str(storage_metrics.get("severity") or "ready")
    runtime_status = _status(runtime)
    runtime_pressure = str(runtime.get("compute_pressure_level") or runtime.get("host_pressure_level") or runtime.get("throttle_profile") or "").lower()
    memory_pressure = str(runtime.get("memory_pressure_level") or "").lower()
    user_active = bool(user.get("user_active", False))
    backlog_pressure = storage_severity in {"critical", "degraded"}
    p_workers = _safe_int(lanes.get("selected_p_core_preprocess_workers"), 1)
    memory_control = _as_dict(_as_dict(lanes.get("p_core_widening_controller")).get("memory_pressure_controller"))
    pressure_policy = _runtime_pressure_attribution_policy(runtime)
    pressure_mode = str(pressure_policy.get("mode") or "clear").strip().lower()
    source_pressure_active = pressure_mode not in {"clear", "legacy_runtime_pressure"}
    watchdog_issue_active = _watchdog_has_active_issues(watchdog_intelligence)
    watchdog_training_blocked = _watchdog_blocks_training(watchdog_intelligence)
    pressure_active = (
        backlog_pressure
        or source_pressure_active
        or watchdog_issue_active
        or status_rank(runtime_status) >= status_rank("degraded")
        or runtime_pressure in {"high", "critical", "protect_live"}
        or memory_pressure in {"high", "red"}
    )
    collector_controller = _collector_reopening_controller(storage_metrics, runtime, user, trend, stability, memory_control, pressure_policy)
    training_gate = _training_reentry_gate(storage_metrics, runtime, user, mlx, writer, trend, stability, memory_control, pressure_policy)
    if storage_severity == "critical":
        collector_ratio = 0.12 if user_active else 0.18
        training = "paused"
        mlx_jobs = 1
        report = "freshness_only"
    elif storage_severity == "degraded":
        collector_ratio = 0.20 if user_active else 0.28
        training = "paused" if user_active or pressure_active else "small_targeted"
        mlx_jobs = 1
        report = "bounded"
    else:
        collector_ratio = 0.35 if user_active else 0.55
        training = "small_targeted" if not pressure_active else "paused"
        mlx_jobs = max(_safe_int(_as_dict(mlx.get("runtime_caps")).get("max_concurrent_mlx_jobs"), 1), 1)
        report = "normal"
    collector_ratio = min(float(collector_ratio), _safe_float(collector_controller.get("max_active_ratio"), collector_ratio))
    training = str(training_gate.get("mode") or training)
    return {
        "live_loops": {
            "mode": "protected_read_only",
            "reason": "live observation remains protected while execution is gated separately",
        },
        "runtime_pressure_source": pressure_policy,
        "backlog_writer": {
            "mode": "catch_up_waves" if backlog_pressure else "maintenance",
            "single_writer_required": True,
            "p_core_preprocess_workers": p_workers,
            "primary_merge_writer_count": 1,
            "shard_writer_lanes": p_workers,
            "writer_lane_policy": "parallel_child_shard_writers_on_p_core_budget_single_serial_primary_merge",
            "max_catch_up_waves": 3 if backlog_pressure else 1,
        },
        "collectors": {
            "mode": "duty_cycle_capped" if backlog_pressure or user_active else "normal",
            "max_active_ratio": round(collector_ratio, 3),
            "pause_optional_collectors": bool(backlog_pressure),
            "adaptive_reopening": collector_controller,
        },
        "training": {
            "mode": training,
            "allowed": bool(training_gate.get("allowed", False)) and not watchdog_training_blocked,
            "profile": str(training_gate.get("profile") or "none"),
            "reentry_gate": training_gate,
            "watchdog_training_blocked": bool(watchdog_training_blocked),
        },
        "mlx_gpu_jobs": {
            "mode": "capped" if pressure_active else "normal",
            "max_concurrent_jobs": mlx_jobs,
            "compile_mode": "off" if pressure_active else _as_dict(mlx.get("runtime_caps")).get("compile_mode", "canary_first"),
        },
        "reports": {"mode": report},
        "maintenance": {"mode": "after_writer" if backlog_pressure else "normal"},
        "watchdogs": {
            "mode": "repair_first" if watchdog_issue_active else "observe",
            "healthy": not watchdog_issue_active,
            "summary": _watchdog_summary(watchdog_intelligence),
            "policy": "watchdog_intelligence_must_be_clean_before_widening_training_or_optional_collector_work",
        },
    }


def _need_items(
    storage_metrics: dict[str, Any],
    runtime: dict[str, Any],
    mlx: dict[str, Any],
    host: dict[str, Any],
    adapter: dict[str, Any],
    budgets: dict[str, Any],
    memory_intelligence: dict[str, Any],
    pressure_policy: dict[str, Any],
    watchdog_intelligence: dict[str, Any],
) -> list[dict[str, Any]]:
    needs: list[dict[str, Any]] = []
    watchdog_needs = _watchdog_blocking_needs(watchdog_intelligence)
    if watchdog_needs:
        need = watchdog_needs[0]
        needs.append(
            {
                "blocker": f"watchdog_{need.get('blocker') or need.get('status') or 'needs_repair'}",
                "exact_file": str(need.get("exact_file") or "governance/health/watchdog_intelligence_latest.json"),
                "exact_shard": str(need.get("target") or ""),
                "command": _as_list(need.get("exact_command")) or ["./scripts/ops/opsctl.sh", "watchdog-intelligence", "--apply", "--json"],
                "current": {
                    "target": need.get("target", ""),
                    "status": need.get("status", ""),
                    "watchdog_status": _status(watchdog_intelligence),
                    "restart_storm_count": _safe_int(watchdog_intelligence.get("restart_storm_count"), 0),
                },
                "expected_impact": str(need.get("expected_impact") or "Repairs watchdog coordination before adding more runtime work."),
                "risk_level": str(need.get("risk_level") or "low"),
                "stop_when": str(need.get("when_to_stop") or "watchdog exact_needs is empty and restart_storm_count is zero."),
            }
        )
    if storage_metrics["severity"] in {"critical", "degraded", "watch"}:
        oldest = storage_metrics.get("oldest_sources", [])
        exact = oldest[0] if oldest and isinstance(oldest[0], dict) else {}
        needs.append(
            {
                "blocker": "backlog_above_target_or_old_pending_work",
                "exact_file": exact.get("source_rel") or "",
                "exact_shard": exact.get("shard") or "",
                "current": {
                    "core_pending_lines": storage_metrics["core_pending_lines"],
                    "total_pending_lines": storage_metrics["total_pending_lines"],
                    "oldest_pending_age_seconds": storage_metrics["oldest_pending_age_seconds"],
                },
                "command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"],
                "expected_impact": "Runs bounded catch-up waves through the active single-writer drainer until pending deltas flatten.",
                "risk_level": "low",
                "stop_when": f"core pending is below {storage_metrics['target_pending_lines']} and oldest pending age is under {int(BACKLOG_GREEN_AGE_SECONDS / 60)} minutes.",
            }
        )
    if status_rank(_status(runtime)) >= status_rank("degraded") and str(pressure_policy.get("mode") or "").strip().lower() not in {"operator_foreground_advisory", "support_maintenance_advisory"}:
        pressure_mode = str(pressure_policy.get("mode") or "").strip().lower()
        if pressure_mode == "macos_system_cooldown":
            blocker = "runtime_pressure_macos_system_cooldown"
            expected = "Keeps bot widening, optional collectors, and training paused while macOS services such as Spotlight or backup jobs cool."
            stop = "host_pressure_attribution.system_cotenant_hot is false and runtime status is ready/advisory."
        elif pressure_mode == "external_cotenant_cooldown":
            blocker = "runtime_pressure_external_cotenant_dominant"
            expected = "Preserves foreground, user, and operator-app headroom before widening collectors, training, or P-core workers."
            stop = "host_pressure_attribution.external_pressure_dominant is false and runtime status is ready/advisory."
        elif pressure_mode == "trim_support_maintenance":
            blocker = "runtime_pressure_support_maintenance_hot"
            expected = "Trims support maintenance first so backlog and live-critical lanes keep priority without crowding the computer."
            stop = "host_pressure_attribution.support_jobs_hot is false and runtime status is ready/advisory."
        elif pressure_mode == "protect_live_or_macro_hot":
            blocker = "runtime_pressure_protected_live_or_macro_hot"
            expected = "Avoids adding training or optional collector work while protected live, paper, or macro lanes are hot."
            stop = "host_pressure_attribution.protected_work_hot is false and runtime status is ready/advisory."
        else:
            blocker = "runtime_pressure_or_soft_cap_active"
            expected = "Refreshes process priority, P-core feedback, and co-tenant headroom before more work is launched."
            stop = "runtime status is ready/advisory and memory pressure is normal."
        needs.append(
            {
                "blocker": blocker,
                "exact_file": "governance/health/runtime_throttle_control_latest.json",
                "exact_shard": "",
                "command": ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
                "current": {
                    "mode": pressure_mode or "unknown",
                    "dominant_bucket": pressure_policy.get("dominant_bucket", "unknown"),
                    "host_saturation_score": pressure_policy.get("host_saturation_score", 0.0),
                    "compute_pressure_level": pressure_policy.get("compute_pressure_level", ""),
                    "memory_pressure_level": pressure_policy.get("memory_pressure_level", ""),
                },
                "expected_impact": expected,
                "risk_level": "low",
                "stop_when": stop,
            }
        )
    memory_classification = _as_dict(memory_intelligence.get("classification"))
    memory_gate = _as_dict(memory_intelligence.get("reopen_gate"))
    memory_status = str(memory_classification.get("status") or "").strip().lower()
    if memory_status not in {"", "clear"} and not bool(memory_gate.get("safe_to_widen_p_core_workers", False)):
        needs.append(
            {
                "blocker": "memory_headroom_not_ready_for_widening",
                "exact_file": "governance/health/memory_pressure_intelligence_latest.json",
                "exact_shard": "",
                "current": {
                    "memory_status": memory_status,
                    "p_core_worker_cap": _safe_int(memory_classification.get("recommended_p_core_worker_cap"), 0),
                    "clear_samples": _safe_int(memory_gate.get("consecutive_memory_clear_samples"), 0),
                },
                "command": ["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--apply", "--json"],
                "expected_impact": "Refreshes the memory headroom gate used before P-core worker widening, collector reopening, or training.",
                "risk_level": "low",
                "stop_when": "memory pressure is clear for two consecutive samples and foreground headroom is no longer blocking.",
            }
        )
    if _status(mlx) in {"blocked", "missing"} or _as_dict(mlx.get("runtime_caps")).get("compile_mode") == "off":
        needs.append(
            {
                "blocker": "mlx_or_gpu_lane_capped",
                "exact_file": "governance/health/mlx_intelligence_router_latest.json",
                "exact_shard": "",
                "command": ["./scripts/ops/opsctl.sh", "mlx-intelligence-router", "--apply", "--json"],
                "expected_impact": "Keeps GPU/MLX jobs capped until host and backlog pressure cool, then restores safe canary-first usage.",
                "risk_level": "low",
                "stop_when": "MLX route is ready and max_concurrent_jobs matches governor budget.",
            }
        )
    if not host:
        needs.append(
            {
                "blocker": "host_capability_contract_missing",
                "exact_file": "governance/health/host_capability_contract_latest.json",
                "exact_shard": "",
                "command": ["./scripts/ops/opsctl.sh", "host-capability", "--json"],
                "expected_impact": "Publishes the body map used by OS adapters and portable runtime limits.",
                "risk_level": "none",
                "stop_when": "host capability contract exists and reports ready/advisory.",
            }
        )
    if adapter and "host_capability_contract_missing" in _as_list(adapter.get("capability_gaps")):
        needs.append(
            {
                "blocker": "os_adapter_waiting_for_host_contract",
                "exact_file": "governance/health/os_adapter_layer_latest.json",
                "exact_shard": "",
                "command": ["./scripts/ops/opsctl.sh", "os-adapter", "--json"],
                "expected_impact": "Refreshes OS-specific actions for priority, services, memory, GPU, and disk probes.",
                "risk_level": "none",
                "stop_when": "adapter no longer reports host contract missing.",
            }
        )
    return needs


def _operator_action_packet(needs: list[dict[str, Any]], writer: dict[str, Any], storage_metrics: dict[str, Any], budgets: dict[str, Any], trend: dict[str, Any], stability: dict[str, Any]) -> dict[str, Any]:
    state = _writer_state(writer)
    planned = _safe_int(state.get("planned_shard_count"), 0)
    completed = _safe_int(state.get("completed_shard_count"), 0)
    writer_active = _writer_active(writer)
    training_gate = _as_dict(_as_dict(budgets.get("training")).get("reentry_gate"))
    if writer_active:
        next_command = ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"]
        posture = "observe_active_writer"
        recheck_seconds = 45
        reason = f"writer is active at {completed}/{planned} shards; avoid duplicate writer launches"
    elif needs:
        next_command = _as_list(needs[0].get("command"))
        posture = "run_next_needed_command"
        recheck_seconds = 60
        reason = str(needs[0].get("blocker") or "needs_action")
    elif training_gate.get("allowed"):
        next_command = _as_list(training_gate.get("recommended_command"))
        posture = "training_reentry_allowed"
        recheck_seconds = 300
        reason = "green gate and runtime stability permit one small targeted training"
    else:
        next_command = ["./scripts/ops/opsctl.sh", "autonomic-governor", "--apply", "--json"]
        posture = "steady_observe"
        recheck_seconds = 300
        reason = "no urgent action; keep refreshing the governor contract"
    return {
        "posture": posture,
        "next_command": next_command,
        "recheck_seconds": recheck_seconds,
        "reason": reason,
        "writer_shards": {"completed": completed, "planned": planned, "current_step": state.get("current_step", ""), "status": state.get("status", "")},
        "stop_conditions": [
            f"oldest pending age below {int(BACKLOG_GREEN_AGE_SECONDS / 60)} minutes",
            f"core pending below {storage_metrics.get('target_pending_lines')}",
            "no active writer before applying a new writer cycle",
        ],
        "trend_status": str(trend.get("status") or "unknown"),
        "green_samples": _safe_int(stability.get("consecutive_green_samples"), 0),
    }


def _env_lines(budgets: dict[str, Any], lanes: dict[str, Any]) -> list[str]:
    collectors = _as_dict(budgets.get("collectors"))
    writer = _as_dict(budgets.get("backlog_writer"))
    training = _as_dict(budgets.get("training"))
    mlx = _as_dict(budgets.get("mlx_gpu_jobs"))
    pressure_source = _as_dict(budgets.get("runtime_pressure_source"))
    memory_control = _as_dict(_as_dict(lanes.get("p_core_widening_controller")).get("memory_pressure_controller"))
    allocation = _as_dict(lanes.get("p_core_allocation_contract"))
    reserve_target = _safe_int(
        allocation.get("user_app_reserve_target_p_cores"),
        _safe_int(os.getenv("AUTONOMIC_PCORE_USER_APP_RESERVE_TARGET"), _safe_int(os.getenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET"), 0)),
    )
    env = {
        "AUTONOMIC_RESOURCE_GOVERNOR_ENABLED": "1",
        "AUTONOMIC_BACKLOG_WRITER_MODE": str(writer.get("mode") or "maintenance"),
        "AUTONOMIC_PCORE_PREPROCESS_WORKERS": str(writer.get("p_core_preprocess_workers") or lanes.get("selected_p_core_preprocess_workers") or 1),
        "SQL_LINK_SERVICE_PREPROCESS_WORKERS": str(writer.get("p_core_preprocess_workers") or lanes.get("selected_p_core_preprocess_workers") or 1),
        "SQL_LINK_SERVICE_SHARD_WRITER_LANES": str(writer.get("shard_writer_lanes") or writer.get("p_core_preprocess_workers") or lanes.get("selected_p_core_preprocess_workers") or 1),
        "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": str(min(max(_safe_int(_as_dict(lanes.get("p_core_widening_controller")).get("max_safe_workers"), lanes.get("selected_p_core_preprocess_workers") or 1), 1), 8)),
        "SQL_LINK_CHILD_WRITER_CPU_POLICY": "performance_core_primary",
        "SQL_LINK_WRITER_BACKGROUND_POLICY": "0",
        "SQL_LINK_WRITER_NICE": "0",
        "AUTONOMIC_MAX_CATCH_UP_WAVES": str(writer.get("max_catch_up_waves") or 1),
        "AUTONOMIC_BACKLOG_GREEN": "1" if bool(writer.get("backlog_green", False)) else "0",
        "AUTONOMIC_BACKLOG_GREEN_AGE_SECONDS": str(int(BACKLOG_GREEN_AGE_SECONDS)),
        "AUTONOMIC_PCORE_WIDENING_MODE": str(_as_dict(lanes.get("p_core_widening_controller")).get("mode") or "steady"),
        "AUTONOMIC_PCORE_MAX_SAFE_WORKERS": str(_as_dict(lanes.get("p_core_widening_controller")).get("max_safe_workers") or lanes.get("selected_p_core_preprocess_workers") or 1),
        "AUTONOMIC_PCORE_NEXT_WORKERS_WHEN_IDLE": str(_as_dict(lanes.get("p_core_widening_controller")).get("prearmed_next_workers_when_idle") or lanes.get("selected_p_core_preprocess_workers") or 1),
        "AUTONOMIC_PCORE_MEMORY_STATUS": str(memory_control.get("status") or "clear"),
        "AUTONOMIC_PCORE_MEMORY_MAX_WORKERS": str(memory_control.get("max_memory_safe_workers") or lanes.get("selected_p_core_preprocess_workers") or 1),
        "AUTONOMIC_PCORE_MEMORY_SAFE_TO_WIDEN": "1" if memory_control.get("safe_to_widen_p_core_workers") else "0",
        "AUTONOMIC_PCORE_MEMORY_SAFE_FOR_TRAINING": "1" if memory_control.get("safe_for_training") else "0",
        "AUTONOMIC_PCORE_MEMORY_CLEAR_SAMPLES": str(memory_control.get("memory_clear_samples") or 0),
        "AUTONOMIC_RUNTIME_PRESSURE_SOURCE_MODE": str(pressure_source.get("mode") or "clear"),
        "AUTONOMIC_RUNTIME_PRESSURE_DOMINANT_BUCKET": str(pressure_source.get("dominant_bucket") or "unknown"),
        "AUTONOMIC_RUNTIME_PRESSURE_ALLOWS_TRAINING": "1" if pressure_source.get("training_allowed", True) else "0",
        "AUTONOMIC_RUNTIME_PRESSURE_ALLOWS_COLLECTORS": "1" if pressure_source.get("collector_reopen_allowed", True) else "0",
        "AUTONOMIC_ECORE_GUARD_MODE": str(_as_dict(lanes.get("efficiency_core_pressure_guard")).get("mode") or "bounded_background_spillover"),
        "AUTONOMIC_ECORE_SPILLOVER_WORKERS": str(lanes.get("efficiency_core_spillover") or 0),
        "BOT_EFFICIENCY_CORE_SPILLOVER_COUNT": str(lanes.get("efficiency_core_spillover") or 0),
        "BOT_CPU_ALLOCATION_POLICY": str(lanes.get("policy") or "performance_core_primary_single_writer_with_user_app_reserve"),
        "BOT_CPU_QOS_POLICY": "performance_core_primary_no_background_writer",
        "AUTONOMIC_PCORE_SYSTEM_WORKERS": str(allocation.get("system_primary_workers") or lanes.get("selected_p_core_preprocess_workers") or 1),
        "AUTONOMIC_PCORE_USER_APP_RESERVE": str(allocation.get("user_app_reserved_p_cores") or 0),
        "AUTONOMIC_PCORE_USER_APP_RESERVE_TARGET": str(reserve_target),
        "BACKLOG_PCORE_USER_APP_RESERVE_TARGET": str(reserve_target),
        "BACKLOG_PCORE_USE_FULL_PERFORMANCE_CORE_BUDGET": "1" if _as_dict(lanes.get("p_core_widening_controller")).get("full_p_core_budget_requested") else "0",
        "BACKLOG_SLEEVE_PUMP_ENABLED": str(os.getenv("BACKLOG_SLEEVE_PUMP_ENABLED") or "0"),
        "BACKLOG_SLEEVE_PUMP_WORKERS": str(os.getenv("BACKLOG_SLEEVE_PUMP_WORKERS") or "1"),
        "BACKLOG_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES": str(os.getenv("BACKLOG_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES") or lanes.get("selected_p_core_preprocess_workers") or 1),
        "SQL_LINK_SERVICE_SLEEVE_PUMP_ENABLED": str(os.getenv("SQL_LINK_SERVICE_SLEEVE_PUMP_ENABLED") or os.getenv("BACKLOG_SLEEVE_PUMP_ENABLED") or "0"),
        "SQL_LINK_SERVICE_SLEEVE_PUMP_WORKERS": str(os.getenv("SQL_LINK_SERVICE_SLEEVE_PUMP_WORKERS") or os.getenv("BACKLOG_SLEEVE_PUMP_WORKERS") or "1"),
        "SQL_LINK_SERVICE_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES": str(os.getenv("SQL_LINK_SERVICE_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES") or os.getenv("BACKLOG_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES") or lanes.get("selected_p_core_preprocess_workers") or 1),
        "AUTONOMIC_BACKLOG_TREND_STATUS": str(writer.get("backlog_trend_status") or "unknown"),
        "AUTONOMIC_BACKLOG_GREEN_SAMPLES": str(writer.get("green_samples") or 0),
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": str(collectors.get("max_active_ratio") or 0.2),
        "AUTONOMIC_COLLECTOR_REOPEN_STAGE": str(_as_dict(collectors.get("adaptive_reopening")).get("stage") or "protect_core"),
        "AUTONOMIC_COLLECTOR_ROLLBACK_IF_AGE_RISES": "1" if _as_dict(collectors.get("adaptive_reopening")).get("rollback_enabled", True) else "0",
        "AUTONOMIC_TRAINING_MODE": str(training.get("mode") or "paused"),
        "AUTONOMIC_TRAINING_ALLOWED": "1" if training.get("allowed") else "0",
        "AUTONOMIC_TRAINING_REENTRY_PROFILE": str(_as_dict(training.get("reentry_gate")).get("profile") or "none"),
        "AUTONOMIC_MLX_MAX_CONCURRENT_JOBS": str(mlx.get("max_concurrent_jobs") or 1),
        "AUTONOMIC_MLX_COMPILE_MODE": str(mlx.get("compile_mode") or "off"),
    }
    return [f"{key}={shlex.quote(value)}" for key, value in env.items()]


def write_outputs(payload: dict[str, Any], *, out_path: Path = DEFAULT_OUT_PATH, override_path: Path = DEFAULT_OVERRIDE_PATH, apply: bool = False) -> dict[str, Any]:
    write_payload(out_path, payload)
    applied = False
    if apply:
        lines = [
            "# Managed by scripts/ops/autonomic_resource_governor.py",
            f"# updated_at_utc={payload.get('timestamp_utc')}",
            *_env_lines(_as_dict(payload.get("budgets")), _as_dict(payload.get("host_lane_budget"))),
            "",
        ]
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text("\n".join(lines), encoding="utf-8")
        applied = True
    return {"out_path": str(out_path), "override_path": str(override_path), "applied": applied}


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    previous = load_json(health / "autonomic_resource_governor_latest.json")
    host = load_json(health / "host_capability_contract_latest.json")
    adapter = load_json(health / "os_adapter_layer_latest.json")
    workload = load_json(health / "workload_class_registry_latest.json")
    runtime = load_json(health / "runtime_throttle_control_latest.json")
    storage = load_json(health / "ingestion_storage_control_latest.json")
    writer = load_json(health / "writer_cycle_coordinator_latest.json")
    drainer = load_json(health / "backpressure_drainer_fleet_latest.json")
    mlx = load_json(health / "mlx_intelligence_router_latest.json")
    computer = load_json(health / "computer_task_intelligence_latest.json")
    benchmark = load_json(health / "host_self_benchmark_latest.json")
    memory_intelligence = load_json(health / "memory_pressure_intelligence_latest.json")
    capital_growth_awareness = load_json(health / "capital_growth_awareness_bridge_latest.json")
    watchdog_raw = load_json(health / "watchdog_intelligence_latest.json")
    process_watchdog = load_json(health / "process_watchdog_latest.json")
    watchdog_intelligence = (
        _as_dict(watchdog_raw)
        if watchdog_raw
        else _as_dict(process_watchdog.get("watchdog_intelligence"))
    )
    storage_metrics = _storage_metrics(storage)
    current_seed = {"timestamp_utc": iso_now(), "storage_metrics": storage_metrics}
    trend = _backlog_trend(previous, current_seed, writer)
    stability = _stability_state(storage_metrics, runtime, writer, trend, previous)
    lanes = _p_core_widening_controller(host, runtime, benchmark, memory_intelligence, writer, storage_metrics, trend, stability)
    user = _user_context(computer, host)
    budgets = _budgets(storage_metrics, runtime, mlx, lanes, user, writer, trend, stability, watchdog_intelligence)
    pressure_policy = _as_dict(budgets.get("runtime_pressure_source")) or _runtime_pressure_attribution_policy(runtime)
    budgets["backlog_writer"]["backlog_green"] = bool(storage_metrics.get("green", False))
    budgets["backlog_writer"]["green_gate"] = _as_dict(storage_metrics.get("green_gate"))
    budgets["backlog_writer"]["backlog_trend_status"] = str(trend.get("status") or "unknown")
    budgets["backlog_writer"]["green_samples"] = _safe_int(stability.get("consecutive_green_samples"), 0)
    needs = _need_items(storage_metrics, runtime, mlx, host, adapter, budgets, memory_intelligence, pressure_policy, watchdog_intelligence)
    statuses = {
        "host_capability": _status(host),
        "os_adapter": _status(adapter),
        "workload_registry": _status(workload),
        "watchdog_intelligence": _status(watchdog_intelligence),
        "runtime_throttle": _status(runtime),
        "ingestion_storage": _status(storage),
        "writer_cycle": _status(writer),
        "drainer_fleet": _status(drainer),
        "mlx_router": _status(mlx),
        "computer_task": _status(computer),
        "capital_growth_awareness": _status(capital_growth_awareness),
    }
    worst = max(status_rank(status) for status in statuses.values()) if statuses else 0
    if storage_metrics["severity"] == "critical":
        overall = "degraded"
    elif needs:
        overall = "advisory"
    elif worst >= status_rank("blocked"):
        overall = "degraded"
    else:
        overall = "ready"
    action_packet = _operator_action_packet(needs, writer, storage_metrics, budgets, trend, stability)
    return {
        "timestamp_utc": current_seed["timestamp_utc"],
        "schema_version": 1,
        "ok": overall in {"ready", "advisory"},
        "overall_status": overall,
        "unified_decision": "backlog_recovery" if storage_metrics["severity"] in {"critical", "degraded"} else "balanced_headroom",
        "statuses": statuses,
        "storage_metrics": storage_metrics,
        "backlog_trend": trend,
        "stability_state": stability,
        "host_lane_budget": lanes,
        "runtime_pressure_source": pressure_policy,
        "memory_pressure_intelligence": {
            "overall_status": _status(memory_intelligence),
            "classification": _as_dict(memory_intelligence.get("classification")),
            "trend": _as_dict(memory_intelligence.get("trend")),
            "reopen_gate": _as_dict(memory_intelligence.get("reopen_gate")),
            "multitasking_headroom": _as_dict(memory_intelligence.get("multitasking_headroom")),
            "observer_overhead": _as_dict(memory_intelligence.get("observer_overhead")),
        },
        "watchdog_intelligence": _watchdog_summary(watchdog_intelligence),
        "capital_growth_awareness": {
            "overall_status": _status(capital_growth_awareness),
            "capital_growth_grade": str(capital_growth_awareness.get("capital_growth_grade") or ""),
            "live_money_scaling_allowed": bool(capital_growth_awareness.get("live_money_scaling_allowed", False)),
            "awareness_scope": _as_dict(capital_growth_awareness.get("awareness_scope")),
            "live_money_scaling_blockers": _as_list(capital_growth_awareness.get("live_money_scaling_blockers"))[:8],
        },
        "user_context": user,
        "budgets": budgets,
        "backlog_green_gate": _as_dict(storage_metrics.get("green_gate")),
        "adaptive_controls": {
            "p_core_widening": _as_dict(lanes.get("p_core_widening_controller")),
            "efficiency_core_pressure_guard": _as_dict(lanes.get("efficiency_core_pressure_guard")),
            "collector_reopening": _as_dict(_as_dict(budgets.get("collectors")).get("adaptive_reopening")),
            "training_reentry": _as_dict(_as_dict(budgets.get("training")).get("reentry_gate")),
        },
        "operator_action_packet": action_packet,
        "what_do_you_need": {
            "status": "needs_action" if needs else "clear",
            "items": needs,
            "next_command": action_packet.get("next_command", []),
        },
        "integration_contract": {
            "reads_host_capability_contract": True,
            "reads_os_adapter_layer": True,
            "reads_workload_class_registry": True,
            "coordinates_live_backlog_collectors_training_mlx_reports": True,
            "protects_user_foreground_apps": True,
            "reads_memory_pressure_intelligence": True,
            "reads_watchdog_intelligence": True,
            "reads_capital_growth_awareness": True,
            "uses_runtime_pressure_attribution": True,
            "never_touch_protected_volumes": ["/Volumes/VIDEO"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish and optionally apply the autonomic resource governor contract.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override", default=str(DEFAULT_OVERRIDE_PATH))
    args = parser.parse_args()
    payload = build_payload(PROJECT_ROOT)
    result = write_outputs(payload, out_path=Path(args.out), override_path=Path(args.override), apply=args.apply)
    payload["write_result"] = result
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"autonomic_resource_governor status={payload['overall_status']} decision={payload['unified_decision']} applied={result['applied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
