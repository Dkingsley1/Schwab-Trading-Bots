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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, LOCAL_TZ, iso_now, load_json, ordered_unique, parse_iso_utc, utc_now, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, LOCAL_TZ, iso_now, load_json, ordered_unique, parse_iso_utc, utc_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "memory_pressure_intelligence_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.memory_pressure_intelligence_override"
CREATIVE_REALTIME_APPS = ("logic", "final cut", "fcpx", "garageband", "ableton", "pro tools", "davinci")
INTERACTIVE_DEVELOPER_APPS = ("pycharm", "xcode", "chrome", "safari", "terminal", "iterm", "codex")
MEDIA_PLAYBACK_APPS = ("music", "itunes", "spotify", "vlc", "quicktime")
OBSERVER_PROCESS_MARKERS = ("asitop", "btop", "powermetrics", "activity monitor")
WEEKEND_TRAINING_ENV = "TRAINING_WEEKEND_BATCH_WINDOW"


def _weekend_training_window_active() -> bool:
    override = str(os.environ.get(WEEKEND_TRAINING_ENV) or "").strip().lower()
    if override in {"1", "true", "yes", "on", "force"}:
        return True
    if override in {"0", "false", "no", "off", "disable"}:
        return False
    try:
        return utc_now().astimezone(LOCAL_TZ).weekday() >= 5
    except Exception:
        return False


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


def _status(payload: dict[str, Any]) -> str:
    for key in ("overall_status", "status"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    if payload.get("ok") is True:
        return "ready"
    if payload.get("ok") is False:
        return "blocked"
    return "missing"


def _max_numeric(*values: Any) -> float:
    return max((_safe_float(value, 0.0) for value in values), default=0.0)


def _payload_newer_or_equal(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    candidate_ts = parse_iso_utc(candidate.get("timestamp_utc"))
    reference_ts = parse_iso_utc(reference.get("timestamp_utc"))
    return bool(candidate_ts is not None and reference_ts is not None and candidate_ts >= reference_ts)


def _non_idle_level(*values: Any) -> bool:
    return any(str(value or "").strip().lower() not in {"", "none", "idle"} for value in values)


def _memory_snapshot(host: dict[str, Any], runtime: dict[str, Any], memory_efficiency: dict[str, Any], computer: dict[str, Any]) -> dict[str, Any]:
    body = _as_dict(host.get("body_map"))
    memory = _as_dict(body.get("memory"))
    host_snapshot = _as_dict(memory.get("memory_snapshot"))
    runtime_snapshot = _as_dict(runtime.get("runtime_snapshot"))
    vm_stat = _as_dict(runtime_snapshot.get("vm_stat"))
    p_feedback = _as_dict(runtime.get("p_core_runtime_feedback"))
    burst = _as_dict(p_feedback.get("p_core_burst_intelligence"))
    burst_inputs = _as_dict(burst.get("inputs"))
    efficiency_snapshot = _as_dict(memory_efficiency.get("memory_snapshot"))
    cotenant = _as_dict(memory_efficiency.get("cotenant_awareness"))
    session = _as_dict(computer.get("session_context"))
    foreground = _as_dict(body.get("foreground_apps_and_user_activity"))
    infrabot = _as_dict(session.get("process_context_infrabot"))

    session_apps = [str(item) for item in _as_list(session.get("open_apps"))]
    cotenant_apps = [str(item) for item in _as_list(cotenant.get("open_apps"))]
    foreground_apps = [str(item) for item in _as_list(foreground.get("open_apps"))]
    computer_supersedes_host = bool(session and _payload_newer_or_equal(computer, host))
    ignore_host_foreground = bool(computer_supersedes_host and foreground_apps != session_apps)
    ignore_memory_efficiency_apps = bool(infrabot.get("ignored_memory_efficiency_app_context", False))
    open_apps = ordered_unique(
        [
            *([] if ignore_memory_efficiency_apps else cotenant_apps),
            *session_apps,
            *([] if ignore_host_foreground else foreground_apps),
        ]
    )
    compressed_store_gb = _max_numeric(
        host_snapshot.get("compressed_store_gb"),
        efficiency_snapshot.get("compressed_store_gb"),
        burst_inputs.get("compressed_store_gb"),
    )
    compressor_gb = _max_numeric(
        memory.get("compressor_gb"),
        host_snapshot.get("compressor_gb"),
        efficiency_snapshot.get("compressor_gb"),
        burst_inputs.get("compressor_gb"),
    )
    compressed_pressure_gb = compressor_gb if compressor_gb > 0 else compressed_store_gb
    swap_gb = _max_numeric(
        memory.get("swap_used_gb"),
        host_snapshot.get("swap_used_gb"),
        efficiency_snapshot.get("swap_used_gb"),
        burst_inputs.get("swap_used_gb"),
    )
    pages_throttled = _max_numeric(
        runtime_snapshot.get("vm_pages_throttled"),
        vm_stat.get("pages_throttled"),
        burst_inputs.get("pages_throttled"),
    )
    pressure_level = str(runtime.get("memory_pressure_level") or memory.get("pressure_level") or "normal").strip().lower()
    pressure_kind = str(
        host_snapshot.get("memory_pressure_kind")
        or efficiency_snapshot.get("memory_pressure_kind")
        or burst_inputs.get("memory_pressure_kind")
        or "none"
    ).strip().lower()
    top_processes = _as_list(runtime.get("top_processes")) or _as_list(runtime_snapshot.get("top_processes"))
    memory_top = sorted(
        [row for row in top_processes if isinstance(row, dict)],
        key=lambda row: _safe_float(row.get("mem_percent"), 0.0),
        reverse=True,
    )[:8]
    cotenant_creative_level = "" if ignore_memory_efficiency_apps else cotenant.get("creative_level")
    cotenant_co_running_level = "" if ignore_memory_efficiency_apps else cotenant.get("co_running_level")
    foreground_creative_level = "" if ignore_host_foreground else foreground.get("creative_level")
    foreground_co_running_level = "" if ignore_host_foreground else foreground.get("co_running_level")
    user_active = bool(
        open_apps
        or _non_idle_level(cotenant_creative_level, session.get("creative_level"), foreground_creative_level)
        or _non_idle_level(cotenant_co_running_level, session.get("co_running_level"), foreground_co_running_level)
    )
    return {
        "memory_gb": _safe_float(memory.get("memory_gb"), 0.0),
        "pressure_level": pressure_level,
        "pressure_kind": pressure_kind,
        "memory_free_pct": round(
            _max_numeric(
                memory.get("memory_free_pct"),
                host_snapshot.get("memory_free_pct"),
                efficiency_snapshot.get("memory_free_pct"),
                burst_inputs.get("memory_free_pct"),
            ),
            3,
        ),
        "swap_used_gb": round(swap_gb, 3),
        "compressed_store_gb": round(compressed_store_gb, 3),
        "compressed_pressure_gb": round(compressed_pressure_gb, 3),
        "compressor_gb": round(compressor_gb, 3),
        "pages_throttled": pages_throttled,
        "runtime_burst_mode": str(burst.get("mode") or "").strip().lower(),
        "memory_efficiency_status": _status(memory_efficiency),
        "open_apps": open_apps[:12],
        "user_active": user_active,
        "top_memory_processes": memory_top,
        "app_context_quality": {
            "session_context_used": bool(session),
            "computer_task_supersedes_host": computer_supersedes_host,
            "ignored_stale_host_foreground": ignore_host_foreground,
            "ignored_memory_efficiency_app_context": ignore_memory_efficiency_apps,
            "host_open_apps": foreground_apps[:12],
            "session_open_apps": session_apps[:12],
            "memory_efficiency_open_apps": cotenant_apps[:12],
        },
    }


def _reconcile_stale_allocation(snapshot: dict[str, Any], swap_pressure_payload: dict[str, Any]) -> dict[str, Any]:
    swap_pressure = _as_dict(swap_pressure_payload.get("swap_pressure"))
    raw_snapshot = dict(snapshot)
    raw_swap_gb = _safe_float(snapshot.get("swap_used_gb"), 0.0)
    current_swap_gb = _safe_float(swap_pressure.get("swap_used_gb"), raw_swap_gb)
    compressed_store_gb = _safe_float(snapshot.get("compressed_store_gb"), 0.0)
    compressor_gb = _safe_float(snapshot.get("compressor_gb"), 0.0)
    free_pct = _safe_float(snapshot.get("memory_free_pct"), 0.0)
    pages_throttled = _safe_float(snapshot.get("pages_throttled"), 0.0)
    pressure_kind = str(snapshot.get("pressure_kind") or "none").strip().lower()
    swap_state = str(swap_pressure.get("memory_pressure_state") or "").strip().lower()
    swap_kind = str(swap_pressure.get("memory_pressure_kind") or "").strip().lower()
    swap_tier = str(swap_pressure.get("tier") or "").strip().lower()
    governor_green = bool(
        swap_pressure
        and swap_tier == "normal"
        and swap_state in {"green", "normal", "none", "clear"}
        and swap_kind in {"", "none", "normal", "green", "clear"}
    )
    vm_clear = bool(pressure_kind in {"", "none", "normal", "green", "clear"} and pages_throttled <= 0.0)
    stale_swap_relief = bool(governor_green and vm_clear and raw_swap_gb >= 2.0 and current_swap_gb + 0.5 < raw_swap_gb)
    inactive_swap_allocation_relief = bool(
        governor_green
        and vm_clear
        and raw_swap_gb >= 2.0
        and (free_pct <= 0.0 or free_pct >= 85.0)
        and compressor_gb <= 1.5
    )
    stale_compression_relief = bool(
        governor_green
        and vm_clear
        and stale_swap_relief
        and (free_pct <= 0.0 or free_pct >= 85.0)
        and current_swap_gb < 3.0
        and compressed_store_gb >= 18.0
        and compressor_gb < 1.5
    )
    if stale_swap_relief:
        snapshot["swap_used_gb"] = round(current_swap_gb, 3)
    if stale_compression_relief:
        snapshot["compressed_store_gb"] = round(max(compressor_gb, 8.0), 3)
    effective_swap_pressure_gb = (
        0.0
        if inactive_swap_allocation_relief
        else _safe_float(snapshot.get("swap_used_gb"), raw_swap_gb)
    )
    snapshot["effective_swap_pressure_gb"] = round(effective_swap_pressure_gb, 3)
    snapshot["swap_allocation_only"] = inactive_swap_allocation_relief
    if stale_swap_relief or stale_compression_relief or inactive_swap_allocation_relief:
        snapshot["pressure_level"] = "normal"
        snapshot["pressure_kind"] = "none"
        snapshot["compressed_pressure_gb"] = round(compressor_gb if compressor_gb > 0 else _safe_float(snapshot.get("compressed_store_gb"), 0.0), 3)
        snapshot["allocation_relief_active"] = True
    snapshot["raw_allocation_snapshot"] = {
        "pressure_level": str(raw_snapshot.get("pressure_level") or ""),
        "pressure_kind": str(raw_snapshot.get("pressure_kind") or ""),
        "swap_used_gb": round(raw_swap_gb, 3),
        "compressed_store_gb": round(compressed_store_gb, 3),
        "compressed_pressure_gb": round(_safe_float(raw_snapshot.get("compressed_pressure_gb"), 0.0), 3),
        "compressor_gb": round(compressor_gb, 3),
    }
    snapshot["memory_truth_reconciliation"] = {
        "active": bool(stale_swap_relief or stale_compression_relief or inactive_swap_allocation_relief),
        "stale_swap_relief": stale_swap_relief,
        "stale_compression_relief": stale_compression_relief,
        "inactive_swap_allocation_relief": inactive_swap_allocation_relief,
        "raw_swap_used_gb": round(raw_swap_gb, 3),
        "effective_swap_used_gb": round(_safe_float(snapshot.get("swap_used_gb"), raw_swap_gb), 3),
        "effective_swap_pressure_gb": round(effective_swap_pressure_gb, 3),
        "raw_compressed_store_gb": round(compressed_store_gb, 3),
        "effective_compressed_store_gb": round(_safe_float(snapshot.get("compressed_store_gb"), compressed_store_gb), 3),
        "compressor_gb": round(compressor_gb, 3),
        "free_pct": round(free_pct, 3),
        "swap_pressure_tier": swap_tier,
        "reason": (
            "fresh_vm_signals_classified_retained_swap_as_allocation_only"
            if inactive_swap_allocation_relief and not (stale_swap_relief or stale_compression_relief)
            else (
                "fresh_swap_pressure_reconciled_stale_allocation"
                if stale_swap_relief or stale_compression_relief
                else "not_applicable"
            )
        ),
        "policy": "classification and reopen gates use effective memory when fresh green VM evidence contradicts stale high-water allocation",
    }
    return snapshot


def _match_app_class(app: str) -> str:
    lowered = app.lower()
    if any(marker in lowered for marker in CREATIVE_REALTIME_APPS):
        return "creative_realtime"
    if any(marker in lowered for marker in INTERACTIVE_DEVELOPER_APPS):
        return "interactive_developer"
    if any(marker in lowered for marker in MEDIA_PLAYBACK_APPS):
        return "media_playback"
    return "foreground_other"


def _multitasking_headroom(snapshot: dict[str, Any]) -> dict[str, Any]:
    open_apps = [str(item) for item in _as_list(snapshot.get("open_apps")) if str(item).strip()]
    classes = ordered_unique([_match_app_class(app) for app in open_apps])
    weekend_window = _weekend_training_window_active()
    weekend_media_window = bool(weekend_window and classes == ["media_playback"])
    if "creative_realtime" in classes:
        level = "realtime_creative"
        cap = 3
        p_core_reserve = 5
        collector_ratio = 0.12
        training_allowed = False
        training_cap = 0
        hard_training_block = True
        reason = "Logic/Final Cut-style foreground work needs the largest memory and scheduler reserve"
    elif "interactive_developer" in classes:
        level = "interactive_developer"
        cap = 5
        p_core_reserve = 3
        collector_ratio = 0.28
        training_allowed = True
        training_cap = 2
        hard_training_block = False
        reason = "developer apps keep a reserve, but guarded small-canary training may run when memory permits"
    elif "media_playback" in classes:
        level = "media_playback"
        cap = 6 if weekend_media_window else 5
        p_core_reserve = 1 if weekend_media_window else 3
        collector_ratio = 0.45 if weekend_media_window else 0.35
        training_allowed = True
        training_cap = 30 if weekend_media_window else 4
        hard_training_block = False
        reason = (
            "weekend media-only foreground activity permits memory-guarded batch training waves"
            if weekend_media_window
            else "media playback keeps a light reserve, but small targeted training may run when memory permits"
        )
    elif open_apps:
        level = "foreground_standard"
        cap = 5
        p_core_reserve = 3
        collector_ratio = 0.28
        training_allowed = True
        training_cap = 2
        hard_training_block = False
        reason = "foreground app activity gets a general reserve, but guarded small-canary training may run"
    else:
        level = "background_available"
        cap = 7
        p_core_reserve = 1
        collector_ratio = 0.55
        training_allowed = True
        training_cap = 30
        hard_training_block = False
        reason = "no foreground apps are asking for extra headroom, so deep-green P7 backlog bursts may be armed"
    return {
        "active": bool(open_apps),
        "level": level,
        "open_apps": open_apps[:12],
        "app_classes": classes,
        "recommended_p_core_cap": cap,
        "recommended_user_p_core_reserve": p_core_reserve,
        "collector_ratio_cap": collector_ratio,
        "training_allowed_by_multitasking": training_allowed,
        "training_hard_block_by_multitasking": hard_training_block,
        "training_max_parallel_trainings": training_cap,
        "weekend_training_window": weekend_window,
        "weekend_media_training_window": weekend_media_window,
        "recommended_background_posture": "foreground_headroom" if open_apps else "normal_background",
        "reason": reason,
        "policy": "foreground_apps_get_memory_and_scheduler_reserve_before_backlog_or_training_widening",
    }


def _observer_overhead(snapshot: dict[str, Any]) -> dict[str, Any]:
    offenders: list[dict[str, Any]] = []
    for row in _as_list(snapshot.get("top_memory_processes")):
        if not isinstance(row, dict):
            continue
        command = str(row.get("command") or "").lower()
        if not any(marker in command for marker in OBSERVER_PROCESS_MARKERS):
            continue
        cpu = _safe_float(row.get("cpu_percent"), 0.0)
        mem = _safe_float(row.get("mem_percent"), 0.0)
        if cpu >= 5.0 or mem >= 5.0:
            offenders.append(
                {
                    "pid": _safe_int(row.get("pid"), 0),
                    "cpu_percent": round(cpu, 3),
                    "mem_percent": round(mem, 3),
                    "command_excerpt": str(row.get("command") or "")[:240],
                }
            )
    total_cpu = round(sum(_safe_float(row.get("cpu_percent"), 0.0) for row in offenders), 3)
    total_mem = round(sum(_safe_float(row.get("mem_percent"), 0.0) for row in offenders), 3)
    return {
        "active": bool(offenders),
        "offenders": offenders[:6],
        "total_cpu_percent": total_cpu,
        "total_mem_percent": total_mem,
        "status": "distorting_observation" if offenders else "clear",
        "reason": "observer tools are consuming enough resources to affect the reading" if offenders else "observer tools are not materially affecting pressure",
        "policy": "report_observer_overhead_separately_so_monitoring_does_not_get_mistaken_for_bot_pressure",
    }


def _memory_trend(previous: dict[str, Any], current_snapshot: dict[str, Any], current_ts: str) -> dict[str, Any]:
    prior_snapshot = _as_dict(previous.get("snapshot"))
    previous_ts = parse_iso_utc(previous.get("timestamp_utc")) if previous else None
    current_dt = parse_iso_utc(current_ts)
    elapsed_seconds = None
    if previous_ts is not None and current_dt is not None:
        elapsed_seconds = max((current_dt - previous_ts).total_seconds(), 0.0)

    previous_compressed_pressure = _safe_float(
        prior_snapshot.get("compressed_pressure_gb"),
        _safe_float(prior_snapshot.get("compressed_store_gb"), 0.0),
    )
    current_compressed_pressure = _safe_float(
        current_snapshot.get("compressed_pressure_gb"),
        _safe_float(current_snapshot.get("compressed_store_gb"), 0.0),
    )
    compressed_delta = round(previous_compressed_pressure - current_compressed_pressure, 3)
    previous_swap_pressure = _safe_float(
        prior_snapshot.get("effective_swap_pressure_gb"),
        _safe_float(prior_snapshot.get("swap_used_gb"), 0.0),
    )
    current_swap_pressure = _safe_float(
        current_snapshot.get("effective_swap_pressure_gb"),
        _safe_float(current_snapshot.get("swap_used_gb"), 0.0),
    )
    swap_delta = round(previous_swap_pressure - current_swap_pressure, 3)
    throttled_delta = round(_safe_float(prior_snapshot.get("pages_throttled"), 0.0) - _safe_float(current_snapshot.get("pages_throttled"), 0.0), 3)
    has_previous = bool(prior_snapshot)
    cooling = bool(has_previous and (compressed_delta >= 0.35 or swap_delta >= 0.1) and throttled_delta >= 0)
    heating = bool(has_previous and (compressed_delta <= -0.35 or swap_delta <= -0.1 or throttled_delta < 0))
    if not has_previous:
        status = "baseline"
    elif heating:
        status = "heating"
    elif cooling:
        status = "cooling"
    else:
        status = "flat"
    previous_gate = _as_dict(previous.get("reopen_gate"))
    previous_clear = _safe_int(previous_gate.get("consecutive_memory_clear_samples"), 0)
    previous_cooling = _safe_int(previous_gate.get("consecutive_cooling_samples"), 0)
    return {
        "status": status,
        "has_previous_sample": has_previous,
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
        "compressed_delta_gb": compressed_delta,
        "swap_delta_gb": swap_delta,
        "pages_throttled_delta": throttled_delta,
        "cooling": cooling,
        "heating": heating,
        "previous_clear_samples": previous_clear,
        "previous_cooling_samples": previous_cooling,
    }


def _classify(snapshot: dict[str, Any], trend: dict[str, Any], multitasking: dict[str, Any]) -> dict[str, Any]:
    level = str(snapshot.get("pressure_level") or "normal").lower()
    kind = str(snapshot.get("pressure_kind") or "none").lower()
    swap_gb = _safe_float(
        snapshot.get("effective_swap_pressure_gb"),
        _safe_float(snapshot.get("swap_used_gb"), 0.0),
    )
    compressed_gb = _safe_float(snapshot.get("compressed_pressure_gb"), _safe_float(snapshot.get("compressed_store_gb"), 0.0))
    pages_throttled = _safe_float(snapshot.get("pages_throttled"), 0.0)
    burst_mode = str(snapshot.get("runtime_burst_mode") or "").lower()
    user_active = bool(snapshot.get("user_active", False))
    multitask_cap = _safe_int(multitasking.get("recommended_p_core_cap"), 6)
    multitask_level = str(multitasking.get("level") or "background_available")
    heating = bool(trend.get("heating", False))
    allocation_only_high = bool(
        level in {"high", "critical"}
        and kind in {"", "none", "normal"}
        and pages_throttled <= 0
        and _safe_float(snapshot.get("memory_free_pct"), 0.0) >= 70.0
        and swap_gb < 3.0
    )
    if (
        (level in {"red", "high", "critical"} and not allocation_only_high)
        or kind in {"red", "critical", "throttled"}
        or pages_throttled > 0
    ):
        status = "hard_relief"
        cap = 2
        decision = "protect_memory_now"
        reason = "memory pressure is high or VM pages are throttled"
    elif (burst_mode.startswith("memory_relief_2") and not allocation_only_high) or swap_gb >= 8.0:
        status = "swap_relief"
        cap = 2
        decision = "protect_memory_now"
        reason = "swap is high or runtime requested strong memory relief"
    elif (burst_mode.startswith("memory_relief_3") and not allocation_only_high) or compressed_gb >= 14.0 or swap_gb >= 4.0:
        status = "compression_relief"
        cap = 3
        decision = "cool_compression_before_widening"
        reason = "compressed memory is elevated enough to hold P-core width"
    elif compressed_gb >= 10.0 or swap_gb >= 2.0 or level == "elevated":
        status = "soft_guard"
        cap = 4
        decision = "cooldown_probe_only"
        reason = "unified memory is warm, so widening should be gradual"
    elif user_active:
        status = "foreground_headroom"
        cap = multitask_cap
        decision = "preserve_user_app_headroom"
        reason = str(multitasking.get("reason") or "foreground apps are active, so the system should keep spare memory headroom")
    else:
        status = "clear"
        deep_clear = bool(
            level == "normal"
            and kind in {"", "none", "normal"}
            and pages_throttled <= 0.0
            and swap_gb < 2.0
            and compressed_gb < 8.0
            and not heating
        )
        cap = 7 if deep_clear else 6
        decision = "safe_to_widen_after_soak"
        reason = (
            "memory pressure is deep-green enough to arm the seventh P-core backlog burst"
            if deep_clear
            else "memory pressure is clear enough for normal widening"
        )
    if heating and cap > 3:
        cap = min(cap, 4)
        if status == "clear":
            status = "heating_guard"
            decision = "hold_while_memory_heats"
            reason = "memory was clear, but the trend is heating, so widening waits"
    cap = min(cap, multitask_cap)
    return {
        "status": status,
        "decision": decision,
        "recommended_p_core_worker_cap": cap,
        "multitasking_level": multitask_level,
        "reason": reason,
    }


def _reopen_gate(classification: dict[str, Any], trend: dict[str, Any], snapshot: dict[str, Any], multitasking: dict[str, Any], observer: dict[str, Any]) -> dict[str, Any]:
    clear_statuses = {"clear", "foreground_headroom"}
    status = str(classification.get("status") or "")
    current_clear = status in clear_statuses and not bool(trend.get("heating", False))
    clear_samples = _safe_int(trend.get("previous_clear_samples"), 0) + 1 if current_clear else 0
    cooling_samples = _safe_int(trend.get("previous_cooling_samples"), 0) + 1 if bool(trend.get("cooling", False)) else 0
    safe_to_widen = bool(clear_samples >= 2)
    safe_for_training = bool(clear_samples >= 3 and str(classification.get("status")) == "clear")
    pressure_level = str(snapshot.get("pressure_level") or "normal").lower()
    pressure_kind = str(snapshot.get("pressure_kind") or "none").lower()
    allocation_only_pressure = bool(
        pressure_level in {"high", "critical"}
        and pressure_kind in {"", "none", "normal"}
        and _safe_float(snapshot.get("pages_throttled"), 0.0) <= 0.0
        and _safe_float(snapshot.get("memory_free_pct"), 0.0) >= 70.0
        and _safe_float(snapshot.get("swap_used_gb"), 0.0) < 3.0
    )
    memory_normal = bool(
        pressure_level == "normal" or allocation_only_pressure
    ) and pressure_kind in {"", "none", "normal"} and _safe_float(snapshot.get("pages_throttled"), 0.0) <= 0.0
    free_pct = _safe_float(snapshot.get("memory_free_pct"), 0.0)
    swap_gb = _safe_float(
        snapshot.get("effective_swap_pressure_gb"),
        _safe_float(snapshot.get("swap_used_gb"), 0.0),
    )
    compressed_gb = _safe_float(snapshot.get("compressed_pressure_gb"), _safe_float(snapshot.get("compressed_store_gb"), 0.0))
    multitasking_allows = bool(multitasking.get("training_allowed_by_multitasking", True))
    trend_heating = bool(trend.get("heating", False))
    observer_clear = not bool(observer.get("active", False))
    prior_clear_samples = _safe_int(trend.get("previous_clear_samples"), 0)
    sequential_guard_soak = max(clear_samples, prior_clear_samples if status == "heating_guard" else 0)
    weekend_large_batch_window = bool(
        status == "soft_guard"
        and bool(multitasking.get("weekend_media_training_window", False))
        and multitasking_allows
        and observer_clear
        and not trend_heating
        and memory_normal
        and (free_pct <= 0.0 or free_pct >= 90.0)
        and compressed_gb < 12.0
        and swap_gb < 3.0
        and _safe_int(classification.get("recommended_p_core_worker_cap"), 0) >= 3
    )
    compression_relief_micro_canary_safe = bool(
        status == "compression_relief"
        and not trend_heating
        and str(snapshot.get("pressure_level") or "normal").lower() not in {"red", "high", "critical"}
        and str(snapshot.get("pressure_kind") or "none").lower() in {"", "none", "normal"}
        and _safe_float(snapshot.get("pages_throttled"), 0.0) <= 0.0
        and swap_gb < 3.0
        and compressed_gb < 20.0
        and (free_pct <= 0.0 or free_pct >= 70.0)
        and multitasking_allows
        and observer_clear
    )
    single_sample_deep_green = bool(
        current_clear
        and memory_normal
        and observer_clear
        and multitasking_allows
        and not trend_heating
        and (free_pct <= 0.0 or free_pct >= 85.0)
        and compressed_gb < 9.0
        and swap_gb < 1.5
        and _safe_int(classification.get("recommended_p_core_worker_cap"), 0) >= 6
    )
    foreground_soft_guard_micro_safe = bool(
        status == "soft_guard"
        and multitasking_allows
        and _safe_int(multitasking.get("training_max_parallel_trainings"), 0) >= 1
        and observer_clear
        and not trend_heating
        and memory_normal
        and _safe_float(snapshot.get("pages_throttled"), 0.0) <= 0.0
        and swap_gb < 2.25
        and compressed_gb < 14.5
        and (free_pct <= 0.0 or free_pct >= 80.0)
    )
    foreground_soft_guard_small_safe = bool(
        status == "soft_guard"
        and multitasking_allows
        and _safe_int(multitasking.get("training_max_parallel_trainings"), 0) >= 2
        and observer_clear
        and not trend_heating
        and memory_normal
        and _safe_float(snapshot.get("pages_throttled"), 0.0) <= 0.0
        and swap_gb < 2.35
        and compressed_gb < 15.25
        and (free_pct <= 0.0 or free_pct >= 80.0)
    )
    small_canary_safe = bool(
        (
            status in {"clear", "foreground_headroom", "soft_guard"}
            and not trend_heating
            and memory_normal
            and swap_gb < 2.0
            and compressed_gb < 12.0
            and multitasking_allows
        )
        or compression_relief_micro_canary_safe
        or foreground_soft_guard_micro_safe
    )
    small_batch_safe = bool(
        (
            small_canary_safe
            and observer_clear
            and compressed_gb < 11.5
            and swap_gb < 1.5
            and status in {"clear", "foreground_headroom", "soft_guard"}
        )
        or foreground_soft_guard_small_safe
    )
    batch10_strict_safe = bool(
        safe_for_training
        and observer_clear
        and memory_normal
        and not trend_heating
        and multitasking_allows
        and clear_samples >= 4
        and compressed_gb < 9.5
        and swap_gb < 1.0
    )
    batch10_wave_safe = bool(
        (
            status in {"clear", "heating_guard"}
            and observer_clear
            and memory_normal
            and multitasking_allows
            and (sequential_guard_soak >= 3 or single_sample_deep_green)
            and (free_pct <= 0.0 or free_pct >= 70.0)
            and compressed_gb < 9.75
            and swap_gb < 1.75
            and _safe_int(classification.get("recommended_p_core_worker_cap"), 0) >= 4
        )
        or weekend_large_batch_window
    )
    batch10_safe = bool(batch10_strict_safe or batch10_wave_safe)
    batch20_strict_safe = bool(
        batch10_strict_safe
        and clear_samples >= 5
        and compressed_gb < 7.5
        and swap_gb < 0.5
        and _safe_int(classification.get("recommended_p_core_worker_cap"), 0) >= 6
    )
    clear_or_heating_batch20_wave_safe = bool(
        not batch20_strict_safe
        and status in {"clear", "heating_guard"}
        and observer_clear
        and memory_normal
        and multitasking_allows
        and (sequential_guard_soak >= 4 or single_sample_deep_green)
        and (free_pct <= 0.0 or free_pct >= 75.0)
        and compressed_gb < 9.5
        and swap_gb < 1.75
        and _safe_int(classification.get("recommended_p_core_worker_cap"), 0) >= 4
    )
    compression_relief_batch20_wave_safe = bool(
        not batch20_strict_safe
        and status == "compression_relief"
        and compression_relief_micro_canary_safe
        and observer_clear
        and multitasking_allows
        and not trend_heating
        and str(snapshot.get("pressure_level") or "normal").lower() not in {"red", "high", "critical"}
        and str(snapshot.get("pressure_kind") or "none").lower() in {"", "none", "normal"}
        and _safe_float(snapshot.get("pages_throttled"), 0.0) <= 0.0
        and (free_pct <= 0.0 or free_pct >= 80.0)
        and compressed_gb < 20.0
        and swap_gb < 2.25
        and _safe_int(classification.get("recommended_p_core_worker_cap"), 0) >= 3
    )
    weekend_soft_guard_batch20_wave_safe = bool(weekend_large_batch_window)
    batch20_wave_safe = bool(clear_or_heating_batch20_wave_safe or compression_relief_batch20_wave_safe or weekend_soft_guard_batch20_wave_safe)
    batch20_safe = bool(batch20_strict_safe or batch20_wave_safe)
    batch20_wave_size = (
        20
        if batch20_strict_safe
        else min(max(_safe_int(classification.get("recommended_p_core_worker_cap"), 3), 1), 4)
        if compression_relief_batch20_wave_safe
        else 4
        if batch20_wave_safe and not batch10_strict_safe
        else 10
        if batch20_wave_safe
        else 0
    )
    batch20_execution_mode = (
        "strict_deep_green"
        if batch20_strict_safe
        else "sequential_memory_guarded_waves"
        if batch20_wave_safe
        else ""
    )
    batch30_strict_safe = bool(
        batch20_strict_safe
        and clear_samples >= 6
        and compressed_gb < 6.75
        and swap_gb < 0.5
        and _safe_int(classification.get("recommended_p_core_worker_cap"), 0) >= 6
    )
    batch30_wave_safe = bool(
        (
            not batch30_strict_safe
            and batch20_safe
            and observer_clear
            and multitasking_allows
            and not trend_heating
            and str(snapshot.get("pressure_level") or "normal").lower() not in {"red", "high", "critical"}
            and str(snapshot.get("pressure_kind") or "none").lower() in {"", "none", "normal"}
            and _safe_float(snapshot.get("pages_throttled"), 0.0) <= 0.0
            and (free_pct <= 0.0 or free_pct >= 80.0)
            and compressed_gb < 20.0
            and swap_gb < 2.25
            and _safe_int(classification.get("recommended_p_core_worker_cap"), 0) >= 3
        )
        or bool(weekend_large_batch_window and batch20_safe)
    )
    batch30_safe = bool(batch30_strict_safe or batch30_wave_safe)
    batch30_wave_size = (
        30
        if batch30_strict_safe
        else min(max(_safe_int(classification.get("recommended_p_core_worker_cap"), 3), 1), 4)
        if status in {"compression_relief", "soft_guard"} and batch30_wave_safe
        else 5
        if batch30_wave_safe
        else 0
    )
    batch30_execution_mode = (
        "strict_deep_green"
        if batch30_strict_safe
        else "sequential_memory_guarded_waves"
        if batch30_wave_safe
        else ""
    )
    training_batch_cap = (
        30
        if batch30_safe
        else 20
        if batch20_safe
        else 10
        if batch10_safe
        else 4
        if safe_for_training
        else 2
        if small_batch_safe
        else 1
        if small_canary_safe
        else 0
    )
    multitasking_training_cap = max(_safe_int(multitasking.get("training_max_parallel_trainings"), 30), 0)
    training_batch_cap = min(training_batch_cap, multitasking_training_cap)
    if multitasking_training_cap < 30:
        batch30_safe = bool(batch30_safe and multitasking_training_cap >= 30)
        batch20_safe = bool(batch20_safe and multitasking_training_cap >= 20)
        batch10_safe = bool(batch10_safe and multitasking_training_cap >= 10)
        safe_for_training = bool(safe_for_training and multitasking_training_cap >= 4)
        small_batch_safe = bool(small_batch_safe and multitasking_training_cap >= 2)
        small_canary_safe = bool(small_canary_safe and multitasking_training_cap >= 1)
    training_profile = (
        "coverage_batch30_canary"
        if batch30_safe
        else "coverage_batch20_canary"
        if batch20_safe
        else "coverage_batch10_canary"
        if batch10_safe
        else "coverage_canary"
        if safe_for_training
        else "coverage_small_canary"
        if small_batch_safe
        else "coverage_micro_canary"
        if small_canary_safe
        else ""
    )
    return {
        "consecutive_memory_clear_samples": clear_samples,
        "consecutive_cooling_samples": cooling_samples,
        "memory_clear_required_samples_for_widening": 2,
        "memory_clear_required_samples_for_training": 3,
        "safe_to_widen_p_core_workers": safe_to_widen,
        "safe_for_training": safe_for_training,
        "small_canary_training_safe": small_canary_safe,
        "small_canary_max_parallel_trainings": 1 if small_canary_safe else 0,
        "small_canary_profile": "coverage_micro_canary" if small_canary_safe and not (small_batch_safe or safe_for_training) else "",
        "small_batch_training_safe": small_batch_safe,
        "small_batch_max_parallel_trainings": 2 if small_batch_safe else 0,
        "small_batch_profile": "coverage_small_canary" if small_batch_safe and not safe_for_training else "",
        "batch10_training_safe": batch10_safe,
        "batch10_max_parallel_trainings": 10 if batch10_safe else 0,
        "batch10_strict_training_safe": batch10_strict_safe,
        "batch10_wave_training_safe": batch10_wave_safe,
        "batch10_profile": "coverage_batch10_canary" if batch10_safe and not batch20_safe else "",
        "batch20_training_safe": batch20_safe,
        "batch20_max_parallel_trainings": 20 if batch20_safe else 0,
        "batch20_strict_training_safe": batch20_strict_safe,
        "batch20_wave_training_safe": batch20_wave_safe,
        "compression_relief_batch20_wave_training_safe": compression_relief_batch20_wave_safe,
        "weekend_large_batch_window": weekend_large_batch_window,
        "weekend_soft_guard_batch20_wave_training_safe": weekend_soft_guard_batch20_wave_safe,
        "batch20_execution_mode": batch20_execution_mode,
        "batch20_wave_size": batch20_wave_size,
        "batch20_requires_between_target_memory_recheck": bool(batch20_wave_safe),
        "batch30_training_safe": batch30_safe,
        "batch30_max_parallel_trainings": 30 if batch30_safe else 0,
        "batch30_strict_training_safe": batch30_strict_safe,
        "batch30_wave_training_safe": batch30_wave_safe,
        "batch30_execution_mode": batch30_execution_mode,
        "batch30_wave_size": batch30_wave_size,
        "batch30_requires_between_target_memory_recheck": bool(batch30_wave_safe),
        "weekend_soft_guard_batch30_wave_training_safe": bool(weekend_large_batch_window and batch30_safe),
        "single_sample_deep_green_batch_widening": single_sample_deep_green,
        "foreground_soft_guard_micro_canary_safe": foreground_soft_guard_micro_safe,
        "foreground_soft_guard_small_canary_safe": foreground_soft_guard_small_safe,
        "compression_relief_micro_canary_safe": compression_relief_micro_canary_safe,
        "batch30_profile": "coverage_batch30_canary" if batch30_safe else "",
        "batch20_profile": "coverage_batch20_canary" if batch20_safe else "",
        "training_batch_cap": training_batch_cap,
        "training_profile": training_profile,
        "multitasking_training_cap": multitasking_training_cap,
        "observer_clear_for_larger_batches": observer_clear,
        "cooling_before_widening": not safe_to_widen,
        "policy": "wide_training_requires_clear_soak; weekend media-only soft-guard windows may use sequential memory-guarded waves when swap, throttle, trend, and observer gates are clear",
    }


def _what_do_you_need(snapshot: dict[str, Any], classification: dict[str, Any], gate: dict[str, Any], observer: dict[str, Any]) -> list[dict[str, Any]]:
    needs: list[dict[str, Any]] = []
    if str(classification.get("status") or "") not in {"clear", "foreground_headroom"}:
        needs.append(
            {
                "blocker": "memory_compression_or_swap_above_widening_gate",
                "exact_file": "governance/health/memory_pressure_intelligence_latest.json",
                "exact_shard": "",
                "command": ["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--apply", "--json"],
                "expected_impact": "Refreshes the memory cap, cooldown gate, and P-core worker limit before the next backlog or training decision.",
                "risk_level": "low",
                "stop_when": "memory status is clear for two consecutive samples and compressed/swap trend is not heating.",
            }
        )
        needs.append(
            {
                "blocker": "runtime_memory_profile_needs_refresh",
                "exact_file": "governance/health/runtime_throttle_control_latest.json",
                "exact_shard": "",
                "command": ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
                "expected_impact": "Refreshes process priority and runtime memory profile using the latest VM/compression state.",
                "risk_level": "low",
                "stop_when": "runtime memory pressure is normal and no pages are throttled.",
            }
        )
    elif not bool(gate.get("safe_to_widen_p_core_workers")):
        needs.append(
            {
                "blocker": "memory_clear_soak_not_finished",
                "exact_file": "governance/health/memory_pressure_intelligence_latest.json",
                "exact_shard": "",
                "command": ["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--json"],
                "expected_impact": "Adds another memory sample so the governor can verify memory stayed clear before widening.",
                "risk_level": "none",
                "stop_when": "two consecutive clear memory samples have been observed.",
            }
        )
    app_classes = ordered_unique([_match_app_class(app) for app in _as_list(snapshot.get("open_apps"))])
    foreground_headroom_managed = bool(
        bool(snapshot.get("user_active", False))
        and str(classification.get("status")) == "foreground_headroom"
        and bool(gate.get("safe_to_widen_p_core_workers"))
        and str(snapshot.get("pressure_level") or "normal").strip().lower() == "normal"
        and str(snapshot.get("pressure_kind") or "none").strip().lower() in {"", "none", "normal"}
        and _safe_float(snapshot.get("pages_throttled"), 0.0) <= 0.0
        and _safe_float(
            snapshot.get("effective_swap_pressure_gb"),
            _safe_float(snapshot.get("swap_used_gb"), 0.0),
        ) < 2.0
        and _safe_float(snapshot.get("compressed_pressure_gb"), 0.0) < 8.0
        and app_classes
        and set(app_classes).issubset({"media_playback"})
    )
    if (
        bool(snapshot.get("user_active", False))
        and str(classification.get("status")) == "foreground_headroom"
        and not foreground_headroom_managed
    ):
        needs.append(
            {
                "blocker": "foreground_app_headroom_reserved",
                "exact_file": "governance/health/computer_task_intelligence_latest.json",
                "exact_shard": "",
                "command": ["./scripts/ops/opsctl.sh", "computer-task-intelligence", "--json"],
                "expected_impact": "Confirms foreground app activity before releasing more memory headroom to backlog workers.",
                "risk_level": "none",
                "stop_when": "foreground creative/developer apps are idle or closed.",
            }
        )
    if bool(observer.get("active", False)):
        needs.append(
            {
                "blocker": "observer_tool_overhead_high",
                "exact_file": "governance/health/memory_pressure_intelligence_latest.json",
                "exact_shard": "",
                "command": [],
                "expected_impact": "Closing or slowing high-overhead observers such as asitop lowers the pressure created by monitoring itself.",
                "risk_level": "operator_choice",
                "stop_when": "observer_overhead.active is false or observer memory/CPU falls below the reporting threshold.",
            }
        )
    return needs


def _managed_controls(snapshot: dict[str, Any], classification: dict[str, Any], gate: dict[str, Any]) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    app_classes = ordered_unique([_match_app_class(app) for app in _as_list(snapshot.get("open_apps"))])
    if bool(snapshot.get("user_active", False)) and str(classification.get("status")) == "foreground_headroom":
        controls.append(
            {
                "id": "foreground_app_headroom_reserved",
                "status": "managed",
                "open_apps": _as_list(snapshot.get("open_apps"))[:12],
                "app_classes": app_classes,
                "safe_to_widen_p_core_workers": bool(gate.get("safe_to_widen_p_core_workers")),
                "training_batch_cap": _safe_int(gate.get("training_batch_cap"), 0),
                "policy": "reserve foreground app headroom by capping background workers instead of asking the operator to close benign apps",
            }
        )
    return controls


def _env_lines(payload: dict[str, Any]) -> list[str]:
    snapshot = _as_dict(payload.get("snapshot"))
    classification = _as_dict(payload.get("classification"))
    gate = _as_dict(payload.get("reopen_gate"))
    multitasking = _as_dict(payload.get("multitasking_headroom"))
    observer = _as_dict(payload.get("observer_overhead"))
    env = {
        "MEMORY_PRESSURE_INTELLIGENCE_ENABLED": "1",
        "MEMORY_PRESSURE_HEADROOM_STATUS": str(classification.get("status") or "unknown"),
        "MEMORY_PRESSURE_DECISION": str(classification.get("decision") or "observe"),
        "MEMORY_PRESSURE_PCORE_WORKER_CAP": str(classification.get("recommended_p_core_worker_cap") or 1),
        "MEMORY_PRESSURE_SAFE_TO_WIDEN": "1" if gate.get("safe_to_widen_p_core_workers") else "0",
        "MEMORY_PRESSURE_SAFE_FOR_TRAINING": "1" if gate.get("safe_for_training") else "0",
        "MEMORY_PRESSURE_SMALL_CANARY_SAFE": "1" if gate.get("small_canary_training_safe") else "0",
        "MEMORY_PRESSURE_SMALL_BATCH_SAFE": "1" if gate.get("small_batch_training_safe") else "0",
        "MEMORY_PRESSURE_BATCH10_SAFE": "1" if gate.get("batch10_training_safe") else "0",
        "MEMORY_PRESSURE_BATCH20_SAFE": "1" if gate.get("batch20_training_safe") else "0",
        "MEMORY_PRESSURE_BATCH20_EXECUTION_MODE": str(gate.get("batch20_execution_mode") or ""),
        "MEMORY_PRESSURE_BATCH20_WAVE_SIZE": str(gate.get("batch20_wave_size") or 0),
        "MEMORY_PRESSURE_BATCH20_RECHECK_BETWEEN_TARGETS": "1" if gate.get("batch20_requires_between_target_memory_recheck") else "0",
        "MEMORY_PRESSURE_BATCH30_SAFE": "1" if gate.get("batch30_training_safe") else "0",
        "MEMORY_PRESSURE_BATCH30_EXECUTION_MODE": str(gate.get("batch30_execution_mode") or ""),
        "MEMORY_PRESSURE_BATCH30_WAVE_SIZE": str(gate.get("batch30_wave_size") or 0),
        "MEMORY_PRESSURE_BATCH30_RECHECK_BETWEEN_TARGETS": "1" if gate.get("batch30_requires_between_target_memory_recheck") else "0",
        "MEMORY_PRESSURE_TRAINING_BATCH_CAP": str(gate.get("training_batch_cap") or 0),
        "MEMORY_PRESSURE_TRAINING_PROFILE": str(gate.get("training_profile") or ""),
        "MEMORY_PRESSURE_FREE_PCT": str(snapshot.get("memory_free_pct") or 0.0),
        "MEMORY_PRESSURE_COMPRESSED_STORE_GB": str(snapshot.get("compressed_store_gb") or 0.0),
        "MEMORY_PRESSURE_COMPRESSED_PRESSURE_GB": str(snapshot.get("compressed_pressure_gb") or snapshot.get("compressed_store_gb") or 0.0),
        "MEMORY_PRESSURE_SWAP_USED_GB": str(snapshot.get("swap_used_gb") or 0.0),
        "MEMORY_PRESSURE_EFFECTIVE_SWAP_GB": str(
            snapshot.get("effective_swap_pressure_gb")
            if snapshot.get("effective_swap_pressure_gb") is not None
            else snapshot.get("swap_used_gb") or 0.0
        ),
        "MEMORY_PRESSURE_CLEAR_SAMPLES": str(gate.get("consecutive_memory_clear_samples") or 0),
        "MULTITASKING_HEADROOM_LEVEL": str(multitasking.get("level") or "background_available"),
        "MULTITASKING_COLLECTOR_RATIO_CAP": str(multitasking.get("collector_ratio_cap") or 0.55),
        "MULTITASKING_TRAINING_ALLOWED": "1" if multitasking.get("training_allowed_by_multitasking") else "0",
        "OBSERVER_OVERHEAD_ACTIVE": "1" if observer.get("active") else "0",
    }
    return [f"{key}={shlex.quote(value)}" for key, value in env.items()]


def write_outputs(
    payload: dict[str, Any],
    *,
    out_path: Path = DEFAULT_OUT_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    apply: bool = False,
) -> dict[str, Any]:
    write_payload(out_path, payload)
    applied = False
    if apply:
        lines = [
            "# Managed by scripts/ops/memory_pressure_intelligence.py",
            f"# updated_at_utc={payload.get('timestamp_utc')}",
            *_env_lines(payload),
            "",
        ]
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text("\n".join(lines), encoding="utf-8")
        applied = True
    return {"out_path": str(out_path), "override_path": str(override_path), "applied": applied}


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    previous = load_json(health / "memory_pressure_intelligence_latest.json")
    host = load_json(health / "host_capability_contract_latest.json")
    runtime = load_json(health / "runtime_throttle_control_latest.json")
    memory_efficiency = load_json(health / "memory_efficiency_control_latest.json") or load_json(health / "memory_efficiency_latest.json")
    computer = load_json(health / "computer_task_intelligence_latest.json")
    swap_pressure = load_json(health / "swap_pressure_governor_latest.json")
    timestamp = iso_now()
    snapshot = _reconcile_stale_allocation(_memory_snapshot(host, runtime, memory_efficiency, computer), swap_pressure)
    multitasking = _multitasking_headroom(snapshot)
    observer = _observer_overhead(snapshot)
    trend = _memory_trend(previous, snapshot, timestamp)
    classification = _classify(snapshot, trend, multitasking)
    gate = _reopen_gate(classification, trend, snapshot, multitasking, observer)
    if bool(multitasking.get("training_hard_block_by_multitasking", False)) or not bool(multitasking.get("training_allowed_by_multitasking", True)):
        gate["safe_for_training"] = False
        gate["small_canary_training_safe"] = False
        gate["small_batch_training_safe"] = False
        gate["batch10_training_safe"] = False
        gate["batch20_training_safe"] = False
        gate["batch30_training_safe"] = False
        gate["small_canary_max_parallel_trainings"] = 0
        gate["small_batch_max_parallel_trainings"] = 0
        gate["batch10_max_parallel_trainings"] = 0
        gate["batch20_max_parallel_trainings"] = 0
        gate["batch30_max_parallel_trainings"] = 0
        gate["training_batch_cap"] = 0
        gate["training_profile"] = ""
        gate["training_blocked_by_multitasking"] = True
    needs = _what_do_you_need(snapshot, classification, gate, observer)
    managed_controls = _managed_controls(snapshot, classification, gate)
    overall = "ready" if not needs and classification["status"] == "clear" else "advisory"
    return {
        "timestamp_utc": timestamp,
        "schema_version": 1,
        "ok": overall == "ready",
        "overall_status": overall,
        "mode": "memory_pressure_intelligence",
        "snapshot": snapshot,
        "multitasking_headroom": multitasking,
        "observer_overhead": observer,
        "trend": trend,
        "classification": classification,
        "reopen_gate": gate,
        "workload_guidance": {
            "p_core_preprocess_worker_cap": classification["recommended_p_core_worker_cap"],
            "collector_bias": "protect_core" if classification["recommended_p_core_worker_cap"] <= 3 else "cautious_reopen",
            "training_allowed_by_memory": bool(gate.get("safe_for_training")),
            "small_canary_training_allowed_by_memory": bool(gate.get("small_canary_training_safe")),
            "small_batch_training_allowed_by_memory": bool(gate.get("small_batch_training_safe")),
            "batch10_training_allowed_by_memory": bool(gate.get("batch10_training_safe")),
            "batch20_training_allowed_by_memory": bool(gate.get("batch20_training_safe")),
            "batch30_training_allowed_by_memory": bool(gate.get("batch30_training_safe")),
            "training_batch_cap": _safe_int(gate.get("training_batch_cap"), 0),
            "training_profile": str(gate.get("training_profile") or ""),
            "mlx_compile_allowed_by_memory": bool(gate.get("safe_for_training")),
            "policy": "memory_intelligence_feeds_autonomic_governor_before_cpu_width_or_training_reopen",
        },
        "what_do_you_need": {
            "status": "needs_action" if needs else "clear",
            "items": needs,
            "next_command": needs[0]["command"] if needs else [],
        },
        "managed_controls": managed_controls,
        "integration_contract": {
            "feeds_autonomic_resource_governor": True,
            "feeds_system_needs_intelligence": True,
            "protects_foreground_apps": True,
            "never_touch_protected_volumes": ["/Volumes/VIDEO"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish memory-pressure intelligence for P-core, training, and backlog headroom.")
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
        classification = _as_dict(payload.get("classification"))
        print(
            "memory_pressure_intelligence "
            f"status={payload['overall_status']} "
            f"headroom={classification.get('status')} "
            f"pcore_cap={classification.get('recommended_p_core_worker_cap')} "
            f"applied={result['applied']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
