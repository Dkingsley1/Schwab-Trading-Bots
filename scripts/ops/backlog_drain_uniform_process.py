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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "backlog_drain_uniform_process_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.backlog_drain_uniform_override"
PROTECTED_VOLUMES = ["/Volumes/VIDEO"]


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _lock_open_enabled() -> bool:
    return any(
        _env_flag(name)
        for name in (
            "BACKLOG_PCORE_DRAIN_LOCK_OPEN",
            "BACKLOG_ACCELERATOR_LOCK_OPEN",
            "BACKLOG_DRAIN_LOCK_OPEN",
            "BACKLOG_FORCE_LOCK_OPEN",
            "OPERATOR_DRAIN_LOCK_OPEN",
        )
    )


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    for key in ("overall_status", "status"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    if payload.get("ok") is True:
        return "ready"
    if payload.get("ok") is False:
        return "blocked"
    return default


def _storage_pressure(storage: dict[str, Any]) -> dict[str, Any]:
    truth = _as_dict(storage.get("backlog_truth"))
    overlay = _as_dict(truth.get("sql_overlay"))
    raw_live = _as_dict(truth.get("raw_live"))
    backpressure = _as_dict(storage.get("backpressure"))
    effective = _as_dict(backpressure.get("effective_raw_live"))
    use_overlay = bool(overlay.get("used_for_pressure", False))
    source = "sql_overlay_attributed" if use_overlay else str(backpressure.get("effective_raw_live_source") or effective.get("source") or "effective_raw_live")
    row = overlay if use_overlay else effective or backpressure
    return {
        "source": source,
        "overall_status": _status(storage),
        "severity": str(storage.get("severity") or ""),
        "pressure_index": round(_safe_float(storage.get("pressure_index"), _safe_float(overlay.get("pressure_ratio"), 0.0)), 3),
        "core_pending_lines": _safe_int(row.get("core_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), 0)),
        "total_pending_lines": _safe_int(row.get("total_pending_lines"), _safe_int(backpressure.get("total_pending_lines"), 0)),
        "oldest_pending_age_seconds": round(_safe_float(row.get("oldest_pending_age_seconds"), _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0)), 3),
        "raw_live": {
            "grade": str(raw_live.get("grade") or ""),
            "core_pending_lines": _safe_int(raw_live.get("core_pending_lines"), 0),
            "total_pending_lines": _safe_int(raw_live.get("total_pending_lines"), 0),
            "oldest_pending_age_seconds": round(_safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0), 3),
        },
        "truth_gap": _as_dict(truth.get("truth_gap")),
    }


def _oldest_sources(storage: dict[str, Any], pump: dict[str, Any], accelerator: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    locator = _as_dict(storage.get("stale_pending_locator"))
    candidates.extend(_as_list(locator.get("oldest_sources")))
    router = _as_dict(_as_dict(pump.get("bots")).get("shard_hotness_router_bot"))
    candidates.extend(_as_list(router.get("focused_sources")))
    storage_contract = _as_dict(accelerator.get("storage_contract"))
    candidates.extend(_as_list(storage_contract.get("oldest_sources")))

    rows: dict[str, dict[str, Any]] = {}
    for raw in candidates:
        row = _as_dict(raw)
        source_rel = str(row.get("source_rel") or "").strip()
        if not source_rel:
            continue
        existing = rows.setdefault(
            source_rel,
            {
                "source_rel": source_rel,
                "shard": str(row.get("shard") or ""),
                "pressure_lane": str(row.get("pressure_lane") or ""),
                "pending_lines": 0,
                "oldest_pending_age_seconds": 0.0,
            },
        )
        existing["pending_lines"] = max(_safe_int(existing.get("pending_lines"), 0), _safe_int(row.get("pending_lines"), 0))
        existing["oldest_pending_age_seconds"] = max(
            _safe_float(existing.get("oldest_pending_age_seconds"), 0.0),
            _safe_float(row.get("oldest_pending_age_seconds"), 0.0),
        )
        for key in ("shard", "pressure_lane"):
            if row.get(key) and not existing.get(key):
                existing[key] = str(row.get(key))
    return sorted(
        rows.values(),
        key=lambda item: (_safe_float(item.get("oldest_pending_age_seconds"), 0.0), _safe_int(item.get("pending_lines"), 0)),
        reverse=True,
    )


def _target_shards(rows: list[dict[str, Any]], pump: dict[str, Any]) -> list[str]:
    router = _as_dict(_as_dict(pump.get("bots")).get("shard_hotness_router_bot"))
    raw_priority = str(_as_dict(router.get("control_env")).get("SQL_LINK_SERVICE_HOT_SHARD_PRIORITY") or "")
    shards = [part.strip() for part in raw_priority.split(",") if part.strip()]
    shards.extend(str(row.get("shard") or "").strip() for row in rows if str(row.get("shard") or "").strip())
    return ordered_unique(shards)[:8]


def _memory_allows_heavier_batch(memory: dict[str, Any], accelerator: dict[str, Any]) -> bool:
    classification = _as_dict(memory.get("classification"))
    snapshot = _as_dict(memory.get("snapshot"))
    host = _as_dict(accelerator.get("host_lane_contract"))
    status_values = [
        str(classification.get("status") or "").lower(),
        str(classification.get("decision") or "").lower(),
        str(snapshot.get("pressure_level") or "").lower(),
        str(snapshot.get("pressure_kind") or "").lower(),
        str(host.get("memory_status") or "").lower(),
    ]
    hard_values = {"hard_relief", "swap_relief", "critical_relief", "memory_critical", "critical", "hard"}
    if any(value in hard_values for value in status_values):
        return False
    if _safe_float(snapshot.get("swap_used_gb"), 0.0) >= 6.0:
        return False
    if _safe_float(snapshot.get("pages_throttled"), 0.0) > 0.0:
        return False
    return True


def _memory_allows_turbo_batch(memory: dict[str, Any], accelerator: dict[str, Any]) -> bool:
    if not _memory_allows_heavier_batch(memory, accelerator):
        return False
    classification = _as_dict(memory.get("classification"))
    snapshot = _as_dict(memory.get("snapshot"))
    host = _as_dict(accelerator.get("host_lane_contract"))
    if str(classification.get("status") or "").lower() not in {"ready", "soft_guard", "cooling", ""}:
        return False
    if str(host.get("memory_status") or "").lower() in {"hard_relief", "swap_relief", "critical_relief"}:
        return False
    if _safe_float(snapshot.get("compressed_pressure_gb"), _safe_float(snapshot.get("compressor_gb"), 0.0)) >= 3.0:
        return False
    if _safe_float(snapshot.get("swap_used_gb"), 0.0) >= 3.5:
        return False
    return True


def _memory_allows_turbo_plus(memory: dict[str, Any], accelerator: dict[str, Any]) -> bool:
    if not _memory_allows_turbo_batch(memory, accelerator):
        return False
    snapshot = _as_dict(memory.get("snapshot"))
    guidance = _as_dict(memory.get("workload_guidance"))
    if _safe_float(snapshot.get("compressed_pressure_gb"), _safe_float(snapshot.get("compressor_gb"), 0.0)) >= 1.5:
        return False
    if _safe_float(snapshot.get("swap_used_gb"), 0.0) >= 3.0:
        return False
    if _safe_int(guidance.get("p_core_preprocess_worker_cap"), 0) and _safe_int(guidance.get("p_core_preprocess_worker_cap"), 0) < 3:
        return False
    return True


def _raw_live_green(pressure: dict[str, Any]) -> bool:
    raw = _as_dict(pressure.get("raw_live"))
    grade = str(raw.get("grade") or "").upper()
    return bool(
        grade in {"A+", "A++", "READY", "GREEN"}
        or (
            _safe_int(raw.get("core_pending_lines"), 0) <= 5000
            and _safe_int(raw.get("total_pending_lines"), 0) <= 5500
            and _safe_float(raw.get("oldest_pending_age_seconds"), 0.0) <= 180.0
        )
    )


def _turbo_decision(
    *,
    pressure: dict[str, Any],
    rows: list[dict[str, Any]],
    shards: list[str],
    writer_state: dict[str, Any],
    memory: dict[str, Any],
    accelerator: dict[str, Any],
) -> dict[str, Any]:
    blockers = ordered_unique(
        [
            "no_hot_target_shards" if not shards else "",
            "raw_live_not_green" if not _raw_live_green(pressure) else "",
            "memory_not_clear_for_turbo_batch" if not _memory_allows_turbo_batch(memory, accelerator) else "",
            "storage_still_critical" if str(pressure.get("overall_status") or "") == "blocked" or str(pressure.get("severity") or "") == "critical" else "",
            "oldest_age_still_above_turbo_threshold" if _safe_float(pressure.get("oldest_pending_age_seconds"), 0.0) > 240.0 else "",
            "writer_not_single_primary" if not bool(writer_state.get("single_primary_merge_writer", True)) else "",
        ]
    )
    truth_gap = _as_dict(pressure.get("truth_gap"))
    overlay_tail_present = bool(
        _safe_int(pressure.get("total_pending_lines"), 0) >= 5000
        or _safe_int(truth_gap.get("pending_line_delta"), 0) >= 5000
        or any(_safe_int(row.get("pending_lines"), 0) > 0 for row in rows)
    )
    if not overlay_tail_present:
        blockers.append("no_material_overlay_tail")
    enabled = not blockers
    turbo_plus_blockers = ordered_unique(
        [
            "turbo_not_enabled" if not enabled else "",
            "memory_not_clear_for_turbo_plus" if enabled and not _memory_allows_turbo_plus(memory, accelerator) else "",
            "pressure_index_above_turbo_plus_ceiling" if enabled and _safe_float(pressure.get("pressure_index"), 0.0) > 0.8 else "",
            "oldest_age_above_turbo_plus_ceiling" if enabled and _safe_float(pressure.get("oldest_pending_age_seconds"), 0.0) > 90.0 else "",
            "status_not_stable_for_turbo_plus"
            if enabled and (str(pressure.get("overall_status") or "") != "ready" or str(pressure.get("severity") or "") not in {"stable", ""})
            else "",
        ]
    )
    turbo_plus_enabled = bool(enabled and not turbo_plus_blockers)
    sentinel_scope = ["health_fast", "writer_progress"]
    shard_scope = ordered_unique([*sentinel_scope, *shards])[:10] if enabled else []
    tier = "turbo_plus_single_writer_catchup" if turbo_plus_enabled else "turbo_single_writer_catchup" if enabled else "guarded_standard"
    return {
        "enabled": enabled,
        "turbo_plus_enabled": turbo_plus_enabled,
        "tier": tier,
        "blockers": blockers,
        "turbo_plus_blockers": turbo_plus_blockers,
        "shard_scope": shard_scope,
        "batch_size": 420000 if turbo_plus_enabled else 360000 if enabled else 0,
        "queue_batch_size": 340000 if turbo_plus_enabled else 300000 if enabled else 0,
        "wave_limit": 6 if enabled else 0,
        "max_seconds_per_cycle": 210 if turbo_plus_enabled else 180 if enabled else 0,
        "poll_seconds": 6 if turbo_plus_enabled else 8 if enabled else 0,
        "wait_timeout_seconds": 240 if turbo_plus_enabled else 300 if enabled else 0,
        "policy": "turbo narrows to sentinel plus hot shards, increases single-writer batch budget, and never adds sqlite writers",
    }


def _storage_cycle_progress(storage_auto: dict[str, Any]) -> dict[str, Any]:
    cycles = [row for row in _as_list(storage_auto.get("cycle_records")) if isinstance(row, dict)]
    latest = cycles[-1] if cycles else {}
    progress = _as_dict(latest.get("progress"))
    before = _as_dict(progress.get("before")) or _as_dict(latest.get("clearance_before"))
    after = _as_dict(progress.get("after")) or _as_dict(latest.get("clearance_after"))
    before_pending = _safe_int(before.get("total_pending_lines"), 0)
    after_pending = _safe_int(after.get("total_pending_lines"), 0)
    reduced = max(_safe_int(progress.get("pending_lines_reduced"), before_pending - after_pending), 0)
    reduction_ratio = round(reduced / max(before_pending, 1), 6) if before_pending > 0 else 0.0
    attempts = [row for row in _as_list(storage_auto.get("attempts")) if isinstance(row, dict)]
    bad_attempts = [
        str(row.get("name") or row.get("status") or "unknown")
        for row in attempts
        if bool(row.get("timed_out", False)) or str(row.get("status") or "").strip().lower() in {"error", "timed_out"}
    ]
    return {
        "observed": bool(progress.get("progress_observed", False) or reduced > 0),
        "before_pending_lines": before_pending,
        "after_pending_lines": after_pending,
        "pending_lines_reduced": reduced,
        "reduction_ratio": reduction_ratio,
        "bad_attempts": bad_attempts,
        "cycle_index": _safe_int(latest.get("cycle_index"), 0),
    }


def _adaptive_convergence_decision(
    *,
    pressure: dict[str, Any],
    shards: list[str],
    writer_state: dict[str, Any],
    memory: dict[str, Any],
    accelerator: dict[str, Any],
    storage_auto: dict[str, Any],
    turbo: dict[str, Any],
) -> dict[str, Any]:
    progress = _storage_cycle_progress(storage_auto)
    status = _status(storage_auto)
    metrics = _as_dict(storage_auto.get("metrics"))
    clearance = _as_dict(storage_auto.get("clearance_state"))
    storage_plane = _as_dict(_as_dict(storage_auto.get("previews")).get("storage_plane"))
    phase = str(storage_plane.get("phase") or metrics.get("storage_plane_phase") or "")
    steady_state_ready = bool(clearance.get("steady_state_ready", False))
    blocker_values = [
        "disabled_by_env" if not _env_flag("BACKLOG_DRAIN_ADAPTIVE_CONVERGENCE_AUTO_ENABLED", True) else "",
        "turbo_plus_not_enabled" if not bool(turbo.get("turbo_plus_enabled", False)) else "",
        "storage_autopilot_history_missing" if not storage_auto else "",
        "storage_autopilot_not_successful" if storage_auto and status not in {"applied", "applied_with_followups", "ready"} else "",
        "last_cycle_had_timeout_or_error" if progress["bad_attempts"] else "",
        "last_cycle_progress_not_observed" if not progress["observed"] else "",
        "last_cycle_reduction_below_convergence_floor"
        if progress["pending_lines_reduced"] < 1000 or progress["reduction_ratio"] < 0.08
        else "",
        "storage_not_stable_for_convergence"
        if str(pressure.get("overall_status") or "") != "ready" or str(pressure.get("severity") or "") not in {"stable", ""}
        else "",
        "pressure_index_above_convergence_ceiling" if _safe_float(pressure.get("pressure_index"), 0.0) > 0.75 else "",
        "raw_live_not_green" if not _raw_live_green(pressure) else "",
        "memory_not_clear_for_convergence" if not _memory_allows_turbo_plus(memory, accelerator) else "",
        "writer_progress_stale" if bool(writer_state.get("active", False)) and _safe_float(writer_state.get("progress_age_minutes"), 0.0) > 2.5 else "",
        "storage_plane_in_recovery_guard" if phase in {"emergency_disk_guard", "storage_reserve_rebuild", "manifest_only_recovery"} else "",
        "steady_state_already_ready" if steady_state_ready else "",
    ]
    blockers = ordered_unique(blocker_values)
    enabled = bool(not blockers)
    shard_scope = ordered_unique(["health_fast", "writer_progress", "hot_path_storage", *shards])[:10] if enabled else []
    return {
        "enabled": enabled,
        "tier": "adaptive_convergence_single_writer_catchup" if enabled else "guarded_turbo_plus",
        "blockers": blockers,
        "progress": progress,
        "storage_autopilot_status": status,
        "storage_plane_phase": phase,
        "steady_state_ready": steady_state_ready,
        "shard_scope": shard_scope,
        "batch_size": 480000 if enabled else 0,
        "queue_batch_size": 380000 if enabled else 0,
        "wave_limit": 6 if enabled else 0,
        "max_seconds_per_cycle": 240 if enabled else 0,
        "poll_seconds": 5 if enabled else 0,
        "wait_timeout_seconds": 210 if enabled else 0,
        "storage_autopilot_cycles": 3 if enabled else 0,
        "policy": "advance only after the last bounded storage cycle proved progress, with one sqlite writer and fresh host headroom",
    }


def _sparse_pressure_context(storage: dict[str, Any]) -> dict[str, Any]:
    backpressure = _as_dict(storage.get("backpressure"))
    raw_live = _as_dict(backpressure.get("raw_live"))
    effective = _as_dict(backpressure.get("effective_raw_live"))
    raw_line = _as_dict(raw_live.get("line_estimation"))
    effective_line = _as_dict(effective.get("line_estimation"))
    line = raw_line or effective_line
    relief = _as_dict(storage.get("backlog_relief_contract"))
    active_issue_ids = [str(item) for item in _as_list(relief.get("active_issue_ids")) if str(item).strip()]
    sparse_issue = {}
    for raw in _as_list(relief.get("issues")):
        row = _as_dict(raw)
        if str(row.get("id") or "") == "sparse_huge_jsonl_files":
            sparse_issue = row
            break
    evidence = _as_dict(sparse_issue.get("evidence"))
    relief_env = _as_dict(relief.get("control_env_recommendations"))
    issue_env = _as_dict(sparse_issue.get("control_env"))
    pcore = _as_dict(relief.get("p_core_backlog_allocation_contract"))
    pcore_env = _as_dict(pcore.get("control_env"))
    pending_bytes = max(
        _safe_int(line.get("sparse_large_line_pending_bytes"), 0),
        _safe_int(evidence.get("sparse_large_line_pending_bytes"), 0),
    )
    pending_lines = max(
        _safe_int(line.get("sparse_large_line_pending_lines"), 0),
        _safe_int(evidence.get("sparse_large_line_pending_lines"), 0),
    )
    file_count = max(
        _safe_int(line.get("sparse_large_line_files"), 0),
        _safe_int(evidence.get("sparse_large_line_files"), 0),
    )
    control_env: dict[str, str] = {}
    for key in ("INGEST_MAX_BYTES_PER_FILE", "SQLITE_BATCH_MAX_BYTES", "INGEST_TOP_PENDING_FILES"):
        value = issue_env.get(key) if key in issue_env else relief_env.get(key)
        if value not in {None, ""}:
            control_env[key] = str(value)
    if pending_bytes > 0:
        control_env.setdefault("SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_DRAIN", "1")
        control_env.setdefault("SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_FILE_COUNT", str(max(file_count, 1)))
    for key in (
        "BACKLOG_PCORE_ALLOCATION_ACTIVE",
        "BACKLOG_PCORE_PREPROCESS_WORKERS",
        "BACKLOG_PCORE_USER_APP_RESERVE_TARGET",
        "BACKLOG_PCORE_BURST_MODE",
        "BACKLOG_PCORE_BURST_REASON",
        "SQL_LINK_SERVICE_PREPROCESS_WORKERS",
        "SQL_LINK_SERVICE_SHARD_WRITER_LANES",
        "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES",
        "SQL_LINK_CHILD_WRITER_CPU_POLICY",
        "SQL_LINK_WRITER_BACKGROUND_POLICY",
        "SQL_LINK_WRITER_NICE",
        "BOT_CPU_ALLOCATION_POLICY",
        "BOT_CPU_QOS_POLICY",
    ):
        value = pcore_env.get(key) if key in pcore_env else relief_env.get(key)
        if value not in {None, ""}:
            control_env[key] = str(value)
    return {
        "active": bool("sparse_huge_jsonl_files" in active_issue_ids or bool(sparse_issue.get("active", False))),
        "active_issue_ids": active_issue_ids,
        "pending_bytes": pending_bytes,
        "pending_lines": pending_lines,
        "file_count": file_count,
        "policy": str(line.get("sparse_large_line_policy") or ""),
        "issue_grade": str(sparse_issue.get("grade") or ""),
        "pressure_ratio": round(_safe_float(sparse_issue.get("pressure_ratio"), 0.0), 3),
        "next_action": str(sparse_issue.get("next_action") or ""),
        "control_env": control_env,
        "pcore_policy": str(pcore.get("policy") or ""),
        "pcore_workers": _safe_int(pcore_env.get("SQL_LINK_SERVICE_PREPROCESS_WORKERS"), _safe_int(relief_env.get("SQL_LINK_SERVICE_PREPROCESS_WORKERS"), 0)),
    }


def _storage_reserve_context(storage: dict[str, Any]) -> dict[str, Any]:
    plane = _as_dict(storage.get("storage_plane_contract"))
    disk = _as_dict(plane.get("disk_contract"))
    efficiency = _as_dict(storage.get("storage_efficiency_contract"))
    metrics = _as_dict(efficiency.get("metrics"))
    free_gb = _safe_float(disk.get("external_available_gb"), 0.0)
    target_gb = _safe_float(metrics.get("safe_space_recovery_target_free_gb"), 64.0 if free_gb > 0.0 else 0.0)
    min_margin_gb = max(_safe_float(os.getenv("BACKLOG_DRAIN_SPARSE_FINALIZER_MIN_FREE_MARGIN_GB"), 1.0), 0.0)
    known = bool(free_gb > 0.0 and target_gb > 0.0)
    margin_gb = round(float(free_gb - target_gb), 3) if known else 0.0
    return {
        "known": known,
        "external_available_gb": round(float(free_gb), 3),
        "target_free_gb": round(float(target_gb), 3),
        "margin_gb": margin_gb,
        "min_margin_gb": round(float(min_margin_gb), 3),
        "enough_margin": bool(not known or margin_gb >= min_margin_gb),
    }


def _sparse_finalizer_progress_context(storage_auto: dict[str, Any]) -> dict[str, Any]:
    progress = _storage_cycle_progress(storage_auto)
    history_known = bool(_safe_int(progress.get("before_pending_lines"), 0) > 0)
    min_reduced = max(_safe_int(os.getenv("BACKLOG_DRAIN_SPARSE_FINALIZER_MIN_REDUCED_LINES"), 100), 0)
    min_ratio = max(_safe_float(os.getenv("BACKLOG_DRAIN_SPARSE_FINALIZER_MIN_REDUCTION_RATIO"), 0.03), 0.0)
    enough = bool(
        not history_known
        or (
            not progress.get("bad_attempts")
            and bool(progress.get("observed", False))
            and (
                _safe_int(progress.get("pending_lines_reduced"), 0) >= min_reduced
                or _safe_float(progress.get("reduction_ratio"), 0.0) >= min_ratio
            )
        )
    )
    return {
        **progress,
        "history_known": history_known,
        "min_reduced_lines": min_reduced,
        "min_reduction_ratio": round(float(min_ratio), 6),
        "enough_progress": enough,
    }


def _sparse_pressure_finalizer_decision(
    *,
    storage: dict[str, Any],
    pressure: dict[str, Any],
    shards: list[str],
    memory: dict[str, Any],
    accelerator: dict[str, Any],
    convergence: dict[str, Any],
    storage_auto: dict[str, Any],
) -> dict[str, Any]:
    context = _sparse_pressure_context(storage)
    reserve = _storage_reserve_context(storage)
    progress = _sparse_finalizer_progress_context(storage_auto)
    target_status = _as_dict(_as_dict(storage.get("steady_state")).get("target_status"))
    breaches = [str(item) for item in _as_list(target_status.get("target_breaches")) if str(item).strip()]
    backpressure = _as_dict(storage.get("backpressure"))
    overlay_clear = bool(backpressure.get("overlay_pressure_clear", False))
    blockers = ordered_unique(
        [
            "disabled_by_env" if not _env_flag("BACKLOG_DRAIN_SPARSE_PRESSURE_FINALIZER_AUTO_ENABLED", True) else "",
            "adaptive_convergence_still_active" if bool(convergence.get("enabled", False)) else "",
            "sparse_pressure_not_active" if not bool(context.get("active", False)) else "",
            "sparse_pending_bytes_missing" if _safe_int(context.get("pending_bytes"), 0) <= 0 else "",
            "pressure_index_not_only_remaining_breach" if breaches and breaches != ["pressure_index"] else "",
            "core_pending_not_under_target" if target_status and not bool(target_status.get("core_pending_lines_ok", False)) else "",
            "drain_minutes_not_under_target" if target_status and not bool(target_status.get("estimated_total_drain_minutes_ok", False)) else "",
            "storage_not_stable_for_sparse_finalizer"
            if str(pressure.get("overall_status") or "") != "ready" or str(pressure.get("severity") or "") not in {"stable", ""}
            else "",
            "raw_live_not_green" if not _raw_live_green(pressure) else "",
            "overlay_not_clear" if not overlay_clear else "",
            "storage_reserve_margin_too_thin" if not bool(reserve.get("enough_margin", True)) else "",
            "last_sparse_finalizer_cycle_had_timeout_or_error" if progress.get("bad_attempts") else "",
            "last_sparse_finalizer_progress_too_low"
            if bool(progress.get("history_known", False)) and not bool(progress.get("enough_progress", True))
            else "",
            "memory_not_clear_for_sparse_finalizer" if not _memory_allows_turbo_batch(memory, accelerator) else "",
        ]
    )
    enabled = bool(not blockers)
    pcore_workers = _safe_int(context.get("pcore_workers"), 0)
    shard_scope = ordered_unique(["health_fast", "writer_progress", "hot_path_storage", *shards])[:10] if enabled else []
    return {
        "enabled": enabled,
        "tier": "sparse_pressure_finalizer_single_writer" if enabled else "guarded_sparse_watch",
        "blockers": blockers,
        "context": context,
        "storage_reserve": reserve,
        "progress": progress,
        "target_breaches": breaches,
        "shard_scope": shard_scope,
        "batch_size": 360000 if enabled else 0,
        "queue_batch_size": 260000 if enabled else 0,
        "wave_limit": 6 if enabled else 0,
        "max_seconds_per_cycle": 180 if enabled else 0,
        "poll_seconds": 7 if enabled else 0,
        "wait_timeout_seconds": 240 if enabled else 0,
        "storage_autopilot_cycles": 2 if enabled else 0,
        "preprocess_workers": pcore_workers if enabled and pcore_workers > 0 else 0,
        "policy": "finish sparse byte pressure with byte-window ingestion caps and p-core preprocessing while preserving one sqlite writer",
    }


def _writer_state(writer: dict[str, Any], writer_intel: dict[str, Any]) -> dict[str, Any]:
    health = _as_dict(writer_intel.get("writer_health"))
    state = _as_dict(writer.get("writer_state_after_wait")) or _as_dict(writer.get("writer_state_before"))
    summary = _as_dict(writer.get("summary"))
    current = health or state
    lane = _as_dict(current.get("shard_writer_lane_contract"))
    completed_handoff_needed = bool(
        current.get("completed_lock_handoff_needed")
        or state.get("complete_lock_handoff_needed")
        or state.get("active_source") == "completed_lock_handoff_needed"
        or (
            not state
            and summary.get("completed_writer_lock_handoff_needed")
            and not summary.get("completed_writer_lock_handoff_released")
        )
    )
    return {
        "active": bool(current.get("active", False) or state.get("active", False)),
        "completed_lock_handoff_needed": completed_handoff_needed,
        "current_step": str(current.get("current_step") or state.get("current_step") or ""),
        "progress_age_minutes": round(_safe_float(current.get("progress_age_minutes"), _safe_float(state.get("progress_age_minutes"), 0.0)), 3),
        "cycle_age_minutes": round(_safe_float(current.get("cycle_age_minutes"), _safe_float(state.get("cycle_age_minutes"), 0.0)), 3),
        "selected_shard_writer_lanes": _safe_int(lane.get("selected_shard_writer_lanes"), _safe_int(current.get("shard_link_writer_lanes"), 0)),
        "max_shard_writer_lanes": _safe_int(lane.get("max_shard_writer_lanes"), 0),
        "primary_merge_writer_count": _safe_int(lane.get("primary_merge_writer_count"), 1),
        "single_primary_merge_writer": bool(lane.get("single_primary_merge_writer", True)),
    }


def _lane_contract(accelerator: dict[str, Any], writer_state: dict[str, Any]) -> dict[str, Any]:
    host = _as_dict(accelerator.get("host_lane_contract"))
    storage_accel = _as_dict(accelerator.get("storage_accelerator_contract"))
    host_workers = max(_safe_int(host.get("selected_p_core_preprocess_workers"), 0), 1)
    desired_workers = max(_safe_int(storage_accel.get("p_core_preprocess_workers"), host_workers), host_workers)
    active_selected = _safe_int(writer_state.get("selected_shard_writer_lanes"), 0)
    live_workers = host_workers
    if bool(writer_state.get("active")) and active_selected > 0:
        live_workers = min(active_selected, host_workers)
    max_lanes = max(_safe_int(storage_accel.get("max_shard_writer_lanes"), 0), live_workers)
    return {
        "live_preprocess_workers": live_workers,
        "desired_preprocess_workers_when_host_clear": desired_workers,
        "max_shard_writer_lanes": max_lanes,
        "source": "host_lane_contract_caps_live_width_while_accelerator_tracks_desired_width",
    }


def _speed_contract(
    storage: dict[str, Any],
    accelerator: dict[str, Any],
    pump: dict[str, Any],
    writer: dict[str, Any],
    writer_intel: dict[str, Any],
    memory: dict[str, Any],
    storage_auto: dict[str, Any],
) -> dict[str, Any]:
    pressure = _storage_pressure(storage)
    rows = _oldest_sources(storage, pump, accelerator)
    shards = _target_shards(rows, pump)
    writer_view = _writer_state(writer, writer_intel)
    lanes = _lane_contract(accelerator, writer_view)
    storage_accel = _as_dict(accelerator.get("storage_accelerator_contract"))
    wave = _as_dict(storage_accel.get("catch_up_wave_controller"))
    pump_wave = _as_dict(_as_dict(_as_dict(pump.get("bots")).get("catch_up_wave_budget_bot")).get("control_env"))
    active_issues = _as_list(_as_dict(storage.get("backlog_relief_contract")).get("active_issue_ids"))
    severe = bool(
        pressure["overall_status"] == "blocked"
        or pressure["severity"] == "critical"
        or pressure["pressure_index"] >= 1.0
        or pressure["oldest_pending_age_seconds"] > 240.0
    )
    turbo = _turbo_decision(
        pressure=pressure,
        rows=rows,
        shards=shards,
        writer_state=writer_view,
        memory=memory,
        accelerator=accelerator,
    )
    convergence = _adaptive_convergence_decision(
        pressure=pressure,
        shards=shards,
        writer_state=writer_view,
        memory=memory,
        accelerator=accelerator,
        storage_auto=storage_auto,
        turbo=turbo,
    )
    finalizer = _sparse_pressure_finalizer_decision(
        storage=storage,
        pressure=pressure,
        shards=shards,
        memory=memory,
        accelerator=accelerator,
        convergence=convergence,
        storage_auto=storage_auto,
    )
    finalizer_blockers = [str(item) for item in _as_list(finalizer.get("blockers"))]
    finalizer_context = _as_dict(finalizer.get("context"))
    finalizer_safety_hold = bool(
        not finalizer.get("enabled", False)
        and finalizer_context.get("active", False)
        and set(finalizer_blockers)
        & {
            "storage_reserve_margin_too_thin",
            "last_sparse_finalizer_cycle_had_timeout_or_error",
            "last_sparse_finalizer_progress_too_low",
        }
    )
    if finalizer_safety_hold:
        turbo = {
            **turbo,
            "enabled": False,
            "turbo_plus_enabled": False,
            "tier": "sparse_finalizer_safety_hold",
            "blockers": ordered_unique([*_as_list(turbo.get("blockers")), "sparse_finalizer_safety_hold"]),
            "turbo_plus_blockers": ordered_unique([*_as_list(turbo.get("turbo_plus_blockers")), "sparse_finalizer_safety_hold"]),
            "shard_scope": [],
            "batch_size": 0,
            "queue_batch_size": 0,
            "wave_limit": 0,
            "max_seconds_per_cycle": 0,
            "poll_seconds": 0,
            "wait_timeout_seconds": 0,
            "policy": "hold broad catch-up when the sparse pressure finalizer is blocked by reserve, errors, or flat progress",
        }
    if bool(convergence.get("enabled", False)) and not finalizer_safety_hold:
        turbo = {
            **turbo,
            "convergence_enabled": True,
            "shard_scope": _as_list(convergence.get("shard_scope")),
        }
    elif bool(finalizer.get("enabled", False)):
        turbo = {
            **turbo,
            "sparse_finalizer_enabled": True,
            "shard_scope": _as_list(finalizer.get("shard_scope")),
        }
    base_wave_limit = max(
        _safe_int(wave.get("max_waves"), 1),
        _safe_int(pump_wave.get("WRITER_CYCLE_MAX_CATCH_UP_WAVES"), 1),
        5 if severe else 1,
    )
    if bool(convergence.get("enabled", False)):
        wave_limit = max(base_wave_limit, _safe_int(convergence.get("wave_limit"), 0))
    elif bool(finalizer.get("enabled", False)):
        wave_limit = max(base_wave_limit, _safe_int(finalizer.get("wave_limit"), 0))
    else:
        wave_limit = max(base_wave_limit, _safe_int(turbo.get("wave_limit"), 0))
    wave_limit = min(max(wave_limit, 1), 6)
    base_seconds_per_cycle = max(
        _safe_int(wave.get("max_seconds_per_writer_cycle"), 30),
        _safe_int(pump_wave.get("SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"), 30),
        120 if severe else 60,
    )
    if bool(convergence.get("enabled", False)):
        seconds_per_cycle = max(base_seconds_per_cycle, _safe_int(convergence.get("max_seconds_per_cycle"), 0))
    elif bool(finalizer.get("enabled", False)):
        seconds_per_cycle = max(base_seconds_per_cycle, _safe_int(finalizer.get("max_seconds_per_cycle"), 0))
    else:
        seconds_per_cycle = max(base_seconds_per_cycle, _safe_int(turbo.get("max_seconds_per_cycle"), 0))
    hot_batch_current = _safe_int(_as_dict(accelerator.get("single_writer_tuning_contract")).get("hot_batch_size"), 0)
    queue_batch_current = _safe_int(_as_dict(accelerator.get("single_writer_tuning_contract")).get("queue_batch_size"), 0)
    hot_batch_size = max(hot_batch_current, 240000) if _memory_allows_heavier_batch(memory, accelerator) else max(hot_batch_current, 120000)
    queue_batch_size = max(queue_batch_current, 240000) if _memory_allows_heavier_batch(memory, accelerator) else max(queue_batch_current, 120000)
    if bool(finalizer.get("enabled", False)):
        hot_batch_size = max(hot_batch_current, _safe_int(finalizer.get("batch_size"), 0), 240000)
        queue_batch_size = max(queue_batch_current, _safe_int(finalizer.get("queue_batch_size"), 0), 240000)
    elif bool(turbo.get("enabled", False)):
        hot_batch_size = max(hot_batch_size, _safe_int(turbo.get("batch_size"), 0))
        queue_batch_size = max(queue_batch_size, _safe_int(turbo.get("queue_batch_size"), 0))
    if bool(convergence.get("enabled", False)):
        hot_batch_size = max(hot_batch_size, _safe_int(convergence.get("batch_size"), 0))
        queue_batch_size = max(queue_batch_size, _safe_int(convergence.get("queue_batch_size"), 0))
    mode = (
        str(convergence.get("tier") or "")
        if bool(convergence.get("enabled", False))
        else str(finalizer.get("tier") or "")
        if bool(finalizer.get("enabled", False))
        else str(turbo.get("tier") or "")
        if bool(turbo.get("enabled", False))
        else "focused_hot_overlay_catchup"
        if severe and shards
        else "steady_uniform_drain"
    )
    poll_seconds = (
        _safe_int(convergence.get("poll_seconds"), 0)
        if bool(convergence.get("enabled", False))
        else _safe_int(finalizer.get("poll_seconds"), 0)
        if bool(finalizer.get("enabled", False))
        else _safe_int(turbo.get("poll_seconds"), 0)
        if bool(turbo.get("enabled", False))
        else 8
        if severe
        else 20
    )
    wait_timeout_seconds = (
        _safe_int(convergence.get("wait_timeout_seconds"), 0)
        if bool(convergence.get("enabled", False))
        else _safe_int(finalizer.get("wait_timeout_seconds"), 0)
        if bool(finalizer.get("enabled", False))
        else _safe_int(turbo.get("wait_timeout_seconds"), 0)
        if bool(turbo.get("enabled", False))
        else 240
        if severe
        else 900
    )
    storage_autopilot_cycles = (
        _safe_int(convergence.get("storage_autopilot_cycles"), 0)
        if bool(convergence.get("enabled", False))
        else _safe_int(finalizer.get("storage_autopilot_cycles"), 0)
        if bool(finalizer.get("enabled", False))
        else 2
        if bool(turbo.get("enabled", False)) or severe
        else 1
    )
    return {
        "mode": mode,
        "canonical_pressure": pressure,
        "target_shards": shards,
        "focused_sources": rows[:8],
        "active_issue_ids": [str(item) for item in active_issues if str(item).strip()],
        "turbo_contract": turbo,
        "adaptive_convergence_contract": convergence,
        "sparse_pressure_finalizer_contract": finalizer,
        "writer_state": writer_view,
        "lane_contract": lanes,
        "wave_limit": wave_limit,
        "max_seconds_per_cycle": seconds_per_cycle,
        "hot_batch_size": hot_batch_size,
        "queue_batch_size": queue_batch_size,
        "poll_seconds": poll_seconds,
        "wait_timeout_seconds": wait_timeout_seconds,
        "storage_autopilot_cycles": storage_autopilot_cycles,
        "single_writer_only": True,
        "adds_parallel_sqlite_writers": False,
    }


def _process_steps(contract: dict[str, Any]) -> list[dict[str, Any]]:
    writer_state = _as_dict(contract.get("writer_state"))
    handoff_needed = bool(writer_state.get("completed_lock_handoff_needed", False))
    writer_active = bool(writer_state.get("active", False)) and not handoff_needed
    poll = str(_safe_int(contract.get("poll_seconds"), 8))
    wait = str(_safe_int(contract.get("wait_timeout_seconds"), 240))
    seconds = str(max(_safe_int(contract.get("max_seconds_per_cycle"), 120), 120))
    writer_command_mode = (
        ["--apply", "--handoff-only"]
        if handoff_needed
        else ([] if writer_active else ["--apply"])
    )
    writer_reason = (
        "clear_completed_writer_lock_handoff"
        if handoff_needed
        else "wait_for_active_writer_progress"
        if writer_active
        else "launch_one_focused_writer_cycle"
    )
    return [
        {
            "step": "refresh_storage_truth",
            "command": ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
            "writes_sqlite": False,
            "parallel_safe": True,
        },
        {
            "step": "sync_pcore_accelerator",
            "command": ["./scripts/ops/opsctl.sh", "backlog-pcore-accelerator", "--apply", "--json"],
            "writes_sqlite": False,
            "parallel_safe": False,
        },
        {
            "step": "sync_pump_infrabots",
            "command": ["./scripts/ops/opsctl.sh", "backlog-pump-infrabots", "--apply", "--json"],
            "writes_sqlite": False,
            "parallel_safe": False,
        },
        {
            "step": "observe_writer_process",
            "command": ["./scripts/ops/opsctl.sh", "writer-process-intelligence", "--json"],
            "writes_sqlite": False,
            "parallel_safe": True,
        },
        {
            "step": "single_writer_handoff_or_wait",
            "command": [
                "./scripts/ops/opsctl.sh",
                "writer-cycle-coordinator",
                *writer_command_mode,
                "--poll-seconds",
                poll,
                "--wait-timeout-seconds",
                wait,
                "--command-timeout-seconds",
                seconds,
                "--json",
            ],
            "writes_sqlite": True,
            "parallel_safe": False,
            "reason": writer_reason,
        },
        {
            "step": "storage_autopilot_followthrough",
            "command": [
                "./scripts/ops/opsctl.sh",
                "storage-backpressure-autopilot",
                "--apply",
                "--poll-seconds",
                poll,
                "--wait-timeout-seconds",
                wait,
                "--max-cycles",
                str(_safe_int(contract.get("storage_autopilot_cycles"), 2)),
                "--json",
            ],
            "writes_sqlite": True,
            "parallel_safe": False,
            "reason": "run_after_writer_idle_or_when_parent_autopilot_owns_lock",
        },
    ]


def env_dict(payload: dict[str, Any]) -> dict[str, str]:
    contract = _as_dict(payload.get("speed_contract"))
    lanes = _as_dict(contract.get("lane_contract"))
    turbo = _as_dict(contract.get("turbo_contract"))
    convergence = _as_dict(contract.get("adaptive_convergence_contract"))
    finalizer = _as_dict(contract.get("sparse_pressure_finalizer_contract"))
    shards = [str(item) for item in _as_list(contract.get("target_shards")) if str(item).strip()]
    shard_scope = [str(item) for item in _as_list(turbo.get("shard_scope")) if str(item).strip()]
    turbo_enabled = bool(turbo.get("enabled", False))
    convergence_enabled = bool(convergence.get("enabled", False))
    finalizer_enabled = bool(finalizer.get("enabled", False))
    lock_open = _lock_open_enabled()
    env = {
        "BACKLOG_PCORE_DRAIN_LOCK_OPEN": "1" if lock_open else "0",
        "BACKLOG_ACCELERATOR_LOCK_OPEN": "1" if lock_open else "0",
        "BACKLOG_DRAIN_LOCK_OPEN": "1" if lock_open else "0",
        "BACKLOG_DRAIN_UNIFORM_PROCESS_ENABLED": "1",
        "BACKLOG_DRAIN_UNIFORM_MODE": "locked_open_uniform_pcore_drain" if lock_open else str(contract.get("mode") or "steady_uniform_drain"),
        "BACKLOG_DRAIN_UNIFORM_PRESSURE_SOURCE": str(_as_dict(contract.get("canonical_pressure")).get("source") or ""),
        "BACKLOG_DRAIN_TURBO_ENABLED": "1" if (turbo_enabled or lock_open) else "0",
        "BACKLOG_DRAIN_TURBO_TIER": str(turbo.get("tier") or "guarded_standard"),
        "BACKLOG_DRAIN_TURBO_BLOCKERS": ",".join(str(item) for item in _as_list(turbo.get("blockers"))),
        "BACKLOG_DRAIN_TURBO_PLUS_ENABLED": "1" if (bool(turbo.get("turbo_plus_enabled", False)) or lock_open) else "0",
        "BACKLOG_DRAIN_TURBO_PLUS_BLOCKERS": ",".join(str(item) for item in _as_list(turbo.get("turbo_plus_blockers"))),
        "BACKLOG_DRAIN_ADAPTIVE_CONVERGENCE_ENABLED": "1" if (convergence_enabled or lock_open) else "0",
        "BACKLOG_DRAIN_ADAPTIVE_CONVERGENCE_TIER": str(convergence.get("tier") or "guarded_turbo_plus"),
        "BACKLOG_DRAIN_ADAPTIVE_CONVERGENCE_BLOCKERS": ",".join(str(item) for item in _as_list(convergence.get("blockers"))),
        "BACKLOG_DRAIN_ADAPTIVE_CONVERGENCE_REDUCTION_RATIO": str(_safe_float(_as_dict(convergence.get("progress")).get("reduction_ratio"), 0.0)),
        "BACKLOG_DRAIN_SPARSE_PRESSURE_FINALIZER_ENABLED": "1" if (finalizer_enabled or lock_open) else "0",
        "BACKLOG_DRAIN_SPARSE_PRESSURE_FINALIZER_TIER": str(finalizer.get("tier") or "guarded_sparse_watch"),
        "BACKLOG_DRAIN_SPARSE_PRESSURE_FINALIZER_BLOCKERS": ",".join(str(item) for item in _as_list(finalizer.get("blockers"))),
        "BACKLOG_DRAIN_SPARSE_PENDING_BYTES": str(_safe_int(_as_dict(finalizer.get("context")).get("pending_bytes"), 0)),
        "BACKLOG_DRAIN_TARGET_SHARDS": ",".join(shards),
        "SQL_LINK_SERVICE_SHARDS": ",".join(shard_scope) if shard_scope else "",
        "SQL_LINK_SERVICE_HOT_SHARD_PRIORITY": ",".join(shards),
        "SQL_LINK_SERVICE_PIN_HOT_SOURCES": "1" if (_as_list(contract.get("focused_sources")) or lock_open) else "0",
        "SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_BOOST": "1",
        "SQL_LINK_SERVICE_COLD_STAGE_YIELDS_TO_RAW_LIVE": "1",
        "SQL_LINK_SERVICE_CATCH_UP_WAVE": "1",
        "WRITER_CYCLE_MAX_CATCH_UP_WAVES": str(max(_safe_int(contract.get("wave_limit"), 1), 9 if lock_open else 1)),
        "BACKLOG_CATCH_UP_WAVE_LIMIT": str(max(_safe_int(contract.get("wave_limit"), 1), 9 if lock_open else 1)),
        "BACKLOG_ACCELERATOR_MAX_SECONDS_PER_CYCLE": str(max(_safe_int(contract.get("max_seconds_per_cycle"), 120), 240 if lock_open else 120)),
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": str(max(_safe_int(contract.get("max_seconds_per_cycle"), 120), 240 if lock_open else 120)),
        "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": str(max(_safe_int(contract.get("max_seconds_per_cycle"), 120) * 3, 900 if lock_open else 420)),
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": str(_safe_int(contract.get("hot_batch_size"), 240000)),
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": str(_safe_int(contract.get("queue_batch_size"), 240000)),
        "SQL_LINK_SERVICE_HOT_MAX_ROWS": str(max(_safe_int(contract.get("hot_batch_size"), 240000) * 15, 3600000)),
        "SQL_LINK_SERVICE_QUEUE_MAX_ROWS": str(max(_safe_int(contract.get("queue_batch_size"), 240000) * 3, 720000)),
        "SQL_LINK_SERVICE_INTERVAL_SECONDS": "8" if convergence_enabled else "10" if turbo_enabled else "20",
        "SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS": "1",
        "SQL_LINK_SERVICE_SKIP_PROMOTION_IDLE_SHARDS": "1" if turbo_enabled else "0",
        "SQL_LINK_SERVICE_SKIP_IDLE_SENTINELS": "0",
        "SQL_LINK_SERVICE_ADAPTIVE_SHARD_ORDER": "1",
        "SQL_LINK_SERVICE_SENTINEL_SHARDS_FIRST": "1",
        "SQL_LINK_SERVICE_STALE_DECISION_SOURCE_CATCH_UP": "1" if (shards or lock_open) else "0",
        "SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS": "0" if lock_open else "45" if convergence_enabled else "60" if turbo_enabled else "90",
        "BACKLOG_DRAIN_SINGLE_WRITER_ONLY": "1",
        "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY": "1",
        "BACKLOG_PCORE_ALLOCATION_ACTIVE": "1",
        "BACKLOG_PCORE_PREPROCESS_WORKERS": str(max(_safe_int(lanes.get("live_preprocess_workers"), 1), _safe_int(os.getenv("BACKLOG_PCORE_PREPROCESS_WORKERS"), 8 if lock_open else 1))),
        "SQL_LINK_SERVICE_PREPROCESS_WORKERS": str(max(_safe_int(lanes.get("live_preprocess_workers"), 1), _safe_int(os.getenv("SQL_LINK_SERVICE_PREPROCESS_WORKERS"), 8 if lock_open else 1))),
        "SQL_LINK_SERVICE_SHARD_WRITER_LANES": str(max(_safe_int(lanes.get("live_preprocess_workers"), 1), _safe_int(os.getenv("SQL_LINK_SERVICE_SHARD_WRITER_LANES"), 8 if lock_open else 1))),
        "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": str(_safe_int(lanes.get("max_shard_writer_lanes"), _safe_int(lanes.get("live_preprocess_workers"), 1))),
        "BACKLOG_DRAIN_UNIFORM_WRITER_POLL_SECONDS": str(_safe_int(contract.get("poll_seconds"), 8)),
        "BACKLOG_DRAIN_UNIFORM_WAIT_TIMEOUT_SECONDS": str(_safe_int(contract.get("wait_timeout_seconds"), 240)),
        "STORAGE_BACKPRESSURE_AUTOPILOT_MAX_CYCLES": str(_safe_int(contract.get("storage_autopilot_cycles"), 2)),
        "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.16",
        "BOT_PROTECTED_VOLUME_DENYLIST": ",".join(PROTECTED_VOLUMES),
        "BOT_NEVER_TOUCH_VIDEO": "1",
    }
    if finalizer_enabled:
        finalizer_context = _as_dict(finalizer.get("context"))
        context_env = _as_dict(finalizer_context.get("control_env"))
        for key, value in context_env.items():
            if value not in {None, ""}:
                env[str(key)] = str(value)
        if _safe_int(finalizer_context.get("pending_bytes"), 0) > 0:
            env["SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_DRAIN"] = "1"
            env["SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_FILE_COUNT"] = str(max(_safe_int(finalizer_context.get("file_count"), 0), 1))
            if shards:
                env["SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_SHARDS"] = ",".join(shards)
        finalizer_workers = _safe_int(finalizer.get("preprocess_workers"), 0)
        if finalizer_workers > 0:
            env["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] = str(finalizer_workers)
            env["SQL_LINK_SERVICE_SHARD_WRITER_LANES"] = str(finalizer_workers)
            env["SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"] = str(max(_safe_int(env.get("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"), 0), finalizer_workers))
    return {str(key): str(value) for key, value in env.items()}


def _env_lines(payload: dict[str, Any]) -> list[str]:
    env = env_dict(payload)
    return [f"{key}={shlex.quote(str(value))}" for key, value in env.items()]


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    storage = load_json(health / "ingestion_storage_control_latest.json")
    accelerator = load_json(health / "backlog_pcore_accelerator_latest.json")
    pump = load_json(health / "backlog_pump_infrabots_latest.json")
    writer = load_json(health / "writer_cycle_coordinator_latest.json")
    writer_intel = load_json(health / "writer_process_intelligence_latest.json")
    storage_auto = load_json(health / "storage_backpressure_autopilot_latest.json")
    memory = load_json(health / "memory_pressure_intelligence_latest.json")
    contract = _speed_contract(storage, accelerator, pump, writer, writer_intel, memory, storage_auto)
    steps = _process_steps(contract)
    statuses = {
        "ingestion_storage_control": _status(storage),
        "backlog_pcore_accelerator": _status(accelerator),
        "backlog_pump_infrabots": _status(pump),
        "writer_cycle_coordinator": _status(writer),
        "writer_process_intelligence": _status(writer_intel),
        "storage_backpressure_autopilot": _status(storage_auto),
        "memory_pressure_intelligence": _status(memory),
    }
    stale_blockers = _as_list(_as_dict(_as_dict(pump.get("bots")).get("stale_signal_arbitrator_bot")).get("blockers"))
    overall = "advisory" if stale_blockers or statuses["ingestion_storage_control"] == "blocked" else "ready"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall == "ready",
        "overall_status": overall,
        "mode": "backlog_drain_uniform_process",
        "input_contracts": statuses,
        "speed_contract": contract,
        "uniform_process_steps": steps,
        "integration_contract": {
            "late_loaded_override": True,
            "unifies_storage_pcore_pump_writer_process": True,
            "single_sqlite_writer_only": True,
            "adds_parallel_sqlite_writers": False,
            "hot_path_speedups": [
                "single late override after pcore and pump contracts",
                "one canonical pressure source for all drain handoffs",
                "hot shard/source pinning before broad rotation",
                "bounded catch-up waves with shorter poll cadence",
                "heavier single-writer batches when memory is not in hard relief",
                "guarded turbo narrows the next cycle to sentinel plus hot shards when raw-live is green",
                "adaptive convergence only advances after the previous bounded cycle proves real drain progress",
                "sparse pressure finalizer switches from broad speed to byte-window cleanup when pressure_index is the last breach",
            ],
            "never_touch_protected_volumes": PROTECTED_VOLUMES,
        },
        "recommended_actions": ordered_unique(
            [
                "load the uniform drain override after pcore and pump overrides so later settings do not fight each other",
                "keep the active writer single-owner; use the uniform process to wait or hand off, not to launch another writer",
                f"pin hot shards first: {','.join(_as_list(contract.get('target_shards')))}" if _as_list(contract.get("target_shards")) else "",
                (
                    "use adaptive convergence 480k hot batches only after the last bounded cycle proved drain progress"
                    if _as_dict(contract.get("adaptive_convergence_contract")).get("enabled")
                    else "use sparse pressure finalizer byte-window caps when pressure_index is the only remaining breach"
                    if _as_dict(contract.get("sparse_pressure_finalizer_contract")).get("enabled")
                    else
                    "use turbo-plus 420k hot batches only when raw-live is green, source age is fresh, "
                    "and memory is clear"
                    if _as_dict(contract.get("turbo_contract")).get("turbo_plus_enabled")
                    else "use turbo 360k hot batches only when raw-live is green and memory is below hard/swap relief"
                )
                if _as_dict(contract.get("turbo_contract")).get("enabled")
                else "use 240k hot/queue batches only while memory is not in hard or swap relief",
                "rerun the uniform process after the current writer finishes to re-score source age and pressure",
            ]
        ),
    }


def write_outputs(payload: dict[str, Any], *, out_path: Path, override_path: Path, apply: bool = False) -> dict[str, Any]:
    write_payload(out_path, payload)
    applied = False
    if apply:
        lines = [
            "# Managed by scripts/ops/backlog_drain_uniform_process.py",
            f"# updated_at_utc={payload.get('timestamp_utc')}",
            *_env_lines(payload),
            "",
        ]
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text("\n".join(lines), encoding="utf-8")
        applied = True
    return {"out_path": str(out_path), "override_path": str(override_path), "applied": applied}


def main() -> int:
    parser = argparse.ArgumentParser(description="Uniform the backpressure drain process into one late-loaded speed and safety contract.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override", default=str(DEFAULT_OVERRIDE_PATH))
    args = parser.parse_args()
    payload = build_payload(PROJECT_ROOT)
    result = write_outputs(payload, out_path=Path(args.out), override_path=Path(args.override), apply=bool(args.apply))
    payload["write_result"] = result
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        contract = _as_dict(payload.get("speed_contract"))
        print(
            "backlog_drain_uniform_process "
            f"status={payload.get('overall_status')} "
            f"mode={contract.get('mode', '')} "
            f"apply={int(bool(args.apply))} "
            f"hot_shards={','.join(_as_list(contract.get('target_shards')))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
