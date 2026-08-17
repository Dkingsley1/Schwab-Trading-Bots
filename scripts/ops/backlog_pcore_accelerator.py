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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "backlog_pcore_accelerator_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.backlog_pcore_accelerator_override"
BACKLOG_GREEN_AGE_SECONDS = 900.0


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


def _storage_metrics(storage: dict[str, Any], governor: dict[str, Any]) -> dict[str, Any]:
    governor_storage = _as_dict(governor.get("storage_metrics"))
    backpressure = _as_dict(storage.get("backpressure"))
    stale = _as_dict(storage.get("stale_pending_locator"))
    oldest_sources = _as_list(governor_storage.get("oldest_sources")) or _as_list(stale.get("oldest_sources"))
    core = _safe_int(governor_storage.get("core_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), 0))
    total = _safe_int(governor_storage.get("total_pending_lines"), _safe_int(backpressure.get("total_pending_lines"), 0))
    overlay = _safe_int(governor_storage.get("overlay_pending_lines"), 0)
    oldest_age = _safe_float(governor_storage.get("oldest_pending_age_seconds"), _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0))
    target = _safe_int(governor_storage.get("target_pending_lines"), _safe_int(backpressure.get("pending_lines_threshold"), 15000)) or 15000
    line_green = core <= target and total <= max(target, core)
    age_green = oldest_age <= BACKLOG_GREEN_AGE_SECONDS
    overlay_green = overlay <= target if overlay > 0 else True
    green = bool(line_green and age_green and overlay_green)
    return {
        "core_pending_lines": core,
        "total_pending_lines": total,
        "overlay_pending_lines": overlay,
        "oldest_pending_age_seconds": round(oldest_age, 3),
        "target_pending_lines": target,
        "line_green": line_green,
        "age_green": age_green,
        "overlay_green": overlay_green,
        "green": green,
        "oldest_sources": oldest_sources[:8],
    }


def _storage_accelerator_contract(storage_payload: dict[str, Any]) -> dict[str, Any]:
    relief = _as_dict(storage_payload.get("backlog_relief_contract"))
    accelerator = _as_dict(relief.get("accelerator_contract"))
    p_core = _as_dict(relief.get("p_core_backlog_allocation_contract"))
    if not accelerator:
        accelerator = _as_dict(p_core.get("accelerator_contract"))
    return accelerator


def _writer_state(writer: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(writer.get("writer_state_before")) or _as_dict(writer.get("writer_state_after_wait"))


def _writer_active(writer: dict[str, Any], writer_intel: dict[str, Any]) -> bool:
    state = _writer_state(writer)
    health = _as_dict(writer_intel.get("writer_health"))
    return bool(
        state.get("active", False)
        or state.get("running", False)
        or health.get("active", False)
        or str(health.get("state") or "") in {"active_progressing", "stale_progress", "stalled"}
    )


def _process_topology(writer_intel: dict[str, Any]) -> dict[str, Any]:
    topology = _as_dict(writer_intel.get("process_topology"))
    return {
        "sql_link_writer_running_count": _safe_int(topology.get("sql_link_writer_running_count"), 1),
        "raw_sql_link_writer_running_count": _safe_int(topology.get("raw_sql_link_writer_running_count"), 1),
        "duplicate_sql_writer_processes": bool(topology.get("duplicate_sql_writer_processes", False)),
        "process_watchdog_status": str(topology.get("process_watchdog_status") or "unknown"),
        "process_fanout_status": str(topology.get("process_fanout_status") or "unknown"),
    }


def _host_lane_contract(governor: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    lanes = _as_dict(governor.get("host_lane_budget"))
    allocation = _as_dict(lanes.get("p_core_allocation_contract"))
    widening = _as_dict(lanes.get("p_core_widening_controller"))
    governor_memory = _as_dict(widening.get("memory_pressure_controller"))
    memory_class = _as_dict(memory.get("classification"))
    observer = _as_dict(memory.get("observer_overhead"))
    p_workers = _safe_int(lanes.get("selected_p_core_preprocess_workers"), _safe_int(memory_class.get("recommended_p_core_worker_cap"), 1))
    p_workers = max(p_workers, 1)
    memory_status = str(governor_memory.get("status") or memory_class.get("status") or "unknown")
    primary_lanes = _safe_int(lanes.get("primary_compute_lanes"), 1)
    env_max_lanes = _safe_int(os.getenv("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"), 0)
    env_workers = max(
        _safe_int(os.getenv("BACKLOG_PCORE_ACCELERATOR_WORKERS"), 0),
        _safe_int(os.getenv("BACKLOG_PCORE_PREPROCESS_WORKERS"), 0),
        _safe_int(os.getenv("SQL_LINK_SERVICE_PREPROCESS_WORKERS"), 0),
        _safe_int(os.getenv("SQL_LINK_SERVICE_SHARD_WRITER_LANES"), 0),
    )
    p_core_budget = max(primary_lanes, env_max_lanes, env_workers, 1)
    child_lane_cap = max(p_core_budget - 1, 1)
    memory_cap = _safe_int(
        governor_memory.get("max_memory_safe_workers"),
        _safe_int(memory_class.get("recommended_p_core_worker_cap"), p_workers),
    )
    memory_relief = memory_status in {"hard_relief", "swap_relief"}
    if env_workers > 0:
        if memory_relief:
            p_workers = min(max(p_workers, min(env_workers, 3)), max(memory_cap, 1), child_lane_cap)
        else:
            p_workers = min(max(p_workers, env_workers), child_lane_cap)
    reserve_target = _safe_int(os.getenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET"), _safe_int(allocation.get("user_app_reserved_p_cores"), 0))
    e_spillover = _safe_int(os.getenv("BACKLOG_ECORE_SPILLOVER_WORKERS"), _safe_int(lanes.get("efficiency_core_spillover"), 0))
    return {
        "primary_compute_lanes": primary_lanes,
        "effective_p_core_budget": p_core_budget,
        "selected_p_core_preprocess_workers": p_workers,
        "user_app_reserved_p_cores": max(reserve_target, 0),
        "efficiency_core_spillover": max(e_spillover, 0),
        "efficiency_core_total": _safe_int(lanes.get("efficiency_core_total"), 0),
        "memory_status": memory_status,
        "memory_worker_cap": memory_cap,
        "full_p_core_budget_requested": _env_flag("BACKLOG_PCORE_USE_FULL_PERFORMANCE_CORE_BUDGET"),
        "elastic_reserve_loan_enabled": _env_flag("BACKLOG_PCORE_ELASTIC_RESERVE_LOAN"),
        "memory_allocation_only_compression": bool(governor_memory.get("allocation_only_compression", False)),
        "memory_safe_to_widen": bool(_as_dict(memory.get("reopen_gate")).get("safe_to_widen_p_core_workers", False)),
        "observer_overhead_active": bool(observer.get("active", False)),
        "policy": str(lanes.get("policy") or "performance_core_primary_single_writer_with_user_app_reserve"),
    }


def _sleeve_pump_contract(host_lanes: dict[str, Any], storage: dict[str, Any]) -> dict[str, Any]:
    enabled = _env_flag("BACKLOG_SLEEVE_PUMP_ENABLED")
    p_workers = max(_safe_int(host_lanes.get("selected_p_core_preprocess_workers"), 1), 1)
    per_sleeve_workers = max(_safe_int(os.getenv("BACKLOG_SLEEVE_PUMP_WORKERS"), 1), 1)
    max_active_sleeves = max(_safe_int(os.getenv("BACKLOG_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES"), p_workers), 1)
    active_slots = max(1, min(max_active_sleeves, p_workers))
    oldest_sources = _as_list(storage.get("oldest_sources"))
    hot_source_count = len(oldest_sources)
    return {
        "enabled": enabled,
        "policy": str(os.getenv("BACKLOG_SLEEVE_PUMP_POLICY") or "per_sleeve_hotness_weighted_pcore_preprocess_single_writer_merge"),
        "share_policy": str(os.getenv("BACKLOG_SLEEVE_PUMP_SHARE_POLICY") or "hot_sleeves_first_then_round_robin"),
        "p_core_shared_preprocess_workers": int(p_workers),
        "per_sleeve_pump_workers": int(per_sleeve_workers),
        "max_active_sleeves_per_wave": int(max_active_sleeves),
        "selected_active_sleeve_slots": int(active_slots),
        "hot_source_count": int(hot_source_count),
        "sqlite_primary_writer_count": 1,
        "sqlite_parallelism": 1,
        "writes_sqlite_in_parallel": False,
        "control_env": {
            "BACKLOG_SLEEVE_PUMP_ENABLED": "1" if enabled else "0",
            "BACKLOG_SLEEVE_PUMP_WORKERS": str(per_sleeve_workers),
            "BACKLOG_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES": str(max_active_sleeves),
            "BACKLOG_SLEEVE_PUMP_POLICY": str(os.getenv("BACKLOG_SLEEVE_PUMP_POLICY") or "per_sleeve_hotness_weighted_pcore_preprocess_single_writer_merge"),
            "BACKLOG_SLEEVE_PUMP_SHARE_POLICY": str(os.getenv("BACKLOG_SLEEVE_PUMP_SHARE_POLICY") or "hot_sleeves_first_then_round_robin"),
            "SQL_LINK_SERVICE_SLEEVE_PUMP_ENABLED": "1" if enabled else "0",
            "SQL_LINK_SERVICE_SLEEVE_PUMP_WORKERS": str(per_sleeve_workers),
            "SQL_LINK_SERVICE_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES": str(max_active_sleeves),
        },
        "stop_conditions": [
            "active writer already owns the SQLite lock",
            "memory enters hard or swap relief",
            "sleeve pump slots stop reducing oldest pending age",
        ],
    }


def _single_writer_tuning_contract() -> dict[str, Any]:
    merge_seconds = max(_safe_int(os.getenv("SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"), 60), 1)
    hot_batch = max(_safe_int(os.getenv("SQL_LINK_SERVICE_HOT_BATCH_SIZE"), 120000), 1)
    queue_batch = max(_safe_int(os.getenv("SQL_LINK_SERVICE_QUEUE_BATCH_SIZE"), 80000), 1)
    sqlite_timeout = max(_safe_int(os.getenv("SQL_LINK_SERVICE_SQLITE_TIMEOUT"), 300), 1)
    lock_retries = max(_safe_int(os.getenv("SQL_LINK_SERVICE_LOCK_RETRIES"), 200), 1)
    cache_kb = max(_safe_int(os.getenv("SQLITE_CACHE_SIZE_KB"), 0), 0)
    mmap_allowed = _env_flag("SQLITE_ALLOW_MMAP", False)
    ops_mmap_allowed = _env_flag("BOT_OPS_SQLITE_ALLOW_MMAP", False)
    mmap_mb = max(_safe_int(os.getenv("SQLITE_MMAP_SIZE_MB"), 0), 0) if mmap_allowed else 0
    ops_mmap_mb = max(_safe_int(os.getenv("BOT_OPS_SQLITE_MMAP_SIZE_MB"), 0), 0) if ops_mmap_allowed else 0
    wal_threshold = _safe_float(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB"), 2.0)
    wal_growth = _safe_float(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB"), 1.5)
    return {
        "enabled": True,
        "policy": "one_heavier_sqlite_merge_writer_with_parallel_p_core_preprocess",
        "primary_merge_writer_count": 1,
        "sqlite_parallelism": 1,
        "merge_max_seconds_per_cycle": int(merge_seconds),
        "sqlite_timeout_seconds": int(sqlite_timeout),
        "sqlite_lock_retries": int(lock_retries),
        "sqlite_lock_retry_delay_seconds": _safe_float(os.getenv("SQL_LINK_SERVICE_LOCK_RETRY_DELAY_SECONDS"), 0.5),
        "hot_batch_size": int(hot_batch),
        "hot_max_rows": max(_safe_int(os.getenv("SQL_LINK_SERVICE_HOT_MAX_ROWS"), 1000000), 1),
        "queue_batch_size": int(queue_batch),
        "wal_checkpoint": {
            "enabled": _env_flag("SQL_LINK_SERVICE_AUTO_WAL_CHECKPOINT", True),
            "threshold_gb": round(wal_threshold, 3),
            "trigger_growth_gb": round(wal_growth, 3),
            "trigger_rows": max(_safe_int(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_ROWS"), 750000), 1),
            "min_interval_seconds": max(_safe_int(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_MIN_INTERVAL_SECONDS"), 900), 1),
            "mode": str(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_MODE") or "auto"),
        },
        "sqlite_memory": {
            "cache_size_kb": int(cache_kb),
            "mmap_size_mb": int(mmap_mb),
            "mmap_enabled": bool(mmap_allowed and mmap_mb > 0),
            "wal_autocheckpoint_pages": max(_safe_int(os.getenv("SQLITE_WAL_AUTOCHECKPOINT_PAGES"), 0), 0),
            "ops_cache_size_kb": max(_safe_int(os.getenv("BOT_OPS_SQLITE_CACHE_SIZE_KB"), 0), 0),
            "ops_mmap_size_mb": int(ops_mmap_mb),
            "ops_mmap_enabled": bool(ops_mmap_allowed and ops_mmap_mb > 0),
            "ops_busy_timeout_ms": max(_safe_int(os.getenv("BOT_OPS_SQLITE_BUSY_TIMEOUT_MS"), 0), 0),
        },
        "control_env": {
            "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": str(merge_seconds),
            "SQL_LINK_SERVICE_SQLITE_TIMEOUT": str(sqlite_timeout),
            "SQL_LINK_SERVICE_LOCK_RETRIES": str(lock_retries),
            "SQL_LINK_SERVICE_LOCK_RETRY_DELAY_SECONDS": str(_safe_float(os.getenv("SQL_LINK_SERVICE_LOCK_RETRY_DELAY_SECONDS"), 0.5)),
            "SQL_LINK_SERVICE_HOT_BATCH_SIZE": str(hot_batch),
            "SQL_LINK_SERVICE_HOT_MAX_ROWS": str(max(_safe_int(os.getenv("SQL_LINK_SERVICE_HOT_MAX_ROWS"), 1000000), 1)),
            "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": str(queue_batch),
            "SQL_LINK_SERVICE_AUTO_WAL_CHECKPOINT": "1" if _env_flag("SQL_LINK_SERVICE_AUTO_WAL_CHECKPOINT", True) else "0",
            "SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB": str(wal_threshold),
            "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_GROWTH_GB": str(wal_growth),
            "SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_ROWS": str(max(_safe_int(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_TRIGGER_ROWS"), 750000), 1)),
            "SQL_LINK_SERVICE_WAL_CHECKPOINT_MIN_INTERVAL_SECONDS": str(max(_safe_int(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_MIN_INTERVAL_SECONDS"), 900), 1)),
            "SQL_LINK_SERVICE_WAL_CHECKPOINT_MODE": str(os.getenv("SQL_LINK_SERVICE_WAL_CHECKPOINT_MODE") or "auto"),
            "SQLITE_CACHE_SIZE_KB": str(cache_kb),
            "SQLITE_MMAP_SIZE_MB": str(mmap_mb),
            "SQLITE_ALLOW_MMAP": "1" if mmap_allowed else "0",
            "SQLITE_WAL_AUTOCHECKPOINT_PAGES": str(max(_safe_int(os.getenv("SQLITE_WAL_AUTOCHECKPOINT_PAGES"), 0), 0)),
            "BOT_OPS_SQLITE_CACHE_SIZE_KB": str(max(_safe_int(os.getenv("BOT_OPS_SQLITE_CACHE_SIZE_KB"), 0), 0)),
            "BOT_OPS_SQLITE_MMAP_SIZE_MB": str(ops_mmap_mb),
            "BOT_OPS_SQLITE_ALLOW_MMAP": "1" if ops_mmap_allowed else "0",
            "BOT_OPS_SQLITE_BUSY_TIMEOUT_MS": str(max(_safe_int(os.getenv("BOT_OPS_SQLITE_BUSY_TIMEOUT_MS"), 0), 0)),
        },
        "stop_conditions": [
            "WAL growth triggers checkpoint pressure",
            "single writer lock wait exceeds configured timeout",
            "memory pressure enters hard relief",
            "merge cycles no longer reduce pending age or lines",
        ],
    }


def _accelerator_lanes(storage: dict[str, Any], host_lanes: dict[str, Any], sleeve_pump: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    p_workers = max(_safe_int(host_lanes.get("selected_p_core_preprocess_workers"), 1), 1)
    oldest_sources = _as_list(storage.get("oldest_sources"))
    hot_source_count = min(len(oldest_sources), max(p_workers, 1))
    density_workers = max(min(4, p_workers - 1), 1)
    stale_workers = max(min(3, p_workers), 1)
    lanes = [
        {
            "lane": "stale_source_locator",
            "class": "p_core_preprocess",
            "workers": stale_workers,
            "writes_sqlite": False,
            "purpose": "identify oldest exact files and shards before a writer pass",
        },
        {
            "lane": "jsonl_density_sampler",
            "class": "p_core_preprocess",
            "workers": density_workers,
            "writes_sqlite": False,
            "purpose": "sample sparse or huge JSONL files without letting them monopolize the writer",
        },
        {
            "lane": "shard_priority_planner",
            "class": "p_core_preprocess",
            "workers": 1,
            "writes_sqlite": False,
            "purpose": "rank hot, warm, and cold shards before handoff to the single writer",
        },
    ]
    sleeve = _as_dict(sleeve_pump)
    if bool(sleeve.get("enabled", False)):
        lanes.append(
            {
                "lane": "per_sleeve_pump_scheduler",
                "class": "p_core_preprocess",
                "workers": max(min(_safe_int(sleeve.get("selected_active_sleeve_slots"), p_workers), p_workers), 1),
                "writes_sqlite": False,
                "purpose": "budget hot sleeve queues across P-core preprocess slots before the one merge writer commits rows",
            }
        )
    lanes.extend(
        [
        {
            "lane": "oldest_work_catchup_scheduler",
            "class": "p_core_preprocess",
            "workers": max(min(hot_source_count, p_workers), 1),
            "writes_sqlite": False,
            "purpose": "schedule bounded catch-up waves around the oldest pending work",
        },
        {
            "lane": "sqlite_single_writer",
            "class": "exclusive_sqlite_writer",
            "workers": 1,
            "writes_sqlite": True,
            "purpose": "perform all SQLite writes through the one lock-owning writer",
        },
        ]
    )
    return lanes


def _wave_policy(storage: dict[str, Any], host_lanes: dict[str, Any], runtime: dict[str, Any], storage_accelerator: dict[str, Any]) -> dict[str, Any]:
    p_workers = max(_safe_int(host_lanes.get("selected_p_core_preprocess_workers"), 1), 1)
    memory_status = str(host_lanes.get("memory_status") or "unknown")
    runtime_status = _status(runtime)
    accelerator_wave = _as_dict(storage_accelerator.get("catch_up_wave_controller"))
    accelerator_limit = _safe_int(accelerator_wave.get("max_waves"), 0)
    accelerator_seconds = _safe_int(accelerator_wave.get("max_seconds_per_writer_cycle"), 0)
    lock_open = _lock_open_enabled()
    if lock_open:
        max_seconds = 240
        waves = 9
        mode = "locked_open_pcore_wave_9"
    elif memory_status in {"hard_relief", "swap_relief", "compression_relief"}:
        max_seconds = 20
        waves = 2
        mode = "memory_relief_bounded"
    elif runtime_status in {"blocked", "critical", "degraded"}:
        max_seconds = 25
        waves = 3
        mode = "runtime_guarded"
    elif storage.get("green"):
        max_seconds = 15
        waves = 1
        mode = "maintenance"
    else:
        max_seconds = 35 if p_workers >= 4 else 25
        waves = 3
        mode = "p_core_catch_up"
    if bool(storage_accelerator.get("enabled", False)) and accelerator_limit > 0 and not lock_open:
        waves = max(waves, accelerator_limit)
        max_seconds = max(max_seconds, accelerator_seconds)
        mode = str(storage_accelerator.get("mode") or mode)
    env_wave_limit = _safe_int(os.getenv("BACKLOG_CATCH_UP_WAVE_LIMIT") or os.getenv("WRITER_CYCLE_MAX_CATCH_UP_WAVES"), 0)
    env_max_seconds = _safe_int(os.getenv("BACKLOG_ACCELERATOR_MAX_SECONDS_PER_CYCLE"), 0)
    if env_wave_limit > 0:
        waves = max(waves, env_wave_limit)
    if env_max_seconds > 0:
        max_seconds = max(max_seconds, env_max_seconds)
    return {
        "mode": mode,
        "bounded_wave_limit": waves,
        "max_seconds_per_writer_cycle": max_seconds,
        "min_recheck_seconds": 45,
        "stop_conditions": [
            "one active SQL writer already owns the lock",
            "oldest pending age is below 15 minutes",
            "memory pressure moves from soft_guard to hard_relief/swap_relief",
            "backlog trend regresses after a writer pass",
        ],
    }


def _grade(storage: dict[str, Any], host_lanes: dict[str, Any], writer_active: bool, topology: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []
    if not bool(topology.get("duplicate_sql_writer_processes", False)):
        score += 20
    else:
        reasons.append("duplicate writer process risk")
    if bool(storage.get("line_green", False)):
        score += 20
    else:
        reasons.append("line backlog above target")
    if bool(storage.get("age_green", False)):
        score += 20
    else:
        reasons.append("oldest pending age not green")
    if str(host_lanes.get("memory_status") or "") in {"clear", "foreground_headroom", "soft_guard", "soft_memory_guard"}:
        score += 15
    else:
        reasons.append("memory still in relief mode")
    if _safe_int(host_lanes.get("selected_p_core_preprocess_workers"), 0) >= 3:
        score += 15
    else:
        reasons.append("P-core accelerator width is too narrow")
    if not bool(_as_dict(memory.get("observer_overhead")).get("active", False)):
        score += 5
    else:
        reasons.append("observer overhead is distorting pressure")
    if writer_active or storage.get("green"):
        score += 5
    else:
        reasons.append("writer is idle while backlog is not green")
    if score >= 90:
        letter = "A"
    elif score >= 80:
        letter = "B"
    elif score >= 70:
        letter = "C"
    elif score >= 60:
        letter = "D"
    else:
        letter = "F"
    return {
        "score": score,
        "letter": letter,
        "reasons": reasons,
        "policy": "grades_backlog_drain_bulletproofing_not_market_or_strategy_quality",
    }


def _decision(storage: dict[str, Any], writer: dict[str, Any], writer_intel: dict[str, Any], host_lanes: dict[str, Any], topology: dict[str, Any]) -> dict[str, Any]:
    state = _writer_state(writer)
    completed = _safe_int(state.get("completed_shard_count"), 0)
    planned = _safe_int(state.get("planned_shard_count"), 0)
    active = _writer_active(writer, writer_intel)
    duplicate = bool(topology.get("duplicate_sql_writer_processes", False))
    memory_status = str(host_lanes.get("memory_status") or "unknown")
    lock_open = _lock_open_enabled()
    if duplicate:
        action = "enforce_single_writer_guard"
        command = ["./scripts/ops/opsctl.sh", "process-fanout-guard", "--apply", "--json"]
        reason = "duplicate SQL writer risk must be cleared before accelerating backlog"
        apply_safe = True
    elif active:
        action = "observe_active_writer_locked_open" if lock_open else "observe_active_writer"
        command = ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"]
        reason = (
            f"writer is active at {completed}/{planned} shards; lock-open accelerators keep preprocessing armed without launching a competing writer"
            if lock_open
            else f"writer is active at {completed}/{planned} shards; do not launch a competing writer"
        )
        apply_safe = False
    elif memory_status in {"hard_relief", "swap_relief"}:
        action = "hold_for_memory_relief"
        command = ["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--apply", "--json"]
        reason = "memory relief is too strong for new backlog waves"
        apply_safe = True
    elif lock_open:
        action = "locked_open_p_core_drain"
        command = ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"]
        reason = "operator lock-open policy keeps P-core drainers and accelerators active even when the backlog snapshot is green"
        apply_safe = True
    elif not bool(storage.get("green", False)):
        action = "run_bounded_p_core_catch_up"
        command = ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"]
        reason = "backlog lines or age are not green; run bounded waves through the single writer"
        apply_safe = True
    else:
        action = "park_to_maintenance"
        command = ["./scripts/ops/opsctl.sh", "autonomic-governor", "--apply", "--json"]
        reason = "backlog is green; keep accelerators in maintenance and allow other gates to re-open gradually"
        apply_safe = True
    return {
        "action": action,
        "next_command": command,
        "apply_safe": apply_safe,
        "reason": reason,
        "writer_shards": {"completed": completed, "planned": planned, "step": state.get("current_step", ""), "status": state.get("status", "")},
    }


def _needs(storage: dict[str, Any], decision: dict[str, Any], host_lanes: dict[str, Any], memory: dict[str, Any]) -> list[dict[str, Any]]:
    needs: list[dict[str, Any]] = []
    if not bool(storage.get("green", False)):
        oldest = _as_list(storage.get("oldest_sources"))
        exact = oldest[0] if oldest and isinstance(oldest[0], dict) else {}
        needs.append(
            {
                "blocker": "backlog_age_or_lines_not_green_for_p_core_acceleration",
                "exact_file": exact.get("source_rel") or "governance/health/ingestion_storage_control_latest.json",
                "exact_shard": exact.get("shard") or "",
                "command": decision.get("next_command", []),
                "expected_impact": "Uses P-core preprocess accelerators to prioritize stale work, then hands one bounded batch to the exclusive SQLite writer.",
                "risk_level": "low" if decision.get("apply_safe") else "observe",
                "stop_when": "oldest pending age is under 15 minutes and core/overlay pending are below target.",
            }
        )
    if str(host_lanes.get("memory_status") or "") not in {"clear", "foreground_headroom"}:
        needs.append(
            {
                "blocker": "memory_headroom_limits_backlog_p_core_width",
                "exact_file": "governance/health/memory_pressure_intelligence_latest.json",
                "exact_shard": "",
                "command": ["./scripts/ops/opsctl.sh", "memory-pressure-intelligence", "--apply", "--json"],
                "expected_impact": "Refreshes the memory gate before widening P-core backlog accelerators.",
                "risk_level": "low",
                "stop_when": "memory is clear for two consecutive samples or the cap reaches the benchmark limit.",
            }
        )
    if bool(_as_dict(memory.get("observer_overhead")).get("active", False)):
        needs.append(
            {
                "blocker": "observer_overhead_distorts_backlog_pressure",
                "exact_file": "governance/health/memory_pressure_intelligence_latest.json",
                "exact_shard": "",
                "command": [],
                "expected_impact": "Closing or reducing high-overhead monitors makes backlog pressure readings cleaner.",
                "risk_level": "operator_choice",
                "stop_when": "observer_overhead.active is false.",
            }
        )
    return needs


def _env_lines(payload: dict[str, Any]) -> list[str]:
    host_lanes = _as_dict(payload.get("host_lane_contract"))
    wave = _as_dict(payload.get("wave_policy"))
    decision = _as_dict(payload.get("decision_packet"))
    sleeve_pump = _as_dict(payload.get("sleeve_pump_contract"))
    sleeve_env = _as_dict(sleeve_pump.get("control_env"))
    writer_tuning = _as_dict(payload.get("single_writer_tuning_contract"))
    writer_env = _as_dict(writer_tuning.get("control_env"))
    lock_open = _lock_open_enabled()
    accelerator_mode = str(wave.get("mode") or ("locked_open_pcore_wave_9" if lock_open else "p_core_catch_up"))
    env = {
        "BACKLOG_PCORE_DRAIN_LOCK_OPEN": "1" if lock_open else "0",
        "BACKLOG_ACCELERATOR_LOCK_OPEN": "1" if lock_open else "0",
        "BACKLOG_DRAIN_LOCK_OPEN": "1" if lock_open else "0",
        "BACKLOG_PCORE_ACCELERATOR_ENABLED": "1",
        "BACKLOG_ACCELERATOR_ENABLED": "1",
        "BACKLOG_ACCELERATOR_MODE": accelerator_mode,
        "SQL_LINK_SERVICE_CATCH_UP_WAVE": "1",
        "BACKLOG_PCORE_ACCELERATOR_ACTION": str(decision.get("action") or "observe"),
        "BACKLOG_PCORE_ACCELERATOR_WORKERS": str(host_lanes.get("selected_p_core_preprocess_workers") or 1),
        "BACKLOG_PCORE_ALLOCATION_ACTIVE": "1",
        "BACKLOG_PCORE_PREPROCESS_WORKERS": str(host_lanes.get("selected_p_core_preprocess_workers") or 1),
        "BACKLOG_PCORE_USER_APP_RESERVE": str(host_lanes.get("user_app_reserved_p_cores") or 0),
        "BACKLOG_ECORE_SPILLOVER_WORKERS": str(host_lanes.get("efficiency_core_spillover") or 0),
        "BOT_EFFICIENCY_CORE_SPILLOVER_COUNT": str(host_lanes.get("efficiency_core_spillover") or 0),
        "BACKLOG_SQLITE_WRITER_WORKERS": "1",
        "BACKLOG_ACCELERATOR_SQLITE_PARALLELISM": "1",
        "BACKLOG_CATCH_UP_WAVE_LIMIT": str(wave.get("bounded_wave_limit") or 1),
        "BACKLOG_ACCELERATOR_MAX_SECONDS_PER_CYCLE": str(wave.get("max_seconds_per_writer_cycle") or 25),
        "BACKLOG_ACCELERATOR_SINGLE_WRITER_GUARD": "1",
        "BACKLOG_PCORE_USE_FULL_PERFORMANCE_CORE_BUDGET": "1" if host_lanes.get("full_p_core_budget_requested") else "0",
        "BACKLOG_PCORE_ELASTIC_RESERVE_LOAN": "1" if host_lanes.get("elastic_reserve_loan_enabled") else "0",
        "BOT_CPU_ALLOCATION_POLICY": str(host_lanes.get("policy") or "performance_core_primary_full_budget_single_writer_with_elastic_user_app_reserve"),
    }
    env.update({str(key): str(value) for key, value in sleeve_env.items()})
    env.update({str(key): str(value) for key, value in writer_env.items()})
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
            "# Managed by scripts/ops/backlog_pcore_accelerator.py",
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
    governor = load_json(health / "autonomic_resource_governor_latest.json")
    storage_payload = load_json(health / "ingestion_storage_control_latest.json")
    writer = load_json(health / "writer_cycle_coordinator_latest.json")
    writer_intel = load_json(health / "writer_process_intelligence_latest.json")
    runtime = load_json(health / "runtime_throttle_control_latest.json")
    memory = load_json(health / "memory_pressure_intelligence_latest.json")
    drainer = load_json(health / "backpressure_drainer_fleet_latest.json")
    host_lanes = _host_lane_contract(governor, memory)
    storage = _storage_metrics(storage_payload, governor)
    storage_accelerator = _storage_accelerator_contract(storage_payload)
    topology = _process_topology(writer_intel)
    writer_is_active = _writer_active(writer, writer_intel)
    sleeve_pump = _sleeve_pump_contract(host_lanes, storage)
    single_writer_tuning = _single_writer_tuning_contract()
    lanes = _accelerator_lanes(storage, host_lanes, sleeve_pump)
    wave = _wave_policy(storage, host_lanes, runtime, storage_accelerator)
    decision = _decision(storage, writer, writer_intel, host_lanes, topology)
    always_armed = bool(
        _env_flag("BACKLOG_ACCELERATOR_ALWAYS_ARMED")
        or _env_flag("STORAGE_BACKPRESSURE_AUTOPILOT_ALWAYS_ARMED")
        or _lock_open_enabled()
    )
    lock_open = _lock_open_enabled()
    if lock_open and str(decision.get("action") or "") == "always_armed_maintenance":
        decision = {
            **decision,
            "action": "locked_open_p_core_drain",
            "reason": "operator lock-open policy keeps drainers active instead of parking to maintenance",
        }
    elif always_armed and str(decision.get("action") or "") == "park_to_maintenance":
        decision = {
            **decision,
            "action": "always_armed_maintenance",
            "reason": "backlog is green right now; accelerators remain prearmed for the next drift without launching a duplicate writer",
        }
    grade = _grade(storage, host_lanes, writer_is_active, topology, memory)
    needs = _needs(storage, decision, host_lanes, memory)
    overall = "ready" if grade["score"] >= 90 and not needs else "advisory"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall == "ready",
        "overall_status": overall,
        "mode": "backlog_pcore_accelerator",
        "input_contracts": {
            "autonomic_resource_governor": _status(governor),
            "ingestion_storage_control": _status(storage_payload),
            "writer_cycle_coordinator": _status(writer),
            "writer_process_intelligence": _status(writer_intel),
            "runtime_throttle_control": _status(runtime),
            "memory_pressure_intelligence": _status(memory),
            "backpressure_drainer_fleet": _status(drainer),
        },
        "host_lane_contract": host_lanes,
        "storage_contract": storage,
        "storage_accelerator_contract": storage_accelerator,
        "always_armed_contract": {
            "enabled": always_armed,
            "locked_open": lock_open,
            "policy": "prearmed_governed_acceleration_single_sqlite_writer_only",
            "apply_when": [
                "pending lines rise above target",
                "oldest pending age breaches target",
                "sparse huge JSONL tails reappear",
                "storage backpressure autopilot reports a repair plan",
            ],
            "hold_when": [
                "another SQL writer already owns the lock",
                "memory enters hard relief or swap relief",
                "protected volume denylist would be touched",
            ],
        },
        "process_topology": topology,
        "sleeve_pump_contract": sleeve_pump,
        "single_writer_tuning_contract": single_writer_tuning,
        "accelerator_lanes": lanes,
        "wave_policy": wave,
        "decision_packet": decision,
        "bulletproof_score": grade,
        "what_do_you_need": {
            "status": "needs_action" if needs else "clear",
            "items": needs,
            "next_command": decision.get("next_command", []),
        },
        "integration_contract": {
            "single_sqlite_writer_only": True,
            "p_core_accelerators_preprocess_only": True,
            "sqlite_write_parallelism": 1,
            "single_writer_tuned_for_heavier_cycles": True,
            "uses_autonomic_resource_governor": True,
            "uses_memory_pressure_intelligence": True,
            "uses_writer_process_intelligence": True,
            "uses_ingestion_storage_control": True,
            "p_cores_are_primary": True,
            "e_cores_are_spillover_only": True,
            "full_p_core_budget_available": bool(host_lanes.get("full_p_core_budget_requested", False)),
            "sleeve_pumps_enabled": bool(sleeve_pump.get("enabled", False)),
            "never_touch_protected_volumes": ["/Volumes/VIDEO"],
            "always_armed_accelerators": always_armed,
            "policy": "accelerate_discovery_priority_and_batch_preparation_not_parallel_sqlite_writes",
        },
        "recommended_actions": ordered_unique(
            [
                "let active writer cycles finish before launching another writer",
                "use P-core workers for stale-source locating, density sampling, shard priority, and catch-up scheduling",
                "route hot sleeves through per-sleeve pump slots before the single merge writer commits rows",
                "keep SQLite writes at one exclusive writer even when accelerators widen",
                "hold training and optional collectors until age, memory, and runtime gates clear",
                "close or reduce high-overhead observers if observer_overhead is active",
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate P-core backlog accelerators around the single SQLite writer.")
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
        decision = _as_dict(payload.get("decision_packet"))
        score = _as_dict(payload.get("bulletproof_score"))
        print(
            "backlog_pcore_accelerator "
            f"status={payload['overall_status']} "
            f"action={decision.get('action')} "
            f"grade={score.get('letter')} "
            f"score={score.get('score')} "
            f"applied={result['applied']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
