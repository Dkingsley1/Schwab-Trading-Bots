#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "platform_settlement_stabilization_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "platform_settlement_stabilization_v1.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.platform_settlement_stabilization_override"

SECTION_KEYS: tuple[str, ...] = (
    "queue_decay_meter",
    "single_writer_guard",
    "market_hours_cadence_smoother",
    "global_clear_settlement_guard",
    "paper_collection_floor_smoother",
    "off_hours_drain_plan",
    "stabilization_effectiveness_memory",
)

CONTROLS: tuple[dict[str, str], ...] = (
    {"id": "queue_decay_meter", "title": "Queue decay meter", "env_key": "SETTLEMENT_QUEUE_DECAY_METER_ENABLED"},
    {"id": "single_writer_guard", "title": "Single writer guard", "env_key": "SETTLEMENT_SINGLE_WRITER_GUARD_ENABLED"},
    {"id": "market_hours_cadence_smoother", "title": "Market-hours cadence smoother", "env_key": "SETTLEMENT_MARKET_HOURS_CADENCE_ENABLED"},
    {"id": "global_clear_settlement_guard", "title": "Global-clear settlement guard", "env_key": "SETTLEMENT_GLOBAL_CLEAR_GUARD_ENABLED"},
    {"id": "paper_collection_floor_smoother", "title": "Paper collection floor smoother", "env_key": "SETTLEMENT_PAPER_COLLECTION_FLOOR_ENABLED"},
    {"id": "off_hours_drain_plan", "title": "Off-hours drain plan", "env_key": "SETTLEMENT_OFF_HOURS_DRAIN_PLAN_ENABLED"},
    {"id": "stabilization_effectiveness_memory", "title": "Stabilization effectiveness memory", "env_key": "SETTLEMENT_EFFECTIVENESS_MEMORY_ENABLED"},
)

INFRA_ASSIGNMENTS: dict[str, list[str]] = {
    "queue_decay_meter": ["queue_decay_meter_bot", "backpressure_slo_bot", "storage_backpressure_autopilot"],
    "single_writer_guard": ["single_writer_settlement_guard", "writer_cycle_coordinator", "sql_link_writer_watchdog"],
    "market_hours_cadence_smoother": ["market_hours_cadence_infrabot", "runtime_throttle_control", "pressure_relief_governor"],
    "global_clear_settlement_guard": ["global_halt_clearance_watcher", "global_killswitch_monitor", "remote_alert_control"],
    "paper_collection_floor_smoother": ["paper_collection_floor_infrabot", "paper_400_ramp_control", "collector_priority_ranker"],
    "off_hours_drain_plan": ["off_hours_drain_scheduler", "external_backlog_retry_bot", "retention_debt_sheriff"],
    "stabilization_effectiveness_memory": ["stabilization_memory_writer", "self_model_lesson_linker", "fix_effectiveness_replay_scorer"],
}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "enabled", "active"}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _health(project_root: Path, name: str) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / name)
    return payload if isinstance(payload, dict) else {}


def _queue_metrics(project_root: Path) -> dict[str, Any]:
    backpressure = _health(project_root, "ingestion_backpressure_latest.json")
    storage = _health(project_root, "ingestion_storage_control_latest.json")
    storage_bp = _as_dict(storage.get("backpressure"))
    throughput = _as_dict(storage.get("throughput"))
    bounded = _as_dict(storage.get("bounded_recovery_contract"))
    core = max(_safe_int(backpressure.get("pending_lines"), 0), _safe_int(storage_bp.get("core_pending_lines"), 0))
    deferred = max(_safe_int(backpressure.get("pending_lines_deferred"), 0), _safe_int(storage_bp.get("deferred_pending_lines"), 0))
    cold = max(_safe_int(backpressure.get("pending_lines_cold"), 0), _safe_int(storage_bp.get("cold_pending_lines"), 0))
    support = max(
        _safe_int(backpressure.get("pending_lines_support_telemetry"), 0),
        _safe_int(storage_bp.get("support_pending_lines"), 0),
    )
    total = max(_safe_int(backpressure.get("pending_lines_total"), 0), _safe_int(storage_bp.get("total_pending_lines"), 0), core + deferred + cold + support)
    threshold = max(_safe_int(storage_bp.get("pending_lines_threshold"), 15000), 1)
    estimated = storage_bp.get("estimated_total_drain_minutes")
    if estimated is None:
        estimated = bounded.get("estimated_total_drain_minutes")
    return {
        "core_pending_lines": core,
        "deferred_pending_lines": deferred,
        "cold_pending_lines": cold,
        "support_pending_lines": support,
        "total_pending_lines": total,
        "pending_lines_threshold": threshold,
        "pending_ratio": round(total / float(threshold), 6),
        "pressure_index": round(_safe_float(storage.get("pressure_index"), total / float(threshold)), 6),
        "severity": str(storage.get("severity") or "unknown"),
        "estimated_total_drain_minutes": round(_safe_float(estimated), 3) if estimated is not None else None,
        "throughput_rows_per_second": round(_safe_float(throughput.get("throughput_rows_per_second"), 0.0), 6),
        "merged_rows_this_cycle": _safe_int(throughput.get("merged_rows_this_cycle"), 0),
        "active_drain_progress": _bool(bounded.get("active_drain_progress")),
        "drain_delta_total_lines": _safe_int(bounded.get("drain_delta_total_lines"), 0),
    }


def _queue_decay_meter(project_root: Path, previous: dict[str, Any]) -> dict[str, Any]:
    metrics = _queue_metrics(project_root)
    prev_metrics = _as_dict(_as_dict(_as_dict(previous.get("sections")).get("queue_decay_meter")).get("metrics"))
    prev_total = _safe_int(prev_metrics.get("total_pending_lines"), metrics["total_pending_lines"])
    delta = metrics["total_pending_lines"] - prev_total
    active = metrics["total_pending_lines"] > metrics["pending_lines_threshold"]
    progress = bool(metrics["active_drain_progress"] or metrics["throughput_rows_per_second"] > 0.0 or delta < 0)
    estimated = _safe_float(metrics.get("estimated_total_drain_minutes"), 0.0)
    status = "ready"
    if active and not progress:
        status = "needs_work"
    elif active or estimated > 15.0:
        status = "watch"
    return {
        "overall_status": status,
        "queue_backpressure_active": bool(active),
        "progress_observed": bool(progress),
        "metrics": {**metrics, "delta_since_previous_total_lines": delta},
        "assigned_infrabots": INFRA_ASSIGNMENTS["queue_decay_meter"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "backpressure-drainers", "--apply", "--ttl-seconds", "900", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "health-fast", "--json"],
        ],
        "settlement_contract": [
            "prove_queue_decay_before_new_expansion",
            "prefer_one_focused_drainer_handoff_over_parallel_writers",
            "do_not_count_queue_as_settled_until_pending_ratio_is_under_one",
        ],
    }


def _writer_single_primary_contract(project_root: Path) -> dict[str, Any]:
    intelligence = _health(project_root, "writer_process_intelligence_latest.json")
    writer_health = _as_dict(intelligence.get("writer_health"))
    lane_contract = _as_dict(writer_health.get("shard_writer_lane_contract"))
    primary_merge_writer_count = max(
        _safe_int(lane_contract.get("primary_merge_writer_count"), 0),
        _safe_int(writer_health.get("primary_merge_writer_count"), 0),
    )
    sqlite_primary_writer_count = _safe_int(lane_contract.get("sqlite_primary_writer_count"), primary_merge_writer_count)
    active_child_writer_count = _safe_int(writer_health.get("active_child_writer_count"), 0)
    lock_held = _bool(writer_health.get("writer_lock_held"))
    single_primary_merge_writer = bool(
        _bool(lane_contract.get("single_primary_merge_writer"))
        and primary_merge_writer_count <= 1
        and sqlite_primary_writer_count <= 1
        and lock_held
    )
    return {
        "overall_status": str(intelligence.get("overall_status") or "missing"),
        "single_primary_merge_writer": single_primary_merge_writer,
        "primary_merge_writer_count": primary_merge_writer_count,
        "sqlite_primary_writer_count": sqlite_primary_writer_count,
        "active_child_writer_count": active_child_writer_count,
        "writer_lock_held": lock_held,
        "writer_lane_policy": str(writer_health.get("writer_lane_policy") or lane_contract.get("policy") or ""),
    }


def _single_writer_guard(project_root: Path, queue: dict[str, Any]) -> dict[str, Any]:
    process = _health(project_root, "process_watchdog_latest.json")
    drainer = _health(project_root, "backpressure_drainer_fleet_latest.json")
    writer_contract = _writer_single_primary_contract(project_root)
    statuses = [row for row in _as_list(process.get("status")) if isinstance(row, dict)]
    writer_rows = [row for row in statuses if str(row.get("name") or "") == "sql_link_writer"]
    raw_running = sum(_safe_int(row.get("running"), 0) for row in writer_rows)
    writer_active = _bool(drainer.get("writer_active"))
    lock_held = _bool(drainer.get("writer_lock_held"))
    wrapper_chain_only = bool(raw_running > 1 and not writer_active and not lock_held)
    guarded_single_writer_chain = bool(
        raw_running > 1
        and (writer_active or lock_held)
        and _bool(writer_contract.get("single_primary_merge_writer"))
    )
    running = 1 if (wrapper_chain_only or guarded_single_writer_chain) else raw_running
    queue_active = _bool(queue.get("queue_backpressure_active"))
    status = "ready"
    if running > 1:
        status = "blocked"
    elif running <= 0 and queue_active:
        status = "needs_work"
    elif queue_active and not (writer_active or lock_held or _bool(queue.get("progress_observed"))):
        status = "watch"
    return {
        "overall_status": status,
        "sql_link_writer_running_count": running,
        "raw_sql_link_writer_running_count": raw_running,
        "wrapper_chain_only": wrapper_chain_only,
        "guarded_single_writer_chain": guarded_single_writer_chain,
        "writer_single_primary_contract": writer_contract,
        "writer_active": writer_active,
        "writer_lock_held": lock_held,
        "queue_backpressure_active": queue_active,
        "assigned_infrabots": INFRA_ASSIGNMENTS["single_writer_guard"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--poll-seconds", "20", "--wait-timeout-seconds", "60", "--command-timeout-seconds", "120", "--maintenance-force", "--json"],
            ["./scripts/ops/opsctl.sh", "backpressure-drainers", "--json"],
        ],
        "writer_contract": [
            "one_sqlite_writer_only",
            "writer_cycle_coordinator_may_follow_through_but_must_not_spawn_parallel_sqlite_writers",
            "writer_progress_is_measured_by_queue_delta_and_throughput_not_process_count_alone",
        ],
    }


def _market_hours_cadence(project_root: Path) -> dict[str, Any]:
    runtime = _health(project_root, "runtime_throttle_control_latest.json")
    pressure = _health(project_root, "pressure_relief_control_latest.json")
    cotenant = _as_dict(runtime.get("cotenant_awareness_contract"))
    saturation = _safe_float(runtime.get("host_saturation_score"), _safe_float(pressure.get("host_saturation_score"), 0.0))
    compute_level = str(runtime.get("compute_pressure_level") or pressure.get("compute_pressure_level") or "unknown")
    memory_level = str(runtime.get("memory_pressure_level") or pressure.get("memory_pressure_level") or "unknown")
    open_apps = _as_list(cotenant.get("open_apps"))
    co_level = str(cotenant.get("co_running_level") or "unknown")
    status = "ready"
    if saturation >= 85.0 or compute_level in {"critical", "high"}:
        status = "needs_work"
    elif saturation >= 65.0 or compute_level == "elevated" or memory_level == "elevated" or co_level == "heavy_competition":
        status = "watch"
    return {
        "overall_status": status,
        "host_saturation_score": round(saturation, 3),
        "compute_pressure_level": compute_level,
        "memory_pressure_level": memory_level,
        "co_running_level": co_level,
        "open_apps": open_apps,
        "assigned_infrabots": INFRA_ASSIGNMENTS["market_hours_cadence_smoother"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "creative-cotenant-guard", "apply", "--json"],
        ],
        "cadence_contract": [
            "foreground_apps_are_cotenants",
            "market_hours_collectors_prefer_thin_sample_under_high_pressure",
            "support_and_report_jobs_wait_for_quiet_windows",
        ],
    }


def _global_clear_guard(project_root: Path, queue: dict[str, Any]) -> dict[str, Any]:
    halt = _health(project_root, "global_halt_auto_clear_latest.json") or _health(project_root, "global_killswitch_latest.json")
    blockers = [str(item) for item in _as_list(halt.get("clear_blockers"))]
    halt_active = _bool(halt.get("halt"))
    queue_active = _bool(queue.get("queue_backpressure_active"))
    status = "ready"
    if halt_active:
        status = "needs_work"
    elif queue_active or "queue_backpressure_active" in blockers:
        status = "watch"
    return {
        "overall_status": status,
        "halt": halt_active,
        "halt_state": str(halt.get("halt_state") or "unknown"),
        "clear_ready": _bool(halt.get("clear_ready")),
        "clear_blockers": blockers,
        "queue_backpressure_active": queue_active,
        "assigned_infrabots": INFRA_ASSIGNMENTS["global_clear_settlement_guard"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "global-halt-status", "--json"],
            ["./scripts/ops/opsctl.sh", "global-halt-auto-clear", "--json"],
        ],
        "clear_contract": [
            "notify_when_global_halt_clears_on_its_own",
            "queue_backpressure_may_block_clear_without_triggering_new_halt",
            "auto_clear_waits_for_queue_and_runtime_settlement",
        ],
    }


def _registry_counts(project_root: Path) -> dict[str, int]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    active = collecting = paper = excluded = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        active += int(_bool(row.get("active")))
        collecting += int(_bool(row.get("data_collection_active")) or str(row.get("lifecycle_state") or "") == "data_collection_only")
        paper += int(_bool(row.get("paper_trade_active")) or _bool(row.get("paper_trading_active")) or _bool(row.get("paper_tagged")))
        excluded += int(_bool(row.get("training_excluded")) or _bool(row.get("exclude_from_training")))
    return {"total_bots": len(rows), "active_bots": active, "collecting_bots": collecting, "paper_tagged_bots": paper, "training_excluded_bots": excluded}


def _paper_floor(project_root: Path) -> dict[str, Any]:
    ramp = _health(project_root, "paper_400_ramp_control_latest.json")
    rollup = _health(project_root, "data_collection_observation_rollup_latest.json")
    registry = _registry_counts(project_root)
    collector_count = _safe_int(rollup.get("collector_count"), registry["collecting_bots"])
    observed = _safe_int(rollup.get("bots_with_observations"), 0)
    zero_obs = _safe_int(rollup.get("zero_observation_count"), max(collector_count - observed, 0))
    coverage = (observed / float(collector_count)) if collector_count > 0 else 1.0
    status = "ready"
    if zero_obs > 0 or coverage < 0.95:
        status = "needs_work"
    elif str(ramp.get("overall_status") or "ready") in {"degraded", "needs_work"}:
        status = "watch"
    return {
        "overall_status": status,
        "collector_count": collector_count,
        "bots_with_observations": observed,
        "zero_observation_count": zero_obs,
        "observation_coverage_ratio": round(coverage, 6),
        "registry": registry,
        "paper_ramp_status": str(ramp.get("overall_status") or "missing"),
        "assigned_infrabots": INFRA_ASSIGNMENTS["paper_collection_floor_smoother"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "data-collection-observation-rollup", "--json"],
            ["./scripts/ops/opsctl.sh", "paper-400-ramp", "--json"],
            ["./scripts/ops/opsctl.sh", "paper-lock", "--json"],
        ],
        "floor_contract": [
            "new_bots_collect_before_training",
            "paper_trade_lock_stays_on_for_all_paper_sleeves",
            "paper_capacity_ramps_only_after_collection_coverage_stays_high",
        ],
    }


def _off_hours_plan(project_root: Path) -> dict[str, Any]:
    drain = _health(project_root, "external_backlog_drain_latest.json")
    storage = _health(project_root, "ingestion_storage_control_latest.json")
    off_hours = _as_dict(drain.get("off_hours_window"))
    blocked = [str(item) for item in _as_list(drain.get("blocked_reasons"))]
    material = _bool(drain.get("material_drain_recommended"))
    market_guard = "market_hours_guard" in blocked
    status = "ready"
    if material and market_guard:
        status = "watch"
    elif material and not _bool(off_hours.get("active")):
        status = "watch"
    elif material and str(drain.get("overall_status") or "") == "blocked":
        status = "needs_work"
    return {
        "overall_status": status,
        "external_drain_status": str(drain.get("overall_status") or "missing"),
        "material_drain_recommended": material,
        "blocked_reasons": blocked,
        "off_hours_window": off_hours,
        "storage_mode": str(drain.get("storage_mode") or _as_dict(storage.get("storage")).get("storage_mode") or "unknown"),
        "assigned_infrabots": INFRA_ASSIGNMENTS["off_hours_drain_plan"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "external-backlog-drain", "--apply", "--follow-through", "--json"],
            ["./scripts/ops/opsctl.sh", "retention-debt-sheriff", "--apply", "--json"],
        ],
        "drain_contract": [
            "market_hours_guard_prevents_heavy_external_drain",
            "off_hours_window_burns_down_deferred_and_cold_lanes",
            "support_watchdog_shard_stays_separate_from_core_decisions",
        ],
    }


def _effectiveness_memory(project_root: Path, previous: dict[str, Any], queue: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    prev_sections = _as_dict(previous.get("sections"))
    prev_queue_metrics = _as_dict(_as_dict(prev_sections.get("queue_decay_meter")).get("metrics"))
    prev_runtime = _as_dict(prev_sections.get("market_hours_cadence_smoother"))
    current_total = _safe_int(_as_dict(queue.get("metrics")).get("total_pending_lines"), 0)
    prev_total = _safe_int(prev_queue_metrics.get("total_pending_lines"), current_total)
    current_sat = _safe_float(runtime.get("host_saturation_score"), 0.0)
    prev_sat = _safe_float(prev_runtime.get("host_saturation_score"), current_sat)
    queue_delta = current_total - prev_total
    saturation_delta = current_sat - prev_sat
    improving = queue_delta < 0 or saturation_delta < 0
    status = "ready" if improving or not previous else "watch"
    if queue_delta > 5000 and saturation_delta > 10.0:
        status = "needs_work"
    return {
        "overall_status": status,
        "previous_artifact_seen": bool(previous),
        "queue_delta_total_lines": queue_delta,
        "host_saturation_delta": round(saturation_delta, 3),
        "improving": bool(improving),
        "assigned_infrabots": INFRA_ASSIGNMENTS["stabilization_effectiveness_memory"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "platform-settlement-stabilization", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "system-self-model", "--json"],
        ],
        "memory_contract": [
            "stabilization_writes_before_after_evidence",
            "self_model_uses_settlement_artifacts_for_future_upgrade_suggestions",
            "do_not_repeat_a_fix_that_failed_to_reduce_pressure",
        ],
    }


def _status_rows(sections: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in SECTION_KEYS:
        section = _as_dict(sections.get(key))
        status = str(section.get("overall_status") or "missing")
        rows.append({"section": key, "overall_status": status, "rank": status_rank(status)})
    return rows


def _worst_status(rows: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("overall_status") or "") for row in rows}
    if statuses & {"blocked", "critical"}:
        return "blocked"
    if statuses & {"degraded"}:
        return "degraded"
    if statuses & {"needs_work"}:
        return "needs_work"
    if statuses & {"watch", "thin", "missing"}:
        return "watch"
    return "ready"


def _next_best_command(sections: dict[str, dict[str, Any]]) -> str:
    for key in (
        "single_writer_guard",
        "queue_decay_meter",
        "market_hours_cadence_smoother",
        "global_clear_settlement_guard",
        "paper_collection_floor_smoother",
        "off_hours_drain_plan",
        "stabilization_effectiveness_memory",
    ):
        section = _as_dict(sections.get(key))
        if str(section.get("overall_status") or "") in {"blocked", "critical", "degraded", "needs_work", "watch"}:
            commands = _as_list(section.get("recommended_commands"))
            if commands:
                return " ".join(str(part) for part in _as_list(commands[0]))
    return "./scripts/ops/opsctl.sh platform-settlement-stabilization --json"


def _env_overrides(payload: dict[str, Any]) -> dict[str, str]:
    sections = _as_dict(payload.get("sections"))
    queue = _as_dict(sections.get("queue_decay_meter"))
    queue_metrics = _as_dict(queue.get("metrics"))
    runtime = _as_dict(sections.get("market_hours_cadence_smoother"))
    clear = _as_dict(sections.get("global_clear_settlement_guard"))
    off_hours = _as_dict(sections.get("off_hours_drain_plan"))
    queue_active = _bool(queue.get("queue_backpressure_active"))
    high_runtime = str(runtime.get("overall_status") or "") in {"needs_work", "degraded", "blocked"}
    env = {
        "PLATFORM_SETTLEMENT_STABILIZATION_ENABLED": "1",
        "PLATFORM_SETTLEMENT_STABILIZATION_VERSION": "1",
        "PLATFORM_SETTLEMENT_STABILIZATION_SECTION_COUNT": str(len(SECTION_KEYS)),
        "PLATFORM_SETTLEMENT_NEXT_BEST_COMMAND": str(payload.get("next_best_command") or ""),
        "SETTLEMENT_MODE": "market_hours_queue_settle",
        "SETTLEMENT_QUEUE_BACKPRESSURE_ACTIVE": "1" if queue_active else "0",
        "SETTLEMENT_QUEUE_PENDING_RATIO": str(queue_metrics.get("pending_ratio", "0")),
        "SETTLEMENT_EXPECTED_DRAIN_MINUTES": str(queue_metrics.get("estimated_total_drain_minutes") or "0"),
        "SETTLEMENT_SINGLE_WRITER_REQUIRED": "1",
        "SETTLEMENT_PARALLEL_SQL_WRITERS_ALLOWED": "0",
        "SETTLEMENT_RUNTIME_PRESSURE_ACTIVE": "1" if high_runtime else "0",
        "SETTLEMENT_MARKET_HOURS_HEAVY_DRAIN_ALLOWED": "0",
        "SETTLEMENT_OFF_HOURS_HEAVY_DRAIN_READY": "1" if _bool(_as_dict(off_hours.get("off_hours_window")).get("active")) else "0",
        "GLOBAL_HALT_CLEAR_REQUIRE_SETTLEMENT": "1",
        "GLOBAL_HALT_CLEAR_REQUIRE_QUEUE_SETTLED": "1",
        "GLOBAL_HALT_NOTIFY_ON_SELF_CLEAR": "1",
        "GLOBAL_HALT_SELF_CLEAR_WATCH_ENABLED": "1",
        "PAPER_COLLECTION_FLOOR_SETTLEMENT_ENABLED": "1",
        "PAPER_TRADE_LOCK": "1",
        "ALLOW_ORDER_EXECUTION": "0",
        "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
        "LIBRARY_DEFAULT_ML_BACKEND": "mlx",
        "EXPANSION_APPLY_ALLOWED": "0" if queue_active or high_runtime or _bool(clear.get("halt")) else "1",
    }
    for control in CONTROLS:
        env[control["env_key"]] = "1"
    return env


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/platform_settlement_stabilization.py"]
    for key in sorted(env):
        lines.append(f"{key}={shlex.quote(str(env[key]))}")
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _write_config(path: Path, payload: dict[str, Any]) -> bool:
    config = {
        "schema_version": 1,
        "updated_utc": payload.get("timestamp_utc"),
        "layer": "platform_settlement_stabilization_v1",
        "section_keys": list(SECTION_KEYS),
        "controls": list(CONTROLS),
        "infra_assignments": INFRA_ASSIGNMENTS,
        "artifacts": payload.get("section_artifacts", {}),
    }
    content = json.dumps(config, ensure_ascii=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    previous = _health(project_root, "platform_settlement_stabilization_latest.json")
    queue = _queue_decay_meter(project_root, previous)
    writer = _single_writer_guard(project_root, queue)
    runtime = _market_hours_cadence(project_root)
    clear = _global_clear_guard(project_root, queue)
    paper = _paper_floor(project_root)
    off_hours = _off_hours_plan(project_root)
    memory = _effectiveness_memory(project_root, previous, queue, runtime)
    sections = {
        "queue_decay_meter": queue,
        "single_writer_guard": writer,
        "market_hours_cadence_smoother": runtime,
        "global_clear_settlement_guard": clear,
        "paper_collection_floor_smoother": paper,
        "off_hours_drain_plan": off_hours,
        "stabilization_effectiveness_memory": memory,
    }
    rows = _status_rows(sections)
    overall = _worst_status(rows)
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall in {"ready", "watch", "needs_work", "degraded"},
        "overall_status": overall,
        "layer_name": "Platform Settlement Stabilization v1",
        "mode": "post_expansion_market_hours_settlement",
        "section_count": len(SECTION_KEYS),
        "section_keys": list(SECTION_KEYS),
        "control_count": len(CONTROLS),
        "controls": [{**control, "enabled": True} for control in CONTROLS],
        "sections": sections,
        "section_statuses": rows,
        "infra_assignments": INFRA_ASSIGNMENTS,
        "next_best_command": "",
        "recommended_env_overrides": {},
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "platform-settlement-stabilization", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "global-halt-status", "--json"],
        ],
        "source_files": {
            "primary_artifact": str(DEFAULT_OUT_PATH),
            "ingestion_storage_control": str(project_root / "governance" / "health" / "ingestion_storage_control_latest.json"),
            "runtime_throttle": str(project_root / "governance" / "health" / "runtime_throttle_control_latest.json"),
            "global_halt": str(project_root / "governance" / "health" / "global_halt_auto_clear_latest.json"),
        },
    }
    payload["next_best_command"] = _next_best_command(sections)
    payload["recommended_env_overrides"] = _env_overrides(payload)
    return payload


def write_section_artifacts(project_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    root = project_root / "governance" / "platform_settlement_stabilization"
    sections = _as_dict(payload.get("sections"))
    written: dict[str, str] = {}
    for key in SECTION_KEYS:
        section = _as_dict(sections.get(key))
        if not section:
            continue
        path = root / f"{key}_latest.json"
        write_payload(path, {"timestamp_utc": payload.get("timestamp_utc"), "schema_version": 1, **section})
        written[key] = str(path)
    assignment_path = root / "infrabot_assignments_latest.json"
    write_payload(
        assignment_path,
        {
            "timestamp_utc": payload.get("timestamp_utc"),
            "schema_version": 1,
            "infra_assignments": INFRA_ASSIGNMENTS,
            "contract": "settlement_infrabots_keep_expansion_pressure_and_queue_drain_coordinated",
        },
    )
    written["infrabot_assignments"] = str(assignment_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the post-expansion settlement stabilizer for queue, writer, runtime, and halt-clear pressure.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_file = Path(args.out_file).expanduser()
    payload = build_payload(project_root)
    write_payload(out_file, payload)
    payload["section_artifacts"] = write_section_artifacts(project_root, payload)
    if args.apply:
        env = {str(k): str(v) for k, v in _as_dict(payload.get("recommended_env_overrides")).items()}
        payload["apply_result"] = {
            "applied": True,
            "override_path": str(Path(args.override_file).expanduser()),
            "override_changed": _write_env_override(Path(args.override_file).expanduser(), env),
            "config_path": str(Path(args.config_file).expanduser()),
            "config_changed": _write_config(Path(args.config_file).expanduser(), payload),
        }
    else:
        payload["apply_result"] = {"applied": False}
    write_payload(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "platform_settlement_stabilization "
            f"overall_status={payload.get('overall_status')} "
            f"sections={payload.get('section_count')} "
            f"next_best_command={payload.get('next_best_command', '')}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
