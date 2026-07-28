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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "platform_stabilization_quality_latest.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "platform_stabilization_quality_v1.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.platform_stabilization_quality_override"

SECTION_KEYS: tuple[str, ...] = (
    "backlog_drain_stabilizer",
    "bot_data_quality_governor",
    "duplicate_alpha_compression",
    "paper_trade_realism_v2",
    "provider_cooldown_failover_v2",
    "ready_only_microtraining",
    "expansion_rehearsal_gate",
)

CONTROLS: tuple[dict[str, str], ...] = (
    {"id": "backlog_drain_stabilizer", "title": "Backlog drain stabilizer", "env_key": "PLATFORM_STABILIZER_BACKLOG_DRAIN_ENABLED"},
    {"id": "bot_data_quality_governor", "title": "Bot data quality governor", "env_key": "PLATFORM_STABILIZER_BOT_QUALITY_ENABLED"},
    {"id": "duplicate_alpha_compression", "title": "Duplicate alpha compression", "env_key": "PLATFORM_STABILIZER_DUPLICATE_ALPHA_ENABLED"},
    {"id": "paper_trade_realism_v2", "title": "Paper trade realism v2", "env_key": "PLATFORM_STABILIZER_PAPER_REALISM_ENABLED"},
    {"id": "provider_cooldown_failover_v2", "title": "Provider cooldown failover v2", "env_key": "PLATFORM_STABILIZER_PROVIDER_FAILOVER_ENABLED"},
    {"id": "ready_only_microtraining", "title": "Ready-only microtraining", "env_key": "PLATFORM_STABILIZER_READY_ONLY_TRAINING_ENABLED"},
    {"id": "expansion_rehearsal_gate", "title": "Expansion rehearsal gate", "env_key": "PLATFORM_STABILIZER_EXPANSION_REHEARSAL_ENABLED"},
)

INFRA_ASSIGNMENTS: dict[str, list[str]] = {
    "backlog_drain_stabilizer": ["backpressure_drainer_fleet", "storage_backpressure_autopilot", "writer_cycle_coordinator"],
    "bot_data_quality_governor": ["bot_quality_autopilot", "training_quality_control", "data_collection_observation_rollup"],
    "duplicate_alpha_compression": ["duplicate_alpha_detector", "correlation_governor", "sleeve_master_rollup"],
    "paper_trade_realism_v2": ["execution_lab", "paper_execution_calibration", "paper_trade_lock_infrabot"],
    "provider_cooldown_failover_v2": ["provider_mesh", "source_verification", "collector_contracts"],
    "ready_only_microtraining": ["training_runtime_control", "training_requalification_lane", "retrain_lane_scheduler"],
    "expansion_rehearsal_gate": ["platform_brain_v5", "expansion_capacity_planner", "paper_400_ramp_control"],
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


def _platform_section(platform: dict[str, Any], key: str) -> dict[str, Any]:
    return _as_dict(_as_dict(platform.get("sections")).get(key))


def _brain_section(brain: dict[str, Any], key: str) -> dict[str, Any]:
    return _as_dict(_as_dict(brain.get("sections")).get(key))


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


def _registry_summary(project_root: Path) -> dict[str, Any]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    active = 0
    collecting = 0
    trainable = 0
    excluded = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        active += int(_bool(row.get("active")))
        collecting += int(_bool(row.get("data_collection_active")) or str(row.get("lifecycle_state") or "") == "data_collection_only")
        excluded += int(_bool(row.get("training_excluded")) or _bool(row.get("exclude_from_training")))
        trainable += int(_bool(row.get("data_collection_training_ready")) and not (_bool(row.get("training_excluded")) or _bool(row.get("exclude_from_training"))))
    return {
        "total_bots": len(rows),
        "active_bots": active,
        "collecting_bots": collecting,
        "registry_trainable_bots": trainable,
        "training_excluded_bots": excluded,
    }


def _pending_metrics(project_root: Path) -> dict[str, Any]:
    backpressure = _health(project_root, "ingestion_backpressure_latest.json")
    storage = _health(project_root, "ingestion_storage_control_latest.json")
    storage_bp = _as_dict(storage.get("backpressure"))
    threshold = max(_safe_int(storage_bp.get("pending_lines_threshold"), 15000), 1)
    storage_total = _safe_int(storage_bp.get("total_pending_lines"), 0)
    storage_oldest = _safe_float(storage_bp.get("oldest_pending_age_seconds"), 0.0)
    storage_status = str(storage.get("overall_status") or "").strip().lower()
    storage_severity = str(storage.get("severity") or "").strip().lower()
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    storage_authoritative = bool(
        storage_bp
        and storage_total <= threshold
        and storage_oldest <= 240.0
        and pressure_index < 1.0
        and storage_status in {"ready", "advisory", "ok", "stable", ""}
        and storage_severity in {"ready", "advisory", "ok", "stable", "watch", ""}
    )
    if storage_authoritative:
        core = _safe_int(storage_bp.get("core_pending_lines"), 0)
        deferred = _safe_int(storage_bp.get("deferred_pending_lines"), 0)
        cold = _safe_int(storage_bp.get("cold_pending_lines"), 0)
        support = _safe_int(storage_bp.get("support_pending_lines"), 0)
        total = storage_total
        oldest = storage_oldest
    else:
        core = max(_safe_int(backpressure.get("pending_lines"), 0), _safe_int(storage_bp.get("core_pending_lines"), 0))
        deferred = max(_safe_int(backpressure.get("pending_lines_deferred"), 0), _safe_int(storage_bp.get("deferred_pending_lines"), 0))
        cold = max(_safe_int(backpressure.get("pending_lines_cold"), 0), _safe_int(storage_bp.get("cold_pending_lines"), 0))
        support = max(
            _safe_int(backpressure.get("pending_lines_support_telemetry"), 0),
            _safe_int(storage_bp.get("support_pending_lines"), 0),
        )
        total = max(_safe_int(backpressure.get("pending_lines_total"), 0), storage_total, core + deferred + cold + support)
        oldest = max(_safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0), storage_oldest)
    ratio = total / float(threshold)
    return {
        "core_pending_lines": core,
        "deferred_pending_lines": deferred,
        "cold_pending_lines": cold,
        "support_pending_lines": support,
        "total_pending_lines": total,
        "pending_lines_threshold": threshold,
        "pending_ratio": round(ratio, 6),
        "oldest_pending_age_seconds": round(oldest, 3),
        "severity": str(storage.get("severity") or "unknown"),
        "pressure_index": round(_safe_float(storage.get("pressure_index"), ratio), 6),
        "storage_live_authoritative": storage_authoritative,
        "estimated_total_drain_minutes": storage_bp.get("estimated_total_drain_minutes"),
    }


def _backlog_drain(project_root: Path) -> dict[str, Any]:
    metrics = _pending_metrics(project_root)
    drainer = _health(project_root, "backpressure_drainer_fleet_latest.json")
    total = _safe_int(metrics.get("total_pending_lines"), 0)
    threshold = max(_safe_int(metrics.get("pending_lines_threshold"), 15000), 1)
    active = total >= threshold or str(metrics.get("severity") or "").lower() in {"high", "critical"}
    status = "needs_work" if active else "ready"
    return {
        "overall_status": status,
        "queue_backpressure_active": bool(active),
        "metrics": metrics,
        "active_drainer": _as_dict(drainer.get("active_drainer")),
        "ready_drainer_count": _safe_int(drainer.get("ready_drainer_count"), 0),
        "assigned_infrabots": INFRA_ASSIGNMENTS["backlog_drain_stabilizer"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "backpressure-drainers", "--apply", "--ttl-seconds", "900", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"],
        ],
        "stabilization_contract": [
            "single_sql_writer_remains_authoritative",
            "drainer_fleet_selects_one_focused_lane_at_a_time",
            "core_decision_and_runtime_backlog_clear_before_expansion",
        ],
    }


def _bot_quality(project_root: Path, platform: dict[str, Any]) -> dict[str, Any]:
    section = _platform_section(platform, "bot_data_quality_scores")
    quality = _health(project_root, "bot_quality_autopilot_latest.json")
    training_quality = _health(project_root, "training_quality_control_latest.json")
    rollup = _health(project_root, "data_collection_observation_rollup_latest.json")
    average = _safe_float(section.get("average_quality_score"), _safe_float(training_quality.get("training_quality_score"), 0.0))
    training_score = _safe_float(training_quality.get("training_quality_score"), 0.0)
    zero_obs = _safe_int(rollup.get("zero_observation_count"), 0)
    quality_status = str(quality.get("overall_status") or "missing")
    recoverable_blocked = len(_as_list(training_quality.get("recoverable_blocked_keys")))
    probation_count = len(_as_list(_as_dict(quality.get("quality_blockers")).get("quality_probation_bot_ids")))
    targeted_retrain_count = len(_as_list(_as_dict(quality.get("quality_blockers")).get("targeted_retrain_bot_ids")))
    status = "ready"
    if zero_obs > 0 or recoverable_blocked > 0 or (average < 55.0 and training_score < 75.0):
        status = "needs_work"
    elif quality_status in {"blocked", "degraded", "needs_work"} or average < 70.0 or probation_count > 0 or targeted_retrain_count > 0:
        status = "watch"
    return {
        "overall_status": status,
        "average_quality_score": round(average, 3),
        "training_quality_score": round(training_score, 3),
        "label_counts": _as_dict(section.get("label_counts")),
        "quality_autopilot_status": quality_status,
        "quality_probation_count": probation_count,
        "targeted_retrain_count": targeted_retrain_count,
        "recoverable_blocked_count": recoverable_blocked,
        "collector_count": _safe_int(rollup.get("collector_count"), 0),
        "bots_with_observations": _safe_int(rollup.get("bots_with_observations"), 0),
        "zero_observation_count": zero_obs,
        "assigned_infrabots": INFRA_ASSIGNMENTS["bot_data_quality_governor"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "bot-quality-autopilot", "--apply", "--timeout-sec", "600", "--json"],
            ["./scripts/ops/opsctl.sh", "training-quality", "--json"],
            ["./scripts/ops/opsctl.sh", "data-collection-observation-rollup", "--json"],
        ],
        "quality_contract": [
            "score_useful_labeled_recent_data_not_raw_volume",
            "new_and_cold_start_bots_collect_until_quality_floor",
            "teacher_and_requalification_queues_stay_fresh_before_training",
        ],
    }


def _duplicate_alpha(platform: dict[str, Any]) -> dict[str, Any]:
    section = _platform_section(platform, "duplicate_alpha_overlap_detector")
    overlap = _safe_int(section.get("overlap_cluster_count"), 0)
    high_overlap = _safe_int(section.get("high_overlap_cluster_count"), 0)
    source_status = str(section.get("overall_status") or "missing")
    novelty_contract_raw = section.get("novelty_contract")
    novelty_contract = _as_dict(novelty_contract_raw)
    controlled_by_novelty_contract = bool(
        novelty_contract_raw
        or novelty_contract
        or _bool(section.get("novelty_contract_present"))
        or _bool(section.get("compression_review_active"))
        or _bool(section.get("review_queue_active"))
    )
    source_needs_work = source_status in {"needs_work", "degraded"}
    if overlap <= 0 and not source_needs_work:
        status = "ready"
    elif overlap > 0 and controlled_by_novelty_contract:
        status = "watch"
    else:
        status = "needs_work"
    return {
        "overall_status": status,
        "overlap_cluster_count": overlap,
        "high_overlap_cluster_count": high_overlap,
        "source_status": source_status,
        "controlled_by_novelty_contract": controlled_by_novelty_contract,
        "compression_review_required": bool(overlap > 0),
        "assigned_infrabots": INFRA_ASSIGNMENTS["duplicate_alpha_compression"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "duplicate-alpha-detector", "--json"],
            ["./scripts/ops/opsctl.sh", "correlation-governor", "--json"],
            ["./scripts/ops/opsctl.sh", "sleeve-masters", "--json"],
        ],
        "compression_contract": [
            "do_not_promote_duplicate_alpha_without_novelty_review",
            "merge_or_thin_redundant_collectors_before_expansion",
            "prefer_unique_cross_sleeve_information_value",
        ],
    }


def _paper_realism(project_root: Path, platform: dict[str, Any]) -> dict[str, Any]:
    section = _platform_section(platform, "execution_paper_trade_realism_layer")
    calibration = _health(project_root, "paper_execution_calibration_latest.json")
    execution_lab = _health(project_root, "execution_lab_latest.json")
    metrics = _as_dict(calibration.get("metrics"))
    mae = _safe_float(metrics.get("mae_bps"), _safe_float(section.get("mae_bps"), 0.0))
    p95 = _safe_float(metrics.get("p95_bps"), 0.0)
    poor = _safe_int(metrics.get("poor_or_fair_fill_count"), 0)
    worst_rows = [row for row in _as_list(execution_lab.get("top_worst_case_scenarios")) if isinstance(row, dict)]
    worst_slippage = max([_safe_float(row.get("slippage_bps"), 0.0) for row in worst_rows] or [0.0])
    status = "ready"
    if mae >= 35.0 or worst_slippage >= 60.0:
        status = "needs_work"
    elif p95 >= 50.0 or worst_slippage >= 35.0 or poor > 0:
        status = "watch"
    return {
        "overall_status": status,
        "mae_bps": round(mae, 3),
        "p95_bps": round(p95, 3),
        "poor_or_fair_fill_count": poor,
        "worst_lab_slippage_bps": round(worst_slippage, 3),
        "assigned_infrabots": INFRA_ASSIGNMENTS["paper_trade_realism_v2"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "execution-lab", "--json"],
            ["./scripts/ops/opsctl.sh", "paper-calibration", "--json"],
            ["./scripts/ops/opsctl.sh", "paper-lock", "--json"],
        ],
        "realism_contract": [
            "paper_pnl_is_discounted_by_spread_slippage_latency_and_fill_quality",
            "options_and_intraday_bots_require_execution_friction_before_promotion",
            "paper_trade_lock_stays_required_for_all_sleeves",
        ],
    }


def _provider_failover(project_root: Path, platform: dict[str, Any]) -> dict[str, Any]:
    provider = _platform_section(platform, "provider_rotation_failover_mesh")
    mesh = _health(project_root, "provider_mesh_latest.json")
    source = _health(project_root, "source_verification_latest.json")
    summary = _as_dict(mesh.get("summary"))
    degraded_count = max(_safe_int(provider.get("degraded_provider_count"), 0), _safe_int(summary.get("required_failure_count"), 0) + _safe_int(summary.get("soft_failure_count"), 0))
    mesh_status = str(mesh.get("overall_status") or "missing")
    required_failures = _safe_int(summary.get("required_failure_count"), 0)
    soft_failures = _safe_int(summary.get("soft_failure_count"), 0)
    source_status = str(source.get("overall_status") or provider.get("source_verification_status") or "missing")
    status = "ready"
    if required_failures > 0 or mesh_status in {"blocked", "critical"}:
        status = "needs_work"
    elif soft_failures > 0 or degraded_count > 0 or mesh_status == "degraded" or source_status == "degraded":
        status = "watch"
    return {
        "overall_status": status,
        "degraded_provider_count": degraded_count,
        "provider_mesh_status": mesh_status,
        "required_failure_count": required_failures,
        "soft_failure_count": soft_failures,
        "source_verification_status": source_status,
        "cooldowns": _as_list(mesh.get("cooldowns")),
        "assigned_infrabots": INFRA_ASSIGNMENTS["provider_cooldown_failover_v2"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "provider-mesh", "--json"],
            ["./scripts/ops/opsctl.sh", "source-verification", "--json"],
            ["./scripts/ops/opsctl.sh", "collector-contracts", "--json"],
        ],
        "failover_contract": [
            "provider_403_429_denials_route_to_cooldown_not_global_halt",
            "last_good_cache_and_proxy_context_keep_collection_alive",
            "training_and_promotion_use_source_confidence",
        ],
    }


def _ready_microtraining(project_root: Path, brain_v4: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    runtime = _health(project_root, "training_runtime_control_latest.json")
    bot_needs = _health(project_root, "bot_needs_intelligence_latest.json")
    training = _brain_section(brain_v4, "training_scheduler_brain")
    economist = _brain_section(brain_v4, "bot_portfolio_economist")
    selector = _as_dict(bot_needs.get("training_candidate_selector"))
    readiness_counts = _as_dict(bot_needs.get("training_readiness_counts"))
    train_allowed = max(
        _safe_int(training.get("train_allowed_count"), _safe_int(economist.get("trainable_bots"), registry.get("registry_trainable_bots", 0))),
        _safe_int(selector.get("selected_count"), 0),
        _safe_int(readiness_counts.get("can_train_now"), 0),
    )
    sample_debt = _safe_int(training.get("sample_debt_count"), 0)
    runtime_status = str(runtime.get("overall_status") or "missing")
    launch_blockers = [str(item) for item in _as_list(runtime.get("launch_blockers")) if str(item).strip()]
    budget_closed_managed = bool(launch_blockers) and set(launch_blockers) <= {"autonomic_training_budget_closed"}
    status = "ready" if train_allowed > 0 and runtime_status in {"ready", "degraded", "constrained"} else "needs_work"
    if status != "ready" and budget_closed_managed and bool(runtime.get("snapshot_ready", False)):
        status = "watch"
    return {
        "overall_status": status,
        "train_allowed_count": train_allowed,
        "bot_needs_selected_count": _safe_int(selector.get("selected_count"), 0),
        "bot_needs_can_train_now": _safe_int(readiness_counts.get("can_train_now"), 0),
        "sample_debt_count": sample_debt,
        "training_policy": str(training.get("training_policy") or "off_hours_micro_batches"),
        "runtime_training_status": runtime_status,
        "training_runtime_launch_blockers": launch_blockers,
        "managed_training_budget_closed": budget_closed_managed,
        "snapshot_ready": bool(runtime.get("snapshot_ready", False)),
        "assigned_infrabots": INFRA_ASSIGNMENTS["ready_only_microtraining"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "runtime-training-snapshot", "--json"],
            ["./scripts/ops/opsctl.sh", "training-runtime-control", "--json"],
            ["./scripts/ops/opsctl.sh", "training-requalification", "--write-queue", "--json"],
        ],
        "training_contract": [
            "train_ready_bots_only",
            "cold_start_and_sample_debt_bots_keep_collecting",
            "micro_batches_prefer_off_hours_and_green_pressure",
        ],
    }


def _expansion_gate(project_root: Path, brain_v5: dict[str, Any], backlog: dict[str, Any]) -> dict[str, Any]:
    roadmap = _brain_section(brain_v5, "strategic_roadmap_synthesizer")
    rehearsal = _brain_section(brain_v5, "scenario_rehearsal_lab")
    runtime = _health(project_root, "runtime_throttle_control_latest.json")
    storage = _pending_metrics(project_root)
    settlement = _health(project_root, "platform_settlement_stabilization_latest.json")
    rollup = _health(project_root, "data_collection_observation_rollup_latest.json")
    halt = _health(project_root, "global_halt_auto_clear_latest.json") or _health(project_root, "global_killswitch_latest.json")
    swap = _health(project_root, "swap_pressure_governor_latest.json")
    swap_payload = _as_dict(swap.get("swap_pressure"))

    expansion_allowed = _bool(roadmap.get("expansion_allowed_now"))
    queue_active = _bool(backlog.get("queue_backpressure_active"))
    clear_blockers = [str(item) for item in _as_list(halt.get("clear_blockers"))]
    halt_active = _bool(halt.get("halt")) or _bool(halt.get("global_halt_active"))
    runtime_status = str(runtime.get("overall_status") or "missing").lower()
    host_saturation = _safe_float(runtime.get("host_saturation_score"), 0.0)
    compute_level = str(runtime.get("compute_pressure_level") or "unknown").lower()
    memory_level = str(runtime.get("memory_pressure_level") or "unknown").lower()
    storage_severity = str(storage.get("severity") or "unknown").lower()
    pending_ratio = _safe_float(storage.get("pending_ratio"), 0.0)
    pressure_index = _safe_float(storage.get("pressure_index"), pending_ratio)
    settlement_sections = _as_dict(settlement.get("sections"))
    settlement_queue = _as_dict(settlement_sections.get("queue_decay_meter"))
    settlement_queue_active = _bool(settlement_queue.get("queue_backpressure_active"))
    settlement_status = str(settlement.get("overall_status") or "missing").lower()
    collector_count = max(_safe_int(rollup.get("collector_count"), 0), 0)
    observed_count = max(
        _safe_int(rollup.get("effective_bots_with_observations", rollup.get("bots_with_observations")), 0),
        0,
    )
    zero_observation_count = max(
        _safe_int(rollup.get("unmanaged_zero_observation_count", rollup.get("zero_observation_count")), 0),
        0,
    )
    managed_zero_observation_count = max(_safe_int(rollup.get("managed_zero_observation_count"), 0), 0)
    collection_coverage = (observed_count / float(collector_count)) if collector_count else 1.0
    swap_tier = str(swap_payload.get("tier") or swap.get("tier") or "normal").lower()
    swap_gb = _safe_float(swap_payload.get("swap_used_gb") or swap.get("swap_used_gb"), 0.0)

    gate_closed_reasons = [
        reason
        for reason in (
            "queue_backpressure_active" if queue_active else "",
            "v5_rehearsal_not_cleared" if not expansion_allowed else "",
            "global_halt_active" if halt_active else "",
            "global_clear_blockers_present" if clear_blockers else "",
            "storage_or_queue_not_settled" if storage_severity in {"high", "critical", "blocked"} or pending_ratio >= 1.0 or pressure_index >= 1.0 else "",
            "runtime_not_calm" if runtime_status in {"blocked", "critical"} or host_saturation >= 65.0 or compute_level in {"elevated", "high", "critical"} or memory_level in {"elevated", "high", "critical"} else "",
            "swap_not_calm" if swap_tier in {"constrained", "pause_research", "survival", "critical"} or swap_gb >= 8.0 else "",
            "collection_floor_not_clean" if collection_coverage < 0.99 or zero_observation_count > 0 else "",
            "settlement_stabilizer_not_clear" if settlement_status in {"blocked", "critical", "degraded", "needs_work"} or settlement_queue_active else "",
        )
        if reason
    ]
    gate_closed = bool(gate_closed_reasons)
    repair_reasons = {
        "queue_backpressure_active",
        "storage_or_queue_not_settled",
        "runtime_not_calm",
        "swap_not_calm",
        "collection_floor_not_clean",
        "settlement_stabilizer_not_clear",
    }
    needs_repair = bool(repair_reasons.intersection(gate_closed_reasons))
    status = "ready"
    if gate_closed:
        if halt_active or runtime_status in {"blocked", "critical"}:
            status = "blocked"
        elif needs_repair:
            status = "needs_work"
        else:
            status = "watch"
    return {
        "overall_status": status,
        "expansion_allowed_now": bool(not gate_closed),
        "gate_closed_reasons": gate_closed_reasons,
        "repair_required_reasons": [reason for reason in gate_closed_reasons if reason in repair_reasons],
        "scenario_count": _safe_int(rehearsal.get("scenario_count"), 0),
        "scenarios": _as_list(rehearsal.get("scenarios")),
        "pre_expansion_snapshot": {
            "runtime_status": runtime_status,
            "host_saturation_score": round(host_saturation, 3),
            "compute_pressure_level": compute_level,
            "memory_pressure_level": memory_level,
            "storage_severity": storage_severity,
            "pending_ratio": round(pending_ratio, 6),
            "pressure_index": round(pressure_index, 6),
            "settlement_status": settlement_status,
            "settlement_queue_backpressure_active": settlement_queue_active,
            "halt_active": halt_active,
            "clear_blockers": clear_blockers,
            "collection_coverage_ratio": round(collection_coverage, 6),
            "zero_observation_count": zero_observation_count,
            "managed_zero_observation_count": managed_zero_observation_count,
            "swap_tier": swap_tier,
            "swap_used_gb": round(swap_gb, 3),
        },
        "calm_requirements": {
            "pending_ratio_under": 1.0,
            "storage_pressure_index_under": 1.0,
            "host_saturation_under": 65.0,
            "collection_coverage_at_least": 0.99,
            "zero_observation_count": 0,
            "global_clear_blockers": 0,
        },
        "assigned_infrabots": INFRA_ASSIGNMENTS["expansion_rehearsal_gate"],
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "platform-settlement-stabilization", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "runtime-throttle", "--apply", "--max-renice-processes", "6", "--json"],
            ["./scripts/ops/opsctl.sh", "backpressure-drainers", "--apply", "--ttl-seconds", "900", "--json"],
            ["./scripts/ops/opsctl.sh", "expansion-capacity", "--json"],
        ],
        "gate_contract": [
            "all_future_expansions_rehearse_before_apply",
            "new_bots_default_to_collect_only_and_training_excluded",
            "expansion_waits_for_backpressure_provider_and_quality_gates",
            "expansion_requires_true_calm_not_just_no_global_halt",
            "queue_storage_runtime_and_collection_floor_must_be_green_before_new_waves",
        ],
    }


def _env_overrides(payload: dict[str, Any]) -> dict[str, str]:
    sections = _as_dict(payload.get("sections"))
    backlog = _as_dict(sections.get("backlog_drain_stabilizer"))
    expansion = _as_dict(sections.get("expansion_rehearsal_gate"))
    next_best = str(payload.get("next_best_command") or "")
    env = {
        "PLATFORM_STABILIZATION_QUALITY_ENABLED": "1",
        "PLATFORM_STABILIZATION_QUALITY_VERSION": "1",
        "PLATFORM_STABILIZATION_QUALITY_SECTION_COUNT": str(len(SECTION_KEYS)),
        "PLATFORM_STABILIZATION_NEXT_BEST_COMMAND": next_best,
        "QUEUE_BACKPRESSURE_AUTODRAIN_ENABLED": "1",
        "STABILITY_HARDENING_V2_ENABLED": "1",
        "HEALTH_ARTIFACT_COALESCE_ENABLED": "1",
        "HEALTH_ARTIFACT_MIN_WRITE_SECONDS": "35",
        "REPORT_REFRESH_DEBOUNCE_ENABLED": "1",
        "REPORT_REFRESH_DEBOUNCE_SECONDS": "1500",
        "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
        "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.82",
        "PAPER_TRADE_EVENT_QUEUE_JITTER_ENABLED": "1",
        "PAPER_TRADE_EVENT_QUEUE_JITTER_SECONDS": "11",
        "PROVIDER_FAILURE_DAMPER_ENABLED": "1",
        "TRAINING_RESEARCH_CIRCUIT_BREAKER_ENABLED": "1",
        "COLD_START_COLLECTOR_THIN_SAMPLE_ENABLED": "1",
        "OPS_LAUNCHD_STAGGER_ENABLED": "1",
        "BACKPRESSURE_DRAINER_FLEET_AUTOPILOT_ENABLED": "1",
        "BACKPRESSURE_DRAINER_TTL_SECONDS": "900",
        "BACKPRESSURE_DRAINER_MODE": "single_writer_focused_handoff",
        "QUEUE_BACKPRESSURE_ACTIVE": "1" if _bool(backlog.get("queue_backpressure_active")) else "0",
        "BOT_DATA_QUALITY_GOVERNOR_ENABLED": "1",
        "BOT_DATA_QUALITY_PROMOTION_FLOOR": "55",
        "BOT_QUALITY_AUTOPILOT_STANDARD_RULE": "1",
        "DUPLICATE_ALPHA_COMPRESSION_ENABLED": "1",
        "DUPLICATE_ALPHA_PROMOTION_REVIEW_REQUIRED": "1",
        "PAPER_EXECUTION_REALISM_V2_ENABLED": "1",
        "PAPER_REALISM_REQUIRE_SPREAD_SLIPPAGE_LATENCY": "1",
        "PROVIDER_COOLDOWN_FAILOVER_V2_ENABLED": "1",
        "PROVIDER_403_429_DEGRADES_NOT_HALTS": "1",
        "PROVIDER_LAST_GOOD_CACHE_REQUIRED": "1",
        "TRAINING_READY_ONLY_MICROBATCH_ENABLED": "1",
        "TRAINING_REQUIRE_BACKPRESSURE_CLEAR": "1",
        "TRAINING_REQUIRE_RESOURCE_GUARD_GREEN": "1",
        "TRAINING_MAX_CONCURRENT_MICROBATCHES": "1",
        "EXPANSION_REHEARSAL_REQUIRED": "1",
        "EXPANSION_CALM_GATE_ENABLED": "1",
        "EXPANSION_PRECHECK_REQUIRED": "1",
        "EXPANSION_REQUIRE_QUEUE_SETTLED": "1",
        "EXPANSION_REQUIRE_STORAGE_GREEN": "1",
        "EXPANSION_REQUIRE_RUNTIME_CALM": "1",
        "EXPANSION_REQUIRE_COLLECTION_FLOOR": "1",
        "EXPANSION_REQUIRE_GLOBAL_CLEAR": "1",
        "EXPANSION_REQUIRE_SETTLEMENT_CLEAR": "1",
        "EXPANSION_REQUIRE_PENDING_RATIO_UNDER": "1.0",
        "EXPANSION_REQUIRE_HOST_SATURATION_UNDER": "65",
        "EXPANSION_APPLY_ALLOWED": "1" if _bool(expansion.get("expansion_allowed_now")) else "0",
        "EXPANSION_CALM_BLOCKERS": ",".join(str(item) for item in _as_list(expansion.get("gate_closed_reasons"))) or "none",
        "ROSTER_EXPANSION_ALLOWED": "1" if _bool(expansion.get("expansion_allowed_now")) else "0",
        "BOT_ADMISSION_EXPANSION_ALLOWED": "1" if _bool(expansion.get("expansion_allowed_now")) else "0",
        "NEW_BOT_DEFAULT_COLLECTION_ONLY": "1",
        "NEW_BOT_DEFAULT_TRAINING_EXCLUDED": "1",
        "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
        "LIBRARY_DEFAULT_ML_BACKEND": "mlx",
        "PAPER_TRADE_LOCK": "1",
        "ALLOW_ORDER_EXECUTION": "0",
    }
    for control in CONTROLS:
        env[control["env_key"]] = "1"
    return env


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/platform_stabilization_quality.py"]
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
        "layer": "platform_stabilization_quality_v1",
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


def _next_best_command(sections: dict[str, dict[str, Any]]) -> str:
    order = (
        "backlog_drain_stabilizer",
        "provider_cooldown_failover_v2",
        "bot_data_quality_governor",
        "duplicate_alpha_compression",
        "paper_trade_realism_v2",
        "ready_only_microtraining",
        "expansion_rehearsal_gate",
    )
    for key in order:
        section = _as_dict(sections.get(key))
        if str(section.get("overall_status") or "") in {"blocked", "critical", "degraded", "needs_work", "watch"}:
            commands = _as_list(section.get("recommended_commands"))
            if commands:
                return " ".join(str(part) for part in _as_list(commands[0]))
    return "./scripts/ops/opsctl.sh platform-stabilization --json"


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    platform = _health(project_root, "platform_intelligence_expansion_latest.json")
    brain_v4 = _health(project_root, "platform_brain_v4_latest.json")
    brain_v5 = _health(project_root, "platform_brain_v5_latest.json")
    registry = _registry_summary(project_root)
    backlog = _backlog_drain(project_root)
    sections = {
        "backlog_drain_stabilizer": backlog,
        "bot_data_quality_governor": _bot_quality(project_root, platform),
        "duplicate_alpha_compression": _duplicate_alpha(platform),
        "paper_trade_realism_v2": _paper_realism(project_root, platform),
        "provider_cooldown_failover_v2": _provider_failover(project_root, platform),
        "ready_only_microtraining": _ready_microtraining(project_root, brain_v4, registry),
        "expansion_rehearsal_gate": _expansion_gate(project_root, brain_v5, backlog),
    }
    rows = _status_rows(sections)
    overall = _worst_status(rows)
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall in {"ready", "watch", "needs_work", "degraded"},
        "overall_status": overall,
        "layer_name": "Platform Stabilization And Quality v1",
        "mode": "guarded_stabilization_quality_before_expansion",
        "section_count": len(SECTION_KEYS),
        "section_keys": list(SECTION_KEYS),
        "control_count": len(CONTROLS),
        "controls": [{**control, "enabled": True} for control in CONTROLS],
        "registry_summary": registry,
        "platform_intelligence_status": platform.get("overall_status", "missing"),
        "platform_brain_v5_status": brain_v5.get("overall_status", "missing"),
        "sections": sections,
        "section_statuses": rows,
        "infra_assignments": INFRA_ASSIGNMENTS,
        "next_best_command": "",
        "recommended_env_overrides": {},
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "platform-stabilization", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "backpressure-drainers", "--apply", "--ttl-seconds", "900", "--json"],
            ["./scripts/ops/opsctl.sh", "provider-mesh", "--json"],
            ["./scripts/ops/opsctl.sh", "bot-quality-autopilot", "--apply", "--timeout-sec", "600", "--json"],
            ["./scripts/ops/opsctl.sh", "platform-brain-v5", "--json"],
        ],
        "source_files": {
            "platform_intelligence": str(project_root / "governance" / "health" / "platform_intelligence_expansion_latest.json"),
            "platform_brain_v5": str(project_root / "governance" / "health" / "platform_brain_v5_latest.json"),
            "primary_artifact": str(DEFAULT_OUT_PATH),
        },
    }
    payload["next_best_command"] = _next_best_command(sections)
    payload["recommended_env_overrides"] = _env_overrides(payload)
    return payload


def write_section_artifacts(project_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    root = project_root / "governance" / "platform_stabilization_quality"
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
            "contract": "stabilization_infrabots_coordinate_existing_ops_surfaces_without_live_execution",
        },
    )
    written["infrabot_assignments"] = str(assignment_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the seven-part stabilization and quality layer before further expansion.")
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
            "platform_stabilization_quality "
            f"overall_status={payload.get('overall_status')} "
            f"sections={payload.get('section_count')} "
            f"next_best_command={payload.get('next_best_command', '')}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
