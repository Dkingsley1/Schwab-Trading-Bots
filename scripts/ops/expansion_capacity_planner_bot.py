#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "expansion_capacity_planner_latest.json"

SUPPORT_INFRABOT_IDS = [
    "brain_refinery_v674_expansion_capacity_planner_bot",
    "brain_refinery_v675_expansion_dependency_graph_guard_bot",
    "brain_refinery_v676_expansion_storage_budget_forecaster_bot",
    "brain_refinery_v677_expansion_training_maturity_gate_bot",
    "brain_refinery_v678_expansion_runtime_isolation_guard_bot",
]


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    if value != value or value in {float("inf"), float("-inf")}:
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
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "active", "enabled", "blocked"}


def _status(raw: Any, default: str = "missing") -> str:
    text = str(raw or "").strip().lower()
    return text or default


def _registry_rows(project_root: Path) -> list[dict[str, Any]]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _bot_version(bot_id: str) -> int:
    match = re.search(r"_v(\d+)", str(bot_id or ""))
    return int(match.group(1)) if match else 0


def _infer_sleeve(row: dict[str, Any]) -> str:
    for key in ("sleeve_profile", "sleeve", "profile", "strategy_sleeve", "slot_kind"):
        value = str(row.get(key) or "").strip().lower()
        if value:
            return value
    text = " ".join(
        str(part or "").lower()
        for part in [
            row.get("bot_id"),
            row.get("bot_role"),
            " ".join(str(item) for item in list(row.get("target_functions") or [])),
            " ".join(str(item) for item in list(row.get("data_intake_collections") or [])),
        ]
    )
    for token in (
        "options_on_futures",
        "intraday_aggressive",
        "day_trading",
        "dividend",
        "conservative",
        "futures",
        "options",
        "crypto",
        "fx",
        "bond",
        "macro",
        "liquidity",
        "execution",
        "feature_quality",
        "system_governor",
    ):
        if token in text:
            return token
    return "default"


def _pressure_snapshot(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    platform_root = project_root / "governance" / "platform_intelligence"
    swap = load_json(health_root / "swap_pressure_governor_latest.json")
    runtime = load_json(health_root / "runtime_throttle_control_latest.json")
    memory = load_json(health_root / "memory_efficiency_control_latest.json")
    storage = load_json(health_root / "ingestion_storage_control_latest.json")
    backpressure = load_json(health_root / "ingestion_backpressure_latest.json")
    data_storage = load_json(health_root / "data_collection_storage_guard_latest.json")
    provider_mesh = load_json(health_root / "provider_mesh_latest.json")
    source_verification = load_json(health_root / "source_verification_latest.json")
    data_rollup = load_json(health_root / "data_collection_observation_rollup_latest.json")
    halt = load_json(health_root / "global_killswitch_latest.json")
    halt_auto_clear = load_json(health_root / "global_halt_auto_clear_latest.json")
    admission = load_json(health_root / "new_bot_admission_guard_latest.json")
    dashboard = load_json(health_root / "runtime_gate_dashboard_latest.json")
    platform_capacity = load_json(platform_root / "capacity_planner_latest.json")
    platform_stabilization = load_json(health_root / "platform_stabilization_quality_latest.json")
    settlement = load_json(health_root / "platform_settlement_stabilization_latest.json")

    swap_pressure = swap.get("swap_pressure") if isinstance(swap.get("swap_pressure"), dict) else {}
    swap_tier = _status(swap_pressure.get("tier") or swap.get("tier"), "normal")
    swap_gb = _safe_float(swap_pressure.get("swap_used_gb") or swap.get("swap_used_gb"), 0.0)
    runtime_status = _status(runtime.get("overall_status"))
    memory_status = _status(memory.get("overall_status"))
    storage_status = _status(storage.get("overall_status"))
    data_storage_status = _status(data_storage.get("overall_status"), storage_status)
    dashboard_status = _status(dashboard.get("overall_status"), "missing")
    platform_capacity_status = _status(platform_capacity.get("overall_status"), "missing")
    storage_pressure = _safe_float(storage.get("pressure_index"), 0.0)
    host_saturation = _safe_float(runtime.get("host_saturation_score"), 0.0)
    compute_level = _status(runtime.get("compute_pressure_level"), "unknown")
    memory_pressure_level = _status(runtime.get("memory_pressure_level"), "unknown")
    storage_backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    backlog_truth = storage.get("backlog_truth") if isinstance(storage.get("backlog_truth"), dict) else {}
    raw_truth = backlog_truth.get("raw_live") if isinstance(backlog_truth.get("raw_live"), dict) else {}
    overlay_truth = backlog_truth.get("sql_overlay") if isinstance(backlog_truth.get("sql_overlay"), dict) else {}
    raw_live_expansion = (
        storage.get("raw_live_expansion_contract")
        if isinstance(storage.get("raw_live_expansion_contract"), dict)
        else {}
    )
    storage_efficiency_contract = (
        storage.get("storage_efficiency_contract")
        if isinstance(storage.get("storage_efficiency_contract"), dict)
        else {}
    )
    storage_efficiency_env = (
        storage_efficiency_contract.get("control_env_recommendations")
        if isinstance(storage_efficiency_contract.get("control_env_recommendations"), dict)
        else {}
    )
    total_pending_lines = max(
        _safe_int(storage_backpressure.get("total_pending_lines"), 0),
        _safe_int(backpressure.get("pending_lines_total"), 0),
        _safe_int(backpressure.get("pending_lines"), 0),
    )
    pending_threshold = max(_safe_int(storage_backpressure.get("pending_lines_threshold"), 15000), 1)
    pending_ratio = total_pending_lines / float(pending_threshold)
    queue_backpressure_active = total_pending_lines >= pending_threshold or _status(storage.get("severity"), "unknown") in {"high", "critical", "blocked"}
    admission_blocking = _safe_int(admission.get("blocking_candidate_count"), 0)
    admission_candidates = _safe_int(admission.get("candidate_bot_count"), 0)
    global_halt_active = any(
        _bool(halt.get(key))
        for key in ("global_halt_active", "halt_active", "hard_halt_active", "blocked", "killswitch_active")
    ) or _bool(halt_auto_clear.get("halt"))
    clear_blockers = halt_auto_clear.get("clear_blockers") if isinstance(halt_auto_clear.get("clear_blockers"), list) else []
    stabilization_sections = platform_stabilization.get("sections") if isinstance(platform_stabilization.get("sections"), dict) else {}
    stabilization_gate = stabilization_sections.get("expansion_rehearsal_gate") if isinstance(stabilization_sections.get("expansion_rehearsal_gate"), dict) else {}
    stabilization_says_no = bool(stabilization_gate) and not _bool(stabilization_gate.get("expansion_allowed_now"))
    settlement_sections = settlement.get("sections") if isinstance(settlement.get("sections"), dict) else {}
    settlement_queue = settlement_sections.get("queue_decay_meter") if isinstance(settlement_sections.get("queue_decay_meter"), dict) else {}
    settlement_queue_active = _bool(settlement_queue.get("queue_backpressure_active"))
    raw_live_expansion_ready = bool(raw_live_expansion.get("expansion_ready", False))
    raw_live_expansion_active = bool(raw_live_expansion.get("active", False))
    raw_live_headroom = raw_live_expansion.get("estimated_expansion_headroom") if isinstance(raw_live_expansion.get("estimated_expansion_headroom"), dict) else {}
    raw_live_estimated_bot_headroom = _safe_int(raw_live_headroom.get("estimated_new_bot_headroom"), 0)
    overlay_total_pending_lines = _safe_int(overlay_truth.get("total_pending_lines"), 0)
    overlay_core_pending_lines = _safe_int(overlay_truth.get("core_pending_lines"), 0)
    storage_allow_expansion_raw = storage_efficiency_env.get("BOT_STORAGE_ALLOW_EXPANSION")
    storage_allow_expansion = True if storage_allow_expansion_raw is None else _bool(storage_allow_expansion_raw)
    source_overall = source_verification.get("overall") if isinstance(source_verification.get("overall"), dict) else {}
    source_counts = source_overall.get("counts") if isinstance(source_overall.get("counts"), dict) else {}
    source_unverified = source_overall.get("unverified_sources") if isinstance(source_overall.get("unverified_sources"), list) else []
    provider_summary = provider_mesh.get("summary") if isinstance(provider_mesh.get("summary"), dict) else {}

    blocking_reasons: list[str] = []
    if global_halt_active:
        blocking_reasons.append("global_halt_active")
    if swap_tier in {"survival", "critical"} or swap_gb >= 20.0:
        blocking_reasons.append("swap_pressure_too_high_for_new_runtime")
    if runtime_status in {"blocked", "critical"} or host_saturation >= 85.0 or compute_level in {"high", "critical"}:
        blocking_reasons.append("runtime_pressure_too_high")
    if storage_status in {"blocked", "critical"} or data_storage_status in {"blocked", "critical"}:
        blocking_reasons.append("storage_pressure_too_high")
    if queue_backpressure_active:
        blocking_reasons.append("queue_backpressure_active")
    if raw_live_expansion and (raw_live_expansion_active or not raw_live_expansion_ready):
        blocking_reasons.append("raw_live_expansion_headroom_not_ready")
    if storage_efficiency_contract and not storage_allow_expansion:
        blocking_reasons.append("storage_efficiency_expansion_locked")
    if clear_blockers:
        blocking_reasons.append("global_clear_blockers_present")
    if stabilization_says_no:
        blocking_reasons.append("pre_expansion_stabilization_gate_closed")
    if settlement_queue_active:
        blocking_reasons.append("settlement_queue_backpressure_active")

    watch_reasons: list[str] = []
    if memory_status in {"needs_work", "degraded", "blocked", "critical"}:
        watch_reasons.append("memory_governor_needs_attention")
    if swap_tier in {"calm", "constrained", "pause_research"} or swap_gb >= 12.0:
        watch_reasons.append("swap_pressure_elevated")
    if storage_pressure >= 0.40:
        watch_reasons.append("storage_pressure_index_elevated")
    if raw_live_expansion and _safe_float(raw_live_expansion.get("pressure_ratio"), 0.0) >= 0.70:
        watch_reasons.append("raw_live_expansion_headroom_warm")
    if overlay_total_pending_lines >= 50_000:
        watch_reasons.append("sql_overlay_backlog_elevated")
    if pending_ratio >= 0.50:
        watch_reasons.append("queue_pending_ratio_elevated")
    if host_saturation >= 65.0 or compute_level == "elevated" or memory_pressure_level == "elevated":
        watch_reasons.append("runtime_not_calm_enough_for_expansion")
    if admission_blocking > 0:
        watch_reasons.append("new_bot_admission_contracts_not_clear")
    if dashboard_status in {"warn", "needs_work", "degraded", "blocked", "critical"}:
        watch_reasons.append("runtime_gate_dashboard_has_attention_items")
    if _status(provider_mesh.get("overall_status")) in {"missing", "needs_work", "degraded", "blocked", "critical"}:
        watch_reasons.append("provider_mesh_not_clean")
    if _status(source_verification.get("overall_status")) in {"missing", "needs_work", "degraded", "blocked", "critical"} or source_unverified:
        watch_reasons.append("source_verification_not_clean")

    if blocking_reasons:
        overall_status = "blocked"
    elif watch_reasons:
        overall_status = "needs_work"
    else:
        overall_status = "ready"

    return {
        "overall_status": overall_status,
        "blocking_reasons": blocking_reasons,
        "watch_reasons": watch_reasons,
        "swap_tier": swap_tier,
        "swap_used_gb": round(swap_gb, 3),
        "runtime_status": runtime_status,
        "memory_status": memory_status,
        "storage_status": storage_status,
        "data_collection_storage_status": data_storage_status,
        "runtime_gate_dashboard_status": dashboard_status,
        "platform_capacity_status": platform_capacity_status,
        "storage_pressure_index": round(storage_pressure, 6),
        "host_saturation_score": round(host_saturation, 3),
        "compute_pressure_level": compute_level,
        "memory_pressure_level": memory_pressure_level,
        "queue_backpressure_active": queue_backpressure_active,
        "total_pending_lines": total_pending_lines,
        "pending_lines_threshold": pending_threshold,
        "pending_ratio": round(pending_ratio, 6),
        "raw_live_expansion": {
            "present": bool(raw_live_expansion),
            "ready": bool(raw_live_expansion_ready),
            "active": bool(raw_live_expansion_active),
            "grade": str(raw_live_expansion.get("grade") or ""),
            "expansion_tier": str(raw_live_expansion.get("expansion_tier") or ""),
            "pressure_ratio": _safe_float(raw_live_expansion.get("pressure_ratio"), 0.0),
            "estimated_new_bot_headroom": int(raw_live_estimated_bot_headroom),
            "raw_core_pending_lines": _safe_int((raw_live_expansion.get("raw_live") or {}).get("core_pending_lines"), 0)
            if isinstance(raw_live_expansion.get("raw_live"), dict)
            else 0,
            "raw_total_pending_lines": _safe_int((raw_live_expansion.get("raw_live") or {}).get("total_pending_lines"), 0)
            if isinstance(raw_live_expansion.get("raw_live"), dict)
            else 0,
        },
        "backlog_truth": {
            "raw_live_grade": str(raw_truth.get("grade") or ""),
            "raw_live_pressure_ratio": _safe_float(raw_truth.get("pressure_ratio"), 0.0),
            "sql_overlay_grade": str(overlay_truth.get("grade") or ""),
            "sql_overlay_pressure_ratio": _safe_float(overlay_truth.get("pressure_ratio"), 0.0),
            "sql_overlay_total_pending_lines": int(overlay_total_pending_lines),
            "sql_overlay_core_pending_lines": int(overlay_core_pending_lines),
            "sql_overlay_used_for_pressure": bool(overlay_truth.get("used_for_pressure", False)),
        },
        "storage_efficiency": {
            "active": bool(storage_efficiency_contract.get("active", False)),
            "storage_plane_phase": str(storage_efficiency_env.get("BOT_STORAGE_PLANE_PHASE") or ""),
            "storage_allow_expansion": bool(storage_allow_expansion),
            "storage_allow_training": _bool(storage_efficiency_env.get("BOT_STORAGE_ALLOW_TRAINING")),
            "storage_space_recovery_required": _bool(storage_efficiency_env.get("BOT_STORAGE_SPACE_RECOVERY_REQUIRED")),
            "storage_space_recovery_deficit_gb": _safe_float(
                storage_efficiency_env.get("BOT_STORAGE_SPACE_RECOVERY_DEFICIT_GB"),
                0.0,
            ),
            "storage_external_free_gb": _safe_float(storage_efficiency_env.get("BOT_STORAGE_EXTERNAL_FREE_GB"), 0.0),
            "storage_external_min_free_gb": _safe_float(storage_efficiency_env.get("BOT_STORAGE_EXTERNAL_MIN_FREE_GB"), 0.0),
            "data_capture_mode": str(storage_efficiency_env.get("BOT_DATA_CAPTURE_MODE") or ""),
            "raw_payload_storage_mode": str(storage_efficiency_env.get("BOT_RAW_PAYLOAD_STORAGE_MODE") or ""),
        },
        "provider_quality": {
            "provider_mesh_status": _status(provider_mesh.get("overall_status")),
            "provider_required_failure_count": _safe_int(provider_summary.get("required_failure_count"), 0),
            "provider_soft_failure_count": _safe_int(provider_summary.get("soft_failure_count"), 0),
            "source_verification_status": _status(source_verification.get("overall_status")),
            "source_all_verified": bool(source_overall.get("all_verified", False)),
            "source_unverified_count": len(source_unverified),
            "source_single_unverified_count": _safe_int(source_counts.get("single_source_unverified"), 0),
            "data_quality_score": _safe_float(data_rollup.get("data_quality_score"), 0.0),
        },
        "global_halt_active": global_halt_active,
        "global_clear_blockers": [str(item) for item in clear_blockers],
        "pre_expansion_stabilization_gate_closed": stabilization_says_no,
        "settlement_queue_backpressure_active": settlement_queue_active,
        "admission_candidate_count": admission_candidates,
        "admission_blocking_candidate_count": admission_blocking,
        "source_files": {
            "swap_pressure_governor": str(health_root / "swap_pressure_governor_latest.json"),
            "runtime_throttle_control": str(health_root / "runtime_throttle_control_latest.json"),
            "memory_efficiency_control": str(health_root / "memory_efficiency_control_latest.json"),
            "ingestion_storage_control": str(health_root / "ingestion_storage_control_latest.json"),
            "data_collection_storage_guard": str(health_root / "data_collection_storage_guard_latest.json"),
            "provider_mesh": str(health_root / "provider_mesh_latest.json"),
            "source_verification": str(health_root / "source_verification_latest.json"),
            "data_collection_observation_rollup": str(health_root / "data_collection_observation_rollup_latest.json"),
            "global_killswitch": str(health_root / "global_killswitch_latest.json"),
            "new_bot_admission_guard": str(health_root / "new_bot_admission_guard_latest.json"),
            "runtime_gate_dashboard": str(health_root / "runtime_gate_dashboard_latest.json"),
            "platform_capacity_planner": str(platform_root / "capacity_planner_latest.json"),
            "platform_stabilization_quality": str(health_root / "platform_stabilization_quality_latest.json"),
            "platform_settlement_stabilization": str(health_root / "platform_settlement_stabilization_latest.json"),
        },
    }


def _capacity_contract(
    rows: list[dict[str, Any]],
    pressure: dict[str, Any],
    *,
    requested_wave_size: int,
    clean_scaling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_bots = len(rows)
    active_bots = sum(1 for row in rows if bool(row.get("active", False)))
    collection_only = sum(1 for row in rows if str(row.get("lifecycle_state") or "").strip().lower() == "data_collection_only")
    training_excluded = sum(1 for row in rows if bool(row.get("training_excluded", False)) or bool(row.get("exclude_from_training", False)))
    max_version = max([_bot_version(str(row.get("bot_id") or "")) for row in rows] or [0])
    sleeve_counts = Counter(_infer_sleeve(row) for row in rows)

    max_new_collectors = 30
    if total_bots >= 900:
        max_new_collectors = 10
    elif total_bots >= 700:
        max_new_collectors = 18
    elif total_bots >= 500:
        max_new_collectors = 24

    if collection_only >= 550:
        max_new_collectors = min(max_new_collectors, 16)
    elif collection_only >= 450:
        max_new_collectors = min(max_new_collectors, 22)

    if pressure["overall_status"] == "blocked":
        max_new_collectors = 0
    elif pressure["overall_status"] == "needs_work":
        max_new_collectors = min(max_new_collectors, 20)

    if pressure["swap_used_gb"] >= 20.0:
        max_new_collectors = min(max_new_collectors, 12)
    elif pressure["swap_used_gb"] >= 16.0:
        max_new_collectors = min(max_new_collectors, 20)

    if pressure["global_halt_active"]:
        max_new_collectors = 0
    clean = clean_scaling if isinstance(clean_scaling, dict) else {}
    if clean:
        max_new_collectors = min(
            max_new_collectors,
            max(_safe_int(clean.get("max_clean_wave_size_now"), max_new_collectors), 0),
        )

    requested = max(int(requested_wave_size), 0)
    recommended = min(requested, max_new_collectors)
    if max_new_collectors <= 0:
        rollout_mode = (
            "blocked_clean_scaling_no_new_runtime_loops"
            if str(clean.get("mode") or "") == "blocked_clean_scaling"
            else "protect_live_no_new_runtime_loops"
        )
    elif recommended < requested:
        rollout_mode = (
            "micro_collection_only_soak"
            if str(clean.get("mode") or "") == "micro_collection_only_soak"
            else "trickle_collection_only_wave"
        )
    else:
        rollout_mode = "collection_only_wave_allowed"

    training_policy = "training_locked_until_admission_and_minimum_observation_contracts_clear"
    if pressure["admission_blocking_candidate_count"] <= 0 and pressure["overall_status"] == "ready":
        training_policy = "training_still_locked_for_new_bots_until_minimum_collection_thresholds"

    sleeve_growth_budget = []
    for sleeve, count in sleeve_counts.most_common():
        if sleeve in {"system_governor_expansion", "feature_quality_data_confidence", "liquidity_regime", "transaction_cost_slippage_intelligence", "provider_adapter_verification"}:
            mode = "support_growth"
        elif count >= 45:
            mode = "thin_sample_only_until_backlog_drops"
        else:
            mode = "normal_collection"
        sleeve_growth_budget.append({"sleeve": sleeve, "registered_bots": count, "recommended_collection_mode": mode})

    status = "ready"
    if max_new_collectors <= 0:
        status = "blocked"
    elif recommended < requested:
        status = "needs_work"

    return {
        "overall_status": status,
        "requested_wave_size": requested,
        "max_new_collectors_now": max_new_collectors,
        "recommended_wave_size_now": recommended,
        "rollout_mode": rollout_mode,
        "training_policy": training_policy,
        "paper_trade_policy": "paper_locked_zero_weight_until_graduation",
        "live_trade_policy": "live_execution_disabled_for_new_expansion_slots",
        "next_bot_id_range": {
            "start": f"brain_refinery_v{max_version + 1}",
            "end_if_full_requested_wave_added": f"brain_refinery_v{max_version + requested}",
            "end_if_recommended_wave_added": f"brain_refinery_v{max_version + recommended}" if recommended else "",
        },
        "fleet_counts": {
            "total_bots": total_bots,
            "active_bots": active_bots,
            "data_collection_only_bots": collection_only,
            "training_excluded_bots": training_excluded,
            "max_bot_version": max_version,
            "sleeve_count": len(sleeve_counts),
        },
        "sleeve_growth_budget": sleeve_growth_budget[:40],
    }


def _grade_from_score(score: float) -> str:
    value = max(0.0, min(float(score), 100.0))
    if value >= 99.0:
        return "A+"
    if value >= 97.0:
        return "A+"
    if value >= 93.0:
        return "A"
    if value >= 85.0:
        return "B"
    if value >= 75.0:
        return "C"
    if value >= 65.0:
        return "D"
    return "F"


def _dimension(name: str, *, status: str, evidence: dict[str, Any], next_action: str) -> dict[str, Any]:
    clean_status = str(status or "blocked").strip().lower()
    if clean_status not in {"ready", "watch", "blocked"}:
        clean_status = "blocked"
    return {
        "name": name,
        "status": clean_status,
        "evidence": evidence,
        "next_action": next_action if clean_status != "ready" else "monitor",
    }


def _clean_scaling_contract(
    rows: list[dict[str, Any]],
    pressure: dict[str, Any],
    *,
    requested_wave_size: int,
) -> dict[str, Any]:
    raw_live = pressure.get("raw_live_expansion") if isinstance(pressure.get("raw_live_expansion"), dict) else {}
    truth = pressure.get("backlog_truth") if isinstance(pressure.get("backlog_truth"), dict) else {}
    storage_efficiency = pressure.get("storage_efficiency") if isinstance(pressure.get("storage_efficiency"), dict) else {}
    provider_quality = pressure.get("provider_quality") if isinstance(pressure.get("provider_quality"), dict) else {}
    raw_headroom = _safe_int(raw_live.get("estimated_new_bot_headroom"), 0)
    raw_core_pending = _safe_int(raw_live.get("raw_core_pending_lines"), 0)
    raw_total_pending = _safe_int(raw_live.get("raw_total_pending_lines"), 0)
    overlay_total = _safe_int(truth.get("sql_overlay_total_pending_lines"), 0)
    host_saturation = _safe_float(pressure.get("host_saturation_score"), 0.0)
    runtime_status = str(pressure.get("runtime_status") or "").strip().lower()
    compute_level = str(pressure.get("compute_pressure_level") or "").strip().lower()
    memory_level = str(pressure.get("memory_pressure_level") or "").strip().lower()
    provider_status = str(provider_quality.get("provider_mesh_status") or "missing").strip().lower()
    source_status = str(provider_quality.get("source_verification_status") or "missing").strip().lower()
    provider_failures = _safe_int(provider_quality.get("provider_required_failure_count"), 0)
    provider_soft_failures = _safe_int(provider_quality.get("provider_soft_failure_count"), 0)
    source_unverified = _safe_int(provider_quality.get("source_unverified_count"), 0)
    data_quality_score = _safe_float(provider_quality.get("data_quality_score"), 0.0)
    admission_blocking = _safe_int(pressure.get("admission_blocking_candidate_count"), 0)
    storage_allow_expansion = bool(storage_efficiency.get("storage_allow_expansion", True))
    storage_phase = str(storage_efficiency.get("storage_plane_phase") or "").strip()
    storage_deficit_gb = _safe_float(storage_efficiency.get("storage_space_recovery_deficit_gb"), 0.0)
    storage_external_free_gb = _safe_float(storage_efficiency.get("storage_external_free_gb"), 0.0)
    storage_external_min_free_gb = _safe_float(storage_efficiency.get("storage_external_min_free_gb"), 0.0)
    storage_manifest_recovery_watch = bool(
        not storage_allow_expansion
        and storage_phase == "manifest_only_recovery"
        and storage_deficit_gb <= 0.0
        and storage_external_min_free_gb > 0.0
        and storage_external_free_gb >= storage_external_min_free_gb
    )
    raw_live_ready = bool(raw_live.get("ready", False)) and not bool(raw_live.get("active", False))
    raw_live_watch = bool(
        not raw_live_ready
        and raw_headroom > 0
        and raw_core_pending <= 5_000
        and raw_total_pending <= 5_000
        and _safe_float(raw_live.get("pressure_ratio"), 0.0) < 3.0
    )
    runtime_hard_block = bool(
        host_saturation >= 85.0
        or compute_level in {"high", "critical"}
        or memory_level in {"critical", "red"}
        or (runtime_status in {"blocked", "critical"} and (host_saturation >= 70.0 or memory_level in {"elevated", "yellow", "red"}))
    )
    provider_hard_block = bool(
        provider_status in {"blocked", "critical"}
        or source_status in {"blocked", "critical"}
        or provider_failures > 0
        or source_unverified > 0
    )
    provider_watch = bool(
        not provider_hard_block
        and (
            source_status in {"missing", "needs_work", "degraded"}
            or (data_quality_score > 0.0 and data_quality_score < 90.0)
            or provider_soft_failures > 2
            or (provider_status in {"missing", "needs_work", "degraded"} and provider_failures > 0)
        )
    )

    dimensions = [
        _dimension(
            "raw_live_headroom",
            status="ready" if raw_live_ready else "watch" if raw_live_watch else "blocked",
            evidence=raw_live,
            next_action="drain hot raw/live queues under reserve before broadening collectors or sleeves",
        ),
        _dimension(
            "sql_overlay_tail_debt",
            status="blocked" if overlay_total >= 100_000 else "watch" if overlay_total >= 50_000 else "ready",
            evidence={
                "sql_overlay_total_pending_lines": int(overlay_total),
                "sql_overlay_grade": str(truth.get("sql_overlay_grade") or ""),
                "sql_overlay_pressure_ratio": _safe_float(truth.get("sql_overlay_pressure_ratio"), 0.0),
            },
            next_action="clear cold/deferred SQL overlay tails before another broad expansion wave",
        ),
        _dimension(
            "storage_efficiency",
            status="ready"
            if storage_allow_expansion
            else "watch"
            if storage_manifest_recovery_watch
            else "blocked",
            evidence=storage_efficiency,
            next_action="let storage efficiency return BOT_STORAGE_ALLOW_EXPANSION=1 before expanding",
        ),
        _dimension(
            "runtime_headroom",
            status="blocked"
            if runtime_hard_block
            else "watch"
            if runtime_status in {"blocked", "critical"}
            or host_saturation >= 65.0
            or compute_level == "elevated"
            or memory_level in {"elevated", "yellow", "red"}
            else "ready",
            evidence={
                "runtime_status": runtime_status,
                "host_saturation_score": round(float(host_saturation), 3),
                "compute_pressure_level": compute_level,
                "memory_pressure_level": memory_level,
            },
            next_action="wait for runtime pressure to cool before adding persistent loops",
        ),
        _dimension(
            "provider_and_data_quality",
            status="blocked"
            if provider_hard_block
            else "watch"
            if provider_watch
            else "ready",
            evidence=provider_quality,
            next_action="refresh provider mesh and source verification before allowing new collectors to influence decisions",
        ),
        _dimension(
            "admission_evidence",
            status="blocked" if admission_blocking > 0 else "ready",
            evidence={
                "admission_candidate_count": _safe_int(pressure.get("admission_candidate_count"), 0),
                "admission_blocking_candidate_count": admission_blocking,
            },
            next_action="clear new-bot admission blockers before training or promotion-capable expansion",
        ),
    ]
    blocked = [row["name"] for row in dimensions if row["status"] == "blocked"]
    watch = [row["name"] for row in dimensions if row["status"] == "watch"]
    if blocked:
        mode = "blocked_clean_scaling"
        max_clean_wave = 0
    elif watch:
        mode = "micro_collection_only_soak"
        max_clean_wave = min(max(raw_headroom, 1), 3, max(int(requested_wave_size), 0))
    else:
        mode = "clean_collection_wave"
        max_clean_wave = min(max(raw_headroom, 1), max(int(requested_wave_size), 0))

    score = 100.0 - len(blocked) * 18.0 - len(watch) * 6.0
    return {
        "overall_status": "blocked" if blocked else "needs_work" if watch else "ready",
        "grade": _grade_from_score(score),
        "score": round(max(score, 0.0), 3),
        "mode": mode,
        "blocked_dimensions": blocked,
        "watch_dimensions": watch,
        "dimension_count": len(dimensions),
        "dimensions": dimensions,
        "max_clean_wave_size_now": int(max_clean_wave),
        "requested_wave_size": int(max(int(requested_wave_size), 0)),
        "clean_scaling_invariants": [
            "raw/live headroom must stay inside reserve before broad expansion",
            "SQL overlay cold tails cannot hide behind a clean raw queue",
            "storage efficiency must explicitly allow expansion",
            "runtime pressure must be calm enough for persistent loops",
            "provider/source verification must be clean before new data influences decisions",
            "admission evidence must be clean before bots leave collect-only probation",
        ],
        "next_action": dimensions[[row["status"] for row in dimensions].index("blocked")]["next_action"]
        if blocked
        else dimensions[[row["status"] for row in dimensions].index("watch")]["next_action"]
        if watch
        else "clean scaling gate is ready for a bounded collection-only wave",
        "fleet_counts": {
            "total_bots": len(rows),
            "data_collection_only_bots": sum(
                1 for row in rows if str(row.get("lifecycle_state") or "").strip().lower() == "data_collection_only"
            ),
        },
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, requested_wave_size: int = 20) -> dict[str, Any]:
    rows = _registry_rows(project_root)
    pressure = _pressure_snapshot(project_root)
    clean_scaling = _clean_scaling_contract(rows, pressure, requested_wave_size=requested_wave_size)
    capacity = _capacity_contract(
        rows,
        pressure,
        requested_wave_size=requested_wave_size,
        clean_scaling=clean_scaling,
    )
    worst_rank = max(
        status_rank(str(pressure.get("overall_status"))),
        status_rank(str(capacity.get("overall_status"))),
        status_rank(str(clean_scaling.get("overall_status"))),
    )
    overall_status = "ready"
    if worst_rank >= status_rank("blocked"):
        overall_status = "blocked"
    elif worst_rank >= status_rank("needs_work"):
        overall_status = "needs_work"

    recommended_actions = ordered_unique(
        [
            "sync roster-expansion with --apply-registry, then materialize core bot files"
            if capacity["recommended_wave_size_now"] > 0 and clean_scaling["overall_status"] == "ready"
            else "",
            str(clean_scaling.get("next_action") or ""),
            "keep every new bot data_collection_only, zero weight, paper locked, and live disabled",
            "run expansion-capacity before each future wave so growth reacts to storage, swap, runtime, halts, and admission pressure",
            "pause any future wave while global halt or high swap pressure is active" if capacity["max_new_collectors_now"] <= 0 else "",
            "clear new-bot admission contracts before allowing any of the expanded roster into training" if pressure["admission_blocking_candidate_count"] > 0 else "",
            "refresh core bot catalog after materialization so PyCharm shows the new files in one place",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "mode": "advisory_expansion_capacity_guard",
        "pressure_snapshot": pressure,
        "clean_scaling_contract": clean_scaling,
        "capacity_contract": capacity,
        "support_infrabots": SUPPORT_INFRABOT_IDS,
        "growth_invariants": [
            "new bots enter as data_collection_only",
            "new bots are excluded from training until minimum observations and days are met",
            "new bots keep zero allocation weight and live execution disabled",
            "future waves must pass storage, queue, swap, runtime, global halt, settlement, and admission gates",
            "pre-expansion stabilization must explicitly allow the wave before any roster growth",
            "registry sync and core materialization must run after each accepted wave",
        ],
        "recommended_actions": recommended_actions,
        "source_files": {
            "master_bot_registry": str(project_root / "master_bot_registry.json"),
            "artifact": str(DEFAULT_OUT_PATH),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan safe expansion capacity for new bot waves.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--wave-size", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root, requested_wave_size=max(int(args.wave_size), 0))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        contract = payload["capacity_contract"]
        print(
            "expansion_capacity_planner "
            f"status={payload.get('overall_status', '')} "
            f"recommended_wave={contract.get('recommended_wave_size_now', 0)} "
            f"max_new_collectors={contract.get('max_new_collectors_now', 0)} "
            f"mode={contract.get('rollout_mode', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
