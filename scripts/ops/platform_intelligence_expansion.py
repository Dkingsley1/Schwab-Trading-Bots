#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, parse_iso_utc, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "platform_intelligence_expansion_latest.json"

SLEEVE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("options_on_futures", "options_on_futures"),
    ("crypto_futures", "crypto_futures"),
    ("intraday_aggressive", "intraday_aggressive"),
    ("swing_aggressive", "swing_aggressive"),
    ("aggressive", "aggressive"),
    ("dividend", "dividend"),
    ("conservative", "conservative"),
    ("futures", "futures"),
    ("options", "options"),
    ("crypto", "crypto"),
    ("bond", "bond"),
    ("macro", "macro"),
    ("fx", "fx"),
    ("volatility", "volatility"),
    ("variance", "volatility"),
    ("correlation", "correlation"),
    ("dispersion", "dispersion"),
    ("microstructure", "microstructure"),
    ("market_making", "market_making"),
    ("infrastructure", "infrastructure"),
    ("governor", "infrastructure"),
    ("default", "default"),
)

REGIME_SLEEVE_ROUTING = {
    "risk_on_trend": {
        "boost": ["intraday_aggressive", "swing_aggressive", "aggressive", "crypto", "futures"],
        "downshift": ["bond", "conservative", "tail_risk"],
    },
    "risk_off_trend": {
        "boost": ["conservative", "bond", "dividend", "macro", "tail_risk", "volatility"],
        "downshift": ["intraday_aggressive", "aggressive", "crypto"],
    },
    "risk_off_shock": {
        "boost": ["tail_risk", "macro", "volatility", "conservative", "bond"],
        "downshift": ["intraday_aggressive", "aggressive", "crypto", "market_making"],
    },
    "fragile_transition": {
        "boost": ["microstructure", "volatility", "macro", "conservative"],
        "downshift": ["intraday_aggressive", "aggressive", "market_making"],
    },
    "rangebound_transition": {
        "boost": ["dividend", "conservative", "options", "dispersion", "market_making"],
        "downshift": ["swing_aggressive"],
    },
    "mixed_transition": {
        "boost": ["conservative", "macro", "microstructure", "options"],
        "downshift": ["intraday_aggressive"],
    },
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


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(float(lo), min(float(value), float(hi)))


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "active", "enabled"}


def _normalize_id(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _latest_age_days(raw: Any) -> float | None:
    parsed = parse_iso_utc(raw)
    if parsed is None:
        return None
    return max((datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0, 0.0)


def _metric01(row: dict[str, Any], *keys: str) -> float:
    values = []
    for key in keys:
        value = _safe_float(row.get(key), -1.0)
        if value >= 0.0:
            values.append(value)
    if not values:
        return 0.0
    return max(min(max(values), 1.0), 0.0)


def _load_registry(project_root: Path) -> list[dict[str, Any]]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _infer_sleeve(row: dict[str, Any]) -> str:
    for key in ("sleeve", "sleeve_profile", "profile", "strategy_sleeve", "slot_kind"):
        text = _normalize_id(row.get(key))
        if text:
            return text
    haystack_parts = [
        row.get("bot_id"),
        row.get("bot_role"),
        row.get("core_module_path"),
        row.get("lifecycle_state"),
        " ".join(str(item) for item in _as_list(row.get("target_functions"))),
        " ".join(str(item) for item in _as_list(row.get("data_intake_collections"))),
        " ".join(str(item) for item in _as_list(row.get("correlation_peer_sleeves"))),
    ]
    haystack = " ".join(str(part or "").lower() for part in haystack_parts)
    for needle, sleeve in SLEEVE_KEYWORDS:
        if needle in haystack:
            return sleeve
    return "default"


def _pressure_snapshot(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    swap = load_json(health_root / "swap_pressure_governor_latest.json")
    runtime = load_json(health_root / "runtime_throttle_control_latest.json")
    ingestion = load_json(health_root / "ingestion_storage_control_latest.json")
    global_halt = load_json(health_root / "global_killswitch_latest.json")
    memory = load_json(health_root / "memory_efficiency_control_latest.json")

    swap_pressure = _as_dict(swap.get("swap_pressure"))
    swap_tier = str(swap_pressure.get("tier") or swap.get("tier") or "normal").strip().lower()
    swap_gb = _safe_float(swap_pressure.get("swap_used_gb"), 0.0)
    host_saturation = _safe_float(runtime.get("host_saturation_score"), 0.0)
    runtime_status = str(runtime.get("overall_status") or "missing").strip().lower()
    memory_status = str(memory.get("overall_status") or "missing").strip().lower()
    memory_profile = str(memory.get("recommended_profile") or memory.get("active_profile") or "").strip().lower()
    storage_status = str(ingestion.get("overall_status") or "missing").strip().lower()
    pressure_index = _safe_float(ingestion.get("pressure_index"), 0.0)
    global_halt_active = any(
        _bool(global_halt.get(key))
        for key in ("global_halt_active", "halt_active", "hard_halt_active", "blocked", "killswitch_active")
    )

    high_swap = swap_tier in {"survival", "critical"} or swap_gb >= 20.0
    elevated_swap = high_swap or swap_tier in {"calm", "constrained", "pause_research"} or swap_gb >= 16.0
    high_runtime = runtime_status in {"blocked", "critical"} or host_saturation >= 85.0
    elevated_runtime = high_runtime or runtime_status == "degraded" or host_saturation >= 65.0
    storage_blocked = storage_status in {"blocked", "critical"}
    elevated_storage = storage_blocked or pressure_index >= 0.25

    overall_status = "ready"
    if global_halt_active or high_swap or high_runtime or storage_blocked:
        overall_status = "blocked"
    elif elevated_runtime or elevated_swap or elevated_storage or storage_status in {"degraded", "needs_work", "watch"}:
        overall_status = "degraded"

    return {
        "overall_status": overall_status,
        "swap_tier": swap_tier,
        "swap_used_gb": round(swap_gb, 3),
        "host_saturation_score": round(host_saturation, 3),
        "runtime_status": runtime_status,
        "memory_status": memory_status,
        "memory_profile": memory_profile,
        "storage_status": storage_status,
        "storage_pressure_index": round(pressure_index, 6),
        "global_halt_active": global_halt_active,
        "high_swap": high_swap,
        "high_runtime": high_runtime,
        "storage_blocked": storage_blocked,
        "elevated_swap": elevated_swap,
        "elevated_storage": elevated_storage,
        "compute_policy": "protect_live" if overall_status == "blocked" else ("sustain" if overall_status == "degraded" else "normal"),
        "source_files": {
            "swap_pressure_governor": str(health_root / "swap_pressure_governor_latest.json"),
            "runtime_throttle": str(health_root / "runtime_throttle_control_latest.json"),
            "ingestion_storage": str(health_root / "ingestion_storage_control_latest.json"),
            "global_killswitch": str(health_root / "global_killswitch_latest.json"),
        },
    }


def _bot_quality_row(row: dict[str, Any], *, sleeve: str) -> dict[str, Any]:
    active = _bool(row.get("active")) and not _bool(row.get("deleted_from_rotation"))
    lifecycle = str(row.get("lifecycle_state") or "").strip().lower()
    quality01 = _metric01(row, "quality_score", "candidate_quality_score", "registry_quality_score")
    accuracy01 = _metric01(row, "test_accuracy", "candidate_test_accuracy", "previous_best_accuracy")
    observations = _safe_int(row.get("data_collection_observations"), 0)
    min_obs = max(_safe_int(row.get("minimum_training_observations"), 1000), 1)
    started_age_days = _latest_age_days(row.get("data_collection_started_utc"))
    min_days = max(_safe_float(row.get("minimum_data_collection_days"), 7.0), 0.0)
    obs_ratio = min(observations / float(min_obs), 1.0)
    age_ratio = min(((started_age_days or 0.0) / min_days), 1.0) if min_days > 0.0 else 1.0
    data_sufficiency = min(obs_ratio, age_ratio) if _bool(row.get("data_collection_active")) else max(quality01, accuracy01)
    no_improvement = _safe_int(row.get("no_improvement_streak"), 0)
    paper_lock = _bool(row.get("paper_trade_lock_required"))
    resource_aware = _bool(row.get("resource_throttle_aware"))
    halt_aware = _bool(row.get("global_halt_aware"))
    rotation_blocked = _bool(row.get("rotation_blocked"))
    training_excluded = _bool(row.get("exclude_from_training")) or _bool(row.get("training_excluded"))

    score = 18.0 if active else 4.0
    score += 24.0 * quality01
    score += 22.0 * accuracy01
    score += 16.0 * data_sufficiency
    score += 5.0 if paper_lock else 0.0
    score += 5.0 if resource_aware else 0.0
    score += 5.0 if halt_aware else 0.0
    score -= min(no_improvement * 3.0, 15.0)
    score -= 8.0 if rotation_blocked and data_sufficiency < 1.0 else 0.0
    score -= 10.0 if lifecycle == "inactive" else 0.0
    score = _clamp(score)

    if score >= 75.0:
        quality_label = "strong"
    elif score >= 55.0:
        quality_label = "watch"
    elif score >= 35.0:
        quality_label = "probation"
    else:
        quality_label = "cold_start"

    return {
        "bot_id": _normalize_id(row.get("bot_id")),
        "bot_role": str(row.get("bot_role") or "unknown"),
        "sleeve": sleeve,
        "active": active,
        "lifecycle_state": lifecycle,
        "quality_score": round(score, 3),
        "quality_label": quality_label,
        "metric_quality": round(quality01, 6),
        "metric_accuracy": round(accuracy01, 6),
        "data_sufficiency": round(_clamp(data_sufficiency, 0.0, 1.0), 6),
        "data_collection_observations": observations,
        "minimum_training_observations": min_obs,
        "data_collection_age_days": round(started_age_days, 3) if started_age_days is not None else None,
        "minimum_data_collection_days": min_days,
        "no_improvement_streak": no_improvement,
        "paper_trade_lock_required": paper_lock,
        "resource_throttle_aware": resource_aware,
        "global_halt_aware": halt_aware,
        "training_excluded": training_excluded,
        "training_candidate_after_threshold": _bool(row.get("training_candidate_after_threshold")),
        "direct_execution_allowed": _bool(row.get("direct_execution_allowed")),
        "correlation_peer_sleeves": [str(item) for item in _as_list(row.get("correlation_peer_sleeves"))],
        "correlation_dependencies": [str(item) for item in _as_list(row.get("correlation_dependencies"))],
    }


def _quality_system(quality_rows: list[dict[str, Any]], *, max_rows: int) -> dict[str, Any]:
    if not quality_rows:
        return {
            "overall_status": "missing",
            "bot_count": 0,
            "average_quality_score": 0.0,
            "label_counts": {},
            "top_probation": [],
            "top_strong": [],
        }
    label_counts = Counter(str(row.get("quality_label") or "") for row in quality_rows)
    avg = sum(_safe_float(row.get("quality_score"), 0.0) for row in quality_rows) / len(quality_rows)
    probation = sorted(
        [row for row in quality_rows if str(row.get("quality_label")) in {"probation", "cold_start"}],
        key=lambda row: (_safe_float(row.get("quality_score"), 0.0), str(row.get("bot_id") or "")),
    )
    strong = sorted(quality_rows, key=lambda row: (-_safe_float(row.get("quality_score"), 0.0), str(row.get("bot_id") or "")))
    overall_status = "ready"
    if label_counts.get("cold_start", 0) + label_counts.get("probation", 0) > max(len(quality_rows) * 0.35, 20):
        overall_status = "needs_work"
    return {
        "overall_status": overall_status,
        "bot_count": len(quality_rows),
        "average_quality_score": round(avg, 3),
        "label_counts": dict(sorted(label_counts.items())),
        "top_probation": probation[:max_rows],
        "top_strong": strong[:max_rows],
        "score_contract": {
            "active_weight": 18,
            "quality_metric_weight": 24,
            "accuracy_weight": 22,
            "data_sufficiency_weight": 16,
            "safety_awareness_bonus": 15,
            "decay_penalty_cap": 15,
        },
    }


def _admission_controller(quality_rows: list[dict[str, Any]], pressure: dict[str, Any], *, max_rows: int) -> dict[str, Any]:
    high_pressure = str(pressure.get("overall_status")) == "blocked"
    global_halt_active = _bool(pressure.get("global_halt_active"))
    swap_paused = str(pressure.get("swap_tier")) in {"pause_research", "survival"}
    rows: list[dict[str, Any]] = []
    counts = Counter()

    for row in quality_rows:
        active = _bool(row.get("active"))
        data_ready = _safe_float(row.get("data_sufficiency"), 0.0) >= 1.0
        score = _safe_float(row.get("quality_score"), 0.0)
        direct_execution_allowed = _bool(row.get("direct_execution_allowed"))
        training_candidate = _bool(row.get("training_candidate_after_threshold")) or not _bool(row.get("training_excluded"))

        collect_allowed = active and not global_halt_active
        collect_mode = "thin_sample" if high_pressure else "normal_sample"
        paper_trade_allowed = active and not global_halt_active and score >= 45.0 and not direct_execution_allowed
        train_allowed = active and data_ready and training_candidate and not high_pressure and not swap_paused and score >= 50.0
        live_trade_allowed = active and direct_execution_allowed and score >= 80.0 and not high_pressure and not global_halt_active

        if collect_allowed:
            counts["collect_allowed"] += 1
        if paper_trade_allowed:
            counts["paper_trade_allowed"] += 1
        if train_allowed:
            counts["train_allowed"] += 1
        if live_trade_allowed:
            counts["live_trade_allowed"] += 1
        if not train_allowed and data_ready and training_candidate:
            counts["train_deferred"] += 1

        if not collect_allowed:
            next_action = "hold_until_global_halt_or_lifecycle_clears"
        elif train_allowed:
            next_action = "eligible_for_train_queue"
        elif data_ready and training_candidate:
            next_action = "defer_training_until_resource_pressure_clears"
        elif paper_trade_allowed:
            next_action = "collect_and_paper_trade"
        else:
            next_action = "collect_until_minimum_data_floor"

        rows.append(
            {
                "bot_id": row["bot_id"],
                "sleeve": row["sleeve"],
                "quality_score": row["quality_score"],
                "data_sufficiency": row["data_sufficiency"],
                "collect_allowed": collect_allowed,
                "collection_mode": collect_mode if collect_allowed else "blocked",
                "paper_trade_allowed": paper_trade_allowed,
                "train_allowed": train_allowed,
                "live_trade_allowed": live_trade_allowed,
                "admission_state": next_action,
            }
        )

    rows.sort(key=lambda row: (-_safe_float(row.get("quality_score"), 0.0), str(row.get("bot_id") or "")))
    overall_status = "ready"
    if high_pressure:
        overall_status = "protect_live"
    if not rows:
        overall_status = "missing"
    return {
        "overall_status": overall_status,
        "mode": "advisory_read_only",
        "pressure_policy": pressure.get("compute_policy", "normal"),
        "bot_count": len(rows),
        "counts": dict(counts),
        "sampled_admissions": rows[:max_rows],
        "rules": {
            "global_halt_blocks_collection_and_paper": True,
            "swap_pause_blocks_training": True,
            "direct_live_execution_requires_explicit_registry_permission": True,
            "under_pressure_collection_downshifts_to_thin_sample": True,
        },
    }


def _sleeve_masters(quality_rows: list[dict[str, Any]], pressure: dict[str, Any], *, max_rows: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in quality_rows:
        grouped[str(row.get("sleeve") or "default")].append(row)
    masters: list[dict[str, Any]] = []
    for sleeve, rows in sorted(grouped.items()):
        avg = sum(_safe_float(row.get("quality_score"), 0.0) for row in rows) / max(len(rows), 1)
        active = sum(1 for row in rows if _bool(row.get("active")))
        probation = sum(1 for row in rows if str(row.get("quality_label")) in {"probation", "cold_start"})
        train_ready = sum(1 for row in rows if _safe_float(row.get("data_sufficiency"), 0.0) >= 1.0 and not _bool(row.get("training_excluded")))
        status = "ready"
        if str(pressure.get("overall_status")) == "blocked":
            status = "protect_live"
        elif probation > max(len(rows) * 0.35, 5):
            status = "needs_work"
        masters.append(
            {
                "virtual_master_bot_id": f"sleeve_master_{sleeve}",
                "sleeve": sleeve,
                "reports_to": "grand_master_bot",
                "overall_status": status,
                "bot_count": len(rows),
                "active_bot_count": active,
                "train_ready_bot_count": train_ready,
                "average_quality_score": round(avg, 3),
                "probation_or_cold_start_count": probation,
                "supervision_contract": [
                    "aggregate sleeve quality",
                    "enforce admission state",
                    "report capacity pressure",
                    "surface correlation concentration",
                    "escalate only material issues to grand master",
                ],
            }
        )
    masters.sort(key=lambda row: (-_safe_int(row.get("active_bot_count"), 0), str(row.get("sleeve") or "")))
    return {
        "overall_status": "ready" if masters else "missing",
        "grand_master_contract": "grand_master_receives_sleeve_rollups_not_raw_bot_noise",
        "sleeve_master_count": len(masters),
        "sleeve_masters": masters[:max_rows],
    }


def _execution_realism(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    calibration = load_json(health_root / "paper_execution_calibration_latest.json")
    execution_lab = load_json(health_root / "execution_lab_latest.json")
    capacity_curves = load_json(project_root / "governance" / "allocator" / "portfolio_capacity_curve_latest.json")
    metrics = _as_dict(calibration.get("metrics"))
    mae_bps = _safe_float(metrics.get("mae_bps"), 0.0)
    poor_fills = _safe_int(metrics.get("poor_or_fair_fill_count"), 0)
    worst = _as_list(execution_lab.get("top_worst_case_scenarios"))
    worst_slippage = max([_safe_float(row.get("slippage_bps"), 0.0) for row in worst if isinstance(row, dict)] or [0.0])
    curve_summary = _as_dict(capacity_curves.get("summary"))
    constrained_curves = _safe_int(curve_summary.get("constrained_curve_count"), 0)
    status = "ready"
    if mae_bps >= 35.0 or worst_slippage >= 40.0 or constrained_curves > 0:
        status = "needs_work"
    if not calibration and not execution_lab:
        status = "missing"
    return {
        "overall_status": status,
        "mae_bps": round(mae_bps, 3),
        "poor_or_fair_fill_count": poor_fills,
        "worst_lab_slippage_bps": round(worst_slippage, 3),
        "constrained_capacity_curve_count": constrained_curves,
        "paper_trade_realism_contract": [
            "slippage",
            "spread",
            "fill_probability",
            "queue_priority",
            "latency_bucket",
            "capacity_fraction",
        ],
        "source_files": {
            "paper_execution_calibration": str(health_root / "paper_execution_calibration_latest.json"),
            "execution_lab": str(health_root / "execution_lab_latest.json"),
            "portfolio_capacity_curves": str(project_root / "governance" / "allocator" / "portfolio_capacity_curve_latest.json"),
        },
    }


def _market_regime_router(project_root: Path, sleeve_masters: dict[str, Any]) -> dict[str, Any]:
    regime = load_json(project_root / "governance" / "health" / "regime_control_plane_latest.json")
    regime_state = str(regime.get("regime_state") or "mixed_transition").strip().lower()
    routing = REGIME_SLEEVE_ROUTING.get(regime_state, REGIME_SLEEVE_ROUTING["mixed_transition"])
    sleeve_set = {str(row.get("sleeve") or "") for row in _as_list(sleeve_masters.get("sleeve_masters")) if isinstance(row, dict)}
    boost = [sleeve for sleeve in routing["boost"] if sleeve in sleeve_set or sleeve in {"tail_risk", "macro", "volatility"}]
    downshift = [sleeve for sleeve in routing["downshift"] if sleeve in sleeve_set or sleeve in {"tail_risk"}]
    return {
        "overall_status": "ready" if regime else "thin",
        "regime_state": regime_state,
        "stance_label": str(regime.get("stance_label") or ""),
        "stance_score": _safe_float(regime.get("stance_score"), 0.0),
        "boost_sleeves": ordered_unique(boost),
        "downshift_sleeves": ordered_unique(downshift),
        "routing_contract": "sleeves are favored or downshifted; individual bots still pass admission control",
        "source_file": str(project_root / "governance" / "health" / "regime_control_plane_latest.json"),
    }


def _capacity_planner(pressure: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
    pressure_status = str(pressure.get("overall_status") or "")
    host = _safe_float(pressure.get("host_saturation_score"), 0.0)
    swap = _safe_float(pressure.get("swap_used_gb"), 0.0)
    counts = _as_dict(admission.get("counts"))
    collect_allowed = _safe_int(counts.get("collect_allowed"), 0)
    if pressure_status == "blocked":
        max_new_collectors = 0
        maintenance_policy = "off_hours_only"
        training_policy = "paused"
    elif pressure_status == "degraded":
        max_new_collectors = max(1, min(10, collect_allowed // 40))
        maintenance_policy = "low_priority_window"
        training_policy = "small_batch_only"
    else:
        max_new_collectors = max(5, min(25, collect_allowed // 20))
        maintenance_policy = "normal"
        training_policy = "allowed"
    return {
        "overall_status": pressure_status or "ready",
        "host_saturation_score": host,
        "swap_used_gb": swap,
        "max_new_collectors_now": max_new_collectors,
        "maintenance_policy": maintenance_policy,
        "training_policy": training_policy,
        "recommended_sql_writer_mode": "thin_sample" if pressure_status == "blocked" else "normal",
        "heavy_job_launch_allowed": pressure_status == "ready",
        "capacity_contract": [
            "predict pressure before launching maintenance",
            "prefer live collection over report/research jobs",
            "resume training only after swap and host saturation clear",
        ],
    }


def _research_pipeline(quality_rows: list[dict[str, Any]], *, max_rows: int) -> dict[str, Any]:
    ideas = []
    for row in quality_rows:
        lifecycle = str(row.get("lifecycle_state") or "")
        data_sufficiency = _safe_float(row.get("data_sufficiency"), 0.0)
        if lifecycle == "data_collection_only" or data_sufficiency < 1.0:
            stage = "paper_only_collecting"
        elif _bool(row.get("training_excluded")):
            stage = "ready_for_training_gate_review"
        elif _safe_float(row.get("quality_score"), 0.0) >= 65.0:
            stage = "candidate_for_retrain"
        else:
            stage = "needs_label_or_data_repair"
        if stage in {"paper_only_collecting", "ready_for_training_gate_review", "candidate_for_retrain"}:
            ideas.append(
                {
                    "bot_id": row["bot_id"],
                    "sleeve": row["sleeve"],
                    "stage": stage,
                    "data_sufficiency": row["data_sufficiency"],
                    "quality_score": row["quality_score"],
                }
            )
    stage_counts = Counter(str(row.get("stage") or "") for row in ideas)
    ideas.sort(key=lambda row: (str(row.get("stage") or ""), -_safe_float(row.get("quality_score"), 0.0), str(row.get("bot_id") or "")))
    return {
        "overall_status": "ready" if ideas else "thin",
        "pipeline_contract": "idea_to_data_to_paper_only_to_trainable_candidate",
        "stage_counts": dict(sorted(stage_counts.items())),
        "research_queue": ideas[:max_rows],
    }


def _correlation_governor(quality_rows: list[dict[str, Any]], *, max_rows: int) -> dict[str, Any]:
    sleeve_counts = Counter(str(row.get("sleeve") or "default") for row in quality_rows)
    dependency_counts: Counter[str] = Counter()
    peer_counts: Counter[str] = Counter()
    for row in quality_rows:
        dependency_counts.update(str(item) for item in _as_list(row.get("correlation_dependencies")) if str(item).strip())
        peer_counts.update(str(item) for item in _as_list(row.get("correlation_peer_sleeves")) if str(item).strip())
    total = max(sum(sleeve_counts.values()), 1)
    concentration = max(sleeve_counts.values() or [0]) / float(total)
    overloaded_dependencies = [
        {"dependency": key, "bot_count": count}
        for key, count in dependency_counts.most_common(max_rows)
        if count >= max(10, total * 0.1)
    ]
    status = "ready"
    if concentration >= 0.30 or overloaded_dependencies:
        status = "needs_work"
    return {
        "overall_status": status,
        "sleeve_concentration": round(concentration, 6),
        "largest_sleeves": [{"sleeve": key, "bot_count": count} for key, count in sleeve_counts.most_common(max_rows)],
        "overloaded_correlation_dependencies": overloaded_dependencies[:max_rows],
        "top_peer_sleeves": [{"peer_sleeve": key, "bot_count": count} for key, count in peer_counts.most_common(max_rows)],
        "governor_contract": [
            "detect hidden same-bet concentration",
            "penalize overloaded correlation dependencies",
            "route sleeve masters through correlation heat before promotion",
        ],
    }


def _model_decay(project_root: Path, quality_rows: list[dict[str, Any]], *, max_rows: int) -> dict[str, Any]:
    decay = load_json(project_root / "governance" / "research" / "decay_monitor_latest.json")
    weak_sleeves = {
        str(row.get("profile") or "").strip().lower()
        for row in _as_list(decay.get("weak_sleeves"))
        if isinstance(row, dict)
    }
    decaying = [
        {
            "bot_id": row["bot_id"],
            "sleeve": row["sleeve"],
            "quality_score": row["quality_score"],
            "no_improvement_streak": row["no_improvement_streak"],
            "decay_reason": "weak_sleeve_or_no_improvement",
        }
        for row in quality_rows
        if _safe_int(row.get("no_improvement_streak"), 0) >= 3 or str(row.get("sleeve") or "") in weak_sleeves
    ]
    decaying.sort(key=lambda row: (-_safe_int(row.get("no_improvement_streak"), 0), _safe_float(row.get("quality_score"), 0.0)))
    status = "ready"
    if decaying or str(decay.get("overall_status") or "") == "needs_work":
        status = "needs_work"
    if not decay:
        status = "thin"
    return {
        "overall_status": status,
        "decay_monitor_status": str(decay.get("overall_status") or "missing"),
        "decaying_bot_count": len(decaying),
        "weak_sleeves": sorted(weak_sleeves),
        "decaying_bots": decaying[:max_rows],
        "source_file": str(project_root / "governance" / "research" / "decay_monitor_latest.json"),
    }


def _system_dashboard(
    sections: dict[str, dict[str, Any]],
    *,
    bot_count: int,
    sleeve_count: int,
) -> dict[str, Any]:
    rows = []
    for key, payload in sections.items():
        status = str(payload.get("overall_status") or "missing")
        rows.append({"section": key, "overall_status": status, "rank": status_rank(status)})
    worst_rank = max([_safe_int(row.get("rank"), 1) for row in rows] or [1])
    if worst_rank >= status_rank("blocked"):
        overall_status = "blocked"
    elif worst_rank >= status_rank("degraded"):
        overall_status = "degraded"
    elif any(str(row.get("overall_status")) in {"needs_work", "thin"} for row in rows):
        overall_status = "needs_work"
    else:
        overall_status = "ready"
    return {
        "overall_status": overall_status,
        "bot_count": bot_count,
        "sleeve_count": sleeve_count,
        "section_count": len(rows),
        "sections": rows,
        "operator_view": {
            "top_command": "./scripts/ops/opsctl.sh platform-intelligence --json",
            "dashboard_artifact": str(DEFAULT_OUT_PATH),
            "purpose": "one report for bot count, sleeve health, halts, swap pressure, data quality, paper realism, training readiness, and weak spots",
        },
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, max_rows: int = 25) -> dict[str, Any]:
    bots = _load_registry(project_root)
    pressure = _pressure_snapshot(project_root)
    quality_rows = [_bot_quality_row(row, sleeve=_infer_sleeve(row)) for row in bots if _normalize_id(row.get("bot_id"))]

    quality = _quality_system(quality_rows, max_rows=max_rows)
    admission = _admission_controller(quality_rows, pressure, max_rows=max_rows)
    sleeve_masters = _sleeve_masters(quality_rows, pressure, max_rows=max_rows)
    execution_realism = _execution_realism(project_root)
    regime_router = _market_regime_router(project_root, sleeve_masters)
    capacity_planner = _capacity_planner(pressure, admission)
    research_pipeline = _research_pipeline(quality_rows, max_rows=max_rows)
    correlation = _correlation_governor(quality_rows, max_rows=max_rows)
    decay = _model_decay(project_root, quality_rows, max_rows=max_rows)

    sections = {
        "bot_admission_controller": admission,
        "per_sleeve_master_bots": sleeve_masters,
        "bot_quality_score_system": quality,
        "execution_realism_engine": execution_realism,
        "market_regime_router": regime_router,
        "swap_cpu_capacity_planner": capacity_planner,
        "research_to_strategy_pipeline": research_pipeline,
        "cross_sleeve_correlation_governor": correlation,
        "model_decay_detector": decay,
    }
    dashboard = _system_dashboard(
        sections,
        bot_count=len(quality_rows),
        sleeve_count=_safe_int(sleeve_masters.get("sleeve_master_count"), 0),
    )
    dashboard["sections"].append(
        {
            "section": "professional_system_dashboard",
            "overall_status": dashboard["overall_status"],
            "rank": status_rank(str(dashboard["overall_status"])),
        }
    )
    dashboard["section_count"] = len(sections) + 1
    sections["professional_system_dashboard"] = dashboard

    top_actions = []
    if pressure.get("overall_status") == "blocked":
        top_actions.append("keep new admissions advisory-only and pause training until swap/runtime pressure clears")
    if quality.get("overall_status") != "ready":
        top_actions.append("work the bot-quality probation queue before adding more runtime-heavy strategies")
    if correlation.get("overall_status") != "ready":
        top_actions.append("use the correlation governor before promoting sleeves that share the same dependencies")
    if execution_realism.get("overall_status") != "ready":
        top_actions.append("refresh execution lab and calibration before trusting paper PnL as live-like")
    if not top_actions:
        top_actions.append("keep this platform-intelligence layer in the dashboard refresh path as the fleet grows")

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": dashboard["overall_status"] in {"ready", "needs_work", "degraded", "blocked"},
        "overall_status": dashboard["overall_status"],
        "mode": "advisory_read_only_no_new_runtime_loops",
        "expansion_count": 10,
        "bot_count": len(quality_rows),
        "sleeve_count": dashboard["sleeve_count"],
        "pressure_snapshot": pressure,
        "sections": sections,
        "top_actions": top_actions,
        "source_files": {
            "master_bot_registry": str(project_root / "master_bot_registry.json"),
            "primary_artifact": str(project_root / "governance" / "health" / "platform_intelligence_expansion_latest.json"),
        },
    }


def write_section_artifacts(project_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    sections = _as_dict(payload.get("sections"))
    platform_intel_root = project_root / "governance" / "platform_intelligence"
    paths = {
        "bot_admission_controller": platform_intel_root / "bot_admission_controller_latest.json",
        "per_sleeve_master_bots": platform_intel_root / "sleeve_masters_latest.json",
        "bot_quality_score_system": platform_intel_root / "bot_quality_scores_latest.json",
        "execution_realism_engine": platform_intel_root / "execution_realism_latest.json",
        "market_regime_router": platform_intel_root / "market_regime_router_latest.json",
        "swap_cpu_capacity_planner": platform_intel_root / "capacity_planner_latest.json",
        "research_to_strategy_pipeline": platform_intel_root / "research_pipeline_latest.json",
        "cross_sleeve_correlation_governor": platform_intel_root / "correlation_governor_latest.json",
        "model_decay_detector": platform_intel_root / "model_decay_detector_latest.json",
        "professional_system_dashboard": platform_intel_root / "system_dashboard_latest.json",
    }
    written: dict[str, str] = {}
    for key, path in paths.items():
        section = sections.get(key)
        if isinstance(section, dict):
            write_payload(path, {"timestamp_utc": payload.get("timestamp_utc"), "schema_version": 1, **section})
            written[key] = str(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the 10-part platform intelligence expansion contract.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--max-rows", type=int, default=25)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root, max_rows=max(int(args.max_rows), 1))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    written = write_section_artifacts(project_root, payload)
    payload["section_artifacts"] = written
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "platform_intelligence_expansion "
            f"status={payload.get('overall_status', '')} "
            f"bots={int(payload.get('bot_count', 0) or 0)} "
            f"sleeves={int(payload.get('sleeve_count', 0) or 0)} "
            f"sections={int(payload.get('expansion_count', 0) or 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
