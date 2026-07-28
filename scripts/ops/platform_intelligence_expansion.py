#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
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
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "platform_intelligence_layer_v2.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.platform_intelligence_override"

PRIMARY_SECTION_KEYS = (
    "bot_lifecycle_manager",
    "bot_data_quality_scores",
    "provider_rotation_failover_mesh",
    "backpressure_prediction_engine",
    "duplicate_alpha_overlap_detector",
    "paper_trade_capacity_governor",
    "self_healing_incident_playbooks",
    "per_sleeve_master_bots",
    "training_readiness_board",
    "market_regime_router",
    "execution_paper_trade_realism_layer",
    "system_black_box_recorder",
)

PLATFORM_INTELLIGENCE_CONTROLS: tuple[dict[str, Any], ...] = (
    {"id": "bot_lifecycle_manager", "title": "Bot lifecycle manager", "env_key": "BOT_LIFECYCLE_MANAGER_ENABLED"},
    {"id": "bot_data_quality_scores", "title": "Data quality score per bot", "env_key": "BOT_DATA_QUALITY_SCORE_ENABLED"},
    {"id": "provider_rotation_failover_mesh", "title": "Provider rotation and failover", "env_key": "PROVIDER_FAILOVER_MESH_ENABLED"},
    {"id": "backpressure_prediction_engine", "title": "Backpressure prediction", "env_key": "BACKPRESSURE_PREDICTOR_ENABLED"},
    {"id": "duplicate_alpha_overlap_detector", "title": "Duplicate alpha overlap detector", "env_key": "DUPLICATE_ALPHA_DETECTOR_ENABLED"},
    {"id": "paper_trade_capacity_governor", "title": "Paper trade capacity governor", "env_key": "PAPER_TRADE_CAPACITY_GOVERNOR_ENABLED"},
    {"id": "self_healing_incident_playbooks", "title": "Self-healing incident playbooks", "env_key": "SELF_HEALING_PLAYBOOKS_ENABLED"},
    {"id": "per_sleeve_master_bots", "title": "Master bot per sleeve rollups", "env_key": "SLEEVE_MASTER_ROLLUP_ENABLED"},
    {"id": "training_readiness_board", "title": "Training readiness board", "env_key": "TRAINING_READINESS_BOARD_ENABLED"},
    {"id": "market_regime_router", "title": "Market regime router", "env_key": "MARKET_REGIME_ROUTER_ENABLED"},
    {"id": "execution_paper_trade_realism_layer", "title": "Execution and paper trade realism", "env_key": "PAPER_EXECUTION_REALISM_ENABLED"},
    {"id": "system_black_box_recorder", "title": "System black box recorder", "env_key": "BLACK_BOX_RECORDER_ENABLED"},
)

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


def _watch_or_needs_work(*, watch: bool, hard: bool = False) -> str:
    if hard:
        return "needs_work"
    if watch:
        return "watch"
    return "ready"


def _read_health(project_root: Path, name: str) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / name)
    return payload if isinstance(payload, dict) else {}


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
    compute_level = str(runtime.get("compute_pressure_level") or runtime.get("compute_level") or "normal").strip().lower()
    runtime_memory_level = str(runtime.get("memory_pressure_level") or runtime.get("memory_level") or "normal").strip().lower()
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
    managed_runtime_degraded = bool(
        runtime_status == "degraded"
        and host_saturation < 50.0
        and compute_level in {"", "normal", "low", "idle"}
        and runtime_memory_level in {"", "normal", "low", "green"}
    )
    elevated_runtime = high_runtime or (runtime_status == "degraded" and not managed_runtime_degraded) or host_saturation >= 65.0
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
        "runtime_compute_level": compute_level,
        "runtime_memory_level": runtime_memory_level,
        "managed_runtime_degraded": managed_runtime_degraded,
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
    lifecycle = str(row.get("lifecycle_state") or "").strip().lower()
    deleted_or_inactive = bool(
        _bool(row.get("deleted_from_rotation"))
        or lifecycle in {"deleted", "inactive", "retired", "disabled"}
    )
    active = _bool(row.get("active")) and not deleted_or_inactive
    collection_active = _bool(row.get("data_collection_active")) or lifecycle == "data_collection_only"
    quality01 = _metric01(row, "quality_score", "candidate_quality_score", "registry_quality_score")
    accuracy01 = _metric01(row, "test_accuracy", "candidate_test_accuracy", "previous_best_accuracy")
    observations = _safe_int(row.get("data_collection_observations"), 0)
    min_obs = max(_safe_int(row.get("minimum_training_observations"), 1000), 1)
    started_age_days = _latest_age_days(row.get("data_collection_started_utc"))
    min_days = max(_safe_float(row.get("minimum_data_collection_days"), 7.0), 0.0)
    obs_ratio = min(observations / float(min_obs), 1.0)
    age_ratio = min(((started_age_days or 0.0) / min_days), 1.0) if min_days > 0.0 else 1.0
    data_sufficiency = min(obs_ratio, age_ratio) if collection_active else max(quality01, accuracy01)
    no_improvement = _safe_int(row.get("no_improvement_streak"), 0)
    paper_lock = _bool(row.get("paper_trade_lock_required"))
    resource_aware = _bool(row.get("resource_throttle_aware"))
    halt_aware = _bool(row.get("global_halt_aware"))
    rotation_blocked = _bool(row.get("rotation_blocked"))
    training_excluded = _bool(row.get("exclude_from_training")) or _bool(row.get("training_excluded"))
    direct_execution_allowed = _bool(row.get("direct_execution_allowed"))
    target_functions = [str(item) for item in _as_list(row.get("target_functions"))]
    correlation_dependencies = [str(item) for item in _as_list(row.get("correlation_dependencies"))]
    generic_paper_collection_contract = bool(
        {"paper_live_data_standard", "paper_trade_lock", "data_collection_floor"}.issubset(set(target_functions))
        and not correlation_dependencies
    )

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
    ignored_for_active_quality = bool(deleted_or_inactive or (not active and not collection_active))
    managed_quality_debt = bool(
        not ignored_for_active_quality
        and not direct_execution_allowed
        and (
            (collection_active and training_excluded and data_sufficiency < 1.0)
            or (paper_lock and generic_paper_collection_contract and quality_label in {"cold_start", "probation"})
        )
    )
    if ignored_for_active_quality:
        subject_state = "ignored_deleted_or_inactive"
    elif managed_quality_debt:
        subject_state = "managed_collect_only_maturity"
    else:
        subject_state = "actionable_quality_subject"

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
        "collection_active": collection_active,
        "ignored_for_active_quality_score": ignored_for_active_quality,
        "managed_quality_debt": managed_quality_debt,
        "generic_paper_collection_contract": generic_paper_collection_contract,
        "quality_subject_state": subject_state,
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
        "direct_execution_allowed": direct_execution_allowed,
        "correlation_peer_sleeves": [str(item) for item in _as_list(row.get("correlation_peer_sleeves"))],
        "correlation_dependencies": correlation_dependencies,
        "target_functions": target_functions,
        "data_intake_collections": [str(item) for item in _as_list(row.get("data_intake_collections"))],
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
    raw_label_counts = Counter(str(row.get("quality_label") or "") for row in quality_rows)
    active_quality_rows = [row for row in quality_rows if not _bool(row.get("ignored_for_active_quality_score"))]
    actionable_quality_rows = [row for row in active_quality_rows if not _bool(row.get("managed_quality_debt"))]
    label_counts = Counter(str(row.get("quality_label") or "") for row in active_quality_rows)
    actionable_label_counts = Counter(str(row.get("quality_label") or "") for row in actionable_quality_rows)
    avg = sum(_safe_float(row.get("quality_score"), 0.0) for row in active_quality_rows) / max(len(active_quality_rows), 1)
    raw_avg = sum(_safe_float(row.get("quality_score"), 0.0) for row in quality_rows) / len(quality_rows)
    actionable_avg = (
        sum(_safe_float(row.get("quality_score"), 0.0) for row in actionable_quality_rows) / len(actionable_quality_rows)
        if actionable_quality_rows
        else 100.0
    )
    probation = sorted(
        [row for row in active_quality_rows if str(row.get("quality_label")) in {"probation", "cold_start"}],
        key=lambda row: (_safe_float(row.get("quality_score"), 0.0), str(row.get("bot_id") or "")),
    )
    strong = sorted(active_quality_rows, key=lambda row: (-_safe_float(row.get("quality_score"), 0.0), str(row.get("bot_id") or "")))
    raw_debt_count = raw_label_counts.get("cold_start", 0) + raw_label_counts.get("probation", 0)
    active_debt_rows = [row for row in active_quality_rows if str(row.get("quality_label")) in {"probation", "cold_start"}]
    managed_debt_rows = [row for row in active_debt_rows if _bool(row.get("managed_quality_debt"))]
    unmanaged_debt_rows = [row for row in active_debt_rows if not _bool(row.get("managed_quality_debt"))]
    ignored_debt_rows = [
        row
        for row in quality_rows
        if _bool(row.get("ignored_for_active_quality_score")) and str(row.get("quality_label")) in {"probation", "cold_start"}
    ]
    debt_count = len(unmanaged_debt_rows)
    debt_ratio = debt_count / max(len(actionable_quality_rows), 1)
    unsafe_live_candidates = [
        row
        for row in active_quality_rows
        if _bool(row.get("direct_execution_allowed")) and _safe_float(row.get("quality_score"), 0.0) < 80.0
    ]
    overall_status = _watch_or_needs_work(
        watch=actionable_avg < 55.0 or debt_count > max(len(actionable_quality_rows) * 0.35, 20),
        hard=bool(unsafe_live_candidates),
    )
    return {
        "overall_status": overall_status,
        "bot_count": len(active_quality_rows),
        "raw_bot_count": len(quality_rows),
        "ignored_bot_count": len(quality_rows) - len(active_quality_rows),
        "actionable_bot_count": len(actionable_quality_rows),
        "average_quality_score": round(avg, 3),
        "raw_average_quality_score": round(raw_avg, 3),
        "actionable_average_quality_score": round(actionable_avg, 3),
        "label_counts": dict(sorted(label_counts.items())),
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "actionable_label_counts": dict(sorted(actionable_label_counts.items())),
        "quality_debt_count": debt_count,
        "raw_quality_debt_count": raw_debt_count,
        "active_quality_debt_count": len(active_debt_rows),
        "managed_quality_debt_count": len(managed_debt_rows),
        "ignored_quality_debt_count": len(ignored_debt_rows),
        "quality_debt_ratio": round(debt_ratio, 6),
        "managed_quality_debt_ratio": round(len(managed_debt_rows) / max(len(active_quality_rows), 1), 6),
        "unsafe_live_candidate_count": len(unsafe_live_candidates),
        "top_probation": probation[:max_rows],
        "top_actionable_probation": unmanaged_debt_rows[:max_rows],
        "top_managed_maturity_debt": managed_debt_rows[:max_rows],
        "top_strong": strong[:max_rows],
        "managed_debt_contract": {
            "active": bool(managed_debt_rows and not unmanaged_debt_rows and not unsafe_live_candidates),
            "policy": "training_excluded_collect_only_bots_count_as_maturity_debt_not_active_soak_repair_debt",
            "unmanaged_quality_debt_count": len(unmanaged_debt_rows),
            "managed_quality_debt_count": len(managed_debt_rows),
            "ignored_quality_debt_count": len(ignored_debt_rows),
        },
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


def _bot_lifecycle_manager(quality_rows: list[dict[str, Any]], admission: dict[str, Any], *, max_rows: int) -> dict[str, Any]:
    admission_by_bot = {
        str(row.get("bot_id") or ""): row
        for row in _as_list(admission.get("sampled_admissions"))
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in quality_rows:
        bot_id = str(row.get("bot_id") or "")
        admission_row = admission_by_bot.get(bot_id, {})
        active = _bool(row.get("active"))
        data_sufficiency = _safe_float(row.get("data_sufficiency"), 0.0)
        quality_score = _safe_float(row.get("quality_score"), 0.0)
        collect_allowed = _bool(admission_row.get("collect_allowed")) if admission_row else active
        paper_allowed = _bool(admission_row.get("paper_trade_allowed"))
        train_allowed = _bool(admission_row.get("train_allowed"))
        direct_allowed = _bool(row.get("direct_execution_allowed"))

        if not active:
            lifecycle_stage = "inactive"
        elif not collect_allowed:
            lifecycle_stage = "blocked_by_safety_gate"
        elif train_allowed:
            lifecycle_stage = "trainable"
        elif paper_allowed and data_sufficiency >= 1.0:
            lifecycle_stage = "paper_ready_train_review"
        elif paper_allowed:
            lifecycle_stage = "paper_collecting"
        elif data_sufficiency >= 1.0 and quality_score >= 45.0:
            lifecycle_stage = "eligible_review"
        else:
            lifecycle_stage = "collecting"

        promotion_blockers: list[str] = []
        if data_sufficiency < 1.0:
            promotion_blockers.append("minimum_data_floor")
        if quality_score < 50.0:
            promotion_blockers.append("quality_floor")
        if _bool(row.get("training_excluded")):
            promotion_blockers.append("training_excluded_until_threshold")
        if direct_allowed:
            promotion_blockers.append("direct_execution_requires_explicit_live_gate")
        if not promotion_blockers and lifecycle_stage in {"trainable", "paper_ready_train_review"}:
            promotion_blockers.append("none")

        counts[lifecycle_stage] += 1
        rows.append(
            {
                "bot_id": bot_id,
                "sleeve": row.get("sleeve"),
                "lifecycle_stage": lifecycle_stage,
                "quality_score": row.get("quality_score"),
                "data_sufficiency": row.get("data_sufficiency"),
                "promotion_blockers": promotion_blockers,
                "next_gate": "training_readiness_board" if lifecycle_stage in {"eligible_review", "paper_ready_train_review", "trainable"} else "continue_collection",
            }
        )

    rows.sort(key=lambda item: (str(item.get("lifecycle_stage") or ""), -_safe_float(item.get("quality_score"), 0.0), str(item.get("bot_id") or "")))
    return {
        "overall_status": "ready" if rows else "missing",
        "mode": "advisory_read_only",
        "bot_count": len(rows),
        "lifecycle_counts": dict(sorted(counts.items())),
        "sampled_lifecycle": rows[:max_rows],
        "lifecycle_contract": [
            "collecting",
            "eligible_review",
            "paper_collecting",
            "paper_ready_train_review",
            "trainable",
            "promoted_requires_separate_live_gate",
        ],
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
    watch_reasons = []
    if mae_bps >= 35.0:
        watch_reasons.append("calibration_mae_watch")
    if worst_slippage >= 40.0:
        watch_reasons.append("worst_case_slippage_watch")
    managed_reasons = []
    if constrained_curves > 0:
        managed_reasons.append("capacity_curves_constrained_haircut_active")
    severe_reasons = []
    if mae_bps >= 60.0:
        severe_reasons.append("calibration_mae_severe")
    if worst_slippage >= 75.0:
        severe_reasons.append("worst_case_slippage_severe")
    status = _watch_or_needs_work(watch=bool(watch_reasons), hard=bool(severe_reasons))
    if not calibration and not execution_lab:
        status = "missing"
    return {
        "overall_status": status,
        "mae_bps": round(mae_bps, 3),
        "poor_or_fair_fill_count": poor_fills,
        "worst_lab_slippage_bps": round(worst_slippage, 3),
        "constrained_capacity_curve_count": constrained_curves,
        "watch_reasons": watch_reasons,
        "managed_reasons": managed_reasons,
        "capacity_curve_haircut_active": constrained_curves > 0,
        "severe_reasons": severe_reasons,
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


def _provider_rotation_failover(project_root: Path, *, max_rows: int) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    provider_rows: list[dict[str, Any]] = []
    for path in sorted(health_root.glob("data_ingress_latest_*.json")):
        payload = load_json(path)
        counts = _as_dict(payload.get("total_counts"))
        api_ok = _safe_int(counts.get("api_ok"), 0)
        api_error = _safe_int(counts.get("api_error"), 0)
        pause_gate = str(payload.get("pause_gate") or "")
        pause_reason = str(payload.get("pause_reason") or "")
        loop_state = str(payload.get("loop_state") or payload.get("overall_status") or payload.get("status") or "unknown")
        provider_name = "unknown"
        stem = path.stem.replace("data_ingress_latest_", "")
        if "schwab" in stem:
            provider_name = "schwab"
        elif "coinbase" in stem:
            provider_name = "coinbase"
        elif "fx" in stem:
            provider_name = "fx"
        elif "macro" in stem:
            provider_name = "macro"
        elif "options" in stem:
            provider_name = "options"
        provider_status = "ready"
        route = "primary"
        if pause_gate:
            pause_text = f"{pause_gate} {pause_reason}".lower()
            if "cooldown" in pause_text or "403" in pause_text or "429" in pause_text:
                provider_status = "cooldown"
                route = "cache_then_slow_retry"
            elif "session_gate" in pause_text or "weekend" in pause_text or "post_window" in pause_text:
                provider_status = "paused_session_gate"
                route = "session_gate_last_good_cache"
            elif "runtime_training_governor" in pause_text or "training" in pause_text or "host_headroom" in pause_text:
                provider_status = "paused_runtime_training_gate"
                route = "runtime_training_gate_cache"
            else:
                provider_status = "degraded"
                route = "cache_then_slow_retry"
        elif api_error > max(api_ok, 0):
            provider_status = "degraded"
            route = "fallback_cache_or_proxy"
        provider_rows.append(
            {
                "provider": provider_name,
                "source_key": stem,
                "overall_status": provider_status,
                "loop_state": loop_state,
                "pause_gate": pause_gate,
                "pause_reason": pause_reason,
                "api_ok": api_ok,
                "api_error": api_error,
                "failover_route": route,
                "source_file": str(path),
            }
        )

    source_verification = load_json(health_root / "source_verification_latest.json")
    provider_mesh = load_json(health_root / "provider_mesh_latest.json")
    mesh_summary = _as_dict(provider_mesh.get("summary"))
    required_failures = _safe_int(mesh_summary.get("required_failure_count"), len(_as_list(provider_mesh.get("required_failures"))))
    soft_failures = _safe_int(mesh_summary.get("soft_failure_count"), len(_as_list(provider_mesh.get("soft_failures"))))
    source_overall = _as_dict(source_verification.get("overall"))
    unverified_sources = len(_as_list(source_overall.get("unverified_sources")))
    stale_sources = len(_as_list(source_overall.get("stale_sources")))
    mesh_status = str(provider_mesh.get("overall_status") or "missing")
    source_status = str(source_verification.get("overall_status") or "missing")
    degraded = [row for row in provider_rows if str(row.get("overall_status")) in {"degraded", "cooldown"}]
    managed_degraded = [
        row
        for row in degraded
        if required_failures == 0
        and mesh_status == "ready"
        and source_status == "ready"
        and str(row.get("failover_route") or "") in {"fallback_cache_or_proxy", "cache_then_slow_retry"}
    ]
    actionable_degraded = [row for row in degraded if row not in managed_degraded]
    hard_degraded = [row for row in actionable_degraded if str(row.get("overall_status")) == "degraded" and required_failures > 0]
    managed_soft_failures = bool(soft_failures > 0 and required_failures == 0 and mesh_status == "ready" and source_status == "ready")
    provider_counts = Counter(str(row.get("provider") or "unknown") for row in provider_rows)
    routes = {
        "schwab": ["latest_good_cache", "ETF_proxy_context", "provider_http_cooldown"],
        "coinbase": ["latest_good_cache", "slower_snapshot_retry", "crypto_futures_context"],
        "fx": ["currency_ETF_proxy", "latest_good_cache", "macro_context_proxy"],
        "macro": ["cached_calendar", "official_source_retry", "manual_review_queue"],
        "options": ["last_chain_cache", "underlying_quote_proxy", "liquidity_filter_only"],
    }
    if not provider_rows:
        status = "thin"
    elif required_failures > 0 or hard_degraded:
        status = "needs_work"
    elif actionable_degraded:
        status = "watch"
    elif (soft_failures > 0 and not managed_soft_failures) or unverified_sources > 0 or stale_sources > 0:
        status = "watch"
    else:
        status = "ready"
    return {
        "overall_status": status,
        "provider_count": len(provider_rows),
        "provider_counts": dict(sorted(provider_counts.items())),
        "degraded_provider_count": len(degraded),
        "managed_degraded_provider_count": len(managed_degraded),
        "actionable_degraded_provider_count": len(actionable_degraded),
        "required_failure_count": required_failures,
        "soft_failure_count": soft_failures,
        "managed_soft_failure_count": soft_failures if managed_soft_failures else 0,
        "unverified_source_count": unverified_sources,
        "stale_source_count": stale_sources,
        "provider_routes": routes,
        "providers": sorted(provider_rows, key=lambda row: (str(row.get("overall_status") or ""), str(row.get("source_key") or "")))[:max_rows],
        "source_verification_status": source_status,
        "provider_mesh_status": mesh_status,
        "managed_failover_contract": {
            "active": bool((managed_degraded or managed_soft_failures) and not actionable_degraded and required_failures == 0),
            "policy": "cache_or_slow_retry_failover_and_optional_provider_soft_failures_do_not_degrade_guarded_collection",
            "managed_degraded_provider_count": len(managed_degraded),
            "actionable_degraded_provider_count": len(actionable_degraded),
            "managed_soft_failure_count": soft_failures if managed_soft_failures else 0,
        },
        "failover_contract": [
            "403_429_provider_denials_go_to_cooldown_not_global_halt",
            "cache_or_proxy_context_is_allowed_for_collection",
            "paper_and_training_use_source_confidence_before_promotion",
            "live_execution_remains_blocked_without_explicit_gate",
        ],
    }


def _backpressure_prediction(project_root: Path, pressure: dict[str, Any]) -> dict[str, Any]:
    ingestion = _read_health(project_root, "ingestion_storage_control_latest.json")
    backpressure = _as_dict(ingestion.get("backpressure"))
    total_pending = _safe_float(backpressure.get("total_pending_lines"), 0.0)
    pending_threshold = max(_safe_float(backpressure.get("pending_lines_threshold"), 15000.0), 1.0)
    oldest_age = _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0)
    age_threshold = max(_safe_float(backpressure.get("oldest_age_threshold_seconds"), 240.0), 1.0)
    drain_minutes = _safe_float(backpressure.get("estimated_total_drain_minutes"), 0.0)
    host = _safe_float(pressure.get("host_saturation_score"), 0.0)
    pending_ratio = total_pending / pending_threshold
    age_ratio = oldest_age / age_threshold
    pressure_score = max(pending_ratio, age_ratio, host / 100.0)
    forecasts = []
    for horizon in (15, 60):
        if drain_minutes <= horizon and pressure_score < 0.75:
            state = "clear"
        elif pressure_score >= 1.0 or drain_minutes > horizon * 2:
            state = "risk"
        else:
            state = "watch"
        forecasts.append({"horizon_minutes": horizon, "predicted_state": state})
    status = "ready"
    if any(row["predicted_state"] == "risk" for row in forecasts):
        status = "needs_work"
    elif any(row["predicted_state"] == "watch" for row in forecasts):
        status = "watch"
    return {
        "overall_status": status,
        "pending_lines": int(total_pending),
        "pending_ratio": round(pending_ratio, 6),
        "oldest_pending_age_seconds": round(oldest_age, 3),
        "age_ratio": round(age_ratio, 6),
        "estimated_total_drain_minutes": round(drain_minutes, 3),
        "host_saturation_score": round(host, 3),
        "forecasts": forecasts,
        "recommended_policy": "increase_drainers_or_thin_sampling" if status == "needs_work" else ("hold_current_relief" if status == "watch" else "normal"),
        "prediction_contract": [
            "forecast_15_and_60_minute_backpressure",
            "slow_noncritical_jobs_before_global_halt",
            "prefer_collection_drain_over_report_render_jobs",
        ],
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


def _paper_trade_capacity_governor(project_root: Path, pressure: dict[str, Any], admission: dict[str, Any], quality_rows: list[dict[str, Any]]) -> dict[str, Any]:
    paper_ramp = _read_health(project_root, "paper_400_ramp_latest.json")
    counts = _as_dict(admission.get("counts"))
    paper_allowed = _safe_int(counts.get("paper_trade_allowed"), 0)
    active_bots = sum(1 for row in quality_rows if _bool(row.get("active")))
    pressure_status = str(pressure.get("overall_status") or "ready")
    ramp_status = str(paper_ramp.get("overall_status") or paper_ramp.get("status") or "missing")
    if pressure_status == "blocked":
        recommended_now = 0
        ramp_stage = "paused_by_pressure"
    elif pressure_status == "degraded":
        recommended_now = min(max(paper_allowed, 50), 400)
        ramp_stage = "guarded_400_or_less"
    else:
        recommended_now = min(max(paper_allowed, 100), 1000)
        ramp_stage = "eligible_to_scale_in_steps"
    return {
        "overall_status": "ready" if pressure_status != "blocked" else "protect_live",
        "active_bot_count": active_bots,
        "paper_allowed_from_admission": paper_allowed,
        "recommended_max_paper_bots_now": recommended_now,
        "ramp_stage": ramp_stage,
        "paper_400_ramp_status": ramp_status,
        "paper_trade_lock_required": True,
        "live_execution_allowed": False,
        "capacity_contract": [
            "paper_trade_lock_stays_on",
            "paper_count_scales_by_runtime_swap_storage_and_provider_health",
            "new_bots_collect_before_training",
            "live_execution_requires_separate_operator_gate",
        ],
        "source_file": str(project_root / "governance" / "health" / "paper_400_ramp_latest.json"),
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


def _training_readiness_board(quality_rows: list[dict[str, Any]], admission: dict[str, Any], *, max_rows: int) -> dict[str, Any]:
    admission_by_bot = {
        str(row.get("bot_id") or ""): row
        for row in _as_list(admission.get("sampled_admissions"))
        if isinstance(row, dict)
    }
    trainable: list[dict[str, Any]] = []
    sample_debt: list[dict[str, Any]] = []
    excluded_ready: list[dict[str, Any]] = []
    for row in quality_rows:
        data_sufficiency = _safe_float(row.get("data_sufficiency"), 0.0)
        quality_score = _safe_float(row.get("quality_score"), 0.0)
        bot_id = str(row.get("bot_id") or "")
        admission_row = admission_by_bot.get(bot_id, {})
        item = {
            "bot_id": bot_id,
            "sleeve": row.get("sleeve"),
            "quality_score": row.get("quality_score"),
            "data_sufficiency": row.get("data_sufficiency"),
            "observations": row.get("data_collection_observations"),
            "minimum_training_observations": row.get("minimum_training_observations"),
        }
        if _bool(admission_row.get("train_allowed")):
            trainable.append({**item, "training_state": "train_allowed"})
        elif data_sufficiency >= 1.0 and _bool(row.get("training_excluded")):
            excluded_ready.append({**item, "training_state": "ready_but_excluded_until_review"})
        elif data_sufficiency < 1.0:
            sample_debt.append({**item, "training_state": "needs_more_collection", "sample_debt_ratio": round(1.0 - data_sufficiency, 6)})
        elif quality_score < 50.0:
            sample_debt.append({**item, "training_state": "needs_quality_repair", "sample_debt_ratio": 0.0})
    trainable.sort(key=lambda item: (-_safe_float(item.get("quality_score"), 0.0), str(item.get("bot_id") or "")))
    sample_debt.sort(key=lambda item: (-_safe_float(item.get("sample_debt_ratio"), 0.0), str(item.get("bot_id") or "")))
    excluded_ready.sort(key=lambda item: (-_safe_float(item.get("quality_score"), 0.0), str(item.get("bot_id") or "")))
    return {
        "overall_status": "ready" if trainable or sample_debt or excluded_ready else "thin",
        "train_allowed_count": len(trainable),
        "sample_debt_count": len(sample_debt),
        "ready_but_excluded_count": len(excluded_ready),
        "trainable_queue": trainable[:max_rows],
        "sample_debt_queue": sample_debt[:max_rows],
        "ready_but_excluded": excluded_ready[:max_rows],
        "readiness_contract": [
            "minimum_observation_floor",
            "minimum_collection_age",
            "label_quality_and_freshness",
            "runtime_pressure_clear",
            "training_exclusion_lifted_only_after_threshold",
        ],
    }


def _correlation_governor(quality_rows: list[dict[str, Any]], *, max_rows: int) -> dict[str, Any]:
    active_rows = [row for row in quality_rows if not _bool(row.get("ignored_for_active_quality_score"))]
    sleeve_counts = Counter(str(row.get("sleeve") or "default") for row in active_rows)
    dependency_counts: Counter[str] = Counter()
    peer_counts: Counter[str] = Counter()
    for row in active_rows:
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
    direct_live_rows = [row for row in active_rows if _bool(row.get("direct_execution_allowed"))]
    managed_concentration = bool(concentration >= 0.30 and not direct_live_rows and not overloaded_dependencies)
    if direct_live_rows and (concentration >= 0.30 or overloaded_dependencies):
        status = "needs_work"
    elif overloaded_dependencies:
        status = "watch"
    elif concentration >= 0.30 and not managed_concentration:
        status = "watch"
    return {
        "overall_status": status,
        "sleeve_concentration": round(concentration, 6),
        "direct_live_candidate_count": len(direct_live_rows),
        "managed_concentration_contract": {
            "active": managed_concentration,
            "policy": "read_only_collect_only_sleeve_concentration_blocks_promotion_not_guarded_collection",
            "raw_sleeve_concentration": round(concentration, 6),
        },
        "largest_sleeves": [{"sleeve": key, "bot_count": count} for key, count in sleeve_counts.most_common(max_rows)],
        "overloaded_correlation_dependencies": overloaded_dependencies[:max_rows],
        "top_peer_sleeves": [{"peer_sleeve": key, "bot_count": count} for key, count in peer_counts.most_common(max_rows)],
        "governor_contract": [
            "detect hidden same-bet concentration",
            "penalize overloaded correlation dependencies",
            "route sleeve masters through correlation heat before promotion",
        ],
    }


def _duplicate_alpha_overlap_detector(quality_rows: list[dict[str, Any]], *, max_rows: int) -> dict[str, Any]:
    clusters: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in quality_rows:
        if _bool(row.get("ignored_for_active_quality_score")):
            continue
        sleeve = str(row.get("sleeve") or "default")
        targets = ",".join(sorted(str(item) for item in _as_list(row.get("target_functions"))[:4]))
        deps = ",".join(sorted(str(item) for item in _as_list(row.get("correlation_dependencies"))[:4]))
        intakes = ",".join(sorted(str(item) for item in _as_list(row.get("data_intake_collections"))[:4]))
        key = (sleeve, targets or "no_target_contract", deps or intakes or "no_dependency_contract")
        clusters[key].append(row)
    overlap_rows = []
    for (sleeve, target_key, dependency_key), rows in clusters.items():
        if len(rows) < 2:
            continue
        avg_quality = sum(_safe_float(row.get("quality_score"), 0.0) for row in rows) / max(len(rows), 1)
        direct_cluster = any(_bool(row.get("direct_execution_allowed")) for row in rows)
        target_set = {item for item in target_key.split(",") if item}
        generic_paper_collection_cluster = bool(
            not direct_cluster
            and {"paper_live_data_standard", "paper_trade_lock", "data_collection_floor"}.issubset(target_set)
            and dependency_key == "no_dependency_contract"
        )
        managed_cluster = bool(
            not direct_cluster
            and (
                all(_bool(row.get("managed_quality_debt")) for row in rows)
                or all(_bool(row.get("generic_paper_collection_contract")) for row in rows)
                or generic_paper_collection_cluster
            )
        )
        if direct_cluster:
            managed_reason = ""
        elif generic_paper_collection_cluster:
            managed_reason = "generic_paper_live_collection_contract"
        elif managed_cluster:
            managed_reason = "collect_only_maturity_contract"
        else:
            managed_reason = ""
        overlap_rows.append(
            {
                "sleeve": sleeve,
                "cluster_size": len(rows),
                "target_key": target_key,
                "dependency_key": dependency_key,
                "average_quality_score": round(avg_quality, 3),
                "sample_bots": [str(row.get("bot_id") or "") for row in sorted(rows, key=lambda item: str(item.get("bot_id") or ""))[:max_rows]],
                "overlap_risk": "high" if len(rows) >= 6 else "medium",
                "managed_by_collect_only_novelty_contract": managed_cluster,
                "managed_overlap_reason": managed_reason,
                "direct_execution_overlap": direct_cluster,
            }
        )
    overlap_rows.sort(key=lambda item: (-_safe_int(item.get("cluster_size"), 0), str(item.get("sleeve") or "")))
    status = "ready"
    high_overlap_rows = [row for row in overlap_rows if str(row.get("overlap_risk")) == "high"]
    direct_overlap = [row for row in overlap_rows if _bool(row.get("direct_execution_overlap"))]
    managed_overlap = [row for row in overlap_rows if _bool(row.get("managed_by_collect_only_novelty_contract"))]
    actionable_overlap = [row for row in overlap_rows if row not in managed_overlap]
    if direct_overlap:
        status = "needs_work"
    elif [row for row in high_overlap_rows if row not in managed_overlap]:
        status = "watch"
    elif actionable_overlap:
        status = "watch"
    return {
        "overall_status": status,
        "overlap_cluster_count": len(overlap_rows),
        "high_overlap_cluster_count": len(high_overlap_rows),
        "direct_execution_overlap_cluster_count": len(direct_overlap),
        "managed_overlap_cluster_count": len(managed_overlap),
        "actionable_overlap_cluster_count": len(actionable_overlap),
        "overlap_clusters": overlap_rows[:max_rows],
        "novelty_contract": [
            "compare_sleeve_target_functions_and_correlation_dependencies",
            "penalize_duplicate_alpha_before_promotion",
            "prefer_new_information_under_capacity_pressure",
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
    direct_decaying = [
        row
        for row in quality_rows
        if _bool(row.get("direct_execution_allowed"))
        and (_safe_int(row.get("no_improvement_streak"), 0) >= 3 or str(row.get("sleeve") or "") in weak_sleeves)
    ]
    status = "ready"
    if direct_decaying:
        status = "needs_work"
    if not decay:
        status = "thin"
    return {
        "overall_status": status,
        "decay_monitor_status": str(decay.get("overall_status") or "missing"),
        "decaying_bot_count": len(decaying),
        "managed_read_only_decaying_bot_count": len(decaying) - len(direct_decaying),
        "direct_execution_decaying_count": len(direct_decaying),
        "decaying_bot_ratio": round(len(decaying) / max(len(quality_rows), 1), 6),
        "managed_decay_contract": {
            "active": bool(decaying and not direct_decaying),
            "policy": "read_only_decay_debt_routes_to_training_or_requalification_without_degrading_guarded_collection",
        },
        "weak_sleeves": sorted(weak_sleeves),
        "decaying_bots": decaying[:max_rows],
        "source_file": str(project_root / "governance" / "research" / "decay_monitor_latest.json"),
    }


def _self_healing_incident_playbooks(project_root: Path, pressure: dict[str, Any], provider: dict[str, Any], backpressure: dict[str, Any], *, max_rows: int) -> dict[str, Any]:
    halt = _read_health(project_root, "global_halt_auto_clear_latest.json") or _read_health(project_root, "global_killswitch_latest.json")
    process = _read_health(project_root, "process_watchdog_latest.json")
    auth = _read_health(project_root, "auth_lease_manager_latest.json")
    storage = _read_health(project_root, "ingestion_storage_control_latest.json")
    provider_degraded = _safe_int(provider.get("actionable_degraded_provider_count"), _safe_int(provider.get("degraded_provider_count"), 0)) > 0
    managed_provider_failover = bool(_as_dict(provider.get("managed_failover_contract")).get("active", False))
    backpressure_risk = str(backpressure.get("overall_status") or "") in {"needs_work", "watch"}
    alerts = _as_list(process.get("alerts"))
    auth_state = str(auth.get("overall_status") or auth.get("lease_state") or auth.get("auth_state") or "").lower()
    playbooks = [
        {
            "id": "safe_global_halt_clear",
            "triggered": _bool(halt.get("halt")) and _bool(halt.get("clear_ready")),
            "auto_allowed": True,
            "command": "./scripts/ops/opsctl.sh global-halt-auto-clear --json",
            "purpose": "clear only when blockers are already gone",
        },
        {
            "id": "provider_cooldown_route",
            "triggered": provider_degraded,
            "auto_allowed": True,
            "command": "./scripts/ops/opsctl.sh pressure-relief --apply --json",
            "purpose": "route provider denials to cooldown/fallback collection",
        },
        {
            "id": "backpressure_drain",
            "triggered": backpressure_risk,
            "auto_allowed": True,
            "command": "./scripts/ops/opsctl.sh storage-backpressure-autopilot --apply --json",
            "purpose": "drain queues before age or pending SLOs trip",
        },
        {
            "id": "auth_lease_refresh",
            "triggered": auth_state in {"warning", "degraded", "critical", "expired"},
            "auto_allowed": False,
            "command": "./scripts/ops/opsctl.sh token-refresh --json",
            "purpose": "refresh credentials through the guarded auth path",
        },
        {
            "id": "process_watchdog_repair",
            "triggered": bool(alerts),
            "auto_allowed": True,
            "command": "./scripts/ops/opsctl.sh infrastructure-autofix --apply --json",
            "purpose": "repair stale or missing launch/watchdog lanes",
        },
        {
            "id": "storage_pressure_clearance",
            "triggered": str(storage.get("overall_status") or storage.get("severity") or "").lower() in {"degraded", "needs_work", "blocked", "critical"},
            "auto_allowed": True,
            "command": "./scripts/ops/opsctl.sh storage-pressure-clearance --apply --json",
            "purpose": "checkpoint/drain/prune storage pressure without deleting live data",
        },
    ]
    triggered = [row for row in playbooks if _bool(row.get("triggered"))]
    manual_triggered = [row for row in triggered if not _bool(row.get("auto_allowed"))]
    status = _watch_or_needs_work(watch=bool(triggered), hard=bool(manual_triggered))
    return {
        "overall_status": status,
        "playbook_count": len(playbooks),
        "triggered_count": len(triggered),
        "manual_triggered_count": len(manual_triggered),
        "managed_provider_failover_active": managed_provider_failover,
        "triggered_playbooks": triggered[:max_rows],
        "available_playbooks": playbooks,
        "incident_contract": [
            "prefer_safe_autoclear_over_manual_flag_removal",
            "separate_provider_denial_from_system_failure",
            "run_storage_and_process_repairs_before escalating global halt",
            "auth_refresh_remains_operator_visible",
        ],
    }


def _black_box_recorder(project_root: Path, sections: dict[str, dict[str, Any]], pressure: dict[str, Any], *, max_rows: int) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    candidates = sorted(
        [path for path in health_root.glob("*.json") if path.is_file()],
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )[: max(max_rows, 8)]
    captured = []
    for path in candidates:
        payload = load_json(path)
        captured.append(
            {
                "name": path.name,
                "timestamp_utc": payload.get("timestamp_utc") or payload.get("updated_at_utc") or "",
                "overall_status": payload.get("overall_status") or payload.get("status") or payload.get("halt_state") or "",
                "loop_state": payload.get("loop_state") or "",
                "pause_gate": payload.get("pause_gate") or "",
                "pause_reason": payload.get("pause_reason") or "",
                "path": str(path),
            }
        )
    section_statuses = {
        key: str(value.get("overall_status") or "missing")
        for key, value in sections.items()
        if isinstance(value, dict)
    }
    return {
        "overall_status": "ready" if captured else "thin",
        "mode": "latest_artifact_snapshot_no_heavy_tail",
        "captured_file_count": len(captured),
        "captured_files": captured,
        "pressure_snapshot": pressure,
        "section_statuses": section_statuses,
        "black_box_contract": [
            "capture_latest_health_artifacts_without_starting_report_jobs",
            "preserve_halt_provider_pressure_and_queue_context",
            "support_incident_replay_and_post_trade_review",
        ],
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
    elif any(str(row.get("overall_status")) == "needs_work" for row in rows):
        overall_status = "needs_work"
    elif any(str(row.get("overall_status")) in {"watch", "thin"} for row in rows):
        overall_status = "watch"
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


def _env_overrides(payload: dict[str, Any]) -> dict[str, str]:
    sections = _as_dict(payload.get("sections"))
    pressure = _as_dict(payload.get("pressure_snapshot"))
    paper = _as_dict(sections.get("paper_trade_capacity_governor"))
    env: dict[str, str] = {
        "PLATFORM_INTELLIGENCE_ENABLED": "1",
        "PLATFORM_INTELLIGENCE_LAYER_VERSION": "2",
        "PLATFORM_INTELLIGENCE_PRIMARY_SECTION_COUNT": str(len(PRIMARY_SECTION_KEYS)),
        "PLATFORM_INTELLIGENCE_READ_ONLY": "1",
        "PRIMARY_ML_RUNTIME_BACKEND": "mlx",
        "LIBRARY_DEFAULT_ML_BACKEND": "mlx",
        "BACKPRESSURE_PREDICTION_HORIZON_MINUTES": "15,60",
        "BLACK_BOX_RECORDER_MAX_SOURCE_FILES": "24",
        "BOT_LIFECYCLE_MIN_TRAINING_OBSERVATIONS_DEFAULT": "1000",
        "PAPER_TRADE_CAPACITY_TARGET": str(_safe_int(paper.get("recommended_max_paper_bots_now"), 0)),
        "PAPER_TRADE_LOCK": "1",
        "ALLOW_ORDER_EXECUTION": "0",
        "PROVIDER_FAILOVER_SCHWAB_403_429_COOLDOWN_SECONDS": "180",
        "PLATFORM_INTELLIGENCE_PRESSURE_POLICY": str(pressure.get("compute_policy") or "normal"),
    }
    for control in PLATFORM_INTELLIGENCE_CONTROLS:
        env[str(control["env_key"])] = "1"
    return env


def _write_env_override(path: Path, env: dict[str, str]) -> bool:
    lines = ["# Auto-managed by scripts/ops/platform_intelligence_expansion.py"]
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
        "schema_version": 2,
        "updated_utc": payload.get("timestamp_utc"),
        "layer": "platform_intelligence_12_part_control_plane",
        "primary_section_keys": list(PRIMARY_SECTION_KEYS),
        "controls": PLATFORM_INTELLIGENCE_CONTROLS,
        "artifacts": payload.get("section_artifacts", {}),
        "recommended_commands": payload.get("recommended_commands", []),
    }
    content = json.dumps(config, ensure_ascii=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def build_payload(project_root: Path = PROJECT_ROOT, *, max_rows: int = 25) -> dict[str, Any]:
    bots = _load_registry(project_root)
    pressure = _pressure_snapshot(project_root)
    quality_rows = [_bot_quality_row(row, sleeve=_infer_sleeve(row)) for row in bots if _normalize_id(row.get("bot_id"))]

    quality = _quality_system(quality_rows, max_rows=max_rows)
    admission = _admission_controller(quality_rows, pressure, max_rows=max_rows)
    lifecycle = _bot_lifecycle_manager(quality_rows, admission, max_rows=max_rows)
    provider_failover = _provider_rotation_failover(project_root, max_rows=max_rows)
    backpressure_prediction = _backpressure_prediction(project_root, pressure)
    duplicate_alpha = _duplicate_alpha_overlap_detector(quality_rows, max_rows=max_rows)
    paper_capacity = _paper_trade_capacity_governor(project_root, pressure, admission, quality_rows)
    self_healing = _self_healing_incident_playbooks(
        project_root,
        pressure,
        provider_failover,
        backpressure_prediction,
        max_rows=max_rows,
    )
    sleeve_masters = _sleeve_masters(quality_rows, pressure, max_rows=max_rows)
    training_readiness = _training_readiness_board(quality_rows, admission, max_rows=max_rows)
    execution_realism = _execution_realism(project_root)
    regime_router = _market_regime_router(project_root, sleeve_masters)
    capacity_planner = _capacity_planner(pressure, admission)
    research_pipeline = _research_pipeline(quality_rows, max_rows=max_rows)
    correlation = _correlation_governor(quality_rows, max_rows=max_rows)
    decay = _model_decay(project_root, quality_rows, max_rows=max_rows)

    sections = {
        "bot_lifecycle_manager": lifecycle,
        "bot_data_quality_scores": quality,
        "provider_rotation_failover_mesh": provider_failover,
        "backpressure_prediction_engine": backpressure_prediction,
        "duplicate_alpha_overlap_detector": duplicate_alpha,
        "paper_trade_capacity_governor": paper_capacity,
        "self_healing_incident_playbooks": self_healing,
        "per_sleeve_master_bots": sleeve_masters,
        "market_regime_router": regime_router,
        "training_readiness_board": training_readiness,
        "execution_paper_trade_realism_layer": execution_realism,
        "bot_admission_controller": admission,
        "bot_quality_score_system": quality,
        "execution_realism_engine": execution_realism,
        "swap_cpu_capacity_planner": capacity_planner,
        "research_to_strategy_pipeline": research_pipeline,
        "cross_sleeve_correlation_governor": correlation,
        "model_decay_detector": decay,
    }
    sections["system_black_box_recorder"] = _black_box_recorder(project_root, sections, pressure, max_rows=max_rows)
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
    if provider_failover.get("overall_status") not in {"ready", "thin"}:
        top_actions.append("let provider failover/cooldown absorb source denials before restarting sleeves")
    if backpressure_prediction.get("overall_status") != "ready":
        top_actions.append("run backpressure drain before increasing paper-trade fanout")
    if not top_actions:
        top_actions.append("keep the 12-layer platform-intelligence control plane in the dashboard refresh path as the fleet grows")

    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": dashboard["overall_status"] in {"ready", "watch", "needs_work", "degraded", "blocked"},
        "overall_status": dashboard["overall_status"],
        "mode": "advisory_read_only_no_new_runtime_loops",
        "expansion_count": len(PRIMARY_SECTION_KEYS),
        "primary_section_keys": list(PRIMARY_SECTION_KEYS),
        "primary_sections": {key: sections[key] for key in PRIMARY_SECTION_KEYS if key in sections},
        "control_count": len(PLATFORM_INTELLIGENCE_CONTROLS),
        "controls": [
            {**control, "enabled": True}
            for control in PLATFORM_INTELLIGENCE_CONTROLS
        ],
        "bot_count": len(quality_rows),
        "sleeve_count": dashboard["sleeve_count"],
        "pressure_snapshot": pressure,
        "sections": sections,
        "top_actions": top_actions,
        "recommended_env_overrides": {},
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "platform-intelligence", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "health-fast", "--json"],
            ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-backpressure-autopilot", "--apply", "--json"],
        ],
        "source_files": {
            "master_bot_registry": str(project_root / "master_bot_registry.json"),
            "primary_artifact": str(project_root / "governance" / "health" / "platform_intelligence_expansion_latest.json"),
        },
    }
    payload["recommended_env_overrides"] = _env_overrides(payload)
    return payload


def write_section_artifacts(project_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    sections = _as_dict(payload.get("sections"))
    platform_intel_root = project_root / "governance" / "platform_intelligence"
    paths = {
        "bot_lifecycle_manager": platform_intel_root / "bot_lifecycle_manager_latest.json",
        "bot_data_quality_scores": platform_intel_root / "bot_data_quality_scores_latest.json",
        "provider_rotation_failover_mesh": platform_intel_root / "provider_rotation_failover_latest.json",
        "backpressure_prediction_engine": platform_intel_root / "backpressure_prediction_latest.json",
        "duplicate_alpha_overlap_detector": platform_intel_root / "duplicate_alpha_overlap_latest.json",
        "paper_trade_capacity_governor": platform_intel_root / "paper_trade_capacity_governor_latest.json",
        "self_healing_incident_playbooks": platform_intel_root / "self_healing_incident_playbooks_latest.json",
        "training_readiness_board": platform_intel_root / "training_readiness_board_latest.json",
        "execution_paper_trade_realism_layer": platform_intel_root / "execution_paper_trade_realism_latest.json",
        "system_black_box_recorder": platform_intel_root / "black_box_recorder_latest.json",
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
    parser = argparse.ArgumentParser(description="Build the 12-part platform intelligence control plane.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--max-rows", type=int, default=25)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root, max_rows=max(int(args.max_rows), 1))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    written = write_section_artifacts(project_root, payload)
    payload["section_artifacts"] = written
    if args.apply:
        env = _as_dict(payload.get("recommended_env_overrides"))
        payload["apply_result"] = {
            "applied": True,
            "override_path": str(Path(args.override_file).expanduser()),
            "override_changed": _write_env_override(Path(args.override_file).expanduser(), {str(k): str(v) for k, v in env.items()}),
            "config_path": str(Path(args.config_file).expanduser()),
            "config_changed": _write_config(Path(args.config_file).expanduser(), payload),
        }
    else:
        payload["apply_result"] = {"applied": False}
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
