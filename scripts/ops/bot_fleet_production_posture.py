#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "bot_fleet_production_posture_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.bot_fleet_production_posture_override"
SCHEMA_VERSION = 1
CONTROL_VERSION = "bot_fleet_production_posture_v1"
GOOD_STATUSES = {"ready", "ok", "advisory", "needs_work"}
BAD_LIFECYCLES = {"deleted", "retired", "archived"}
ACTIVE_LIFECYCLES = {"active", "paper_live_data", "data_collection_only"}

ARTIFACT_SPECS: dict[str, tuple[str, float, bool]] = {
    "registry": ("master_bot_registry.json", 1440.0, True),
    "paper_standard": ("governance/health/paper_live_data_standard_latest.json", 240.0, True),
    "observation_rollup": ("governance/health/data_collection_observation_rollup_latest.json", 240.0, True),
    "health_fast": ("governance/health/health_fast_latest.json", 90.0, True),
    "sleeve_ingestion": ("governance/health/sleeve_ingestion_production_control_latest.json", 240.0, True),
    "bot_quality": ("governance/health/bot_quality_autopilot_latest.json", 240.0, True),
    "bot_mesh": ("governance/health/bot_intelligence_mesh_latest.json", 240.0, True),
    "training_quality": ("governance/health/training_quality_control_latest.json", 240.0, True),
    "supportability": ("governance/health/supportability_control_latest.json", 240.0, True),
    "teacher_quality": ("governance/distillation/teacher_quality_latest.json", 240.0, True),
    "overfitting_awareness": ("governance/health/overfitting_awareness_latest.json", 240.0, True),
}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


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


def _status(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _grade(score: float) -> str:
    if score >= 98.0:
        return "A+"
    if score >= 94.0:
        return "A"
    if score >= 90.0:
        return "A-"
    if score >= 85.0:
        return "B+"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    return "D"


def _bot_id(row: dict[str, Any]) -> str:
    return str(row.get("bot_id") or row.get("id") or row.get("name") or "").strip()


def _lifecycle(row: dict[str, Any]) -> str:
    return str(row.get("lifecycle_state") or "").strip().lower()


def _is_deleted(row: dict[str, Any]) -> bool:
    return bool(row.get("deleted", False) or row.get("deleted_from_rotation", False) or _lifecycle(row) in BAD_LIFECYCLES)


def _is_non_deleted(row: dict[str, Any]) -> bool:
    return not _is_deleted(row)


def _is_active(row: dict[str, Any]) -> bool:
    return bool(_is_non_deleted(row) and row.get("active", False))


def _has_label_contract(row: dict[str, Any]) -> bool:
    return isinstance(row.get("label_contract"), dict) or isinstance(row.get("universal_label_contract"), dict)


def _has_target_functions(row: dict[str, Any]) -> bool:
    return bool([item for item in _as_list(row.get("target_functions")) if str(item or "").strip()])


def _has_live_authority(row: dict[str, Any]) -> bool:
    return any(
        bool(row.get(key, False))
        for key in (
            "direct_execution_allowed",
            "trading_enabled",
            "live_trading_enabled",
            "execution_enabled",
            "allocation_enabled",
        )
    )


def _load_sources(project_root: Path) -> dict[str, dict[str, Any]]:
    return {name: load_json(project_root / rel_path) for name, (rel_path, _max_age, _required) in ARTIFACT_SPECS.items()}


def _source_freshness_contract(project_root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stale_or_missing: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    for name, (rel_path, max_age_minutes, required) in ARTIFACT_SPECS.items():
        path = project_root / rel_path
        payload = _as_dict(sources.get(name))
        age = payload_age_minutes(payload, path)
        loaded = bool(payload)
        fresh = bool(loaded and age is not None and float(age) <= float(max_age_minutes))
        if required and not fresh:
            stale_or_missing.append(name)
        rows[name] = {
            "path": rel_path,
            "loaded": loaded,
            "required": bool(required),
            "fresh": fresh,
            "age_minutes": round(float(age), 3) if age is not None else None,
            "max_age_minutes": float(max_age_minutes),
        }
    return {
        "all_required_fresh": not stale_or_missing,
        "stale_or_missing": stale_or_missing,
        "sources": rows,
    }


def _registry_contract(registry: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in _as_list(registry.get("sub_bots")) if isinstance(row, dict)]
    non_deleted = [row for row in rows if _is_non_deleted(row)]
    active = [row for row in non_deleted if bool(row.get("active", False))]
    deleted = [row for row in rows if _is_deleted(row)]
    role_counts = Counter(str(row.get("bot_role") or "missing").strip() or "missing" for row in non_deleted)
    lifecycle_counts = Counter(str(row.get("lifecycle_state") or "missing").strip() or "missing" for row in non_deleted)
    missing_bot_ids = [_bot_id(row) or "<missing>" for row in non_deleted if not _bot_id(row)]
    missing_roles = [_bot_id(row) for row in non_deleted if not str(row.get("bot_role") or "").strip()]
    missing_labels = [_bot_id(row) for row in non_deleted if not _has_label_contract(row)]
    missing_targets = [_bot_id(row) for row in non_deleted if not _has_target_functions(row)]
    unsupported_lifecycles = [_bot_id(row) for row in non_deleted if _lifecycle(row) and _lifecycle(row) not in ACTIVE_LIFECYCLES]
    inactive_non_deleted = [_bot_id(row) for row in non_deleted if not bool(row.get("active", False))]
    non_collecting = [_bot_id(row) for row in non_deleted if not bool(row.get("data_collection_active", False))]
    live_authority = [_bot_id(row) for row in non_deleted if _has_live_authority(row)]
    deleted_active = [_bot_id(row) for row in deleted if bool(row.get("active", False))]
    paper_enabled = [
        row
        for row in non_deleted
        if bool(row.get("paper_live_data_enabled", False) or row.get("paper_trading_enabled", False) or row.get("paper_trade_enabled", False))
    ]
    return {
        "total_bots": len(rows),
        "non_deleted_bots": len(non_deleted),
        "active_bots": len(active),
        "deleted_or_retired_bots": len(deleted),
        "paper_live_data_enabled_bots": len(paper_enabled),
        "data_collection_active_bots": len([row for row in non_deleted if bool(row.get("data_collection_active", False))]),
        "inactive_non_deleted_count": len(inactive_non_deleted),
        "inactive_non_deleted_sample": inactive_non_deleted[:20],
        "missing_bot_id_count": len(missing_bot_ids),
        "missing_bot_id_sample": missing_bot_ids[:20],
        "missing_role_count": len(missing_roles),
        "missing_role_sample": missing_roles[:20],
        "missing_label_contract_count": len(missing_labels),
        "missing_label_contract_sample": missing_labels[:20],
        "missing_target_functions_count": len(missing_targets),
        "missing_target_functions_sample": missing_targets[:20],
        "unsupported_lifecycle_count": len(unsupported_lifecycles),
        "unsupported_lifecycle_sample": unsupported_lifecycles[:20],
        "non_collecting_count": len(non_collecting),
        "non_collecting_sample": non_collecting[:20],
        "live_authority_count": len(live_authority),
        "live_authority_sample": live_authority[:20],
        "deleted_active_count": len(deleted_active),
        "deleted_active_sample": deleted_active[:20],
        "role_counts": dict(role_counts.most_common()),
        "lifecycle_counts": dict(lifecycle_counts.most_common()),
        "identity_contract_ready": bool(not missing_bot_ids and not missing_roles),
        "label_contract_ready": bool(not missing_labels),
        "target_function_contract_ready": bool(not missing_targets),
        "all_non_deleted_active": bool(non_deleted and len(active) == len(non_deleted)),
        "all_non_deleted_collecting": bool(non_deleted and not non_collecting),
        "deleted_rows_quarantined": bool(not deleted_active),
        "live_authority_absent": bool(not live_authority),
        "supported_lifecycles": bool(not unsupported_lifecycles),
    }


def _paper_contract(paper_standard: dict[str, Any]) -> dict[str, Any]:
    counts = _as_dict(paper_standard.get("counts_after"))
    safety = _as_dict(paper_standard.get("safety_contract"))
    direct = _safe_int(counts.get("direct_execution_allowed_bots"), 0)
    live = _safe_int(counts.get("live_trading_enabled_bots"), 0)
    live_locked = bool(
        direct <= 0
        and live <= 0
        and str(safety.get("allow_order_execution") or "0") == "0"
        and str(safety.get("market_data_only") or "1") == "1"
        and safety.get("live_execution_allowed") is False
    )
    return {
        "status": str(paper_standard.get("overall_status") or "missing"),
        "ok": bool(paper_standard.get("ok", False)),
        "non_deleted_bots": _safe_int(counts.get("non_deleted_bots"), 0),
        "data_collection_active_bots": _safe_int(counts.get("data_collection_active_bots"), 0),
        "paper_live_data_enabled_bots": _safe_int(counts.get("paper_live_data_enabled_bots"), 0),
        "collection_until_standard_bots": _safe_int(counts.get("collection_until_standard_bots"), 0),
        "direct_execution_allowed_bots": direct,
        "live_trading_enabled_bots": live,
        "live_execution_locked": live_locked,
        "paper_mirror_all_active_sub_bots": str(safety.get("paper_mirror_all_active_sub_bots") or ""),
        "paper_lock": str(safety.get("paper_trade_lock") or ""),
        "market_data_only": str(safety.get("market_data_only") or "1"),
        "allow_order_execution": str(safety.get("allow_order_execution") or "0"),
    }


def _collection_contract(rollup: dict[str, Any]) -> dict[str, Any]:
    collectors = _safe_int(rollup.get("collector_count"), 0)
    observed = _safe_int(rollup.get("effective_bots_with_observations"), _safe_int(rollup.get("bots_with_observations"), 0))
    unmanaged_zero = _safe_int(rollup.get("unmanaged_zero_observation_count"), _safe_int(rollup.get("zero_observation_count"), 0))
    return {
        "status": str(rollup.get("overall_status") or "missing"),
        "collector_count": collectors,
        "effective_bots_with_observations": observed,
        "unmanaged_zero_observation_count": unmanaged_zero,
        "total_observations": _safe_int(rollup.get("total_observations"), 0),
        "training_ready_count": _safe_int(rollup.get("training_ready_count"), 0),
        "coverage_score": _safe_float(rollup.get("collection_coverage_score"), 0.0),
        "data_quality_score": _safe_float(rollup.get("data_quality_score"), 0.0),
        "coverage_ready": bool(collectors > 0 and observed >= collectors and unmanaged_zero <= 0),
    }


def _runtime_contract(health_fast: dict[str, Any]) -> dict[str, Any]:
    readiness = _as_dict(health_fast.get("operational_readiness"))
    guarded = _as_dict(readiness.get("guarded_paper"))
    watchdog = _as_dict(health_fast.get("process_watchdog"))
    all_sleeves = _as_dict(watchdog.get("all_sleeves_effective_runtime"))
    return {
        "health_fast_status": str(health_fast.get("overall_status") or "missing"),
        "health_fast_ok": bool(health_fast.get("ok", False)),
        "guarded_paper_status": str(guarded.get("status") or ""),
        "guarded_paper_ok": bool(guarded.get("ok", False)),
        "guarded_blockers": [str(item) for item in _as_list(guarded.get("blockers"))],
        "paper_ramp_stage": str(guarded.get("paper_ramp_stage") or ""),
        "all_sleeves_status": str(all_sleeves.get("status") or ""),
        "all_sleeves_ready": bool(all_sleeves.get("ok", False) or str(all_sleeves.get("status") or "") == "ready"),
        "child_process_count": _safe_int(all_sleeves.get("child_process_count"), 0),
        "child_fanout_ok": bool(all_sleeves.get("child_fanout_ok", False)),
        "heartbeat_ok": bool(all_sleeves.get("heartbeat_ok", False)),
    }


def _mesh_contract(mesh: dict[str, Any]) -> dict[str, Any]:
    hierarchy = _as_dict(mesh.get("hierarchy_edge_summary"))
    quality = _as_dict(mesh.get("a_plus_target_contract"))
    quality_summary = _as_dict(quality.get("summary"))
    return {
        "status": str(mesh.get("overall_status") or "missing"),
        "communication_readiness_score": _safe_float(mesh.get("communication_readiness_score"), 0.0),
        "quality_readiness_score": _safe_float(mesh.get("quality_readiness_score"), 0.0),
        "quality_target_status": str(mesh.get("quality_target_status") or ""),
        "bot_count": _safe_int(mesh.get("bot_count"), 0),
        "active_bot_count": _safe_int(mesh.get("active_bot_count"), 0),
        "missing_tiers": [str(item) for item in _as_list(mesh.get("missing_tiers"))],
        "edge_count_total": _safe_int(hierarchy.get("edge_count_total"), 0),
        "active_sub_or_infra_route_ratio": _safe_float(hierarchy.get("active_sub_or_infra_route_ratio"), 0.0),
        "active_master_route_ratio": _safe_float(hierarchy.get("active_master_route_ratio"), 0.0),
        "quality_blocker_count": _safe_int(quality.get("blocker_count"), 0),
        "training_readiness_gap": _safe_int(quality_summary.get("training_ready_gap"), 0),
        "route_ready": bool(
            not _as_list(mesh.get("missing_tiers"))
            and _safe_float(hierarchy.get("active_sub_or_infra_route_ratio"), 0.0) >= 0.99
            and _safe_float(hierarchy.get("active_master_route_ratio"), 0.0) >= 0.99
        ),
    }


def _quality_lane_contract(bot_quality: dict[str, Any], training_quality: dict[str, Any], supportability: dict[str, Any]) -> dict[str, Any]:
    blockers = _as_dict(bot_quality.get("quality_blockers"))
    targeted = _as_dict(training_quality.get("targeted_actions"))
    attempts = [_as_dict(row) for row in _as_list(bot_quality.get("attempts"))]
    hard_failed_attempts: list[dict[str, Any]] = []
    for row in attempts:
        if bool(row.get("timed_out", False)):
            hard_failed_attempts.append(row)
            continue
        if "rc" not in row or str(row.get("rc") or "").strip() == "":
            continue
        if _safe_int(row.get("rc"), 0) != 0:
            hard_failed_attempts.append(row)
    repair_runtime = [str(item) for item in _as_list(blockers.get("repair_runtime_input_bot_ids"))]
    students_without_teachers = _safe_int(blockers.get("students_without_teachers"), _safe_int(supportability.get("students_without_teachers"), 0))
    coverage_shortfall = _safe_int(blockers.get("coverage_shortfall_bots"), 0)
    hard_quality_debt = bool(repair_runtime or students_without_teachers > 0 or hard_failed_attempts)
    planned_queue_count = _safe_int(blockers.get("planned_queue_count"), len(_as_list(bot_quality.get("quality_upgrade_queue"))))
    selected_retrain = [str(item) for item in _as_list(targeted.get("selected_targeted_retrain_bot_ids"))]
    precompute = [str(item) for item in _as_list(targeted.get("precompute_target_bot_ids"))]
    refresh_diagnostics = [str(item) for item in _as_list(blockers.get("refresh_diagnostics_bot_ids"))]
    weak_sleeves = [_as_dict(row) for row in _as_list(targeted.get("weak_sleeves"))]
    return {
        "status": str(bot_quality.get("overall_status") or "missing"),
        "training_quality_status": str(training_quality.get("overall_status") or "missing"),
        "training_quality_score": _safe_float(training_quality.get("training_quality_score"), 0.0),
        "planned_queue_count": planned_queue_count,
        "refresh_diagnostics_bot_count": len(refresh_diagnostics),
        "refresh_diagnostics_bot_ids": refresh_diagnostics[:20],
        "repair_runtime_input_bot_count": len(repair_runtime),
        "repair_runtime_input_bot_ids": repair_runtime[:20],
        "students_without_teachers": students_without_teachers,
        "coverage_shortfall_bots": coverage_shortfall,
        "hard_failed_attempt_count": len(hard_failed_attempts),
        "selected_targeted_retrain_bot_count": len(selected_retrain),
        "selected_targeted_retrain_bot_ids": selected_retrain[:20],
        "precompute_target_bot_count": len(precompute),
        "precompute_target_bot_ids": precompute[:20],
        "weak_sleeve_count": len(weak_sleeves),
        "weak_sleeves": weak_sleeves[:12],
        "top_label_actions": [str(item) for item in _as_list(targeted.get("top_label_actions"))[:12]],
        "debt_owned": bool(not hard_quality_debt and _status(bot_quality.get("overall_status")) in GOOD_STATUSES),
        "quality_debt_mode": "owned_repair_lanes" if not hard_quality_debt else "hard_repair_required",
    }


def _teacher_contract(teacher_quality: dict[str, Any]) -> dict[str, Any]:
    summary = _as_dict(teacher_quality.get("summary"))
    coverage = _as_dict(teacher_quality.get("student_role_coverage"))
    return {
        "status": str(teacher_quality.get("overall_status") or "missing"),
        "qualified_teacher_count": _safe_int(summary.get("qualified_teacher_count"), 0),
        "elite_teacher_count": _safe_int(summary.get("elite_teacher_count"), 0),
        "strong_teacher_count": _safe_int(summary.get("strong_teacher_count"), 0),
        "uncovered_student_role_count": _safe_int(summary.get("uncovered_student_role_count"), len(_as_list(coverage.get("uncovered_roles")))),
        "uncovered_student_roles": [str(item) for item in _as_list(coverage.get("uncovered_roles"))],
        "teacher_pool_ready": bool(
            _status(teacher_quality.get("overall_status")) in {"ready", "ok"}
            and _safe_int(summary.get("qualified_teacher_count"), 0) > 0
            and _safe_int(summary.get("elite_teacher_count"), 0) > 0
            and not _as_list(coverage.get("uncovered_roles"))
        ),
    }


def _overfit_contract(overfit: dict[str, Any]) -> dict[str, Any]:
    risk_count = _safe_int(overfit.get("risk_bot_count"), 0)
    hard_risk = _safe_int(overfit.get("hard_risk_bot_count"), 0)
    blocked_teachers = _safe_int(overfit.get("blocked_teacher_bot_count"), len(_as_list(overfit.get("blocked_teacher_bot_ids"))))
    teacher_ineligible = _safe_int(overfit.get("teacher_ineligible_bot_count"), 0)
    return {
        "status": str(overfit.get("overall_status") or "missing"),
        "risk_bot_count": risk_count,
        "hard_risk_bot_count": hard_risk,
        "guarded_bot_count": _safe_int(overfit.get("guarded_bot_count"), 0),
        "high_accuracy_guarded_bot_count": _safe_int(overfit.get("high_accuracy_guarded_bot_count"), 0),
        "blocked_teacher_bot_count": blocked_teachers,
        "blocked_teacher_bot_ids": [str(item) for item in _as_list(overfit.get("blocked_teacher_bot_ids"))[:20]],
        "teacher_ineligible_bot_count": teacher_ineligible,
        "teacher_lockout_enforced": bool(blocked_teachers > 0 or teacher_ineligible > 0),
        "generalization_guard_ready": bool(
            _status(overfit.get("overall_status")) in {"ready", "ok", "guarded"}
            and risk_count <= 0
            and hard_risk <= 0
        ),
    }


def _sleeve_ingestion_contract(payload: dict[str, Any]) -> dict[str, Any]:
    grade = _as_dict(payload.get("production_grade_contract"))
    mode = _as_dict(payload.get("ingestion_mode_contract"))
    return {
        "status": str(payload.get("overall_status") or "missing"),
        "ok": bool(payload.get("ok", False)),
        "grade": str(grade.get("grade") or ""),
        "score": _safe_float(grade.get("score"), 0.0),
        "missing": [str(item) for item in _as_list(grade.get("missing"))],
        "mode": str(mode.get("mode") or ""),
        "paper_soak_allowed": bool(mode.get("paper_soak_allowed", False)),
        "live_money_blocked": bool(mode.get("live_money_blocked", True)),
        "ready": bool(payload.get("ok", False) and str(payload.get("overall_status") or "") == "ready" and not _as_list(grade.get("missing"))),
    }


def _env_values(payload: dict[str, Any]) -> dict[str, str]:
    posture = _as_dict(payload.get("production_posture_contract"))
    registry = _as_dict(payload.get("registry_contract"))
    quality = _as_dict(payload.get("quality_lane_contract"))
    mesh = _as_dict(payload.get("mesh_contract"))
    return {
        "BOT_FLEET_PRODUCTION_POSTURE_ENABLED": "1",
        "BOT_FLEET_PRODUCTION_POSTURE_VERSION": CONTROL_VERSION,
        "BOT_FLEET_PRODUCTION_GRADE": str(posture.get("grade") or ""),
        "BOT_FLEET_PRODUCTION_STATE": str(posture.get("state") or ""),
        "BOT_FLEET_ACTIVE_BOT_COUNT": str(registry.get("active_bots") or 0),
        "BOT_FLEET_NON_DELETED_BOT_COUNT": str(registry.get("non_deleted_bots") or 0),
        "BOT_FLEET_COLLECTION_POSTURE": "all_active_collect",
        "BOT_FLEET_SAFETY_BOUNDARY": "paper_only_market_data",
        "BOT_FLEET_WEAK_BOT_ROUTING": str(quality.get("quality_debt_mode") or "owned_repair_lanes"),
        "BOT_FLEET_REQUIRE_LABEL_CONTRACTS": "1",
        "BOT_FLEET_REQUIRE_TARGET_FUNCTIONS": "1",
        "BOT_FLEET_REQUIRE_TEACHER_ROUTES": "1",
        "BOT_FLEET_REQUIRE_MASTER_ROUTES": "1",
        "BOT_FLEET_COMMUNICATION_SCORE": f"{_safe_float(mesh.get('communication_readiness_score'), 0.0):g}",
        "BOT_FLEET_QUALITY_READINESS_SCORE": f"{_safe_float(mesh.get('quality_readiness_score'), 0.0):g}",
        "BOT_FLEET_OVERFIT_RISK_BLOCK_TEACHING": "1",
        "BOT_FLEET_LIVE_MONEY_BLOCKED": "1",
        "PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS": "0",
        "PAPER_BROKER_BRIDGE_ENABLED": "1",
        "MARKET_DATA_ONLY": "1",
        "ALLOW_ORDER_EXECUTION": "0",
    }


def _write_override(path: Path, payload: dict[str, Any]) -> None:
    env = _env_values(payload)
    lines = [
        "# Managed by scripts/ops/bot_fleet_production_posture.py",
        f"# updated_at_utc={payload.get('timestamp_utc')}",
    ]
    lines.extend(f"{key}={shlex.quote(str(value))}" for key, value in sorted(env.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False, override_path: Path = DEFAULT_OVERRIDE_PATH) -> dict[str, Any]:
    project_root = Path(project_root)
    sources = _load_sources(project_root)
    registry = _registry_contract(_as_dict(sources.get("registry")))
    paper = _paper_contract(_as_dict(sources.get("paper_standard")))
    collection = _collection_contract(_as_dict(sources.get("observation_rollup")))
    runtime = _runtime_contract(_as_dict(sources.get("health_fast")))
    sleeve_ingestion = _sleeve_ingestion_contract(_as_dict(sources.get("sleeve_ingestion")))
    mesh = _mesh_contract(_as_dict(sources.get("bot_mesh")))
    quality = _quality_lane_contract(
        _as_dict(sources.get("bot_quality")),
        _as_dict(sources.get("training_quality")),
        _as_dict(sources.get("supportability")),
    )
    teacher = _teacher_contract(_as_dict(sources.get("teacher_quality")))
    overfit = _overfit_contract(_as_dict(sources.get("overfitting_awareness")))
    freshness = _source_freshness_contract(project_root, sources)

    must_haves = {
        "registry_identity_complete": bool(registry.get("identity_contract_ready", False)),
        "registry_roles_complete": _safe_int(registry.get("missing_role_count"), 0) <= 0,
        "registry_label_contracts_complete": bool(registry.get("label_contract_ready", False)),
        "registry_target_functions_complete": bool(registry.get("target_function_contract_ready", False)),
        "all_non_deleted_bots_active": bool(registry.get("all_non_deleted_active", False)),
        "all_non_deleted_bots_collecting": bool(registry.get("all_non_deleted_collecting", False)),
        "deleted_rows_quarantined": bool(registry.get("deleted_rows_quarantined", False)),
        "registry_live_authority_absent": bool(registry.get("live_authority_absent", False)),
        "paper_standard_ready": bool(paper.get("ok", False) and str(paper.get("status") or "") == "ready"),
        "paper_live_execution_locked": bool(paper.get("live_execution_locked", False)),
        "collection_coverage_ready": bool(collection.get("coverage_ready", False)),
        "health_fast_guarded_paper_ready": bool(runtime.get("health_fast_ok", False) and str(runtime.get("guarded_paper_status") or "") == "ready"),
        "all_sleeves_runtime_ready": bool(runtime.get("all_sleeves_ready", False)),
        "sleeve_ingestion_ready": bool(sleeve_ingestion.get("ready", False)),
        "mesh_routes_ready": bool(mesh.get("route_ready", False)),
        "teacher_pool_ready": bool(teacher.get("teacher_pool_ready", False)),
        "quality_debt_owned": bool(quality.get("debt_owned", False)),
        "overfitting_guard_ready": bool(overfit.get("generalization_guard_ready", False)),
        "source_artifacts_fresh": bool(freshness.get("all_required_fresh", False)),
        "live_money_blocked": bool(sleeve_ingestion.get("live_money_blocked", True)),
    }
    missing = [key for key, value in must_haves.items() if not bool(value)]

    score = 100.0
    if not registry.get("identity_contract_ready", False):
        score -= 18.0
    if not registry.get("label_contract_ready", False):
        score -= min(20.0, 4.0 + _safe_int(registry.get("missing_label_contract_count"), 0) * 0.5)
    if not registry.get("target_function_contract_ready", False):
        score -= min(16.0, 4.0 + _safe_int(registry.get("missing_target_functions_count"), 0) * 0.5)
    if not registry.get("all_non_deleted_active", False):
        score -= min(22.0, 8.0 + _safe_int(registry.get("inactive_non_deleted_count"), 0) * 0.3)
    if not registry.get("all_non_deleted_collecting", False):
        score -= min(28.0, 10.0 + _safe_int(registry.get("non_collecting_count"), 0) * 0.4)
    if not registry.get("live_authority_absent", False) or not paper.get("live_execution_locked", False):
        score -= 50.0
    if not collection.get("coverage_ready", False):
        score -= 18.0
    if not runtime.get("all_sleeves_ready", False):
        score -= 12.0
    if not sleeve_ingestion.get("ready", False):
        score -= 12.0
    if not mesh.get("route_ready", False):
        score -= 14.0
    if not teacher.get("teacher_pool_ready", False):
        score -= 10.0
    if not quality.get("debt_owned", False):
        score -= 16.0
    if not overfit.get("generalization_guard_ready", False):
        score -= 12.0
    if not freshness.get("all_required_fresh", False):
        score -= min(24.0, 4.0 * len(_as_list(freshness.get("stale_or_missing"))))
    score = round(max(min(score, 100.0), 0.0), 2)
    grade = _grade(score)
    state = "production_controlled_paper_soak" if not missing else "production_attention_required"
    ok = not missing

    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": SCHEMA_VERSION,
        "control_version": CONTROL_VERSION,
        "ok": ok,
        "overall_status": "ready" if ok else "blocked",
        "production_posture_contract": {
            "grade": grade,
            "score": score,
            "state": state,
            "must_haves": must_haves,
            "missing": missing,
            "policy": "grades whole-bot-fleet production posture for paper soak; it does not certify raw profitability or live-money readiness",
        },
        "registry_contract": registry,
        "paper_standard_contract": paper,
        "collection_contract": collection,
        "runtime_contract": runtime,
        "sleeve_ingestion_contract": sleeve_ingestion,
        "mesh_contract": mesh,
        "quality_lane_contract": quality,
        "teacher_contract": teacher,
        "overfitting_contract": overfit,
        "source_freshness_contract": freshness,
        "bot_lanes": {
            "paper_live_data": {
                "bot_count": _safe_int(paper.get("paper_live_data_enabled_bots"), 0),
                "mode": "may paper trade on live data when local paper/profitability guards say yes",
                "live_execution_authority": False,
            },
            "collection_until_standard": {
                "bot_count": _safe_int(paper.get("collection_until_standard_bots"), 0),
                "mode": "collect observations, labels, and runtime context until paper standard clears",
                "live_execution_authority": False,
            },
            "quality_repair": {
                "bot_count": _safe_int(quality.get("planned_queue_count"), 0),
                "mode": str(quality.get("quality_debt_mode") or ""),
                "selected_retrain_bot_count": _safe_int(quality.get("selected_targeted_retrain_bot_count"), 0),
                "precompute_target_bot_count": _safe_int(quality.get("precompute_target_bot_count"), 0),
                "weak_sleeve_count": _safe_int(quality.get("weak_sleeve_count"), 0),
                "live_execution_authority": False,
            },
            "overfit_containment": {
                "risk_bot_count": _safe_int(overfit.get("risk_bot_count"), 0),
                "guarded_bot_count": _safe_int(overfit.get("guarded_bot_count"), 0),
                "policy": "overfit-risk bots cannot teach, promote, or carry master-vote authority until generalization canaries clear",
                "live_execution_authority": False,
            },
        },
        "control_env_recommendations": {},
        "regression_guards": [
            "every non-deleted bot must have identity, role, label contract, and target-function contracts",
            "every non-deleted bot must be active live-data collection or explicitly deleted/quarantined",
            "direct/live execution must stay absent while this fleet posture control is active",
            "all active sub/infrastructure bots must route to a master and all masters must route to a grandmaster",
            "quality debt is production-safe only when owned by explicit diagnostics, precompute, coverage, teacher, or targeted-retrain lanes",
            "overfit-risk bots cannot teach or promote until generalization evidence clears",
            "sleeve ingestion must be A+/ready before whole-fleet posture can be A+",
            "source artifacts must be fresh before claiming the whole bot fleet is production controlled",
        ],
        "recommended_actions": ordered_unique(
            [
                "keep bot fleet production posture loaded through runtime env",
                "refresh overfitting-awareness before trusting teacher or promotion lanes" if not overfit.get("generalization_guard_ready", False) else "",
                "run bot-quality-autopilot apply on a bounded schedule so quality debt stays owned" if not quality.get("debt_owned", False) else "",
                "keep weak sleeves in diagnostics, precompute, and targeted retrain lanes before promotion" if _safe_int(quality.get("weak_sleeve_count"), 0) > 0 else "",
                "keep all bots paper/data-only until live canary gates independently clear",
            ]
        ),
        "apply_result": {
            "applied": bool(apply),
            "override_path": str(override_path),
        },
    }
    payload["control_env_recommendations"] = _env_values(payload)
    if apply:
        _write_override(override_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade and control whole-bot-fleet production posture for paper-soak operation.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_path = Path(args.out_file).expanduser()
    override_path = Path(args.override).expanduser()
    payload = build_payload(project_root, apply=bool(args.apply), override_path=override_path)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        posture = _as_dict(payload.get("production_posture_contract"))
        print(
            "bot_fleet_production_posture "
            f"status={payload.get('overall_status')} "
            f"grade={posture.get('grade')} "
            f"missing={len(_as_list(posture.get('missing')))}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
