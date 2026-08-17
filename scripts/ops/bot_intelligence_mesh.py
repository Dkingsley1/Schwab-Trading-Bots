#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "bot_intelligence_mesh_latest.json"

TIER_ORDER = ("infrastructure", "sub", "master", "grand_master")
QUALITY_TARGET = 100.0
OVERFIT_RISK_STATUSES = {"leak_like", "severe_overfit", "overfit_watch"}

GRAND_MARKERS = (
    "grandmaster",
    "grand_master",
    "grand master",
)
MASTER_MARKERS = (
    "sleeve_master",
    "master_bot",
    "master_coordination",
    "per_sleeve_master_bots",
)
INFRA_MARKERS = (
    "infrastructure",
    "infra",
    "guard",
    "watchdog",
    "supervisor",
    "validator",
)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _as_str_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw or "").strip()
    return [text] if text else []


def _bot_id(row: dict[str, Any]) -> str:
    return str(row.get("bot_id") or "").strip()


def _lower_bot_id(row: dict[str, Any]) -> str:
    return _bot_id(row).lower()


def _registry_rows(project_root: Path) -> list[dict[str, Any]]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _is_active(row: dict[str, Any]) -> bool:
    if row.get("deleted") is True:
        return False
    lifecycle = str(row.get("lifecycle_state") or "").strip().lower()
    if lifecycle in {"deleted", "retired", "archived"}:
        return False
    return bool(row.get("active", False))


def _bot_text(row: dict[str, Any]) -> str:
    parts: list[str] = [
        str(row.get("bot_id") or ""),
        str(row.get("bot_role") or ""),
        str(row.get("slot_kind") or ""),
        str(row.get("bot_intelligence_layer") or ""),
        str(row.get("sleeve_profile") or ""),
    ]
    parts.extend(_as_str_list(row.get("target_functions")))
    return " ".join(parts).lower()


def classify_tier(row: dict[str, Any]) -> str:
    identity_text = " ".join(
        [
            str(row.get("bot_id") or ""),
            str(row.get("slot_kind") or ""),
            str(row.get("bot_intelligence_layer") or ""),
        ]
    ).lower()
    functions = {item.lower() for item in _as_str_list(row.get("target_functions"))}
    role = str(row.get("bot_role") or "").strip().lower()
    if any(marker in identity_text for marker in GRAND_MARKERS) or "grand_master" in functions:
        return "grand_master"
    if (
        any(marker in identity_text for marker in MASTER_MARKERS)
        or "sleeve_master" in functions
        or "master_bot" in functions
        or "sleeve_masters" in functions
    ):
        return "master"
    if role == "infrastructure_sub_bot" or any(marker in identity_text for marker in INFRA_MARKERS):
        return "infrastructure"
    return "sub"


def _compact_bot(row: dict[str, Any], tier: str) -> dict[str, Any]:
    return {
        "bot_id": _bot_id(row),
        "tier": tier,
        "bot_role": str(row.get("bot_role") or ""),
        "active": _is_active(row),
        "sleeve_profile": str(row.get("sleeve_profile") or ""),
        "slot_kind": str(row.get("slot_kind") or ""),
    }


def _quality_score(training_quality: dict[str, Any]) -> float:
    for key in ("training_quality_score", "quality_score", "score"):
        if key in training_quality:
            return round(_safe_float(training_quality.get(key), 0.0), 3)
    quality_system = _as_dict(training_quality.get("quality_score_system"))
    for key in ("training_quality_score", "overall_score", "score"):
        if key in quality_system:
            return round(_safe_float(quality_system.get(key), 0.0), 3)
    return 0.0


def _data_quality_components(data_rollup: dict[str, Any]) -> dict[str, Any]:
    collector_count = _safe_int(data_rollup.get("collector_count"), 0)
    observed = _safe_int(data_rollup.get("bots_with_observations"), 0)
    ready = _safe_int(data_rollup.get("training_ready_count"), 0)
    zero = _safe_int(data_rollup.get("zero_observation_count"), len(_as_list(data_rollup.get("zero_observation_bot_ids"))))
    if collector_count <= 0:
        return {
            "collector_count": collector_count,
            "bots_with_observations": observed,
            "training_ready_count": ready,
            "zero_observation_count": zero,
            "data_quality_score": 0.0,
            "collection_coverage_score": 0.0,
            "training_readiness_score": 0.0,
            "training_ready_gap": 0,
        }
    default_coverage = (observed / collector_count) * 100.0
    default_training_ready = (ready / collector_count) * 100.0
    zero_penalty = min((zero / collector_count) * 100.0, 100.0)
    coverage_score = (
        _safe_float(data_rollup.get("collection_coverage_score"), default_coverage)
        if "collection_coverage_score" in data_rollup
        else default_coverage
    )
    training_readiness_score = (
        _safe_float(data_rollup.get("training_readiness_score"), default_training_ready)
        if "training_readiness_score" in data_rollup
        else default_training_ready
    )
    if "data_quality_score" in data_rollup:
        data_score = _safe_float(data_rollup.get("data_quality_score"), default_coverage)
    else:
        data_score = max(0.0, min(100.0, coverage_score - zero_penalty))
    return {
        "collector_count": collector_count,
        "bots_with_observations": observed,
        "training_ready_count": ready,
        "zero_observation_count": zero,
        "data_quality_score": round(max(0.0, min(data_score, 100.0)), 3),
        "collection_coverage_score": round(max(0.0, min(coverage_score, 100.0)), 3),
        "training_readiness_score": round(max(0.0, min(training_readiness_score, 100.0)), 3),
        "training_ready_gap": max(collector_count - ready, 0),
    }


def _data_quality_score(data_rollup: dict[str, Any]) -> float:
    return _safe_float(_data_quality_components(data_rollup).get("data_quality_score"), 0.0)


def _link_key(edge: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(edge.get("route_kind") or ""),
        str(edge.get("source_bot_id") or ""),
        str(edge.get("target_bot_id") or ""),
        str(edge.get("source") or ""),
    )


def _edge(
    *,
    source_bot_id: str,
    source_tier: str,
    target_bot_id: str,
    target_tier: str,
    route_kind: str,
    source: str,
    confidence: float,
    purpose: str,
) -> dict[str, Any]:
    return {
        "source_bot_id": source_bot_id,
        "source_tier": source_tier,
        "target_bot_id": target_bot_id,
        "target_tier": target_tier,
        "route_kind": route_kind,
        "source": source,
        "confidence": round(max(0.0, min(float(confidence), 1.0)), 3),
        "purpose": purpose,
    }


def _first_active_bot(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        bot_id = _bot_id(row)
        if bot_id and _is_active(row):
            return bot_id
    for row in rows:
        bot_id = _bot_id(row)
        if bot_id:
            return bot_id
    return ""


def _build_hierarchy_edges(
    rows: list[dict[str, Any]],
    tiers_by_id: dict[str, str],
    *,
    edge_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_id = {_lower_bot_id(row): row for row in rows if _lower_bot_id(row)}
    active_rows = [row for row in rows if _is_active(row)]
    masters = [row for row in active_rows if tiers_by_id.get(_lower_bot_id(row)) == "master"]
    grandmasters = [row for row in active_rows if tiers_by_id.get(_lower_bot_id(row)) == "grand_master"]
    masters_by_sleeve: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grand_by_sleeve: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in masters:
        sleeve = str(row.get("sleeve_profile") or "").strip().lower()
        if sleeve:
            masters_by_sleeve[sleeve].append(row)
    for row in grandmasters:
        sleeve = str(row.get("sleeve_profile") or "").strip().lower()
        if sleeve:
            grand_by_sleeve[sleeve].append(row)

    default_master = _first_active_bot(masters)
    default_grand = _first_active_bot(grandmasters)
    edges: list[dict[str, Any]] = []
    stats = Counter()

    def add(edge: dict[str, Any]) -> None:
        if edge["source_bot_id"] and edge["target_bot_id"]:
            edges.append(edge)
            stats[str(edge["route_kind"])] += 1

    for row in active_rows:
        bot_id = _bot_id(row)
        if not bot_id:
            continue
        lower_id = bot_id.lower()
        tier = tiers_by_id.get(lower_id, "sub")
        sleeve = str(row.get("sleeve_profile") or "").strip().lower()
        explicit_master = str(row.get("reports_to_sleeve_master_bot_id") or row.get("sleeve_master_bot_id") or "").strip()
        explicit_grand = str(row.get("grandmaster_bridge_bot_id") or "").strip()

        if tier in {"sub", "infrastructure"}:
            target_master = explicit_master
            link_source = "registry_explicit"
            confidence = 1.0
            if not target_master and sleeve and masters_by_sleeve.get(sleeve):
                target_master = _first_active_bot(masters_by_sleeve[sleeve])
                link_source = "sleeve_profile_inferred"
                confidence = 0.72
            if not target_master and default_master:
                target_master = default_master
                link_source = "global_master_bus_inferred"
                confidence = 0.55
            if target_master:
                target_tier = tiers_by_id.get(target_master.lower(), "master")
                add(
                    _edge(
                        source_bot_id=bot_id,
                        source_tier=tier,
                        target_bot_id=target_master,
                        target_tier=target_tier,
                        route_kind=f"{tier}_to_master",
                        source=link_source,
                        confidence=confidence,
                        purpose="publish observations, quality evidence, and runtime needs upward",
                    )
                )

        if tier == "master":
            target_grand = explicit_grand
            link_source = "registry_explicit"
            confidence = 1.0
            if not target_grand and sleeve and grand_by_sleeve.get(sleeve):
                target_grand = _first_active_bot(grand_by_sleeve[sleeve])
                link_source = "sleeve_profile_inferred"
                confidence = 0.72
            if not target_grand and default_grand:
                target_grand = default_grand
                link_source = "global_grandmaster_bus_inferred"
                confidence = 0.58
            if target_grand:
                add(
                    _edge(
                        source_bot_id=bot_id,
                        source_tier="master",
                        target_bot_id=target_grand,
                        target_tier=tiers_by_id.get(target_grand.lower(), "grand_master"),
                        route_kind="master_to_grand_master",
                        source=link_source,
                        confidence=confidence,
                        purpose="publish sleeve summaries, risk votes, and promotion evidence upward",
                    )
                )

        if tier == "grand_master" and default_master:
            add(
                _edge(
                    source_bot_id=bot_id,
                    source_tier="grand_master",
                    target_bot_id=default_master,
                    target_tier=tiers_by_id.get(default_master.lower(), "master"),
                    route_kind="grand_master_to_master_policy",
                    source="control_plane_policy_bus",
                    confidence=0.65,
                    purpose="broadcast policy, backlog priorities, training gates, and quality targets downward",
                )
            )

        for teacher in _as_str_list(row.get("bootstrap_teacher_bot_ids")):
            teacher_row = rows_by_id.get(teacher.lower(), {})
            teacher_tier = tiers_by_id.get(teacher.lower(), classify_tier(teacher_row) if teacher_row else "sub")
            add(
                _edge(
                    source_bot_id=teacher,
                    source_tier=teacher_tier,
                    target_bot_id=bot_id,
                    target_tier=tier,
                    route_kind="bootstrap_teacher_to_student",
                    source="registry_bootstrap_teacher",
                    confidence=0.8,
                    purpose="seed student calibration and abstention behavior from known teacher bots",
                )
            )

    unique_edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for edge in edges:
        key = _link_key(edge)
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append(edge)

    active_counts = Counter(tiers_by_id.get(_lower_bot_id(row), "sub") for row in active_rows)
    expected_sub_or_infra = active_counts.get("sub", 0) + active_counts.get("infrastructure", 0)
    sub_or_infra_routes = stats.get("sub_to_master", 0) + stats.get("infrastructure_to_master", 0)
    master_routes = stats.get("master_to_grand_master", 0)
    stats_payload = {
        "edge_count_total": len(unique_edges),
        "edge_count_exported": min(len(unique_edges), edge_limit),
        "route_counts": dict(stats),
        "active_sub_or_infra_route_ratio": round(sub_or_infra_routes / expected_sub_or_infra, 4)
        if expected_sub_or_infra
        else 1.0,
        "active_master_route_ratio": round(master_routes / active_counts.get("master", 1), 4)
        if active_counts.get("master", 0)
        else 1.0,
    }
    return unique_edges[: max(int(edge_limit), 1)], stats_payload


def _teacher_student_edges(
    teacher_plan: dict[str, Any],
    tiers_by_id: dict[str, str],
    *,
    edge_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignments = _as_list(teacher_plan.get("assignments"))
    edges: list[dict[str, Any]] = []
    teacher_ids: list[str] = []
    student_ids: list[str] = []
    for assignment in assignments:
        row = _as_dict(assignment)
        student_id = str(row.get("student_bot_id") or row.get("bot_id") or "").strip()
        if not student_id:
            continue
        student_ids.append(student_id)
        student_tier = tiers_by_id.get(student_id.lower(), "sub")
        for teacher in _as_list(row.get("teachers")):
            teacher_row = _as_dict(teacher)
            teacher_id = str(teacher_row.get("bot_id") or teacher_row.get("teacher_bot_id") or "").strip()
            if not teacher_id:
                continue
            teacher_ids.append(teacher_id)
            edges.append(
                _edge(
                    source_bot_id=teacher_id,
                    source_tier=tiers_by_id.get(teacher_id.lower(), "sub"),
                    target_bot_id=student_id,
                    target_tier=student_tier,
                    route_kind="teacher_to_student",
                    source="teacher_student_plan",
                    confidence=0.9,
                    purpose="distill decision examples, abstention boundaries, and side-specific calibration",
                )
            )
    unique_edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for edge in edges:
        key = _link_key(edge)
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append(edge)
    summary = _as_dict(teacher_plan.get("summary"))
    stats = {
        "teacher_count": _safe_int(summary.get("teacher_count"), len(set(teacher_ids))),
        "student_count": _safe_int(summary.get("student_count"), len(set(student_ids))),
        "assignment_count": _safe_int(summary.get("assignment_count"), len(assignments)),
        "edge_count_total": len(unique_edges),
        "edge_count_exported": min(len(unique_edges), edge_limit),
    }
    return unique_edges[: max(int(edge_limit), 1)], stats


def _teacher_quality_summary(teacher_quality: dict[str, Any], teacher_plan: dict[str, Any]) -> dict[str, Any]:
    tq_summary = _as_dict(teacher_quality.get("summary"))
    plan_summary = _as_dict(teacher_plan.get("summary"))
    role_coverage = teacher_quality.get("role_coverage") if isinstance(teacher_quality.get("role_coverage"), list) else []
    student_coverage = _as_dict(teacher_quality.get("student_role_coverage"))
    return {
        "status": str(teacher_quality.get("overall_status") or "missing" if not teacher_quality else teacher_quality.get("overall_status") or "ready"),
        "teacher_count": _safe_int(plan_summary.get("teacher_count"), len(_as_list(teacher_plan.get("teachers")))),
        "student_count": _safe_int(plan_summary.get("student_count"), len(_as_list(teacher_plan.get("assignments")))),
        "assignment_count": _safe_int(plan_summary.get("assignment_count"), len(_as_list(teacher_plan.get("assignments")))),
        "qualified_teacher_count": _safe_int(tq_summary.get("qualified_teacher_count"), len(_as_list(teacher_quality.get("qualified_teachers")))),
        "elite_teacher_count": _safe_int(tq_summary.get("elite_teacher_count"), 0),
        "strong_teacher_count": _safe_int(tq_summary.get("strong_teacher_count"), 0),
        "uncovered_student_role_count": _safe_int(tq_summary.get("uncovered_student_role_count"), len(_as_list(student_coverage.get("uncovered_roles")))),
        "uncovered_student_roles": [str(item) for item in _as_list(student_coverage.get("uncovered_roles"))],
        "role_coverage": role_coverage[:12],
    }


def _overfitting_summary(overfitting_awareness: dict[str, Any]) -> dict[str, Any]:
    if not overfitting_awareness:
        return {
            "overall_status": "missing",
            "risk_bot_count": 0,
            "hard_risk_bot_count": 0,
            "guarded_bot_count": 0,
            "high_accuracy_guarded_bot_count": 0,
            "blocked_teacher_count": 0,
            "active_status_counts": {},
            "broadcast_contract": {},
            "top_risk_bots": [],
        }
    return {
        "overall_status": str(overfitting_awareness.get("overall_status") or "ready"),
        "risk_bot_count": _safe_int(overfitting_awareness.get("risk_bot_count"), 0),
        "hard_risk_bot_count": _safe_int(overfitting_awareness.get("hard_risk_bot_count"), 0),
        "guarded_bot_count": _safe_int(overfitting_awareness.get("guarded_bot_count"), 0),
        "high_accuracy_guarded_bot_count": _safe_int(overfitting_awareness.get("high_accuracy_guarded_bot_count"), 0),
        "blocked_teacher_count": _safe_int(
            overfitting_awareness.get("blocked_teacher_bot_count"),
            len(_as_list(overfitting_awareness.get("blocked_teacher_bot_ids"))),
        ),
        "teacher_ineligible_bot_count": _safe_int(overfitting_awareness.get("teacher_ineligible_bot_count"), 0),
        "active_status_counts": _as_dict(overfitting_awareness.get("active_status_counts")),
        "broadcast_contract": _as_dict(overfitting_awareness.get("broadcast_contract")),
        "top_risk_bots": _as_list(overfitting_awareness.get("top_risk_bots"))[:12],
    }


def _quality_contract(
    *,
    training_quality: dict[str, Any],
    data_rollup: dict[str, Any],
    teacher_quality: dict[str, Any],
    supportability: dict[str, Any],
    mesh_stats: dict[str, Any],
    overfitting_awareness: dict[str, Any],
) -> dict[str, Any]:
    training_score = _quality_score(training_quality)
    data_components = _data_quality_components(data_rollup)
    data_score = _safe_float(data_components.get("data_quality_score"), 0.0)
    collection_coverage_score = _safe_float(data_components.get("collection_coverage_score"), 0.0)
    training_readiness_score = _safe_float(data_components.get("training_readiness_score"), 0.0)
    training_ready_gap = _safe_int(data_components.get("training_ready_gap"), 0)
    targeted = _as_dict(training_quality.get("targeted_actions"))
    runtime_depth_rows = [_as_dict(row) for row in _as_list(targeted.get("runtime_input_depth_debt_rows"))]
    quality_probation = [str(item) for item in _as_list(targeted.get("quality_probation_bot_ids"))]
    targeted_retrain = [str(item) for item in _as_list(targeted.get("targeted_retrain_bot_ids"))]
    repair_ids = [str(item) for item in _as_list(targeted.get("repair_runtime_input_bot_ids"))]
    zero_ids = [str(item) for item in _as_list(data_rollup.get("zero_observation_bot_ids"))]
    teacher_summary = _teacher_quality_summary(teacher_quality, {})
    overfit_summary = _overfitting_summary(overfitting_awareness)
    support_students_without_teachers = _safe_int(
        supportability.get("students_without_teachers"),
        len(_as_list(supportability.get("students_without_teachers"))),
    )

    blockers: list[dict[str, Any]] = []
    if training_score < QUALITY_TARGET:
        blockers.append(
            {
                "key": "training_quality_below_100",
                "current": training_score,
                "target": QUALITY_TARGET,
                "need": "clear probation, depth debt, and retrain debt before claiming A+ training quality",
            }
        )
    if data_score < QUALITY_TARGET:
        blockers.append(
            {
                "key": "data_quality_below_100",
                "current": data_score,
                "target": QUALITY_TARGET,
                "need": "keep observation rollup fresh and eliminate zero-observation collection gaps",
            }
        )
    if training_readiness_score < QUALITY_TARGET:
        blockers.append(
            {
                "key": "training_readiness_below_100",
                "current": training_readiness_score,
                "target": QUALITY_TARGET,
                "training_ready_count": _safe_int(data_components.get("training_ready_count"), 0),
                "training_ready_gap": training_ready_gap,
                "need": "keep collecting until intended training bots clear minimum observation/day floors; do not confuse this with raw data coverage quality",
            }
        )
    if runtime_depth_rows:
        blockers.append(
            {
                "key": "runtime_input_depth_debt",
                "bot_count": len(runtime_depth_rows),
                "bots": [str(row.get("bot_id") or "") for row in runtime_depth_rows[:12]],
                "need": "target deeper sequence/context collection for these bots before more blind retrains",
            }
        )
    if quality_probation:
        blockers.append(
            {
                "key": "quality_probation",
                "bot_count": len(quality_probation),
                "bots": quality_probation[:12],
                "need": "repair labels, abstention, and side calibration before using these bots as teachers or promotion evidence",
            }
        )
    if repair_ids:
        blockers.append(
            {
                "key": "runtime_input_repair",
                "bot_count": len(repair_ids),
                "bots": repair_ids[:12],
                "need": "repair missing feature/runtime inputs before the next training wave",
            }
        )
    if targeted_retrain:
        blockers.append(
            {
                "key": "targeted_retrain_debt",
                "bot_count": len(targeted_retrain),
                "bots": targeted_retrain[:12],
                "need": "run bounded recovery canaries and keep only non-regressing artifacts",
            }
        )
    if zero_ids:
        blockers.append(
            {
                "key": "zero_observation_bots",
                "bot_count": len(zero_ids),
                "bots": zero_ids[:12],
                "need": "run targeted collection for these bots before training them",
            }
        )
    if support_students_without_teachers > 0:
        blockers.append(
            {
                "key": "students_without_teachers",
                "count": support_students_without_teachers,
                "need": "assign qualified teachers by bot role before distillation training",
            }
        )
    if _safe_int(teacher_summary.get("elite_teacher_count"), 0) <= 0:
        blockers.append(
            {
                "key": "elite_teacher_pool_missing",
                "need": "promote at least one clean elite teacher per active student role",
            }
        )
    if _safe_float(mesh_stats.get("active_sub_or_infra_route_ratio"), 0.0) < 0.95:
        blockers.append(
            {
                "key": "sub_or_infra_master_routes_incomplete",
                "current": mesh_stats.get("active_sub_or_infra_route_ratio"),
                "target": 0.95,
                "need": "add explicit sleeve master or global master bus routes for unlinked active bots",
            }
        )
    if _safe_float(mesh_stats.get("active_master_route_ratio"), 0.0) < 0.95:
        blockers.append(
            {
                "key": "master_grandmaster_routes_incomplete",
                "current": mesh_stats.get("active_master_route_ratio"),
                "target": 0.95,
                "need": "add explicit grandmaster bridge routes for sleeve masters",
            }
        )
    overfit_risk_count = _safe_int(overfit_summary.get("risk_bot_count"), 0)
    guarded_count = _safe_int(overfit_summary.get("guarded_bot_count"), 0)
    high_accuracy_guarded_count = _safe_int(overfit_summary.get("high_accuracy_guarded_bot_count"), 0)
    overfit_status = str(overfit_summary.get("overall_status") or "")
    if overfit_risk_count > 0 or overfit_status == "blocked":
        blockers.append(
            {
                "key": "overfitting_awareness_risk",
                "overall_status": overfit_summary.get("overall_status"),
                "risk_bot_count": overfit_risk_count,
                "hard_risk_bot_count": overfit_summary.get("hard_risk_bot_count"),
                "guarded_bot_count": guarded_count,
                "high_accuracy_guarded_bot_count": high_accuracy_guarded_count,
                "active_status_counts": overfit_summary.get("active_status_counts"),
                "need": "exclude overfit-risk bots from teacher, promotion, and master-vote duty until generalization canaries clear",
            }
        )
    elif guarded_count > 0 or overfit_status == "guarded":
        blockers.append(
            {
                "key": "overfitting_awareness_guarded",
                "overall_status": overfit_summary.get("overall_status"),
                "risk_bot_count": overfit_risk_count,
                "guarded_bot_count": guarded_count,
                "high_accuracy_guarded_bot_count": high_accuracy_guarded_count,
                "active_status_counts": overfit_summary.get("active_status_counts"),
                "need": "keep high-accuracy bots provisional until cross-regime and sequence evidence confirms they generalize",
            }
        )

    commands = [
        ["./scripts/ops/opsctl.sh", "overfitting-awareness", "--json"],
        ["./scripts/ops/opsctl.sh", "data-collection-observation-rollup", "--apply", "--json"],
        ["./scripts/ops/opsctl.sh", "teacher-quality", "--json"],
        ["./scripts/ops/opsctl.sh", "supportability-control", "--json"],
        ["./scripts/ops/opsctl.sh", "training-quality", "--json"],
        ["./scripts/ops/opsctl.sh", "bot-needs", "--limit", "40", "--json"],
        ["./scripts/ops/opsctl.sh", "bot-intelligence-mesh", "--json"],
    ]
    return {
        "target_score": QUALITY_TARGET,
        "current_training_quality_score": training_score,
        "current_data_quality_score": data_score,
        "current_collection_coverage_score": collection_coverage_score,
        "current_training_readiness_score": training_readiness_score,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "summary": {
            "training_gap": round(max(QUALITY_TARGET - training_score, 0.0), 3),
            "data_gap": round(max(QUALITY_TARGET - data_score, 0.0), 3),
            "collection_coverage_gap": round(max(QUALITY_TARGET - collection_coverage_score, 0.0), 3),
            "training_readiness_gap": round(max(QUALITY_TARGET - training_readiness_score, 0.0), 3),
            "training_ready_gap": training_ready_gap,
            "runtime_depth_debt_count": len(runtime_depth_rows),
            "quality_probation_count": len(quality_probation),
            "targeted_retrain_count": len(targeted_retrain),
            "overfit_risk_bot_count": _safe_int(overfit_summary.get("risk_bot_count"), 0),
            "overfit_hard_risk_bot_count": _safe_int(overfit_summary.get("hard_risk_bot_count"), 0),
            "overfit_guarded_bot_count": guarded_count,
            "overfit_high_accuracy_guarded_bot_count": high_accuracy_guarded_count,
        },
        "safe_next_commands": commands,
        "stop_condition": "stop widening training when memory/runtime/backlog gates degrade or a recovery canary regresses",
    }


def _communication_matrix() -> list[dict[str, Any]]:
    return [
        {
            "from_tier": "infrastructure",
            "to_tier": "sub",
            "contract": "runtime, storage, data freshness, feature availability, and repair needs",
            "artifact_channel": "governance/health/*_latest.json",
            "may_execute_trades": False,
        },
        {
            "from_tier": "sub",
            "to_tier": "master",
            "contract": "observations, decisions, abstentions, label quality, and strategy evidence",
            "artifact_channel": "decision logs, training snapshots, registry counters",
            "may_execute_trades": False,
        },
        {
            "from_tier": "master",
            "to_tier": "grand_master",
            "contract": "sleeve rollups, quality votes, capacity needs, and promotion evidence",
            "artifact_channel": "governance/health and governance/distillation",
            "may_execute_trades": False,
        },
        {
            "from_tier": "grand_master",
            "to_tier": "master",
            "contract": "global policy, priority routing, resource budgets, and quality targets",
            "artifact_channel": "system signal bus and bot intelligence mesh",
            "may_execute_trades": False,
        },
        {
            "from_tier": "teacher",
            "to_tier": "student",
            "contract": "distillation examples, abstention thresholds, and side-specific calibration",
            "artifact_channel": "governance/distillation/teacher_student_plan_latest.json",
            "may_execute_trades": False,
        },
    ]


def _communication_readiness_score(
    *,
    active_tier_counts: dict[str, int],
    hierarchy_stats: dict[str, Any],
    teacher_stats: dict[str, Any],
) -> float:
    present_tiers = sum(1 for tier in TIER_ORDER if int(active_tier_counts.get(tier, 0)) > 0)
    tier_score = (present_tiers / len(TIER_ORDER)) * 35.0
    hierarchy_score = (
        min(_safe_float(hierarchy_stats.get("active_sub_or_infra_route_ratio"), 0.0), 1.0) * 25.0
        + min(_safe_float(hierarchy_stats.get("active_master_route_ratio"), 0.0), 1.0) * 20.0
    )
    teacher_edge_count = _safe_int(teacher_stats.get("edge_count_total"), 0)
    teacher_student_count = _safe_int(teacher_stats.get("student_count"), 0)
    teacher_score = 20.0 if teacher_edge_count and teacher_student_count else 0.0
    return round(max(0.0, min(tier_score + hierarchy_score + teacher_score, 100.0)), 3)


def _quality_readiness_score(
    *,
    communication_score: float,
    training_score: float,
    data_score: float,
    training_readiness_score: float,
    blocker_count: int,
) -> float:
    score = (
        min(max(communication_score, 0.0), 100.0) * 0.25
        + min(max(training_score, 0.0), 100.0) * 0.35
        + min(max(data_score, 0.0), 100.0) * 0.25
        + min(max(training_readiness_score, 0.0), 100.0) * 0.15
    )
    if blocker_count > 0:
        score = min(score, 94.0)
    if training_score < QUALITY_TARGET or data_score < QUALITY_TARGET or training_readiness_score < QUALITY_TARGET:
        score = min(score, 96.0)
    return round(max(0.0, min(score, 100.0)), 3)


def _status_for_communication(score: float, missing_tiers: list[str]) -> str:
    if missing_tiers:
        return "blocked"
    if score >= 99.0:
        return "ready"
    if score >= 80.0:
        return "advisory"
    return "needs_work"


def build_payload(project_root: Path = PROJECT_ROOT, *, edge_limit: int = 250) -> dict[str, Any]:
    project_root = Path(project_root)
    rows = _registry_rows(project_root)
    tiers_by_id = {_lower_bot_id(row): classify_tier(row) for row in rows if _lower_bot_id(row)}
    tier_counts = Counter(tiers_by_id.values())
    active_rows = [row for row in rows if _is_active(row)]
    active_tier_counts = Counter(tiers_by_id.get(_lower_bot_id(row), "sub") for row in active_rows)

    hierarchy_edges, hierarchy_stats = _build_hierarchy_edges(rows, tiers_by_id, edge_limit=edge_limit)

    teacher_plan = load_json(project_root / "governance" / "distillation" / "teacher_student_plan_latest.json")
    teacher_quality = load_json(project_root / "governance" / "distillation" / "teacher_quality_latest.json")
    supportability = load_json(project_root / "governance" / "health" / "supportability_control_latest.json")
    training_quality = load_json(project_root / "governance" / "health" / "training_quality_control_latest.json")
    data_rollup = load_json(project_root / "governance" / "health" / "data_collection_observation_rollup_latest.json")
    overfitting_awareness = load_json(project_root / "governance" / "health" / "overfitting_awareness_latest.json")

    teacher_edges, teacher_stats = _teacher_student_edges(teacher_plan, tiers_by_id, edge_limit=edge_limit)
    teacher_summary = _teacher_quality_summary(teacher_quality, teacher_plan)
    overfit_summary = _overfitting_summary(overfitting_awareness)
    quality_contract = _quality_contract(
        training_quality=training_quality,
        data_rollup=data_rollup,
        teacher_quality=teacher_quality,
        supportability=supportability,
        mesh_stats=hierarchy_stats,
        overfitting_awareness=overfitting_awareness,
    )
    missing_tiers = [tier for tier in TIER_ORDER if _safe_int(active_tier_counts.get(tier), 0) <= 0]
    training_score = _safe_float(quality_contract.get("current_training_quality_score"), 0.0)
    data_score = _safe_float(quality_contract.get("current_data_quality_score"), 0.0)
    training_readiness_score = _safe_float(quality_contract.get("current_training_readiness_score"), 0.0)
    communication_score = _communication_readiness_score(
        active_tier_counts=dict(active_tier_counts),
        hierarchy_stats=hierarchy_stats,
        teacher_stats=teacher_stats,
    )
    quality_readiness_score = _quality_readiness_score(
        communication_score=communication_score,
        training_score=training_score,
        data_score=data_score,
        training_readiness_score=training_readiness_score,
        blocker_count=_safe_int(quality_contract.get("blocker_count"), 0),
    )
    overall_status = _status_for_communication(communication_score, missing_tiers)

    tier_samples: dict[str, list[dict[str, Any]]] = {}
    for tier in TIER_ORDER:
        tier_rows = [row for row in rows if tiers_by_id.get(_lower_bot_id(row)) == tier]
        tier_samples[tier] = [_compact_bot(row, tier) for row in tier_rows[:12]]

    needs = [
        str(blocker.get("need") or blocker.get("key") or "")
        for blocker in _as_list(quality_contract.get("blockers"))
        if str(_as_dict(blocker).get("need") or _as_dict(blocker).get("key") or "").strip()
    ]
    recommended_commands = _as_list(quality_contract.get("safe_next_commands"))
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "overall_status": overall_status,
        "communication_readiness_score": communication_score,
        "quality_readiness_score": quality_readiness_score,
        "quality_target_status": "ready" if _safe_int(quality_contract.get("blocker_count"), 0) == 0 else "needs_work",
        "quality_target_note": "100 is treated as an A+ target. The mesh reports gaps honestly and does not inflate scores.",
        "bot_count": len(rows),
        "active_bot_count": len(active_rows),
        "tier_counts": {tier: int(tier_counts.get(tier, 0)) for tier in TIER_ORDER},
        "active_tier_counts": {tier: int(active_tier_counts.get(tier, 0)) for tier in TIER_ORDER},
        "missing_tiers": missing_tiers,
        "tier_samples": tier_samples,
        "communication_matrix": _communication_matrix(),
        "hierarchy_edges": hierarchy_edges,
        "hierarchy_edge_summary": hierarchy_stats,
        "teacher_student_edges": teacher_edges,
        "teacher_student_edge_summary": teacher_stats,
        "teacher_student_intelligence": {
            "summary": teacher_summary,
            "policy": {
                "probation_bots_may_teach": False,
                "runtime_input_debt_bots_may_teach": False,
                "overfit_risk_bots_may_teach": False,
                "high_accuracy_without_generalization_is_provisional": True,
                "teacher_blend": "use elite/strong teachers first, cap weak-role fallback teachers to repair-only guidance",
                "student_update_rule": "students may absorb examples and abstention thresholds, but promotion still requires clean walk-forward evidence",
            },
        },
        "overfitting_awareness": overfit_summary,
        "a_plus_target_contract": quality_contract,
        "what_the_system_needs": ordered_unique(needs)[:24],
        "recommended_commands": recommended_commands,
        "integration_contract": {
            "signal_bus_name": "bot_intelligence_mesh",
            "output_path": str(DEFAULT_OUT_PATH.relative_to(project_root) if DEFAULT_OUT_PATH.is_relative_to(project_root) else DEFAULT_OUT_PATH),
            "does_not_execute_trades": True,
            "protect_live_execution": True,
            "intended_consumers": [
                "system_intelligence_coordinator",
                "supportability_control",
                "teacher_quality_guard",
                "bot_needs_intelligence",
                "overfitting_awareness_layer",
                "training_runtime_control",
            ],
        },
    }
    return payload


def write_outputs(payload: dict[str, Any], out_path: Path = DEFAULT_OUT_PATH) -> None:
    write_payload(out_path, payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the bot tier communication and teacher/student intelligence mesh.")
    parser.add_argument("--edge-limit", type=int, default=250)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(PROJECT_ROOT, edge_limit=args.edge_limit)
    write_outputs(payload, args.out)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "bot_intelligence_mesh "
            f"status={payload['overall_status']} "
            f"score={payload['communication_readiness_score']} "
            f"blockers={payload['a_plus_target_contract']['blocker_count']}"
        )
    return 0 if payload.get("overall_status") != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
