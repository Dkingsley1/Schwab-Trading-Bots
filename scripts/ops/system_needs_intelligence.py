#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import time
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_needs_intelligence_latest.json"
DEFAULT_LOG_PATH = PROJECT_ROOT / "governance" / "health" / "system_needs_fix_log.jsonl"
LOW_GRADE_VALUES = {"D", "F"}
LOW_GRADE_AUDIT_EXCLUDED_FILES = {
    "low_grade_finalizer_latest.json",
    "system_needs_intelligence_latest.json",
}
LOW_GRADE_ARTIFACT_ALIASES = {
    "system_cell_federation_latest.json": "distributed_cell_architecture_latest.json",
}
SOAK_MANAGED_TRAINING_BLOCKERS = {
    "training_runtime_pretraining_drain_buffer_active",
    "training_runtime_autonomic_training_budget_closed",
    "training_runtime_training_quality_blocked",
}
SOAK_MANAGED_GOVERNOR_BLOCKERS = {
    "mlx_or_gpu_lane_capped",
}
SOAK_MANAGED_MEMORY_BLOCKERS = {
    "foreground_app_headroom_reserved",
    "memory_clear_soak_not_finished",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def _command(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _is_grade_field(key: str) -> bool:
    lowered = str(key or "").lower()
    return "grade" in lowered


def _low_grade_command(source_file: str, json_path: str) -> list[Any]:
    text = f"{source_file} {json_path}".lower()
    if "quant_strategy_storage_backlog_accommodation" in text:
        return ["./scripts/ops/opsctl.sh", "quant-storage-backlog-accommodation", "--apply", "--json"]
    if "paper_profitability" in text or "profit_harvest" in text or "profit_grade" in text:
        return ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"]
    if "system_self_intelligence" in text or "codex_handoff" in text or "whole_system_intelligence" in text:
        return ["./scripts/ops/opsctl.sh", "system-intelligence", "--apply", "--json"]
    if "quant_strategy_gap" in text:
        return ["./scripts/ops/opsctl.sh", "quant-strategy-gap", "--apply", "--json"]
    if "backlog" in text or "storage" in text or "ingestion" in text:
        return ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"]
    return ["./scripts/ops/opsctl.sh", "grade-regression-guard", "--json"]


def _low_grade_category(source_file: str, json_path: str) -> str:
    text = f"{source_file} {json_path}".lower()
    if "base_raw" in text:
        return "base_evidence_grade"
    if "profit_grade" in text:
        if "contained" in text:
            return "contained_profit_grade"
        if "probationary" in text:
            return "probationary_profit_grade"
        return "profile_profit_grade"
    if "self_awareness" in text or "awareness_state_vector" in text:
        return "self_awareness_grade"
    if "backlog_letter_grade" in text:
        return "backlog_accommodation_snapshot"
    return "low_grade_layer"


def _low_grade_expected_impact(category: str) -> tuple[str, str]:
    if category == "base_evidence_grade":
        return (
            "Keeps the base/raw grade visible and routes the subsystem toward real outcome improvement instead of only control-credit improvement.",
            "base/raw grade is C or better and the headline/control grade no longer depends on rescue credit.",
        )
    if category in {"contained_profit_grade", "probationary_profit_grade", "profile_profit_grade"}:
        return (
            "Repairs or deweights weak paper profiles using hard-negative labels, tighter entries/exits, and fresh paper evidence.",
            "profile profit grade is C or better, or the profile is explicitly quarantined/probationary with no active new-entry path.",
        )
    if category == "self_awareness_grade":
        return (
            "Refreshes stale self-awareness surfaces so the handoff stops reasoning from old artifacts.",
            "system_self_intelligence.awareness_state_vector.grade is C or better with stale/blind-spot count reduced.",
        )
    if category == "backlog_accommodation_snapshot":
        return (
            "Refreshes the stale quant/backlog accommodation snapshot against current storage truth.",
            "backlog accommodation snapshot is current and backlog_letter_grade is C or better.",
        )
    return (
        "Refreshes the owning health surface and keeps the low grade visible for targeted repair.",
        "the same JSON path no longer reports D/F.",
    )


def _skip_low_grade_path(json_path: str) -> bool:
    lowered = str(json_path or "").lower()
    return bool(
        lowered.startswith("remaining_low_grade_layers.")
        or ".remaining_low_grade_layers." in lowered
        or lowered.startswith("low_grade_layer_summary.")
        or ".low_grade_layer_summary." in lowered
        or lowered.startswith("low_grade_control_report_card.")
        or ".low_grade_control_report_card." in lowered
    )


def _is_embedded_snapshot_path(json_path: str) -> bool:
    parts = [part.lower() for part in str(json_path or "").split(".") if part]
    if "parsed" in parts or "embedded_payload" in parts:
        return True
    snapshot_roots = {"steps", "refresh_steps", "repair_steps", "command_results", "results"}
    if parts and parts[0] in snapshot_roots and "payload" in parts:
        return True
    return bool(parts and parts[0] in {"production_excellence", "system_signal_bus"})


def _is_propagated_grade_path(source_file: str, json_path: str) -> bool:
    source = str(source_file or "").strip().lower()
    lowered = str(json_path or "").strip().lower()
    return bool(
        source == "system_signal_bus_latest.json"
        and lowered.startswith("signals.")
        and ".metrics." in lowered
    )


def _is_historical_grade_path(json_path: str) -> bool:
    return any("historical" in part.lower() for part in str(json_path or "").split("."))


def _low_grade_scope(row: dict[str, Any]) -> str:
    source = str(row.get("exact_file") or "").lower()
    if bool(row.get("stale_artifact", False)) or bool(row.get("embedded_snapshot", False)) or bool(row.get("historical_snapshot", False)):
        return "historical_or_superseded"
    if bool(row.get("propagated_snapshot", False)):
        return "propagated_runtime_signal"
    if any(token in source for token in ("production_excellence", "live_money_readiness", "promotion", "canary")):
        return "live_promotion_evidence"
    if "paper_profitability" in source or "paper_runtime_profitability" in source:
        return "paper_outcome_evidence"
    return "runtime_operational"


def _canonical_low_grade_key(source_file: str, json_path: str, grade: str, category: str) -> tuple[str, str, str]:
    parts = str(json_path or "").split(".")
    if category == "profile_profit_grade":
        for marker in ("active_profile_controls", "profile_controls"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    return (category, f"profile_profit_grade.{parts[idx + 1]}", grade)
    if category in {"contained_profit_grade", "probationary_profit_grade"}:
        return (category, json_path, grade)
    if category == "self_awareness_grade":
        return (category, "system_self_awareness.grade", grade)
    if category == "base_evidence_grade" and "base_raw_outcome_grade" in str(json_path):
        return (category, "profit_harvest_report_card.base_raw_outcome_grade", grade)
    if category == "backlog_accommodation_snapshot":
        return (category, "quant_strategy_storage_backlog_accommodation.storage_snapshot.backlog_letter_grade", grade)
    return (category, f"{source_file}:{json_path}", grade)


def _low_grade_control_context(health: Path) -> dict[str, Any]:
    paper = load_json(health / "paper_profitability_control_latest.json")
    self_intelligence = load_json(health / "system_self_intelligence_latest.json")
    non_blocking_profiles = {
        str(row.get("profile") or "")
        for row in _as_list(_as_dict(paper).get("remaining_low_grade_layers"))
        if isinstance(row, dict) and str(row.get("profile") or "") and not bool(row.get("active_blocker", False))
    }
    return {
        "paper_control_posture_grade": str(
            _as_dict(_as_dict(paper).get("low_grade_control_report_card")).get("control_posture_grade")
            or _as_dict(_as_dict(paper).get("low_grade_layer_summary")).get("control_posture_grade")
            or ""
        ).upper(),
        "paper_active_blocker_count": _safe_int(
            _as_dict(_as_dict(paper).get("low_grade_layer_summary")).get("active_blocker_count"),
            _safe_int(_as_dict(_as_dict(paper).get("low_grade_control_report_card")).get("active_blocker_count"), 999),
        ),
        "paper_non_blocking_profiles": non_blocking_profiles,
        "self_awareness_control_posture_grade": str(
            _as_dict(_as_dict(self_intelligence).get("awareness_state_vector")).get("control_posture_grade")
            or _as_dict(_as_dict(self_intelligence).get("awareness_state_vector")).get("control_grade")
            or ""
        ).upper(),
    }


def _profile_from_canonical_path(canonical_path: str) -> str:
    prefix = "profile_profit_grade."
    text = str(canonical_path or "")
    return text[len(prefix) :] if text.startswith(prefix) else ""


def _low_grade_control_state(row: dict[str, Any], context: dict[str, Any]) -> tuple[str, bool]:
    category = str(row.get("category") or "")
    canonical_path = str(row.get("canonical_json_path") or "")
    if bool(row.get("embedded_snapshot", False)):
        return ("superseded_embedded_snapshot", False)
    if bool(row.get("historical_snapshot", False)):
        return ("historical_evidence_preserved", False)
    if bool(row.get("propagated_snapshot", False)):
        return ("propagated_dependency_signal", False)
    if bool(row.get("stale_artifact", False)):
        return ("stale_artifact_not_current", False)
    if str(row.get("scope") or "") == "live_promotion_evidence":
        return ("live_promotion_evidence_debt", False)
    if category in {"contained_profit_grade", "probationary_profit_grade"}:
        return ("contained_or_probationary", False)
    if category == "profile_profit_grade" and _profile_from_canonical_path(canonical_path) in set(
        context.get("paper_non_blocking_profiles") or set()
    ):
        return ("contained_by_paper_profitability_control", False)
    source = str(row.get("exact_file") or "").lower()
    paper_control_ready = (
        _safe_int(context.get("paper_active_blocker_count"), 999) == 0
        and str(context.get("paper_control_posture_grade") or "") == "A+"
    )
    if paper_control_ready and ("paper_profitability_control" in source or "paper_runtime_profitability_controls" in source):
        return ("raw_paper_outcome_under_a_plus_control", False)
    if (
        category == "base_evidence_grade"
        and canonical_path == "profit_harvest_report_card.base_raw_outcome_grade"
        and _safe_int(context.get("paper_active_blocker_count"), 999) == 0
        and str(context.get("paper_control_posture_grade") or "") in {"A+", "A+"}
    ):
        return ("raw_harvest_evidence_under_a_plus_control", False)
    if category == "self_awareness_grade" and str(context.get("self_awareness_control_posture_grade") or "") in {"A+", "A+"}:
        return ("self_awareness_under_a_plus_control", False)
    return ("actionable_low_grade_blocker", True)


def _low_grade_audit_control_grade(active_blocker_count: int) -> str:
    if active_blocker_count <= 0:
        return "A+"
    if active_blocker_count <= 2:
        return "B"
    if active_blocker_count <= 5:
        return "C"
    return "D"


def _iter_low_grade_fields(payload: Any, path: list[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_path = [*path, str(key)]
            if isinstance(value, str) and value.strip().upper() in LOW_GRADE_VALUES and _is_grade_field(str(key)):
                rows.append((".".join(next_path), value.strip().upper()))
            rows.extend(_iter_low_grade_fields(value, next_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            rows.extend(_iter_low_grade_fields(value, [*path, str(index)]))
    return rows


def _low_grade_layer_audit(project_root: Path) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    control_context = _low_grade_control_context(health)
    hits: list[dict[str, Any]] = []
    duplicate_sources: dict[tuple[str, str, str], int] = {}
    canonical: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_alias_file_count = 0
    for path in sorted(health.glob("*latest*.json")):
        if path.name in LOW_GRADE_AUDIT_EXCLUDED_FILES:
            continue
        canonical_alias = LOW_GRADE_ARTIFACT_ALIASES.get(path.name)
        if canonical_alias and load_json(path) == load_json(health / canonical_alias):
            duplicate_alias_file_count += 1
            continue
        try:
            artifact_age_hours = max(0.0, (time() - path.stat().st_mtime) / 3600.0)
        except Exception:
            artifact_age_hours = 0.0
        stale_artifact = artifact_age_hours >= 24.0
        payload = load_json(path)
        if not payload:
            continue
        for json_path, grade in _iter_low_grade_fields(payload, []):
            if _skip_low_grade_path(json_path):
                continue
            category = _low_grade_category(path.name, json_path)
            expected_impact, stop_when = _low_grade_expected_impact(category)
            command = _low_grade_command(path.name, json_path)
            key = _canonical_low_grade_key(path.name, json_path, grade, category)
            duplicate_sources[key] = duplicate_sources.get(key, 0) + 1
            if key not in canonical:
                canonical[key] = {
                    "layer_id": f"{key[0]}:{key[1]}",
                    "category": category,
                    "current_grade": grade,
                    "exact_file": str(path.relative_to(project_root)),
                    "exact_json_path": json_path,
                    "canonical_json_path": key[1],
                    "artifact_age_hours": round(float(artifact_age_hours), 3),
                    "stale_artifact": bool(stale_artifact),
                    "embedded_snapshot": _is_embedded_snapshot_path(json_path),
                    "historical_snapshot": _is_historical_grade_path(json_path),
                    "propagated_snapshot": _is_propagated_grade_path(path.name, json_path),
                    "command": command,
                    "expected_impact": expected_impact,
                    "risk_level": "low",
                    "when_to_stop": stop_when,
                    "source": "low_grade_layer_audit",
                }
            hits.append({"file": str(path.relative_to(project_root)), "json_path": json_path, "grade": grade, "category": category})
    layers = list(canonical.values())
    for row in layers:
        key = (str(row.get("category") or ""), str(row.get("canonical_json_path") or ""), str(row.get("current_grade") or ""))
        row["duplicate_surface_count"] = duplicate_sources.get(key, 1)
        row["scope"] = _low_grade_scope(row)
        control_state, active_blocker = _low_grade_control_state(row, control_context)
        row["control_state"] = control_state
        row["active_blocker"] = bool(active_blocker)
        row["effective_grade"] = str(row.get("current_grade") or "")
        row["raw_grade_preserved"] = True
    layers.sort(
        key=lambda row: (
            0 if bool(row.get("active_blocker", False)) else 1,
            1 if bool(row.get("stale_artifact", False)) else 0,
            0 if str(row.get("category") or "") in {"base_evidence_grade", "self_awareness_grade", "backlog_accommodation_snapshot"} else 1,
            str(row.get("category") or ""),
            str(row.get("exact_file") or ""),
            str(row.get("exact_json_path") or ""),
        )
    )
    by_category: dict[str, int] = {}
    for row in layers:
        category = str(row.get("category") or "low_grade_layer")
        by_category[category] = by_category.get(category, 0) + 1
    active_blocker_count = sum(1 for row in layers if bool(row.get("active_blocker", False)))
    stale_artifact_count = sum(1 for row in layers if bool(row.get("stale_artifact", False)))
    embedded_snapshot_count = sum(1 for row in layers if bool(row.get("embedded_snapshot", False)))
    historical_snapshot_count = sum(1 for row in layers if bool(row.get("historical_snapshot", False)))
    propagated_snapshot_count = sum(1 for row in layers if bool(row.get("propagated_snapshot", False)))
    promotion_evidence_layer_count = sum(1 for row in layers if str(row.get("scope") or "") == "live_promotion_evidence")
    contained_or_controlled_count = sum(1 for row in layers if not bool(row.get("active_blocker", False)))
    effective_low_grade_layer_count = sum(1 for row in layers if str(row.get("effective_grade") or row.get("current_grade") or "").upper() in LOW_GRADE_VALUES)
    next_commands: list[list[Any]] = []
    seen_commands: set[tuple[str, ...]] = set()
    for row in [row for row in layers if bool(row.get("active_blocker", False))] or layers:
        command = _command(row.get("command"))
        key = tuple(str(part) for part in command)
        if command and key not in seen_commands:
            seen_commands.add(key)
            next_commands.append(command)
    return {
        "active": bool(layers),
        "raw_hit_count": len(hits),
        "unique_low_grade_layer_count": len(layers),
        "active_blocker_count": active_blocker_count,
        "actionable_low_grade_layer_count": active_blocker_count,
        "effective_low_grade_layer_count": effective_low_grade_layer_count,
        "contained_or_controlled_count": contained_or_controlled_count,
        "stale_artifact_count": stale_artifact_count,
        "embedded_snapshot_count": embedded_snapshot_count,
        "historical_snapshot_count": historical_snapshot_count,
        "propagated_snapshot_count": propagated_snapshot_count,
        "promotion_evidence_layer_count": promotion_evidence_layer_count,
        "duplicate_alias_file_count": duplicate_alias_file_count,
        "control_posture_grade": _low_grade_audit_control_grade(active_blocker_count),
        "control_posture_status": "a_plus_control_ready" if active_blocker_count == 0 else "actionable_low_grade_blockers",
        "finalization_contract": {
            "active": True,
            "mode": "truthful_low_grade_classification_v2",
            "effective_control_posture_grade": _low_grade_audit_control_grade(active_blocker_count),
            "raw_grades_preserved": True,
            "rewrites_raw_evidence": False,
            "cosmetic_grade_uplift_allowed": False,
        },
        "by_category": by_category,
        "layers": layers,
        "actionable_layers": [row for row in layers if bool(row.get("active_blocker", False))],
        "next_commands": next_commands,
        "reporting_rule": "D/F evidence is never relabeled. Current blockers, controlled outcomes, stale artifacts, propagated signals, and superseded embedded snapshots are classified separately.",
    }


def _need_from_low_grade_audit(audit: dict[str, Any]) -> list[dict[str, Any]]:
    if _safe_int(audit.get("active_blocker_count"), 0) <= 0:
        return []
    layers = _as_list(audit.get("actionable_layers")) or _as_list(audit.get("layers"))
    if not layers:
        return []
    top = layers[0] if isinstance(layers[0], dict) else {}
    return [
        {
            "blocker": "low_grade_layers_still_present",
            "exact_file": top.get("exact_file", "governance/health"),
            "exact_shard": top.get("exact_json_path", ""),
            "command": _command(top.get("command")),
            "expected_impact": (
                f"Surfaces and starts the first repair path for {audit.get('unique_low_grade_layer_count', 0)} "
                "remaining D/F grade layers instead of hiding them behind headline grades."
            ),
            "risk_level": "low",
            "when_to_stop": "low_grade_layer_audit.unique_low_grade_layer_count is 0, or every remaining low grade is marked contained/probationary with an explicit repair path.",
            "source": "low_grade_layer_audit",
            "low_grade_layer_count": _safe_int(audit.get("unique_low_grade_layer_count"), 0),
            "low_grade_categories": _as_dict(audit.get("by_category")),
        }
    ]


def _profitability_grade_below_a(value: Any) -> bool:
    grade = str(value or "").strip().upper()
    return bool(grade and grade not in {"A", "A+"})


def _nested_dict(payload: dict[str, Any], *path: str) -> dict[str, Any]:
    current: Any = payload
    for key in path:
        current = _as_dict(current).get(key)
    return _as_dict(current)


def _first_nested_dict(*values: dict[str, Any]) -> dict[str, Any]:
    for value in values:
        if value:
            return value
    return {}


def _loss_cause_names(*contracts: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for contract in contracts:
        for row in _as_list(contract.get("top_loss_causes")):
            if isinstance(row, dict):
                cause = str(row.get("cause") or "").strip()
            else:
                cause = str(row or "").strip()
            if cause and cause not in names:
                names.append(cause)
    return names


def _raw_profitability_recovery_context(
    *,
    paper_profitability: dict[str, Any],
    paper_runtime_profitability: dict[str, Any],
    live_canary_readiness: dict[str, Any],
) -> dict[str, Any]:
    source = paper_profitability if paper_profitability else paper_runtime_profitability
    runtime_source = paper_runtime_profitability if paper_runtime_profitability else paper_profitability
    raw_grade = str(
        source.get("raw_profitability_grade")
        or runtime_source.get("raw_profitability_grade")
        or ""
    ).strip().upper()
    controlled_grade = str(
        source.get("controlled_profitability_grade")
        or runtime_source.get("controlled_profitability_grade")
        or source.get("profitability_grade")
        or runtime_source.get("profitability_grade")
        or ""
    ).strip().upper()
    financial_grade = str(
        source.get("financial_profitability_grade")
        or runtime_source.get("financial_profitability_grade")
        or source.get("financial_display_grade")
        or runtime_source.get("financial_display_grade")
        or ""
    ).strip().upper()
    a_plus = _first_nested_dict(
        _nested_dict(paper_runtime_profitability, "a_plus_target_contract"),
        _nested_dict(paper_profitability, "a_plus_target_contract"),
    )
    current = _as_dict(a_plus.get("current"))
    thresholds = _as_dict(a_plus.get("thresholds"))
    raw_improvement = _first_nested_dict(
        _nested_dict(paper_runtime_profitability, "raw_profitability_improvement_contract"),
        _nested_dict(paper_profitability, "raw_profitability_improvement_contract"),
    )
    raw_a_recovery = _first_nested_dict(
        _nested_dict(paper_runtime_profitability, "raw_profitability_a_recovery_contract"),
        _nested_dict(paper_profitability, "raw_profitability_a_recovery_contract"),
    )
    raw_six = _first_nested_dict(
        _nested_dict(paper_runtime_profitability, "raw_profitability_six_point_recovery_contract"),
        _nested_dict(paper_profitability, "raw_profitability_six_point_recovery_contract"),
    )
    burn_down = _first_nested_dict(
        _nested_dict(raw_improvement, "burn_down_contract"),
        _nested_dict(raw_a_recovery, "burn_down_contract"),
        _nested_dict(raw_six, "burn_down_contract"),
    )
    loss_feedback = _first_nested_dict(
        _nested_dict(raw_improvement, "loss_cause_training_feedback_contract"),
        _nested_dict(raw_six, "loss_cause_filter_contract"),
        raw_a_recovery,
    )
    top_loss_causes = _loss_cause_names(loss_feedback, raw_a_recovery)
    requirements = _as_list(raw_improvement.get("requirements"))
    ready_requirement_count = sum(1 for row in requirements if isinstance(row, dict) and bool(row.get("ready", False)))
    live_blockers = [
        str(item or "").strip()
        for item in _as_list(live_canary_readiness.get("blockers"))
        if str(item or "").strip()
    ]
    raw_live_blockers = [item for item in live_blockers if "raw_profitability" in item]
    net_pnl = _safe_float(current.get("net_pnl"), _safe_float(burn_down.get("current_net_pnl"), 0.0))
    raw_ready = bool(
        raw_grade in {"A", "A+"}
        and net_pnl >= 0.0
        and not raw_live_blockers
        and (not bool(a_plus) or bool(a_plus.get("combined_a_plus_ready", True)))
    )
    active = bool(
        raw_grade
        and (
            _profitability_grade_below_a(raw_grade)
            or net_pnl < 0.0
            or _profitability_grade_below_a(financial_grade)
            or raw_live_blockers
            or (bool(a_plus) and not bool(a_plus.get("combined_a_plus_ready", False)))
        )
    )
    return {
        "active": bool(active and not raw_ready),
        "raw_profitability_grade": raw_grade,
        "controlled_profitability_grade": controlled_grade,
        "financial_profitability_grade": financial_grade,
        "net_pnl": net_pnl,
        "realized_pnl": _safe_float(current.get("realized_pnl"), 0.0),
        "unrealized_pnl": _safe_float(current.get("unrealized_pnl"), 0.0),
        "change_vs_previous_day": _safe_float(current.get("change_vs_previous_day"), 0.0),
        "executions": _safe_int(current.get("executions"), 0),
        "weak_profile_count": _safe_int(current.get("weak_profile_count"), 0),
        "strategy_control_count": _safe_int(current.get("strategy_control_count"), 0),
        "unprotected_weak_profile_count": _safe_int(current.get("unprotected_weak_profile_count"), 0),
        "unprotected_strategy_control_count": _safe_int(current.get("unprotected_strategy_control_count"), 0),
        "min_net_pnl": _safe_float(thresholds.get("min_net_pnl"), 0.0),
        "combined_a_plus_ready": bool(a_plus.get("combined_a_plus_ready", False)) if a_plus else False,
        "daily_net_improvement_target": max(
            _safe_float(burn_down.get("required_average_daily_net_improvement"), 0.0),
            _safe_float(_as_dict(raw_improvement.get("runtime_enforcement")).get("raw_d_daily_net_improvement_target"), 0.0),
        ),
        "top_loss_causes": top_loss_causes,
        "requirement_count": len(requirements),
        "ready_requirement_count": ready_requirement_count,
        "all_requirements_ready": bool(requirements) and ready_requirement_count == len(requirements),
        "runtime_enforcement": _as_dict(raw_improvement.get("runtime_enforcement")),
        "top_drag_profiles": _as_list(burn_down.get("top_drag_profiles"))[:5],
        "largest_drag_profile": _as_dict(burn_down.get("largest_drag_profile")),
        "live_canary_raw_blockers": raw_live_blockers,
        "source_file": (
            "governance/health/paper_runtime_profitability_controls_latest.json"
            if paper_runtime_profitability
            else "governance/health/paper_profitability_control_latest.json"
        ),
        "stop_condition": (
            str(raw_improvement.get("stop_condition") or "")
            or str(raw_a_recovery.get("stop_condition") or "")
            or str(raw_six.get("stop_condition") or "")
            or "raw_profitability_grade is A or better and net_pnl_total >= 0"
        ),
    }


def _need_from_raw_profitability_recovery(context: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(context.get("active", False)):
        return []
    top_drags = []
    for row in _as_list(context.get("top_drag_profiles")):
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "").strip()
        if profile:
            top_drags.append(profile)
    evidence = [
        f"raw_profitability_grade={context.get('raw_profitability_grade') or 'unknown'}",
        f"controlled_profitability_grade={context.get('controlled_profitability_grade') or 'unknown'}",
        f"financial_profitability_grade={context.get('financial_profitability_grade') or 'unknown'}",
        f"raw_net_pnl={_safe_float(context.get('net_pnl'), 0.0):.6f}",
        f"realized_pnl={_safe_float(context.get('realized_pnl'), 0.0):.6f}",
        f"unrealized_pnl={_safe_float(context.get('unrealized_pnl'), 0.0):.6f}",
        f"change_vs_previous_day={_safe_float(context.get('change_vs_previous_day'), 0.0):.6f}",
        f"weak_profile_count={_safe_int(context.get('weak_profile_count'), 0)}",
        f"strategy_control_count={_safe_int(context.get('strategy_control_count'), 0)}",
        f"daily_net_improvement_target={_safe_float(context.get('daily_net_improvement_target'), 0.0):.6f}",
        f"top_loss_causes={','.join(_as_list(context.get('top_loss_causes'))[:6]) or 'none'}",
        f"top_drag_profiles={','.join(top_drags[:5]) or 'none'}",
        f"live_canary_raw_blockers={','.join(_as_list(context.get('live_canary_raw_blockers'))[:6]) or 'none'}",
        f"raw_recovery_requirements={_safe_int(context.get('ready_requirement_count'), 0)}/{_safe_int(context.get('requirement_count'), 0)}",
    ]
    return [
        {
            "blocker": "raw_profitability_burn_down",
            "exact_file": str(context.get("source_file") or "governance/health/paper_runtime_profitability_controls_latest.json"),
            "exact_shard": "raw_profitability_improvement_contract",
            "command": ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
            "expected_impact": (
                "Keeps raw PnL recovery visible and routes zero-entry weak sleeves, reduce-only drag burn-down, "
                "strict clean-sleeve admission, loss-cause filters, training feedback, and three-profitable-refresh re-entry."
            ),
            "risk_level": "low",
            "when_to_stop": (
                "raw_profitability_grade is A or better, raw net PnL is non-negative, weak profiles and losing strategy pairs "
                "have three profitable refreshes or remain quarantined, and live-canary raw profitability blockers are empty."
            ),
            "source": "raw_profitability_recovery",
            "evidence": evidence,
            "target_capabilities": [
                "paper_profitability_control",
                "paper_performance_refresh",
                "runtime_paper_regression_guard",
                "paper_execution_truth_layer",
                "training_data_intake_labeling",
                "training_labeling_intelligence",
                "master_grandmaster_profitability_trainer",
                "live_canary_readiness_contract",
            ],
            "control_policy": {
                "do_not_force_trades": True,
                "paper_only": True,
                "live_execution_allowed": False,
                "raw_truth_preserved": True,
            },
        }
    ]


def _load_fix_log(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    for raw in lines[-limit:]:
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _append_fix_log(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _need_from_governor(governor: dict[str, Any]) -> list[dict[str, Any]]:
    needs = _as_list(_as_dict(governor.get("what_do_you_need")).get("items"))
    out: list[dict[str, Any]] = []
    for item in needs:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "blocker": item.get("blocker", "unknown"),
                "exact_file": item.get("exact_file", ""),
                "exact_shard": item.get("exact_shard", ""),
                "command": item.get("command", []),
                "expected_impact": item.get("expected_impact", ""),
                "risk_level": item.get("risk_level", "unknown"),
                "when_to_stop": item.get("stop_when", ""),
                "source": "autonomic_resource_governor",
            }
        )
    return out


def _need_from_storage(storage: dict[str, Any]) -> list[dict[str, Any]]:
    needs: list[dict[str, Any]] = []
    backpressure = _as_dict(storage.get("backpressure"))
    stale = _as_dict(storage.get("stale_pending_locator"))
    oldest = _as_list(stale.get("oldest_sources"))
    core = _safe_int(backpressure.get("core_pending_lines"), 0)
    target = _safe_int(backpressure.get("pending_lines_threshold"), 5000) or 5000
    if core > target and not oldest:
        needs.append(
            {
                "blocker": "core_backlog_above_target_without_source_locator",
                "exact_file": "governance/health/ingestion_storage_control_latest.json",
                "exact_shard": "",
                "command": ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
                "expected_impact": "Refreshes truth reconciliation and stale pending source locator.",
                "risk_level": "none",
                "when_to_stop": "stale_pending_locator has oldest_sources or core backlog is under target.",
                "source": "ingestion_storage_control",
            }
        )
    return needs


def _need_from_memory(memory: dict[str, Any]) -> list[dict[str, Any]]:
    needs = _as_list(_as_dict(memory.get("what_do_you_need")).get("items"))
    out: list[dict[str, Any]] = []
    for item in needs:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "blocker": item.get("blocker", "unknown"),
                "exact_file": item.get("exact_file", ""),
                "exact_shard": item.get("exact_shard", ""),
                "command": item.get("command", []),
                "expected_impact": item.get("expected_impact", ""),
                "risk_level": item.get("risk_level", "unknown"),
                "when_to_stop": item.get("stop_when", ""),
                "source": "memory_pressure_intelligence",
            }
        )
    return out


def _need_from_training_runtime(training_runtime: dict[str, Any]) -> list[dict[str, Any]]:
    contract = _as_dict(training_runtime.get("training_launch_contract"))
    blockers = [str(item or "").strip() for item in _as_list(contract.get("launch_blockers")) if str(item or "").strip()]
    if bool(contract.get("launch_allowed", False)) or not blockers:
        return []
    prep_commands = [_command(item) for item in _as_list(contract.get("recommended_prep_commands"))]
    quota_gate = _as_dict(contract.get("storage_quota_training_gate"))
    blocked_quota_families = [str(item or "").strip() for item in _as_list(quota_gate.get("blocked_families")) if str(item or "").strip()]

    def command_for(blocker: str) -> list[Any]:
        command_needle = ""
        if "storage_quota" in blocker:
            if "governance_telemetry" in blocked_quota_families:
                return ["./scripts/ops/opsctl.sh", "governance-telemetry-compactor", "--apply", "--json"]
            command_needle = "storage-quota-guard"
        elif "writer" in blocker or "drain" in blocker:
            command_needle = "writer-cycle-coordinator"
        elif "memory" in blocker or "headroom" in blocker or "multitasking" in blocker:
            command_needle = "memory-pressure-intelligence"
        elif "runtime_snapshot" in blocker:
            command_needle = "runtime-training-snapshot"
        for command in prep_commands:
            if command_needle and command_needle in " ".join(str(part) for part in command):
                return command
        if prep_commands:
            return prep_commands[0]
        return ["./scripts/ops/opsctl.sh", "training-runtime-control", "--limit", str(_safe_int(contract.get("requested_batch_size"), 30) or 30), "--json"]

    needs: list[dict[str, Any]] = []
    for blocker in blockers[:4]:
        if "storage_quota" in blocker:
            expected = "Refreshes storage quota truth and keeps batch training gated until hard-breached families are below quota."
            if "governance_telemetry" in blocked_quota_families:
                expected = "Rotates oversized governance channel telemetry out of the hot quota lane, then lets batch training recheck storage quota truth."
            stop = "storage_quota_training_gate.status is ready and storage_quota_hard_breach is gone from launch_blockers."
        else:
            expected = "Refreshes the training launch contract and clears the next prep step before widening retrains."
            stop = "training_launch_contract.launch_allowed is true or the blocker list changes."
        needs.append(
            {
                "blocker": f"training_runtime_{blocker}",
                "exact_file": "governance/health/training_runtime_control_latest.json",
                "exact_shard": "",
                "command": command_for(blocker),
                "expected_impact": expected,
                "risk_level": "low",
                "when_to_stop": stop,
                "source": "training_runtime_control",
            }
        )
    return needs


def _list_of_strings(value: Any) -> list[str]:
    return [str(item or "").strip() for item in _as_list(value) if str(item or "").strip()]


def _status_needs_repair(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text and text not in {"ok", "ready", "running", "healthy", "clear", "stable"})


def _soak_management_context(project_root: Path, health_fast: dict[str, Any]) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    soak = load_json(health / "unattended_soak_readiness_latest.json")
    paper_guard = load_json(health / "runtime_paper_regression_guard_latest.json")
    soak_status = str(soak.get("overall_status") or soak.get("status") or "").strip().lower()
    soak_grade = str(soak.get("overall_grade") or soak.get("grade") or "").strip().upper()
    soak_ready = bool(soak.get("safe_to_leave_unattended", False)) and soak_status in {"ready", "ok", "healthy"}
    if soak_grade and soak_grade not in {"A", "A+"}:
        soak_ready = False
    paper_status = str(paper_guard.get("overall_status") or paper_guard.get("status") or "").strip().lower()
    paper_guard_clean = (
        bool(paper_guard.get("ok", False))
        and paper_status in {"ready", "ok", "healthy"}
        and _safe_int(paper_guard.get("failed_guard_count"), 0) <= 0
        and not _as_list(paper_guard.get("failed_guards"))
    )
    health_status = str(health_fast.get("overall_status") or health_fast.get("status") or "").strip().lower()
    return {
        "enabled": bool(soak_ready and paper_guard_clean),
        "soak_ready": bool(soak_ready),
        "soak_status": soak_status,
        "soak_grade": soak_grade,
        "paper_guard_clean": bool(paper_guard_clean),
        "paper_guard_status": paper_status,
        "paper_stage": str(paper_guard.get("paper_stage") or ""),
        "paper_armed": bool(paper_guard.get("paper_armed", False)),
        "paper_blocked": bool(paper_guard.get("paper_blocked", False)),
        "failed_guard_count": _safe_int(paper_guard.get("failed_guard_count"), 0),
        "health_fast_status": health_status,
    }


def _managed_soak_reason(item: dict[str, Any], context: dict[str, Any]) -> str:
    if not bool(context.get("enabled", False)):
        return ""
    blocker = str(item.get("blocker") or "")
    source = str(item.get("source") or "")
    if blocker in SOAK_MANAGED_TRAINING_BLOCKERS and source == "training_runtime_control":
        return "training_expansion_parked_for_unattended_soak"
    if blocker in SOAK_MANAGED_GOVERNOR_BLOCKERS and source == "autonomic_resource_governor":
        return "optional_mlx_capacity_deferred_during_unattended_soak"
    if blocker in SOAK_MANAGED_MEMORY_BLOCKERS and source == "memory_pressure_intelligence":
        if blocker == "memory_clear_soak_not_finished":
            return "memory_widening_soak_deferred_during_unattended_soak"
        return "foreground_headroom_reserved_for_unattended_soak"
    return ""


def _split_managed_soak_controls(
    needs: list[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actionable: list[dict[str, Any]] = []
    managed: list[dict[str, Any]] = []
    for item in needs:
        reason = _managed_soak_reason(item, context)
        if not reason:
            actionable.append(item)
            continue
        row = dict(item)
        row.update(
            {
                "managed_control_state": reason,
                "managed_by": "unattended_soak_readiness",
                "soak_ready": bool(context.get("soak_ready", False)),
                "paper_guard_clean": bool(context.get("paper_guard_clean", False)),
                "paper_stage": str(context.get("paper_stage") or ""),
                "paper_armed": bool(context.get("paper_armed", False)),
                "action_policy": "defer_until_soak_not_green_or_operator_explicitly_widens_training_mlx",
                "when_to_unmanage": (
                    "surface as an actionable need if unattended-soak-readiness is no longer ready/safe, "
                    "runtime-paper-regression-guard has failed guards, or the blocker changes into a storage, "
                    "writer, runtime, or live-paper defect."
                ),
            }
        )
        managed.append(row)
    return actionable, managed


def _need_from_runtime_surfaces(
    *,
    health_fast: dict[str, Any],
    process_watchdog: dict[str, Any],
    health_gates: dict[str, Any],
    collector_contracts: dict[str, Any],
    global_halt: dict[str, Any],
    paper_ramp: dict[str, Any],
    plumbing: dict[str, Any],
) -> list[dict[str, Any]]:
    needs: list[dict[str, Any]] = []
    health_fast_process = _as_dict(health_fast.get("process_watchdog"))
    all_sleeves = _as_dict(health_fast_process.get("all_sleeves_effective_runtime"))
    if not all_sleeves:
        all_sleeves = _as_dict(process_watchdog.get("all_sleeves_effective_runtime"))
    all_sleeves_status = str(all_sleeves.get("status") or "")
    launcher_live = bool(all_sleeves.get("launcher_live", True))
    child_fanout_ok = bool(all_sleeves.get("child_fanout_ok", True))
    heartbeat_ok = bool(all_sleeves.get("heartbeat_fresh", all_sleeves.get("heartbeat", True)))
    if all_sleeves and (_status_needs_repair(all_sleeves_status) or not launcher_live or not child_fanout_ok or not heartbeat_ok):
        needs.append(
            {
                "blocker": "all_sleeves_launcher_fanout_needs_repair",
                "exact_file": "governance/health/process_watchdog_latest.json",
                "exact_shard": "all_sleeves",
                "command": ["./scripts/ops/opsctl.sh", "start", "--paper", "--run-all-sleeves"],
                "expected_impact": "Restarts the paper all-sleeves launcher path so the launcher heartbeat and child fanout converge on the active paper sleeve set.",
                "risk_level": "medium",
                "when_to_stop": "all_sleeves_effective_runtime.status is ready/running, launcher_live is true, child_fanout_ok is true, and heartbeat is fresh.",
                "source": "process_watchdog",
            }
        )

    hard_gates = _as_dict(health_gates.get("hard_gates"))
    inputs = _as_dict(health_gates.get("inputs"))
    collector_required_failures = _list_of_strings(
        collector_contracts.get("required_failures")
        or inputs.get("collector_required_failures")
        or _as_dict(health_fast.get("collector_contracts")).get("required_failures")
    )
    if collector_required_failures:
        needs.append(
            {
                "blocker": "collector_contracts_required_failures",
                "exact_file": "governance/health/collector_contracts_latest.json",
                "exact_shard": ",".join(collector_required_failures),
                "command": ["./scripts/ops/opsctl.sh", "source-verification-refresh", "--apply", "--json"],
                "expected_impact": "Refreshes required context collectors and reruns the collector contract evidence used by health gates.",
                "risk_level": "low",
                "when_to_stop": "collector_contracts.required_failures is empty and health_gates.hard_gates.collector_contracts is false.",
                "source": "collector_contracts",
            }
        )

    if bool(hard_gates.get("ingestion_backpressure_overload", False)):
        override = _as_dict(inputs.get("backpressure_storage_control_override"))
        command = ["./scripts/ops/opsctl.sh", "health-gates", "--json"] if bool(override.get("active", False)) else [
            "./scripts/ops/opsctl.sh",
            "storage-pressure-clearance",
            "--apply",
            "--json",
        ]
        needs.append(
            {
                "blocker": "ingestion_backpressure_health_gate_needs_reconciliation",
                "exact_file": "governance/health/health_gates_latest.json",
                "exact_shard": "ingestion_backpressure_overload",
                "command": command,
                "expected_impact": "Refreshes stale backpressure evidence against storage-control truth; storage-pressure-clearance keeps duplicate writer/drain jobs out when autopilot is already active.",
                "risk_level": "low",
                "when_to_stop": "health_gates.hard_gates.ingestion_backpressure_overload is false or storage-control override is active with queue_clear true.",
                "source": "health_gates",
            }
        )

    paper_blockers = _list_of_strings(paper_ramp.get("blockers"))
    halt_clear = bool(
        not global_halt.get("halt", False)
        and not global_halt.get("global_halt", False)
        and not global_halt.get("halt_latched", False)
        and not global_halt.get("halt_required", False)
        and not global_halt.get("would_rehalt", False)
        and not _list_of_strings(global_halt.get("clear_blockers"))
    )
    if halt_clear and "global_halt_or_clear_blocker_active" in paper_blockers:
        needs.append(
            {
                "blocker": "paper_ramp_global_halt_state_stale",
                "exact_file": "governance/health/paper_400_ramp_latest.json",
                "exact_shard": "global_halt",
                "command": ["./scripts/ops/opsctl.sh", "paper-400-ramp", "--apply", "--json"],
                "expected_impact": "Recomputes the paper ramp gate from the current clean global-halt artifact instead of a stale clear-blocker snapshot.",
                "risk_level": "low",
                "when_to_stop": "paper_400_ramp.gates.global_halt.ok is true and global_halt_or_clear_blocker_active is absent from blockers.",
                "source": "paper_400_ramp",
            }
        )

    platform_repair = _as_dict(_as_dict(health_fast.get("operational_readiness")).get("platform_repair"))
    platform_issues = _list_of_strings(platform_repair.get("issues"))
    plumbing_status = str(plumbing.get("overall_status") or "")
    if platform_issues or _status_needs_repair(plumbing_status):
        command = _command(platform_repair.get("next_best_command")) or [
            "./scripts/ops/opsctl.sh",
            "system-plumbing-control",
            "--json",
        ]
        needs.append(
            {
                "blocker": "platform_plumbing_repair_needed",
                "exact_file": "governance/health/system_plumbing_control_latest.json",
                "exact_shard": ",".join(platform_issues),
                "command": command,
                "expected_impact": "Refreshes platform plumbing/hardening evidence and routes the next repair command when operational readiness is degraded.",
                "risk_level": "low",
                "when_to_stop": "system_plumbing_control.overall_status is ready and platform_repair.issues is empty.",
                "source": "system_plumbing_control",
            }
        )

    return needs


def _ready_actions_from_training_runtime(training_runtime: dict[str, Any]) -> list[dict[str, Any]]:
    contract = _as_dict(training_runtime.get("training_launch_contract"))
    if not bool(contract.get("launch_allowed", False)):
        return []
    command = _command(contract.get("recommended_retrain_command"))
    if not command:
        return []
    host_gate = _as_dict(contract.get("host_training_headroom_gate"))
    batch_size = _safe_int(contract.get("recommended_batch_size"), 0)
    profile = str(host_gate.get("selected_training_profile") or host_gate.get("governor_profile") or "")
    batch20_mode = str(host_gate.get("batch20_execution_mode") or "")
    wave_size = _safe_int(host_gate.get("batch20_wave_size"), 0)
    batch30_mode = str(host_gate.get("batch30_execution_mode") or "")
    batch30_wave_size = _safe_int(host_gate.get("batch30_wave_size"), 0)
    quality_recovery = bool(contract.get("training_quality_recovery_canary", False))
    expected = f"Runs the guarded {batch_size}-bot retrain batch under {profile or 'the selected canary profile'}."
    if quality_recovery:
        expected += " This is a quality-recovery canary with master promotion skipped."
    if batch20_mode == "sequential_memory_guarded_waves" and wave_size > 0:
        expected += f" Batch-20 is executed as sequential memory-guarded waves of {wave_size}."
    if batch30_mode == "sequential_memory_guarded_waves" and batch30_wave_size > 0:
        expected += f" Batch-30 is executed as sequential memory-guarded waves of {batch30_wave_size}."
    return [
        {
            "action": "run_guarded_training_batch",
            "exact_file": "governance/health/training_runtime_control_latest.json",
            "command": command,
            "expected_impact": expected,
            "risk_level": "medium" if batch_size >= 20 else "low",
            "when_to_stop": "stop if memory pressure rises, thermal guard trips, or any target fails outside the recovery canary guard.",
            "source": "training_runtime_control",
            "batch_size": batch_size,
            "profile": profile,
            "batch20_execution_mode": batch20_mode,
            "batch20_wave_size": wave_size,
            "batch30_execution_mode": batch30_mode,
            "batch30_wave_size": batch30_wave_size,
            "quality_recovery_canary": quality_recovery,
        }
    ]


def _need_from_uniform_hardening(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not payload:
        return [
            {
                "blocker": "uniform_hardening_contract_missing",
                "exact_file": "governance/health/uniform_hardening_contract_latest.json",
                "exact_shard": "",
                "command": ["./scripts/ops/opsctl.sh", "uniform-hardening", "--json"],
                "expected_impact": "Builds the shared structural and freshness floor across every critical production domain.",
                "risk_level": "low",
                "when_to_stop": "uniform_floor_ready and critical_runtime_ready are both true.",
                "source": "uniform_hardening_contract",
            }
        ]
    structural = _list_of_strings(payload.get("structural_blockers"))
    critical = _list_of_strings(payload.get("critical_runtime_blockers"))
    if not structural and not critical:
        return []
    commands = _as_list(payload.get("recommended_recovery_commands"))
    command = commands[0] if commands and isinstance(commands[0], list) else ["./scripts/ops/opsctl.sh", "uniform-hardening", "--json"]
    blocker = "uniform_structural_floor_not_ready" if structural else "uniform_critical_runtime_not_ready"
    return [
        {
            "blocker": blocker,
            "exact_file": "governance/health/uniform_hardening_contract_latest.json",
            "exact_shard": ",".join((structural or critical)[:8]),
            "command": command,
            "expected_impact": "Repairs the first failed shared control or stale critical artifact without relabeling domain evidence.",
            "risk_level": "low",
            "when_to_stop": "uniform_floor_ready and critical_runtime_ready are both true; evidence-only debt remains separately visible.",
            "source": "uniform_hardening_contract",
        }
    ]


def build_payload(project_root: Path = PROJECT_ROOT, *, fix_log_path: Path = DEFAULT_LOG_PATH) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    governor = load_json(health / "autonomic_resource_governor_latest.json")
    storage = load_json(health / "ingestion_storage_control_latest.json")
    writer = load_json(health / "writer_cycle_coordinator_latest.json")
    runtime = load_json(health / "runtime_throttle_control_latest.json")
    benchmark = load_json(health / "host_self_benchmark_latest.json")
    migration = load_json(health / "migration_readiness_report_latest.json")
    memory = load_json(health / "memory_pressure_intelligence_latest.json")
    training_runtime = load_json(health / "training_runtime_control_latest.json")
    health_fast = load_json(health / "health_fast_latest.json")
    process_watchdog = load_json(health / "process_watchdog_latest.json")
    health_gates = load_json(health / "health_gates_latest.json")
    collector_contracts = load_json(health / "collector_contracts_latest.json")
    global_halt = load_json(health / "global_halt_auto_clear_latest.json") or load_json(health / "global_killswitch_latest.json")
    paper_ramp = load_json(health / "paper_400_ramp_latest.json")
    plumbing = load_json(health / "system_plumbing_control_latest.json")
    paper_profitability = load_json(health / "paper_profitability_control_latest.json")
    paper_runtime_profitability = load_json(health / "paper_runtime_profitability_controls_latest.json")
    live_canary_readiness = load_json(health / "live_canary_readiness_contract_latest.json")
    uniform_hardening = load_json(health / "uniform_hardening_contract_latest.json")
    uniform_hardening_enabled = (project_root / "config" / "production_uniform_hardening_v1.json").is_file()
    low_grade_audit = _low_grade_layer_audit(project_root)
    raw_profitability_recovery = _raw_profitability_recovery_context(
        paper_profitability=paper_profitability,
        paper_runtime_profitability=paper_runtime_profitability,
        live_canary_readiness=live_canary_readiness,
    )
    soak_management = _soak_management_context(project_root, health_fast)
    needs = [
        *_need_from_governor(governor),
        *_need_from_memory(memory),
        *_need_from_storage(storage),
        *_need_from_training_runtime(training_runtime),
        *_need_from_low_grade_audit(low_grade_audit),
        *_need_from_raw_profitability_recovery(raw_profitability_recovery),
        *(_need_from_uniform_hardening(uniform_hardening) if uniform_hardening_enabled else []),
        *_need_from_runtime_surfaces(
            health_fast=health_fast,
            process_watchdog=process_watchdog,
            health_gates=health_gates,
            collector_contracts=collector_contracts,
            global_halt=global_halt,
            paper_ramp=paper_ramp,
            plumbing=plumbing,
        ),
    ]
    ready_actions = _ready_actions_from_training_runtime(training_runtime)
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in needs:
        key = (str(item.get("blocker")), str(item.get("exact_file")), str(item.get("exact_shard")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    actionable_needs, managed_controls = _split_managed_soak_controls(deduped, soak_management)
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "needs_action" if actionable_needs else "ready",
        "what_do_you_need": actionable_needs,
        "needs": actionable_needs,
        "managed_controls": managed_controls,
        "next_command": actionable_needs[0]["command"] if actionable_needs else [],
        "ready_actions": ready_actions,
        "next_ready_command": ready_actions[0]["command"] if ready_actions else [],
        "frames_of_reference": {
            "latest_writer_effectiveness": _as_dict(writer.get("drain_effectiveness")),
            "latest_benchmark_limits": _as_dict(benchmark.get("self_tuned_limits")),
            "backlog_green_gate": _as_dict(governor.get("backlog_green_gate")),
            "backlog_trend": _as_dict(governor.get("backlog_trend")),
            "stability_state": _as_dict(governor.get("stability_state")),
            "adaptive_controls": _as_dict(governor.get("adaptive_controls")),
            "runtime_pressure_source": _as_dict(governor.get("runtime_pressure_source")),
            "host_pressure_attribution": _as_dict(runtime.get("host_pressure_attribution")),
            "memory_pressure_intelligence": {
                "classification": _as_dict(memory.get("classification")),
                "trend": _as_dict(memory.get("trend")),
                "reopen_gate": _as_dict(memory.get("reopen_gate")),
                "multitasking_headroom": _as_dict(memory.get("multitasking_headroom")),
                "observer_overhead": _as_dict(memory.get("observer_overhead")),
            },
            "training_runtime_control": {
                "overall_status": str(training_runtime.get("overall_status") or ""),
                "launch_contract": _as_dict(training_runtime.get("training_launch_contract")),
                "host_training_headroom_gate": _as_dict(_as_dict(training_runtime.get("training_launch_contract")).get("host_training_headroom_gate")),
                "bot_needs": _as_dict(training_runtime.get("bot_needs")),
            },
            "health_fast": {
                "overall_status": str(health_fast.get("overall_status") or health_fast.get("status") or ""),
                "process_watchdog": _as_dict(health_fast.get("process_watchdog")),
                "operational_readiness": _as_dict(health_fast.get("operational_readiness")),
            },
            "health_gates": {
                "hard_gates": _as_dict(health_gates.get("hard_gates")),
                "inputs": _as_dict(health_gates.get("inputs")),
            },
            "collector_contracts": {
                "required_failures": _as_list(collector_contracts.get("required_failures")),
                "soft_failures": _as_list(collector_contracts.get("soft_failures")),
            },
            "global_halt": {
                "halt": bool(global_halt.get("halt", False) or global_halt.get("global_halt", False)),
                "halt_latched": bool(global_halt.get("halt_latched", False)),
                "halt_required": bool(global_halt.get("halt_required", False) or global_halt.get("would_rehalt", False)),
                "clear_blockers": _as_list(global_halt.get("clear_blockers")),
                "halt_posture": str(global_halt.get("halt_posture") or ""),
            },
            "paper_400_ramp": {
                "stage": str(paper_ramp.get("stage") or ""),
                "blockers": _as_list(paper_ramp.get("blockers")),
                "global_halt_gate": _as_dict(_as_dict(paper_ramp.get("gates")).get("global_halt")),
            },
            "system_plumbing_control": {
                "overall_status": str(plumbing.get("overall_status") or ""),
                "plumbing_score": plumbing.get("plumbing_score"),
            },
            "low_grade_layer_audit": low_grade_audit,
            "raw_profitability_recovery": raw_profitability_recovery,
            "uniform_hardening_contract": {
                "overall_status": str(uniform_hardening.get("overall_status") or ""),
                "uniform_floor_ready": bool(uniform_hardening.get("uniform_floor_ready", False)),
                "critical_runtime_ready": bool(uniform_hardening.get("critical_runtime_ready", False)),
                "domain_statuses": _as_dict(uniform_hardening.get("domain_statuses")),
                "structural_blockers": _as_list(uniform_hardening.get("structural_blockers")),
                "critical_runtime_blockers": _as_list(uniform_hardening.get("critical_runtime_blockers")),
                "evidence_debt_domains": _as_list(uniform_hardening.get("evidence_debt_domains")),
            },
            "soak_management_context": soak_management,
            "operator_action_packet": _as_dict(governor.get("operator_action_packet")),
            "migration_binder": _as_dict(migration.get("migration_binder")),
            "recent_fix_log": _load_fix_log(fix_log_path),
        },
        "contract": {
            "always_include_exact_blocker": True,
            "always_include_exact_file_or_shard_when_known": True,
            "always_include_expected_impact_risk_and_stop_rule": True,
            "include_ready_actions_when_no_blockers_exist": True,
            "include_remaining_low_grade_layers": True,
            "include_raw_profitability_recovery_need": True,
            "include_uniform_hardening_floor_need": True,
            "split_managed_soak_controls_from_actionable_needs": True,
            "fixes_logged_to": str(fix_log_path),
            "protected_volumes": ["/Volumes/VIDEO"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain exactly what the system needs next and preserve fix frames of reference.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--fix-log", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--log-fix", default="")
    parser.add_argument("--fix-result", default="")
    args = parser.parse_args()
    fix_log_path = Path(args.fix_log)
    if args.log_fix:
        _append_fix_log(
            fix_log_path,
            {
                "timestamp_utc": iso_now(),
                "fix": args.log_fix,
                "result": args.fix_result,
            },
        )
    payload = build_payload(PROJECT_ROOT, fix_log_path=fix_log_path)
    write_payload(Path(args.out), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"system_needs_intelligence status={payload['overall_status']} needs={len(payload['what_do_you_need'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
