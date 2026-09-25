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
    from scripts.ops.long_runtime_common import (
        PROJECT_ROOT,
        iso_now,
        load_json,
        write_payload,
    )
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "system_needs_intelligence_latest.json"
)
DEFAULT_LOG_PATH = PROJECT_ROOT / "governance" / "health" / "system_needs_fix_log.jsonl"
DEFAULT_MARKDOWN_PATH = (
    PROJECT_ROOT / "exports" / "reports" / "operator" / "system_needs_latest.md"
)
DEFAULT_ROLE_CONTRACTS_PATH = PROJECT_ROOT / "config" / "system_role_contracts_v1.json"
DEFAULT_SLEEVE_STRATEGY_CONTRACTS_PATH = (
    PROJECT_ROOT / "config" / "sleeve_strategy_contracts_v1.json"
)
DEFAULT_MASTER_GRANDMASTER_EVIDENCE_PATH = (
    PROJECT_ROOT / "config" / "master_grandmaster_evidence_v2.json"
)
LOW_GRADE_VALUES = {"C", "D", "F"}
LOW_GRADE_AUDIT_EXCLUDED_FILES = {
    "low_grade_finalizer_latest.json",
    "profitability_self_assessment_latest.json",
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
        return [
            "./scripts/ops/opsctl.sh",
            "quant-storage-backlog-accommodation",
            "--apply",
            "--json",
        ]
    if (
        "paper_profitability" in text
        or "profit_harvest" in text
        or "profit_grade" in text
    ):
        return [
            "./scripts/ops/opsctl.sh",
            "paper-profitability-control",
            "--apply",
            "--json",
        ]
    if (
        "system_self_intelligence" in text
        or "codex_handoff" in text
        or "whole_system_intelligence" in text
    ):
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
            "base/raw grade is B or better and the headline/control grade no longer depends on rescue credit.",
        )
    if category in {
        "contained_profit_grade",
        "probationary_profit_grade",
        "profile_profit_grade",
    }:
        return (
            "Repairs or deweights weak paper profiles using hard-negative labels, tighter entries/exits, and fresh paper evidence.",
            "profile profit grade is B or better, or the profile is explicitly quarantined/probationary with no active new-entry path.",
        )
    if category == "self_awareness_grade":
        return (
            "Refreshes stale self-awareness surfaces so the handoff stops reasoning from old artifacts.",
            "system_self_intelligence.awareness_state_vector.grade is B or better with stale/blind-spot count reduced.",
        )
    if category == "backlog_accommodation_snapshot":
        return (
            "Refreshes the stale quant/backlog accommodation snapshot against current storage truth.",
            "backlog accommodation snapshot is current and backlog_letter_grade is B or better.",
        )
    return (
        "Refreshes the owning health surface and keeps the low grade visible for targeted repair.",
        "the same JSON path no longer reports C/D/F.",
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
    snapshot_roots = {
        "steps",
        "refresh_steps",
        "repair_steps",
        "command_results",
        "results",
    }
    if parts and parts[0] in snapshot_roots and "payload" in parts:
        return True
    return bool(
        parts
        and parts[0]
        in {
            "production_excellence",
            "system_signal_bus",
            "root_causes",
            "readiness_evidence",
        }
    )


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
    json_path = str(row.get("exact_json_path") or "").lower()
    if (
        bool(row.get("stale_artifact", False))
        or bool(row.get("embedded_snapshot", False))
        or bool(row.get("historical_snapshot", False))
    ):
        return "historical_or_superseded"
    if bool(row.get("propagated_snapshot", False)):
        return "propagated_runtime_signal"
    if any(
        token in source
        for token in (
            "production_excellence",
            "live_money_readiness",
            "live_transition",
            "promotion",
            "canary",
            "profitability_evidence_firewall",
        )
    ):
        return "live_promotion_evidence"
    if "evidence_grade" in json_path:
        return "evidence_debt"
    if (
        "paper_profitability" in source
        or "paper_runtime_profitability" in source
        or "sleeve_profitability_dashboard" in source
    ):
        return "paper_outcome_evidence"
    return "runtime_operational"


def _canonical_low_grade_key(
    source_file: str, json_path: str, grade: str, category: str
) -> tuple[str, str, str]:
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
        return (
            category,
            "quant_strategy_storage_backlog_accommodation.storage_snapshot.backlog_letter_grade",
            grade,
        )
    return (category, f"{source_file}:{json_path}", grade)


def _low_grade_control_context(health: Path) -> dict[str, Any]:
    paper = load_json(health / "paper_profitability_control_latest.json")
    self_intelligence = load_json(health / "system_self_intelligence_latest.json")
    non_blocking_profiles = {
        str(row.get("profile") or "")
        for row in _as_list(_as_dict(paper).get("remaining_low_grade_layers"))
        if isinstance(row, dict)
        and str(row.get("profile") or "")
        and not bool(row.get("active_blocker", False))
    }
    return {
        "paper_control_ok": bool(paper.get("ok", False)),
        "paper_control_posture_grade": str(
            _as_dict(_as_dict(paper).get("low_grade_control_report_card")).get(
                "control_posture_grade"
            )
            or _as_dict(_as_dict(paper).get("low_grade_layer_summary")).get(
                "control_posture_grade"
            )
            or ""
        ).upper(),
        "paper_active_blocker_count": _safe_int(
            _as_dict(_as_dict(paper).get("low_grade_layer_summary")).get(
                "active_blocker_count"
            ),
            _safe_int(
                _as_dict(_as_dict(paper).get("low_grade_control_report_card")).get(
                    "active_blocker_count"
                ),
                999,
            ),
        ),
        "paper_non_blocking_profiles": non_blocking_profiles,
        "self_awareness_control_posture_grade": str(
            _as_dict(_as_dict(self_intelligence).get("awareness_state_vector")).get(
                "control_posture_grade"
            )
            or _as_dict(_as_dict(self_intelligence).get("awareness_state_vector")).get(
                "control_grade"
            )
            or ""
        ).upper(),
    }


def _profile_from_canonical_path(canonical_path: str) -> str:
    prefix = "profile_profit_grade."
    text = str(canonical_path or "")
    return text[len(prefix) :] if text.startswith(prefix) else ""


def _low_grade_control_state(
    row: dict[str, Any], context: dict[str, Any]
) -> tuple[str, bool]:
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
    if category == "profile_profit_grade" and _profile_from_canonical_path(
        canonical_path
    ) in set(context.get("paper_non_blocking_profiles") or set()):
        return ("contained_by_paper_profitability_control", False)
    source = str(row.get("exact_file") or "").lower()
    paper_control_ready = (
        _safe_int(context.get("paper_active_blocker_count"), 999) == 0
        and str(context.get("paper_control_posture_grade") or "") == "A+"
    )
    if paper_control_ready and (
        "paper_profitability_control" in source
        or "paper_runtime_profitability_controls" in source
    ):
        return ("raw_paper_outcome_under_a_plus_control", False)
    if (
        category == "base_evidence_grade"
        and canonical_path == "profit_harvest_report_card.base_raw_outcome_grade"
        and _safe_int(context.get("paper_active_blocker_count"), 999) == 0
        and str(context.get("paper_control_posture_grade") or "") in {"A+", "A+"}
    ):
        return ("raw_harvest_evidence_under_a_plus_control", False)
    if category == "self_awareness_grade" and str(
        context.get("self_awareness_control_posture_grade") or ""
    ) in {"A+", "A+"}:
        return ("self_awareness_under_a_plus_control", False)
    if str(row.get("scope") or "") == "paper_outcome_evidence" and bool(
        context.get("paper_control_ok", False)
    ):
        return ("raw_paper_outcome_under_operational_control", False)
    if str(row.get("scope") or "") == "evidence_debt":
        return ("elapsed_or_qualification_evidence_debt", False)
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
            if (
                isinstance(value, str)
                and value.strip().upper() in LOW_GRADE_VALUES
                and _is_grade_field(str(key))
            ):
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
        if canonical_alias and (health / canonical_alias).exists():
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
                    "propagated_snapshot": _is_propagated_grade_path(
                        path.name, json_path
                    ),
                    "command": command,
                    "expected_impact": expected_impact,
                    "risk_level": "low",
                    "when_to_stop": stop_when,
                    "source": "low_grade_layer_audit",
                }
            hits.append(
                {
                    "file": str(path.relative_to(project_root)),
                    "json_path": json_path,
                    "grade": grade,
                    "category": category,
                }
            )
    layers = list(canonical.values())
    for row in layers:
        key = (
            str(row.get("category") or ""),
            str(row.get("canonical_json_path") or ""),
            str(row.get("current_grade") or ""),
        )
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
            (
                0
                if str(row.get("category") or "")
                in {
                    "base_evidence_grade",
                    "self_awareness_grade",
                    "backlog_accommodation_snapshot",
                }
                else 1
            ),
            str(row.get("category") or ""),
            str(row.get("exact_file") or ""),
            str(row.get("exact_json_path") or ""),
        )
    )
    by_category: dict[str, int] = {}
    for row in layers:
        category = str(row.get("category") or "low_grade_layer")
        by_category[category] = by_category.get(category, 0) + 1
    active_blocker_count = sum(
        1 for row in layers if bool(row.get("active_blocker", False))
    )
    stale_artifact_count = sum(
        1 for row in layers if bool(row.get("stale_artifact", False))
    )
    embedded_snapshot_count = sum(
        1 for row in layers if bool(row.get("embedded_snapshot", False))
    )
    historical_snapshot_count = sum(
        1 for row in layers if bool(row.get("historical_snapshot", False))
    )
    propagated_snapshot_count = sum(
        1 for row in layers if bool(row.get("propagated_snapshot", False))
    )
    promotion_evidence_layer_count = sum(
        1 for row in layers if str(row.get("scope") or "") == "live_promotion_evidence"
    )
    contained_or_controlled_count = sum(
        1 for row in layers if not bool(row.get("active_blocker", False))
    )
    effective_low_grade_layer_count = sum(
        1
        for row in layers
        if str(row.get("effective_grade") or row.get("current_grade") or "").upper()
        in LOW_GRADE_VALUES
    )
    next_commands: list[list[Any]] = []
    seen_commands: set[tuple[str, ...]] = set()
    for row in [
        row for row in layers if bool(row.get("active_blocker", False))
    ] or layers:
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
        "control_posture_status": (
            "a_plus_control_ready"
            if active_blocker_count == 0
            else "actionable_low_grade_blockers"
        ),
        "finalization_contract": {
            "active": True,
            "mode": "truthful_low_grade_classification_v2",
            "effective_control_posture_grade": _low_grade_audit_control_grade(
                active_blocker_count
            ),
            "raw_grades_preserved": True,
            "rewrites_raw_evidence": False,
            "cosmetic_grade_uplift_allowed": False,
        },
        "by_category": by_category,
        "layers": layers,
        "actionable_layers": [
            row for row in layers if bool(row.get("active_blocker", False))
        ],
        "next_commands": next_commands,
        "reporting_rule": "C/D/F evidence is never relabeled. Current blockers, controlled outcomes, stale artifacts, propagated signals, and superseded embedded snapshots are classified separately.",
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
                "remaining C/D/F grade layers instead of hiding them behind headline grades."
            ),
            "risk_level": "low",
            "when_to_stop": "low_grade_layer_audit.unique_low_grade_layer_count is 0, or every remaining low grade is marked contained/probationary with an explicit repair path.",
            "source": "low_grade_layer_audit",
            "low_grade_layer_count": _safe_int(
                audit.get("unique_low_grade_layer_count"), 0
            ),
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
    runtime_source = (
        paper_runtime_profitability
        if paper_runtime_profitability
        else paper_profitability
    )
    raw_grade = (
        str(
            source.get("raw_profitability_grade")
            or runtime_source.get("raw_profitability_grade")
            or ""
        )
        .strip()
        .upper()
    )
    controlled_grade = (
        str(
            source.get("controlled_profitability_grade")
            or runtime_source.get("controlled_profitability_grade")
            or source.get("profitability_grade")
            or runtime_source.get("profitability_grade")
            or ""
        )
        .strip()
        .upper()
    )
    financial_grade = (
        str(
            source.get("financial_profitability_grade")
            or runtime_source.get("financial_profitability_grade")
            or source.get("financial_display_grade")
            or runtime_source.get("financial_display_grade")
            or ""
        )
        .strip()
        .upper()
    )
    a_plus = _first_nested_dict(
        _nested_dict(paper_runtime_profitability, "a_plus_target_contract"),
        _nested_dict(paper_profitability, "a_plus_target_contract"),
    )
    current = _as_dict(a_plus.get("current"))
    thresholds = _as_dict(a_plus.get("thresholds"))
    raw_improvement = _first_nested_dict(
        _nested_dict(
            paper_runtime_profitability, "raw_profitability_improvement_contract"
        ),
        _nested_dict(paper_profitability, "raw_profitability_improvement_contract"),
    )
    raw_a_recovery = _first_nested_dict(
        _nested_dict(
            paper_runtime_profitability, "raw_profitability_a_recovery_contract"
        ),
        _nested_dict(paper_profitability, "raw_profitability_a_recovery_contract"),
    )
    raw_six = _first_nested_dict(
        _nested_dict(
            paper_runtime_profitability, "raw_profitability_six_point_recovery_contract"
        ),
        _nested_dict(
            paper_profitability, "raw_profitability_six_point_recovery_contract"
        ),
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
    ready_requirement_count = sum(
        1
        for row in requirements
        if isinstance(row, dict) and bool(row.get("ready", False))
    )
    live_blockers = [
        str(item or "").strip()
        for item in _as_list(live_canary_readiness.get("blockers"))
        if str(item or "").strip()
    ]
    raw_live_blockers = [item for item in live_blockers if "raw_profitability" in item]
    net_pnl = _safe_float(
        current.get("net_pnl"), _safe_float(burn_down.get("current_net_pnl"), 0.0)
    )
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
        "change_vs_previous_day": _safe_float(
            current.get("change_vs_previous_day"), 0.0
        ),
        "executions": _safe_int(current.get("executions"), 0),
        "weak_profile_count": _safe_int(current.get("weak_profile_count"), 0),
        "strategy_control_count": _safe_int(current.get("strategy_control_count"), 0),
        "unprotected_weak_profile_count": _safe_int(
            current.get("unprotected_weak_profile_count"), 0
        ),
        "unprotected_strategy_control_count": _safe_int(
            current.get("unprotected_strategy_control_count"), 0
        ),
        "min_net_pnl": _safe_float(thresholds.get("min_net_pnl"), 0.0),
        "combined_a_plus_ready": (
            bool(a_plus.get("combined_a_plus_ready", False)) if a_plus else False
        ),
        "daily_net_improvement_target": max(
            _safe_float(burn_down.get("required_average_daily_net_improvement"), 0.0),
            _safe_float(
                _as_dict(raw_improvement.get("runtime_enforcement")).get(
                    "raw_d_daily_net_improvement_target"
                ),
                0.0,
            ),
        ),
        "top_loss_causes": top_loss_causes,
        "requirement_count": len(requirements),
        "ready_requirement_count": ready_requirement_count,
        "all_requirements_ready": bool(requirements)
        and ready_requirement_count == len(requirements),
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


def _need_from_raw_profitability_recovery(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
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
            "exact_file": str(
                context.get("source_file")
                or "governance/health/paper_runtime_profitability_controls_latest.json"
            ),
            "exact_shard": "raw_profitability_improvement_contract",
            "command": [
                "./scripts/ops/opsctl.sh",
                "paper-profitability-control",
                "--apply",
                "--json",
            ],
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


def _profitability_self_assessment_context(payload: dict[str, Any]) -> dict[str, Any]:
    binding = _as_dict(payload.get("candidate_binding"))
    candidate = _as_dict(payload.get("candidate"))
    grades = _as_dict(payload.get("grades"))
    scorecard = _as_dict(payload.get("scorecard"))
    implementation_scorecard = _as_dict(scorecard.get("implementation"))
    economic_scorecard = _as_dict(scorecard.get("economic_evidence"))
    measurement = _as_dict(payload.get("measurement"))
    evidence_gaps = _as_dict(payload.get("profitability_evidence_gaps"))
    developmental = _as_dict(payload.get("developmental_soak_learning"))
    needs = [row for row in _as_list(payload.get("needs")) if isinstance(row, dict)]
    return {
        "present": bool(payload),
        "overall_status": str(payload.get("overall_status") or "missing"),
        "assessment_status": str(
            payload.get("assessment_status")
            or payload.get("overall_status")
            or "missing"
        ),
        "system_statement": str(payload.get("system_statement") or ""),
        "candidate_id": str(
            binding.get("candidate_id")
            or candidate.get("candidate_id")
            or measurement.get("candidate_id")
            or ""
        ),
        "candidate_identity_consistent": bool(
            binding.get(
                "identity_consistent", candidate.get("identity_consistent", False)
            )
        ),
        "candidate_identity_complete": bool(
            binding.get("identity_complete", candidate.get("identity_complete", False))
        ),
        "implementation_grade": str(
            grades.get("implementation_grade")
            or implementation_scorecard.get("grade")
            or ""
        ),
        "implementation_score": _safe_float(
            grades.get("implementation_score"),
            _safe_float(implementation_scorecard.get("score"), 0.0),
        ),
        "economic_evidence_grade": str(
            grades.get("economic_evidence_grade")
            or economic_scorecard.get("grade")
            or ""
        ),
        "economic_evidence_score": _safe_float(
            grades.get("economic_evidence_score"),
            _safe_float(economic_scorecard.get("score"), 0.0),
        ),
        "economic_evidence_ready": bool(
            grades.get(
                "economic_evidence_ready", economic_scorecard.get("ready", False)
            )
        ),
        "candidate_post_cost_sample_count": _safe_int(
            measurement.get("candidate_post_cost_sample_count"), 0
        ),
        "candidate_post_cost_minimum_samples": _safe_int(
            measurement.get("candidate_post_cost_minimum_samples"),
            _safe_int(evidence_gaps.get("candidate_post_cost_minimum_samples"), 30),
        ),
        "candidate_observed_days": _safe_int(
            measurement.get("candidate_observed_days"),
            _safe_int(evidence_gaps.get("candidate_observed_days"), 0),
        ),
        "candidate_minimum_observed_days": _safe_int(
            measurement.get("candidate_minimum_observed_days"),
            _safe_int(evidence_gaps.get("candidate_minimum_observed_days"), 3),
        ),
        "candidate_independent_fill_records": _safe_int(
            measurement.get("candidate_independent_fill_records"),
            _safe_int(evidence_gaps.get("candidate_independent_fill_records"), 0),
        ),
        "candidate_independent_fill_minimum_records": _safe_int(
            measurement.get("candidate_independent_fill_minimum_records"),
            _safe_int(
                evidence_gaps.get("candidate_independent_fill_minimum_records"), 30
            ),
        ),
        "positive_post_cost_lower_confidence_bound_95": bool(
            measurement.get(
                "positive_post_cost_lower_confidence_bound_95",
                evidence_gaps.get("positive_post_cost_lcb", False),
            )
        ),
        "historical_active_book_net_pnl": _safe_float(
            measurement.get("historical_active_book_net_pnl"), 0.0
        ),
        "historical_active_book_candidate_grade_eligible": bool(
            measurement.get("historical_active_book_candidate_grade_eligible", False)
        ),
        "developmental_learning_status": str(developmental.get("status") or "missing"),
        "accepted_generation_count": _safe_int(
            developmental.get("accepted_generation_count"), 0
        ),
        "attributable_generation_count": _safe_int(
            developmental.get("attributable_generation_count"), 0
        ),
        "mature_developmental_generation_count": _safe_int(
            developmental.get("mature_developmental_generation_count"), 0
        ),
        "bounded_paper_action_plan": [
            row
            for row in _as_list(developmental.get("bounded_paper_action_plan"))
            if isinstance(row, dict)
        ],
        "historical_generations_grade_current_candidate": False,
        "clean_720_hour_live_promotion_gate_unchanged": True,
        "needs": needs,
        "next_safe_action": _as_dict(payload.get("next_safe_action")),
        "assessment_sha256": str(payload.get("assessment_sha256") or ""),
    }


def _needs_from_profitability_self_assessment(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    if not context.get("present", False):
        return []
    normalized: list[dict[str, Any]] = []
    for raw in _as_list(context.get("needs")):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row.setdefault("source", "profitability_self_assessment")
        row.setdefault("risk_level", "low")
        row.setdefault(
            "command",
            ["./scripts/ops/opsctl.sh", "profitability-self-assessment", "--json"],
        )
        row.setdefault(
            "expected_impact",
            "Advances current-candidate profitability evidence without relabeling historical results.",
        )
        row.setdefault(
            "when_to_stop", "the candidate-bound assessment marks this need complete"
        )
        row["canonical_candidate_bound_need"] = True
        normalized.append(row)
    return normalized


def _market_pattern_feedback_context(payload: dict[str, Any]) -> dict[str, Any]:
    patterns = [
        row for row in _as_list(payload.get("patterns")) if isinstance(row, dict)
    ]
    dominant = [
        row
        for row in _as_list(payload.get("dominant_patterns"))
        if isinstance(row, dict)
    ]
    sleeves = [
        row for row in _as_list(payload.get("sleeve_feedback")) if isinstance(row, dict)
    ]
    prioritized = [
        str(row.get("sleeve") or "")
        for row in sleeves
        if str(row.get("paper_sampling_posture") or "")
        == "prioritize_bounded_paper_sampling"
    ]
    downshift = [
        str(row.get("sleeve") or "")
        for row in sleeves
        if str(row.get("paper_sampling_posture") or "") == "downshift_or_context_first"
    ]
    context_only = [
        str(row.get("sleeve") or "")
        for row in sleeves
        if str(row.get("paper_sampling_posture") or "") == "context_only"
    ]
    contract = _as_dict(payload.get("platform_feedback_contract"))
    evidence_gaps = _as_dict(payload.get("profitability_evidence_gaps"))
    dominant_rows = dominant or patterns[:5]
    pattern_ids = [str(row.get("pattern_id") or "") for row in patterns]
    return {
        "present": bool(payload),
        "overall_status": str(payload.get("overall_status") or "missing"),
        "pattern_count": _safe_int(payload.get("pattern_count"), len(patterns)),
        "dominant_pattern_ids": _unique(
            [str(row.get("pattern_id") or "") for row in dominant_rows]
        ),
        "dominant_patterns": [
            {
                "pattern_id": str(row.get("pattern_id") or ""),
                "label": str(row.get("label") or ""),
                "strength": _safe_float(row.get("strength"), 0.0),
                "confidence": _safe_float(row.get("confidence"), 0.0),
                "direction": str(row.get("direction") or ""),
            }
            for row in dominant_rows[:8]
        ],
        "pattern_ids": _unique(pattern_ids),
        "has_symbol_specific_evidence_gap": "symbol_specific_evidence_gap"
        in set(pattern_ids),
        "observable_dimension_count": _safe_int(
            payload.get("observable_dimension_count"), 0
        ),
        "source_context_usable_count": _safe_int(
            _as_dict(payload.get("source_snapshot")).get("context_usable_source_count"),
            0,
        ),
        "source_count": _safe_int(
            _as_dict(payload.get("source_snapshot")).get("source_count"), 0
        ),
        "sleeve_feedback_count": len(sleeves),
        "prioritized_sleeves": _unique(prioritized),
        "downshift_sleeves": _unique(downshift),
        "context_only_sleeves": _unique(context_only),
        "top_sleeve_feedback": [
            {
                "sleeve": str(row.get("sleeve") or ""),
                "paper_sampling_posture": str(row.get("paper_sampling_posture") or ""),
                "boost_score": _safe_float(row.get("boost_score"), 0.0),
                "caution_score": _safe_float(row.get("caution_score"), 0.0),
                "context_score": _safe_float(row.get("context_score"), 0.0),
                "pattern_ids": _list_of_strings(row.get("pattern_ids")),
                "collection_focus": _list_of_strings(row.get("collection_focus")),
            }
            for row in sleeves[:10]
        ],
        "profitability_evidence_gaps": evidence_gaps,
        "recommended_actions": _list_of_strings(payload.get("recommended_actions")),
        "paper_only": bool(payload.get("paper_only", True)),
        "live_execution_allowed": bool(payload.get("live_execution_allowed", False)),
        "profitability_claim_allowed": bool(
            payload.get("profitability_claim_allowed", False)
        ),
        "can_route_paper_collection_priority": bool(
            contract.get("can_route_paper_collection_priority", False)
        ),
        "can_change_live_execution": bool(
            contract.get("can_change_live_execution", False)
        ),
        "can_claim_profitability": bool(contract.get("can_claim_profitability", False)),
        "source_file": "governance/health/market_pattern_feedback_latest.json",
    }


def _paper_evidence_collection_context(payload: dict[str, Any]) -> dict[str, Any]:
    defaults = _as_dict(payload.get("defaults"))
    profiles = _as_dict(payload.get("profiles"))
    forbidden_enabled = [
        key
        for key in (
            "force_trade_allowed",
            "loss_recovery_size_increase_allowed",
            "profitability_claim_allowed",
        )
        if bool(defaults.get(key, False))
    ]
    safe = bool(
        payload
        and bool(payload.get("enabled", False))
        and bool(payload.get("paper_only", False))
        and not bool(payload.get("live_execution_allowed", False))
        and not forbidden_enabled
    )
    return {
        "present": bool(payload),
        "policy_id": str(payload.get("policy_id") or ""),
        "enabled": bool(payload.get("enabled", False)),
        "paper_only": bool(payload.get("paper_only", False)),
        "live_execution_allowed": bool(payload.get("live_execution_allowed", True)),
        "safe_for_evidence_collection": safe,
        "profile_count": len(profiles),
        "allowed_guard_reasons": _list_of_strings(
            defaults.get("allowed_guard_reasons")
        ),
        "allowed_entry_policy_blockers": _list_of_strings(
            defaults.get("allowed_entry_policy_blockers")
        ),
        "allowed_clean_gate_failures": _list_of_strings(
            defaults.get("allowed_clean_gate_failures")
        ),
        "minimum_model_score_edge_over_threshold": _safe_float(
            defaults.get("minimum_model_score_edge_over_threshold"), 0.0
        ),
        "minimum_known_core_channels": _safe_int(
            defaults.get("minimum_known_core_channels"), 0
        ),
        "forbidden_enabled": forbidden_enabled,
        "non_relaxed_controls": _list_of_strings(payload.get("non_relaxed_controls")),
        "source_file": "config/paper_evidence_collection_controls_v1.json",
    }


def _brain_boundary_category_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _as_list(value):
        item = _as_dict(row)
        category_id = str(item.get("category_id") or "").strip()
        if not category_id:
            continue
        rows.append(
            {
                "category_id": category_id,
                "label": str(item.get("label") or "").strip(),
                "purpose": str(item.get("purpose") or "").strip(),
                "scope": _list_of_strings(item.get("scope")),
                "current_posture": str(item.get("current_posture") or "").strip(),
                "allowed_actions": _list_of_strings(item.get("allowed_actions")),
                "frozen_or_review_actions": _list_of_strings(
                    item.get("frozen_or_review_actions")
                ),
                "never_actions": _list_of_strings(item.get("never_actions")),
            }
        )
    return rows


def _brain_boundary_category_ids(value: Any) -> list[str]:
    return [
        str(_as_dict(row).get("category_id") or "").strip()
        for row in _as_list(value)
        if str(_as_dict(row).get("category_id") or "").strip()
    ]


def _route_category(route: str) -> tuple[str, str]:
    parts = str(route or "").strip().split(".")
    if len(parts) < 2:
        return "", ""
    return parts[0], parts[1]


def _operator_brain_boundary_hardening(
    *,
    boundary: dict[str, Any],
    trading: dict[str, Any],
    ops: dict[str, Any],
    trading_categories: list[dict[str, Any]],
    ops_categories: list[dict[str, Any]],
    routing: dict[str, Any],
) -> dict[str, Any]:
    invariants = _as_dict(boundary.get("hardening_invariants"))
    required_fields = _list_of_strings(invariants.get("required_category_fields")) or [
        "category_id",
        "purpose",
        "scope",
        "current_posture",
    ]
    required_trading = set(
        _list_of_strings(invariants.get("required_trading_categories"))
    )
    required_ops = set(_list_of_strings(invariants.get("required_ops_categories")))
    trading_ids = _brain_boundary_category_ids(trading_categories)
    ops_ids = _brain_boundary_category_ids(ops_categories)
    trading_set = set(trading_ids)
    ops_set = set(ops_ids)
    checks: dict[str, dict[str, Any]] = {}

    def record(check_id: str, ok: bool, detail: Any) -> None:
        checks[check_id] = {"ok": bool(ok), "detail": detail}

    def missing_category_fields(rows: list[dict[str, Any]]) -> list[str]:
        missing: list[str] = []
        for row in rows:
            category_id = str(row.get("category_id") or "")
            for field in required_fields:
                value = row.get(field)
                if isinstance(value, list):
                    empty = not _list_of_strings(value)
                else:
                    empty = not str(value or "").strip()
                if empty:
                    missing.append(f"{category_id}.{field}")
        return missing

    trading_missing_fields = missing_category_fields(trading_categories)
    ops_missing_fields = missing_category_fields(ops_categories)
    record(
        "trading_categories_have_required_fields",
        not trading_missing_fields,
        trading_missing_fields,
    )
    record(
        "ops_categories_have_required_fields",
        not ops_missing_fields,
        ops_missing_fields,
    )
    record(
        "trading_category_ids_unique",
        len(trading_ids) == len(trading_set),
        trading_ids,
    )
    record("ops_category_ids_unique", len(ops_ids) == len(ops_set), ops_ids)
    record(
        "required_trading_categories_present",
        required_trading.issubset(trading_set),
        sorted(required_trading - trading_set),
    )
    record(
        "required_ops_categories_present",
        required_ops.issubset(ops_set),
        sorted(required_ops - ops_set),
    )

    route_fields = ("safe_now", "freeze_now", "read_only_now", "review_required")
    unresolved: list[str] = []
    for field in route_fields:
        for route in _list_of_strings(routing.get(field)):
            brain, category = _route_category(route)
            if brain == "trading_brain" and category in trading_set:
                continue
            if brain == "ops_brain" and category in ops_set:
                continue
            unresolved.append(f"{field}:{route}")
    record("routing_references_resolve", not unresolved, unresolved)

    safe_now = _list_of_strings(routing.get("safe_now"))
    freeze_now = _list_of_strings(routing.get("freeze_now"))
    read_only_now = _list_of_strings(routing.get("read_only_now"))
    review_required = _list_of_strings(routing.get("review_required"))
    record(
        "safe_now_routes_are_ops_only",
        all(_route_category(route)[0] == "ops_brain" for route in safe_now),
        safe_now,
    )
    record(
        "freeze_now_routes_are_trading_only",
        all(_route_category(route)[0] == "trading_brain" for route in freeze_now),
        freeze_now,
    )
    record(
        "read_only_routes_are_market_decision_only",
        all(
            _route_category(route) == ("trading_brain", "market_decision")
            for route in read_only_now
        ),
        read_only_now,
    )
    candidate = _as_dict(
        next(
            (
                row
                for row in trading_categories
                if row.get("category_id") == "candidate_evidence"
            ),
            {},
        )
    )
    candidate_posture = str(candidate.get("current_posture") or "")
    record(
        "candidate_evidence_remains_frozen_or_collecting",
        "frozen" in candidate_posture
        or "collecting" in candidate_posture
        or "do_not_reset" in candidate_posture,
        candidate_posture,
    )
    review_text = ",".join(review_required)
    trading_review = _list_of_strings(trading.get("requires_explicit_operator_review"))
    record(
        "live_authority_changes_remain_review_required",
        "authority" in review_text
        and "live_execution_authority_change" in trading_review,
        {"review_required": review_required, "trading_review": trading_review},
    )
    unsafe_tokens = {
        "change_trade_logic",
        "restart_trading_with_new_parameters",
        "grant_live_authority",
        "submit_live_order",
        "submit_order",
        "force_trade",
        "increase_size",
    }
    unsafe_ops_actions = [
        f"{row.get('category_id')}:{action}"
        for row in ops_categories
        for action in _list_of_strings(row.get("allowed_actions"))
        if action in unsafe_tokens
    ]
    record(
        "ops_allowed_actions_do_not_mutate_trade_logic",
        not unsafe_ops_actions,
        unsafe_ops_actions,
    )
    failed = [check_id for check_id, row in checks.items() if not row.get("ok")]
    return {
        "overall_status": "needs_action" if failed else "ready",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "checks": checks,
        "invariants": invariants,
    }


def _operator_brain_boundary_context(payload: dict[str, Any]) -> dict[str, Any]:
    boundary = _as_dict(payload.get("operator_brain_boundary"))
    trading = _as_dict(boundary.get("trading_brain"))
    ops = _as_dict(boundary.get("ops_brain"))
    rules = _as_dict(boundary.get("boundary_rules"))
    routing = _as_dict(boundary.get("routing_matrix"))
    handoff = _as_dict(boundary.get("handoff_language"))
    trading_categories = _brain_boundary_category_rows(trading.get("categories"))
    ops_categories = _brain_boundary_category_rows(ops.get("categories"))
    hardening = _operator_brain_boundary_hardening(
        boundary=boundary,
        trading=trading,
        ops=ops,
        trading_categories=trading_categories,
        ops_categories=ops_categories,
        routing=routing,
    )
    return {
        "present": bool(boundary),
        "boundary_id": str(boundary.get("boundary_id") or ""),
        "purpose": str(boundary.get("purpose") or ""),
        "summary": str(handoff.get("summary") or ""),
        "trading_brain": {
            "display_name": str(trading.get("display_name") or "Trading Brain"),
            "definition": str(trading.get("definition") or ""),
            "scope": _list_of_strings(trading.get("scope")),
            "characteristics": _as_dict(trading.get("characteristics")),
            "current_posture": str(trading.get("current_posture") or ""),
            "allowed_during_collection": _list_of_strings(
                trading.get("allowed_during_collection")
            ),
            "categories": trading_categories,
            "requires_explicit_operator_review": _list_of_strings(
                trading.get("requires_explicit_operator_review")
            ),
        },
        "ops_brain": {
            "display_name": str(ops.get("display_name") or "Ops Brain"),
            "definition": str(ops.get("definition") or ""),
            "scope": _list_of_strings(ops.get("scope")),
            "characteristics": _as_dict(ops.get("characteristics")),
            "current_posture": str(ops.get("current_posture") or ""),
            "allowed_during_collection": _list_of_strings(
                ops.get("allowed_during_collection")
            ),
            "categories": ops_categories,
        },
        "boundary_rules": {
            "trading_brain_changes_may_reset_or_invalidate_candidate_evidence": bool(
                rules.get(
                    "trading_brain_changes_may_reset_or_invalidate_candidate_evidence",
                    False,
                )
            ),
            "ops_brain_never_changes_trade_logic": bool(
                rules.get("ops_brain_never_changes_trade_logic", False)
            ),
            "ops_brain_never_grants_live_authority": bool(
                rules.get("ops_brain_never_grants_live_authority", False)
            ),
            "paper_collection_controls_may_collect_more_evidence_without_live_authority": bool(
                rules.get(
                    "paper_collection_controls_may_collect_more_evidence_without_live_authority",
                    False,
                )
            ),
            "live_money_authority_remains_separate": bool(
                rules.get("live_money_authority_remains_separate", False)
            ),
            "operator_question_default_route": str(
                rules.get("operator_question_default_route") or ""
            ),
        },
        "routing_matrix": {
            "classification_rule": str(routing.get("classification_rule") or ""),
            "safe_now": _list_of_strings(routing.get("safe_now")),
            "freeze_now": _list_of_strings(routing.get("freeze_now")),
            "read_only_now": _list_of_strings(routing.get("read_only_now")),
            "review_required": _list_of_strings(routing.get("review_required")),
        },
        "hardening": hardening,
        "safe_work_now": str(handoff.get("safe_work_now") or ""),
        "frozen_work_now": str(handoff.get("frozen_work_now") or ""),
        "trading_freeze_reason": str(handoff.get("trading_freeze_reason") or ""),
        "source_file": "config/system_role_contracts_v1.json",
    }


def _field_present(value: Any) -> bool:
    if isinstance(value, list):
        return bool(_list_of_strings(value))
    if isinstance(value, dict):
        return bool(value)
    return bool(str(value or "").strip())


def _sleeve_characteristics_hardening(
    *,
    config: dict[str, Any],
    hardening_invariants: dict[str, Any],
    rows: list[dict[str, Any]],
    sleeves: dict[str, Any],
    objective_class_ids: list[str],
    objective_characteristics: dict[str, Any],
    brain_boundary: dict[str, Any],
) -> dict[str, Any]:
    required_objective_fields = _list_of_strings(
        hardening_invariants.get("required_objective_characteristic_fields")
    ) or [
        "primary_character",
        "market_question",
        "naturally_helps_when",
        "naturally_hurts_when",
        "sensitive_to",
        "profitability_evidence_type",
        "trading_brain_categories",
        "ops_brain_dependencies",
    ]
    required_sleeve_fields = _list_of_strings(
        hardening_invariants.get("required_sleeve_fields")
    ) or [
        "objective_class",
        "economic_thesis",
        "universe",
        "decision_horizon",
        "holding_horizon",
        "benchmark",
        "risk_budget",
    ]
    authority_false_fields = _list_of_strings(
        hardening_invariants.get("authority_false_fields")
    ) or [
        "can_change_trade_logic",
        "can_change_sizing",
        "can_submit_order",
        "can_claim_profitability",
        "can_promote_candidate",
    ]
    authority = _as_dict(config.get("authority"))
    min_habitats = _safe_int(
        hardening_invariants.get("minimum_natural_habitat_count"), 1
    )
    min_hostile = _safe_int(
        hardening_invariants.get("minimum_hostile_condition_count"), 1
    )
    min_sensitive = _safe_int(hardening_invariants.get("minimum_sensitivity_count"), 1)
    trading_categories = set(
        _brain_boundary_category_ids(
            _as_dict(_as_dict(brain_boundary).get("trading_brain")).get("categories")
        )
    )
    ops_categories = set(
        _brain_boundary_category_ids(
            _as_dict(_as_dict(brain_boundary).get("ops_brain")).get("categories")
        )
    )
    checks: dict[str, dict[str, Any]] = {}

    def record(check_id: str, ok: bool, detail: Any) -> None:
        checks[check_id] = {"ok": bool(ok), "detail": detail}

    false_authority_violations = [
        field for field in authority_false_fields if bool(authority.get(field, False))
    ]
    record(
        "authority_is_metadata_only",
        bool(authority.get("metadata_only", False)) and not false_authority_violations,
        {
            "metadata_only": bool(authority.get("metadata_only", False)),
            "false_field_violations": false_authority_violations,
        },
    )
    missing_objective_characteristics = [
        objective_id
        for objective_id in objective_class_ids
        if not _as_dict(objective_characteristics.get(objective_id))
    ]
    record(
        "every_objective_class_has_characteristics",
        not missing_objective_characteristics,
        missing_objective_characteristics,
    )
    missing_objective_fields: list[str] = []
    insufficient_habitats: list[str] = []
    insufficient_hostile: list[str] = []
    insufficient_sensitive: list[str] = []
    unresolved_category_refs: list[str] = []
    for objective_id in objective_class_ids:
        characteristic = _as_dict(objective_characteristics.get(objective_id))
        if not characteristic:
            continue
        for field in required_objective_fields:
            if objective_id == "control_only" and field == "trading_brain_categories":
                continue
            if not _field_present(characteristic.get(field)):
                missing_objective_fields.append(f"{objective_id}.{field}")
        if (
            len(_list_of_strings(characteristic.get("naturally_helps_when")))
            < min_habitats
        ):
            insufficient_habitats.append(objective_id)
        if (
            len(_list_of_strings(characteristic.get("naturally_hurts_when")))
            < min_hostile
        ):
            insufficient_hostile.append(objective_id)
        if len(_list_of_strings(characteristic.get("sensitive_to"))) < min_sensitive:
            insufficient_sensitive.append(objective_id)
        for category_id in _list_of_strings(
            characteristic.get("trading_brain_categories")
        ):
            if trading_categories and category_id not in trading_categories:
                unresolved_category_refs.append(f"{objective_id}.trading:{category_id}")
        for category_id in _list_of_strings(
            characteristic.get("ops_brain_dependencies")
        ):
            if ops_categories and category_id not in ops_categories:
                unresolved_category_refs.append(f"{objective_id}.ops:{category_id}")
    record(
        "objective_characteristics_have_required_fields",
        not missing_objective_fields,
        missing_objective_fields,
    )
    record(
        "objective_characteristics_have_enough_habitats",
        not insufficient_habitats,
        insufficient_habitats,
    )
    record(
        "objective_characteristics_have_enough_hostile_conditions",
        not insufficient_hostile,
        insufficient_hostile,
    )
    record(
        "objective_characteristics_have_enough_sensitivities",
        not insufficient_sensitive,
        insufficient_sensitive,
    )
    record(
        "brain_category_references_resolve",
        not unresolved_category_refs,
        unresolved_category_refs,
    )
    missing_sleeve_fields: list[str] = []
    for sleeve_id, raw_sleeve in sorted(sleeves.items()):
        sleeve = _as_dict(raw_sleeve)
        for field in required_sleeve_fields:
            if not _field_present(sleeve.get(field)):
                missing_sleeve_fields.append(f"{sleeve_id}.{field}")
    record(
        "sleeves_have_required_fields", not missing_sleeve_fields, missing_sleeve_fields
    )
    missing_measurement_focus = [
        str(row.get("sleeve_id") or "")
        for row in rows
        if not _list_of_strings(row.get("measurement_focus"))
    ]
    record(
        "every_sleeve_has_measurement_focus",
        not missing_measurement_focus,
        missing_measurement_focus,
    )
    control_only = _as_dict(objective_characteristics.get("control_only"))
    control_only_trading = _list_of_strings(
        control_only.get("trading_brain_categories")
    )
    record(
        "control_only_has_no_trading_brain_categories",
        not control_only_trading,
        control_only_trading,
    )
    failed = [check_id for check_id, row in checks.items() if not row.get("ok")]
    return {
        "overall_status": "needs_action" if failed else "ready",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "checks": checks,
        "invariants": hardening_invariants,
    }


def _sleeve_characteristics_context(
    payload: dict[str, Any],
    market_patterns: dict[str, Any],
    brain_boundary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _as_dict(payload.get("sleeve_characteristics"))
    sleeves = _as_dict(payload.get("sleeves"))
    measurements = _as_dict(payload.get("sleeve_measurement_parameters"))
    defaults = _as_dict(payload.get("measurement_parameter_defaults"))
    objective_characteristics = _as_dict(config.get("objective_class_characteristics"))
    routing_defaults = _as_dict(config.get("routing_defaults"))
    hardening_invariants = _as_dict(config.get("hardening_invariants"))
    prioritized = set(_list_of_strings(market_patterns.get("prioritized_sleeves")))
    downshift = set(_list_of_strings(market_patterns.get("downshift_sleeves")))
    context_only = set(_list_of_strings(market_patterns.get("context_only_sleeves")))
    default_trading_categories = _list_of_strings(
        routing_defaults.get("trading_brain_categories")
    )
    default_ops_dependencies = _list_of_strings(
        routing_defaults.get("ops_brain_dependencies")
    )
    rows: list[dict[str, Any]] = []
    objective_counts: dict[str, int] = {}
    for sleeve_id in sorted(str(key) for key in sleeves):
        sleeve = _as_dict(sleeves.get(sleeve_id))
        objective_class = str(sleeve.get("objective_class") or "").strip()
        if objective_class:
            objective_counts[objective_class] = (
                objective_counts.get(objective_class, 0) + 1
            )
        objective = _as_dict(objective_characteristics.get(objective_class))
        measurement = _as_dict(measurements.get(sleeve_id))
        route = "neutral"
        if sleeve_id in prioritized:
            route = "prioritized_by_market_pattern"
        elif sleeve_id in downshift:
            route = "downshift_by_market_pattern"
        elif sleeve_id in context_only:
            route = "context_only_by_market_pattern"
        rows.append(
            {
                "sleeve_id": sleeve_id,
                "objective_class": objective_class,
                "primary_character": str(
                    objective.get("primary_character") or "unclassified"
                ),
                "market_question": str(objective.get("market_question") or ""),
                "universe": str(sleeve.get("universe") or ""),
                "decision_horizon": str(sleeve.get("decision_horizon") or ""),
                "holding_horizon": str(sleeve.get("holding_horizon") or ""),
                "benchmark": str(sleeve.get("benchmark") or ""),
                "risk_budget": str(sleeve.get("risk_budget") or ""),
                "natural_habitats": _list_of_strings(
                    objective.get("naturally_helps_when")
                ),
                "hostile_conditions": _list_of_strings(
                    objective.get("naturally_hurts_when")
                ),
                "sensitive_to": _list_of_strings(objective.get("sensitive_to")),
                "profitability_evidence_type": str(
                    objective.get("profitability_evidence_type") or ""
                ),
                "measurement_focus": _list_of_strings(
                    measurement.get("measurement_focus")
                    or defaults.get("measurement_focus")
                )[:8],
                "trading_brain_categories": _list_of_strings(
                    objective.get("trading_brain_categories")
                )
                or default_trading_categories,
                "ops_brain_dependencies": _list_of_strings(
                    objective.get("ops_brain_dependencies")
                )
                or default_ops_dependencies,
                "current_market_route": route,
            }
        )
    objective_class_ids = sorted(
        set(str(key) for key in _as_dict(payload.get("objective_classes")))
        | set(objective_counts)
    )
    hardening = _sleeve_characteristics_hardening(
        config=config,
        hardening_invariants=hardening_invariants,
        rows=rows,
        sleeves=sleeves,
        objective_class_ids=objective_class_ids,
        objective_characteristics=objective_characteristics,
        brain_boundary=_as_dict(brain_boundary),
    )
    return {
        "present": bool(payload),
        "policy_id": str(payload.get("policy_id") or ""),
        "purpose": str(config.get("purpose") or ""),
        "authority": _as_dict(config.get("authority")),
        "classification_rule": str(routing_defaults.get("classification_rule") or ""),
        "hardening": hardening,
        "sleeve_count": len(rows),
        "characterized_sleeve_count": len(
            [row for row in rows if row.get("primary_character") != "unclassified"]
        ),
        "objective_class_count": len(objective_counts),
        "objective_class_counts": dict(sorted(objective_counts.items())),
        "objective_characteristics": objective_characteristics,
        "prioritized_sleeves": [
            row
            for row in rows
            if row.get("current_market_route") == "prioritized_by_market_pattern"
        ],
        "downshift_sleeves": [
            row
            for row in rows
            if row.get("current_market_route") == "downshift_by_market_pattern"
        ],
        "context_only_sleeves": [
            row
            for row in rows
            if row.get("current_market_route") == "context_only_by_market_pattern"
        ],
        "sleeves": rows,
        "source_file": "config/sleeve_strategy_contracts_v1.json",
    }


def _strategy_group_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _as_list(value):
        item = _as_dict(row)
        group_id = str(item.get("group_id") or "").strip()
        if not group_id:
            continue
        rows.append(
            {
                "group_id": group_id,
                "label": str(item.get("label") or "").strip(),
                "objective_classes": _list_of_strings(item.get("objective_classes")),
                "primary_sleeves": _list_of_strings(item.get("primary_sleeves")),
                "strategy_archetypes": _list_of_strings(
                    item.get("strategy_archetypes")
                ),
                "expected_edge_source": str(
                    item.get("expected_edge_source") or ""
                ).strip(),
                "measurement_focus": _list_of_strings(item.get("measurement_focus")),
                "failure_modes": _list_of_strings(item.get("failure_modes")),
                "master_bot_needs": _list_of_strings(item.get("master_bot_needs")),
                "grandmaster_bot_needs": _list_of_strings(
                    item.get("grandmaster_bot_needs")
                ),
            }
        )
    return rows


def _strategy_organization_hardening(
    *,
    organization: dict[str, Any],
    groups: list[dict[str, Any]],
    sleeve_objectives: dict[str, str],
    objective_class_ids: set[str],
    strategy_additions: dict[str, Any],
) -> dict[str, Any]:
    invariants = _as_dict(organization.get("hardening_invariants"))
    authority = _as_dict(organization.get("authority"))
    authority_false_fields = _list_of_strings(
        invariants.get("authority_false_fields")
    ) or [
        "can_change_trade_logic",
        "can_change_sizing",
        "can_submit_order",
        "can_claim_profitability",
        "can_promote_candidate",
        "can_allocate_capital",
    ]
    required_group_fields = _list_of_strings(
        invariants.get("required_group_fields")
    ) or [
        "group_id",
        "label",
        "objective_classes",
        "primary_sleeves",
        "strategy_archetypes",
        "expected_edge_source",
        "measurement_focus",
        "failure_modes",
        "master_bot_needs",
        "grandmaster_bot_needs",
    ]
    checks: dict[str, dict[str, Any]] = {}

    def record(check_id: str, ok: bool, detail: Any) -> None:
        checks[check_id] = {"ok": bool(ok), "detail": detail}

    false_authority_violations = [
        field for field in authority_false_fields if bool(authority.get(field, False))
    ]
    record(
        "authority_is_metadata_only",
        bool(authority.get("metadata_only", False)) and not false_authority_violations,
        {
            "metadata_only": bool(authority.get("metadata_only", False)),
            "false_field_violations": false_authority_violations,
        },
    )

    group_ids = [str(row.get("group_id") or "") for row in groups]
    record(
        "strategy_group_ids_unique", len(group_ids) == len(set(group_ids)), group_ids
    )

    missing_group_fields: list[str] = []
    for row in groups:
        group_id = str(row.get("group_id") or "")
        for field in required_group_fields:
            if not _field_present(row.get(field)):
                missing_group_fields.append(f"{group_id}.{field}")
    record(
        "strategy_groups_have_required_fields",
        not missing_group_fields,
        missing_group_fields,
    )

    known_sleeves = set(sleeve_objectives)
    unresolved_sleeves = sorted(
        {
            sleeve_id
            for row in groups
            for sleeve_id in _list_of_strings(row.get("primary_sleeves"))
            if sleeve_id not in known_sleeves
        }
    )
    record(
        "group_sleeve_references_resolve", not unresolved_sleeves, unresolved_sleeves
    )

    unresolved_objective_classes = sorted(
        {
            objective_id
            for row in groups
            for objective_id in _list_of_strings(row.get("objective_classes"))
            if objective_id not in objective_class_ids
        }
    )
    record(
        "group_objective_class_references_resolve",
        not unresolved_objective_classes,
        unresolved_objective_classes,
    )

    groups_without_tier_needs = [
        str(row.get("group_id") or "")
        for row in groups
        if not _list_of_strings(row.get("master_bot_needs"))
        or not _list_of_strings(row.get("grandmaster_bot_needs"))
    ]
    record(
        "groups_define_master_and_grandmaster_needs",
        not groups_without_tier_needs,
        groups_without_tier_needs,
    )

    mapped_sleeves = {
        sleeve_id
        for row in groups
        for sleeve_id in _list_of_strings(row.get("primary_sleeves"))
    }
    trading_sleeves = {
        sleeve_id
        for sleeve_id, objective_class in sleeve_objectives.items()
        if objective_class != "control_only"
    }
    unmapped_trading_sleeves = sorted(trading_sleeves - mapped_sleeves)
    record(
        "every_trading_sleeve_is_grouped",
        not unmapped_trading_sleeves,
        unmapped_trading_sleeves,
    )

    unmapped_strategy_additions = sorted(
        {
            sleeve_id
            for sleeve_id, strategies in strategy_additions.items()
            if sleeve_id in trading_sleeves
            and _list_of_strings(strategies)
            and sleeve_id not in mapped_sleeves
        }
    )
    record(
        "every_sleeve_strategy_addition_inherits_group",
        not unmapped_strategy_additions,
        unmapped_strategy_additions,
    )

    grouped_control_only = sorted(
        {
            sleeve_id
            for sleeve_id in mapped_sleeves
            if sleeve_objectives.get(sleeve_id) == "control_only"
        }
    )
    record(
        "control_only_sleeves_are_excluded_from_strategy_groups",
        not grouped_control_only,
        grouped_control_only,
    )

    failed = [check_id for check_id, row in checks.items() if not row.get("ok")]
    return {
        "overall_status": "needs_action" if failed else "ready",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "checks": checks,
        "invariants": invariants,
    }


def _strategy_organization_context(
    payload: dict[str, Any],
    sleeve_characteristics: dict[str, Any],
) -> dict[str, Any]:
    organization = _as_dict(payload.get("strategy_organization"))
    groups = _strategy_group_rows(organization.get("groups"))
    sleeve_rows = [
        row
        for row in _as_list(sleeve_characteristics.get("sleeves"))
        if isinstance(row, dict)
    ]
    sleeve_objectives = {
        str(row.get("sleeve_id") or ""): str(row.get("objective_class") or "")
        for row in sleeve_rows
        if str(row.get("sleeve_id") or "")
    }
    objective_class_ids = set(
        _as_dict(sleeve_characteristics.get("objective_characteristics"))
    )
    objective_class_ids.update(
        str(objective_id)
        for objective_id in _as_dict(
            sleeve_characteristics.get("objective_class_counts")
        )
    )
    strategy_additions = _as_dict(payload.get("strategy_additions"))
    hardening = _strategy_organization_hardening(
        organization=organization,
        groups=groups,
        sleeve_objectives=sleeve_objectives,
        objective_class_ids=objective_class_ids,
        strategy_additions=strategy_additions,
    )
    mapped_sleeves = sorted(
        {
            sleeve_id
            for row in groups
            for sleeve_id in _list_of_strings(row.get("primary_sleeves"))
            if sleeve_id in sleeve_objectives
        }
    )
    trading_sleeves = sorted(
        sleeve_id
        for sleeve_id, objective_class in sleeve_objectives.items()
        if objective_class != "control_only"
    )
    sleeve_group_map: list[dict[str, Any]] = []
    for sleeve_id in sorted(sleeve_objectives):
        group_ids = [
            str(row.get("group_id") or "")
            for row in groups
            if sleeve_id in _list_of_strings(row.get("primary_sleeves"))
        ]
        strategies = _list_of_strings(strategy_additions.get(sleeve_id))
        if not group_ids and not strategies:
            continue
        sleeve_group_map.append(
            {
                "sleeve_id": sleeve_id,
                "objective_class": sleeve_objectives.get(sleeve_id, ""),
                "strategy_groups": group_ids,
                "strategy_count": len(strategies),
                "strategies": strategies[:12],
            }
        )
    return {
        "present": bool(organization),
        "policy_id": str(payload.get("policy_id") or ""),
        "purpose": str(organization.get("purpose") or ""),
        "authority": _as_dict(organization.get("authority")),
        "hardening": hardening,
        "group_count": len(groups),
        "group_ids": [row["group_id"] for row in groups],
        "mapped_sleeve_count": len(mapped_sleeves),
        "mapped_sleeves": mapped_sleeves,
        "trading_sleeve_count": len(trading_sleeves),
        "unmapped_trading_sleeves": sorted(set(trading_sleeves) - set(mapped_sleeves)),
        "groups": groups,
        "sleeve_group_map": sleeve_group_map,
        "operator_output_requirements": _list_of_strings(
            organization.get("operator_output_requirements")
        ),
        "source_file": "config/sleeve_strategy_contracts_v1.json",
    }


def _coordination_success_need_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _as_list(value):
        item = _as_dict(row)
        need_id = str(item.get("need_id") or "").strip()
        if not need_id:
            continue
        rows.append(
            {
                "need_id": need_id,
                "why": str(item.get("why") or "").strip(),
                "evidence_artifacts": _list_of_strings(item.get("evidence_artifacts")),
                "success_signal": str(item.get("success_signal") or "").strip(),
            }
        )
    return rows


def _coordination_success_hardening(
    success: dict[str, Any],
    master_needs: list[dict[str, Any]],
    grandmaster_needs: list[dict[str, Any]],
) -> dict[str, Any]:
    invariants = _as_dict(success.get("hardening_invariants"))
    authority = _as_dict(success.get("authority"))
    authority_false_fields = _list_of_strings(
        invariants.get("authority_false_fields")
    ) or [
        "can_submit_order",
        "can_change_trade_logic",
        "can_change_sizing",
        "can_allocate_capital",
        "can_promote_candidate",
        "can_claim_profitability",
        "can_override_halt",
    ]
    required_need_fields = _list_of_strings(invariants.get("required_need_fields")) or [
        "need_id",
        "why",
        "evidence_artifacts",
        "success_signal",
    ]
    minimum_need_count = _safe_int(invariants.get("minimum_need_count_per_tier"), 6)
    checks: dict[str, dict[str, Any]] = {}

    def record(check_id: str, ok: bool, detail: Any) -> None:
        checks[check_id] = {"ok": bool(ok), "detail": detail}

    false_authority_violations = [
        field for field in authority_false_fields if bool(authority.get(field, False))
    ]
    record(
        "authority_is_metadata_only",
        bool(authority.get("metadata_only", False)) and not false_authority_violations,
        {
            "metadata_only": bool(authority.get("metadata_only", False)),
            "false_field_violations": false_authority_violations,
        },
    )
    master = _as_dict(success.get("master_bot"))
    grandmaster = _as_dict(success.get("grandmaster_bot"))
    record(
        "master_and_grandmaster_views_present",
        bool(str(master.get("view") or "").strip())
        and bool(str(grandmaster.get("view") or "").strip()),
        {
            "master_view": str(master.get("view") or ""),
            "grandmaster_view": str(grandmaster.get("view") or ""),
        },
    )
    record(
        "master_need_count_meets_floor",
        len(master_needs) >= minimum_need_count,
        {"current": len(master_needs), "minimum": minimum_need_count},
    )
    record(
        "grandmaster_need_count_meets_floor",
        len(grandmaster_needs) >= minimum_need_count,
        {"current": len(grandmaster_needs), "minimum": minimum_need_count},
    )

    missing_need_fields: list[str] = []
    for tier_id, rows in (
        ("master_bot", master_needs),
        ("grandmaster_bot", grandmaster_needs),
    ):
        for row in rows:
            need_id = str(row.get("need_id") or "")
            for field in required_need_fields:
                if not _field_present(row.get(field)):
                    missing_need_fields.append(f"{tier_id}.{need_id}.{field}")
    record(
        "success_needs_have_required_fields",
        not missing_need_fields,
        missing_need_fields,
    )
    record(
        "master_and_grandmaster_must_not_lists_present",
        bool(_list_of_strings(master.get("must_not")))
        and bool(_list_of_strings(grandmaster.get("must_not"))),
        {
            "master_bot": _list_of_strings(master.get("must_not")),
            "grandmaster_bot": _list_of_strings(grandmaster.get("must_not")),
        },
    )
    failed = [check_id for check_id, row in checks.items() if not row.get("ok")]
    return {
        "overall_status": "needs_action" if failed else "ready",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "checks": checks,
        "invariants": invariants,
    }


def _master_grandmaster_success_context(policy: dict[str, Any]) -> dict[str, Any]:
    success = _as_dict(policy.get("coordination_success_needs"))
    master = _as_dict(success.get("master_bot"))
    grandmaster = _as_dict(success.get("grandmaster_bot"))
    master_needs = _coordination_success_need_rows(master.get("needs"))
    grandmaster_needs = _coordination_success_need_rows(grandmaster.get("needs"))
    hardening = _coordination_success_hardening(
        success,
        master_needs,
        grandmaster_needs,
    )
    return {
        "present": bool(success),
        "policy_id": str(policy.get("policy_id") or ""),
        "purpose": str(success.get("purpose") or ""),
        "authority": _as_dict(success.get("authority")),
        "hardening": hardening,
        "master_bot": {
            "view": str(master.get("view") or ""),
            "need_count": len(master_needs),
            "need_ids": [row["need_id"] for row in master_needs],
            "needs": master_needs,
            "must_not": _list_of_strings(master.get("must_not")),
        },
        "grandmaster_bot": {
            "view": str(grandmaster.get("view") or ""),
            "need_count": len(grandmaster_needs),
            "need_ids": [row["need_id"] for row in grandmaster_needs],
            "needs": grandmaster_needs,
            "must_not": _list_of_strings(grandmaster.get("must_not")),
        },
        "source_file": "config/master_grandmaster_evidence_v2.json",
    }


def _need_from_operator_brain_boundary(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(context.get("present", False)):
        return [
            {
                "blocker": "operator_brain_boundary_missing",
                "exact_file": "config/system_role_contracts_v1.json",
                "exact_shard": "operator_brain_boundary",
                "command": ["./scripts/ops/opsctl.sh", "system-needs", "--json"],
                "expected_impact": "Defines the Trading Brain versus Ops Brain boundary so operational hardening can continue without accidentally mutating the current trading candidate.",
                "risk_level": "low",
                "when_to_stop": "operator_brain_boundary exists with trading_brain, ops_brain, and boundary_rules sections.",
                "source": "system_role_contracts",
            }
        ]
    hardening = _as_dict(context.get("hardening"))
    failed = _list_of_strings(hardening.get("failed_checks"))
    if not failed:
        return []
    return [
        {
            "blocker": "operator_brain_boundary_hardening_failed",
            "exact_file": "config/system_role_contracts_v1.json",
            "exact_shard": ",".join(failed),
            "command": ["./scripts/ops/opsctl.sh", "system-needs", "--json"],
            "expected_impact": "Repairs the checked Trading Brain and Ops Brain taxonomy before routing operator work across trading, ops, sleeve, or evidence boundaries.",
            "risk_level": "low",
            "when_to_stop": "brain_boundary_readout.hardening.overall_status is ready with zero failed checks.",
            "source": "system_role_contracts",
        }
    ]


def _need_from_market_pattern_feedback(context: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(context.get("present", False)):
        return [
            {
                "blocker": "market_pattern_feedback_missing",
                "exact_file": "governance/health/market_pattern_feedback_latest.json",
                "exact_shard": "",
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "market-pattern-feedback",
                    "--json",
                ],
                "expected_impact": "Builds the current pattern readout so sleeve paper collection can be segmented by regime, cycle phase, symbol drivers, and post-cost outcome.",
                "risk_level": "none",
                "when_to_stop": "market_pattern_feedback_latest.json exists with at least one pattern or an explicit no-pattern status.",
                "source": "market_pattern_feedback",
            }
        ]
    if (
        _safe_int(context.get("pattern_count"), 0) <= 0
        or _safe_int(context.get("observable_dimension_count"), 0) <= 0
    ):
        return [
            {
                "blocker": "market_pattern_feedback_thin",
                "exact_file": "governance/health/market_pattern_feedback_latest.json",
                "exact_shard": "patterns,observable_market_dimensions",
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "market-pattern-feedback",
                    "--json",
                ],
                "expected_impact": "Refreshes pattern detection and the list of observable market dimensions before the platform uses context to prioritize evidence collection.",
                "risk_level": "none",
                "when_to_stop": "pattern_count and observable_dimension_count are nonzero, or source_snapshot explains why market context is unavailable.",
                "source": "market_pattern_feedback",
            }
        ]
    needs: list[dict[str, Any]] = []
    if not bool(context.get("can_route_paper_collection_priority", False)):
        needs.append(
            {
                "blocker": "market_pattern_feedback_contract_not_routeable",
                "exact_file": "governance/health/market_pattern_feedback_latest.json",
                "exact_shard": "platform_feedback_contract.can_route_paper_collection_priority",
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "market-pattern-feedback",
                    "--json",
                ],
                "expected_impact": "Restores the advisory contract that lets patterns route paper-only collection priority without changing execution authority.",
                "risk_level": "low",
                "when_to_stop": "platform_feedback_contract.can_route_paper_collection_priority is true while live execution remains false.",
                "source": "market_pattern_feedback",
            }
        )
    if bool(context.get("has_symbol_specific_evidence_gap", False)):
        needs.append(
            {
                "blocker": "symbol_level_driver_attribution_collecting",
                "exact_file": "governance/health/market_move_explainer_latest.json",
                "exact_shard": "ranked_drivers,unknowns",
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "decision-intelligence",
                    "--symbol",
                    "BTC",
                    "--json",
                ],
                "expected_impact": "Turns a market-wide regime readout into symbol-level driver rows that can be joined to each paper intent and post-cost outcome.",
                "risk_level": "none",
                "when_to_stop": "market_move_explainer has ranked drivers for the watched symbol, or the unknowns list is empty for the current collection window.",
                "source": "market_pattern_feedback",
            }
        )
    return needs


def _need_from_paper_evidence_collection_context(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    if bool(context.get("safe_for_evidence_collection", False)):
        return []
    return [
        {
            "blocker": "paper_evidence_collection_controls_not_safe",
            "exact_file": "config/paper_evidence_collection_controls_v1.json",
            "exact_shard": "enabled,paper_only,live_execution_allowed,forbidden_shortcuts",
            "command": ["./scripts/ops/opsctl.sh", "system-needs", "--json"],
            "expected_impact": "Requires the paper collection loosening contract to remain paper-only, non-forcing, non-martingale, and unable to claim profitability.",
            "risk_level": "medium",
            "when_to_stop": "controls are enabled, paper_only is true, live_execution_allowed is false, and forbidden shortcuts remain false.",
            "source": "paper_evidence_collection_controls",
        }
    ]


def _need_from_sleeve_characteristics_context(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(context.get("present", False)):
        return [
            {
                "blocker": "sleeve_characteristics_missing",
                "exact_file": "config/sleeve_strategy_contracts_v1.json",
                "exact_shard": "sleeve_characteristics",
                "command": ["./scripts/ops/opsctl.sh", "system-needs", "--json"],
                "expected_impact": "Defines sleeve-specific market character, evidence type, and brain-routing metadata without changing strategy parameters.",
                "risk_level": "low",
                "when_to_stop": "sleeve_characteristics exists and describes every active sleeve objective class with metadata-only authority.",
                "source": "sleeve_strategy_contracts",
            }
        ]
    hardening = _as_dict(context.get("hardening"))
    failed = _list_of_strings(hardening.get("failed_checks"))
    if not failed:
        return []
    return [
        {
            "blocker": "sleeve_characteristics_hardening_failed",
            "exact_file": "config/sleeve_strategy_contracts_v1.json",
            "exact_shard": ",".join(failed),
            "command": ["./scripts/ops/opsctl.sh", "system-needs", "--json"],
            "expected_impact": "Repairs sleeve-characteristic coverage, authority locks, measurement focus, or brain-category references before using sleeve traits in operator decisions.",
            "risk_level": "low",
            "when_to_stop": "sleeve_characteristics_readout.hardening.overall_status is ready with zero failed checks.",
            "source": "sleeve_strategy_contracts",
        }
    ]


def _need_from_strategy_organization_context(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(context.get("present", False)):
        return [
            {
                "blocker": "strategy_organization_missing",
                "exact_file": "config/sleeve_strategy_contracts_v1.json",
                "exact_shard": "strategy_organization",
                "command": ["./scripts/ops/opsctl.sh", "system-needs", "--json"],
                "expected_impact": "Groups every trading sleeve strategy into market archetypes so Master and Grandmaster bots can compare like with like without changing parameters.",
                "risk_level": "low",
                "when_to_stop": "strategy_organization exists with metadata-only authority and every trading sleeve mapped to at least one group.",
                "source": "sleeve_strategy_contracts",
            }
        ]
    hardening = _as_dict(context.get("hardening"))
    failed = _list_of_strings(hardening.get("failed_checks"))
    if not failed:
        return []
    return [
        {
            "blocker": "strategy_organization_hardening_failed",
            "exact_file": "config/sleeve_strategy_contracts_v1.json",
            "exact_shard": ",".join(failed),
            "command": ["./scripts/ops/opsctl.sh", "system-needs", "--json"],
            "expected_impact": "Repairs strategy group coverage, authority locks, sleeve/objective references, and tier-specific aggregation needs before using group taxonomy in operator decisions.",
            "risk_level": "low",
            "when_to_stop": "strategy_organization_readout.hardening.overall_status is ready with zero failed checks.",
            "source": "sleeve_strategy_contracts",
        }
    ]


def _need_from_master_grandmaster_success_context(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(context.get("present", False)):
        return [
            {
                "blocker": "master_grandmaster_success_needs_missing",
                "exact_file": "config/master_grandmaster_evidence_v2.json",
                "exact_shard": "coordination_success_needs",
                "command": ["./scripts/ops/opsctl.sh", "system-needs", "--json"],
                "expected_impact": "Defines what Master and Grandmaster bots need beyond raw data collection to make useful advisory coordination decisions.",
                "risk_level": "low",
                "when_to_stop": "coordination_success_needs exists with master_bot and grandmaster_bot needs, success signals, and metadata-only authority.",
                "source": "master_grandmaster_evidence",
            }
        ]
    hardening = _as_dict(context.get("hardening"))
    failed = _list_of_strings(hardening.get("failed_checks"))
    if not failed:
        return []
    return [
        {
            "blocker": "master_grandmaster_success_needs_hardening_failed",
            "exact_file": "config/master_grandmaster_evidence_v2.json",
            "exact_shard": ",".join(failed),
            "command": ["./scripts/ops/opsctl.sh", "system-needs", "--json"],
            "expected_impact": "Repairs Master/Grandmaster success needs, success signals, must-not rules, or authority locks before the tiered bot hierarchy can explain what it needs.",
            "risk_level": "low",
            "when_to_stop": "master_grandmaster_success_readout.hardening.overall_status is ready with zero failed checks.",
            "source": "master_grandmaster_evidence",
        }
    ]


def _need_domain(item: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(item.get("blocker") or ""),
            str(item.get("source") or ""),
            str(item.get("exact_file") or ""),
        ]
    ).lower()
    if (
        "profit" in text
        or "post_cost" in text
        or "fill" in text
        or "sleeve_economic" in text
    ):
        return "profitability_evidence"
    if "market_pattern" in text or "symbol_level" in text or "market_move" in text:
        return "market_context"
    if "storage" in text or "backpressure" in text or "writer" in text:
        return "storage_writer"
    if "training" in text or "quality" in text or "low_grade" in text:
        return "training_quality"
    if "memory" in text or "runtime" in text or "mlx" in text:
        return "runtime_resources"
    if "halt" in text or "auth" in text or "live" in text:
        return "safety"
    return "system"


def _priority_for_need(item: dict[str, Any]) -> int:
    blocker = str(item.get("blocker") or "").lower()
    domain = _need_domain(item)
    if "live" in blocker or "halt" in blocker or "auth" in blocker:
        return 10
    if domain == "storage_writer":
        return 20
    if blocker in {
        "candidate_post_cost_observations_collecting",
        "independent_fill_evidence_by_market_collecting",
        "candidate_bound_threshold_and_exit_replay_collecting",
        "economic_profitability_evidence_collecting",
    }:
        return 30
    if domain == "market_context":
        return 40
    if domain == "training_quality":
        return 50
    if domain == "runtime_resources":
        return 60
    return 70


def _need_priority_ladder(needs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = sorted(
        needs,
        key=lambda item: (
            _priority_for_need(item),
            str(item.get("blocker") or ""),
            str(item.get("exact_file") or ""),
        ),
    )
    for idx, item in enumerate(ordered[:15], start=1):
        rows.append(
            {
                "rank": idx,
                "domain": _need_domain(item),
                "blocker": str(item.get("blocker") or ""),
                "exact_file": str(item.get("exact_file") or ""),
                "exact_shard": str(item.get("exact_shard") or ""),
                "command": _command(item.get("command")),
                "expected_impact": str(item.get("expected_impact") or ""),
                "risk_level": str(item.get("risk_level") or ""),
                "when_to_stop": str(item.get("when_to_stop") or ""),
                "source": str(item.get("source") or ""),
            }
        )
    return rows


def _operator_communication_packet(
    *,
    actionable_needs: list[dict[str, Any]],
    managed_controls: list[dict[str, Any]],
    profitability: dict[str, Any],
    raw_profitability_recovery: dict[str, Any],
    market_patterns: dict[str, Any],
    paper_collection: dict[str, Any],
    brain_boundary: dict[str, Any],
    sleeve_characteristics: dict[str, Any],
    strategy_organization: dict[str, Any],
    master_grandmaster_success: dict[str, Any],
) -> dict[str, Any]:
    market_gaps = _as_dict(market_patterns.get("profitability_evidence_gaps"))
    sample_count = _safe_int(
        profitability.get("candidate_post_cost_sample_count"),
        _safe_int(market_gaps.get("candidate_post_cost_sample_count"), 0),
    )
    min_samples = _safe_int(
        profitability.get("candidate_post_cost_minimum_samples"),
        _safe_int(market_gaps.get("candidate_post_cost_minimum_samples"), 30),
    )
    observed_days = _safe_int(
        profitability.get("candidate_observed_days"),
        _safe_int(market_gaps.get("candidate_observed_days"), 0),
    )
    min_days = _safe_int(
        profitability.get("candidate_minimum_observed_days"),
        _safe_int(market_gaps.get("candidate_minimum_observed_days"), 3),
    )
    fill_records = _safe_int(
        profitability.get("candidate_independent_fill_records"),
        _safe_int(market_gaps.get("candidate_independent_fill_records"), 0),
    )
    min_fill_records = _safe_int(
        profitability.get("candidate_independent_fill_minimum_records"), 30
    )
    positive_lcb = bool(
        profitability.get(
            "positive_post_cost_lower_confidence_bound_95",
            market_gaps.get("positive_post_cost_lcb", False),
        )
    )
    implementation_grade = str(profitability.get("implementation_grade") or "unknown")
    economic_grade = str(profitability.get("economic_evidence_grade") or "unknown")
    candidate_id = str(profitability.get("candidate_id") or "")
    dominant = _list_of_strings(market_patterns.get("dominant_pattern_ids"))
    prioritized = _list_of_strings(market_patterns.get("prioritized_sleeves"))
    downshift = _list_of_strings(market_patterns.get("downshift_sleeves"))
    plain_status = str(profitability.get("system_statement") or "").strip() or (
        f"Candidate {candidate_id or 'unknown'} is implementation grade "
        f"{implementation_grade}, but economic evidence is {economic_grade}."
    )
    if not bool(profitability.get("economic_evidence_ready", False)):
        direct_answer = (
            "It is not profitable on evidence yet: the code/control layer is ready, "
            "but the current candidate still needs post-cost samples, persistence by day, "
            "independent fill evidence, and a positive conservative lower bound."
        )
    else:
        direct_answer = (
            "The current candidate has passed the economic evidence gate; live authority "
            "still depends on the separate live/promotion controls."
        )
    if bool(paper_collection.get("safe_for_evidence_collection", False)):
        paper_answer = (
            "Paper evidence collection can be loosened inside the bounded sleeve policy: "
            "more samples are allowed only when point-in-time context, model edge, "
            "liquidity, spread, quote age, and reversal checks pass."
        )
    else:
        paper_answer = (
            "Paper evidence collection should not loosen until the collection-control "
            "artifact is safely enabled and explicitly paper-only."
        )
    trading_boundary = _as_dict(brain_boundary.get("trading_brain"))
    ops_boundary = _as_dict(brain_boundary.get("ops_brain"))
    trading_categories = _as_list(trading_boundary.get("categories"))
    ops_categories = _as_list(ops_boundary.get("categories"))
    master_success = _as_dict(master_grandmaster_success.get("master_bot"))
    grandmaster_success = _as_dict(master_grandmaster_success.get("grandmaster_bot"))
    return {
        "plain_english_status": plain_status,
        "direct_profitability_answer": direct_answer,
        "paper_collection_answer": paper_answer,
        "current_candidate": {
            "candidate_id": candidate_id,
            "identity_consistent": bool(
                profitability.get("candidate_identity_consistent", False)
            ),
            "implementation_grade": implementation_grade,
            "implementation_score": _safe_float(
                profitability.get("implementation_score"), 0.0
            ),
            "economic_evidence_grade": economic_grade,
            "economic_evidence_score": _safe_float(
                profitability.get("economic_evidence_score"), 0.0
            ),
            "economic_evidence_ready": bool(
                profitability.get("economic_evidence_ready", False)
            ),
        },
        "why_not_profitable_yet": _unique(
            [
                f"current_candidate_post_cost_samples={sample_count}/{min_samples}",
                f"candidate_observed_days={observed_days}/{min_days}",
                f"candidate_independent_fill_records={fill_records}/{min_fill_records}",
                f"positive_post_cost_lcb_95={positive_lcb}",
                f"historical_active_book_net_pnl={_safe_float(profitability.get('historical_active_book_net_pnl'), 0.0):.6f}",
                (
                    "historical_active_book_not_current_candidate_grade_evidence"
                    if not bool(
                        profitability.get(
                            "historical_active_book_candidate_grade_eligible", False
                        )
                    )
                    else ""
                ),
                f"market_patterns={','.join(dominant) or 'none'}",
                (
                    "symbol_level_driver_evidence_gap"
                    if bool(
                        market_patterns.get("has_symbol_specific_evidence_gap", False)
                    )
                    else ""
                ),
            ]
        ),
        "exactly_needed_to_call_it_profitable": [
            {
                "need": "current_candidate_post_cost_outcomes",
                "current": sample_count,
                "target": min_samples,
                "success_condition": "at least the configured minimum schema-v2 post-cost observations exist for the current candidate",
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "paper-performance",
                    "--week-days",
                    "7",
                    "--json",
                ],
            },
            {
                "need": "persistence_across_days",
                "current": observed_days,
                "target": min_days,
                "success_condition": "candidate evidence spans the configured minimum observed days",
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "profitability-self-assessment",
                    "--json",
                ],
            },
            {
                "need": "independent_fill_evidence",
                "current": fill_records,
                "target": min_fill_records,
                "success_condition": "independent fills exist by required market type and are bound to the candidate",
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "independent-fill-acquisition",
                    "--apply",
                    "--json",
                ],
            },
            {
                "need": "positive_conservative_post_cost_lower_bound",
                "current": positive_lcb,
                "target": True,
                "success_condition": "the current candidate has positive 95 percent post-cost lower confidence evidence",
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "profitability-evidence-firewall",
                    "--json",
                ],
            },
            {
                "need": "sleeve_breadth_and_context",
                "current": len(prioritized),
                "target": 4,
                "success_condition": "enough independently profitable, low-correlation sleeves qualify without relying on one cluster",
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "sleeve-profitability-dashboard",
                    "--json",
                ],
            },
        ],
        "market_pattern_readout": {
            "overall_status": str(market_patterns.get("overall_status") or ""),
            "pattern_count": _safe_int(market_patterns.get("pattern_count"), 0),
            "dominant_patterns": _as_list(market_patterns.get("dominant_patterns")),
            "prioritized_sleeves": prioritized,
            "downshift_sleeves": downshift,
            "context_only_sleeves": _list_of_strings(
                market_patterns.get("context_only_sleeves")
            )[:12],
            "recommended_actions": _list_of_strings(
                market_patterns.get("recommended_actions")
            )[:8],
        },
        "sleeve_characteristics_readout": {
            "policy_id": str(sleeve_characteristics.get("policy_id") or ""),
            "purpose": str(sleeve_characteristics.get("purpose") or ""),
            "classification_rule": str(
                sleeve_characteristics.get("classification_rule") or ""
            ),
            "authority": _as_dict(sleeve_characteristics.get("authority")),
            "sleeve_count": _safe_int(sleeve_characteristics.get("sleeve_count"), 0),
            "characterized_sleeve_count": _safe_int(
                sleeve_characteristics.get("characterized_sleeve_count"), 0
            ),
            "objective_class_count": _safe_int(
                sleeve_characteristics.get("objective_class_count"), 0
            ),
            "objective_class_counts": _as_dict(
                sleeve_characteristics.get("objective_class_counts")
            ),
            "hardening": _as_dict(sleeve_characteristics.get("hardening")),
            "prioritized_sleeves": _as_list(
                sleeve_characteristics.get("prioritized_sleeves")
            )[:12],
            "downshift_sleeves": _as_list(
                sleeve_characteristics.get("downshift_sleeves")
            )[:12],
            "context_only_sleeves": _as_list(
                sleeve_characteristics.get("context_only_sleeves")
            )[:12],
            "sleeves": _as_list(sleeve_characteristics.get("sleeves")),
            "source_file": str(sleeve_characteristics.get("source_file") or ""),
        },
        "strategy_organization_readout": {
            "policy_id": str(strategy_organization.get("policy_id") or ""),
            "purpose": str(strategy_organization.get("purpose") or ""),
            "authority": _as_dict(strategy_organization.get("authority")),
            "group_count": _safe_int(strategy_organization.get("group_count"), 0),
            "group_ids": _list_of_strings(strategy_organization.get("group_ids")),
            "mapped_sleeve_count": _safe_int(
                strategy_organization.get("mapped_sleeve_count"), 0
            ),
            "trading_sleeve_count": _safe_int(
                strategy_organization.get("trading_sleeve_count"), 0
            ),
            "unmapped_trading_sleeves": _list_of_strings(
                strategy_organization.get("unmapped_trading_sleeves")
            ),
            "hardening": _as_dict(strategy_organization.get("hardening")),
            "groups": _as_list(strategy_organization.get("groups")),
            "sleeve_group_map": _as_list(strategy_organization.get("sleeve_group_map")),
            "source_file": str(strategy_organization.get("source_file") or ""),
        },
        "master_grandmaster_success_readout": {
            "policy_id": str(master_grandmaster_success.get("policy_id") or ""),
            "purpose": str(master_grandmaster_success.get("purpose") or ""),
            "authority": _as_dict(master_grandmaster_success.get("authority")),
            "master_bot": master_success,
            "grandmaster_bot": grandmaster_success,
            "master_need_count": _safe_int(master_success.get("need_count"), 0),
            "grandmaster_need_count": _safe_int(
                grandmaster_success.get("need_count"), 0
            ),
            "master_need_ids": _list_of_strings(master_success.get("need_ids")),
            "grandmaster_need_ids": _list_of_strings(
                grandmaster_success.get("need_ids")
            ),
            "hardening": _as_dict(master_grandmaster_success.get("hardening")),
            "source_file": str(master_grandmaster_success.get("source_file") or ""),
        },
        "paper_evidence_collection_readout": {
            "safe_for_evidence_collection": bool(
                paper_collection.get("safe_for_evidence_collection", False)
            ),
            "policy_id": str(paper_collection.get("policy_id") or ""),
            "profile_count": _safe_int(paper_collection.get("profile_count"), 0),
            "paper_only": bool(paper_collection.get("paper_only", False)),
            "live_execution_allowed": False,
            "non_relaxed_controls": _list_of_strings(
                paper_collection.get("non_relaxed_controls")
            ),
        },
        "brain_boundary_readout": {
            "boundary_id": str(brain_boundary.get("boundary_id") or ""),
            "summary": str(brain_boundary.get("summary") or ""),
            "purpose": str(brain_boundary.get("purpose") or ""),
            "trading_brain": trading_boundary,
            "ops_brain": ops_boundary,
            "trading_category_count": len(trading_categories),
            "ops_category_count": len(ops_categories),
            "trading_categories": _brain_boundary_category_ids(trading_categories),
            "ops_categories": _brain_boundary_category_ids(ops_categories),
            "boundary_rules": _as_dict(brain_boundary.get("boundary_rules")),
            "routing_matrix": _as_dict(brain_boundary.get("routing_matrix")),
            "hardening": _as_dict(brain_boundary.get("hardening")),
            "safe_work_now": str(brain_boundary.get("safe_work_now") or ""),
            "frozen_work_now": str(brain_boundary.get("frozen_work_now") or ""),
            "trading_freeze_reason": str(
                brain_boundary.get("trading_freeze_reason") or ""
            ),
            "source_file": str(brain_boundary.get("source_file") or ""),
        },
        "priority_ladder": _need_priority_ladder(actionable_needs),
        "managed_control_count": len(managed_controls),
        "next_command": (
            _command(actionable_needs[0].get("command")) if actionable_needs else []
        ),
        "what_not_to_do": [
            "do_not_claim_profitability_from_implementation_grade",
            "do_not_enable_live_execution_from_market_patterns",
            "do_not_force_trades_to_create_samples",
            "do_not_increase_size_to_recover_losses",
            "do_not_count_historical_losses_or_old_generations_as_current_candidate_profit_proof",
            "do_not_change_trading_brain_while_current_candidate_collects_evidence",
            "do_not_treat_ops_brain_repairs_as_profitability_evidence",
            "do_not_treat_strategy_groups_as_trade_authority",
            "do_not_promote_strategy_groups_without_candidate_bound_post_cost_evidence",
            "do_not_treat_master_grandmaster_success_needs_as_live_authority",
        ],
        "confidence": {
            "level": (
                "medium"
                if bool(market_patterns.get("present", False))
                and bool(profitability.get("present", False))
                else "low"
            ),
            "reason": "needs are exact when profitability assessment and market-pattern feedback are both present",
        },
        "raw_profitability_recovery": {
            "historical_context_only": bool(
                raw_profitability_recovery.get("historical_context_only", False)
            ),
            "active": bool(raw_profitability_recovery.get("active", False)),
            "top_loss_causes": _list_of_strings(
                raw_profitability_recovery.get("top_loss_causes")
            )[:8],
            "top_drag_profiles": _as_list(
                raw_profitability_recovery.get("top_drag_profiles")
            )[:5],
        },
    }


def render_operator_needs_markdown(payload: dict[str, Any]) -> str:
    communication = _as_dict(payload.get("operator_communication"))
    ladder = _as_list(communication.get("priority_ladder"))
    exact = _as_list(communication.get("exactly_needed_to_call_it_profitable"))
    market = _as_dict(communication.get("market_pattern_readout"))
    sleeve_chars = _as_dict(communication.get("sleeve_characteristics_readout"))
    strategy_org = _as_dict(communication.get("strategy_organization_readout"))
    master_grandmaster = _as_dict(
        communication.get("master_grandmaster_success_readout")
    )
    boundary = _as_dict(communication.get("brain_boundary_readout"))
    trading = _as_dict(boundary.get("trading_brain"))
    ops = _as_dict(boundary.get("ops_brain"))
    lines = [
        "# System Needs",
        "",
        f"- Timestamp UTC: `{payload.get('timestamp_utc', '')}`",
        f"- Status: `{payload.get('overall_status', '')}`",
        f"- Need Count: `{payload.get('need_count', 0)}`",
        f"- Managed Control Count: `{payload.get('managed_control_count', 0)}`",
        "",
        "## Direct Read",
        "",
        communication.get("plain_english_status", ""),
        "",
        communication.get("direct_profitability_answer", ""),
        "",
        communication.get("paper_collection_answer", ""),
        "",
        "## Why Not Profitable Yet",
        "",
    ]
    for item in _list_of_strings(communication.get("why_not_profitable_yet")):
        lines.append(f"- {item}")
    lines.extend(["", "## Exactly Needed", ""])
    for row in exact:
        item = _as_dict(row)
        lines.append(
            f"- `{item.get('need', '')}` current `{item.get('current', '')}` target `{item.get('target', '')}`: "
            f"{item.get('success_condition', '')}"
        )
    if boundary:
        lines.extend(
            [
                "",
                "## Trading/Ops Boundary",
                "",
                f"- Summary: {boundary.get('summary', '')}",
                f"- Trading Brain: `{trading.get('current_posture', '')}` - {trading.get('definition', '')}",
                f"- Ops Brain: `{ops.get('current_posture', '')}` - {ops.get('definition', '')}",
                f"- Safe Work Now: {boundary.get('safe_work_now', '')}",
                f"- Frozen Work Now: {boundary.get('frozen_work_now', '')}",
                f"- Freeze Reason: {boundary.get('trading_freeze_reason', '')}",
            ]
        )
        trading_categories = _as_list(trading.get("categories"))
        ops_categories = _as_list(ops.get("categories"))
        if trading_categories:
            lines.append(
                f"- Trading Categories: `{','.join(_brain_boundary_category_ids(trading_categories))}`"
            )
            for row in trading_categories[:8]:
                item = _as_dict(row)
                lines.append(
                    f"- Trading `{item.get('category_id', '')}`: {item.get('purpose', '')} "
                    f"posture `{item.get('current_posture', '')}`"
                )
        if ops_categories:
            lines.append(
                f"- Ops Categories: `{','.join(_brain_boundary_category_ids(ops_categories))}`"
            )
            for row in ops_categories[:8]:
                item = _as_dict(row)
                lines.append(
                    f"- Ops `{item.get('category_id', '')}`: {item.get('purpose', '')} "
                    f"posture `{item.get('current_posture', '')}`"
                )
        routing = _as_dict(boundary.get("routing_matrix"))
        if routing:
            if str(routing.get("classification_rule") or ""):
                lines.append(
                    f"- Routing Rule: {routing.get('classification_rule', '')}"
                )
            lines.append(
                f"- Safe Now Categories: `{','.join(_list_of_strings(routing.get('safe_now'))) or 'none'}`"
            )
            lines.append(
                f"- Frozen Categories: `{','.join(_list_of_strings(routing.get('freeze_now'))) or 'none'}`"
            )
            lines.append(
                f"- Review Required: `{','.join(_list_of_strings(routing.get('review_required'))) or 'none'}`"
            )
        boundary_hardening = _as_dict(boundary.get("hardening"))
        if boundary_hardening:
            lines.append(
                f"- Boundary Hardening: `{boundary_hardening.get('overall_status', '')}` "
                f"failed `{boundary_hardening.get('failed_check_count', 0)}`"
            )
            failed = _list_of_strings(boundary_hardening.get("failed_checks"))
            if failed:
                lines.append(f"- Boundary Failed Checks: `{','.join(failed)}`")
    if sleeve_chars:
        lines.extend(
            [
                "",
                "## Sleeve Characteristics",
                "",
                f"- Policy: `{sleeve_chars.get('policy_id', '')}`",
                f"- Sleeve Count: `{sleeve_chars.get('characterized_sleeve_count', 0)}/{sleeve_chars.get('sleeve_count', 0)}` characterized across `{sleeve_chars.get('objective_class_count', 0)}` objective classes",
                f"- Rule: {sleeve_chars.get('classification_rule', '')}",
            ]
        )
        objective_counts = _as_dict(sleeve_chars.get("objective_class_counts"))
        if objective_counts:
            lines.append(
                "- Objective Classes: "
                + ", ".join(
                    f"`{key}`={objective_counts[key]}"
                    for key in sorted(objective_counts)
                )
            )
        sleeve_hardening = _as_dict(sleeve_chars.get("hardening"))
        if sleeve_hardening:
            lines.append(
                f"- Sleeve Hardening: `{sleeve_hardening.get('overall_status', '')}` "
                f"failed `{sleeve_hardening.get('failed_check_count', 0)}`"
            )
            failed = _list_of_strings(sleeve_hardening.get("failed_checks"))
            if failed:
                lines.append(f"- Sleeve Failed Checks: `{','.join(failed)}`")
        for label, rows in (
            ("Prioritized", _as_list(sleeve_chars.get("prioritized_sleeves"))),
            ("Downshift", _as_list(sleeve_chars.get("downshift_sleeves"))),
            ("Context Only", _as_list(sleeve_chars.get("context_only_sleeves"))),
        ):
            if not rows:
                continue
            lines.append(f"- {label} Sleeve Traits:")
            for row in rows[:8]:
                item = _as_dict(row)
                lines.append(
                    f"- `{item.get('sleeve_id', '')}` `{item.get('objective_class', '')}` "
                    f"`{item.get('primary_character', '')}` sees `{item.get('market_question', '')}`"
                )
    if strategy_org:
        lines.extend(
            [
                "",
                "## Strategy Organization",
                "",
                f"- Policy: `{strategy_org.get('policy_id', '')}`",
                f"- Groups: `{strategy_org.get('group_count', 0)}` mapped sleeves `{strategy_org.get('mapped_sleeve_count', 0)}/{strategy_org.get('trading_sleeve_count', 0)}`",
                f"- Unmapped Trading Sleeves: `{','.join(_list_of_strings(strategy_org.get('unmapped_trading_sleeves'))) or 'none'}`",
            ]
        )
        strategy_hardening = _as_dict(strategy_org.get("hardening"))
        if strategy_hardening:
            lines.append(
                f"- Strategy Hardening: `{strategy_hardening.get('overall_status', '')}` "
                f"failed `{strategy_hardening.get('failed_check_count', 0)}`"
            )
            failed = _list_of_strings(strategy_hardening.get("failed_checks"))
            if failed:
                lines.append(f"- Strategy Failed Checks: `{','.join(failed)}`")
        for row in _as_list(strategy_org.get("groups")):
            item = _as_dict(row)
            lines.append(
                f"- Group `{item.get('group_id', '')}` {item.get('label', '')}: "
                f"sleeves `{','.join(_list_of_strings(item.get('primary_sleeves'))) or 'none'}`; "
                f"archetypes `{','.join(_list_of_strings(item.get('strategy_archetypes'))[:4]) or 'none'}`"
            )
            lines.append(
                f"- Group `{item.get('group_id', '')}` master needs `{','.join(_list_of_strings(item.get('master_bot_needs'))) or 'none'}`; "
                f"grandmaster needs `{','.join(_list_of_strings(item.get('grandmaster_bot_needs'))) or 'none'}`"
            )
        sleeve_group_map = _as_list(strategy_org.get("sleeve_group_map"))
        if sleeve_group_map:
            lines.append("- Sleeve Strategy Map:")
            for row in sleeve_group_map[:12]:
                item = _as_dict(row)
                lines.append(
                    f"- `{item.get('sleeve_id', '')}` strategies `{item.get('strategy_count', 0)}` "
                    f"groups `{','.join(_list_of_strings(item.get('strategy_groups'))) or 'none'}`"
                )
    if master_grandmaster:
        master = _as_dict(master_grandmaster.get("master_bot"))
        grandmaster = _as_dict(master_grandmaster.get("grandmaster_bot"))
        lines.extend(
            [
                "",
                "## Master/Grandmaster Needs",
                "",
                f"- Master View: `{master.get('view', '')}`",
                f"- Grandmaster View: `{grandmaster.get('view', '')}`",
                f"- Master Needs: `{','.join(_list_of_strings(master.get('need_ids'))) or 'none'}`",
                f"- Grandmaster Needs: `{','.join(_list_of_strings(grandmaster.get('need_ids'))) or 'none'}`",
            ]
        )
        success_hardening = _as_dict(master_grandmaster.get("hardening"))
        if success_hardening:
            lines.append(
                f"- Master/Grandmaster Hardening: `{success_hardening.get('overall_status', '')}` "
                f"failed `{success_hardening.get('failed_check_count', 0)}`"
            )
            failed = _list_of_strings(success_hardening.get("failed_checks"))
            if failed:
                lines.append(
                    f"- Master/Grandmaster Failed Checks: `{','.join(failed)}`"
                )
        for label, rows in (
            ("Master", _as_list(master.get("needs"))),
            ("Grandmaster", _as_list(grandmaster.get("needs"))),
        ):
            for row in rows[:6]:
                item = _as_dict(row)
                lines.append(
                    f"- {label} `{item.get('need_id', '')}`: {item.get('success_signal', '')}"
                )
    lines.extend(
        [
            "",
            "## Market Patterns",
            "",
            f"- Status: `{market.get('overall_status', '')}`",
            f"- Pattern Count: `{market.get('pattern_count', 0)}`",
            f"- Prioritized Sleeves: `{','.join(_list_of_strings(market.get('prioritized_sleeves'))) or 'none'}`",
            f"- Downshift Sleeves: `{','.join(_list_of_strings(market.get('downshift_sleeves'))) or 'none'}`",
        ]
    )
    for row in _as_list(market.get("dominant_patterns")):
        item = _as_dict(row)
        lines.append(
            f"- `{item.get('pattern_id', '')}` {item.get('label', '')} strength `{item.get('strength', 0)}`"
        )
    lines.extend(["", "## Priority Ladder", ""])
    if not ladder:
        lines.append("- None.")
    for row in ladder[:12]:
        item = _as_dict(row)
        lines.append(
            f"- `{item.get('rank', '')}` `{item.get('domain', '')}` `{item.get('blocker', '')}` "
            f"file `{item.get('exact_file', '')}` command `{' '.join(str(part) for part in _as_list(item.get('command')))}`"
        )
    lines.extend(["", "## Do Not Do", ""])
    for item in _list_of_strings(communication.get("what_not_to_do")):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


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
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "ingestion-storage-control",
                    "--json",
                ],
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


def _need_from_training_runtime(
    training_runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    contract = _as_dict(training_runtime.get("training_launch_contract"))
    blockers = [
        str(item or "").strip()
        for item in _as_list(contract.get("launch_blockers"))
        if str(item or "").strip()
    ]
    if bool(contract.get("launch_allowed", False)) or not blockers:
        return []
    prep_commands = [
        _command(item) for item in _as_list(contract.get("recommended_prep_commands"))
    ]
    quota_gate = _as_dict(contract.get("storage_quota_training_gate"))
    blocked_quota_families = [
        str(item or "").strip()
        for item in _as_list(quota_gate.get("blocked_families"))
        if str(item or "").strip()
    ]

    def command_for(blocker: str) -> list[Any]:
        command_needle = ""
        if "storage_quota" in blocker:
            if "governance_telemetry" in blocked_quota_families:
                return [
                    "./scripts/ops/opsctl.sh",
                    "governance-telemetry-compactor",
                    "--apply",
                    "--json",
                ]
            command_needle = "storage-quota-guard"
        elif "writer" in blocker or "drain" in blocker:
            command_needle = "writer-cycle-coordinator"
        elif "memory" in blocker or "headroom" in blocker or "multitasking" in blocker:
            command_needle = "memory-pressure-intelligence"
        elif "runtime_snapshot" in blocker:
            command_needle = "runtime-training-snapshot"
        for command in prep_commands:
            if command_needle and command_needle in " ".join(
                str(part) for part in command
            ):
                return command
        if prep_commands:
            return prep_commands[0]
        return [
            "./scripts/ops/opsctl.sh",
            "training-runtime-control",
            "--limit",
            str(_safe_int(contract.get("requested_batch_size"), 30) or 30),
            "--json",
        ]

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
    return [
        str(item or "").strip() for item in _as_list(value) if str(item or "").strip()
    ]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _status_needs_repair(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(
        text and text not in {"ok", "ready", "running", "healthy", "clear", "stable"}
    )


def _soak_management_context(
    project_root: Path, health_fast: dict[str, Any]
) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    soak = load_json(health / "unattended_soak_readiness_latest.json")
    paper_guard = load_json(health / "runtime_paper_regression_guard_latest.json")
    soak_status = (
        str(soak.get("overall_status") or soak.get("status") or "").strip().lower()
    )
    soak_grade = (
        str(soak.get("overall_grade") or soak.get("grade") or "").strip().upper()
    )
    soak_ready = bool(soak.get("safe_to_leave_unattended", False)) and soak_status in {
        "ready",
        "ok",
        "healthy",
    }
    if soak_grade and soak_grade not in {"A", "A+"}:
        soak_ready = False
    paper_status = (
        str(paper_guard.get("overall_status") or paper_guard.get("status") or "")
        .strip()
        .lower()
    )
    paper_guard_clean = (
        bool(paper_guard.get("ok", False))
        and paper_status in {"ready", "ok", "healthy"}
        and _safe_int(paper_guard.get("failed_guard_count"), 0) <= 0
        and not _as_list(paper_guard.get("failed_guards"))
    )
    health_status = (
        str(health_fast.get("overall_status") or health_fast.get("status") or "")
        .strip()
        .lower()
    )
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
    if (
        blocker in SOAK_MANAGED_TRAINING_BLOCKERS
        and source == "training_runtime_control"
    ):
        return "training_expansion_parked_for_unattended_soak"
    if (
        blocker in SOAK_MANAGED_GOVERNOR_BLOCKERS
        and source == "autonomic_resource_governor"
    ):
        return "optional_mlx_capacity_deferred_during_unattended_soak"
    if (
        blocker in SOAK_MANAGED_MEMORY_BLOCKERS
        and source == "memory_pressure_intelligence"
    ):
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
    capability_materialization: dict[str, Any],
    capability_materialization_configured: bool,
    collector_capabilities: dict[str, Any],
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
    heartbeat_ok = bool(
        all_sleeves.get("heartbeat_fresh", all_sleeves.get("heartbeat", True))
    )
    if all_sleeves and (
        _status_needs_repair(all_sleeves_status)
        or not launcher_live
        or not child_fanout_ok
        or not heartbeat_ok
    ):
        needs.append(
            {
                "blocker": "all_sleeves_launcher_fanout_needs_repair",
                "exact_file": "governance/health/process_watchdog_latest.json",
                "exact_shard": "all_sleeves",
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "start",
                    "--paper",
                    "--run-all-sleeves",
                ],
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
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "source-verification-refresh",
                    "--apply",
                    "--json",
                ],
                "expected_impact": "Refreshes required context collectors and reruns the collector contract evidence used by health gates.",
                "risk_level": "low",
                "when_to_stop": "collector_contracts.required_failures is empty and health_gates.hard_gates.collector_contracts is false.",
                "source": "collector_contracts",
            }
        )

    materialized_rows = [
        row
        for row in _as_list(capability_materialization.get("capabilities"))
        if isinstance(row, dict)
    ]
    direct_ready_count = sum(
        1
        for row in materialized_rows
        if row.get("usable") is True
        and str(row.get("proof_semantics") or "") == "direct"
        and bool(str(row.get("proof_receipt_sha256") or ""))
    )
    if capability_materialization_configured and (
        str(capability_materialization.get("overall_status") or "") != "ready"
        or capability_materialization.get("live_promotion_ready") is not True
        or direct_ready_count < 4
    ):
        needs.append(
            {
                "blocker": "capability_materialization_not_ready",
                "exact_file": "governance/collector_capabilities/materialized_capabilities_latest.json",
                "exact_shard": ",".join(
                    _list_of_strings(capability_materialization.get("errors"))
                )
                or f"direct_proofs={direct_ready_count}/4",
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "capability-materialization",
                    "--json",
                ],
                "expected_impact": "Rebuilds exchange-session, derivative-contract, and stress-scenario capability receipts from versioned sources before capability routing runs.",
                "risk_level": "low",
                "when_to_stop": "materialized_capabilities_latest is fresh, ready, has 4/4 direct proof receipts, and retains zero execution or promotion authority.",
                "source": "capability_materialization",
            }
        )

    capability_blockers = _list_of_strings(
        collector_capabilities.get("structural_blockers")
        or collector_capabilities.get("paper_soak_blockers")
    )
    if collector_capabilities and (
        collector_capabilities.get("ok") is not True
        or collector_capabilities.get("paper_soak_ready") is not True
    ):
        needs.append(
            {
                "blocker": "collector_capability_routing_not_ready",
                "exact_file": "governance/health/collector_capability_control_latest.json",
                "exact_shard": ",".join(capability_blockers) or "capability_router",
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "collector-capability-control",
                    "--json",
                ],
                "expected_impact": "Revalidates the logical capability catalog, current collector mappings, shared bot subscriptions, input freshness, and zero-authority contract.",
                "risk_level": "low",
                "when_to_stop": "collector_capability_control.ok and paper_soak_ready are true with complete collector mapping and bot binding coverage.",
                "source": "collector_capability_control",
            }
        )

    if bool(hard_gates.get("ingestion_backpressure_overload", False)):
        override = _as_dict(inputs.get("backpressure_storage_control_override"))
        command = (
            ["./scripts/ops/opsctl.sh", "health-gates", "--json"]
            if bool(override.get("active", False))
            else [
                "./scripts/ops/opsctl.sh",
                "storage-pressure-clearance",
                "--apply",
                "--json",
            ]
        )
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
                "command": [
                    "./scripts/ops/opsctl.sh",
                    "paper-400-ramp",
                    "--apply",
                    "--json",
                ],
                "expected_impact": "Recomputes the paper ramp gate from the current clean global-halt artifact instead of a stale clear-blocker snapshot.",
                "risk_level": "low",
                "when_to_stop": "paper_400_ramp.gates.global_halt.ok is true and global_halt_or_clear_blocker_active is absent from blockers.",
                "source": "paper_400_ramp",
            }
        )

    platform_repair = _as_dict(
        _as_dict(health_fast.get("operational_readiness")).get("platform_repair")
    )
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


def _ready_actions_from_training_runtime(
    training_runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    contract = _as_dict(training_runtime.get("training_launch_contract"))
    if not bool(contract.get("launch_allowed", False)):
        return []
    command = _command(contract.get("recommended_retrain_command"))
    if not command:
        return []
    host_gate = _as_dict(contract.get("host_training_headroom_gate"))
    batch_size = _safe_int(contract.get("recommended_batch_size"), 0)
    profile = str(
        host_gate.get("selected_training_profile")
        or host_gate.get("governor_profile")
        or ""
    )
    batch20_mode = str(host_gate.get("batch20_execution_mode") or "")
    wave_size = _safe_int(host_gate.get("batch20_wave_size"), 0)
    batch30_mode = str(host_gate.get("batch30_execution_mode") or "")
    batch30_wave_size = _safe_int(host_gate.get("batch30_wave_size"), 0)
    quality_recovery = bool(contract.get("training_quality_recovery_canary", False))
    expected = f"Runs the guarded {batch_size}-bot retrain batch under {profile or 'the selected canary profile'}."
    if quality_recovery:
        expected += " This is a quality-recovery canary with master promotion skipped."
    if batch20_mode == "sequential_memory_guarded_waves" and wave_size > 0:
        expected += (
            f" Batch-20 is executed as sequential memory-guarded waves of {wave_size}."
        )
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
    command = (
        commands[0]
        if commands and isinstance(commands[0], list)
        else ["./scripts/ops/opsctl.sh", "uniform-hardening", "--json"]
    )
    blocker = (
        "uniform_structural_floor_not_ready"
        if structural
        else "uniform_critical_runtime_not_ready"
    )
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


def build_payload(
    project_root: Path = PROJECT_ROOT, *, fix_log_path: Path = DEFAULT_LOG_PATH
) -> dict[str, Any]:
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
    capability_materialization = load_json(
        project_root
        / "governance"
        / "collector_capabilities"
        / "materialized_capabilities_latest.json"
    )
    capability_materialization_configured = (
        project_root / "config" / "capability_materialization_v1.json"
    ).is_file()
    collector_capabilities = load_json(
        health / "collector_capability_control_latest.json"
    )
    global_halt = load_json(health / "global_halt_auto_clear_latest.json") or load_json(
        health / "global_killswitch_latest.json"
    )
    paper_ramp = load_json(health / "paper_400_ramp_latest.json")
    plumbing = load_json(health / "system_plumbing_control_latest.json")
    paper_profitability = load_json(health / "paper_profitability_control_latest.json")
    paper_runtime_profitability = load_json(
        health / "paper_runtime_profitability_controls_latest.json"
    )
    profitability_self_assessment = load_json(
        health / "profitability_self_assessment_latest.json"
    )
    market_pattern_feedback = load_json(health / "market_pattern_feedback_latest.json")
    system_role_contracts = load_json(
        project_root / "config" / DEFAULT_ROLE_CONTRACTS_PATH.name
    )
    sleeve_strategy_contracts = load_json(
        project_root / "config" / DEFAULT_SLEEVE_STRATEGY_CONTRACTS_PATH.name
    )
    master_grandmaster_evidence_policy = load_json(
        project_root / "config" / DEFAULT_MASTER_GRANDMASTER_EVIDENCE_PATH.name
    )
    paper_evidence_collection_controls = load_json(
        project_root / "config" / "paper_evidence_collection_controls_v1.json"
    )
    live_canary_readiness = load_json(
        health / "live_canary_readiness_contract_latest.json"
    )
    uniform_hardening = load_json(health / "uniform_hardening_contract_latest.json")
    uniform_hardening_enabled = (
        project_root / "config" / "production_uniform_hardening_v1.json"
    ).is_file()
    low_grade_audit = _low_grade_layer_audit(project_root)
    raw_profitability_recovery = _raw_profitability_recovery_context(
        paper_profitability=paper_profitability,
        paper_runtime_profitability=paper_runtime_profitability,
        live_canary_readiness=live_canary_readiness,
    )
    profitability_assessment_context = _profitability_self_assessment_context(
        profitability_self_assessment
    )
    market_pattern_context = _market_pattern_feedback_context(market_pattern_feedback)
    brain_boundary_context = _operator_brain_boundary_context(system_role_contracts)
    sleeve_characteristics_context = _sleeve_characteristics_context(
        sleeve_strategy_contracts,
        market_pattern_context,
        brain_boundary_context,
    )
    strategy_organization_context = _strategy_organization_context(
        sleeve_strategy_contracts,
        sleeve_characteristics_context,
    )
    master_grandmaster_success_context = _master_grandmaster_success_context(
        master_grandmaster_evidence_policy
    )
    paper_evidence_collection_context = _paper_evidence_collection_context(
        paper_evidence_collection_controls
    )
    if profitability_assessment_context.get(
        "candidate_id"
    ) and profitability_assessment_context.get("candidate_identity_consistent", False):
        raw_profitability_recovery["active"] = False
        raw_profitability_recovery["historical_context_only"] = True
        raw_profitability_recovery["superseded_by_candidate_assessment"] = True
        raw_profitability_recovery["candidate_id"] = str(
            profitability_assessment_context.get("candidate_id") or ""
        )
    soak_management = _soak_management_context(project_root, health_fast)
    needs = [
        *_need_from_governor(governor),
        *_need_from_memory(memory),
        *_need_from_storage(storage),
        *_need_from_training_runtime(training_runtime),
        *_need_from_low_grade_audit(low_grade_audit),
        *_needs_from_profitability_self_assessment(profitability_assessment_context),
        *_need_from_raw_profitability_recovery(raw_profitability_recovery),
        *_need_from_market_pattern_feedback(market_pattern_context),
        *_need_from_operator_brain_boundary(brain_boundary_context),
        *_need_from_sleeve_characteristics_context(sleeve_characteristics_context),
        *_need_from_strategy_organization_context(strategy_organization_context),
        *_need_from_master_grandmaster_success_context(
            master_grandmaster_success_context
        ),
        *_need_from_paper_evidence_collection_context(
            paper_evidence_collection_context
        ),
        *(
            _need_from_uniform_hardening(uniform_hardening)
            if uniform_hardening_enabled
            else []
        ),
        *_need_from_runtime_surfaces(
            health_fast=health_fast,
            process_watchdog=process_watchdog,
            health_gates=health_gates,
            collector_contracts=collector_contracts,
            capability_materialization=capability_materialization,
            capability_materialization_configured=capability_materialization_configured,
            collector_capabilities=collector_capabilities,
            global_halt=global_halt,
            paper_ramp=paper_ramp,
            plumbing=plumbing,
        ),
    ]
    ready_actions = _ready_actions_from_training_runtime(training_runtime)
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in needs:
        key = (
            str(item.get("blocker")),
            str(item.get("exact_file")),
            str(item.get("exact_shard")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    actionable_needs, managed_controls = _split_managed_soak_controls(
        deduped, soak_management
    )
    operator_communication = _operator_communication_packet(
        actionable_needs=actionable_needs,
        managed_controls=managed_controls,
        profitability=profitability_assessment_context,
        raw_profitability_recovery=raw_profitability_recovery,
        market_patterns=market_pattern_context,
        paper_collection=paper_evidence_collection_context,
        brain_boundary=brain_boundary_context,
        sleeve_characteristics=sleeve_characteristics_context,
        strategy_organization=strategy_organization_context,
        master_grandmaster_success=master_grandmaster_success_context,
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "needs_action" if actionable_needs else "ready",
        "what_do_you_need": actionable_needs,
        "needs": actionable_needs,
        "need_count": len(actionable_needs),
        "managed_controls": managed_controls,
        "managed_control_count": len(managed_controls),
        "next_command": actionable_needs[0]["command"] if actionable_needs else [],
        "ready_actions": ready_actions,
        "next_ready_command": ready_actions[0]["command"] if ready_actions else [],
        "operator_communication": operator_communication,
        "need_priority_ladder": operator_communication["priority_ladder"],
        "current_candidate": operator_communication["current_candidate"],
        "profitability_requirements": operator_communication[
            "exactly_needed_to_call_it_profitable"
        ],
        "market_pattern_readout": operator_communication["market_pattern_readout"],
        "sleeve_characteristics_readout": operator_communication[
            "sleeve_characteristics_readout"
        ],
        "strategy_organization_readout": operator_communication[
            "strategy_organization_readout"
        ],
        "master_grandmaster_success_readout": operator_communication[
            "master_grandmaster_success_readout"
        ],
        "paper_evidence_collection_readout": operator_communication[
            "paper_evidence_collection_readout"
        ],
        "brain_boundary_readout": operator_communication["brain_boundary_readout"],
        "direct_system_statement": operator_communication["plain_english_status"],
        "direct_profitability_answer": operator_communication[
            "direct_profitability_answer"
        ],
        "frames_of_reference": {
            "latest_writer_effectiveness": _as_dict(writer.get("drain_effectiveness")),
            "latest_benchmark_limits": _as_dict(benchmark.get("self_tuned_limits")),
            "backlog_green_gate": _as_dict(governor.get("backlog_green_gate")),
            "backlog_trend": _as_dict(governor.get("backlog_trend")),
            "stability_state": _as_dict(governor.get("stability_state")),
            "adaptive_controls": _as_dict(governor.get("adaptive_controls")),
            "runtime_pressure_source": _as_dict(
                governor.get("runtime_pressure_source")
            ),
            "host_pressure_attribution": _as_dict(
                runtime.get("host_pressure_attribution")
            ),
            "memory_pressure_intelligence": {
                "classification": _as_dict(memory.get("classification")),
                "trend": _as_dict(memory.get("trend")),
                "reopen_gate": _as_dict(memory.get("reopen_gate")),
                "multitasking_headroom": _as_dict(memory.get("multitasking_headroom")),
                "observer_overhead": _as_dict(memory.get("observer_overhead")),
            },
            "training_runtime_control": {
                "overall_status": str(training_runtime.get("overall_status") or ""),
                "launch_contract": _as_dict(
                    training_runtime.get("training_launch_contract")
                ),
                "host_training_headroom_gate": _as_dict(
                    _as_dict(training_runtime.get("training_launch_contract")).get(
                        "host_training_headroom_gate"
                    )
                ),
                "bot_needs": _as_dict(training_runtime.get("bot_needs")),
            },
            "health_fast": {
                "overall_status": str(
                    health_fast.get("overall_status") or health_fast.get("status") or ""
                ),
                "process_watchdog": _as_dict(health_fast.get("process_watchdog")),
                "operational_readiness": _as_dict(
                    health_fast.get("operational_readiness")
                ),
            },
            "health_gates": {
                "hard_gates": _as_dict(health_gates.get("hard_gates")),
                "inputs": _as_dict(health_gates.get("inputs")),
            },
            "collector_contracts": {
                "required_failures": _as_list(
                    collector_contracts.get("required_failures")
                ),
                "soft_failures": _as_list(collector_contracts.get("soft_failures")),
            },
            "capability_materialization": {
                "overall_status": str(
                    capability_materialization.get("overall_status") or ""
                ),
                "live_promotion_ready": bool(
                    capability_materialization.get("live_promotion_ready", False)
                ),
                "direct_proof_count": sum(
                    1
                    for row in _as_list(capability_materialization.get("capabilities"))
                    if isinstance(row, dict)
                    and row.get("usable") is True
                    and str(row.get("proof_semantics") or "") == "direct"
                    and bool(str(row.get("proof_receipt_sha256") or ""))
                ),
                "errors": _as_list(capability_materialization.get("errors")),
            },
            "collector_capabilities": {
                "overall_status": str(
                    collector_capabilities.get("overall_status") or ""
                ),
                "paper_soak_ready": bool(
                    collector_capabilities.get("paper_soak_ready", False)
                ),
                "live_promotion_ready": bool(
                    collector_capabilities.get("live_promotion_ready", False)
                ),
                "summary": _as_dict(collector_capabilities.get("summary")),
                "coverage_debt": _as_dict(collector_capabilities.get("coverage_debt")),
                "structural_blockers": _as_list(
                    collector_capabilities.get("structural_blockers")
                ),
                "paper_soak_blockers": _as_list(
                    collector_capabilities.get("paper_soak_blockers")
                ),
            },
            "global_halt": {
                "halt": bool(
                    global_halt.get("halt", False)
                    or global_halt.get("global_halt", False)
                ),
                "halt_latched": bool(global_halt.get("halt_latched", False)),
                "halt_required": bool(
                    global_halt.get("halt_required", False)
                    or global_halt.get("would_rehalt", False)
                ),
                "clear_blockers": _as_list(global_halt.get("clear_blockers")),
                "halt_posture": str(global_halt.get("halt_posture") or ""),
            },
            "paper_400_ramp": {
                "stage": str(paper_ramp.get("stage") or ""),
                "blockers": _as_list(paper_ramp.get("blockers")),
                "global_halt_gate": _as_dict(
                    _as_dict(paper_ramp.get("gates")).get("global_halt")
                ),
            },
            "system_plumbing_control": {
                "overall_status": str(plumbing.get("overall_status") or ""),
                "plumbing_score": plumbing.get("plumbing_score"),
            },
            "low_grade_layer_audit": low_grade_audit,
            "raw_profitability_recovery": raw_profitability_recovery,
            "profitability_self_assessment": profitability_assessment_context,
            "market_pattern_feedback": market_pattern_context,
            "sleeve_strategy_characteristics": sleeve_characteristics_context,
            "strategy_organization": strategy_organization_context,
            "paper_evidence_collection_controls": paper_evidence_collection_context,
            "operator_brain_boundary": brain_boundary_context,
            "master_grandmaster_success_needs": master_grandmaster_success_context,
            "uniform_hardening_contract": {
                "overall_status": str(uniform_hardening.get("overall_status") or ""),
                "uniform_floor_ready": bool(
                    uniform_hardening.get("uniform_floor_ready", False)
                ),
                "critical_runtime_ready": bool(
                    uniform_hardening.get("critical_runtime_ready", False)
                ),
                "domain_statuses": _as_dict(uniform_hardening.get("domain_statuses")),
                "structural_blockers": _as_list(
                    uniform_hardening.get("structural_blockers")
                ),
                "critical_runtime_blockers": _as_list(
                    uniform_hardening.get("critical_runtime_blockers")
                ),
                "evidence_debt_domains": _as_list(
                    uniform_hardening.get("evidence_debt_domains")
                ),
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
            "prefer_candidate_bound_profitability_assessment_over_historical_ledger_burn_down": True,
            "include_uniform_hardening_floor_need": True,
            "split_managed_soak_controls_from_actionable_needs": True,
            "include_plain_english_operator_communication": True,
            "include_market_pattern_feedback_needs": True,
            "include_paper_evidence_collection_control_status": True,
            "include_operator_brain_boundary": True,
            "include_strategy_organization": True,
            "include_master_grandmaster_success_needs": True,
            "system_needs_markdown": str(DEFAULT_MARKDOWN_PATH),
            "fixes_logged_to": str(fix_log_path),
            "protected_volumes": ["/Volumes/VIDEO"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explain exactly what the system needs next and preserve fix frames of reference."
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--no-md", action="store_true")
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
    if not args.no_md:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_operator_needs_markdown(payload), encoding="utf-8"
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"system_needs_intelligence status={payload['overall_status']} needs={len(payload['what_do_you_need'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
