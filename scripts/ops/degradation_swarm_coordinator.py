#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import (
        iso_now,
        load_json,
        ordered_unique,
        parse_iso_utc,
        run_bounded_process_group,
        write_payload,
    )
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import (
        iso_now,
        load_json,
        ordered_unique,
        parse_iso_utc,
        run_bounded_process_group,
        write_payload,
    )

from core.operating_contracts import REQUIRED_OPERATING_CONTRACT_FIELDS

DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "degradation_swarm_coordinator_latest.json"
)
DEFAULT_CONTEXT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "degradation_swarm_context_latest.json"
)
DEFAULT_STATE_PATH = (
    PROJECT_ROOT / "governance" / "health" / "degradation_swarm_state.json"
)
STORAGE_BACKPRESSURE_OUT_PATH = (
    PROJECT_ROOT
    / "governance"
    / "health"
    / "storage_backpressure_autopilot_latest.json"
)
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "degradation_swarm.lock"
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "logs" / "degradation_swarm_repair_ledger.jsonl"
SCHEMA_VERSION = 1
OWNER_OUTPUT_RELEASE_SIGNAL = "owner_artifact_operating_contract_complete_true_and_release_conditions_met_or_named"

SAFE_EXEC_ENV = {
    "MARKET_DATA_ONLY": "1",
    "ALLOW_ORDER_EXECUTION": "0",
    "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
    "EXECUTION_LANE_LIVE_ENABLED": "0",
    "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR": "0",
    "INLINE_PAPER_EXECUTION_ENABLED": "0",
    "DEGRADATION_SWARM_SAFE_REPAIR_EXECUTOR": "1",
}

READ_ONLY_OPSCTL_SUBCOMMANDS = {
    "artifact-freshness-slo",
    "bot-organization",
    "bot-profitability-scalability",
    "canonical-representation-audit",
    "control-surface-ownership",
    "decision-intelligence",
    "infrabot-adaptive-governor",
    "infrabot-gap-roster",
    "local-storage-reserve-guard",
    "master-grandmaster-evidence",
    "master-infra-supervisor",
    "operator-cockpit",
    "paper-performance",
    "paper-profitability-control",
    "paper-truth",
    "promotion-quality-gate",
    "provider-mesh",
    "roster-resilience",
    "runtime-gate-dashboard",
    "runtime-paper-regression-guard",
    "sleeve-profitability-dashboard",
    "sleeve-scalability-selector",
    "source-verification",
    "storage-prune-standby",
    "storage-retention-unison",
    "training-quality",
}

APPLY_SAFE_OPSCTL_SUBCOMMANDS = {
    "backlog-pcore-accelerator",
    "daily-verify-remediation",
    "external-backlog-drain",
    "external-backlog-retry-bot",
    "infrastructure-autofix",
    "local-storage-reserve-guard",
    "paper-truth-refresh",
    "raw-training-compaction",
    "source-verification-refresh",
    "storage-backpressure-autopilot",
    "storage-pressure-clearance",
    "storage-prune-standby",
    "system-drift-autopilot",
    "writer-cycle-coordinator",
}

FORBIDDEN_COMMAND_PARTS = {
    "start-live",
    "live-canary-preflight",
    "live-canary-dress-rehearsal",
    "--allow-orders",
    "--confirm-all",
    "--issue-attestation",
    "--issue-allowlist",
}

ATTENTION_TO_LANE = {
    "paper_execution_safety_guard_active": "paper_execution",
    "external_backlog_drain_recommended": "storage_reserve",
    "external_backlog_drain_writer_busy": "storage_reserve",
    "external_backlog_retry_bot_followups": "storage_reserve",
    "retrain_artifact_freshness_not_ok": "training_promotion",
    "promotion_not_ready": "training_promotion",
    "training_quality_control_blocked": "training_promotion",
    "bot_quality_autopilot_evidence_pending": "profitability_evidence",
    "infrastructure_autofix_bot_needs_work": "ops_self_audit",
    "infrastructure_autofix_bot_blocked": "ops_self_audit",
    "coordination_state_control_blocked": "ops_self_audit",
    "coordination_state_control_needs_work": "ops_self_audit",
    "roster_resilience_planner_needs_work": "bot_organization",
    "source_verification_context_debt": "source_verification",
    "source_verification_decision_critical_blocked": "source_verification",
    "master_grandmaster_evidence_v2_not_ok": "master_grandmaster",
    "bot_profitability_scalability_control_not_ok": "profitability_evidence",
    "sleeve_scalability_selector_not_ok": "sleeve_selection",
}

DOMAIN_TO_LANE = {
    "bot_roster": "bot_organization",
    "execution_safety": "paper_execution",
    "market_context": "source_verification",
    "ops_self_healing": "ops_self_audit",
    "storage_backlog": "storage_reserve",
    "training_promotion": "training_promotion",
}

PHASE_RANK = {
    "observe": 10,
    "stabilize": 20,
    "repair": 30,
    "verify": 40,
}

SWARM_SECTION_DEFINITIONS = {
    "trading_brain": {
        "label": "Trading Brain",
        "mission": "Decide whether trade, sleeve, model, promotion, and profitability evidence is strong enough to earn authority.",
        "primary_question": "Can this trading surface prove post-cost, persistent, attributable edge without relying on unsafe execution authority?",
        "allowed_authority": [
            "observe_market_and_paper_evidence",
            "refresh_readiness_and_profitability_reports",
            "verify_execution_and_promotion_gates",
        ],
        "blocked_authority": [
            "live_order_submission",
            "paper_order_submission_while_guarded",
            "model_promotion_without_gradeable_evidence",
        ],
    },
    "ops_brain": {
        "label": "Ops Brain",
        "mission": "Keep storage, process ownership, bot organization, and repair lanes healthy so data collection can continue cleanly.",
        "primary_question": "Can the platform contain infrastructure debt without hiding it or interrupting the collection hot path?",
        "allowed_authority": [
            "bounded_storage_and_writer_repairs",
            "runtime_health_refresh",
            "bot_roster_and_tripwire_verification",
        ],
        "blocked_authority": [
            "changing_trading_thresholds",
            "starting_competing_sqlite_writers",
            "granting_execution_or_promotion_authority",
        ],
    },
}

PHASE_DEFINITIONS = {
    "observe": {
        "purpose": "Refresh truth and evidence surfaces before repairs are chosen.",
        "entry_condition": "contained lane has stale, missing, or advisory evidence",
        "exit_signal": "owner surface is fresh enough to identify the exact blocker",
        "authority": "read_only_or_refresh_only",
    },
    "stabilize": {
        "purpose": "Reduce immediate operational pressure before heavier repair work.",
        "entry_condition": "writer, queue, lease, or runtime pressure can interfere with repair order",
        "exit_signal": "handoff or pressure signal is bounded and safe to continue",
        "authority": "bounded_infrastructure_repair_only",
    },
    "repair": {
        "purpose": "Run the owner command that can actually clear contained debt.",
        "entry_condition": "lane has a specific safe remediation command and no uncontained blast radius",
        "exit_signal": "repair command reports ready or a smaller explicit follow-up",
        "authority": "safe_apply_allowlist_only",
    },
    "verify": {
        "purpose": "Confirm the lane is contained, ready, or still blocked for a named reason.",
        "entry_condition": "truth or repair step completed or prior state says it already ran",
        "exit_signal": "release condition is met or remaining blocker is named",
        "authority": "read_only_or_gate_verification",
    },
}

LANE_DEFINITIONS = {
    "collection_hot_path": {
        "section": "ops_brain",
        "domain": "collection_runtime",
        "canonical_owner": "operator_cockpit",
        "mission": "Keep market/account evidence collection moving while degradation is isolated elsewhere.",
        "contained_definition": "Collection is healthy enough that advisory debt does not stop fresh evidence accrual.",
        "uncontained_definition": "Collection is halted, stale, or unsafe, so repair work cannot rely on fresh evidence.",
        "primary_release_signal": "collection_surface_ready_true",
        "escalates_when": [
            "collection_surface_ready_false",
            "runtime_hot_path_blocked_true",
            "global_halt_active",
        ],
        "safe_repair_scope": [
            "observe_collection_health",
            "refresh_runtime_readiness",
        ],
    },
    "paper_execution": {
        "section": "trading_brain",
        "domain": "execution_safety",
        "canonical_owner": "runtime_paper_regression_guard",
        "mission": "Prove paper execution is gradeable, post-cost, attributable, and guarded before order authority returns.",
        "contained_definition": "Paper order submission is blocked while collection and evidence refreshes continue.",
        "uncontained_definition": "Paper or live order authority is open while post-cost execution evidence is missing or degraded.",
        "primary_release_signal": "paper_execution_safety_guard_clears_after_profitability_and_execution_evidence_are_gradeable",
        "escalates_when": [
            "paper_order_submission_authority_true_while_guard_active",
            "broker_or_fill_truth_missing",
            "execution_ledger_unattributed",
        ],
        "safe_repair_scope": [
            "refresh_paper_truth",
            "verify_runtime_paper_contract",
            "explain_post_cost_execution_blockers",
        ],
    },
    "storage_reserve": {
        "section": "ops_brain",
        "domain": "storage_backlog",
        "canonical_owner": "storage_tier_policy",
        "mission": "Keep storage pressure, writer handoff, and backlog drain bounded without interrupting hot-path collection.",
        "contained_definition": "Queues and writer pressure are bounded and repairs are serialized through single-writer commands.",
        "uncontained_definition": "Storage pressure blocks collection, starts competing writers, or crosses hard queue/reserve limits.",
        "primary_release_signal": "storage_pressure_and_backlog_drain_report_ready",
        "escalates_when": [
            "hard_queue_watermark_breach",
            "writer_shedding_active",
            "collection_blocked_by_storage",
        ],
        "safe_repair_scope": [
            "single_writer_handoff",
            "bounded_backlog_drain",
            "quick_backpressure_verification",
        ],
    },
    "bot_organization": {
        "section": "ops_brain",
        "domain": "bot_roster",
        "canonical_owner": "bot_organization_control",
        "mission": "Keep bot roles, sleeve ownership, review debt, and tripwires explicit enough that repair ownership is not ambiguous.",
        "contained_definition": "Blocking tripwires are zero and any remaining review debt is visible and assigned.",
        "uncontained_definition": "Blocking tripwires, missing owners, or ambiguous bot authority can hide a degradation source.",
        "primary_release_signal": "bot_organization_zero_blocking_tripwires",
        "escalates_when": [
            "blocking_tripwire_count_positive",
            "lane_without_owner",
            "repair_gap_without_assigned_infrabot",
        ],
        "safe_repair_scope": [
            "refresh_bot_roster",
            "verify_infrabot_gap_roster",
            "surface_operator_review_items",
        ],
    },
    "profitability_evidence": {
        "section": "trading_brain",
        "domain": "profitability_proof",
        "canonical_owner": "bot_profitability_scalability_control",
        "mission": "Separate real edge from ungradeable paper losses by requiring post-cost persistence, attribution, and capacity evidence.",
        "contained_definition": "Profitability claims and scaling are blocked while evidence collection and diagnosis continue.",
        "uncontained_definition": "The system claims profitability, scales, or grants execution authority without gradeable post-cost evidence.",
        "primary_release_signal": "profitability_evidence_gradeable_post_cost_and_persistent",
        "escalates_when": [
            "profitability_claim_ready_true_without_gradeable_evidence",
            "negative_paper_balance_unexplained_by_attribution",
            "capacity_or_cost_model_missing",
        ],
        "safe_repair_scope": [
            "refresh_paper_performance",
            "refresh_bot_profitability_scalability",
            "verify_profitability_gate",
        ],
    },
    "sleeve_selection": {
        "section": "trading_brain",
        "domain": "sleeve_authority",
        "canonical_owner": "sleeve_scalability_selector",
        "mission": "Make each sleeve earn authority from sleeve-specific edge, capacity, drawdown, and execution evidence.",
        "contained_definition": "Sleeve recommendation authority is blocked until each sleeve has earned evidence.",
        "uncontained_definition": "A sleeve is promoted or widened without sleeve-specific evidence and risk constraints.",
        "primary_release_signal": "sleeve_selector_has_eligible_sleeves_with_gradeable_evidence",
        "escalates_when": [
            "recommendation_ready_true_with_evidence_debt",
            "sleeve_without_capacity_or_drawdown_contract",
            "cross_sleeve_correlation_unknown",
        ],
        "safe_repair_scope": [
            "refresh_sleeve_selector",
            "refresh_sleeve_profitability_dashboard",
            "verify_sleeve_evidence_debt",
        ],
    },
    "master_grandmaster": {
        "section": "trading_brain",
        "domain": "meta_coordination",
        "canonical_owner": "master_grandmaster_evidence_v2",
        "mission": "Let master and grandmaster bots coordinate only after structural, evidence, and promotion blockers are explicit.",
        "contained_definition": "Coordination or promotion is held while structural grade, runtime capacity, or evidence blockers remain visible.",
        "uncontained_definition": "Master/grandmaster authority can promote, scale, or route orders while blockers are unresolved.",
        "primary_release_signal": "master_grandmaster_structural_grade_a_and_promotion_blockers_explicit_or_clear",
        "escalates_when": [
            "automatic_live_promotion_allowed_true_with_blockers",
            "runtime_capacity_not_ready",
            "required_master_source_stale",
        ],
        "safe_repair_scope": [
            "refresh_master_grandmaster_evidence",
            "verify_promotion_quality_gate",
        ],
    },
    "source_verification": {
        "section": "trading_brain",
        "domain": "market_context",
        "canonical_owner": "source_verification_report",
        "mission": "Keep market-context claims honest by distinguishing decision-critical blockers from optional context debt.",
        "contained_definition": "Decision-critical sources are usable and context debt blocks claims or promotion, not collection.",
        "uncontained_definition": "Decision-critical sources are unavailable, stale, or used for trading claims without verification.",
        "primary_release_signal": "source_verification_all_sources_verified_or_context_debt_empty",
        "escalates_when": [
            "decision_critical_source_blocked",
            "market_context_claim_uses_unverified_source",
            "source_context_debt_becomes_execution_dependency",
        ],
        "safe_repair_scope": [
            "refresh_source_verification",
            "verify_provider_mesh",
            "quarantine_optional_context_debt",
        ],
    },
    "training_promotion": {
        "section": "trading_brain",
        "domain": "model_training_and_promotion",
        "canonical_owner": "training_quality_control",
        "mission": "Keep training refresh and model promotion frozen until artifact freshness, lineage, and quality gates are clean.",
        "contained_definition": "Training/promotion is blocked while evidence collection continues and stale artifacts are named.",
        "uncontained_definition": "A model can retrain, promote, or replace production logic while inputs or lineage are degraded.",
        "primary_release_signal": "training_quality_and_promotion_quality_gates_ready",
        "escalates_when": [
            "promotion_authority_true_with_training_debt",
            "retrain_artifact_freshness_not_ok",
            "model_lineage_or_source_gate_missing",
        ],
        "safe_repair_scope": [
            "refresh_training_quality",
            "verify_retrain_artifact_freshness",
            "verify_promotion_gate",
        ],
    },
    "ops_self_audit": {
        "section": "ops_brain",
        "domain": "ops_self_healing",
        "canonical_owner": "master_infrastructure_supervisor",
        "mission": "Keep the platform honest about stale governance, repair debt, command hygiene, and self-healing coverage.",
        "contained_definition": "Ops debt is advisory or bounded and does not block collection, storage, or execution safety gates.",
        "uncontained_definition": "Ops debt masks runtime truth, prevents troubleshooting, or blocks the collection hot path.",
        "primary_release_signal": "master_infrastructure_supervisor_ready_or_contained_without_hot_path_block",
        "escalates_when": [
            "truth_layer_not_ready",
            "command_surface_not_clean",
            "self_healing_infrabot_missing_for_active_lane",
        ],
        "safe_repair_scope": [
            "bounded_infrastructure_autofix",
            "system_drift_repair",
            "master_infra_verification",
        ],
    },
}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _status(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _opsctl(*args: str) -> list[str]:
    return ["./scripts/ops/opsctl.sh", *args]


def _py_script(script: str, *args: str) -> list[str]:
    return ["python", script, *args]


def _command_hash(command: list[str]) -> str:
    data = json.dumps(command, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:12]


def _incident_id(model: dict[str, Any]) -> str:
    compact = {
        "contained_degradation_lanes": sorted(
            str(item)
            for item in _as_list(model.get("contained_degradation_lanes"))
            if str(item).strip()
        ),
        "uncontained_lanes": sorted(
            str(item)
            for item in _as_list(model.get("uncontained_lanes"))
            if str(item).strip()
        ),
        "runtime_attention": sorted(
            str(item)
            for item in _as_list(model.get("runtime_attention"))
            if str(item).strip()
        ),
    }
    data = json.dumps(compact, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _opsctl_subcommand(command: list[str]) -> str:
    for index, part in enumerate(command):
        if Path(str(part)).name == "opsctl.sh" and index + 1 < len(command):
            return str(command[index + 1]).strip()
    return ""


def _normalize_command(project_root: Path, command: list[str]) -> list[str]:
    clean = [str(part).strip() for part in command if str(part).strip()]
    if not clean:
        return []
    first = clean[0]
    if first == "python":
        normalized = [str(Path(sys.executable))]
        for index, part in enumerate(clean[1:], start=1):
            path = Path(part)
            if index == 1 and not path.is_absolute() and str(path).endswith(".py"):
                normalized.append(str((project_root / path).resolve()))
            else:
                normalized.append(part)
        return normalized
    path = Path(first).expanduser()
    if first == "./scripts/ops/opsctl.sh":
        clean[0] = str(project_root / "scripts" / "ops" / "opsctl.sh")
    elif not path.is_absolute() and first.endswith("opsctl.sh"):
        clean[0] = str((project_root / path).resolve())
    return clean


def _command_forbidden(command: list[str]) -> bool:
    lowered = {str(part).strip().lower() for part in command}
    return bool(lowered & FORBIDDEN_COMMAND_PARTS)


def _command_safe(command: list[str]) -> bool:
    if not command or _command_forbidden(command):
        return False
    subcommand = _opsctl_subcommand(command)
    if subcommand:
        if subcommand == "storage-switch-external":
            return "--dry-run" in command and "--apply" not in command
        if "--apply" in command:
            return subcommand in APPLY_SAFE_OPSCTL_SUBCOMMANDS
        return subcommand in READ_ONLY_OPSCTL_SUBCOMMANDS.union(
            APPLY_SAFE_OPSCTL_SUBCOMMANDS
        )
    if Path(command[0]).name.startswith("python") and len(command) > 1:
        return Path(command[1]).name in {"retrain_artifact_freshness_guard.py"}
    return False


def _lane_definition(lane: str) -> dict[str, Any]:
    definition = _as_dict(LANE_DEFINITIONS.get(str(lane or "").strip()))
    if definition:
        return dict(definition)
    return {
        "section": "ops_brain",
        "domain": "unclassified",
        "canonical_owner": "operator_review",
        "mission": "Classify this lane before it can participate in automated repair.",
        "contained_definition": "The lane is explicitly assigned to an owner with a bounded blast radius.",
        "uncontained_definition": "The lane has no known owner, authority boundary, or release signal.",
        "primary_release_signal": "owner_mapping_added_or_attention_clears",
        "escalates_when": ["owner_mapping_missing"],
        "safe_repair_scope": ["operator_review_only"],
    }


def _section_definition(section: str) -> dict[str, Any]:
    definition = _as_dict(SWARM_SECTION_DEFINITIONS.get(str(section or "").strip()))
    if definition:
        return dict(definition)
    return {
        "label": "Unclassified",
        "mission": "Hold unknown work outside authority-bearing repair until ownership is defined.",
        "primary_question": "What owns this signal and what authority can it safely use?",
        "allowed_authority": ["observe_only"],
        "blocked_authority": ["automated_repair_without_owner_mapping"],
    }


def _phase_definition(phase: str) -> dict[str, Any]:
    definition = _as_dict(PHASE_DEFINITIONS.get(str(phase or "").strip()))
    if definition:
        return dict(definition)
    return {
        "purpose": "Unclassified phase; do not execute without explicit review.",
        "entry_condition": "unknown",
        "exit_signal": "phase_definition_added",
        "authority": "observe_only",
    }


def _phase_order_for_lane(assignments: list[dict[str, Any]], lane: str) -> list[str]:
    return ordered_unique(
        [
            str(row.get("phase") or "")
            for row in assignments
            if isinstance(row, dict)
            and str(row.get("lane") or "") == lane
            and str(row.get("phase") or "").strip()
        ]
    )


def _worst_runtime_tier(rows: list[dict[str, Any]]) -> str:
    rank = {"critical": 0, "degraded": 1, "watch": 2, "advisory": 3}
    tiers = [
        str(row.get("tier") or "").strip().lower()
        for row in rows
        if isinstance(row, dict) and str(row.get("tier") or "").strip()
    ]
    if not tiers:
        return ""
    return sorted(tiers, key=lambda item: rank.get(item, 99))[0]


def _lane_operating_model(
    model: dict[str, Any], lane: str, assignments: list[dict[str, Any]]
) -> dict[str, Any]:
    definition = _lane_definition(lane)
    lane_row = _as_dict(_as_dict(model.get("cockpit_lanes")).get(lane))
    runtime_rows = [
        row
        for row in _as_list(
            _as_dict(model.get("runtime_containment_rows_by_lane")).get(lane)
        )
        if isinstance(row, dict)
    ]
    runtime_attentions = ordered_unique(
        [
            str(row.get("attention") or "")
            for row in runtime_rows
            if str(row.get("attention") or "").strip()
        ]
    )
    release_signals = ordered_unique(
        [
            str(definition.get("primary_release_signal") or ""),
            *[
                str(row.get("release_condition") or "")
                for row in runtime_rows
                if str(row.get("release_condition") or "").strip()
            ],
        ]
    )
    lane_assignments = [
        row
        for row in assignments
        if isinstance(row, dict) and str(row.get("lane") or "") == lane
    ]
    safe_assignment_count = sum(
        1 for row in lane_assignments if bool(row.get("safe_execute_allowed", False))
    )
    blocked_assignment_count = len(lane_assignments) - safe_assignment_count
    current_contained = bool(lane_row.get("contained", False))
    runtime_tier = _worst_runtime_tier(runtime_rows)
    lane_status = _status(lane_row.get("status"))
    if lane in set(str(item) for item in _as_list(model.get("uncontained_lanes"))):
        current_level = "owner_unresolved"
    elif runtime_tier:
        current_level = runtime_tier
    elif lane_status in {"ready", "clear", "ok", "stable"}:
        current_level = "clear"
    elif current_contained:
        current_level = "contained_debt"
    else:
        current_level = "needs_owner_classification"

    section = str(definition.get("section") or "ops_brain")
    return {
        "lane": lane,
        "section": section,
        "section_label": _section_definition(section).get("label", section),
        "domain": definition.get("domain"),
        "canonical_owner": definition.get("canonical_owner"),
        "mission": definition.get("mission"),
        "contained_definition": definition.get("contained_definition"),
        "uncontained_definition": definition.get("uncontained_definition"),
        "current_status": lane_row.get("status", "unknown"),
        "current_contained": current_contained,
        "current_level": current_level,
        "trade_impact": lane_row.get("trade_impact", ""),
        "runtime_attentions": runtime_attentions,
        "release_signals": release_signals,
        "owner_output_contract_required_fields": list(
            REQUIRED_OPERATING_CONTRACT_FIELDS
        ),
        "owner_output_release_signal": OWNER_OUTPUT_RELEASE_SIGNAL,
        "escalates_when": _as_list(definition.get("escalates_when")),
        "safe_repair_scope": _as_list(definition.get("safe_repair_scope")),
        "phase_order": _phase_order_for_lane(assignments, lane),
        "assignment_count": len(lane_assignments),
        "safe_executable_assignment_count": safe_assignment_count,
        "blocked_assignment_count": blocked_assignment_count,
    }


def _build_swarm_operating_model(
    model: dict[str, Any], assignments: list[dict[str, Any]]
) -> dict[str, Any]:
    active_lanes = ordered_unique(
        [
            *[
                str(item)
                for item in _as_list(model.get("contained_degradation_lanes"))
                if str(item).strip()
            ],
            *[
                str(item)
                for item in _as_list(model.get("uncontained_lanes"))
                if str(item).strip()
            ],
            *[
                str(row.get("lane") or "")
                for row in assignments
                if isinstance(row, dict) and str(row.get("lane") or "").strip()
            ],
        ]
    )
    lane_models = {
        lane: _lane_operating_model(model, lane, assignments) for lane in active_lanes
    }
    sections: dict[str, dict[str, Any]] = {}
    for section, definition in SWARM_SECTION_DEFINITIONS.items():
        sections[section] = {
            **definition,
            "lanes": [],
            "assignment_count": 0,
            "safe_executable_assignment_count": 0,
            "blocked_assignment_count": 0,
            "phase_counts": {},
        }
    for lane, lane_model in lane_models.items():
        section = str(lane_model.get("section") or "ops_brain")
        sections.setdefault(
            section,
            {
                **_section_definition(section),
                "lanes": [],
                "assignment_count": 0,
                "safe_executable_assignment_count": 0,
                "blocked_assignment_count": 0,
                "phase_counts": {},
            },
        )
        section_row = sections[section]
        section_row["lanes"].append(lane)
        section_row["assignment_count"] = _safe_int(
            section_row.get("assignment_count"), 0
        ) + _safe_int(lane_model.get("assignment_count"), 0)
        section_row["safe_executable_assignment_count"] = _safe_int(
            section_row.get("safe_executable_assignment_count"), 0
        ) + _safe_int(lane_model.get("safe_executable_assignment_count"), 0)
        section_row["blocked_assignment_count"] = _safe_int(
            section_row.get("blocked_assignment_count"), 0
        ) + _safe_int(lane_model.get("blocked_assignment_count"), 0)
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        lane = str(assignment.get("lane") or "")
        section = str(_as_dict(lane_models.get(lane)).get("section") or "ops_brain")
        phase = str(assignment.get("phase") or "unknown")
        phase_counts = _as_dict(
            sections.setdefault(section, {}).setdefault("phase_counts", {})
        )
        phase_counts[phase] = _safe_int(phase_counts.get(phase), 0) + 1
        sections[section]["phase_counts"] = phase_counts

    refinement_backlog = _build_refinement_backlog(model, assignments, lane_models)
    definition_maturity = {
        "section_definition_count": len(SWARM_SECTION_DEFINITIONS),
        "lane_definition_count": len(LANE_DEFINITIONS),
        "phase_definition_count": len(PHASE_DEFINITIONS),
        "required_owner_output_field_count": len(REQUIRED_OPERATING_CONTRACT_FIELDS),
        "active_lane_count": len(lane_models),
        "active_lanes_with_release_signal_count": sum(
            1 for row in lane_models.values() if _as_list(row.get("release_signals"))
        ),
        "active_lanes_with_owner_contract_count": sum(
            1
            for row in lane_models.values()
            if _as_list(row.get("owner_output_contract_required_fields"))
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "sections": sections,
        "phase_definitions": PHASE_DEFINITIONS,
        "lane_definitions": {
            lane: _lane_definition(lane) for lane in sorted(LANE_DEFINITIONS)
        },
        "active_lanes": lane_models,
        "active_sections": ordered_unique(
            [
                str(row.get("section") or "")
                for row in lane_models.values()
                if str(row.get("section") or "").strip()
            ]
        ),
        "active_phase_order": ordered_unique(
            [
                str(row.get("phase") or "")
                for row in assignments
                if isinstance(row, dict) and str(row.get("phase") or "").strip()
            ]
        ),
        "refinement_backlog": refinement_backlog,
        "definition_maturity": definition_maturity,
        "required_owner_output_fields": list(REQUIRED_OPERATING_CONTRACT_FIELDS),
        "swarm_protocol": {
            "step_order": [
                "classify",
                "contain",
                "observe",
                "stabilize",
                "repair",
                "verify",
            ],
            "classification_required_before_execution": True,
            "state_memory_required": True,
            "owner_release_condition_required": True,
            "owner_operating_contract_required": True,
            "owner_output_release_signal": OWNER_OUTPUT_RELEASE_SIGNAL,
            "live_or_promotion_authority_allowed": False,
        },
    }


def _build_refinement_backlog(
    model: dict[str, Any],
    assignments: list[dict[str, Any]],
    lane_models: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    backlog: list[dict[str, Any]] = []
    for lane, lane_model in lane_models.items():
        if _safe_int(lane_model.get("blocked_assignment_count"), 0) > 0:
            backlog.append(
                {
                    "lane": lane,
                    "section": lane_model.get("section"),
                    "need": "blocked_swarm_assignment_needs_owner_safe_command_or_explicit_manual_review",
                    "reason": "one_or_more_assignments_failed_safe_execute_allowlist_or_authority_boundary",
                }
            )
        if str(lane_model.get("current_level") or "") in {
            "owner_unresolved",
            "needs_owner_classification",
        }:
            backlog.append(
                {
                    "lane": lane,
                    "section": lane_model.get("section"),
                    "need": "lane_contract_needs_clearer_contained_vs_owner_unresolved_definition",
                    "reason": "lane_is_globally_safe_but_owner_surface_is_not_structurally_ready",
                }
            )
    for row in _as_dict(model.get("runtime_remediation_by_attention")).values():
        if not isinstance(row, dict):
            continue
        command = _as_list(row.get("command"))
        if str(row.get("owner") or "") == "operator_review" or not command:
            backlog.append(
                {
                    "lane": ATTENTION_TO_LANE.get(
                        str(row.get("attention") or ""), "operator_review"
                    ),
                    "section": "ops_brain",
                    "need": "runtime_attention_needs_concrete_owner_command",
                    "reason": str(row.get("attention") or "unmapped_attention"),
                }
            )
    if _as_list(model.get("source_context_debt")):
        backlog.append(
            {
                "lane": "source_verification",
                "section": "trading_brain",
                "need": "context_sources_need_release_criteria_and_confidence_budget",
                "reason": ",".join(
                    str(item) for item in _as_list(model.get("source_context_debt"))
                ),
            }
        )
    return backlog


def _playbook_task(
    *,
    lane: str,
    infrabot_id: str,
    title: str,
    command: list[str],
    phase: str,
    priority: int,
    repair_intent: str,
    stop_when: str,
    auto_execute: bool = True,
    resource_lock: str = "none",
    max_attempts: int = 2,
    allowed_when_uncontained: bool = False,
    requires_collection_hot_path: bool = True,
) -> dict[str, Any]:
    return {
        "lane": lane,
        "infrabot_id": infrabot_id,
        "title": title,
        "command": list(command),
        "phase": phase,
        "priority": int(priority),
        "repair_intent": repair_intent,
        "stop_when": stop_when,
        "auto_execute_requested": bool(auto_execute),
        "resource_lock": resource_lock,
        "max_attempts_per_incident": max(int(max_attempts), 1),
        "allowed_when_uncontained": bool(allowed_when_uncontained),
        "requires_collection_hot_path": bool(requires_collection_hot_path),
        "authority_boundary": "repair_refresh_or_observe_only_no_live_execution_no_order_submission_no_promotion",
        "required_owner_output_fields": list(REQUIRED_OPERATING_CONTRACT_FIELDS),
        "owner_output_release_signal": OWNER_OUTPUT_RELEASE_SIGNAL,
    }


def _lane_playbooks() -> dict[str, list[dict[str, Any]]]:
    return {
        "paper_execution": [
            _playbook_task(
                lane="paper_execution",
                infrabot_id="paper_execution_truth_infrabot",
                title="Paper Execution Truth Infrabot",
                command=_opsctl("paper-truth", "--json"),
                phase="observe",
                priority=10,
                repair_intent="refresh paper truth and identify missing broker, ledger, or attribution evidence",
                stop_when="paper-execution-truth reports ready or watch with no failed checks",
            ),
            _playbook_task(
                lane="paper_execution",
                infrabot_id="runtime_paper_contract_infrabot",
                title="Runtime Paper Contract Infrabot",
                command=_opsctl("runtime-paper-regression-guard", "--json"),
                phase="verify",
                priority=42,
                repair_intent="verify the paper runtime remains locked while evidence is ungradeable",
                stop_when="runtime-paper-regression-guard reports ready",
            ),
            _playbook_task(
                lane="paper_execution",
                infrabot_id="paper_profitability_gate_infrabot",
                title="Paper Profitability Gate Infrabot",
                command=_opsctl("paper-profitability-control", "--json"),
                phase="verify",
                priority=44,
                repair_intent="explain exactly which post-cost persistence and execution evidence blocks paper order submission",
                stop_when="paper-profitability-control reports gradeable post-cost evidence and no unsafe paper execution blockers",
            ),
        ],
        "storage_reserve": [
            _playbook_task(
                lane="storage_reserve",
                infrabot_id="local_storage_reserve_infrabot",
                title="Local Storage Reserve Infrabot",
                command=_opsctl("local-storage-reserve-guard", "--apply", "--json"),
                phase="observe",
                priority=8,
                repair_intent="refresh internal reserve pressure and publish the ordered storage recovery pipeline",
                stop_when="local-storage-reserve-guard reports ready or names the exact route/storage blocker",
                resource_lock="local_storage_guard",
                allowed_when_uncontained=True,
                requires_collection_hot_path=False,
            ),
            _playbook_task(
                lane="storage_reserve",
                infrabot_id="writer_cycle_coordinator_infrabot",
                title="Writer Cycle Coordinator Infrabot",
                command=_opsctl(
                    "writer-cycle-coordinator", "--apply", "--fast-handoff", "--json"
                ),
                phase="stabilize",
                priority=20,
                repair_intent="clear or progress writer handoff before heavier drainers run",
                stop_when="writer-cycle-coordinator reports no blocked writer handoff or queue pressure remains under stable watermarks",
                resource_lock="single_writer",
                allowed_when_uncontained=True,
                requires_collection_hot_path=False,
            ),
            _playbook_task(
                lane="storage_reserve",
                infrabot_id="pcore_storage_contract_infrabot",
                title="P-Core Storage Contract Infrabot",
                command=_opsctl("backlog-pcore-accelerator", "--apply", "--json"),
                phase="repair",
                priority=24,
                repair_intent="enable bounded P-core file compaction while keeping SQLite checkpointing on the exclusive writer lane",
                stop_when="backlog-pcore-accelerator reports file compaction ready and sqlite_write_parallelism remains 1",
                resource_lock="p_core_file_compaction",
                allowed_when_uncontained=True,
                requires_collection_hot_path=False,
            ),
            _playbook_task(
                lane="storage_reserve",
                infrabot_id="raw_file_compaction_infrabot",
                title="Raw File Compaction Infrabot",
                command=_opsctl(
                    "raw-training-compaction",
                    "--apply",
                    "--max-files",
                    "50",
                    "--max-gb",
                    "8",
                    "--jumbo-gb",
                    "7",
                    "--compaction-workers",
                    "4",
                    "--json",
                ),
                phase="repair",
                priority=26,
                repair_intent="clear eligible raw file debt with bounded independent P-core compaction workers",
                stop_when="raw-training-compaction reports ready with zero apply failures",
                resource_lock="p_core_file_compaction",
                allowed_when_uncontained=True,
                requires_collection_hot_path=False,
            ),
            _playbook_task(
                lane="storage_reserve",
                infrabot_id="sqlite_checkpoint_infrabot",
                title="SQLite Checkpoint Infrabot",
                command=_opsctl(
                    "storage-pressure-clearance",
                    "--apply",
                    "--force-clear-stale-gate",
                    "--checkpoint-mode",
                    "passive",
                    "--json",
                ),
                phase="repair",
                priority=28,
                repair_intent="checkpoint WAL and clear stale storage pressure through the single SQLite writer path",
                stop_when="storage-pressure-clearance reports active_pressure=0 and error_count=0",
                resource_lock="single_writer",
                allowed_when_uncontained=True,
                requires_collection_hot_path=False,
            ),
            _playbook_task(
                lane="storage_reserve",
                infrabot_id="external_backlog_drain_infrabot",
                title="External Backlog Drain Infrabot",
                command=_opsctl(
                    "external-backlog-drain", "--apply", "--follow-through", "--json"
                ),
                phase="repair",
                priority=30,
                repair_intent="drain aged external backlog without touching hot-path collection",
                stop_when="external-backlog-drain reports no material drain recommended and no aged candidates",
                resource_lock="single_writer",
                allowed_when_uncontained=True,
                requires_collection_hot_path=False,
            ),
            _playbook_task(
                lane="storage_reserve",
                infrabot_id="external_route_rehome_planner",
                title="External Route Rehome Planner",
                command=_opsctl("storage-switch-external", "--dry-run"),
                phase="repair",
                priority=38,
                repair_intent="surface the explicit external route rehome command when local fallback is active under internal reserve pressure",
                stop_when="storage-switch-external dry-run shows target_mode=external and the operator chooses whether to mutate route",
                auto_execute=False,
                resource_lock="storage_route",
                allowed_when_uncontained=True,
                requires_collection_hot_path=False,
            ),
            _playbook_task(
                lane="storage_reserve",
                infrabot_id="verified_standby_prune_infrabot",
                title="Verified Standby Prune Infrabot",
                command=_opsctl("storage-prune-standby", "--apply", "--json"),
                phase="repair",
                priority=40,
                repair_intent="delete only route-verified local standby SQLite copies after external failback is certified",
                stop_when="storage-prune-standby reports pruned, no_eligible_standby, or ready_idle_active_local_route",
                resource_lock="storage_standby_prune",
                allowed_when_uncontained=True,
                requires_collection_hot_path=False,
            ),
            _playbook_task(
                lane="storage_reserve",
                infrabot_id="storage_backpressure_infrabot",
                title="Storage Backpressure Infrabot",
                command=_opsctl(
                    "storage-backpressure-autopilot",
                    "--apply",
                    "--quick-bounded",
                    "--wait-timeout-seconds",
                    "20",
                    "--command-timeout-seconds",
                    "60",
                    "--backpressure-command-timeout-seconds",
                    "30",
                    "--json",
                ),
                phase="verify",
                priority=46,
                repair_intent="verify backlog and reserve pressure remain bounded after drain/handoff work",
                stop_when="storage-backpressure-autopilot reports ready or only physical reserve debt remains",
                resource_lock="single_writer",
                allowed_when_uncontained=True,
                requires_collection_hot_path=False,
            ),
            _playbook_task(
                lane="storage_reserve",
                infrabot_id="storage_retention_unison_infrabot",
                title="Storage Retention Unison Infrabot",
                command=_opsctl("storage-retention-unison", "--json"),
                phase="verify",
                priority=50,
                repair_intent="verify reserve, route, backlog, and retention controllers agree after repairs",
                stop_when="storage-retention-unison reports ready/watch or names a stable external blocker",
                resource_lock="storage_readiness",
                allowed_when_uncontained=True,
                requires_collection_hot_path=False,
            ),
        ],
        "bot_organization": [
            _playbook_task(
                lane="bot_organization",
                infrabot_id="bot_organization_control_infrabot",
                title="Bot Organization Control Infrabot",
                command=_opsctl("bot-organization", "--json"),
                phase="observe",
                priority=12,
                repair_intent="refresh sleeve, role, review-debt, and tripwire organization",
                stop_when="bot-organization reports zero blocking tripwires",
            ),
            _playbook_task(
                lane="bot_organization",
                infrabot_id="infrabot_gap_roster_infrabot",
                title="Infrabot Gap Roster Infrabot",
                command=_opsctl("infrabot-gap-roster", "--json"),
                phase="verify",
                priority=48,
                repair_intent="find degradation lanes that still lack repair-bot coverage",
                stop_when="infrabot-gap-roster reports active_count=0 or all active gaps have assigned bots",
            ),
        ],
        "profitability_evidence": [
            _playbook_task(
                lane="profitability_evidence",
                infrabot_id="paper_performance_evidence_infrabot",
                title="Paper Performance Evidence Infrabot",
                command=_opsctl("paper-performance", "--json"),
                phase="observe",
                priority=11,
                repair_intent="refresh paper PnL, fill, and sleeve evidence without changing controls",
                stop_when="paper-performance is fresh enough for downstream evidence controls",
            ),
            _playbook_task(
                lane="profitability_evidence",
                infrabot_id="bot_profitability_scalability_infrabot",
                title="Bot Profitability Scalability Infrabot",
                command=_opsctl("bot-profitability-scalability", "--json"),
                phase="verify",
                priority=43,
                repair_intent="refresh per-bot post-cost persistence and capacity evidence debt",
                stop_when="bot-profitability-scalability reports selected scalable bots or explicitly contained evidence debt",
            ),
            _playbook_task(
                lane="profitability_evidence",
                infrabot_id="paper_profitability_gate_infrabot",
                title="Paper Profitability Gate Infrabot",
                command=_opsctl("paper-profitability-control", "--json"),
                phase="verify",
                priority=45,
                repair_intent="keep weak-profile containment aligned with current paper evidence",
                stop_when="paper-profitability-control reports gradeable evidence and safe paper controls",
            ),
        ],
        "sleeve_selection": [
            _playbook_task(
                lane="sleeve_selection",
                infrabot_id="sleeve_scalability_selector_infrabot",
                title="Sleeve Scalability Selector Infrabot",
                command=_opsctl("sleeve-scalability-selector", "--json"),
                phase="verify",
                priority=47,
                repair_intent="refresh sleeve-specific eligibility, capacity, and evidence gates",
                stop_when="sleeve selector reports eligible sleeves or contained evidence debt only",
            ),
            _playbook_task(
                lane="sleeve_selection",
                infrabot_id="sleeve_profitability_dashboard_infrabot",
                title="Sleeve Profitability Dashboard Infrabot",
                command=_opsctl("sleeve-profitability-dashboard", "--json"),
                phase="observe",
                priority=13,
                repair_intent="refresh sleeve-level PnL attribution before sleeve eligibility changes",
                stop_when="sleeve dashboard is fresh and no unaccounted sleeve drag remains",
            ),
        ],
        "master_grandmaster": [
            _playbook_task(
                lane="master_grandmaster",
                infrabot_id="master_grandmaster_evidence_infrabot",
                title="Master Grandmaster Evidence Infrabot",
                command=_opsctl("master-grandmaster-evidence", "--json"),
                phase="verify",
                priority=49,
                repair_intent="refresh master/grandmaster structural and evidence blockers",
                stop_when="master-grandmaster evidence reports structural grade A and promotion blockers are explicit",
            ),
            _playbook_task(
                lane="master_grandmaster",
                infrabot_id="promotion_quality_gate_infrabot",
                title="Promotion Quality Gate Infrabot",
                command=_opsctl("promotion-quality-gate", "--json"),
                phase="verify",
                priority=50,
                repair_intent="keep promotion blocked until evidence, lineage, and source gates are clean",
                stop_when="promotion quality gate is ready or explicitly lists non-runtime blockers",
            ),
        ],
        "source_verification": [
            _playbook_task(
                lane="source_verification",
                infrabot_id="source_verification_refresh_infrabot",
                title="Source Verification Refresh Infrabot",
                command=_opsctl("source-verification-refresh", "--apply", "--json"),
                phase="repair",
                priority=22,
                repair_intent="refresh stale or low-confidence context sources in dependency order",
                stop_when="decision-critical sources are ready and context debt is empty or quarantined",
            ),
            _playbook_task(
                lane="source_verification",
                infrabot_id="source_verification_report_infrabot",
                title="Source Verification Report Infrabot",
                command=_opsctl("source-verification", "--json"),
                phase="verify",
                priority=51,
                repair_intent="verify whether source debt remains context-only or blocks runtime",
                stop_when="source verification reports ready or contained_context_debt without decision-critical blockers",
            ),
            _playbook_task(
                lane="source_verification",
                infrabot_id="provider_mesh_infrabot",
                title="Provider Mesh Infrabot",
                command=_opsctl("provider-mesh", "--json"),
                phase="observe",
                priority=14,
                repair_intent="refresh provider context that explains why source confidence is low",
                stop_when="provider mesh is ready or lists bounded provider followups",
            ),
        ],
        "training_promotion": [
            _playbook_task(
                lane="training_promotion",
                infrabot_id="training_quality_infrabot",
                title="Training Quality Infrabot",
                command=_opsctl("training-quality", "--json"),
                phase="observe",
                priority=15,
                repair_intent="refresh training blockers while promotion stays frozen",
                stop_when="training quality is ready or blockers are evidence-only",
            ),
            _playbook_task(
                lane="training_promotion",
                infrabot_id="retrain_artifact_freshness_infrabot",
                title="Retrain Artifact Freshness Infrabot",
                command=_py_script(
                    "scripts/retrain_artifact_freshness_guard.py", "--json"
                ),
                phase="verify",
                priority=52,
                repair_intent="verify retrain artifact freshness without launching training",
                stop_when="retrain artifact freshness is ok or stale reason is contained",
            ),
            _playbook_task(
                lane="training_promotion",
                infrabot_id="promotion_quality_gate_infrabot",
                title="Promotion Quality Gate Infrabot",
                command=_opsctl("promotion-quality-gate", "--json"),
                phase="verify",
                priority=53,
                repair_intent="make promotion blockers explicit without granting promotion authority",
                stop_when="promotion-quality-gate reports ready or explicit held blockers",
            ),
        ],
        "ops_self_audit": [
            _playbook_task(
                lane="ops_self_audit",
                infrabot_id="infrastructure_autofix_infrabot",
                title="Infrastructure Autofix Infrabot",
                command=_opsctl("infrastructure-autofix", "--apply", "--json"),
                phase="repair",
                priority=24,
                repair_intent="run bounded infrastructure repairs only through existing owner commands",
                stop_when="infrastructure-autofix reports ready or no hot-path dependent finding remains",
            ),
            _playbook_task(
                lane="ops_self_audit",
                infrabot_id="system_drift_autopilot_infrabot",
                title="System Drift Autopilot Infrabot",
                command=_opsctl("system-drift-autopilot", "--apply", "--json"),
                phase="repair",
                priority=26,
                repair_intent="repair stale governance and system drift without touching live execution",
                stop_when="system drift artifacts and governance freshness no longer block master infra",
            ),
            _playbook_task(
                lane="ops_self_audit",
                infrabot_id="master_infra_supervisor_infrabot",
                title="Master Infrastructure Supervisor Infrabot",
                command=_opsctl("master-infra-supervisor", "--json"),
                phase="verify",
                priority=54,
                repair_intent="verify whether the ops debt remains contained after child repairs",
                stop_when="master-infra-supervisor reports ready or contained_degradation with no hot-path block",
            ),
        ],
    }


def _try_singleton_lock(path: Path) -> Any | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started_utc={iso_now()}\n")
    handle.flush()
    return handle


def _load_artifacts(project_root: Path) -> dict[str, dict[str, Any]]:
    health = project_root / "governance" / "health"
    return {
        "operator_cockpit": load_json(health / "operator_cockpit_latest.json"),
        "runtime_gate_dashboard": load_json(
            health / "runtime_gate_dashboard_latest.json"
        ),
        "master_infrastructure_supervisor": load_json(
            health / "master_infrastructure_supervisor_latest.json"
        ),
        "source_verification": load_json(health / "source_verification_latest.json"),
        "infrabot_adaptive_governor": load_json(
            health / "infrabot_adaptive_governor_latest.json"
        ),
    }


def _runtime_containment(runtime: dict[str, Any]) -> dict[str, Any]:
    top = _as_dict(runtime.get("degradation_containment"))
    nested = _as_dict(_as_dict(runtime.get("overall")).get("degradation_containment"))
    return top or nested


def _runtime_actions(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _as_list(_as_dict(runtime.get("overall")).get("remediation_actions"))
    return [row for row in rows if isinstance(row, dict)]


def _runtime_rows(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    containment = _runtime_containment(runtime)
    rows = _as_list(containment.get("containment_rows"))
    return [row for row in rows if isinstance(row, dict)]


def _lane_for_runtime_row(row: dict[str, Any]) -> str:
    attention = str(row.get("attention") or "").strip()
    domain = str(row.get("domain") or "").strip()
    return ATTENTION_TO_LANE.get(attention) or DOMAIN_TO_LANE.get(domain) or domain


def _build_degradation_model(project_root: Path) -> dict[str, Any]:
    artifacts = _load_artifacts(project_root)
    cockpit = artifacts["operator_cockpit"]
    runtime = artifacts["runtime_gate_dashboard"]
    master = artifacts["master_infrastructure_supervisor"]
    source = artifacts["source_verification"]

    cockpit_containment = _as_dict(cockpit.get("degradation_containment"))
    runtime_containment = _runtime_containment(runtime)
    master_containment = _as_dict(master.get("degradation_containment"))
    source_contract = _as_dict(source.get("containment_contract"))
    runtime_overall = _as_dict(runtime.get("overall"))
    runtime_attention = ordered_unique(
        [str(item) for item in _as_list(runtime_overall.get("attention"))]
    )
    cockpit_lanes = _as_dict(cockpit_containment.get("lanes"))
    contained_degradation_lanes = ordered_unique(
        [
            str(item)
            for item in _as_list(cockpit_containment.get("contained_degradation_lanes"))
            if str(item).strip()
        ]
    )
    uncontained_lanes = ordered_unique(
        [
            str(item)
            for item in _as_list(cockpit_containment.get("uncontained_lanes"))
            if str(item).strip()
        ]
    )
    if not contained_degradation_lanes and cockpit_lanes:
        contained_degradation_lanes = ordered_unique(
            [
                str(lane)
                for lane, row in cockpit_lanes.items()
                if isinstance(row, dict)
                and bool(row.get("contained", False))
                and _status(row.get("status")) not in {"ready", "clear", "ok", "stable"}
            ]
        )
    if not uncontained_lanes and cockpit_lanes:
        uncontained_lanes = ordered_unique(
            [
                str(lane)
                for lane, row in cockpit_lanes.items()
                if isinstance(row, dict) and not bool(row.get("contained", False))
            ]
        )

    rows_by_lane: dict[str, list[dict[str, Any]]] = {}
    for row in _runtime_rows(runtime):
        lane = _lane_for_runtime_row(row)
        if lane:
            rows_by_lane.setdefault(lane, []).append(row)

    remediation_by_attention = {
        str(row.get("attention") or "").strip(): row
        for row in _runtime_actions(runtime)
    }
    source_runtime = _as_dict(source.get("source_runtime_contract"))
    source_context_debt = ordered_unique(
        [
            *[
                str(item)
                for item in _as_list(source_contract.get("decision_context_debt"))
                if str(item).strip()
            ],
            *[
                str(item)
                for item in _as_list(source_runtime.get("decision_context_debt"))
                if str(item).strip()
            ],
            *[
                str(item)
                for item in _as_list(source_runtime.get("optional_enrichment_debt"))
                if str(item).strip()
            ],
        ]
    )
    source_decision_blockers = ordered_unique(
        [
            *[
                str(item)
                for item in _as_list(source_contract.get("decision_critical_blockers"))
                if str(item).strip()
            ],
            *[
                str(item)
                for item in _as_list(source_runtime.get("decision_critical_blockers"))
                if str(item).strip()
            ],
        ]
    )
    safe_to_collect = bool(cockpit_containment.get("safe_to_keep_collecting", False))
    status = str(cockpit_containment.get("status") or "").strip() or str(
        runtime_containment.get("status") or "unknown"
    )
    model = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "status": status,
        "operator_cockpit_status": cockpit.get("overall_status", "missing"),
        "runtime_gate_status": runtime_overall.get("status", "missing"),
        "master_infra_status": master.get("overall_status", "missing"),
        "source_verification_status": source.get("overall_status", "missing"),
        "safe_to_keep_collecting": safe_to_collect,
        "safe_to_submit_paper_orders": bool(
            cockpit_containment.get("safe_to_submit_paper_orders", False)
        ),
        "safe_to_train_or_promote": bool(
            cockpit_containment.get("safe_to_train_or_promote", False)
        ),
        "hot_path_blocked": bool(
            runtime_containment.get(
                "hot_path_blocked", master_containment.get("hot_path_blocked", False)
            )
        ),
        "contained_degradation_lanes": contained_degradation_lanes,
        "uncontained_lanes": uncontained_lanes,
        "runtime_attention": runtime_attention,
        "runtime_containment_rows_by_lane": rows_by_lane,
        "runtime_remediation_by_attention": remediation_by_attention,
        "cockpit_lanes": cockpit_lanes,
        "master_contained_lanes": _as_list(master_containment.get("contained_lanes")),
        "master_uncontained_lanes": _as_list(
            master_containment.get("uncontained_lanes")
        ),
        "master_containment_rows": _as_list(master_containment.get("rows")),
        "source_context_debt": source_context_debt,
        "source_decision_critical_blockers": source_decision_blockers,
        "source_containment": source_contract,
        "policy": "contained degradation becomes a bounded repair incident with lane context, owner commands, release conditions, and live-disabled execution controls",
    }
    model["incident_id"] = _incident_id(model)
    return model


def _command_from_runtime_remediation(
    project_root: Path,
    action: dict[str, Any],
    lane: str,
    priority: int,
) -> dict[str, Any] | None:
    command = [str(item) for item in _as_list(action.get("command"))]
    if not command:
        return None
    owner = str(action.get("owner") or "runtime_remediation_owner").strip()
    return _playbook_task(
        lane=lane,
        infrabot_id=f"{owner}_infrabot",
        title=f"{owner.replace('_', ' ').title()} Infrabot",
        command=command,
        phase="repair" if "--apply" in command else "verify",
        priority=priority,
        repair_intent=str(action.get("success_condition") or "").strip()
        or "run the runtime dashboard owner command for this contained signal",
        stop_when=str(action.get("success_condition") or "").strip()
        or "runtime dashboard no longer reports this attention item",
        auto_execute=True,
    )


def _lane_context(model: dict[str, Any], lane: str) -> dict[str, Any]:
    definition = _lane_definition(lane)
    section = str(definition.get("section") or "ops_brain")
    lane_row = _as_dict(_as_dict(model.get("cockpit_lanes")).get(lane))
    runtime_rows = [
        row
        for row in _as_list(
            _as_dict(model.get("runtime_containment_rows_by_lane")).get(lane)
        )
        if isinstance(row, dict)
    ]
    runtime_attentions = ordered_unique(
        [
            str(row.get("attention") or "")
            for row in runtime_rows
            if row.get("attention")
        ]
    )
    release_conditions = ordered_unique(
        [
            str(row.get("release_condition") or "")
            for row in runtime_rows
            if str(row.get("release_condition") or "").strip()
        ]
        + [
            str(lane_row.get("trade_impact") or ""),
        ]
    )
    return {
        "incident_id": model.get("incident_id"),
        "lane": lane,
        "section": section,
        "section_definition": _section_definition(section),
        "lane_definition": definition,
        "lane_status": lane_row.get("status", "unknown"),
        "contained": bool(lane_row.get("contained", False)),
        "owner": lane_row.get("owner", ""),
        "trade_impact": lane_row.get("trade_impact", ""),
        "runtime_attentions": runtime_attentions,
        "runtime_containment_rows": runtime_rows,
        "release_conditions": release_conditions,
        "source_context_debt": _as_list(model.get("source_context_debt")),
        "source_decision_critical_blockers": _as_list(
            model.get("source_decision_critical_blockers")
        ),
        "safe_to_keep_collecting": bool(model.get("safe_to_keep_collecting", False)),
        "safe_to_submit_paper_orders": bool(
            model.get("safe_to_submit_paper_orders", False)
        ),
        "safe_to_train_or_promote": bool(model.get("safe_to_train_or_promote", False)),
        "hot_path_blocked": bool(model.get("hot_path_blocked", False)),
        "hard_limits": [
            "do_not_place_live_orders",
            "do_not_submit_paper_orders_when_paper_execution_is_blocked",
            "do_not_promote_models_when_source_or_profitability_evidence_is_degraded",
            "do_not_start_competing_sqlite_writers",
            "do_not_loosen_trading_thresholds_as_a_repair_action",
        ],
    }


def _build_assignments(
    project_root: Path, model: dict[str, Any]
) -> list[dict[str, Any]]:
    playbooks = _lane_playbooks()
    lanes = ordered_unique(
        [
            str(item)
            for item in _as_list(model.get("contained_degradation_lanes"))
            if str(item).strip()
        ]
    )
    for lane in [
        str(item)
        for item in _as_list(model.get("uncontained_lanes"))
        if str(item).strip()
    ]:
        if any(
            bool(row.get("allowed_when_uncontained", False))
            for row in playbooks.get(lane, [])
        ):
            lanes.append(lane)
    lanes = ordered_unique(lanes)
    if not lanes and _status(model.get("status")) in {
        "contained",
        "contained_degradation",
    }:
        lanes = ordered_unique(
            [
                str(lane)
                for lane, row in _as_dict(model.get("cockpit_lanes")).items()
                if isinstance(row, dict)
                and bool(row.get("contained", False))
                and _status(row.get("status")) not in {"ready", "clear", "ok", "stable"}
            ]
        )

    raw_tasks: list[dict[str, Any]] = []
    for lane in lanes:
        raw_tasks.extend(playbooks.get(lane, []))
        for row in _as_list(
            _as_dict(model.get("runtime_containment_rows_by_lane")).get(lane)
        ):
            if not isinstance(row, dict):
                continue
            attention = str(row.get("attention") or "").strip()
            action = _as_dict(
                _as_dict(model.get("runtime_remediation_by_attention")).get(attention)
            )
            if action:
                task = _command_from_runtime_remediation(
                    project_root,
                    action,
                    lane,
                    PHASE_RANK.get("repair", 30) + len(raw_tasks),
                )
                if task:
                    raw_tasks.append(task)

    assignments: list[dict[str, Any]] = []
    seen_commands: set[tuple[str, ...]] = set()
    for task in sorted(
        raw_tasks,
        key=lambda row: (
            PHASE_RANK.get(str(row.get("phase") or ""), 99),
            _safe_int(row.get("priority"), 999),
            str(row.get("infrabot_id") or ""),
        ),
    ):
        command = [str(item) for item in _as_list(task.get("command"))]
        exec_command = _normalize_command(project_root, command)
        if not exec_command:
            continue
        command_key = tuple(exec_command)
        if command_key in seen_commands:
            continue
        seen_commands.add(command_key)
        lane = str(task.get("lane") or "").strip()
        safe = _command_safe(exec_command)
        auto_requested = bool(task.get("auto_execute_requested", False))
        blocked_by: list[str] = []
        if not safe:
            blocked_by.append("command_not_in_safe_allowlist")
        if not auto_requested:
            blocked_by.append("auto_execute_not_requested")
        if lane == "paper_execution" and "--apply" in exec_command:
            blocked_by.append("paper_execution_apply_blocked_while_guard_active")
        if (
            lane in {"training_promotion", "master_grandmaster"}
            and "--apply" in exec_command
        ):
            blocked_by.append("training_or_promotion_apply_blocked")
        if bool(model.get("uncontained_lanes")) and not bool(
            task.get("allowed_when_uncontained", False)
        ):
            blocked_by.append("uncontained_degradation_present")
        if not bool(model.get("safe_to_keep_collecting", False)) and bool(
            task.get("requires_collection_hot_path", True)
        ):
            blocked_by.append("collection_hot_path_not_ready")
        assignment_id = (
            f"{lane}:{task.get('infrabot_id', 'unknown')}:{_command_hash(exec_command)}"
        )
        lane_definition = _lane_definition(lane)
        phase = str(task.get("phase") or "")
        assignments.append(
            {
                "assignment_id": assignment_id,
                "incident_id": model.get("incident_id"),
                "lane": lane,
                "section": lane_definition.get("section", "ops_brain"),
                "domain": lane_definition.get("domain", "unclassified"),
                "canonical_owner": lane_definition.get("canonical_owner", ""),
                "lane_mission": lane_definition.get("mission", ""),
                "lane_release_signal": lane_definition.get(
                    "primary_release_signal", ""
                ),
                "phase": phase,
                "phase_definition": _phase_definition(phase),
                "priority": _safe_int(task.get("priority"), 999),
                "infrabot_id": task.get("infrabot_id"),
                "title": task.get("title"),
                "repair_intent": task.get("repair_intent"),
                "stop_when": task.get("stop_when"),
                "command": command,
                "exec_command": exec_command,
                "resource_lock": task.get("resource_lock", "none"),
                "max_attempts_per_incident": _safe_int(
                    task.get("max_attempts_per_incident"), 2
                ),
                "allowed_when_uncontained": bool(
                    task.get("allowed_when_uncontained", False)
                ),
                "requires_collection_hot_path": bool(
                    task.get("requires_collection_hot_path", True)
                ),
                "auto_execute_requested": auto_requested,
                "safe_execute_allowed": bool(
                    safe and auto_requested and not blocked_by
                ),
                "blocked_by": ordered_unique(blocked_by),
                "degradation_context": _lane_context(model, lane),
                "authority_boundary": task.get("authority_boundary"),
            }
        )
    return assignments


def _load_state(path: Path) -> dict[str, Any]:
    state = load_json(path)
    if not state:
        return {"schema_version": SCHEMA_VERSION, "assignments": {}}
    if not isinstance(state.get("assignments"), dict):
        state["assignments"] = {}
    return state


def _state_key(assignment: dict[str, Any]) -> str:
    return str(assignment.get("assignment_id") or "")


def _state_gate(
    state: dict[str, Any], assignment: dict[str, Any], *, now: datetime
) -> dict[str, Any]:
    assignments = _as_dict(state.get("assignments"))
    row = _as_dict(assignments.get(_state_key(assignment)))
    if not row:
        return {"active": False}
    incident_id = str(assignment.get("incident_id") or "")
    if str(row.get("incident_id") or "") != incident_id:
        return {"active": False}
    legacy_summary = _as_dict(row.get("last_summary"))
    legacy_reported_followup = bool(
        str(row.get("last_outcome") or "") == "failed"
        and (
            legacy_summary.get("ok") is True
            or _status(legacy_summary.get("overall_status"))
            in {
                "applied_still_degraded",
                "blocked",
                "degraded",
                "needs_attention",
                "warn",
                "warning",
            }
        )
    )
    if legacy_reported_followup:
        return {
            "active": False,
            "state": row,
            "legacy_reported_followup_state_ignored": True,
        }
    if str(row.get("last_outcome") or "") in {"success", "completed_with_followups"}:
        return {
            "active": True,
            "gate": "already_completed",
            "reason": "assignment_already_completed_for_current_incident",
            "state": row,
        }
    if bool(row.get("retry_budget_exhausted", False)):
        return {
            "active": True,
            "gate": "retry_budget",
            "reason": "assignment_retry_budget_exhausted_for_current_incident",
            "state": row,
        }
    cooldown = parse_iso_utc(row.get("cooldown_until_utc"))
    if cooldown is not None and cooldown > now:
        return {
            "active": True,
            "gate": "cooldown",
            "reason": "assignment_cooling_down_after_retryable_result",
            "cooldown_until_utc": cooldown.isoformat(),
            "state": row,
        }
    return {"active": False, "state": row}


def _classify_result(
    result: dict[str, Any],
    assignment: dict[str, Any] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    payload = _as_dict(result.get("payload"))
    status = _status(payload.get("overall_status") or payload.get("status"))
    ok = payload.get("ok")
    rc = _safe_int(result.get("rc"), 1)
    timed_out = bool(result.get("timed_out", False))
    timeout_progressing = False
    if timed_out:
        progress_payload = _timeout_progress_artifact(project_root, assignment)
        if progress_payload:
            payload = progress_payload
            status = _status(payload.get("overall_status") or payload.get("status"))
            ok = payload.get("ok")
            outcome = "completed_with_followups"
            success_like = True
            retryable = False
            command_failed = False
            requires_followup = True
            lane_unresolved = status not in {"ready", "applied"}
            timeout_progressing = True
        else:
            outcome = "timeout"
            success_like = False
            retryable = True
            command_failed = True
            requires_followup = True
            lane_unresolved = True
    elif (
        payload
        and rc in {0, 2, 3}
        and (
            ok is False
            or status
            in {
                "applied_still_degraded",
                "blocked",
                "degraded",
                "needs_attention",
                "warn",
                "warning",
            }
            or rc in {2, 3}
        )
    ):
        outcome = "completed_with_followups"
        success_like = True
        retryable = False
        command_failed = False
        requires_followup = True
        lane_unresolved = True
    elif rc == 0:
        outcome = "success"
        success_like = True
        retryable = False
        command_failed = False
        requires_followup = False
        lane_unresolved = False
    else:
        outcome = "failed"
        success_like = False
        retryable = True
        command_failed = True
        requires_followup = True
        lane_unresolved = True
    return {
        "outcome": outcome,
        "success_like": success_like,
        "retryable": retryable,
        "command_failed": command_failed,
        "requires_followup": requires_followup,
        "lane_unresolved": lane_unresolved,
        "timeout_progressing": timeout_progressing,
        "summary": {
            "overall_status": payload.get("overall_status") or payload.get("status"),
            "ok": ok,
            "busy": payload.get("busy"),
            "quick_bounded": payload.get("quick_bounded"),
            "timeout_progress_artifact": payload.get("timeout_progress_artifact"),
            "timeout_progress_classification": payload.get(
                "timeout_progress_classification"
            ),
            "recommended_actions_count": len(
                _as_list(payload.get("recommended_actions"))
            ),
        },
    }


def _update_state(
    state_path: Path,
    state: dict[str, Any],
    assignment: dict[str, Any],
    classification: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    assignments = _as_dict(state.setdefault("assignments", {}))
    key = _state_key(assignment)
    previous = _as_dict(assignments.get(key))
    success_like = bool(classification.get("success_like", False))
    retryable = bool(classification.get("retryable", False))
    previous_failures = _safe_int(previous.get("failure_count"), 0)
    failure_count = 0 if success_like and not retryable else previous_failures
    if not success_like and retryable:
        failure_count = previous_failures + 1
    max_attempts = _safe_int(assignment.get("max_attempts_per_incident"), 2)
    retry_budget_exhausted = bool(failure_count >= max_attempts and retryable)
    cooldown_seconds = 0
    if retryable and not retry_budget_exhausted:
        cooldown_seconds = 300 * max(1, min(failure_count + 1, 6))
    cooldown_until = (
        datetime.now(timezone.utc) + timedelta(seconds=cooldown_seconds)
        if cooldown_seconds
        else None
    )
    row = {
        "assignment_id": key,
        "incident_id": assignment.get("incident_id"),
        "lane": assignment.get("lane"),
        "infrabot_id": assignment.get("infrabot_id"),
        "last_seen_utc": iso_now(),
        "last_command": assignment.get("exec_command"),
        "last_returncode": result.get("rc"),
        "last_timed_out": bool(result.get("timed_out", False)),
        "last_outcome": classification.get("outcome"),
        "last_summary": _as_dict(classification.get("summary")),
        "failure_count": failure_count,
        "max_attempts_per_incident": max_attempts,
        "retry_budget_exhausted": retry_budget_exhausted,
        "cooldown_seconds": cooldown_seconds,
        "cooldown_until_utc": cooldown_until.isoformat() if cooldown_until else "",
    }
    assignments[key] = row
    state["assignments"] = assignments
    state["timestamp_utc"] = iso_now()
    write_payload(state_path, state)
    return row


def _parse_json_stdout(stdout: str) -> dict[str, Any]:
    for raw in reversed(
        [line.strip() for line in str(stdout or "").splitlines() if line.strip()]
    ):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _recent_owner_artifact(
    payload: dict[str, Any], *, max_age_seconds: int = 900
) -> bool:
    timestamp = parse_iso_utc(payload.get("timestamp_utc"))
    if timestamp is None:
        return False
    age = datetime.now(timezone.utc) - timestamp
    return age.total_seconds() <= max(int(max_age_seconds), 1)


def _timeout_progress_artifact(
    project_root: Path, assignment: dict[str, Any] | None
) -> dict[str, Any]:
    if not assignment:
        return {}
    command = _as_list(assignment.get("exec_command")) or _as_list(
        assignment.get("command")
    )
    subcommand = _opsctl_subcommand([str(item) for item in command])
    progress_artifacts = {
        "storage-backpressure-autopilot": {
            "path": STORAGE_BACKPRESSURE_OUT_PATH.name,
            "statuses": {
                "running",
                "already_running",
                "ready",
                "applied",
                "applied_with_followups",
            },
        },
        "external-backlog-drain": {
            "path": "external_backlog_drain_latest.json",
            "statuses": {
                "drain_active",
                "ready",
                "waiting_for_off_hours",
                "applied",
                "applied_with_followups",
            },
        },
    }
    artifact_contract = _as_dict(progress_artifacts.get(subcommand))
    if not artifact_contract:
        return {}
    artifact_path = (
        project_root / "governance" / "health" / str(artifact_contract.get("path"))
    )
    payload = load_json(artifact_path)
    if not payload or not _recent_owner_artifact(payload):
        return {}
    status = _status(payload.get("overall_status") or payload.get("status"))
    allowed_statuses = {
        str(item) for item in _as_list(list(artifact_contract.get("statuses") or []))
    }
    if payload.get("ok") is True and status in allowed_statuses:
        out = dict(payload)
        out["timeout_progress_artifact"] = str(artifact_path)
        out["timeout_progress_classification"] = (
            "owner_artifact_recent_ok_and_progressing"
        )
        return out
    return {}


def _owner_progress_classification(payload: dict[str, Any]) -> dict[str, Any]:
    status = _status(payload.get("overall_status") or payload.get("status"))
    unresolved = status not in {"ready", "applied"}
    return {
        "outcome": "completed_with_followups" if unresolved else "success",
        "success_like": True,
        "retryable": False,
        "command_failed": False,
        "requires_followup": unresolved,
        "lane_unresolved": unresolved,
        "timeout_progressing": True,
        "summary": {
            "overall_status": status,
            "ok": payload.get("ok"),
            "timeout_progress_artifact": payload.get("timeout_progress_artifact"),
            "timeout_progress_classification": payload.get(
                "timeout_progress_classification"
            ),
        },
    }


def _append_ledger(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")


def _execute_assignments(
    project_root: Path,
    payload: dict[str, Any],
    *,
    max_execute_actions: int,
    command_timeout_seconds: int,
    context_path: Path,
    state_path: Path,
    lock_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    lock = _try_singleton_lock(lock_path)
    if lock is None:
        return {
            "enabled": True,
            "executed_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "blocked": True,
            "reason": "degradation_swarm_lock_active",
            "lock_path": str(lock_path),
        }
    try:
        max_execute = max(int(max_execute_actions), 0)
        timeout = max(int(command_timeout_seconds), 10)
        state = _load_state(state_path)
        now = datetime.now(timezone.utc)
        assignments = [
            row
            for row in _as_list(payload.get("assignments"))
            if isinstance(row, dict) and bool(row.get("safe_execute_allowed", False))
        ]
        assignments = sorted(
            assignments,
            key=lambda row: (
                PHASE_RANK.get(str(row.get("phase") or ""), 99),
                _safe_int(row.get("priority"), 999),
                str(row.get("assignment_id") or ""),
            ),
        )

        env = os.environ.copy()
        env.update(SAFE_EXEC_ENV)
        env["BOT_RUNTIME_PROFILE"] = env.get("BOT_RUNTIME_PROFILE", "live")
        env["PYTHONUNBUFFERED"] = "1"
        env["DEGRADATION_SWARM_CONTEXT_PATH"] = str(context_path)
        env["DEGRADATION_SWARM_INCIDENT_ID"] = str(payload.get("incident_id") or "")
        stack = [
            item.strip()
            for item in str(env.get("INFRA_REPAIR_CALL_STACK", "")).split(",")
            if item.strip()
        ]
        if "degradation_swarm_coordinator" not in stack:
            stack.append("degradation_swarm_coordinator")
        env["INFRA_REPAIR_CALL_STACK"] = ",".join(stack)

        results: list[dict[str, Any]] = []
        executed_count = 0
        failed_count = 0
        followup_count = 0
        skipped_count = 0
        cooldown_skipped_count = 0
        retry_budget_skipped_count = 0
        already_completed_skipped_count = 0
        already_completed_success_count = 0
        already_completed_followup_count = 0
        completed_from_state_count = 0
        timed_out_count = 0
        raw_timed_out_count = 0
        progress_reclassified_timeout_count = 0
        deadline = time.monotonic() + max_execute * max(timeout, 1) + 5
        for assignment in assignments:
            if executed_count >= max_execute:
                skipped_count += 1
                results.append(
                    {
                        "assignment_id": assignment.get("assignment_id"),
                        "infrabot_id": assignment.get("infrabot_id"),
                        "lane": assignment.get("lane"),
                        "executed": False,
                        "reason": "max_execute_actions_reached",
                    }
                )
                continue
            gate = _state_gate(state, assignment, now=now)
            if bool(gate.get("active", False)):
                if str(gate.get("gate") or "") in {"cooldown", "retry_budget"}:
                    progress_payload = _timeout_progress_artifact(
                        project_root, assignment
                    )
                    if progress_payload:
                        classification = _owner_progress_classification(
                            progress_payload
                        )
                        state_row = _update_state(
                            state_path,
                            state,
                            assignment,
                            classification,
                            {
                                "rc": 0,
                                "timed_out": False,
                                "payload": progress_payload,
                            },
                        )
                        skipped_count += 1
                        completed_from_state_count += 1
                        followup_count += 1
                        results.append(
                            {
                                "assignment_id": assignment.get("assignment_id"),
                                "infrabot_id": assignment.get("infrabot_id"),
                                "lane": assignment.get("lane"),
                                "executed": False,
                                "reason": "cooldown_timeout_reclassified_from_recent_owner_progress",
                                "self_healing_gate": gate,
                                "classification": classification,
                                "state": state_row,
                            }
                        )
                        continue
                skipped_count += 1
                if str(gate.get("gate") or "") == "retry_budget":
                    retry_budget_skipped_count += 1
                elif str(gate.get("gate") or "") == "already_completed":
                    already_completed_skipped_count += 1
                    completed_from_state_count += 1
                    state_row = _as_dict(gate.get("state"))
                    if (
                        str(state_row.get("last_outcome") or "")
                        == "completed_with_followups"
                    ):
                        already_completed_followup_count += 1
                        followup_count += 1
                    else:
                        already_completed_success_count += 1
                else:
                    cooldown_skipped_count += 1
                state_row = _as_dict(gate.get("state"))
                previous_outcome = str(state_row.get("last_outcome") or "")
                results.append(
                    {
                        "assignment_id": assignment.get("assignment_id"),
                        "infrabot_id": assignment.get("infrabot_id"),
                        "lane": assignment.get("lane"),
                        "executed": False,
                        "reason": gate.get("reason"),
                        "self_healing_gate": gate,
                        "classification": (
                            {
                                "outcome": previous_outcome,
                                "success_like": previous_outcome
                                in {"success", "completed_with_followups"},
                                "retryable": False,
                                "command_failed": False,
                                "requires_followup": previous_outcome
                                == "completed_with_followups",
                                "lane_unresolved": previous_outcome
                                == "completed_with_followups",
                                "summary": _as_dict(state_row.get("last_summary")),
                            }
                            if previous_outcome
                            else {}
                        ),
                    }
                )
                continue
            progress_payload = _timeout_progress_artifact(project_root, assignment)
            if progress_payload:
                classification = _owner_progress_classification(progress_payload)
                state_row = _update_state(
                    state_path,
                    state,
                    assignment,
                    classification,
                    {
                        "rc": 0,
                        "timed_out": False,
                        "payload": progress_payload,
                    },
                )
                skipped_count += 1
                completed_from_state_count += 1
                followup_count += (
                    1 if bool(classification.get("requires_followup", False)) else 0
                )
                results.append(
                    {
                        "assignment_id": assignment.get("assignment_id"),
                        "infrabot_id": assignment.get("infrabot_id"),
                        "lane": assignment.get("lane"),
                        "executed": False,
                        "reason": "recent_owner_progress_detected",
                        "classification": classification,
                        "state": state_row,
                    }
                )
                continue
            remaining = int(max(deadline - time.monotonic(), 0))
            if remaining <= 0:
                skipped_count += 1
                results.append(
                    {
                        "assignment_id": assignment.get("assignment_id"),
                        "infrabot_id": assignment.get("infrabot_id"),
                        "lane": assignment.get("lane"),
                        "executed": False,
                        "reason": "swarm_timeout_budget_exhausted",
                    }
                )
                continue
            command_timeout = min(timeout, max(remaining, 1))
            env["DEGRADATION_SWARM_ASSIGNMENT_ID"] = str(
                assignment.get("assignment_id") or ""
            )
            env["DEGRADATION_SWARM_LANE"] = str(assignment.get("lane") or "")
            result = run_bounded_process_group(
                [str(part) for part in _as_list(assignment.get("exec_command"))],
                cwd=project_root,
                timeout_seconds=command_timeout,
                env=env,
            )
            result["payload"] = _parse_json_stdout(str(result.get("stdout") or ""))
            classification = _classify_result(
                result, assignment=assignment, project_root=project_root
            )
            state_row = _update_state(
                state_path, state, assignment, classification, result
            )
            executed_count += 1
            failed = not bool(classification.get("success_like", False))
            failed_count += 1 if failed else 0
            followup_count += (
                1 if bool(classification.get("requires_followup", False)) else 0
            )
            raw_timed_out = bool(result.get("timed_out", False))
            timeout_progressing = bool(classification.get("timeout_progressing", False))
            raw_timed_out_count += 1 if raw_timed_out else 0
            progress_reclassified_timeout_count += (
                1 if raw_timed_out and timeout_progressing else 0
            )
            timed_out_count += 1 if raw_timed_out and not timeout_progressing else 0
            event = {
                "timestamp_utc": iso_now(),
                "incident_id": payload.get("incident_id"),
                "assignment_id": assignment.get("assignment_id"),
                "lane": assignment.get("lane"),
                "infrabot_id": assignment.get("infrabot_id"),
                "command": assignment.get("exec_command"),
                "classification": classification,
                "state": state_row,
            }
            _append_ledger(ledger_path, event)
            results.append(
                {
                    "assignment_id": assignment.get("assignment_id"),
                    "infrabot_id": assignment.get("infrabot_id"),
                    "lane": assignment.get("lane"),
                    "phase": assignment.get("phase"),
                    "executed": True,
                    "returncode": result.get("rc"),
                    "timed_out": bool(result.get("timed_out", False)),
                    "failed": failed,
                    "classification": classification,
                    "state": state_row,
                    "stdout_tail": str(result.get("stdout") or "")[-1200:],
                    "stderr_tail": str(result.get("stderr") or "")[-1200:],
                }
            )
        return {
            "enabled": True,
            "executed_count": executed_count,
            "failed_count": failed_count,
            "followup_count": followup_count,
            "skipped_count": skipped_count,
            "cooldown_skipped_count": cooldown_skipped_count,
            "retry_budget_skipped_count": retry_budget_skipped_count,
            "already_completed_skipped_count": already_completed_skipped_count,
            "already_completed_success_count": already_completed_success_count,
            "already_completed_followup_count": already_completed_followup_count,
            "completed_from_state_count": completed_from_state_count,
            "effective_completed_count": executed_count + completed_from_state_count,
            "timed_out_count": timed_out_count,
            "raw_timed_out_count": raw_timed_out_count,
            "progress_reclassified_timeout_count": progress_reclassified_timeout_count,
            "max_execute_actions": max_execute,
            "command_timeout_seconds": timeout,
            "state_path": str(state_path),
            "ledger_path": str(ledger_path),
            "live_execution_authority": False,
            "paper_order_submission_authority": False,
            "results": results,
        }
    finally:
        try:
            lock.close()
        except Exception:
            pass


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    execute_safe_repairs: bool = False,
    max_execute_actions: int = 4,
    command_timeout_seconds: int = 180,
    out_path: Path = DEFAULT_OUT_PATH,
    context_path: Path = DEFAULT_CONTEXT_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    lock_path: Path = DEFAULT_LOCK_PATH,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    model = _build_degradation_model(project_root)
    assignments = _build_assignments(project_root, model)
    operating_model = _build_swarm_operating_model(model, assignments)
    executable_assignments = [
        row for row in assignments if bool(row.get("safe_execute_allowed", False))
    ]
    blocked_assignments = [
        row for row in assignments if not bool(row.get("safe_execute_allowed", False))
    ]
    uncontained_lanes = _as_list(model.get("uncontained_lanes"))
    storage_recovery_assignments = [
        row
        for row in executable_assignments
        if str(row.get("lane") or "") == "storage_reserve"
        and bool(row.get("allowed_when_uncontained", False))
    ]
    standard_contained_ready = bool(
        _status(model.get("status")) in {"contained", "contained_degradation"}
        and not uncontained_lanes
        and bool(model.get("safe_to_keep_collecting", False))
        and bool(assignments)
    )
    storage_recovery_ready = bool(uncontained_lanes and storage_recovery_assignments)
    ready_to_swarm = bool(standard_contained_ready or storage_recovery_ready)
    can_execute = bool(ready_to_swarm and executable_assignments)
    execution_policy = {
        "safe_apply_only": True,
        "exact_allowlist_enforced": True,
        "bounded_action_budget": max(int(max_execute_actions), 0),
        "per_command_timeout_seconds": max(int(command_timeout_seconds), 10),
        "execute_requires_apply": True,
        "live_execution_authority": False,
        "paper_order_submission_authority": False,
        "promotion_authority": False,
        "training_launch_authority": False,
        "single_writer_commands_are_serialized": True,
        "storage_recovery_can_run_while_other_lanes_uncontained": True,
        "infrabot_context_env": "DEGRADATION_SWARM_CONTEXT_PATH",
    }
    swarm = {
        "ready_to_swarm": ready_to_swarm,
        "mode": (
            "contained_lane_repair"
            if standard_contained_ready
            else (
                "storage_recovery_while_global_degradation_uncontained"
                if storage_recovery_ready
                else "blocked"
            )
        ),
        "can_execute_safe_repairs": can_execute,
        "active_lane_count": len(_as_list(model.get("contained_degradation_lanes"))),
        "active_assignment_count": len(assignments),
        "safe_executable_assignment_count": len(executable_assignments),
        "storage_recovery_assignment_count": len(storage_recovery_assignments),
        "blocked_assignment_count": len(blocked_assignments),
        "active_sections": _as_list(operating_model.get("active_sections")),
        "active_phase_order": _as_list(operating_model.get("active_phase_order")),
        "refinement_backlog_count": len(
            _as_list(operating_model.get("refinement_backlog"))
        ),
        "contained_lane_repair_order": _as_list(
            model.get("contained_degradation_lanes")
        ),
        "policy": "lane-specific infrabots get the same incident context and only run bounded repair or verification commands",
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "overall_status": (
            "storage_recovery"
            if storage_recovery_ready
            else (
                "blocked"
                if uncontained_lanes
                or (
                    assignments
                    and not bool(model.get("safe_to_keep_collecting", False))
                )
                else "ready" if not assignments else "coordinating"
            )
        ),
        "incident_id": model.get("incident_id"),
        "source_incident_id": model.get("incident_id"),
        "active_lane_count": len(_as_list(model.get("contained_degradation_lanes"))),
        "active_assignment_count": len(assignments),
        "safe_executable_assignment_count": len(executable_assignments),
        "blocked_assignment_count": len(blocked_assignments),
        "degradation_model": model,
        "operating_model": operating_model,
        "swarm": swarm,
        "assignments": assignments,
        "execution_policy": execution_policy,
        "apply_result": {
            "applied": False,
            "context_written": False,
            "executed_safe_repairs": False,
            "note": "Dry run only; use --apply to publish the swarm context and --execute-safe-repairs to run bounded safe repairs.",
        },
        "execution_summary": {
            "enabled": False,
            "executed_count": 0,
            "failed_count": 0,
            "followup_count": 0,
            "skipped_count": 0,
            "already_completed_skipped_count": 0,
        },
    }

    if apply:
        context = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": iso_now(),
            "incident_id": payload.get("incident_id"),
            "degradation_model": model,
            "operating_model": operating_model,
            "assignments": assignments,
            "execution_policy": execution_policy,
            "policy": "infrabots should read this packet before repair so they preserve containment boundaries and lane release conditions",
        }
        write_payload(context_path, context)
        payload["apply_result"] = {
            "applied": True,
            "context_written": True,
            "context_path": str(context_path),
            "executed_safe_repairs": False,
        }
        if execute_safe_repairs:
            execution = _execute_assignments(
                project_root,
                payload,
                max_execute_actions=max_execute_actions,
                command_timeout_seconds=command_timeout_seconds,
                context_path=context_path,
                state_path=state_path,
                lock_path=lock_path,
                ledger_path=ledger_path,
            )
            payload["execution_summary"] = execution
            _as_dict(payload["apply_result"])["executed_safe_repairs"] = True
            if _safe_int(execution.get("failed_count"), 0) > 0:
                payload["overall_status"] = "executed_with_followups"
            elif _safe_int(execution.get("followup_count"), 0) > 0:
                payload["overall_status"] = "executed_with_followups"
            elif _safe_int(execution.get("effective_completed_count"), 0) > 0:
                payload["overall_status"] = "executed"
            elif bool(execution.get("blocked", False)):
                payload["overall_status"] = "blocked"
        write_payload(out_path, payload)
    else:
        write_payload(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Coordinate bounded infrabot repair swarms for contained degradation lanes."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--context-file", default=str(DEFAULT_CONTEXT_PATH))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--ledger-file", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--execute-safe-repairs",
        action="store_true",
        help="Run exact allowlisted repair commands for contained lanes.",
    )
    parser.add_argument("--max-execute-actions", type=int, default=4)
    parser.add_argument("--command-timeout-seconds", type=int, default=180)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.execute_safe_repairs and not args.apply:
        parser.error("--execute-safe-repairs requires --apply")
    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        execute_safe_repairs=bool(args.execute_safe_repairs),
        max_execute_actions=int(args.max_execute_actions),
        command_timeout_seconds=int(args.command_timeout_seconds),
        out_path=Path(args.out_file).expanduser(),
        context_path=Path(args.context_file).expanduser(),
        state_path=Path(args.state_file).expanduser(),
        lock_path=Path(args.lock_file).expanduser(),
        ledger_path=Path(args.ledger_file).expanduser(),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "degradation_swarm_coordinator "
            f"overall_status={payload.get('overall_status', '')} "
            f"ready_to_swarm={bool(_as_dict(payload.get('swarm')).get('ready_to_swarm', False))} "
            f"assignments={len(_as_list(payload.get('assignments')))} "
            f"safe_executable={_safe_int(_as_dict(payload.get('swarm')).get('safe_executable_assignment_count'), 0)}"
        )
    return 0 if payload.get("overall_status") != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
