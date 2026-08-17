#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_VERSION = 819
CONTROL_VERSION = "cognitive_control_plane_v1"
LABEL_CONTRACT_VERSION = "cognitive_control_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 8000
MINIMUM_COLLECTION_DAYS = 30
PAPER_RUNTIME_CAPACITY_FLOOR = 700

CONTROL_SLUG = "cognitive_control_plane"
CONTROL_DISPLAY_NAME = "Cognitive Control Plane"
CONTROL_FAMILY = "cognitive_governance"
CONTROL_PROFILE = "cognitive_control_plane"

DATA_INTAKES = [
    "cognitive_plan_trace",
    "belief_state_posterior_trace",
    "multi_agent_debate_score_trace",
    "formal_alignment_invariant_trace",
    "strategy_synthesis_candidate_trace",
    "curriculum_learning_gap_trace",
    "memory_distillation_digest_trace",
    "epistemic_risk_budget_trace",
    "counterfactual_plan_replay_trace",
    "research_backlog_priority_trace",
    "policy_simulation_sandbox_trace",
    "safety_case_evidence_trace",
    "feedback_credit_assignment_trace",
    "agent_market_routing_trace",
    "grandmaster_cognitive_packet_trace",
]

STORAGE_TARGETS = [
    "governance/cognitive_control_plane",
    "governance/cognitive_control_plane/plans",
    "governance/cognitive_control_plane/debate",
    "governance/cognitive_control_plane/safety_cases",
    "governance/health/cognitive_control_plane",
    "data/jsonl_link.sqlite3",
]

REQUIRED_LABELS = [
    "plan_quality_bucket",
    "belief_confidence_bucket",
    "debate_consensus_score",
    "alignment_invariant_status",
    "strategy_candidate_grade",
    "epistemic_budget_bucket",
    "safety_case_status",
]

CONTROL_CONTRACT = {
    "contract_version": CONTROL_VERSION,
    "purpose": "recursive_cognitive_governance_for_planning_debate_memory_distillation_and_safe_strategy_synthesis",
    "cognitive_depth": [
        "hierarchical_planning",
        "belief_state_fusion",
        "multi_agent_debate",
        "formal_objective_alignment",
        "strategy_synthesis",
        "curriculum_learning",
        "memory_distillation",
        "epistemic_risk_budgeting",
        "counterfactual_planning",
        "autonomous_research_backlog",
        "policy_simulation_sandbox",
        "safety_case_construction",
        "feedback_credit_assignment",
        "agent_market_routing",
        "grandmaster_cognition_bridge",
    ],
    "graduation_policy": "collection_only_until_long_horizon_evidence_replay_and_alignment_guards_clear",
    "global_halt_contract": "cognitive_expansion_must_downshift_before_global_halt",
    "paper_lock_contract": "paper_trade_lock_required_before_any_execution_path",
    "resource_contract": "heavy_cognition_runs_off_hot_path_and_prefers_cold_lane_windows",
}

BOTS: list[dict[str, Any]] = [
    {
        "role_slug": "planning_orchestrator_master",
        "slug": "cognitive_planning_orchestrator_master_bot",
        "label": "Cognitive Planning Orchestrator Master",
        "bot_role": "infrastructure_sub_bot",
        "cognitive_layer": "hierarchical_planning",
        "priority": "critical",
        "objective": "Turn platform goals into bounded, ordered, pressure-aware plans for sleeves, guards, and research lanes.",
        "target_functions": ["sleeve_master", "operator_cockpit", "platform_control_plane", "grand_master_reporting"],
    },
    {
        "role_slug": "belief_state_fusion_engine",
        "slug": "cognitive_belief_state_fusion_engine_bot",
        "label": "Cognitive Belief State Fusion Engine",
        "bot_role": "signal_sub_bot",
        "cognitive_layer": "belief_state_fusion",
        "priority": "critical",
        "objective": "Maintain posterior-style belief state across events, regimes, data quality, model confidence, and resource pressure.",
        "target_functions": ["market_regime_router", "event_intelligence", "model_lifecycle", "risk_service"],
    },
    {
        "role_slug": "multi_agent_debate_panel",
        "slug": "cognitive_multi_agent_debate_panel_bot",
        "label": "Cognitive Multi-Agent Debate Panel",
        "bot_role": "infrastructure_sub_bot",
        "cognitive_layer": "multi_agent_debate",
        "priority": "high",
        "objective": "Compare competing sleeve hypotheses and score consensus, dissent, and evidence quality before promotion.",
        "target_functions": ["research_pipeline", "decision_provenance", "multiple_testing_guard"],
    },
    {
        "role_slug": "formal_objective_alignment_guard",
        "slug": "cognitive_formal_objective_alignment_guard_bot",
        "label": "Cognitive Formal Objective Alignment Guard",
        "bot_role": "infrastructure_sub_bot",
        "cognitive_layer": "formal_objective_alignment",
        "priority": "critical",
        "objective": "Check whether proposed learning and strategy actions still obey paper lock, halt, retention, and no-lookahead invariants.",
        "target_functions": ["formal_verification", "paper_trade_lock", "global_halt_status", "regression_guard"],
    },
    {
        "role_slug": "strategy_synthesis_generator",
        "slug": "cognitive_strategy_synthesis_generator_bot",
        "label": "Cognitive Strategy Synthesis Generator",
        "bot_role": "signal_sub_bot",
        "cognitive_layer": "strategy_synthesis",
        "priority": "high",
        "objective": "Generate candidate strategy ideas from gaps, regimes, research evidence, and undercovered market structure.",
        "target_functions": ["research_automation", "strategy_coverage", "new_bot_admission_guard"],
    },
    {
        "role_slug": "curriculum_learning_scheduler",
        "slug": "cognitive_curriculum_learning_scheduler_bot",
        "label": "Cognitive Curriculum Learning Scheduler",
        "bot_role": "infrastructure_sub_bot",
        "cognitive_layer": "curriculum_learning",
        "priority": "high",
        "objective": "Order training, replay, and data-collection tasks from easiest confidence wins to hardest gaps.",
        "target_functions": ["coverage_gap_closer", "training_requalification", "model_lifecycle"],
    },
    {
        "role_slug": "memory_distillation_compression",
        "slug": "cognitive_memory_distillation_compression_bot",
        "label": "Cognitive Memory Distillation Compression",
        "bot_role": "infrastructure_sub_bot",
        "cognitive_layer": "memory_distillation",
        "priority": "critical",
        "objective": "Compress experience memory into durable summaries and retire low-value raw traces under retention guard.",
        "target_functions": ["experience_accumulation_memory", "data_retention", "storage_quota_guard"],
    },
    {
        "role_slug": "epistemic_risk_budget_allocator",
        "slug": "cognitive_epistemic_risk_budget_allocator_bot",
        "label": "Cognitive Epistemic Risk Budget Allocator",
        "bot_role": "signal_sub_bot",
        "cognitive_layer": "epistemic_risk_budgeting",
        "priority": "critical",
        "objective": "Allocate attention and collection budget to the highest uncertainty/highest consequence decisions.",
        "target_functions": ["uncertainty_calibration", "runtime_throttle", "portfolio_risk_layer"],
    },
    {
        "role_slug": "counterfactual_scenario_planner",
        "slug": "cognitive_counterfactual_scenario_planner_bot",
        "label": "Cognitive Counterfactual Scenario Planner",
        "bot_role": "signal_sub_bot",
        "cognitive_layer": "counterfactual_planning",
        "priority": "high",
        "objective": "Design what-if replays that test whether strategy improvements survive alternate event and liquidity paths.",
        "target_functions": ["stress_lab", "causal_counterfactual_evidence", "golden_replay_guard"],
    },
    {
        "role_slug": "autonomous_research_backlog_prioritizer",
        "slug": "cognitive_autonomous_research_backlog_prioritizer_bot",
        "label": "Cognitive Autonomous Research Backlog Prioritizer",
        "bot_role": "infrastructure_sub_bot",
        "cognitive_layer": "autonomous_research_backlog",
        "priority": "medium",
        "objective": "Rank research backlog items by novelty, evidence, system need, cost, and overlap with existing sleeves.",
        "target_functions": ["research_pipeline", "experiment_ledger", "duplicate_alpha_detector"],
    },
    {
        "role_slug": "policy_simulation_sandbox",
        "slug": "cognitive_policy_simulation_sandbox_bot",
        "label": "Cognitive Policy Simulation Sandbox",
        "bot_role": "infrastructure_sub_bot",
        "cognitive_layer": "policy_simulation_sandbox",
        "priority": "critical",
        "objective": "Replay proposed policy and throttle changes before they can touch runtime or training gates.",
        "target_functions": ["runtime_throttle", "release_freeze_guard", "golden_replay_guard"],
    },
    {
        "role_slug": "safety_case_builder",
        "slug": "cognitive_safety_case_builder_bot",
        "label": "Cognitive Safety Case Builder",
        "bot_role": "infrastructure_sub_bot",
        "cognitive_layer": "safety_case_construction",
        "priority": "critical",
        "objective": "Assemble proof-style safety cases for expansions, paper trading, report readiness, and global halt posture.",
        "target_functions": ["formal_verification", "reporting_layer", "global_halt_status", "commands_hygiene"],
    },
    {
        "role_slug": "feedback_credit_assignment",
        "slug": "cognitive_feedback_credit_assignment_bot",
        "label": "Cognitive Feedback Credit Assignment",
        "bot_role": "signal_sub_bot",
        "cognitive_layer": "feedback_credit_assignment",
        "priority": "high",
        "objective": "Attribute improvements and regressions to data, labels, model changes, runtime policy, or market regime shifts.",
        "target_functions": ["model_lifecycle", "decision_provenance", "grade_regression_guard"],
    },
    {
        "role_slug": "agent_market_specialist_router",
        "slug": "cognitive_agent_market_specialist_router_bot",
        "label": "Cognitive Agent Market Specialist Router",
        "bot_role": "infrastructure_sub_bot",
        "cognitive_layer": "agent_market_routing",
        "priority": "high",
        "objective": "Route problems to the best specialist sleeve or infrabot based on evidence, confidence, and resource cost.",
        "target_functions": ["sleeve_masters", "infrastructure_autofix", "operator_cockpit"],
    },
    {
        "role_slug": "grandmaster_cognition_bridge",
        "slug": "cognitive_grandmaster_cognition_bridge_bot",
        "label": "Cognitive Grandmaster Cognition Bridge",
        "bot_role": "infrastructure_sub_bot",
        "cognitive_layer": "grandmaster_cognition_bridge",
        "priority": "critical",
        "objective": "Convert cognitive-control evidence into compact grandmaster-ready decisions with safety and pressure context.",
        "target_functions": ["grand_master_reporting", "system_summary", "platform_control_plane", "operator_cockpit"],
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _version_from_bot_id(bot_id: str) -> int | None:
    match = re.match(r"^brain_refinery_v(?P<version>\d+)", bot_id)
    return int(match.group("version")) if match else None


def _next_available_version(used_versions: set[int], start: int) -> int:
    version = start
    while version in used_versions:
        version += 1
    used_versions.add(version)
    return version


def _slot_kind(bot: dict[str, Any]) -> str:
    return f"{CONTROL_SLUG}_{bot['role_slug']}"


def _assign_bot_ids(rows: list[dict[str, Any]]) -> dict[str, str]:
    existing_by_slot_kind = {
        str(row.get("slot_kind") or ""): str(row.get("bot_id") or "")
        for row in rows
        if isinstance(row, dict) and str(row.get("slot_kind") or "") and str(row.get("bot_id") or "")
    }
    used_versions = {
        version
        for row in rows
        if isinstance(row, dict)
        for version in [_version_from_bot_id(str(row.get("bot_id") or ""))]
        if version is not None
    }
    assigned: dict[str, str] = {}
    for index, bot in enumerate(BOTS):
        slot = _slot_kind(bot)
        if slot in existing_by_slot_kind:
            assigned[slot] = existing_by_slot_kind[slot]
            continue
        desired_version = BASE_VERSION + index
        if desired_version not in used_versions:
            version = desired_version
            used_versions.add(version)
        else:
            version = _next_available_version(used_versions, max(max(used_versions, default=BASE_VERSION - 1) + 1, desired_version))
        assigned[slot] = f"brain_refinery_v{version}_{bot['slug']}"
    return assigned


def _threshold_progress() -> dict[str, Any]:
    return {
        "observations": 0,
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "observations_ready": False,
        "collection_age_days": 0.0,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "days_ready": False,
        "training_ready": False,
    }


def _control_contract(assigned_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": CONTROL_VERSION,
        "new_sleeve_or_subsleeve": {
            "sleeve_family": CONTROL_FAMILY,
            "sleeve_profile": CONTROL_PROFILE,
            "display_name": CONTROL_DISPLAY_NAME,
        },
        "bot_pack_size": len(BOTS),
        "bot_pack_size_rule": "5_to_15_bots_max",
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "cognitive_hot_7d_warm_90d_cold_365d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": 45,
            "capture_mode": "sampled",
            "sample_rate": 0.25,
            "dedupe_required": True,
            "stale_deletion_policy": "distill_then_stage_low_value_cognitive_raw_traces_under_quota_guard",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "sleeve_master_bot_id": assigned_ids.get(f"{CONTROL_SLUG}_planning_orchestrator_master", ""),
        "regression_guard_bot_id": assigned_ids.get(f"{CONTROL_SLUG}_formal_objective_alignment_guard", ""),
        "capacity_check": {
            "active_bot_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
            "runtime_stability_mode": "full_force_buffered",
            "runtime_control_refresh_seconds": 240,
            "queue_policy": "buffered_jsonl_batching",
            "compute_guard_mode": "sustain",
            "global_halt_mode": "downshift_and_distill_before_hard_halt",
            "heavy_reasoning_mode": "cold_lane_only_when_host_pressure_is_clear",
        },
        "cognitive_control_contract": CONTROL_CONTRACT,
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    contract = _control_contract(assigned_ids)
    advanced_contract = {
        "contract_version": "advanced_intelligence_layers_v4",
        "cognitive_control_version": CONTROL_VERSION,
        "capability_pack": CONTROL_SLUG,
        "bot_intelligence_layer": bot["cognitive_layer"],
        "reports_to_sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "critic_guard_bot_id": contract["regression_guard_bot_id"],
        "planning": "cognitive_plan_trace",
        "belief_state": "belief_state_posterior_trace",
        "debate": "multi_agent_debate_score_trace",
        "alignment": "formal_alignment_invariant_trace",
        "strategy_synthesis": "strategy_synthesis_candidate_trace",
        "curriculum": "curriculum_learning_gap_trace",
        "memory_distillation": "memory_distillation_digest_trace",
        "epistemic_budget": "epistemic_risk_budget_trace",
        "safety_case": "safety_case_evidence_trace",
        "resource_budget": "compute_capital_allocation_trace",
    }
    return {
        "bot_id": bot_id,
        "bot_role": bot["bot_role"],
        "active": True,
        "reason": "planned_roster_expansion_slot",
        "weight": 0.0,
        "preference_score": 0.0,
        "quality_score": 0.0,
        "test_accuracy": None,
        "candidate_test_accuracy": None,
        "candidate_quality_score": 0.0,
        "previous_best_accuracy": None,
        "no_improvement_streak": 0,
        "deleted_from_rotation": False,
        "delete_reason": "",
        "promoted": False,
        "promotion_reason": "planned_roster_expansion_slot",
        "model_path": "",
        "log_file": "",
        "candidate_log_file": "",
        "lifecycle_state": "data_collection_only",
        "slot_label": bot["label"],
        "slot_kind": _slot_kind(bot),
        "slot_priority": bot["priority"],
        "slot_objective": bot["objective"],
        "target_functions": list(bot["target_functions"]),
        "preferred_regimes": ["all_weather", "low_pressure", "research_cycle", "regime_shift", "model_uncertainty"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v804_advanced_mesh_metacognitive_state_router_master_bot",
            "brain_refinery_v808_advanced_mesh_self_correction_regression_guard_bot",
            "brain_refinery_v818_advanced_mesh_grandmaster_intelligence_bridge_bot",
        ],
        "data_intake_collections": list(DATA_INTAKES),
        "storage_targets": list(STORAGE_TARGETS),
        "freshness_slo_seconds": 1200,
        "retention_profile": "cognitive_hot_7d_warm_90d_cold_365d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "cognitive_control_observer_until_alignment_and_evidence_thresholds_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "data_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_cognitive_control_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "training_threshold_policy": "eligible_when_minimum_observations_days_guarded_replay_and_alignment_case_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "promotion_blocked_until": "minimum_data_collection_threshold_met",
        "promotion_block_reason": "cognitive_control_collection_only_no_training_yet",
        "data_collection_storage_guarded": True,
        "data_collection_storage_guard_mode": "normal",
        "data_collection_capture_mode": "sampled",
        "data_collection_sample_rate": 0.25,
        "data_collection_max_daily_storage_mb": 45,
        "data_collection_storage_guard_updated_utc": now,
        "storage_pressure_capture_reason": "cognitive_control_distilled_low_volume_collection",
        "data_collection_compute_guard_mode": "sustain",
        "data_collection_resource_guard_reason": "heavy_cognitive_reasoning_off_hot_path",
        "data_collection_max_daily_mb": 45,
        "collected_observation_count": 0,
        "data_collection_last_counted_utc": "",
        "data_collection_observation_rollup_source": "awaiting_first_incremental_rollup",
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "sleeve_profile": CONTROL_PROFILE,
        "sleeve_family": CONTROL_FAMILY,
        "correlation_peer_sleeves": [
            "advanced_intelligence_mesh",
            "research_automation",
            "model_lifecycle",
            "stress_lab",
            "reporting_layer",
            "portfolio_risk_layer",
        ],
        "correlation_dependencies": [
            "decision_provenance",
            "formal_verification",
            "runtime_throttle",
            "global_halt_status",
            "operator_cockpit",
        ],
        "provider_capability_profile": "internal_cognitive_governance_only",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "registry",
            "advanced_intelligence_mesh",
            "decision_provenance",
            "experiment_ledger",
            "health_gates",
            "runtime_throttle",
            "training_diagnostics",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "cognitive_control_has_no_direct_execution_or_broker_dependency",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.7,
            "freshness_slo_seconds": 1200,
            "regression_guard_bot_id": contract["regression_guard_bot_id"],
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": [
            "research_only",
            "collection_only",
            "execution_blocked",
            "paper_only_floor",
            f"sleeve_family:{CONTROL_FAMILY}",
            f"sleeve_profile:{CONTROL_PROFILE}",
            f"capability_pack:{CONTROL_SLUG}",
            f"cognitive_layer:{bot['cognitive_layer']}",
            f"data_floor:{MINIMUM_TRAINING_OBSERVATIONS}",
            "training_after_threshold",
            "global_halt_aware",
            "cognitive_governance",
        ],
        "execution_policy_label": "collection_only_cognitive_control_no_execution",
        "eligible_for_master_vote": False,
        "data_collection_runtime_dependency_profile": "cold_lane_low_pressure_sampled_distilled",
        "founder_bot_id": "brain_refinery_v1",
        "founder_dna_version": "founder_dna_v1",
        "founder_dna_scope": "brain_refinery_v1_to_full_fleet",
        "founder_dna_source": "registry_inferred_full_fleet_contract",
        "founder_dna_confidence": 0.78,
        "founder_dna_traits": [
            "market_data_observation",
            "paper_first_safety",
            "global_halt_awareness",
            "resource_throttle_awareness",
            "training_lineage",
            "decision_explanation_contract",
            "data_collection_before_training",
            "registry_auditable_identity",
            "cognitive_control_governance",
        ],
        "founder_dna_inheritance_mode": "explicit_contract_metadata",
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "lineage_regression_guard": "fail_if_founder_dna_missing_or_stale",
        "lineage_generation": 4,
        "paper_runtime_stability_mode": "full_force_buffered",
        "paper_execution_queue_policy": "buffered_jsonl_batching",
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "paper_trade_lock_required": True,
        "paper_runtime_control_refresh_seconds": 240,
        "capability_pack_version": CONTROL_VERSION,
        "capability_pack_slug": CONTROL_SLUG,
        "capability_pack_display_name": CONTROL_DISPLAY_NAME,
        "cognitive_control_version": CONTROL_VERSION,
        "capability_pack_contract": contract,
        "advanced_intelligence_layer_contract": advanced_contract,
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    signal_inactive = [row for row in inactive if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_inactive = [row for row in inactive if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    control = [row for row in rows if str(row.get("cognitive_control_version") or "") == CONTROL_VERSION]
    summary = dict(registry.get("summary") or {})
    summary.update(
        {
            "total_bots": len(rows),
            "active_bots": len(active),
            "inactive_bots": len(inactive),
            "active_signal_sub_bots": len(signal_active),
            "active_infrastructure_sub_bots": len(infra_active),
            "inactive_signal_sub_bots": len(signal_inactive),
            "inactive_infrastructure_sub_bots": len(infra_inactive),
            "data_collection_active_bots": sum(1 for row in rows if bool(row.get("data_collection_active"))),
            "training_excluded_bots": sum(1 for row in rows if bool(row.get("training_excluded"))),
            "structured_capability_pack_bot_count": len(structured),
            "cognitive_control_plane_bot_count": len(control),
            "latest_cognitive_control_plane": CONTROL_VERSION,
        }
    )
    registry["summary"] = summary


def plan_registry_expansion(registry: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    existing_slot_kinds = {str(row.get("slot_kind") or "") for row in rows}
    assigned_ids = _assign_bot_ids(rows)
    now = _utc_now()
    planned_rows: list[dict[str, Any]] = []
    skipped_existing: list[str] = []
    for bot in BOTS:
        slot = _slot_kind(bot)
        if slot in existing_slot_kinds:
            skipped_existing.append(slot)
            continue
        planned_rows.append(_row_for_bot(bot, assigned_ids[slot], assigned_ids, now))
    return {
        "generated_at_utc": now,
        "cognitive_control_version": CONTROL_VERSION,
        "pack_count": 1,
        "bot_count": len(BOTS),
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped_existing),
        "planned_rows": planned_rows,
        "skipped_existing_slot_kinds": skipped_existing,
        "control_plane": _control_summary(assigned_ids),
    }


def _control_summary(assigned_ids: dict[str, str]) -> dict[str, Any]:
    contract = _control_contract(assigned_ids)
    return {
        "slug": CONTROL_SLUG,
        "display_name": CONTROL_DISPLAY_NAME,
        "sleeve_family": CONTROL_FAMILY,
        "sleeve_profile": CONTROL_PROFILE,
        "objective": "Push the platform into recursive cognitive governance: planning, debate, alignment checks, memory distillation, and safe strategy synthesis.",
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "regression_guard_bot_id": contract["regression_guard_bot_id"],
        "capacity_check": contract["capacity_check"],
        "cognitive_depth": list(CONTROL_CONTRACT["cognitive_depth"]),
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry = _load_json(project_root / "master_bot_registry.json")
    plan = plan_registry_expansion(registry)
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    return {
        "ok": True,
        "generated_at_utc": plan["generated_at_utc"],
        "mode": "dry_run",
        "registry_path": str((project_root / "master_bot_registry.json").resolve()),
        "current_total_bots": len(rows),
        "current_active_bots": sum(1 for row in rows if bool(row.get("active"))),
        "cognitive_control_version": CONTROL_VERSION,
        "pack_count": plan["pack_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "control_plane": plan["control_plane"],
        "cognitive_control_contract": CONTROL_CONTRACT,
        "recommended_apply_command": "./scripts/ops/opsctl.sh cognitive-control-plane --apply --json",
    }


def apply_registry(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    registry = _load_json(registry_path)
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    plan = plan_registry_expansion(registry)
    added_rows = list(plan["planned_rows"])
    backup_path = ""
    if added_rows:
        backup_dir = project_root / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"master_bot_registry_before_cognitive_control_plane_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(registry_path, backup)
        backup_path = str(backup)
        rows.extend(added_rows)
        registry["sub_bots"] = rows
        registry["updated_at_utc"] = _utc_now()
        _refresh_summary(registry)
        _write_json(registry_path, registry)

    payload = build_payload(project_root)
    payload.update(
        {
            "mode": "applied",
            "added_bot_count": len(added_rows),
            "added_bot_ids": [str(row.get("bot_id") or "") for row in added_rows],
            "backup_path": backup_path,
            "new_total_bots": len(rows),
            "new_active_bots": sum(1 for row in rows if bool(row.get("active"))),
        }
    )
    _write_json(
        project_root / "config" / "cognitive_control_plane_v1.json",
        {
            "generated_at_utc": _utc_now(),
            "cognitive_control_version": CONTROL_VERSION,
            "control_plane": payload["control_plane"],
            "cognitive_control_contract": payload["cognitive_control_contract"],
        },
    )
    _write_json(project_root / "governance" / "health" / "cognitive_control_plane_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add a 15-bot cognitive control plane expansion.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = apply_registry(project_root) if args.apply else build_payload(project_root)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "cognitive_control_plane "
            f"mode={payload['mode']} bots={payload['bot_count']} "
            f"planned={payload['planned_bot_count']} added={payload.get('added_bot_count', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
