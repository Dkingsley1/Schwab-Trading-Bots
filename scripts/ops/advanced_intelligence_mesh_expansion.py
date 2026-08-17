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
BASE_VERSION = 804
MESH_VERSION = "advanced_intelligence_mesh_v1"
LABEL_CONTRACT_VERSION = "advanced_mesh_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 5000
MINIMUM_COLLECTION_DAYS = 21
PAPER_RUNTIME_CAPACITY_FLOOR = 700

MESH_SLUG = "advanced_intelligence_mesh"
MESH_DISPLAY_NAME = "Advanced Intelligence Mesh"
MESH_FAMILY = "meta_intelligence"
MESH_PROFILE = "advanced_intelligence_mesh"

DATA_INTAKES = [
    "metacognitive_state_trace",
    "experience_accumulation_memory_trace",
    "causal_counterfactual_evidence_trace",
    "uncertainty_calibration_trace",
    "observer_critic_self_correction_trace",
    "active_learning_query_trace",
    "neuro_symbolic_rule_trace",
    "world_model_scenario_trace",
    "cross_modal_context_embedding_trace",
    "adversarial_robustness_probe_trace",
    "compute_capital_allocation_trace",
    "hypothesis_retrieval_context_trace",
    "explainability_decision_trace",
    "safety_invariant_verification_trace",
    "grandmaster_intelligence_bridge_trace",
]

STORAGE_TARGETS = [
    "governance/advanced_intelligence_mesh",
    "governance/advanced_intelligence_mesh/memory",
    "governance/advanced_intelligence_mesh/critic",
    "governance/advanced_intelligence_mesh/explainability",
    "governance/health/advanced_intelligence_mesh",
    "data/jsonl_link.sqlite3",
]

REQUIRED_LABELS = [
    "intelligence_layer",
    "evidence_confidence_bucket",
    "uncertainty_bucket",
    "self_correction_action",
    "resource_budget_bucket",
    "safety_invariant_status",
]

ADVANCED_MESH_CONTRACT = {
    "contract_version": MESH_VERSION,
    "purpose": "higher_order_reasoning_mesh_for_cross_sleeve_memory_critic_and_resource_aware_intelligence",
    "intelligence_depth": [
        "metacognition",
        "experience_memory",
        "causal_counterfactual_reasoning",
        "uncertainty_calibration",
        "self_correction_critic_loop",
        "active_learning",
        "neuro_symbolic_reasoning",
        "world_model_simulation",
        "cross_modal_context_fusion",
        "adversarial_robustness",
        "resource_aware_reasoning",
        "retrieval_augmented_hypothesis_memory",
        "explainability_traceability",
        "safety_invariant_verification",
        "grandmaster_bridge",
    ],
    "graduation_policy": "collection_only_until_minimum_evidence_and_guarded_replay",
    "global_halt_contract": "expansion_pressure_aware_soft_cap_before_hard_halt",
    "paper_lock_contract": "paper_trade_lock_required_before_any_execution_path",
}

BOTS: list[dict[str, Any]] = [
    {
        "role_slug": "metacognitive_state_router_master",
        "slug": "advanced_mesh_metacognitive_state_router_master_bot",
        "label": "Advanced Mesh Metacognitive State Router Master",
        "bot_role": "infrastructure_sub_bot",
        "intelligence_layer": "metacognition",
        "priority": "critical",
        "objective": "Route the whole intelligence stack by confidence, regime, uncertainty, pressure, and evidence age.",
        "target_functions": ["sleeve_master", "market_regime_router", "operator_cockpit", "grand_master_reporting"],
    },
    {
        "role_slug": "experience_memory_consolidator",
        "slug": "advanced_mesh_experience_memory_consolidator_bot",
        "label": "Advanced Mesh Experience Memory Consolidator",
        "bot_role": "infrastructure_sub_bot",
        "intelligence_layer": "memory",
        "priority": "critical",
        "objective": "Compress useful observations into reusable experience memory without keeping noisy raw traces forever.",
        "target_functions": ["experience_accumulation_memory", "data_retention", "feature_store"],
    },
    {
        "role_slug": "causal_counterfactual_evaluator",
        "slug": "advanced_mesh_causal_counterfactual_evaluator_bot",
        "label": "Advanced Mesh Causal Counterfactual Evaluator",
        "bot_role": "signal_sub_bot",
        "intelligence_layer": "causal_reasoning",
        "priority": "high",
        "objective": "Separate correlation from plausible cause by scoring counterfactual and natural-experiment evidence.",
        "target_functions": ["causal_intervention_ledger", "research_pipeline", "model_lifecycle"],
    },
    {
        "role_slug": "uncertainty_calibration_guard",
        "slug": "advanced_mesh_uncertainty_calibration_guard_bot",
        "label": "Advanced Mesh Uncertainty Calibration Guard",
        "bot_role": "infrastructure_sub_bot",
        "intelligence_layer": "uncertainty",
        "priority": "critical",
        "objective": "Track confidence drift, calibration error, conformal interval width, and abstention quality.",
        "target_functions": ["calibration_control", "multiple_testing_guard", "training_quality"],
    },
    {
        "role_slug": "self_correction_regression_guard",
        "slug": "advanced_mesh_self_correction_regression_guard_bot",
        "label": "Advanced Mesh Self-Correction Regression Guard",
        "bot_role": "infrastructure_sub_bot",
        "intelligence_layer": "critic",
        "priority": "critical",
        "objective": "Audit whether critic-loop recommendations improved behavior or created new regressions.",
        "target_functions": ["regression_guard", "grade_regression_guard", "decision_provenance"],
    },
    {
        "role_slug": "active_learning_query_planner",
        "slug": "advanced_mesh_active_learning_query_planner_bot",
        "label": "Advanced Mesh Active Learning Query Planner",
        "bot_role": "signal_sub_bot",
        "intelligence_layer": "active_learning",
        "priority": "high",
        "objective": "Prioritize the next observations, replays, and labels that would most improve weak sleeves.",
        "target_functions": ["coverage_gap_closer", "collector_observation_rollup", "training_requalification"],
    },
    {
        "role_slug": "neuro_symbolic_rule_bridge",
        "slug": "advanced_mesh_neuro_symbolic_rule_bridge_bot",
        "label": "Advanced Mesh Neuro-Symbolic Rule Bridge",
        "bot_role": "signal_sub_bot",
        "intelligence_layer": "neuro_symbolic",
        "priority": "high",
        "objective": "Link learned signals with explicit risk, execution, labeling, and safety rules.",
        "target_functions": ["label_taxonomy", "risk_service", "formal_verification"],
    },
    {
        "role_slug": "world_model_market_simulator",
        "slug": "advanced_mesh_world_model_market_simulator_bot",
        "label": "Advanced Mesh World Model Market Simulator",
        "bot_role": "signal_sub_bot",
        "intelligence_layer": "world_model",
        "priority": "high",
        "objective": "Build lightweight scenario beliefs that connect event, microstructure, risk, and stress-lab evidence.",
        "target_functions": ["stress_lab", "golden_replay_guard", "market_regime_router"],
    },
    {
        "role_slug": "cross_modal_context_encoder",
        "slug": "advanced_mesh_cross_modal_context_encoder_bot",
        "label": "Advanced Mesh Cross-Modal Context Encoder",
        "bot_role": "infrastructure_sub_bot",
        "intelligence_layer": "context_fusion",
        "priority": "high",
        "objective": "Unify transcripts, calendars, prices, flows, reports, and replay traces into typed context.",
        "target_functions": ["event_intelligence", "research_automation", "feature_store"],
    },
    {
        "role_slug": "adversarial_robustness_guard",
        "slug": "advanced_mesh_adversarial_robustness_guard_bot",
        "label": "Advanced Mesh Adversarial Robustness Guard",
        "bot_role": "infrastructure_sub_bot",
        "intelligence_layer": "robustness",
        "priority": "critical",
        "objective": "Probe source disagreement, spoofed signals, bad labels, and brittle model behavior.",
        "target_functions": ["source_verification", "provider_adapter_verification", "regression_guard"],
    },
    {
        "role_slug": "resource_aware_reasoning_scheduler",
        "slug": "advanced_mesh_resource_aware_reasoning_scheduler_bot",
        "label": "Advanced Mesh Resource-Aware Reasoning Scheduler",
        "bot_role": "infrastructure_sub_bot",
        "intelligence_layer": "capacity",
        "priority": "critical",
        "objective": "Schedule heavy reasoning, replay, and MLX work around live loops, swap pressure, and storage drains.",
        "target_functions": ["runtime_throttle", "memory_efficiency", "swap_pressure_governor", "ingestion_storage_control"],
    },
    {
        "role_slug": "hypothesis_retrieval_rag",
        "slug": "advanced_mesh_hypothesis_retrieval_rag_bot",
        "label": "Advanced Mesh Hypothesis Retrieval RAG Bot",
        "bot_role": "infrastructure_sub_bot",
        "intelligence_layer": "retrieval_memory",
        "priority": "medium",
        "objective": "Retrieve prior hypotheses, reports, replays, and paper notes before new experiments are admitted.",
        "target_functions": ["research_pipeline", "experiment_ledger", "reporting_layer"],
    },
    {
        "role_slug": "explainability_trace_builder",
        "slug": "advanced_mesh_explainability_trace_builder_bot",
        "label": "Advanced Mesh Explainability Trace Builder",
        "bot_role": "infrastructure_sub_bot",
        "intelligence_layer": "explainability",
        "priority": "high",
        "objective": "Build concise evidence chains from raw inputs to sleeve decisions, reports, and guard outcomes.",
        "target_functions": ["decision_provenance", "reporting_layer", "operator_cockpit"],
    },
    {
        "role_slug": "safety_invariant_verifier",
        "slug": "advanced_mesh_safety_invariant_verifier_bot",
        "label": "Advanced Mesh Safety Invariant Verifier",
        "bot_role": "infrastructure_sub_bot",
        "intelligence_layer": "formal_safety",
        "priority": "critical",
        "objective": "Verify paper lock, no-live-execution, no-lookahead, retention, and halt invariants after expansions.",
        "target_functions": ["formal_verification", "paper_trade_lock", "global_halt_status", "commands_hygiene"],
    },
    {
        "role_slug": "grandmaster_intelligence_bridge",
        "slug": "advanced_mesh_grandmaster_intelligence_bridge_bot",
        "label": "Advanced Mesh Grandmaster Intelligence Bridge",
        "bot_role": "infrastructure_sub_bot",
        "intelligence_layer": "grandmaster_bridge",
        "priority": "critical",
        "objective": "Summarize mesh intelligence into master/grand master ready packets with confidence and pressure context.",
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
    return f"{MESH_SLUG}_{bot['role_slug']}"


def _assign_bot_ids(rows: list[dict[str, Any]]) -> dict[str, str]:
    existing_by_slot_kind = {
        str(row.get("slot_kind") or ""): str(row.get("bot_id") or "")
        for row in rows
        if isinstance(row, dict) and str(row.get("slot_kind") or "") and str(row.get("bot_id") or "")
    }
    used_ids = {str(row.get("bot_id") or "") for row in rows if isinstance(row, dict)}
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
        bot_id = f"brain_refinery_v{version}_{bot['slug']}"
        used_ids.add(bot_id)
        assigned[slot] = bot_id
    return assigned


def _training_threshold_progress() -> dict[str, Any]:
    return {
        "observations": 0,
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "observations_ready": False,
        "collection_age_days": 0.0,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "days_ready": False,
        "training_ready": False,
    }


def _mesh_contract(assigned_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": MESH_VERSION,
        "new_sleeve_or_subsleeve": {
            "sleeve_family": MESH_FAMILY,
            "sleeve_profile": MESH_PROFILE,
            "display_name": MESH_DISPLAY_NAME,
        },
        "bot_pack_size": len(BOTS),
        "bot_pack_size_rule": "5_to_15_bots_max",
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "mesh_hot_14d_warm_120d_cold_365d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": 60,
            "capture_mode": "sampled",
            "sample_rate": 0.35,
            "dedupe_required": True,
            "stale_deletion_policy": "compress_memory_then_stage_low_value_raw_traces_under_quota_guard",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "sleeve_master_bot_id": assigned_ids.get(f"{MESH_SLUG}_metacognitive_state_router_master", ""),
        "regression_guard_bot_id": assigned_ids.get(f"{MESH_SLUG}_self_correction_regression_guard", ""),
        "capacity_check": {
            "active_bot_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
            "runtime_stability_mode": "full_force_buffered",
            "runtime_control_refresh_seconds": 240,
            "queue_policy": "buffered_jsonl_batching",
            "compute_guard_mode": "soft_cap",
            "global_halt_mode": "soft_cap_and_backpressure_before_hard_halt",
            "heavy_reasoning_mode": "off_hot_path_low_pressure_only",
        },
        "advanced_mesh_contract": ADVANCED_MESH_CONTRACT,
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    contract = _mesh_contract(assigned_ids)
    label_tags = [
        "research_only",
        "collection_only",
        "execution_blocked",
        "paper_only_floor",
        f"sleeve_family:{MESH_FAMILY}",
        f"sleeve_profile:{MESH_PROFILE}",
        f"capability_pack:{MESH_SLUG}",
        f"intelligence_layer:{bot['intelligence_layer']}",
        f"data_floor:{MINIMUM_TRAINING_OBSERVATIONS}",
        "training_after_threshold",
        "global_halt_aware",
        "meta_intelligence",
    ]
    advanced_contract = {
        "contract_version": "advanced_intelligence_layers_v3",
        "mesh_version": MESH_VERSION,
        "capability_pack": MESH_SLUG,
        "bot_intelligence_layer": bot["intelligence_layer"],
        "reports_to_sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "critic_guard_bot_id": contract["regression_guard_bot_id"],
        "memory": "experience_accumulation_memory_trace",
        "metacognition": "metacognitive_state_trace",
        "causal_evidence": "causal_counterfactual_evidence_trace",
        "uncertainty": "uncertainty_calibration_trace",
        "critic_loop": "observer_critic_self_correction_trace",
        "world_model": "world_model_scenario_trace",
        "context_fusion": "cross_modal_context_embedding_trace",
        "safety": "safety_invariant_verification_trace",
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
        "preferred_regimes": ["all_weather", "model_uncertainty", "regime_shift", "research_cycle", "low_pressure"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v617_experience_accumulation_memory_design_bot",
            "brain_refinery_v757_compute_capital_allocator_bot",
            "brain_refinery_v761_semantic_feature_ontology_harmonizer_bot",
        ],
        "data_intake_collections": list(DATA_INTAKES),
        "storage_targets": list(STORAGE_TARGETS),
        "freshness_slo_seconds": 900,
        "retention_profile": "mesh_hot_14d_warm_120d_cold_365d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "advanced_mesh_observer_until_minimum_evidence",
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
        "training_exclusion_reason": "collecting_mesh_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "training_threshold_policy": "eligible_when_minimum_observations_days_and_guarded_replay_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "promotion_blocked_until": "minimum_data_collection_threshold_met",
        "promotion_block_reason": "advanced_mesh_collection_only_no_training_yet",
        "data_collection_storage_guarded": True,
        "data_collection_storage_guard_mode": "normal",
        "data_collection_capture_mode": "sampled",
        "data_collection_sample_rate": 0.35,
        "data_collection_max_daily_storage_mb": 60,
        "data_collection_storage_guard_updated_utc": now,
        "storage_pressure_capture_reason": "advanced_mesh_budgeted_memory_compressed_collection",
        "data_collection_compute_guard_mode": "soft_cap",
        "data_collection_resource_guard_reason": "higher_order_reasoning_off_hot_path",
        "data_collection_max_daily_mb": 60,
        "collected_observation_count": 0,
        "data_collection_last_counted_utc": "",
        "data_collection_observation_rollup_source": "awaiting_first_incremental_rollup",
        "data_collection_threshold_progress": _training_threshold_progress(),
        "data_collection_training_ready": False,
        "sleeve_profile": MESH_PROFILE,
        "sleeve_family": MESH_FAMILY,
        "correlation_peer_sleeves": [
            "execution_intelligence",
            "portfolio_risk_layer",
            "event_intelligence",
            "market_microstructure",
            "research_automation",
            "stress_lab",
            "model_lifecycle",
            "reporting_layer",
        ],
        "correlation_dependencies": [
            "decision_provenance",
            "model_lifecycle",
            "research_pipeline",
            "runtime_throttle",
            "global_halt_status",
        ],
        "provider_capability_profile": "internal_context_and_governance_mesh",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "registry",
            "decision_provenance",
            "health_gates",
            "training_diagnostics",
            "experiment_ledger",
            "report_quality_guard",
            "stress_replay_artifacts",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "meta_intelligence_no_direct_execution_or_broker_dependency",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.65,
            "freshness_slo_seconds": 900,
            "regression_guard_bot_id": contract["regression_guard_bot_id"],
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": label_tags,
        "execution_policy_label": "collection_only_meta_intelligence_no_execution",
        "eligible_for_master_vote": False,
        "data_collection_runtime_dependency_profile": "low_pressure_sampled_async_memory_compressed",
        "founder_bot_id": "brain_refinery_v1",
        "founder_dna_version": "founder_dna_v1",
        "founder_dna_scope": "brain_refinery_v1_to_full_fleet",
        "founder_dna_source": "registry_inferred_full_fleet_contract",
        "founder_dna_confidence": 0.76,
        "founder_dna_traits": [
            "market_data_observation",
            "paper_first_safety",
            "global_halt_awareness",
            "resource_throttle_awareness",
            "training_lineage",
            "decision_explanation_contract",
            "data_collection_before_training",
            "registry_auditable_identity",
            "advanced_meta_intelligence_governance",
        ],
        "founder_dna_inheritance_mode": "explicit_contract_metadata",
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "lineage_regression_guard": "fail_if_founder_dna_missing_or_stale",
        "lineage_generation": 3,
        "paper_runtime_stability_mode": "full_force_buffered",
        "paper_execution_queue_policy": "buffered_jsonl_batching",
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "paper_trade_lock_required": True,
        "paper_runtime_control_refresh_seconds": 240,
        "capability_pack_version": MESH_VERSION,
        "capability_pack_slug": MESH_SLUG,
        "capability_pack_display_name": MESH_DISPLAY_NAME,
        "advanced_mesh_version": MESH_VERSION,
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
    mesh = [row for row in rows if str(row.get("advanced_mesh_version") or "") == MESH_VERSION]
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
            "advanced_intelligence_mesh_bot_count": len(mesh),
            "latest_advanced_intelligence_mesh": MESH_VERSION,
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
        "advanced_mesh_version": MESH_VERSION,
        "pack_count": 1,
        "bot_count": len(BOTS),
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped_existing),
        "planned_rows": planned_rows,
        "skipped_existing_slot_kinds": skipped_existing,
        "mesh": _mesh_summary(assigned_ids),
    }


def _mesh_summary(assigned_ids: dict[str, str]) -> dict[str, Any]:
    contract = _mesh_contract(assigned_ids)
    return {
        "slug": MESH_SLUG,
        "display_name": MESH_DISPLAY_NAME,
        "sleeve_family": MESH_FAMILY,
        "sleeve_profile": MESH_PROFILE,
        "objective": "Push the platform into higher-order self-awareness, memory, causal reasoning, critic-loop correction, and resource-aware intelligence.",
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "regression_guard_bot_id": contract["regression_guard_bot_id"],
        "capacity_check": contract["capacity_check"],
        "intelligence_depth": list(ADVANCED_MESH_CONTRACT["intelligence_depth"]),
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
        "advanced_mesh_version": MESH_VERSION,
        "pack_count": plan["pack_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "mesh": plan["mesh"],
        "advanced_mesh_contract": ADVANCED_MESH_CONTRACT,
        "recommended_apply_command": "./scripts/ops/opsctl.sh advanced-intelligence-mesh --apply --json",
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
        backup = backup_dir / f"master_bot_registry_before_advanced_intelligence_mesh_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
        project_root / "config" / "advanced_intelligence_mesh_v1.json",
        {
            "generated_at_utc": _utc_now(),
            "advanced_mesh_version": MESH_VERSION,
            "mesh": payload["mesh"],
            "advanced_mesh_contract": payload["advanced_mesh_contract"],
        },
    )
    _write_json(project_root / "governance" / "health" / "advanced_intelligence_mesh_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add a 15-bot advanced intelligence mesh expansion.")
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
            "advanced_intelligence_mesh "
            f"mode={payload['mode']} bots={payload['bot_count']} "
            f"planned={payload['planned_bot_count']} added={payload.get('added_bot_count', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
