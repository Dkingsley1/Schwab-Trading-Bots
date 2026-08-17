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
BASE_VERSION = 879
KERNEL_VERSION = "adaptive_intelligence_kernel_v1"
LABEL_CONTRACT_VERSION = "adaptive_kernel_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 10000
MINIMUM_COLLECTION_DAYS = 35
PAPER_RUNTIME_CAPACITY_FLOOR = 700

KERNEL_SLUG = "adaptive_intelligence_kernel"
KERNEL_DISPLAY_NAME = "Adaptive Intelligence Kernel"
KERNEL_FAMILY = "adaptive_meta_learning"
KERNEL_PROFILE = "adaptive_intelligence_kernel"

DATA_INTAKES = [
    "meta_learning_update_trace",
    "continual_learning_rehearsal_trace",
    "catastrophic_forgetting_probe_trace",
    "regime_transfer_evidence_trace",
    "self_supervised_state_embedding_trace",
    "causal_representation_trace",
    "bayesian_model_router_trace",
    "meta_gradient_reward_trace",
    "operator_preference_alignment_trace",
    "tool_reliability_observation_trace",
    "teacher_committee_distillation_trace",
    "rl_curriculum_environment_trace",
    "simulation_to_reality_gap_trace",
    "data_valuation_priority_trace",
    "grandmaster_adaptive_kernel_packet_trace",
]

STORAGE_TARGETS = [
    "governance/adaptive_intelligence_kernel",
    "governance/adaptive_intelligence_kernel/meta_learning",
    "governance/adaptive_intelligence_kernel/rehearsal_memory",
    "governance/adaptive_intelligence_kernel/sim_to_real",
    "governance/health/adaptive_intelligence_kernel",
    "data/jsonl_link.sqlite3",
]

REQUIRED_LABELS = [
    "adaptation_quality_bucket",
    "transfer_learning_regime_pair",
    "forgetting_risk_bucket",
    "state_embedding_stability",
    "causal_representation_confidence",
    "sim_to_real_gap_bucket",
    "operator_alignment_status",
]

KERNEL_CONTRACT = {
    "contract_version": KERNEL_VERSION,
    "purpose": "adaptive_meta_learning_kernel_for_continual_learning_transfer_safety_and_grandmaster_intelligence_routing",
    "adaptive_depth": [
        "online_meta_learning",
        "continual_learning_rehearsal",
        "catastrophic_forgetting_control",
        "regime_transfer_learning",
        "self_supervised_market_state_encoding",
        "causal_representation_learning",
        "bayesian_model_routing",
        "meta_gradient_reward_shaping",
        "operator_preference_alignment",
        "tool_reliability_auditing",
        "teacher_committee_distillation",
        "rl_environment_curriculum",
        "simulation_to_reality_gap_detection",
        "data_valuation",
        "grandmaster_kernel_bridge",
    ],
    "graduation_policy": "collection_only_until_long_horizon_replay_forgetting_and_alignment_guards_clear",
    "global_halt_contract": "adaptive_kernel_must_downshift_or_distill_before_hard_global_halt",
    "paper_lock_contract": "paper_trade_lock_required_before_any_execution_path",
    "resource_contract": "sampled_observer_mode_with_heavy_adaptation_off_hot_path",
}

BOTS: list[dict[str, Any]] = [
    {
        "role_slug": "online_meta_learning_master",
        "slug": "adaptive_kernel_online_meta_learning_master_bot",
        "label": "Adaptive Kernel Online Meta-Learning Master",
        "bot_role": "infrastructure_sub_bot",
        "adaptive_layer": "online_meta_learning",
        "priority": "critical",
        "objective": "Track which model, label, regime, and throttle changes actually improve the platform over time.",
        "target_functions": ["model_lifecycle", "operator_cockpit", "platform_control_plane", "grand_master_reporting"],
    },
    {
        "role_slug": "regime_transfer_learning_bridge",
        "slug": "adaptive_kernel_regime_transfer_learning_bridge_bot",
        "label": "Adaptive Kernel Regime Transfer Learning Bridge",
        "bot_role": "signal_sub_bot",
        "adaptive_layer": "regime_transfer_learning",
        "priority": "high",
        "objective": "Score when lessons from one regime or sleeve can transfer safely into another without overfitting.",
        "target_functions": ["regime_control_plane", "training_requalification", "strategy_coverage"],
    },
    {
        "role_slug": "continual_rehearsal_memory_buffer",
        "slug": "adaptive_kernel_continual_rehearsal_memory_buffer_bot",
        "label": "Adaptive Kernel Continual Rehearsal Memory Buffer",
        "bot_role": "infrastructure_sub_bot",
        "adaptive_layer": "continual_learning_rehearsal",
        "priority": "critical",
        "objective": "Keep a compact rehearsal set so new learning does not erase hard-won older behavior.",
        "target_functions": ["experience_accumulation_memory", "data_retention", "training_runtime_control"],
    },
    {
        "role_slug": "catastrophic_forgetting_guard",
        "slug": "adaptive_kernel_catastrophic_forgetting_guard_bot",
        "label": "Adaptive Kernel Catastrophic Forgetting Guard",
        "bot_role": "infrastructure_sub_bot",
        "adaptive_layer": "forgetting_control",
        "priority": "critical",
        "objective": "Detect when a retrain or expansion improves a slice while degrading legacy market behavior.",
        "target_functions": ["golden_replay_guard", "grade_regression_guard", "model_lifecycle"],
    },
    {
        "role_slug": "self_supervised_state_encoder",
        "slug": "adaptive_kernel_self_supervised_state_encoder_bot",
        "label": "Adaptive Kernel Self-Supervised State Encoder",
        "bot_role": "signal_sub_bot",
        "adaptive_layer": "market_state_representation",
        "priority": "high",
        "objective": "Learn compact market-state embeddings from unlabeled events, prices, flows, reports, and health surfaces.",
        "target_functions": ["feature_store", "event_intelligence", "advanced_intelligence_mesh"],
    },
    {
        "role_slug": "causal_representation_disentangler",
        "slug": "adaptive_kernel_causal_representation_disentangler_bot",
        "label": "Adaptive Kernel Causal Representation Disentangler",
        "bot_role": "signal_sub_bot",
        "adaptive_layer": "causal_representation",
        "priority": "high",
        "objective": "Separate durable causes from coincidental features before labels or strategies are promoted.",
        "target_functions": ["causal_counterfactual_evidence", "multiple_testing_guard", "feature_quality_layer"],
    },
    {
        "role_slug": "bayesian_model_router",
        "slug": "adaptive_kernel_bayesian_model_router_bot",
        "label": "Adaptive Kernel Bayesian Model Router",
        "bot_role": "infrastructure_sub_bot",
        "adaptive_layer": "bayesian_model_routing",
        "priority": "critical",
        "objective": "Route sleeves to model families by uncertainty, evidence strength, and downside of being wrong.",
        "target_functions": ["calibration_control", "model_lifecycle", "sleeve_master"],
    },
    {
        "role_slug": "meta_gradient_reward_guard",
        "slug": "adaptive_kernel_meta_gradient_reward_guard_bot",
        "label": "Adaptive Kernel Meta-Gradient Reward Guard",
        "bot_role": "infrastructure_sub_bot",
        "adaptive_layer": "reward_shaping_safety",
        "priority": "critical",
        "objective": "Check that adaptive rewards improve long-run behavior without teaching shortcut or unsafe incentives.",
        "target_functions": ["formal_verification", "policy_simulation_sandbox", "regression_guard"],
    },
    {
        "role_slug": "operator_preference_alignment_recorder",
        "slug": "adaptive_kernel_operator_preference_alignment_recorder_bot",
        "label": "Adaptive Kernel Operator Preference Alignment Recorder",
        "bot_role": "infrastructure_sub_bot",
        "adaptive_layer": "operator_alignment",
        "priority": "high",
        "objective": "Record operator choices as bounded preferences for safety, reporting, throttle, and research-priority behavior.",
        "target_functions": ["operator_cockpit", "decision_provenance", "system_summary"],
    },
    {
        "role_slug": "tool_reliability_auditor",
        "slug": "adaptive_kernel_tool_reliability_auditor_bot",
        "label": "Adaptive Kernel Tool Reliability Auditor",
        "bot_role": "infrastructure_sub_bot",
        "adaptive_layer": "tool_reliability",
        "priority": "critical",
        "objective": "Score command, collector, report, broker, and storage tools by reliability before automation depends on them.",
        "target_functions": ["command_validity", "collector_contracts", "report_quality_guard", "storage_resilience"],
    },
    {
        "role_slug": "teacher_committee_distillation_router",
        "slug": "adaptive_kernel_teacher_committee_distillation_router_bot",
        "label": "Adaptive Kernel Teacher Committee Distillation Router",
        "bot_role": "infrastructure_sub_bot",
        "adaptive_layer": "teacher_committee_distillation",
        "priority": "critical",
        "objective": "Route weak bots to teacher committees and distill disagreement into compact, trainable lessons.",
        "target_functions": ["teacher_quality_guard", "bot_quality_autopilot", "supportability_control"],
    },
    {
        "role_slug": "rl_curriculum_environment_builder",
        "slug": "adaptive_kernel_rl_curriculum_environment_builder_bot",
        "label": "Adaptive Kernel RL Curriculum Environment Builder",
        "bot_role": "signal_sub_bot",
        "adaptive_layer": "rl_curriculum",
        "priority": "high",
        "objective": "Design safe paper-only learning environments from historical, synthetic, and stress-lab scenarios.",
        "target_functions": ["stress_lab", "golden_replay_guard", "cognitive_control_plane"],
    },
    {
        "role_slug": "simulation_to_reality_gap_detector",
        "slug": "adaptive_kernel_simulation_to_reality_gap_detector_bot",
        "label": "Adaptive Kernel Simulation-to-Reality Gap Detector",
        "bot_role": "infrastructure_sub_bot",
        "adaptive_layer": "sim_to_real_gap",
        "priority": "critical",
        "objective": "Detect when replay, synthetic, or paper behavior diverges from live-observable market conditions.",
        "target_functions": ["paper_execution_calibration", "execution_lab", "stress_lab", "live_runtime_separation"],
    },
    {
        "role_slug": "adaptive_data_valuation_prioritizer",
        "slug": "adaptive_kernel_data_valuation_prioritizer_bot",
        "label": "Adaptive Kernel Data Valuation Prioritizer",
        "bot_role": "infrastructure_sub_bot",
        "adaptive_layer": "data_valuation",
        "priority": "high",
        "objective": "Spend collection, storage, and labeling budget on the highest marginal-value observations.",
        "target_functions": ["ingestion_priority_queue", "data_collection_observation_rollup", "storage_tier_policy"],
    },
    {
        "role_slug": "grandmaster_adaptive_kernel_bridge",
        "slug": "adaptive_kernel_grandmaster_adaptive_kernel_bridge_bot",
        "label": "Adaptive Kernel Grandmaster Adaptive Kernel Bridge",
        "bot_role": "infrastructure_sub_bot",
        "adaptive_layer": "grandmaster_kernel_bridge",
        "priority": "critical",
        "objective": "Summarize adaptive learning posture into Grand Master packets with confidence, cost, and safety context.",
        "target_functions": ["grand_master_reporting", "coordination_intelligence", "operator_cockpit", "system_summary"],
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
    return f"{KERNEL_SLUG}_{bot['role_slug']}"


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


def _kernel_contract(assigned_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": KERNEL_VERSION,
        "new_sleeve_or_subsleeve": {
            "sleeve_family": KERNEL_FAMILY,
            "sleeve_profile": KERNEL_PROFILE,
            "display_name": KERNEL_DISPLAY_NAME,
        },
        "bot_pack_size": len(BOTS),
        "bot_pack_size_rule": "5_to_15_bots_max",
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "adaptive_kernel_hot_7d_warm_90d_cold_365d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": 40,
            "capture_mode": "sampled_distilled",
            "sample_rate": 0.20,
            "dedupe_required": True,
            "stale_deletion_policy": "distill_rehearsal_memory_then_stage_low_value_raw_traces_under_quota_guard",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "sleeve_master_bot_id": assigned_ids.get(f"{KERNEL_SLUG}_online_meta_learning_master", ""),
        "regression_guard_bot_id": assigned_ids.get(f"{KERNEL_SLUG}_catastrophic_forgetting_guard", ""),
        "capacity_check": {
            "active_bot_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
            "runtime_stability_mode": "adaptive_ready_with_followups",
            "runtime_control_refresh_seconds": 300,
            "queue_policy": "buffered_jsonl_batching",
            "compute_guard_mode": "cold_lane_preferred",
            "global_halt_mode": "adaptive_downshift_before_hard_halt",
            "heavy_reasoning_mode": "off_hot_path_low_pressure_or_simulation_window_only",
        },
        "adaptive_kernel_contract": KERNEL_CONTRACT,
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    contract = _kernel_contract(assigned_ids)
    advanced_contract = {
        "contract_version": "advanced_intelligence_layers_v5",
        "adaptive_intelligence_kernel_version": KERNEL_VERSION,
        "capability_pack": KERNEL_SLUG,
        "bot_intelligence_layer": bot["adaptive_layer"],
        "reports_to_sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "critic_guard_bot_id": contract["regression_guard_bot_id"],
        "meta_learning": "meta_learning_update_trace",
        "continual_memory": "continual_learning_rehearsal_trace",
        "forgetting_guard": "catastrophic_forgetting_probe_trace",
        "transfer_learning": "regime_transfer_evidence_trace",
        "state_representation": "self_supervised_state_embedding_trace",
        "causal_representation": "causal_representation_trace",
        "bayesian_routing": "bayesian_model_router_trace",
        "sim_to_real": "simulation_to_reality_gap_trace",
        "data_valuation": "data_valuation_priority_trace",
        "operator_alignment": "operator_preference_alignment_trace",
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
        "preferred_regimes": ["all_weather", "regime_shift", "model_uncertainty", "research_cycle", "low_pressure"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v804_advanced_mesh_metacognitive_state_router_master_bot",
            "brain_refinery_v819_cognitive_planning_orchestrator_master_bot",
            "brain_refinery_v849_coordination_lineage_genome_mapper_bot",
        ],
        "data_intake_collections": list(DATA_INTAKES),
        "storage_targets": list(STORAGE_TARGETS),
        "freshness_slo_seconds": 1200,
        "retention_profile": "adaptive_kernel_hot_7d_warm_90d_cold_365d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "adaptive_kernel_observer_until_forgetting_transfer_and_alignment_thresholds_clear",
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
        "training_exclusion_reason": "collecting_adaptive_kernel_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "training_threshold_policy": "eligible_when_observation_days_forgetting_replay_transfer_and_alignment_guards_clear",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "promotion_blocked_until": "minimum_data_collection_threshold_met",
        "promotion_block_reason": "adaptive_kernel_collection_only_no_training_yet",
        "data_collection_storage_guarded": True,
        "data_collection_storage_guard_mode": "normal",
        "data_collection_capture_mode": "sampled_distilled",
        "data_collection_sample_rate": 0.20,
        "data_collection_max_daily_storage_mb": 40,
        "data_collection_storage_guard_updated_utc": now,
        "storage_pressure_capture_reason": "adaptive_kernel_distilled_low_volume_collection",
        "data_collection_compute_guard_mode": "cold_lane_preferred",
        "data_collection_resource_guard_reason": "adaptive_kernel_heavy_learning_off_hot_path",
        "data_collection_max_daily_mb": 40,
        "collected_observation_count": 0,
        "data_collection_last_counted_utc": "",
        "data_collection_observation_rollup_source": "awaiting_first_incremental_rollup",
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "sleeve_profile": KERNEL_PROFILE,
        "sleeve_family": KERNEL_FAMILY,
        "correlation_peer_sleeves": [
            "advanced_intelligence_mesh",
            "cognitive_control_plane",
            "recursive_research_foundry",
            "coordination_intelligence",
            "model_lifecycle",
            "stress_lab",
            "execution_intelligence",
        ],
        "correlation_dependencies": [
            "operator_cockpit",
            "model_lifecycle",
            "golden_replay_guard",
            "runtime_throttle",
            "global_halt_status",
            "training_quality",
        ],
        "provider_capability_profile": "internal_adaptive_governance_only",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "registry",
            "advanced_intelligence_mesh",
            "cognitive_control_plane",
            "coordination_intelligence",
            "decision_provenance",
            "experiment_ledger",
            "health_gates",
            "training_diagnostics",
            "stress_replay_artifacts",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "adaptive_kernel_has_no_direct_execution_or_broker_dependency",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.72,
            "freshness_slo_seconds": 1200,
            "regression_guard_bot_id": contract["regression_guard_bot_id"],
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": [
            "research_only",
            "collection_only",
            "execution_blocked",
            "paper_only_floor",
            f"sleeve_family:{KERNEL_FAMILY}",
            f"sleeve_profile:{KERNEL_PROFILE}",
            f"capability_pack:{KERNEL_SLUG}",
            f"adaptive_layer:{bot['adaptive_layer']}",
            f"data_floor:{MINIMUM_TRAINING_OBSERVATIONS}",
            "training_after_threshold",
            "global_halt_aware",
            "adaptive_meta_learning",
        ],
        "execution_policy_label": "collection_only_adaptive_kernel_no_execution",
        "eligible_for_master_vote": False,
        "data_collection_runtime_dependency_profile": "cold_lane_low_pressure_sampled_distilled",
        "founder_bot_id": "brain_refinery_v1",
        "founder_dna_version": "founder_dna_v1",
        "founder_dna_scope": "brain_refinery_v1_to_full_fleet",
        "founder_dna_source": "registry_inferred_full_fleet_contract",
        "founder_dna_confidence": 0.79,
        "founder_dna_traits": [
            "market_data_observation",
            "paper_first_safety",
            "global_halt_awareness",
            "resource_throttle_awareness",
            "training_lineage",
            "decision_explanation_contract",
            "data_collection_before_training",
            "registry_auditable_identity",
            "adaptive_meta_learning_governance",
        ],
        "founder_dna_inheritance_mode": "explicit_contract_metadata",
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "lineage_regression_guard": "fail_if_founder_dna_missing_or_stale",
        "lineage_generation": 5,
        "paper_runtime_stability_mode": "full_force_buffered",
        "paper_execution_queue_policy": "buffered_jsonl_batching",
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "paper_trade_lock_required": True,
        "paper_runtime_control_refresh_seconds": 300,
        "capability_pack_version": KERNEL_VERSION,
        "capability_pack_slug": KERNEL_SLUG,
        "capability_pack_display_name": KERNEL_DISPLAY_NAME,
        "adaptive_intelligence_kernel_version": KERNEL_VERSION,
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
    kernel = [row for row in rows if str(row.get("adaptive_intelligence_kernel_version") or "") == KERNEL_VERSION]
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
            "adaptive_intelligence_kernel_bot_count": len(kernel),
            "latest_adaptive_intelligence_kernel": KERNEL_VERSION,
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
        "adaptive_intelligence_kernel_version": KERNEL_VERSION,
        "pack_count": 1,
        "bot_count": len(BOTS),
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped_existing),
        "planned_rows": planned_rows,
        "skipped_existing_slot_kinds": skipped_existing,
        "kernel": _kernel_summary(assigned_ids),
    }


def _kernel_summary(assigned_ids: dict[str, str]) -> dict[str, Any]:
    contract = _kernel_contract(assigned_ids)
    return {
        "slug": KERNEL_SLUG,
        "display_name": KERNEL_DISPLAY_NAME,
        "sleeve_family": KERNEL_FAMILY,
        "sleeve_profile": KERNEL_PROFILE,
        "objective": "Add an adaptive meta-learning kernel that watches how the whole platform learns, forgets, transfers, and aligns over time.",
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "regression_guard_bot_id": contract["regression_guard_bot_id"],
        "capacity_check": contract["capacity_check"],
        "adaptive_depth": list(KERNEL_CONTRACT["adaptive_depth"]),
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
        "adaptive_intelligence_kernel_version": KERNEL_VERSION,
        "pack_count": plan["pack_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "kernel": plan["kernel"],
        "adaptive_kernel_contract": KERNEL_CONTRACT,
        "recommended_apply_command": "./scripts/ops/opsctl.sh adaptive-intelligence-kernel --apply --json",
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
        backup = backup_dir / f"master_bot_registry_before_adaptive_intelligence_kernel_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
        project_root / "config" / "adaptive_intelligence_kernel_v1.json",
        {
            "generated_at_utc": _utc_now(),
            "adaptive_intelligence_kernel_version": KERNEL_VERSION,
            "kernel": payload["kernel"],
            "adaptive_kernel_contract": payload["adaptive_kernel_contract"],
        },
    )
    _write_json(project_root / "governance" / "health" / "adaptive_intelligence_kernel_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add a 15-bot adaptive intelligence kernel expansion.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--apply", action="store_true", help="Append missing adaptive-kernel bots to master_bot_registry.json.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = apply_registry(project_root) if args.apply else build_payload(project_root)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "adaptive_intelligence_kernel "
            f"mode={payload['mode']} bots={payload['bot_count']} "
            f"planned={payload['planned_bot_count']} added={payload.get('added_bot_count', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
