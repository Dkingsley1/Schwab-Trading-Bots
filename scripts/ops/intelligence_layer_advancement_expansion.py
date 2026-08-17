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
BASE_VERSION = 934
PACK_VERSION = "intelligence_layer_advancement_v1"
PACK_SLUG = "intelligence_layer_advancement"
PACK_DISPLAY_NAME = "Intelligence Layer Advancement Pack"
SLEEVE_FAMILY = "meta_intelligence_control_plane"
SLEEVE_PROFILE = "intelligence_layer_advancement"
LABEL_CONTRACT_VERSION = "intelligence_layer_advancement_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 24000
MINIMUM_COLLECTION_DAYS = 90
PAPER_RUNTIME_CAPACITY_FLOOR = 925

SYSTEMS: list[dict[str, Any]] = [
    {
        "slug": "metacognitive_routing_v2",
        "display_name": "Metacognitive Routing v2",
        "objective": "Route reasoning effort, model families, sleeves, and memory calls by uncertainty, pressure, and payoff.",
        "outputs": ["reasoning_budget_packet", "expert_route_vote", "uncertainty_escalation_state"],
    },
    {
        "slug": "world_model_counterfactual_lab",
        "display_name": "World Model / Counterfactual Lab",
        "objective": "Simulate what-if market states and compare current decisions against plausible alternate paths.",
        "outputs": ["counterfactual_scenario_packet", "world_model_residual", "decision_alternative_set"],
    },
    {
        "slug": "alpha_evaluation_benchmark_suite",
        "display_name": "Alpha Evaluation Benchmark Suite",
        "objective": "Continuously benchmark alphas against golden replays, leakage checks, walk-forward windows, and crisis scenes.",
        "outputs": ["alpha_benchmark_score", "leakage_red_flag", "walk_forward_evidence_packet"],
    },
    {
        "slug": "memory_compression_retrieval_v2",
        "display_name": "Memory Compression / Retrieval v2",
        "objective": "Compress experience into useful memories and test whether retrieved memories actually help current decisions.",
        "outputs": ["memory_distillation_packet", "retrieval_quality_score", "memory_decay_vote"],
    },
    {
        "slug": "multi_agent_debate_critic_board",
        "display_name": "Multi-Agent Debate / Critic Board",
        "objective": "Let proposer, skeptic, and arbiter agents pressure-test high-impact advisory conclusions.",
        "outputs": ["debate_packet", "critic_objection_set", "arbiter_resolution_vote"],
    },
    {
        "slug": "active_learning_experiment_design_v2",
        "display_name": "Active Learning / Experiment Design v2",
        "objective": "Choose which labels, sources, regimes, or simulations are worth collecting next.",
        "outputs": ["active_learning_query", "experiment_value_score", "exploration_budget_vote"],
    },
    {
        "slug": "ensemble_governance_uncertainty",
        "display_name": "Ensemble Governance / Uncertainty",
        "objective": "Govern ensembles by diversity, calibration, disagreement, failure modes, and uncertainty routing.",
        "outputs": ["ensemble_diversity_score", "calibration_packet", "failure_mode_cluster"],
    },
    {
        "slug": "library_tool_intelligence_router",
        "display_name": "Library / Tool Intelligence Router",
        "objective": "Route MLX, QuantLib, sklearn, stats, and internal engines to the tasks where they add the most value.",
        "outputs": ["library_route_packet", "mlx_utilization_vote", "tool_reliability_score"],
    },
    {
        "slug": "safety_invariant_verification_v2",
        "display_name": "Safety Invariant Verification v2",
        "objective": "Verify no intelligence layer bypasses paper locks, halt contracts, storage guards, or training gates.",
        "outputs": ["safety_invariant_report", "canary_violation_packet", "halt_preemption_vote"],
    },
    {
        "slug": "self_improvement_backlog_planner",
        "display_name": "Self-Improvement Backlog Planner",
        "objective": "Rank future upgrades by payoff, risk, dependency complexity, data availability, and resource load.",
        "outputs": ["upgrade_backlog_rank", "implementation_dependency_graph", "risk_payoff_packet"],
    },
]

DATA_INTAKES = [
    "metacognitive_routing_trace_v2",
    "reasoning_budget_uncertainty_trace",
    "world_model_counterfactual_trace",
    "alpha_benchmark_replay_trace",
    "memory_distillation_retrieval_trace",
    "multi_agent_debate_critic_trace",
    "active_learning_experiment_trace_v2",
    "ensemble_uncertainty_calibration_trace",
    "library_tool_route_trace",
    "safety_invariant_verification_trace_v2",
    "self_improvement_backlog_trace",
]

STORAGE_TARGETS = [
    "governance/intelligence_layer",
    "governance/intelligence_layer/metacognition",
    "governance/intelligence_layer/world_model",
    "governance/intelligence_layer/benchmarks",
    "governance/intelligence_layer/memory",
    "governance/intelligence_layer/debate",
    "governance/intelligence_layer/active_learning",
    "governance/intelligence_layer/ensemble",
    "governance/intelligence_layer/tool_router",
    "governance/intelligence_layer/safety",
    "governance/intelligence_layer/backlog",
    "governance/health/intelligence_layer_advancement_latest.json",
]

REQUIRED_LABELS = [
    "reasoning_budget_bucket",
    "uncertainty_escalation_state",
    "counterfactual_scenario_id",
    "alpha_benchmark_bucket",
    "retrieval_quality_bucket",
    "critic_objection_bucket",
    "experiment_value_bucket",
    "ensemble_calibration_bucket",
    "tool_route_status",
    "safety_invariant_status",
]

BOTS: list[dict[str, Any]] = [
    ("metacognitive_reasoning_budget_allocator", "Meta Reasoning Budget Allocator", "metacognitive_routing_v2", "infrastructure_sub_bot", "critical", "Allocate reasoning depth and compute budget by uncertainty, pressure, and expected value.", ["system_self_model", "runtime_throttle", "mlx_intelligence_router"]),
    ("expert_route_selector", "Expert Route Selector", "metacognitive_routing_v2", "signal_sub_bot", "high", "Choose which specialist sleeves, libraries, and masters should evaluate a state.", ["sleeve_masters", "library_router", "coordination_intelligence"]),
    ("uncertainty_escalation_guard", "Uncertainty Escalation Guard", "metacognitive_routing_v2", "infrastructure_sub_bot", "critical", "Escalate or downshift decisions when uncertainty, disagreement, or resource pressure is elevated.", ["global_halt_status", "operator_cockpit", "ensemble_uncertainty"]),
    ("counterfactual_scenario_generator", "Counterfactual Scenario Generator", "world_model_counterfactual_lab", "signal_sub_bot", "high", "Generate plausible alternate market paths for current regime and event context.", ["stress_lab", "market_regime_memory", "macro_context_sync"]),
    ("world_model_residual_scorer", "World Model Residual Scorer", "world_model_counterfactual_lab", "signal_sub_bot", "high", "Score when live behavior diverges from the platform's expected world model.", ["regime_router", "feature_quality", "operator_cockpit"]),
    ("decision_alternative_critic", "Decision Alternative Critic", "world_model_counterfactual_lab", "infrastructure_sub_bot", "critical", "Compare candidate actions against counterfactual paths before trust is lifted.", ["decision_provenance", "execution_reality_lab", "alpha_intelligence_evolution"]),
    ("golden_replay_alpha_benchmarker", "Golden Replay Alpha Benchmarker", "alpha_evaluation_benchmark_suite", "infrastructure_sub_bot", "critical", "Benchmark alphas against replay scenes, crisis windows, and known fragile regimes.", ["golden_replay_guard", "stress_lab", "model_lifecycle"]),
    ("walk_forward_evidence_scorer", "Walk-Forward Evidence Scorer", "alpha_evaluation_benchmark_suite", "signal_sub_bot", "critical", "Score whether alpha survives walk-forward windows without leakage or decay.", ["training_quality", "model_decay_detector", "multiple_testing_guard"]),
    ("leakage_red_team_auditor", "Leakage Red-Team Auditor", "alpha_evaluation_benchmark_suite", "infrastructure_sub_bot", "critical", "Attack labels and features for lookahead, duplicate joins, and retrospective leakage.", ["label_quality", "feature_store", "point_in_time_event_store"]),
    ("episodic_memory_distiller_v2", "Episodic Memory Distiller v2", "memory_compression_retrieval_v2", "infrastructure_sub_bot", "high", "Compress incidents, regimes, and decisions into small reusable memory packets.", ["system_self_model", "incident_memory", "regime_playbook_memory_v2"]),
    ("retrieval_relevance_scorer_v2", "Retrieval Relevance Scorer v2", "memory_compression_retrieval_v2", "signal_sub_bot", "high", "Score whether retrieved memories are relevant enough to influence current routing.", ["memory_retrieval", "decision_provenance", "adaptive_intelligence_kernel"]),
    ("memory_decay_forgetting_guard_v2", "Memory Decay / Forgetting Guard v2", "memory_compression_retrieval_v2", "infrastructure_sub_bot", "high", "Forget or down-rank stale lessons that no longer match market structure.", ["model_decay_detector", "feature_freshness", "self_model_regression_guard"]),
    ("proposal_agent_board", "Proposal Agent Board", "multi_agent_debate_critic_board", "signal_sub_bot", "medium", "Generate structured proposals for high-impact advisory changes.", ["cognitive_control_plane", "research_foundry", "alpha_research_os"]),
    ("skeptic_critic_board", "Skeptic Critic Board", "multi_agent_debate_critic_board", "infrastructure_sub_bot", "critical", "List objections, missing evidence, and unsafe assumptions before trust lift.", ["advanced_mesh_self_correction", "safety_invariant_verification", "bot_admission_guard"]),
    ("arbiter_resolution_master", "Arbiter Resolution Master", "multi_agent_debate_critic_board", "infrastructure_sub_bot", "critical", "Resolve proposer and skeptic disagreement into an auditable advisory vote.", ["grandmaster_control", "coordination_intelligence", "operator_cockpit"]),
    ("active_learning_query_planner_v2", "Active Learning Query Planner v2", "active_learning_experiment_design_v2", "signal_sub_bot", "high", "Choose which samples, labels, sources, or regimes are most valuable to collect next.", ["coverage_gap_closer", "research_automation", "feature_quality"]),
    ("experiment_value_of_information_scorer", "Experiment Value-of-Information Scorer", "active_learning_experiment_design_v2", "signal_sub_bot", "high", "Rank experiments by expected information gain minus resource and storage cost.", ["experiment_ledger", "expansion_capacity", "training_quality"]),
    ("exploration_budget_guard", "Exploration Budget Guard", "active_learning_experiment_design_v2", "infrastructure_sub_bot", "critical", "Limit exploration when storage, CPU, memory, or global halt pressure is elevated.", ["runtime_throttle", "storage_backpressure_autopilot", "global_halt"]),
    ("ensemble_diversity_governor", "Ensemble Diversity Governor", "ensemble_governance_uncertainty", "infrastructure_sub_bot", "critical", "Prevent ensembles from becoming many copies of the same alpha.", ["duplicate_alpha_novelty", "correlation_cluster_governor", "ensemble_disagreement"]),
    ("calibration_uncertainty_mapper", "Calibration Uncertainty Mapper", "ensemble_governance_uncertainty", "signal_sub_bot", "high", "Map calibration errors, confidence drift, and uncertainty by sleeve and regime.", ["training_quality", "model_lifecycle", "bayesian_uncertainty"]),
    ("failure_mode_clusterer", "Failure Mode Clusterer", "ensemble_governance_uncertainty", "infrastructure_sub_bot", "high", "Cluster recurring model, feed, label, and execution failure modes.", ["incident_review_packet", "system_self_model", "reporting_layer"]),
    ("mlx_route_optimizer_v2", "MLX Route Optimizer v2", "library_tool_intelligence_router", "infrastructure_sub_bot", "high", "Route MLX kernels, compiled paths, and batch sizing to the right intelligence jobs.", ["mlx_intelligence_router", "mlx_utilization", "memory_efficiency"]),
    ("quant_library_route_scorer", "Quant Library Route Scorer", "library_tool_intelligence_router", "infrastructure_sub_bot", "medium", "Route QuantLib, stats, simulation, and internal engines by suitability and cost.", ["library_utilization_router", "quant_model_control", "pricing_grad"]),
    ("tool_reliability_canary", "Tool Reliability Canary", "library_tool_intelligence_router", "infrastructure_sub_bot", "critical", "Detect broken or degraded libraries before intelligence layers depend on them.", ["dependency_utilization", "command_validity", "daily_verify"]),
    ("policy_invariant_verifier_v2", "Policy Invariant Verifier v2", "safety_invariant_verification_v2", "infrastructure_sub_bot", "critical", "Verify paper lock, halt, storage, and training gates cannot be bypassed by intelligence layers.", ["paper_trade_lock_guard", "global_halt", "new_bot_admission_guard"]),
    ("canary_violation_simulator", "Canary Violation Simulator", "safety_invariant_verification_v2", "infrastructure_sub_bot", "critical", "Run dry canary violations to ensure guards catch unsafe promotions or execution paths.", ["golden_replay_guard", "release_freeze", "regression_guard"]),
    ("halt_preemption_intelligence_guard", "Halt Preemption Intelligence Guard", "safety_invariant_verification_v2", "infrastructure_sub_bot", "critical", "Recommend throttles and summary-only modes before pressure becomes a global halt.", ["global_halt_preemption", "runtime_throttle", "ingestion_backpressure"]),
    ("upgrade_candidate_ranker_v2", "Upgrade Candidate Ranker v2", "self_improvement_backlog_planner", "infrastructure_sub_bot", "high", "Rank next system upgrades by payoff, risk, dependency complexity, and resource load.", ["system_self_model", "optimization_recommendation_ranker", "commands_hygiene"]),
    ("implementation_dependency_mapper", "Implementation Dependency Mapper", "self_improvement_backlog_planner", "infrastructure_sub_bot", "high", "Map dependencies needed before future expansions are safe to apply.", ["dependency_graph", "expansion_capacity", "system_drift_guard"]),
    ("risk_payoff_backlog_master", "Risk / Payoff Backlog Master", "self_improvement_backlog_planner", "infrastructure_sub_bot", "critical", "Produce a grandmaster-ready backlog of safe next upgrades and blocked ideas.", ["grandmaster_self_awareness_bridge", "operator_cockpit", "institutional_readiness"]),
]

BOTS = [
    {
        "role_slug": role_slug,
        "slug": f"intelligence_{role_slug}_bot",
        "label": label,
        "system": system,
        "bot_role": bot_role,
        "priority": priority,
        "objective": objective,
        "target_functions": targets,
    }
    for role_slug, label, system, bot_role, priority, objective, targets in BOTS
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


def _ensure_storage_targets(project_root: Path) -> list[str]:
    ready: list[str] = []
    for target in STORAGE_TARGETS:
        path = project_root / target
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            ready.append(str(path.parent.relative_to(project_root)))
        else:
            path.mkdir(parents=True, exist_ok=True)
            ready.append(str(path.relative_to(project_root)))
    return sorted(set(ready))


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
    return f"{PACK_SLUG}_{bot['role_slug']}"


def _assign_bot_ids(rows: list[dict[str, Any]]) -> dict[str, str]:
    existing_by_slot = {
        str(row.get("slot_kind") or ""): str(row.get("bot_id") or "")
        for row in rows
        if str(row.get("slot_kind") or "") and str(row.get("bot_id") or "")
    }
    used_versions = {
        version
        for row in rows
        for version in [_version_from_bot_id(str(row.get("bot_id") or ""))]
        if version is not None
    }
    assigned: dict[str, str] = {}
    for index, bot in enumerate(BOTS):
        slot = _slot_kind(bot)
        if slot in existing_by_slot:
            assigned[slot] = existing_by_slot[slot]
            continue
        desired = BASE_VERSION + index
        version = desired if desired not in used_versions else _next_available_version(
            used_versions, max(max(used_versions, default=BASE_VERSION - 1) + 1, desired)
        )
        used_versions.add(version)
        assigned[slot] = f"brain_refinery_v{version}_{bot['slug']}"
    return assigned


def _system(bot: dict[str, Any]) -> dict[str, Any]:
    for system in SYSTEMS:
        if system["slug"] == bot["system"]:
            return system
    return {"slug": bot["system"], "display_name": bot["system"], "objective": bot["objective"], "outputs": []}


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


def _pack_contract(assigned_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": PACK_VERSION,
        "new_sleeve_or_subsleeve": {
            "sleeve_family": SLEEVE_FAMILY,
            "sleeve_profile": SLEEVE_PROFILE,
            "display_name": PACK_DISPLAY_NAME,
        },
        "system_count": len(SYSTEMS),
        "bot_count": len(BOTS),
        "bots_per_system": 3,
        "bot_pack_size_rule": "10_systems_3_bots_each_30_total_max",
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "intelligence_layer_hot_7d_warm_60d_cold_365d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": 22,
            "capture_mode": "digest_first_reasoning_trace",
            "sample_rate": 0.14,
            "dedupe_required": True,
            "stale_deletion_policy": "retain_decision_digests_stage_raw_reasoning_traces",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "sleeve_master_bot_id": assigned_ids.get(f"{PACK_SLUG}_metacognitive_reasoning_budget_allocator", ""),
        "regression_guard_bot_id": assigned_ids.get(f"{PACK_SLUG}_policy_invariant_verifier_v2", ""),
        "grandmaster_bridge_bot_id": assigned_ids.get(f"{PACK_SLUG}_risk_payoff_backlog_master", ""),
        "runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "intelligence_advancements": [
            "metacognitive_routing_v2",
            "world_model_counterfactual_lab",
            "alpha_evaluation_benchmark_suite",
            "memory_compression_retrieval_v2",
            "multi_agent_debate_critic_board",
            "active_learning_experiment_design_v2",
            "ensemble_governance_uncertainty",
            "library_tool_intelligence_router",
            "safety_invariant_verification_v2",
            "self_improvement_backlog_planner",
        ],
        "global_halt_contract": "reasoning_and_experiment_layers_downshift_to_digest_only_before_halt_pressure_escalates",
        "paper_lock_contract": "no_direct_execution_no_allocation_no_training_until_benchmark_safety_and_memory_quality_gates_clear",
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    system = _system(bot)
    contract = _pack_contract(assigned_ids)
    advanced_contract = {
        "contract_version": "intelligence_layer_advancement_layers_v1",
        "capability_pack": PACK_SLUG,
        "system": bot["system"],
        "system_display_name": system["display_name"],
        "system_outputs": list(system.get("outputs", [])),
        "reports_to_sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "regression_guard_bot_id": contract["regression_guard_bot_id"],
        "grandmaster_bridge_bot_id": contract["grandmaster_bridge_bot_id"],
        "intelligence_advancement": "evaluate_route_memory_debate_and_verify_before_trust_lift",
        "reasoning_safety": "digest_first_collect_only_no_hidden_execution_paths",
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
        "preferred_regimes": ["mixed_transition", "fragile_transition", "risk_off_shock", "risk_on_trend", "low_pressure"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v804_advanced_mesh_metacognitive_state_router_master_bot",
            "brain_refinery_v879_adaptive_kernel_online_meta_learning_master_bot",
            "brain_refinery_v894_self_model_identity_cartographer_bot",
            "brain_refinery_v904_alpha_training_graduation_scorer_bot",
        ],
        "data_intake_collections": list(DATA_INTAKES),
        "storage_targets": list(STORAGE_TARGETS),
        "freshness_slo_seconds": 900,
        "retention_profile": "intelligence_layer_hot_7d_warm_60d_cold_365d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "intelligence_layer_collect_only_until_benchmark_memory_safety_and_routing_gates_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "intelligence_layer_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_intelligence_layer_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "training_threshold_policy": {
            "minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_collection_days": MINIMUM_COLLECTION_DAYS,
            "requires_alpha_benchmark_clearance": True,
            "requires_safety_invariant_clearance": True,
            "requires_memory_quality_clearance": True,
            "requires_runtime_pressure_clear": True,
        },
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "digest_first_reasoning_trace",
        "data_collection_sample_rate": 0.14,
        "data_collection_max_daily_storage_mb": 22,
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "sleeve_profile": SLEEVE_PROFILE,
        "sleeve_family": SLEEVE_FAMILY,
        "correlation_peer_sleeves": [
            "advanced_intelligence_mesh",
            "adaptive_intelligence_kernel",
            "system_self_awareness",
            "alpha_intelligence_evolution",
            "coordination_intelligence",
        ],
        "correlation_dependencies": [
            "system_self_model",
            "operator_cockpit",
            "alpha_intelligence_evolution",
            "training_quality_control",
            "global_halt_status",
            "mlx_intelligence_router",
            "library_utilization_router",
        ],
        "provider_capability_profile": "internal_intelligence_control_plane_only",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "master_bot_registry",
            "system_self_model",
            "operator_cockpit",
            "alpha_intelligence_evolution",
            "training_quality_control",
            "source_verification",
            "decision_provenance",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "intelligence_layer_advancement_has_no_direct_execution_or_broker_dependency",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.78,
            "freshness_slo_seconds": 900,
            "regression_guard_bot_id": contract["regression_guard_bot_id"],
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": [
            "research_only",
            "collection_only",
            "execution_blocked",
            "paper_only_floor",
            f"sleeve_family:{SLEEVE_FAMILY}",
            f"sleeve_profile:{SLEEVE_PROFILE}",
            f"capability_pack:{PACK_SLUG}",
            f"intelligence_system:{bot['system']}",
            "intelligence_advancement",
            "metacognitive_control",
            "training_after_threshold",
            "global_halt_aware",
        ],
        "execution_policy_label": "collection_only_intelligence_layer_no_execution",
        "eligible_for_master_vote": False,
        "founder_bot_id": "brain_refinery_v1",
        "founder_dna_version": "founder_dna_v1",
        "founder_dna_traits": [
            "market_data_observation",
            "paper_first_safety",
            "global_halt_awareness",
            "resource_throttle_awareness",
            "decision_explanation_contract",
            "registry_auditable_identity",
            "metacognitive_routing",
            "safety_invariant_verification",
        ],
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "paper_runtime_stability_mode": "full_force_buffered",
        "paper_trade_lock_required": True,
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "capability_pack_version": PACK_VERSION,
        "capability_pack_slug": PACK_SLUG,
        "capability_pack_display_name": PACK_DISPLAY_NAME,
        "intelligence_layer_advancement_version": PACK_VERSION,
        "capability_pack_contract": contract,
        "advanced_intelligence_layer_contract": advanced_contract,
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    pack_rows = [row for row in rows if str(row.get("intelligence_layer_advancement_version") or "") == PACK_VERSION]
    summary = dict(registry.get("summary") or {})
    summary.update(
        {
            "total_bots": len(rows),
            "active_bots": len(active),
            "inactive_bots": len(inactive),
            "active_signal_sub_bots": len(signal_active),
            "active_infrastructure_sub_bots": len(infra_active),
            "data_collection_active_bots": sum(1 for row in rows if bool(row.get("data_collection_active"))),
            "training_excluded_bots": sum(1 for row in rows if bool(row.get("training_excluded"))),
            "structured_capability_pack_bot_count": len(structured),
            "intelligence_layer_advancement_bot_count": len(pack_rows),
            "latest_intelligence_layer_advancement": PACK_VERSION,
        }
    )
    registry["summary"] = summary


def _pack_summary(assigned_ids: dict[str, str]) -> dict[str, Any]:
    contract = _pack_contract(assigned_ids)
    return {
        "slug": PACK_SLUG,
        "display_name": PACK_DISPLAY_NAME,
        "sleeve_family": SLEEVE_FAMILY,
        "sleeve_profile": SLEEVE_PROFILE,
        "objective": "Advance the system's intelligence layer through routing, world models, benchmarks, memory, debate, active learning, uncertainty, tool selection, safety verification, and self-improvement planning.",
        "system_count": len(SYSTEMS),
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "systems": list(SYSTEMS),
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "regression_guard_bot_id": contract["regression_guard_bot_id"],
        "grandmaster_bridge_bot_id": contract["grandmaster_bridge_bot_id"],
        "runtime_capacity_floor": contract["runtime_capacity_floor"],
        "intelligence_advancements": list(contract["intelligence_advancements"]),
    }


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
        "intelligence_layer_advancement_version": PACK_VERSION,
        "system_count": len(SYSTEMS),
        "bot_count": len(BOTS),
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped_existing),
        "planned_rows": planned_rows,
        "skipped_existing_slot_kinds": skipped_existing,
        "pack": _pack_summary(assigned_ids),
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
        "intelligence_layer_advancement_version": PACK_VERSION,
        "system_count": plan["system_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "pack": plan["pack"],
        "recommended_apply_command": "./scripts/ops/opsctl.sh intelligence-layer-advancement --apply --json",
    }


def apply_registry(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    registry = _load_json(registry_path)
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    plan = plan_registry_expansion(registry)
    added_rows = list(plan["planned_rows"])
    storage_targets_ready = _ensure_storage_targets(project_root)
    backup_path = ""
    if added_rows:
        backup_dir = project_root / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"master_bot_registry_before_intelligence_layer_advancement_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
            "storage_targets_ready": storage_targets_ready,
        }
    )
    _write_json(
        project_root / "config" / "intelligence_layer_advancement_v1.json",
        {
            "generated_at_utc": _utc_now(),
            "intelligence_layer_advancement_version": PACK_VERSION,
            "pack": payload["pack"],
        },
    )
    _write_json(project_root / "governance" / "health" / "intelligence_layer_advancement_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add the next intelligence layer advancement pack.")
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
            "intelligence_layer_advancement "
            f"mode={payload['mode']} systems={payload['system_count']} "
            f"bots={payload['bot_count']} planned={payload['planned_bot_count']} "
            f"added={payload.get('added_bot_count', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
