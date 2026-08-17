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
BASE_VERSION = 964
TARGET_PLATFORM_TOTAL_BOTS = 1000
PACK_VERSION = "apex_self_awareness_intelligence_v1"
PACK_SLUG = "apex_self_awareness_intelligence"
PACK_DISPLAY_NAME = "Apex Self-Awareness Intelligence Pack"
SLEEVE_FAMILY = "apex_meta_intelligence_control_plane"
SLEEVE_PROFILE = "apex_self_awareness_intelligence"
LABEL_CONTRACT_VERSION = "apex_self_awareness_intelligence_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 30000
MINIMUM_COLLECTION_DAYS = 120
PAPER_RUNTIME_CAPACITY_FLOOR = 1000

SYSTEMS: list[dict[str, Any]] = [
    {
        "slug": "self_model_deep_introspection",
        "display_name": "Self-Model Deep Introspection",
        "objective": "Keep a live, contradiction-aware map of identity, capabilities, assumptions, lineage, and current posture.",
        "outputs": ["self_state_vector", "assumption_inventory", "contradiction_packet"],
    },
    {
        "slug": "meta_reasoning_policy_engine",
        "display_name": "Meta-Reasoning Policy Engine",
        "objective": "Route reasoning depth, specialist selection, confidence budgets, and trace quality by payoff and pressure.",
        "outputs": ["reasoning_policy_packet", "confidence_budget_vote", "reasoning_trace_quality"],
    },
    {
        "slug": "experience_memory_os",
        "display_name": "Experience Memory OS",
        "objective": "Compress, retrieve, replay, and retire system lessons with point-in-time evidence.",
        "outputs": ["memory_index_packet", "causal_lesson_score", "memory_compaction_vote"],
    },
    {
        "slug": "world_model_scenario_oracle",
        "display_name": "World Model Scenario Oracle",
        "objective": "Compare live state against counterfactual scenarios, regime maps, and cross-sleeve consistency checks.",
        "outputs": ["scenario_path_packet", "world_model_consistency_score", "scenario_decay_flag"],
    },
    {
        "slug": "autonomous_upgrade_foundry",
        "display_name": "Autonomous Upgrade Foundry",
        "objective": "Propose, rank, dependency-map, and park future upgrades before they touch production paths.",
        "outputs": ["upgrade_hypothesis", "dependency_safe_plan", "payoff_risk_rank"],
    },
    {
        "slug": "alpha_safety_causal_judge",
        "display_name": "Alpha Safety / Causal Judge",
        "objective": "Judge alphas for causal evidence, duplicate behavior, leakage, promotion safety, and paper/live drift.",
        "outputs": ["causal_evidence_packet", "promotion_safety_vote", "duplicate_alpha_flag"],
    },
    {
        "slug": "resource_autonomy_governor",
        "display_name": "Resource Autonomy Governor",
        "objective": "Keep the 1000-bot platform calm by learning host pressure, app co-tenancy, storage pressure, and lane budgets.",
        "outputs": ["resource_autonomy_packet", "lane_budget_vote", "thousand_bot_smoothing_state"],
    },
    {
        "slug": "operator_copilot_narrative",
        "display_name": "Operator Copilot Narrative",
        "objective": "Turn state, incidents, commands, and reports into concise operator-ready system explanations.",
        "outputs": ["operator_brief_packet", "command_recommendation", "report_readiness_flag"],
    },
    {
        "slug": "grandmaster_collective_intelligence",
        "display_name": "Grand Master Collective Intelligence",
        "objective": "Harmonize sleeve master votes, disagreement, context packets, and collective health for Grand Master routing.",
        "outputs": ["grandmaster_context_packet", "sleeve_vote_harmony_score", "collective_health_flag"],
    },
    {
        "slug": "adaptive_research_frontier",
        "display_name": "Adaptive Research Frontier",
        "objective": "Watch frontier research ideas and convert only evidence-backed ideas into safe bot blueprints.",
        "outputs": ["frontier_watchlist_packet", "experiment_design_candidate", "research_blueprint_guard_vote"],
    },
]

DATA_INTAKES = [
    "apex_self_state_vector_trace",
    "assumption_inventory_contradiction_trace",
    "meta_reasoning_policy_trace",
    "experience_memory_index_trace",
    "world_model_scenario_oracle_trace",
    "upgrade_foundry_hypothesis_trace",
    "causal_alpha_safety_judge_trace",
    "resource_autonomy_pressure_trace",
    "operator_copilot_narrative_trace",
    "grandmaster_collective_intelligence_trace",
    "adaptive_research_frontier_trace",
]

STORAGE_TARGETS = [
    "governance/apex_intelligence",
    "governance/apex_intelligence/self_model",
    "governance/apex_intelligence/meta_reasoning",
    "governance/apex_intelligence/memory",
    "governance/apex_intelligence/world_model",
    "governance/apex_intelligence/upgrade_foundry",
    "governance/apex_intelligence/alpha_safety",
    "governance/apex_intelligence/resource_autonomy",
    "governance/apex_intelligence/operator_copilot",
    "governance/apex_intelligence/grandmaster",
    "governance/apex_intelligence/research_frontier",
    "governance/health/apex_self_awareness_intelligence_latest.json",
]

REQUIRED_LABELS = [
    "self_state_vector_bucket",
    "assumption_status",
    "reasoning_policy_bucket",
    "memory_retrieval_value_bucket",
    "scenario_consistency_bucket",
    "upgrade_payoff_risk_bucket",
    "causal_evidence_status",
    "resource_autonomy_status",
    "operator_brief_quality_bucket",
    "grandmaster_vote_harmony_bucket",
]

BOTS: list[dict[str, Any]] = [
    ("self_model_state_vector_builder", "Self-Model State Vector Builder", "self_model_deep_introspection", "infrastructure_sub_bot", "critical", "Build compact state vectors summarizing identity, sleeves, resources, gates, and current operating mode.", ["system_self_model", "operator_cockpit", "global_halt_status"]),
    ("capability_graph_integrity_mapper", "Capability Graph Integrity Mapper", "self_model_deep_introspection", "infrastructure_sub_bot", "critical", "Map capability packs, core files, commands, and registry rows into one integrity graph.", ["core_bot_catalog", "commands_hygiene", "bot_founder_dna"]),
    ("live_assumption_inventory_guard", "Live Assumption Inventory Guard", "self_model_deep_introspection", "infrastructure_sub_bot", "critical", "Track assumptions the platform is currently relying on and flag stale or unsupported ones.", ["system_self_model", "source_verification", "feature_quality"]),
    ("internal_contradiction_detector", "Internal Contradiction Detector", "self_model_deep_introspection", "infrastructure_sub_bot", "critical", "Detect mismatches between registry truth, health files, cockpit summaries, and command surfaces.", ["operator_cockpit", "core_bot_catalog", "daily_verify"]),
    ("identity_lineage_checkpoint_master", "Identity Lineage Checkpoint Master", "self_model_deep_introspection", "infrastructure_sub_bot", "high", "Checkpoint founder DNA, capability lineage, and materialized file coverage after each expansion.", ["bot_founder_dna", "core_bot_materialize", "new_bot_admission_guard"]),
    ("reasoning_policy_router", "Reasoning Policy Router", "meta_reasoning_policy_engine", "infrastructure_sub_bot", "critical", "Route each advisory task to fast, careful, replay, or debate mode by uncertainty and payoff.", ["advanced_intelligence_mesh", "cognitive_control_plane", "runtime_throttle"]),
    ("confidence_budget_allocator", "Confidence Budget Allocator", "meta_reasoning_policy_engine", "signal_sub_bot", "high", "Allocate confidence budgets by data quality, source agreement, regime stability, and model decay.", ["training_quality", "source_confidence", "ensemble_uncertainty"]),
    ("deliberation_depth_throttle", "Deliberation Depth Throttle", "meta_reasoning_policy_engine", "infrastructure_sub_bot", "critical", "Reduce noncritical reasoning depth when CPU, memory, swap, or backlog pressure rises.", ["memory_efficiency", "swap_pressure_governor", "runtime_throttle"]),
    ("specialist_consensus_weighting", "Specialist Consensus Weighting Bot", "meta_reasoning_policy_engine", "signal_sub_bot", "high", "Weight specialist opinions by recent reliability, coverage, uncertainty, and overlap risk.", ["coordination_intelligence", "sleeve_masters", "duplicate_alpha_novelty"]),
    ("reasoning_trace_regression_guard", "Reasoning Trace Regression Guard", "meta_reasoning_policy_engine", "infrastructure_sub_bot", "critical", "Guard reasoning traces against missing evidence links, unsafe claims, or hidden execution paths.", ["decision_provenance", "reporter_quality", "safety_invariants"]),
    ("episodic_memory_index_builder", "Episodic Memory Index Builder", "experience_memory_os", "infrastructure_sub_bot", "high", "Index incidents, expansions, guard trips, retrains, and successful stabilizations into compact memory.", ["incident_timeline", "system_self_model", "regime_playbook_memory"]),
    ("causal_lesson_retrieval_scorer", "Causal Lesson Retrieval Scorer", "experience_memory_os", "signal_sub_bot", "high", "Score retrieved lessons by causal relevance to the current state instead of keyword similarity.", ["memory_retrieval", "causal_discovery", "operator_cockpit"]),
    ("failure_memory_replay_picker", "Failure Memory Replay Picker", "experience_memory_os", "infrastructure_sub_bot", "critical", "Pick the best historical failure replay before changing halt, feed, storage, or training controls.", ["golden_replay_guard", "incident_review_packet", "global_halt"]),
    ("memory_compaction_quality_guard", "Memory Compaction Quality Guard", "experience_memory_os", "infrastructure_sub_bot", "critical", "Verify memory compaction keeps decisions, dates, thresholds, and outcomes intact.", ["system_self_model", "report_quality_guard", "commands_hygiene"]),
    ("stale_lesson_retirement_master", "Stale Lesson Retirement Master", "experience_memory_os", "infrastructure_sub_bot", "high", "Down-rank old lessons when market structure, providers, or runtime architecture changes.", ["model_decay_detector", "feature_freshness", "adaptive_intelligence_kernel"]),
    ("regime_state_space_mapper", "Regime State Space Mapper", "world_model_scenario_oracle", "signal_sub_bot", "high", "Map current macro, microstructure, volatility, liquidity, and cross-asset regime state.", ["market_regime_router", "macro_context_sync", "microstructure"]),
    ("shock_path_counterfactual_generator", "Shock Path Counterfactual Generator", "world_model_scenario_oracle", "signal_sub_bot", "high", "Generate digest-sized shock paths before high-impact advisory or expansion decisions.", ["stress_lab", "scenario_lab", "world_model"]),
    ("cross_sleeve_scenario_consistency_guard", "Cross-Sleeve Scenario Consistency Guard", "world_model_scenario_oracle", "infrastructure_sub_bot", "critical", "Check that sleeve assumptions agree under the same scenario before trust is lifted.", ["sleeve_masters", "coordination_intelligence", "portfolio_exposure"]),
    ("market_microstructure_scenario_bridge", "Market Microstructure Scenario Bridge", "world_model_scenario_oracle", "signal_sub_bot", "high", "Bridge order-flow and liquidity states into broader scenario decisions.", ["order_flow_microstructure", "liquidity_regime", "execution_reality_lab"]),
    ("scenario_truth_decay_auditor", "Scenario Truth Decay Auditor", "world_model_scenario_oracle", "infrastructure_sub_bot", "critical", "Audit when old scenario assumptions stop matching live market behavior.", ["scenario_lab", "model_decay_detector", "feature_quality"]),
    ("upgrade_hypothesis_generator", "Upgrade Hypothesis Generator", "autonomous_upgrade_foundry", "infrastructure_sub_bot", "high", "Generate safe candidate upgrades from failures, missing coverage, and operator intent.", ["system_self_model", "operator_copilot", "research_foundry"]),
    ("dependency_safe_patch_planner", "Dependency-Safe Patch Planner", "autonomous_upgrade_foundry", "infrastructure_sub_bot", "critical", "Plan upgrades by dependency order, write scope, tests, and rollback evidence.", ["dependency_graph", "commands_hygiene", "regression_guard"]),
    ("regression_surface_forecaster", "Regression Surface Forecaster", "autonomous_upgrade_foundry", "infrastructure_sub_bot", "critical", "Forecast which tests and runtime artifacts need protection before a change lands.", ["daily_verify", "test_inventory", "system_drift_guard"]),
    ("payoff_risk_upgrade_ranker", "Payoff / Risk Upgrade Ranker", "autonomous_upgrade_foundry", "signal_sub_bot", "high", "Rank upgrade ideas by expected payoff, resource load, operator value, and implementation risk.", ["optimization_recommendation_ranker", "expansion_capacity", "operator_cockpit"]),
    ("blocked_idea_parking_lot_curator", "Blocked Idea Parking Lot Curator", "autonomous_upgrade_foundry", "infrastructure_sub_bot", "medium", "Keep risky or unsupported ideas parked with explicit unblock criteria instead of silently losing them.", ["research_pipeline", "new_bot_admission_guard", "institutional_readiness"]),
    ("causal_evidence_gatekeeper", "Causal Evidence Gatekeeper", "alpha_safety_causal_judge", "infrastructure_sub_bot", "critical", "Require point-in-time causal evidence before any alpha can graduate from collection.", ["training_quality", "point_in_time_event_store", "causal_discovery"]),
    ("duplicate_alpha_semantic_guard", "Duplicate Alpha Semantic Guard", "alpha_safety_causal_judge", "infrastructure_sub_bot", "critical", "Detect semantically duplicate alpha ideas across bot labels, features, and behavior.", ["duplicate_alpha_novelty", "feature_store", "bot_catalog"]),
    ("overfit_leakage_prosecutor", "Overfit / Leakage Prosecutor", "alpha_safety_causal_judge", "infrastructure_sub_bot", "critical", "Actively attack candidate alphas for leakage, overfit, hindsight joins, and survivorship bias.", ["multiple_testing_guard", "leakage_red_team", "golden_replay_guard"]),
    ("promotion_safety_judge", "Promotion Safety Judge", "alpha_safety_causal_judge", "infrastructure_sub_bot", "critical", "Judge promotions against paper locks, halt state, resource pressure, and execution realism.", ["promotion_autopilot", "paper_trade_lock_guard", "execution_lab"]),
    ("paper_live_gap_preemption_guard", "Paper / Live Gap Preemption Guard", "alpha_safety_causal_judge", "infrastructure_sub_bot", "critical", "Flag slippage, fill, latency, and data entitlement gaps before live trust can rise.", ["execution_reality_lab", "broker_adapter_mesh", "transaction_cost_intelligence"]),
    ("host_app_awareness_scheduler", "Host App Awareness Scheduler", "resource_autonomy_governor", "infrastructure_sub_bot", "critical", "Downshift collection and reasoning automatically when Logic Pro, Final Cut, PyCharm, or browsers need room.", ["creative_cotenant_guard", "memory_efficiency", "runtime_throttle"]),
    ("swap_pressure_learning_controller", "Swap Pressure Learning Controller", "resource_autonomy_governor", "infrastructure_sub_bot", "critical", "Learn which queues, feeds, and reports push swap pressure and preemptively calm them.", ["swap_pressure_governor", "memory_efficiency", "backpressure_drainers"]),
    ("cpu_memory_lane_arbiter", "CPU / Memory Lane Arbiter", "resource_autonomy_governor", "infrastructure_sub_bot", "critical", "Assign lane budgets across live feeds, paper loops, reporters, retrains, and MLX jobs.", ["runtime_throttle", "mlx_intelligence_router", "writer_cycle_coordinator"]),
    ("storage_backpressure_predictor_v2", "Storage Backpressure Predictor v2", "resource_autonomy_governor", "infrastructure_sub_bot", "critical", "Predict storage and queue pressure before writers or feeds trip global halt gates.", ["storage_backpressure_autopilot", "ingestion_storage_control", "sql_link_shards"]),
    ("thousand_bot_runtime_smoother", "1000-Bot Runtime Smoother", "resource_autonomy_governor", "infrastructure_sub_bot", "critical", "Smooth collection cadence, sampling, and paper queues for the 1000-bot platform.", ["backpressure_slo", "data_collection_storage_guard", "global_halt_preemption"]),
    ("operator_intent_memory_mapper", "Operator Intent Memory Mapper", "operator_copilot_narrative", "infrastructure_sub_bot", "high", "Remember operator preferences and repeated goals so future recommendations fit the way the system is used.", ["operator_cockpit", "commands_hygiene", "system_summary"]),
    ("daily_system_brief_composer", "Daily System Brief Composer", "operator_copilot_narrative", "infrastructure_sub_bot", "high", "Compose concise daily briefs explaining bot counts, data quality, halts, backlog, and readiness.", ["system_summary", "one_numbers", "reporting_layer"]),
    ("command_recommendation_router", "Command Recommendation Router", "operator_copilot_narrative", "infrastructure_sub_bot", "high", "Recommend the safest next command from current health, intent, and operating mode.", ["commands_hygiene", "operator_cockpit", "global_halt_status"]),
    ("report_readiness_editor_guard", "Report Readiness Editor Guard", "operator_copilot_narrative", "infrastructure_sub_bot", "high", "Guard external-facing reports for stale metrics, broken PDFs, weak narrative, and mismatched labels.", ["report_quality_guard", "system_explainers", "strategy_inventory"]),
    ("sleeve_master_vote_harmonizer", "Sleeve Master Vote Harmonizer", "grandmaster_collective_intelligence", "infrastructure_sub_bot", "critical", "Normalize sleeve master votes into a comparable, evidence-weighted packet.", ["sleeve_masters", "coordination_intelligence", "grandmaster_control"]),
    ("grandmaster_context_packet_builder", "Grand Master Context Packet Builder", "grandmaster_collective_intelligence", "infrastructure_sub_bot", "critical", "Build compact Grand Master packets from self-model, resource, alpha, and scenario state.", ["grandmaster_self_awareness_bridge", "system_self_model", "apex_intelligence"]),
    ("cross_sleeve_disagreement_referee", "Cross-Sleeve Disagreement Referee", "grandmaster_collective_intelligence", "infrastructure_sub_bot", "critical", "Referee conflicts between sleeves before any allocation, training, or confidence lift.", ["coordination_intelligence", "portfolio_exposure", "ensemble_uncertainty"]),
    ("collective_intelligence_health_guard", "Collective Intelligence Health Guard", "grandmaster_collective_intelligence", "infrastructure_sub_bot", "critical", "Guard the intelligence stack for stale, circular, overconfident, or resource-heavy conclusions.", ["advanced_intelligence_mesh", "cognitive_control_plane", "safety_invariants"]),
    ("frontier_model_watchlist_curator", "Frontier Model Watchlist Curator", "adaptive_research_frontier", "signal_sub_bot", "medium", "Track advanced model ideas and library candidates without promoting them before evidence exists.", ["research_foundry", "library_utilization_router", "mlx_intelligence_router"]),
    ("experiment_design_frontier_scout", "Experiment Design Frontier Scout", "adaptive_research_frontier", "signal_sub_bot", "medium", "Turn frontier concepts into small, testable, resource-bounded experiments.", ["experiment_ledger", "active_learning", "recursive_research_foundry"]),
    ("research_to_bot_blueprint_guard", "Research-to-Bot Blueprint Guard", "adaptive_research_frontier", "infrastructure_sub_bot", "critical", "Require source, label, data, safety, and resource contracts before frontier ideas become bots.", ["new_bot_admission_guard", "source_verification", "training_quality"]),
]

BOTS = [
    {
        "role_slug": role_slug,
        "slug": f"apex_{role_slug}_bot",
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
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "new_sleeve_or_subsleeve": {
            "sleeve_family": SLEEVE_FAMILY,
            "sleeve_profile": SLEEVE_PROFILE,
            "display_name": PACK_DISPLAY_NAME,
        },
        "system_count": len(SYSTEMS),
        "bot_count": len(BOTS),
        "bot_pack_size_rule": "46_bots_to_reach_1000_total_platform_bots_from_954",
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "apex_intelligence_hot_7d_warm_90d_cold_365d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": 18,
            "capture_mode": "sampled_digest_first_control_trace",
            "sample_rate": 0.12,
            "dedupe_required": True,
            "stale_deletion_policy": "retain_decision_and_control_digests_stage_raw_traces",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "sleeve_master_bot_id": assigned_ids.get(f"{PACK_SLUG}_self_model_state_vector_builder", ""),
        "regression_guard_bot_id": assigned_ids.get(f"{PACK_SLUG}_collective_intelligence_health_guard", ""),
        "grandmaster_bridge_bot_id": assigned_ids.get(f"{PACK_SLUG}_grandmaster_context_packet_builder", ""),
        "resource_governor_bot_id": assigned_ids.get(f"{PACK_SLUG}_thousand_bot_runtime_smoother", ""),
        "runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "self_awareness_advancements": [
            "deep_self_state_vectors",
            "live_assumption_inventory",
            "internal_contradiction_detection",
            "operator_intent_memory",
            "grandmaster_context_packets",
        ],
        "intelligence_advancements": [
            "meta_reasoning_policy",
            "experience_memory_os",
            "world_model_scenario_oracle",
            "autonomous_upgrade_foundry",
            "alpha_safety_causal_judge",
            "adaptive_research_frontier",
        ],
        "global_halt_contract": "apex_layers_preemptively_downshift_to_sampled_digest_mode_before_global_halt_pressure",
        "paper_lock_contract": "no_direct_execution_no_allocation_no_training_until_120_days_30000_observations_and_safety_gates_clear",
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    system = _system(bot)
    contract = _pack_contract(assigned_ids)
    apex_contract = {
        "contract_version": "apex_self_awareness_intelligence_layers_v1",
        "capability_pack": PACK_SLUG,
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "system": bot["system"],
        "system_display_name": system["display_name"],
        "system_outputs": list(system.get("outputs", [])),
        "reports_to_sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "regression_guard_bot_id": contract["regression_guard_bot_id"],
        "grandmaster_bridge_bot_id": contract["grandmaster_bridge_bot_id"],
        "resource_governor_bot_id": contract["resource_governor_bot_id"],
        "intelligence_advancement": "observe_route_memory_scenario_judge_and_smooth_before_trust_lift",
        "self_awareness_boundary": "system_self_modeling_only_no_claim_of_consciousness_or_autonomous_trading_authority",
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
        "preferred_regimes": ["all_weather", "expansion_pressure", "resource_pressure", "fragile_transition", "incident_recovery"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v894_self_model_identity_cartographer_bot",
            "brain_refinery_v904_alpha_training_graduation_scorer_bot",
            "brain_refinery_v934_intelligence_metacognitive_reasoning_budget_allocator_bot",
            "brain_refinery_v963_intelligence_risk_payoff_backlog_master_bot",
        ],
        "data_intake_collections": list(DATA_INTAKES),
        "storage_targets": list(STORAGE_TARGETS),
        "freshness_slo_seconds": 900,
        "retention_profile": "apex_intelligence_hot_7d_warm_90d_cold_365d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "apex_self_awareness_intelligence_collect_only_until_safety_memory_reasoning_and_resource_gates_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "apex_intelligence_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_apex_intelligence_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "training_threshold_policy": {
            "minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_collection_days": MINIMUM_COLLECTION_DAYS,
            "requires_assumption_inventory_clearance": True,
            "requires_causal_evidence_clearance": True,
            "requires_safety_invariant_clearance": True,
            "requires_memory_quality_clearance": True,
            "requires_runtime_pressure_clear": True,
        },
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "sampled",
        "data_collection_sample_rate": 0.12,
        "data_collection_max_daily_storage_mb": 18,
        "data_collection_max_daily_mb": 18.0,
        "data_collection_compute_guard_mode": "sustain",
        "data_collection_resource_guard_reason": "1000_bot_platform_uses_sampled_digest_first_control_traces",
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "paper_execution_queue_policy": "buffered_jsonl_batching",
        "paper_runtime_control_refresh_seconds": 240,
        "sleeve_profile": SLEEVE_PROFILE,
        "sleeve_family": SLEEVE_FAMILY,
        "correlation_peer_sleeves": [
            "system_self_awareness",
            "intelligence_layer_advancement",
            "alpha_intelligence_evolution",
            "adaptive_intelligence_kernel",
            "advanced_intelligence_mesh",
            "coordination_intelligence",
        ],
        "correlation_dependencies": [
            "system_self_model",
            "operator_cockpit",
            "global_halt_status",
            "runtime_throttle",
            "memory_efficiency",
            "mlx_intelligence_router",
            "library_utilization_router",
        ],
        "provider_capability_profile": "internal_apex_control_plane_only",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "master_bot_registry",
            "system_self_model",
            "operator_cockpit",
            "alpha_intelligence_evolution",
            "intelligence_layer_advancement",
            "training_quality_control",
            "decision_provenance",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "apex_self_awareness_intelligence_has_no_direct_execution_or_broker_dependency",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.8,
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
            f"apex_system:{bot['system']}",
            "apex_self_awareness",
            "apex_intelligence",
            "training_after_threshold",
            "global_halt_aware",
            "1000_bot_platform",
        ],
        "execution_policy_label": "collection_only_apex_intelligence_no_execution",
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
            "system_self_model_awareness",
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
        "apex_self_awareness_intelligence_version": PACK_VERSION,
        "capability_pack_contract": contract,
        "apex_self_awareness_intelligence_contract": apex_contract,
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    pack_rows = [row for row in rows if str(row.get("apex_self_awareness_intelligence_version") or "") == PACK_VERSION]
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
            "apex_self_awareness_intelligence_bot_count": len(pack_rows),
            "latest_apex_self_awareness_intelligence": PACK_VERSION,
            "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
            "target_platform_total_bots_met": len(rows) == TARGET_PLATFORM_TOTAL_BOTS,
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
        "objective": "Push the self-awareness and intelligence layers into a 1000-bot control plane that can observe itself, route reasoning, remember lessons, test scenarios, rank upgrades, judge alpha safety, smooth resources, and brief the operator.",
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
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
        "resource_governor_bot_id": contract["resource_governor_bot_id"],
        "runtime_capacity_floor": contract["runtime_capacity_floor"],
        "self_awareness_advancements": list(contract["self_awareness_advancements"]),
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
        "apex_self_awareness_intelligence_version": PACK_VERSION,
        "system_count": len(SYSTEMS),
        "bot_count": len(BOTS),
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped_existing),
        "planned_total_after_apply": len(rows) + len(planned_rows),
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "planned_reaches_target_total": len(rows) + len(planned_rows) == TARGET_PLATFORM_TOTAL_BOTS,
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
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "planned_total_after_apply": plan["planned_total_after_apply"],
        "planned_reaches_target_total": plan["planned_reaches_target_total"],
        "apex_self_awareness_intelligence_version": PACK_VERSION,
        "system_count": plan["system_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "pack": plan["pack"],
        "recommended_apply_command": "./scripts/ops/opsctl.sh apex-self-awareness-intelligence --apply --json",
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
        backup = backup_dir / f"master_bot_registry_before_apex_self_awareness_intelligence_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
            "target_platform_total_bots_met": len(rows) == TARGET_PLATFORM_TOTAL_BOTS,
            "storage_targets_ready": storage_targets_ready,
        }
    )
    _write_json(
        project_root / "config" / "apex_self_awareness_intelligence_v1.json",
        {
            "generated_at_utc": _utc_now(),
            "apex_self_awareness_intelligence_version": PACK_VERSION,
            "pack": payload["pack"],
        },
    )
    _write_json(project_root / "governance" / "health" / "apex_self_awareness_intelligence_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add the 46-bot apex self-awareness and intelligence pack.")
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
            "apex_self_awareness_intelligence "
            f"mode={payload['mode']} systems={payload['system_count']} "
            f"bots={payload['bot_count']} planned={payload['planned_bot_count']} "
            f"added={payload.get('added_bot_count', 0)} "
            f"target_total={payload.get('planned_total_after_apply') or payload.get('new_total_bots')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
