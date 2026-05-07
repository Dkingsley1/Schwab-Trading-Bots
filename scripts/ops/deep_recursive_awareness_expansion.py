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
BASE_VERSION = 1010
TARGET_PLATFORM_TOTAL_BOTS = 1028
PACK_VERSION = "deep_recursive_awareness_v1"
PACK_SLUG = "deep_recursive_awareness"
PACK_DISPLAY_NAME = "Deep Recursive Awareness Pack"
SLEEVE_FAMILY = "recursive_meta_awareness_control_plane"
SLEEVE_PROFILE = "deep_recursive_awareness"
LABEL_CONTRACT_VERSION = "deep_recursive_awareness_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 36000
MINIMUM_COLLECTION_DAYS = 150
PAPER_RUNTIME_CAPACITY_FLOOR = 1000

SYSTEMS: list[dict[str, Any]] = [
    {
        "slug": "causal_self_diagnosis",
        "display_name": "Causal Self-Diagnosis",
        "objective": "Explain platform incidents as causal chains across halts, feeds, auth, storage, queues, and resource pressure.",
        "outputs": ["causal_incident_graph", "root_cause_rank", "clearance_explanation_packet"],
    },
    {
        "slug": "predictive_runtime_oracle",
        "display_name": "Predictive Runtime Oracle",
        "objective": "Forecast CPU, memory, swap, writer backlog, and feed load before large bot waves or heavy views run.",
        "outputs": ["runtime_pressure_forecast", "launch_preflight_vote", "throttle_shape_plan"],
    },
    {
        "slug": "experience_memory_core",
        "display_name": "Experience Memory Core",
        "objective": "Remember which fixes, restarts, cleanups, and expansions helped or hurt the platform over time.",
        "outputs": ["stabilization_memory_event", "fix_effectiveness_score", "command_failure_lesson"],
    },
    {
        "slug": "self_upgrade_critic_board",
        "display_name": "Self-Upgrade Critic Board",
        "objective": "Propose only evidence-backed upgrades and challenge them for dependency, regression, and payoff risk.",
        "outputs": ["upgrade_red_team_packet", "dependency_risk_map", "regression_guard_plan"],
    },
    {
        "slug": "operator_context_governor",
        "display_name": "Operator Context Governor",
        "objective": "Adapt platform intensity to operator context, especially when creative or development apps need headroom.",
        "outputs": ["operator_context_state", "host_cotenant_mode_vote", "calm_transition_plan"],
    },
    {
        "slug": "internal_critic_board",
        "display_name": "Internal Critic Board",
        "objective": "Challenge overconfidence, circular reasoning, duplicate alpha, and autonomy-boundary violations.",
        "outputs": ["critic_objection_packet", "circular_reasoning_flag", "autonomy_boundary_vote"],
    },
    {
        "slug": "recursive_platform_map",
        "display_name": "Recursive Platform Map",
        "objective": "Keep a live map of bots, sleeves, masters, Grand Master, data sources, storage, guards, reports, and failure modes.",
        "outputs": ["recursive_platform_graph", "sleeve_master_edge_audit", "framework_map_regression_flag"],
    },
]

DATA_INTAKES = [
    "deep_recursive_causal_incident_trace",
    "deep_recursive_halt_feed_auth_storage_trace",
    "deep_recursive_runtime_pressure_forecast_trace",
    "deep_recursive_launch_preflight_trace",
    "deep_recursive_experience_memory_trace",
    "deep_recursive_fix_effectiveness_trace",
    "deep_recursive_upgrade_critic_trace",
    "deep_recursive_dependency_regression_trace",
    "deep_recursive_operator_context_trace",
    "deep_recursive_internal_critic_trace",
    "deep_recursive_platform_map_trace",
]

STORAGE_TARGETS = [
    "governance/deep_recursive_awareness",
    "governance/deep_recursive_awareness/causal_self_diagnosis",
    "governance/deep_recursive_awareness/predictive_runtime_oracle",
    "governance/deep_recursive_awareness/experience_memory_core",
    "governance/deep_recursive_awareness/self_upgrade_critic_board",
    "governance/deep_recursive_awareness/operator_context_governor",
    "governance/deep_recursive_awareness/internal_critic_board",
    "governance/deep_recursive_awareness/recursive_platform_map",
    "governance/health/deep_recursive_awareness_latest.json",
]

REQUIRED_LABELS = [
    "causal_chain_status",
    "root_cause_confidence_bucket",
    "runtime_pressure_forecast_bucket",
    "launch_preflight_status",
    "fix_effectiveness_bucket",
    "upgrade_dependency_risk_bucket",
    "operator_context_mode",
    "critic_objection_status",
    "recursive_map_integrity_bucket",
]

BOTS: list[tuple[str, str, str, str, str, str, list[str]]] = [
    (
        "causal_incident_root_cause_builder",
        "Causal Incident Root Cause Builder",
        "causal_self_diagnosis",
        "infrastructure_sub_bot",
        "critical",
        "Build causal incident graphs from halt, feed, auth, queue, storage, and resource evidence.",
        ["global_halt_status", "incident_timeline", "operator_cockpit"],
    ),
    (
        "global_halt_causal_chain_explainer",
        "Global Halt Causal Chain Explainer",
        "causal_self_diagnosis",
        "infrastructure_sub_bot",
        "critical",
        "Explain whether a halt came from real risk, stale health, backlog pressure, auth, storage, or sleeve collapse.",
        ["global_halt_clear_blockers", "health_gates", "data_plane_recovery_controller"],
    ),
    (
        "backlog_pressure_attribution_scorer",
        "Backlog Pressure Attribution Scorer",
        "causal_self_diagnosis",
        "infrastructure_sub_bot",
        "critical",
        "Attribute writer and feed backlog pressure to sources, sleeves, reporters, or external-drive state.",
        ["backpressure_slo", "writer_cycle_coordinator", "storage_backpressure_autopilot"],
    ),
    (
        "feed_auth_storage_failure_triage_guard",
        "Feed / Auth / Storage Failure Triage Guard",
        "causal_self_diagnosis",
        "infrastructure_sub_bot",
        "critical",
        "Triage live feed gaps by separating provider, auth, storage, and local resource causes.",
        ["schwab_auth_supervisor", "coinbase_api_health", "storage_mount_guard"],
    ),
    (
        "runtime_pressure_forecaster",
        "Runtime Pressure Forecaster",
        "predictive_runtime_oracle",
        "infrastructure_sub_bot",
        "critical",
        "Forecast CPU, memory, swap, backlog, and IO pressure before they become operator-visible problems.",
        ["memory_efficiency", "swap_pressure_governor", "runtime_throttle"],
    ),
    (
        "thousand_bot_launch_preflight_oracle",
        "1000-Bot Launch Preflight Oracle",
        "predictive_runtime_oracle",
        "infrastructure_sub_bot",
        "critical",
        "Produce launch readiness votes before full-force paper collection or large expansion waves.",
        ["expansion_capacity", "core_bot_catalog", "global_halt_preemption"],
    ),
    (
        "adaptive_throttle_shape_planner",
        "Adaptive Throttle Shape Planner",
        "predictive_runtime_oracle",
        "infrastructure_sub_bot",
        "critical",
        "Shape throttles smoothly by sleeve, data source, queue, and host app context.",
        ["runtime_throttle", "creative_cotenant_guard", "ingestion_priority_queue"],
    ),
    (
        "paper_queue_load_smoother",
        "Paper Queue Load Smoother",
        "predictive_runtime_oracle",
        "infrastructure_sub_bot",
        "critical",
        "Smooth paper-trade queue cadence so collection stays useful without spiking writers.",
        ["paper_trade_lock_guard", "writer_cycle_coordinator", "backpressure_drainers"],
    ),
    (
        "stabilization_memory_writer",
        "Stabilization Memory Writer",
        "experience_memory_core",
        "infrastructure_sub_bot",
        "high",
        "Write compact memories of cleanups, restarts, throttles, and guard changes with point-in-time evidence.",
        ["system_self_model", "incident_review_packet", "daily_verify"],
    ),
    (
        "fix_effectiveness_replay_scorer",
        "Fix Effectiveness Replay Scorer",
        "experience_memory_core",
        "signal_sub_bot",
        "high",
        "Score whether a prior fix actually reduced halts, backlog, stale feeds, or swap pressure.",
        ["golden_replay_guard", "one_numbers", "health_gates"],
    ),
    (
        "expansion_pressure_memory_indexer",
        "Expansion Pressure Memory Indexer",
        "experience_memory_core",
        "infrastructure_sub_bot",
        "high",
        "Index which expansion waves increased pressure and which support layers absorbed it.",
        ["expansion_capacity", "apex_self_awareness_intelligence", "core_bot_catalog"],
    ),
    (
        "command_failure_lesson_linker",
        "Command Failure Lesson Linker",
        "experience_memory_core",
        "infrastructure_sub_bot",
        "high",
        "Link command failures to fixes and follow-up regression guards.",
        ["commands_hygiene", "command_validity", "operator_cockpit"],
    ),
    (
        "upgrade_hypothesis_red_team",
        "Upgrade Hypothesis Red Team",
        "self_upgrade_critic_board",
        "infrastructure_sub_bot",
        "critical",
        "Challenge candidate upgrades for weak evidence, hidden resource cost, or safety holes.",
        ["self_improvement_backlog", "safety_invariants_v2", "research_foundry"],
    ),
    (
        "dependency_risk_board",
        "Dependency Risk Board",
        "self_upgrade_critic_board",
        "infrastructure_sub_bot",
        "critical",
        "Map dependency, data, command, and library risk before upgrades are applied.",
        ["dependency_utilization", "library_utilization_router", "commands_hygiene"],
    ),
    (
        "regression_guard_design_planner",
        "Regression Guard Design Planner",
        "self_upgrade_critic_board",
        "infrastructure_sub_bot",
        "critical",
        "Design focused regression guards before touching halt, feed, storage, training, or reporting contracts.",
        ["regression_guard", "system_drift_guard", "daily_verify"],
    ),
    (
        "payoff_evidence_gatekeeper",
        "Payoff Evidence Gatekeeper",
        "self_upgrade_critic_board",
        "infrastructure_sub_bot",
        "critical",
        "Require measurable payoff evidence before self-upgrade ideas become implementation work.",
        ["optimization_recommendation_ranker", "experiment_ledger", "alpha_benchmark_suite"],
    ),
    (
        "host_cotenant_context_mapper",
        "Host Co-Tenant Context Mapper",
        "operator_context_governor",
        "infrastructure_sub_bot",
        "critical",
        "Map local app context and reserve headroom for PyCharm, browsers, Logic Pro, and Final Cut.",
        ["creative_cotenant_guard", "memory_efficiency", "operator_control"],
    ),
    (
        "creative_app_mode_governor",
        "Creative App Mode Governor",
        "operator_context_governor",
        "infrastructure_sub_bot",
        "critical",
        "Move nonessential collection, reporters, and reasoning into calmer modes while creative apps are active.",
        ["creative_cotenant_guard", "runtime_throttle", "chrome_headless_guard"],
    ),
    (
        "operator_preference_state_bridge",
        "Operator Preference State Bridge",
        "operator_context_governor",
        "infrastructure_sub_bot",
        "high",
        "Bridge repeated operator preferences into command recommendations and runtime modes.",
        ["operator_copilot", "commands_hygiene", "system_summary"],
    ),
    (
        "calm_mode_transition_controller",
        "Calm Mode Transition Controller",
        "operator_context_governor",
        "infrastructure_sub_bot",
        "critical",
        "Transition smoothly between full-force collection, digest mode, creative mode, and recovery mode.",
        ["mode_switchboard", "runtime_throttle", "global_halt_preemption"],
    ),
    (
        "overconfidence_challenge_board",
        "Overconfidence Challenge Board",
        "internal_critic_board",
        "infrastructure_sub_bot",
        "critical",
        "Challenge overconfident conclusions when evidence is thin, stale, or circular.",
        ["critic_board", "ensemble_uncertainty", "report_quality_guard"],
    ),
    (
        "circular_reasoning_detector",
        "Circular Reasoning Detector",
        "internal_critic_board",
        "infrastructure_sub_bot",
        "critical",
        "Detect when self-model outputs are being reused as evidence for themselves.",
        ["system_self_model", "decision_provenance", "causal_evidence_gatekeeper"],
    ),
    (
        "duplicate_sleeve_alpha_referee",
        "Duplicate Sleeve Alpha Referee",
        "internal_critic_board",
        "infrastructure_sub_bot",
        "critical",
        "Challenge sleeve or bot additions that duplicate existing alpha behavior under a new label.",
        ["duplicate_alpha_novelty", "sleeve_strategy_coverage", "core_bot_catalog"],
    ),
    (
        "unsafe_autonomy_boundary_guard",
        "Unsafe Autonomy Boundary Guard",
        "internal_critic_board",
        "infrastructure_sub_bot",
        "critical",
        "Guard recursive awareness layers from implying consciousness, authority, or execution permission.",
        ["safety_invariants_v2", "paper_trade_lock_guard", "operator_release"],
    ),
    (
        "recursive_platform_graph_builder",
        "Recursive Platform Graph Builder",
        "recursive_platform_map",
        "infrastructure_sub_bot",
        "critical",
        "Build the living graph from bots to sleeves, masters, data sources, storage, guards, reports, and incidents.",
        ["core_bot_catalog", "system_self_model", "framework_map"],
    ),
    (
        "sleeve_master_grandmaster_edge_auditor",
        "Sleeve Master / Grand Master Edge Auditor",
        "recursive_platform_map",
        "infrastructure_sub_bot",
        "critical",
        "Audit the edges between sub bots, sleeve masters, Grand Master packets, and vote harmonizers.",
        ["sleeve_masters", "grandmaster_collective_intelligence", "coordination_intelligence"],
    ),
    (
        "data_source_storage_route_mapper",
        "Data Source / Storage Route Mapper",
        "recursive_platform_map",
        "infrastructure_sub_bot",
        "critical",
        "Map how every source routes into shards, cold storage, reports, health files, and retention controls.",
        ["data_retention", "sql_audit", "storage_resilience"],
    ),
    (
        "living_framework_map_regression_guard",
        "Living Framework Map Regression Guard",
        "recursive_platform_map",
        "infrastructure_sub_bot",
        "critical",
        "Keep framework-map and system-overview reports aligned with registry truth after expansions.",
        ["report_quality_guard", "system_explainers", "strategy_inventory"],
    ),
]

BOTS = [
    {
        "role_slug": role_slug,
        "slug": f"recursive_awareness_{role_slug}_bot",
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
        "bot_pack_size_rule": "28_bots_deep_recursive_awareness_layer_after_1000_bot_apex_pack",
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "deep_recursive_awareness_hot_7d_warm_120d_cold_540d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": 14,
            "capture_mode": "sampled_digest_first_causal_trace",
            "sample_rate": 0.1,
            "dedupe_required": True,
            "stale_deletion_policy": "retain_causal_decision_digests_stage_raw_recursive_traces",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "causal_diagnosis_bot_id": assigned_ids.get(f"{PACK_SLUG}_global_halt_causal_chain_explainer", ""),
        "runtime_oracle_bot_id": assigned_ids.get(f"{PACK_SLUG}_runtime_pressure_forecaster", ""),
        "experience_memory_bot_id": assigned_ids.get(f"{PACK_SLUG}_stabilization_memory_writer", ""),
        "upgrade_critic_bot_id": assigned_ids.get(f"{PACK_SLUG}_upgrade_hypothesis_red_team", ""),
        "operator_context_bot_id": assigned_ids.get(f"{PACK_SLUG}_host_cotenant_context_mapper", ""),
        "internal_critic_bot_id": assigned_ids.get(f"{PACK_SLUG}_overconfidence_challenge_board", ""),
        "recursive_map_bot_id": assigned_ids.get(f"{PACK_SLUG}_recursive_platform_graph_builder", ""),
        "runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "awareness_advancements": [
            "causal_self_diagnosis",
            "predictive_self_regulation",
            "experience_memory_core",
            "self_upgrade_reasoning",
            "operator_aware_runtime",
            "internal_critic_board",
            "recursive_platform_map",
        ],
        "global_halt_contract": "deep_recursive_awareness_explains_and_preempts_halts_but_never_force_clears_without_existing_guard_contracts",
        "paper_lock_contract": "no_direct_execution_no_allocation_no_training_until_150_days_36000_observations_and_recursive_awareness_gates_clear",
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    system = _system(bot)
    contract = _pack_contract(assigned_ids)
    deep_contract = {
        "contract_version": "deep_recursive_awareness_layers_v1",
        "capability_pack": PACK_SLUG,
        "system": bot["system"],
        "system_display_name": system["display_name"],
        "system_outputs": list(system.get("outputs", [])),
        "causal_diagnosis_bot_id": contract["causal_diagnosis_bot_id"],
        "runtime_oracle_bot_id": contract["runtime_oracle_bot_id"],
        "experience_memory_bot_id": contract["experience_memory_bot_id"],
        "upgrade_critic_bot_id": contract["upgrade_critic_bot_id"],
        "operator_context_bot_id": contract["operator_context_bot_id"],
        "internal_critic_bot_id": contract["internal_critic_bot_id"],
        "recursive_map_bot_id": contract["recursive_map_bot_id"],
        "recursive_awareness_boundary": "system_self_modeling_and_control_plane_advice_only_no_consciousness_claim_no_execution_authority",
    }
    return {
        "bot_id": bot_id,
        "bot_role": bot["bot_role"],
        "active": True,
        "reason": "deep_recursive_awareness_expansion_slot",
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
        "promotion_reason": "deep_recursive_awareness_expansion_slot",
        "model_path": "",
        "log_file": "",
        "candidate_log_file": "",
        "lifecycle_state": "data_collection_only",
        "slot_label": bot["label"],
        "slot_kind": _slot_kind(bot),
        "slot_priority": bot["priority"],
        "slot_objective": bot["objective"],
        "target_functions": list(bot["target_functions"]),
        "preferred_regimes": [
            "global_halt_recovery",
            "backpressure_spike",
            "heavy_operator_workload",
            "rapid_expansion",
            "feed_gap_recovery",
            "resource_pressure",
        ],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v934_intelligence_metacognitive_reasoning_budget_allocator_bot",
            "brain_refinery_v964_apex_self_model_state_vector_builder_bot",
            "brain_refinery_v984_apex_upgrade_hypothesis_generator_bot",
            "brain_refinery_v998_apex_thousand_bot_runtime_smoother_bot",
        ],
        "data_intake_collections": list(DATA_INTAKES),
        "storage_targets": list(STORAGE_TARGETS),
        "freshness_slo_seconds": 600,
        "retention_profile": "deep_recursive_awareness_hot_7d_warm_120d_cold_540d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "deep_recursive_awareness_collect_only_until_causal_runtime_memory_critic_and_operator_context_gates_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "deep_recursive_awareness_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_recursive_awareness_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "training_threshold_policy": {
            "minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_collection_days": MINIMUM_COLLECTION_DAYS,
            "requires_causal_diagnosis_quality_clearance": True,
            "requires_runtime_forecast_clearance": True,
            "requires_experience_memory_clearance": True,
            "requires_critic_board_clearance": True,
            "requires_operator_context_clearance": True,
            "requires_recursive_map_integrity_clearance": True,
        },
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "sampled",
        "data_collection_sample_rate": 0.1,
        "data_collection_max_daily_storage_mb": 14,
        "data_collection_max_daily_mb": 14.0,
        "data_collection_compute_guard_mode": "sustain",
        "data_collection_resource_guard_reason": "deep_recursive_awareness_uses_digest_first_control_traces_for_1000_plus_bot_platform",
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "paper_execution_queue_policy": "buffered_jsonl_batching",
        "paper_runtime_control_refresh_seconds": 180,
        "sleeve_profile": SLEEVE_PROFILE,
        "sleeve_family": SLEEVE_FAMILY,
        "correlation_peer_sleeves": [
            "system_self_awareness",
            "intelligence_layer_advancement",
            "apex_self_awareness_intelligence",
            "adaptive_intelligence_kernel",
            "coordination_intelligence",
        ],
        "correlation_dependencies": [
            "system_self_model",
            "operator_cockpit",
            "global_halt_status",
            "runtime_throttle",
            "memory_efficiency",
            "backpressure_slo",
            "core_bot_catalog",
        ],
        "provider_capability_profile": "internal_recursive_awareness_control_plane_only",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "master_bot_registry",
            "governance_health",
            "logs",
            "system_self_model",
            "operator_cockpit",
            "apex_self_awareness_intelligence",
            "decision_provenance",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "deep_recursive_awareness_has_no_direct_broker_dependency_or_execution_authority",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.82,
            "freshness_slo_seconds": 600,
            "regression_guard_bot_id": contract["internal_critic_bot_id"],
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
            f"recursive_system:{bot['system']}",
            "deep_recursive_awareness",
            "causal_self_diagnosis",
            "predictive_self_regulation",
            "training_after_threshold",
            "global_halt_aware",
            "operator_context_aware",
            "1000_plus_bot_platform",
        ],
        "execution_policy_label": "collection_only_deep_recursive_awareness_no_execution",
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
            "recursive_awareness_boundary",
        ],
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "paper_runtime_stability_mode": "digest_first_recursive_awareness",
        "paper_trade_lock_required": True,
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "capability_pack_version": PACK_VERSION,
        "capability_pack_slug": PACK_SLUG,
        "capability_pack_display_name": PACK_DISPLAY_NAME,
        "deep_recursive_awareness_version": PACK_VERSION,
        "capability_pack_contract": contract,
        "deep_recursive_awareness_contract": deep_contract,
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    pack_rows = [row for row in rows if str(row.get("deep_recursive_awareness_version") or "") == PACK_VERSION]
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
            "deep_recursive_awareness_bot_count": len(pack_rows),
            "latest_deep_recursive_awareness": PACK_VERSION,
            "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
            "target_platform_total_bots_met": len(rows) >= TARGET_PLATFORM_TOTAL_BOTS,
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
        "objective": "Add a deeper recursive awareness layer that diagnoses causes, predicts runtime pressure, remembers what worked, critiques upgrades, adapts to operator context, challenges overconfidence, and keeps a living platform map.",
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "system_count": len(SYSTEMS),
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "systems": list(SYSTEMS),
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "runtime_capacity_floor": contract["runtime_capacity_floor"],
        "awareness_advancements": list(contract["awareness_advancements"]),
        "anchor_bot_ids": {
            "causal_diagnosis": contract["causal_diagnosis_bot_id"],
            "runtime_oracle": contract["runtime_oracle_bot_id"],
            "experience_memory": contract["experience_memory_bot_id"],
            "upgrade_critic": contract["upgrade_critic_bot_id"],
            "operator_context": contract["operator_context_bot_id"],
            "internal_critic": contract["internal_critic_bot_id"],
            "recursive_map": contract["recursive_map_bot_id"],
        },
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
        "deep_recursive_awareness_version": PACK_VERSION,
        "system_count": len(SYSTEMS),
        "bot_count": len(BOTS),
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped_existing),
        "planned_total_after_apply": len(rows) + len(planned_rows),
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "planned_reaches_target_total": len(rows) + len(planned_rows) >= TARGET_PLATFORM_TOTAL_BOTS,
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
        "deep_recursive_awareness_version": PACK_VERSION,
        "system_count": plan["system_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "pack": plan["pack"],
        "recommended_apply_command": "./scripts/ops/opsctl.sh deep-recursive-awareness --apply --json",
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
        backup = backup_dir / f"master_bot_registry_before_deep_recursive_awareness_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
            "target_platform_total_bots_met": len(rows) >= TARGET_PLATFORM_TOTAL_BOTS,
            "storage_targets_ready": storage_targets_ready,
        }
    )
    _write_json(
        project_root / "config" / "deep_recursive_awareness_v1.json",
        {
            "generated_at_utc": _utc_now(),
            "deep_recursive_awareness_version": PACK_VERSION,
            "pack": payload["pack"],
        },
    )
    _write_json(project_root / "governance" / "health" / "deep_recursive_awareness_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add the 28-bot deep recursive awareness control-plane pack.")
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
            "deep_recursive_awareness "
            f"mode={payload['mode']} systems={payload['system_count']} "
            f"bots={payload['bot_count']} planned={payload['planned_bot_count']} "
            f"added={payload.get('added_bot_count', 0)} "
            f"target_total={payload.get('planned_total_after_apply') or payload.get('new_total_bots')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
