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
BASE_VERSION = 1478
TARGET_PLATFORM_TOTAL_BOTS = 1548
PACK_VERSION = "quant_operational_intelligence_v1"
PACK_SLUG = "quant_operational_intelligence"
PACK_DISPLAY_NAME = "Quant And Operational Intelligence Pack"
SLEEVE_FAMILY = "quant_operational_intelligence"
LABEL_CONTRACT_VERSION = "quant_operational_intelligence_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 65000
MINIMUM_COLLECTION_DAYS = 160
PAPER_RUNTIME_CAPACITY_FLOOR = 1000
SAMPLE_RATE = 0.018
MAX_DAILY_MB_PER_BOT = 3


INTELLIGENCE_SYSTEMS: list[dict[str, Any]] = [
    {
        "slug": "alpha_factor_court",
        "domain": "quant",
        "display_name": "Alpha Factor Court",
        "objective": "Judge factor candidates with evidence quality, regime dependency, turnover cost, duplication risk, and promotion readiness.",
        "outputs": ["factor_case_packet", "alpha_evidence_verdict", "promotion_readiness_vote"],
        "peer_systems": ["alpha_evidence_court", "duplicate_alpha_novelty", "model_governance_board"],
        "data_intakes": ["factor_candidate_trace", "alpha_evidence_trace", "promotion_gate_trace"],
    },
    {
        "slug": "cross_asset_lead_lag_map",
        "domain": "quant",
        "display_name": "Cross-Asset Lead Lag Map",
        "objective": "Map lead-lag edges across equities, rates, credit, FX, commodities, options, and crypto under changing regimes.",
        "outputs": ["lead_lag_graph", "cross_asset_causality_score", "regime_transfer_packet"],
        "peer_systems": ["cross_asset_risk_transfer_lab", "regime_transition_engine", "market_regime_router"],
        "data_intakes": ["cross_asset_return_trace", "regime_label_trace", "event_calendar_trace"],
    },
    {
        "slug": "regime_conditional_edge_lab",
        "domain": "quant",
        "display_name": "Regime Conditional Edge Lab",
        "objective": "Separate durable edge from regime-specific edge using volatility, liquidity, macro, event, and crowding states.",
        "outputs": ["regime_edge_matrix", "conditional_alpha_vote", "regime_overfit_alert"],
        "peer_systems": ["regime_router", "uncertainty_risk_calibration", "scenario_generation_synthetic_markets_v2"],
        "data_intakes": ["regime_transition_trace", "alpha_outcome_trace", "liquidity_state_trace"],
    },
    {
        "slug": "slippage_capacity_alpha_adjuster",
        "domain": "quant",
        "display_name": "Slippage Capacity Alpha Adjuster",
        "objective": "Haircut alpha by estimated fill quality, slippage, capacity, adverse selection, spread width, and market impact.",
        "outputs": ["alpha_after_cost_packet", "capacity_haircut_curve", "slippage_guard_vote"],
        "peer_systems": ["execution_quality_lab_v2", "execution_realism_layer", "liquidity_stress_market_impact_lab"],
        "data_intakes": ["paper_fill_trace", "spread_cost_trace", "market_impact_trace"],
    },
    {
        "slug": "portfolio_risk_budget_optimizer",
        "domain": "quant",
        "display_name": "Portfolio Risk Budget Optimizer",
        "objective": "Allocate research and paper attention by marginal utility, correlation crowding, capital pressure, and tail exposure.",
        "outputs": ["risk_budget_packet", "marginal_utility_rank", "crowding_rebalance_vote"],
        "peer_systems": ["portfolio_intelligence_layer", "portfolio_brain", "funding_collateral_margin_intelligence"],
        "data_intakes": ["portfolio_exposure_trace", "correlation_governor_trace", "margin_pressure_trace"],
    },
    {
        "slug": "tail_hedge_convexity_lab",
        "domain": "quant",
        "display_name": "Tail Hedge Convexity Lab",
        "objective": "Score hedges by convexity, carry drag, crash responsiveness, liquidity, basis risk, and scenario fit.",
        "outputs": ["tail_hedge_scorecard", "convexity_cost_curve", "crisis_response_packet"],
        "peer_systems": ["options_risk_intelligence_v2", "macro_crisis_scenario_lab", "formal_safety_verification"],
        "data_intakes": ["option_surface_trace", "stress_replay_trace", "tail_event_trace"],
    },
    {
        "slug": "model_uncertainty_calibration",
        "domain": "quant",
        "display_name": "Model Uncertainty Calibration",
        "objective": "Calibrate predictive confidence, ensemble disagreement, outcome variance, and drawdown risk before promotion.",
        "outputs": ["uncertainty_calibration_packet", "confidence_reliability_curve", "ensemble_disagreement_alert"],
        "peer_systems": ["ensemble_uncertainty", "model_governance_board", "training_quality_control"],
        "data_intakes": ["prediction_outcome_trace", "ensemble_vote_trace", "calibration_bucket_trace"],
    },
    {
        "slug": "feature_decay_drift_watch",
        "domain": "quant",
        "display_name": "Feature Decay And Drift Watch",
        "objective": "Track feature stability, label drift, missingness, provider drift, leakage risk, and decay before retraining.",
        "outputs": ["feature_decay_curve", "label_drift_alert", "retrain_or_retire_vote"],
        "peer_systems": ["feature_store_dataset_registry", "data_quality_observatory", "model_decay_detector"],
        "data_intakes": ["feature_lineage_trace", "label_quality_trace", "provider_drift_trace"],
    },
    {
        "slug": "synthetic_scenario_alpha_lab",
        "domain": "quant",
        "display_name": "Synthetic Scenario Alpha Lab",
        "objective": "Stress strategy ideas through synthetic market scenarios, bootstrapped paths, crash windows, and liquidity droughts.",
        "outputs": ["synthetic_scenario_result", "path_robustness_score", "stress_survival_vote"],
        "peer_systems": ["replay_scenario_lab_v2", "scenario_generation_synthetic_markets_v2", "golden_replay_regression"],
        "data_intakes": ["scenario_path_trace", "replay_hash_trace", "stress_driver_trace"],
    },
    {
        "slug": "execution_microstructure_alpha_router",
        "domain": "quant",
        "display_name": "Execution Microstructure Alpha Router",
        "objective": "Route candidate alpha to auction, passive, aggressive, spread, and event execution simulations before paper promotion.",
        "outputs": ["execution_route_vote", "microstructure_fit_score", "paper_route_contract"],
        "peer_systems": ["order_flow_market_microstructure", "execution_realism_layer", "passive_liquidity_provision_sim"],
        "data_intakes": ["auction_trace", "order_book_proxy_trace", "execution_sim_trace"],
    },
    {
        "slug": "incident_causal_replay_router",
        "domain": "operational",
        "display_name": "Incident Causal Replay Router",
        "objective": "Turn incidents, halts, auth decay, storage pressure, writer waits, and feed issues into replayable causal chains.",
        "outputs": ["incident_causal_chain", "replay_command_packet", "root_cause_confidence"],
        "peer_systems": ["system_self_intelligence", "failure_memory", "halt_recovery_intelligence"],
        "data_intakes": ["incident_timeline_trace", "self_intelligence_memory_trace", "global_halt_trace"],
    },
    {
        "slug": "backlog_outcome_verifier",
        "domain": "operational",
        "display_name": "Backlog Outcome Verifier",
        "objective": "Verify whether pressure relief, writer cycles, and drainer waves actually reduce pending lines and pressure index.",
        "outputs": ["drain_effect_verdict", "backlog_clearance_delta", "playbook_change_vote"],
        "peer_systems": ["backpressure_storage_brain_v2", "drainer_intelligence_layer", "system_self_intelligence"],
        "data_intakes": ["ingestion_storage_trace", "writer_cycle_trace", "drainer_wave_trace"],
    },
    {
        "slug": "safe_command_router",
        "domain": "operational",
        "display_name": "Safe Command Router",
        "objective": "Map each blocked surface to one bounded command, required prechecks, post-verifiers, and do-not-do constraints.",
        "outputs": ["safe_command_packet", "precheck_contract", "post_verify_sequence"],
        "peer_systems": ["codex_handoff", "operator_cockpit", "system_brain"],
        "data_intakes": ["codex_handoff_trace", "opsctl_command_contract_trace", "health_surface_trace"],
    },
    {
        "slug": "capacity_forecast_scheduler",
        "domain": "operational",
        "display_name": "Capacity Forecast Scheduler",
        "objective": "Forecast CPU, memory, swap, MLX, IO, writer, and report pressure so heavy work can move to safe windows.",
        "outputs": ["capacity_forecast", "safe_work_window_vote", "throttle_shape_recommendation"],
        "peer_systems": ["runtime_throttle_control", "host_pressure_intelligence", "mlx_intelligence_router"],
        "data_intakes": ["runtime_throttle_trace", "memory_efficiency_trace", "mlx_runtime_trace"],
    },
    {
        "slug": "data_lineage_contract_enforcer",
        "domain": "operational",
        "display_name": "Data Lineage Contract Enforcer",
        "objective": "Enforce point-in-time joins, label lineage, schema compatibility, retention class, and source confidence contracts.",
        "outputs": ["lineage_contract_status", "schema_compatibility_alert", "label_join_guard_vote"],
        "peer_systems": ["feature_store_dataset_registry", "data_quality_v2", "point_in_time_event_store"],
        "data_intakes": ["feature_store_lineage_trace", "schema_migration_trace", "label_contract_trace"],
    },
    {
        "slug": "service_dependency_map",
        "domain": "operational",
        "display_name": "Service Dependency Map",
        "objective": "Map health surfaces, launchers, writer lanes, collectors, reports, and storage paths into dependency edges.",
        "outputs": ["dependency_graph_delta", "blocked_edge_packet", "last_good_surface_map"],
        "peer_systems": ["system_self_model", "dependency_memory", "master_infrastructure_supervisor"],
        "data_intakes": ["dependency_memory_trace", "process_watchdog_trace", "artifact_freshness_trace"],
    },
    {
        "slug": "storage_memory_cotenant_balancer",
        "domain": "operational",
        "display_name": "Storage Memory Cotenant Balancer",
        "objective": "Balance storage drains, memory pressure, foreground apps, external volume state, and runtime throttle posture.",
        "outputs": ["cotenant_balance_packet", "storage_memory_tradeoff_vote", "pressure_relief_sequence"],
        "peer_systems": ["memory_efficiency_control", "pressure_relief_control", "storage_tier_policy"],
        "data_intakes": ["memory_pressure_trace", "storage_pressure_trace", "cotenant_awareness_trace"],
    },
    {
        "slug": "paper_live_separation_auditor",
        "domain": "operational",
        "display_name": "Paper Live Separation Auditor",
        "objective": "Continuously audit that expansion, quant, training, and reporting bots cannot add live execution authority.",
        "outputs": ["separation_contract_status", "execution_authority_diff", "unsafe_promotion_alert"],
        "peer_systems": ["live_runtime_separation", "paper_trade_lock_guard", "global_halt"],
        "data_intakes": ["registry_diff_trace", "paper_lock_trace", "execution_policy_trace"],
    },
    {
        "slug": "report_signal_freshness_sentinel",
        "domain": "operational",
        "display_name": "Report Signal Freshness Sentinel",
        "objective": "Watch reports, markdown, JSON health artifacts, PDF bundles, and dashboards for stale or conflicting signals.",
        "outputs": ["freshness_slo_packet", "report_staleness_alert", "surface_conflict_vote"],
        "peer_systems": ["artifact_freshness_slo", "report_quality_guard", "system_signal_bus"],
        "data_intakes": ["artifact_freshness_trace", "report_quality_trace", "signal_bus_trace"],
    },
    {
        "slug": "operator_decision_packet_builder",
        "domain": "operational",
        "display_name": "Operator Decision Packet Builder",
        "objective": "Compress causal root, effect memory, safe commands, blockers, capability gaps, and next actions into one operator packet.",
        "outputs": ["operator_decision_packet", "attention_queue_rank", "decision_reason_codes"],
        "peer_systems": ["operator_cockpit_v2", "codex_handoff", "system_self_model"],
        "data_intakes": ["operator_cockpit_trace", "codex_handoff_trace", "self_summary_trace"],
    },
]


ROLE_TEMPLATES: list[dict[str, Any]] = [
    {"suffix": "telemetry_collector", "label": "Telemetry Collector", "bot_role": "infrastructure_sub_bot", "priority": "high"},
    {"suffix": "hypothesis_modeler", "label": "Hypothesis Modeler", "bot_role": "signal_sub_bot", "priority": "high"},
    {"suffix": "policy_guard", "label": "Policy Guard", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
    {"suffix": "master_bridge", "label": "Master Bridge", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
]


BASE_DATA_INTAKES = [
    "system_self_intelligence_trace",
    "whole_system_intelligence_trace",
    "quant_model_control_trace",
    "platform_organ_systems_trace",
    "institutional_alpha_validation_trace",
    "runtime_pressure_trace",
    "backpressure_causal_trace",
    "paper_live_safety_trace",
]

REQUIRED_LABELS = [
    "quant_edge_quality_bucket",
    "operational_effectiveness_bucket",
    "causal_root_confidence_bucket",
    "action_effect_verdict_bucket",
    "runtime_pressure_bucket",
    "data_lineage_status",
    "paper_live_separation_status",
    "promotion_gate_status",
]

STORAGE_TARGETS = [
    "governance/quant_operational_intelligence",
    *[f"governance/quant_operational_intelligence/{system['slug']}" for system in INTELLIGENCE_SYSTEMS],
    "governance/health/quant_operational_intelligence_latest.json",
]


def _bot_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for system in INTELLIGENCE_SYSTEMS:
        for role in ROLE_TEMPLATES:
            role_slug = f"{system['slug']}_{role['suffix']}"
            specs.append(
                {
                    "role_slug": role_slug,
                    "slug": f"quant_operational_{role_slug}_bot",
                    "label": f"{system['display_name']} {role['label']}",
                    "system": system["slug"],
                    "domain": system["domain"],
                    "bot_role": role["bot_role"],
                    "priority": role["priority"],
                    "objective": f"{role['label']} for {system['objective']}",
                    "target_functions": list(system.get("outputs", [])),
                }
            )
    return specs


BOTS = _bot_specs()


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
        if desired not in used_versions:
            version = desired
            used_versions.add(version)
        else:
            version = _next_available_version(used_versions, max(max(used_versions, default=BASE_VERSION - 1) + 1, desired))
        assigned[slot] = f"brain_refinery_v{version}_{bot['slug']}"
    return assigned


def _system(bot: dict[str, Any]) -> dict[str, Any]:
    for system in INTELLIGENCE_SYSTEMS:
        if system["slug"] == bot["system"]:
            return system
    return {"slug": bot["system"], "domain": bot.get("domain", ""), "display_name": bot["system"], "objective": bot["objective"], "outputs": []}


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
        "sleeve_family": SLEEVE_FAMILY,
        "display_name": PACK_DISPLAY_NAME,
        "system_count": len(INTELLIGENCE_SYSTEMS),
        "quant_system_count": sum(1 for system in INTELLIGENCE_SYSTEMS if system["domain"] == "quant"),
        "operational_system_count": sum(1 for system in INTELLIGENCE_SYSTEMS if system["domain"] == "operational"),
        "bot_count": len(BOTS),
        "bot_pack_size_rule": "20_intelligence_systems_4_bots_each_80_bot_quant_operational_layer",
        "intelligence_systems": [system["slug"] for system in INTELLIGENCE_SYSTEMS],
        "dedicated_data_intake": list(BASE_DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "quant_operational_intelligence_hot_3d_warm_120d_cold_540d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": MAX_DAILY_MB_PER_BOT,
            "capture_mode": "thin_digest_first_quant_operational_trace",
            "sample_rate": SAMPLE_RATE,
            "dedupe_required": True,
            "stale_deletion_policy": "retain_verdicts_routes_and_effect_scores_stage_raw_traces",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "paper_trading_enabled": False,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "anchor_bot_ids": {
            bot["system"]: assigned_ids.get(_slot_kind(bot), "")
            for bot in BOTS
            if bot["role_slug"].endswith("telemetry_collector")
        },
        "runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "global_halt_contract": "quant_operational_intelligence_can_explain_and_route_blockers_but_never_force_clear_halts",
        "paper_lock_contract": "no_execution_no_allocation_no_training_until_160_days_65000_observations_and_quant_ops_gates_clear",
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    system = _system(bot)
    contract = _pack_contract(assigned_ids)
    system_slug = str(system["slug"])
    domain = str(system["domain"])
    data_intakes = list(BASE_DATA_INTAKES) + list(system.get("data_intakes", [])) + [
        f"{system_slug}_effect_trace",
        f"{system_slug}_label_quality_trace",
    ]
    peer_sleeves = [
        "system_self_intelligence",
        "whole_system_intelligence",
        "platform_organ_systems",
        "quant_strategy_gap",
        "institutional_alpha_validation",
        "frontier_intelligence",
        *list(system.get("peer_systems", [])),
    ]
    system_contract = {
        "contract_version": "quant_operational_intelligence_layers_v1",
        "capability_pack": PACK_SLUG,
        "intelligence_system": system_slug,
        "intelligence_domain": domain,
        "system_display_name": system["display_name"],
        "system_outputs": list(system.get("outputs", [])),
        "authority_boundary": "collection_only_advisory_no_execution_no_allocation_no_halt_clearance",
        "pressure_boundary": "thin_digest_storage_low_compute_collect_only",
        "integration_contract": "feeds_quant_model_control_platform_organs_system_self_intelligence_and_codex_handoff_as_advisory_evidence",
    }
    return {
        "bot_id": bot_id,
        "bot_role": bot["bot_role"],
        "active": True,
        "reason": "quant_operational_intelligence_expansion_slot",
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
        "promotion_reason": "quant_operational_intelligence_expansion_slot",
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
            "normal_collection",
            "market_hours_pressure",
            "post_expansion_settlement",
            "overnight_drain",
            "stress_replay_window",
            "global_halt_review",
            "regime_transition",
        ],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v1034_recursive_awareness_recursive_platform_graph_builder_bot",
            "brain_refinery_v1086_institutional_alpha_evidence_court_evidence_collector_bot",
            "brain_refinery_v1136_institutional_backpressure_storage_brain_v2_evidence_collector_bot",
            "brain_refinery_v1326_platform_organ_data_quality_v2_telemetry_collector_bot",
        ],
        "data_intake_collections": data_intakes,
        "storage_targets": [
            "governance/quant_operational_intelligence",
            f"governance/quant_operational_intelligence/{system_slug}",
            "governance/health/quant_operational_intelligence_latest.json",
        ],
        "freshness_slo_seconds": 1200,
        "retention_profile": "quant_operational_intelligence_hot_3d_warm_120d_cold_540d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "quant_operational_intelligence_collect_only_until_evidence_resource_runtime_and_safety_gates_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "quant_operational_intelligence_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_quant_operational_intelligence_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "training_threshold_policy": {
            "minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_collection_days": MINIMUM_COLLECTION_DAYS,
            "requires_alpha_evidence_court_clearance": domain == "quant",
            "requires_execution_quality_clearance": domain == "quant",
            "requires_runtime_pressure_clearance": True,
            "requires_backpressure_clearance": True,
            "requires_data_quality_clearance": True,
            "requires_duplicate_alpha_clearance": True,
            "requires_paper_live_separation_clearance": True,
            "requires_global_halt_clear": True,
        },
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "thin_sampled",
        "data_collection_sample_rate": SAMPLE_RATE,
        "data_collection_max_daily_storage_mb": MAX_DAILY_MB_PER_BOT,
        "data_collection_max_daily_mb": float(MAX_DAILY_MB_PER_BOT),
        "data_collection_compute_guard_mode": "thin_digest",
        "data_collection_resource_guard_reason": "quant_operational_intelligence_uses_digest_only_capture_to_protect_cpu_memory_storage",
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "paper_execution_queue_policy": "blocked_until_quant_operational_evidence_thresholds_clear",
        "paper_runtime_control_refresh_seconds": 300,
        "sleeve_profile": system_slug,
        "sleeve_family": SLEEVE_FAMILY,
        "intelligence_domain": domain,
        "intelligence_system": system_slug,
        "strategy_family": "quant_and_operational_control_intelligence",
        "correlation_peer_sleeves": sorted(set(peer_sleeves)),
        "correlation_dependencies": [
            "system_self_model",
            "system_self_intelligence",
            "platform_brain_v6",
            "quant_model_control",
            "platform_organ_systems",
            "backpressure_storage_brain_v2",
            "paper_trade_lock_guard",
            "global_halt_guard",
        ],
        "provider_capability_profile": "internal_quant_ops_governance_and_market_context_collect_only",
        "direct_market_data_available": domain == "quant",
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "master_bot_registry",
            "governance_health",
            "quant_model_control",
            "system_self_intelligence_memory",
            "codex_handoff",
            "platform_organ_systems",
            "paper_trade_lock_trace",
        ],
        "schwab_direct_inputs": ["quotes", "chains", "market_hours", "fundamentals"] if domain == "quant" else [],
        "proxy_only_reason": "quant_operational_intelligence_collects_evidence_and_routes_only_until_training_and_paper_gates_clear",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "primary_horizon": f"{system_slug}_decision_quality_after_bounded_action",
            "required_context": data_intakes,
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.87,
            "freshness_slo_seconds": 1200,
            "regression_guard_bot_id": contract["anchor_bot_ids"].get(system_slug, ""),
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": [
            "research_only",
            "collection_only",
            "execution_blocked",
            "paper_only_floor",
            f"sleeve_family:{SLEEVE_FAMILY}",
            f"sleeve_profile:{system_slug}",
            f"intelligence_domain:{domain}",
            f"capability_pack:{PACK_SLUG}",
            "quant_operational_intelligence",
            "point_in_time_only",
            "training_after_threshold",
            "global_halt_aware",
            "pressure_safe",
            "mlx_default",
        ],
        "execution_policy_label": "collection_only_quant_operational_intelligence_no_execution",
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
            "quant_decision_quality",
            "operational_causal_routing",
            "point_in_time_labeling",
        ],
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "paper_runtime_stability_mode": "thin_digest_quant_operational_intelligence",
        "paper_trade_lock_required": True,
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "capability_pack_version": PACK_VERSION,
        "capability_pack_slug": PACK_SLUG,
        "capability_pack_display_name": PACK_DISPLAY_NAME,
        "quant_operational_intelligence_version": PACK_VERSION,
        "capability_pack_contract": contract,
        "quant_operational_intelligence_contract": system_contract,
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    pack_rows = [row for row in rows if str(row.get("quant_operational_intelligence_version") or "") == PACK_VERSION]
    versions = [
        int(match.group(1))
        for row in rows
        for match in [re.match(r"^brain_refinery_v(\d+)", str(row.get("bot_id") or ""))]
        if match
    ]
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
            "quant_operational_intelligence_bot_count": len(pack_rows),
            "latest_quant_operational_intelligence": PACK_VERSION,
            "max_bot_version": max(versions) if versions else None,
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
        "objective": "Add 10 quant intelligence systems and 10 operational intelligence systems that score edge quality, action effects, causal routing, capacity, lineage, and safe command decisions.",
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "system_count": len(INTELLIGENCE_SYSTEMS),
        "quant_system_count": contract["quant_system_count"],
        "operational_system_count": contract["operational_system_count"],
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "intelligence_systems": list(INTELLIGENCE_SYSTEMS),
        "dedicated_data_intake": list(BASE_DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "runtime_capacity_floor": contract["runtime_capacity_floor"],
        "anchor_bot_ids": contract["anchor_bot_ids"],
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
        "quant_operational_intelligence_version": PACK_VERSION,
        "system_count": len(INTELLIGENCE_SYSTEMS),
        "quant_system_count": sum(1 for system in INTELLIGENCE_SYSTEMS if system["domain"] == "quant"),
        "operational_system_count": sum(1 for system in INTELLIGENCE_SYSTEMS if system["domain"] == "operational"),
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
        "quant_operational_intelligence_version": PACK_VERSION,
        "system_count": plan["system_count"],
        "quant_system_count": plan["quant_system_count"],
        "operational_system_count": plan["operational_system_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "pack": plan["pack"],
        "recommended_apply_command": "./scripts/ops/opsctl.sh quant-operational-intelligence --apply --json",
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
        backup = backup_dir / f"master_bot_registry_before_quant_operational_intelligence_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
        project_root / "config" / "quant_operational_intelligence_v1.json",
        {
            "generated_at_utc": _utc_now(),
            "quant_operational_intelligence_version": PACK_VERSION,
            "pack": payload["pack"],
        },
    )
    _write_json(project_root / "governance" / "health" / "quant_operational_intelligence_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add the 80-bot quant and operational intelligence collect-only pack.")
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
            "quant_operational_intelligence "
            f"mode={payload['mode']} systems={payload['system_count']} "
            f"quant={payload['quant_system_count']} operational={payload['operational_system_count']} "
            f"bots={payload['bot_count']} planned={payload['planned_bot_count']} "
            f"added={payload.get('added_bot_count', 0)} "
            f"target_total={payload.get('planned_total_after_apply') or payload.get('new_total_bots')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
