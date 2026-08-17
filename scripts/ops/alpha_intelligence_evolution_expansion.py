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
BASE_VERSION = 904
PACK_VERSION = "alpha_intelligence_evolution_v1"
PACK_SLUG = "alpha_intelligence_evolution"
PACK_DISPLAY_NAME = "Alpha Intelligence Evolution Pack"
SLEEVE_FAMILY = "alpha_intelligence_control_plane"
SLEEVE_PROFILE = "alpha_intelligence_evolution"
LABEL_CONTRACT_VERSION = "alpha_intelligence_evolution_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 20000
MINIMUM_COLLECTION_DAYS = 75
PAPER_RUNTIME_CAPACITY_FLOOR = 900

SYSTEMS: list[dict[str, Any]] = [
    {
        "slug": "training_readiness_brain",
        "display_name": "Training Readiness Brain",
        "objective": "Graduate bots from collect-only to trainable only when evidence, label quality, freshness, and runtime stability are real.",
        "outputs": ["training_graduation_score", "sample_debt_queue", "retrain_admission_packet"],
    },
    {
        "slug": "execution_reality_lab",
        "display_name": "Execution Reality Lab",
        "objective": "Stress paper decisions against slippage, spread, fill probability, latency, partial fills, and paper-live drift.",
        "outputs": ["execution_reality_score", "paper_live_drift_packet", "route_quality_surface"],
    },
    {
        "slug": "portfolio_exposure_brain",
        "display_name": "Portfolio Exposure Brain",
        "objective": "Net cross-sleeve exposure across beta, delta, duration, sector, vol, crypto beta, margin, and correlation clusters.",
        "outputs": ["exposure_netting_packet", "convexity_margin_surface", "correlation_cluster_map"],
    },
    {
        "slug": "data_source_confidence_engine",
        "display_name": "Data Source Confidence Engine",
        "objective": "Score every data source by freshness, field completeness, source disagreement, credential health, and ingestion lag.",
        "outputs": ["source_confidence_score", "source_disagreement_packet", "freshness_sla_route"],
    },
    {
        "slug": "research_intake_pipeline",
        "display_name": "Research Intake Pipeline",
        "objective": "Turn papers, notes, and model ideas into bounded candidates with data, label, risk, and implementation contracts.",
        "outputs": ["research_candidate_card", "feature_model_candidate", "implementation_risk_vote"],
    },
    {
        "slug": "duplicate_alpha_novelty_engine",
        "display_name": "Duplicate Alpha / Novelty Engine",
        "objective": "Stop duplicate alphas from bloating the platform and reward strategies that add truly new information.",
        "outputs": ["alpha_similarity_score", "novelty_vote", "duplicate_alpha_reject_packet"],
    },
    {
        "slug": "regime_playbook_memory_v2",
        "display_name": "Regime Playbook Memory v2",
        "objective": "Remember what worked, failed, decayed, or overloaded in comparable market regimes.",
        "outputs": ["regime_playbook_memory", "regime_retrieval_packet", "regime_lesson_decay_vote"],
    },
    {
        "slug": "professional_dashboard_v2",
        "display_name": "Professional System Dashboard v2",
        "objective": "Expose a clean report-ready cockpit for bot count, collection health, training maturity, halts, pressure, and top issues.",
        "outputs": ["system_dashboard_v2", "operator_narrative", "dashboard_regression_packet"],
    },
    {
        "slug": "broker_data_adapter_mesh",
        "display_name": "Broker / Data Adapter Mesh",
        "objective": "Normalize Schwab, Coinbase, IBKR, paper broker, free data, and future adapters behind one guarded contract.",
        "outputs": ["adapter_contract_matrix", "entitlement_gap_packet", "failover_route_map"],
    },
    {
        "slug": "autonomous_cleanup_governor",
        "display_name": "Autonomous Cleanup Governor",
        "objective": "Rank and apply low-risk cleanup for stale files, duplicated artifacts, old logs, failed launches, and cache growth.",
        "outputs": ["cleanup_action_score", "stale_artifact_quarantine", "cleanup_regression_report"],
    },
]

DATA_INTAKES = [
    "training_readiness_threshold_trace",
    "label_quality_sample_debt_trace",
    "execution_reality_slippage_trace",
    "paper_live_drift_trace",
    "portfolio_exposure_netting_trace",
    "data_source_confidence_trace",
    "research_candidate_intake_trace",
    "duplicate_alpha_similarity_trace",
    "regime_playbook_memory_trace_v2",
    "professional_dashboard_status_trace",
    "broker_adapter_contract_trace",
    "autonomous_cleanup_action_trace",
    "self_awareness_intelligence_upgrade_trace",
]

STORAGE_TARGETS = [
    "governance/alpha_intelligence",
    "governance/alpha_intelligence/training_readiness",
    "governance/alpha_intelligence/execution_reality",
    "governance/alpha_intelligence/portfolio_exposure",
    "governance/alpha_intelligence/source_confidence",
    "governance/alpha_intelligence/research_intake",
    "governance/alpha_intelligence/duplicate_alpha",
    "governance/alpha_intelligence/regime_memory",
    "governance/alpha_intelligence/dashboard",
    "governance/alpha_intelligence/adapter_mesh",
    "governance/alpha_intelligence/cleanup_governor",
    "governance/health/alpha_intelligence_evolution_latest.json",
]

REQUIRED_LABELS = [
    "training_readiness_state",
    "label_quality_bucket",
    "execution_reality_bucket",
    "paper_live_drift_bucket",
    "source_confidence_bucket",
    "alpha_novelty_bucket",
    "portfolio_exposure_bucket",
    "regime_memory_key",
    "cleanup_action_risk",
    "adapter_contract_status",
]

BOTS: list[dict[str, Any]] = [
    {
        "role_slug": "training_graduation_scorer",
        "slug": "alpha_training_graduation_scorer_bot",
        "label": "Alpha Training Graduation Scorer",
        "bot_role": "infrastructure_sub_bot",
        "system": "training_readiness_brain",
        "priority": "critical",
        "objective": "Score when collect-only bots have enough clean samples, freshness, label quality, and runtime stability to train.",
        "target_functions": ["training_requalification", "new_bot_admission_guard", "training_quality_control"],
    },
    {
        "role_slug": "label_sample_debt_guard",
        "slug": "alpha_label_sample_debt_guard_bot",
        "label": "Alpha Label Sample Debt Guard",
        "bot_role": "infrastructure_sub_bot",
        "system": "training_readiness_brain",
        "priority": "critical",
        "objective": "Rank sample debt, delayed labels, stale joins, leakage risk, and point-in-time coverage before retrain admission.",
        "target_functions": ["label_quality", "point_in_time_event_store", "coverage_gap_closer"],
    },
    {
        "role_slug": "retrain_admission_master",
        "slug": "alpha_retrain_admission_master_bot",
        "label": "Alpha Retrain Admission Master",
        "bot_role": "infrastructure_sub_bot",
        "system": "training_readiness_brain",
        "priority": "critical",
        "objective": "Open training only for bots whose readiness, novelty, and resource budgets clear the control-plane gate.",
        "target_functions": ["weekly_retrain", "training_runtime_control", "model_lifecycle"],
    },
    {
        "role_slug": "slippage_fill_probability_scorer",
        "slug": "alpha_slippage_fill_probability_scorer_bot",
        "label": "Alpha Slippage Fill Probability Scorer",
        "bot_role": "signal_sub_bot",
        "system": "execution_reality_lab",
        "priority": "critical",
        "objective": "Estimate fill quality, spread drag, partial fill probability, and latency penalty for every paper decision.",
        "target_functions": ["execution_lab", "transaction_cost_slippage_intelligence", "paper_broker_bridge"],
    },
    {
        "role_slug": "paper_live_drift_auditor",
        "slug": "alpha_paper_live_drift_auditor_bot",
        "label": "Alpha Paper-Live Drift Auditor",
        "bot_role": "infrastructure_sub_bot",
        "system": "execution_reality_lab",
        "priority": "critical",
        "objective": "Compare paper assumptions against broker truth, route availability, entitlement state, and live-feed timing.",
        "target_functions": ["paper_live_execution_parity", "broker_truth", "provider_adapter_verification"],
    },
    {
        "role_slug": "route_quality_guard",
        "slug": "alpha_route_quality_guard_bot",
        "label": "Alpha Route Quality Guard",
        "bot_role": "infrastructure_sub_bot",
        "system": "execution_reality_lab",
        "priority": "high",
        "objective": "Down-rank strategies when route quality, stale quotes, or venue liquidity make paper edge unrealistic.",
        "target_functions": ["slippage_venue_quality_router", "execution_quality_regression_guard", "sleeve_isolation"],
    },
    {
        "role_slug": "cross_sleeve_exposure_netter",
        "slug": "alpha_cross_sleeve_exposure_netter_bot",
        "label": "Alpha Cross-Sleeve Exposure Netter",
        "bot_role": "infrastructure_sub_bot",
        "system": "portfolio_exposure_brain",
        "priority": "critical",
        "objective": "Net beta, delta, duration, sector, vol, crypto, macro, and hedge exposure across all sleeves.",
        "target_functions": ["portfolio_allocator", "risk_service", "cross_sleeve_exposure_netting"],
    },
    {
        "role_slug": "convexity_margin_sentinel",
        "slug": "alpha_convexity_margin_sentinel_bot",
        "label": "Alpha Convexity Margin Sentinel",
        "bot_role": "infrastructure_sub_bot",
        "system": "portfolio_exposure_brain",
        "priority": "critical",
        "objective": "Watch options convexity, futures exposure, margin sensitivity, and liquidation cascades before they trip guards.",
        "target_functions": ["margin_guard", "options_greeks", "collateral_margin_regression_guard"],
    },
    {
        "role_slug": "correlation_cluster_governor",
        "slug": "alpha_correlation_cluster_governor_bot",
        "label": "Alpha Correlation Cluster Governor",
        "bot_role": "signal_sub_bot",
        "system": "portfolio_exposure_brain",
        "priority": "high",
        "objective": "Detect when independent-looking bots are really one crowded correlation cluster.",
        "target_functions": ["correlation_cluster_risk", "duplicate_alpha_similarity", "market_regime_router"],
    },
    {
        "role_slug": "source_confidence_scorer",
        "slug": "alpha_source_confidence_scorer_bot",
        "label": "Alpha Source Confidence Scorer",
        "bot_role": "infrastructure_sub_bot",
        "system": "data_source_confidence_engine",
        "priority": "critical",
        "objective": "Score each data source by freshness, missing fields, stale ticks, credential health, and ingestion lag.",
        "target_functions": ["collector_contracts", "source_verification", "data_ingress_health"],
    },
    {
        "role_slug": "adapter_disagreement_guard",
        "slug": "alpha_adapter_disagreement_guard_bot",
        "label": "Alpha Adapter Disagreement Guard",
        "bot_role": "infrastructure_sub_bot",
        "system": "data_source_confidence_engine",
        "priority": "critical",
        "objective": "Flag source disagreement between Schwab, Coinbase, IBKR, macro, options, and fallback data adapters.",
        "target_functions": ["provider_adapter_verification", "broker_truth", "source_confidence"],
    },
    {
        "role_slug": "freshness_sla_master",
        "slug": "alpha_freshness_sla_master_bot",
        "label": "Alpha Freshness SLA Master",
        "bot_role": "infrastructure_sub_bot",
        "system": "data_source_confidence_engine",
        "priority": "high",
        "objective": "Route stale feeds to degraded collection or summary-only mode before global halts need to fire.",
        "target_functions": ["artifact_freshness_slo", "data_plane_recovery_controller", "global_halt_preemption"],
    },
    {
        "role_slug": "research_candidate_compiler",
        "slug": "alpha_research_candidate_compiler_bot",
        "label": "Alpha Research Candidate Compiler",
        "bot_role": "infrastructure_sub_bot",
        "system": "research_intake_pipeline",
        "priority": "high",
        "objective": "Compile papers, operator notes, and model ideas into structured candidate cards.",
        "target_functions": ["recursive_research_foundry", "research_automation", "strategy_inventory_report"],
    },
    {
        "role_slug": "feature_model_candidate_builder",
        "slug": "alpha_feature_model_candidate_builder_bot",
        "label": "Alpha Feature / Model Candidate Builder",
        "bot_role": "signal_sub_bot",
        "system": "research_intake_pipeline",
        "priority": "high",
        "objective": "Turn approved research into feature, model, data, sleeve, or bot specs with bounded assumptions.",
        "target_functions": ["feature_store", "advanced_quant_models", "bot_admission_guard"],
    },
    {
        "role_slug": "implementation_risk_triager",
        "slug": "alpha_implementation_risk_triager_bot",
        "label": "Alpha Implementation Risk Triager",
        "bot_role": "infrastructure_sub_bot",
        "system": "research_intake_pipeline",
        "priority": "critical",
        "objective": "Block research ideas that need paid data, unavailable broker support, excessive compute, or unsafe labels.",
        "target_functions": ["source_verification", "expansion_capacity_planner", "security_audit"],
    },
    {
        "role_slug": "alpha_similarity_detector",
        "slug": "alpha_similarity_detector_bot",
        "label": "Alpha Similarity Detector",
        "bot_role": "infrastructure_sub_bot",
        "system": "duplicate_alpha_novelty_engine",
        "priority": "critical",
        "objective": "Compare new and existing bots to detect duplicate alpha, duplicated features, and redundant sleeve intent.",
        "target_functions": ["bot_admission_committee", "feature_overlap_map", "lineage_trait_drift_guard"],
    },
    {
        "role_slug": "novelty_score_master",
        "slug": "alpha_novelty_score_master_bot",
        "label": "Alpha Novelty Score Master",
        "bot_role": "signal_sub_bot",
        "system": "duplicate_alpha_novelty_engine",
        "priority": "high",
        "objective": "Reward alphas that add new information after accounting for correlation, regime, and feature overlap.",
        "target_functions": ["alpha_research_os", "bayesian_evidence_score", "model_decay_detector"],
    },
    {
        "role_slug": "duplicate_alpha_reject_guard",
        "slug": "alpha_duplicate_reject_guard_bot",
        "label": "Alpha Duplicate Reject Guard",
        "bot_role": "infrastructure_sub_bot",
        "system": "duplicate_alpha_novelty_engine",
        "priority": "critical",
        "objective": "Reject or quarantine low-novelty bots before they consume storage, CPU, labels, or report space.",
        "target_functions": ["new_bot_admission_guard", "storage_backpressure_autopilot", "training_label_guard"],
    },
    {
        "role_slug": "regime_playbook_writer_v2",
        "slug": "alpha_regime_playbook_writer_v2_bot",
        "label": "Alpha Regime Playbook Writer v2",
        "bot_role": "infrastructure_sub_bot",
        "system": "regime_playbook_memory_v2",
        "priority": "high",
        "objective": "Write compact lessons about which sleeves helped or hurt during each market regime.",
        "target_functions": ["market_regime_memory", "stress_lab", "system_self_model"],
    },
    {
        "role_slug": "regime_retrieval_router_v2",
        "slug": "alpha_regime_retrieval_router_v2_bot",
        "label": "Alpha Regime Retrieval Router v2",
        "bot_role": "signal_sub_bot",
        "system": "regime_playbook_memory_v2",
        "priority": "high",
        "objective": "Retrieve comparable regimes and route the relevant playbook to sleeve masters and grandmaster.",
        "target_functions": ["regime_router", "adaptive_intelligence_kernel", "coordination_intelligence"],
    },
    {
        "role_slug": "regime_lesson_decay_guard_v2",
        "slug": "alpha_regime_lesson_decay_guard_v2_bot",
        "label": "Alpha Regime Lesson Decay Guard v2",
        "bot_role": "infrastructure_sub_bot",
        "system": "regime_playbook_memory_v2",
        "priority": "high",
        "objective": "Retire stale regime lessons when market structure, liquidity, volatility, or data quality changes.",
        "target_functions": ["model_decay_detector", "feature_freshness", "self_model_optimization_ranker"],
    },
    {
        "role_slug": "dashboard_surface_builder_v2",
        "slug": "alpha_dashboard_surface_builder_v2_bot",
        "label": "Alpha Dashboard Surface Builder v2",
        "bot_role": "infrastructure_sub_bot",
        "system": "professional_dashboard_v2",
        "priority": "high",
        "objective": "Build a concise operator dashboard for bot count, collection, training, halts, pressure, and top issues.",
        "target_functions": ["operator_cockpit", "system_dashboard", "reporting_layer"],
    },
    {
        "role_slug": "operator_narrative_brief_v2",
        "slug": "alpha_operator_narrative_brief_v2_bot",
        "label": "Alpha Operator Narrative Brief v2",
        "bot_role": "infrastructure_sub_bot",
        "system": "professional_dashboard_v2",
        "priority": "medium",
        "objective": "Explain what changed, why it matters, and which safe commands should be run next.",
        "target_functions": ["system_self_model", "system_summary", "incident_review_packet"],
    },
    {
        "role_slug": "dashboard_regression_guard_v2",
        "slug": "alpha_dashboard_regression_guard_v2_bot",
        "label": "Alpha Dashboard Regression Guard v2",
        "bot_role": "infrastructure_sub_bot",
        "system": "professional_dashboard_v2",
        "priority": "critical",
        "objective": "Prevent stale, missing, or misleading dashboard surfaces from hiding pressure or halt states.",
        "target_functions": ["operator_cockpit", "report_quality_guard", "commands_hygiene"],
    },
    {
        "role_slug": "adapter_contract_master",
        "slug": "alpha_adapter_contract_master_bot",
        "label": "Alpha Adapter Contract Master",
        "bot_role": "infrastructure_sub_bot",
        "system": "broker_data_adapter_mesh",
        "priority": "critical",
        "objective": "Maintain one contract for broker, paper, macro, crypto, options, and fallback data adapters.",
        "target_functions": ["broker_registry", "collector_contracts", "provider_adapter_verification"],
    },
    {
        "role_slug": "entitlement_handshake_guard",
        "slug": "alpha_entitlement_handshake_guard_bot",
        "label": "Alpha Entitlement Handshake Guard",
        "bot_role": "infrastructure_sub_bot",
        "system": "broker_data_adapter_mesh",
        "priority": "critical",
        "objective": "Detect credential, entitlement, and handshake gaps before live feeds or paper truth degrade.",
        "target_functions": ["auth_lease_manager", "schwab_auth_supervisor", "interactive_broker_auth_steps"],
    },
    {
        "role_slug": "adapter_failover_route_mapper",
        "slug": "alpha_adapter_failover_route_mapper_bot",
        "label": "Alpha Adapter Failover Route Mapper",
        "bot_role": "infrastructure_sub_bot",
        "system": "broker_data_adapter_mesh",
        "priority": "high",
        "objective": "Map safe read-only fallback routes when a source, broker, or external drive path degrades.",
        "target_functions": ["storage_failback_sync", "data_plane_recovery", "source_confidence"],
    },
    {
        "role_slug": "cleanup_action_scorer",
        "slug": "alpha_cleanup_action_scorer_bot",
        "label": "Alpha Cleanup Action Scorer",
        "bot_role": "infrastructure_sub_bot",
        "system": "autonomous_cleanup_governor",
        "priority": "high",
        "objective": "Rank cleanup actions by risk, storage payoff, freshness, and impact on collection or paper trading.",
        "target_functions": ["data_retention_policy", "stale_artifact_reaper", "storage_quota_guard"],
    },
    {
        "role_slug": "stale_artifact_quarantine_master",
        "slug": "alpha_stale_artifact_quarantine_master_bot",
        "label": "Alpha Stale Artifact Quarantine Master",
        "bot_role": "infrastructure_sub_bot",
        "system": "autonomous_cleanup_governor",
        "priority": "high",
        "objective": "Quarantine low-value stale artifacts before deletion while preserving report, training, and audit evidence.",
        "target_functions": ["stale_sweeper", "retention_debt_sheriff", "storage_backpressure_autopilot"],
    },
    {
        "role_slug": "cleanup_regression_guard",
        "slug": "alpha_cleanup_regression_guard_bot",
        "label": "Alpha Cleanup Regression Guard",
        "bot_role": "infrastructure_sub_bot",
        "system": "autonomous_cleanup_governor",
        "priority": "critical",
        "objective": "Block cleanup from deleting active feeds, core bot files, reports, recent training data, or audit evidence.",
        "target_functions": ["stateful_storage_guard", "core_bot_file_guard", "report_quality_guard"],
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


def _ensure_storage_targets(project_root: Path) -> list[str]:
    created_or_confirmed: list[str] = []
    for target in STORAGE_TARGETS:
        path = project_root / target
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            created_or_confirmed.append(str(path.parent.relative_to(project_root)))
        else:
            path.mkdir(parents=True, exist_ok=True)
            created_or_confirmed.append(str(path.relative_to(project_root)))
    return sorted(set(created_or_confirmed))


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
            "retention_profile": "alpha_intelligence_hot_7d_warm_60d_cold_365d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": 24,
            "capture_mode": "digest_first_delta_snapshots",
            "sample_rate": 0.16,
            "dedupe_required": True,
            "stale_deletion_policy": "quarantine_then_summary_digest_before_delete",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "sleeve_master_bot_id": assigned_ids.get(f"{PACK_SLUG}_retrain_admission_master", ""),
        "regression_guard_bot_id": assigned_ids.get(f"{PACK_SLUG}_cleanup_regression_guard", ""),
        "alpha_admission_guard_bot_id": assigned_ids.get(f"{PACK_SLUG}_duplicate_alpha_reject_guard", ""),
        "self_awareness_bridge_bot_id": assigned_ids.get(f"{PACK_SLUG}_operator_narrative_brief_v2", ""),
        "runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "intelligence_upgrades": [
            "training_readiness_gating",
            "execution_reality_scoring",
            "portfolio_exposure_netting",
            "source_confidence_scoring",
            "research_to_candidate_pipeline",
            "duplicate_alpha_novelty_control",
            "regime_playbook_memory_v2",
            "professional_dashboard_v2",
            "broker_data_adapter_mesh",
            "autonomous_cleanup_governor",
        ],
        "self_awareness_upgrades": [
            "self_model_feeds_alpha_readiness",
            "resource_pressure_context_before_expansion",
            "failure_memory_informs_admission",
            "operator_narratives_include_why_and_next_safe_action",
            "cleanup_governor_requires_regression_guard_clearance",
        ],
        "global_halt_contract": "all_bots_switch_to_summary_only_and_zero_weight_when_halt_or_resource_pressure_is_active",
        "paper_lock_contract": "collection_only_until_training_readiness_execution_reality_source_confidence_and_duplicate_alpha_gates_clear",
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    system = _system(bot)
    contract = _pack_contract(assigned_ids)
    advanced_contract = {
        "contract_version": "alpha_intelligence_evolution_layers_v1",
        "capability_pack": PACK_SLUG,
        "system": bot["system"],
        "system_display_name": system["display_name"],
        "system_outputs": list(system.get("outputs", [])),
        "reports_to_sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "regression_guard_bot_id": contract["regression_guard_bot_id"],
        "alpha_admission_guard_bot_id": contract["alpha_admission_guard_bot_id"],
        "self_awareness_bridge_bot_id": contract["self_awareness_bridge_bot_id"],
        "alpha_advancement": "novelty_execution_reality_data_confidence_and_regime_memory_must_all_clear_before_training_or_trust_lift",
        "self_awareness": "system_self_model_resource_pressure_failure_memory_and_operator_narrative_are_required_context",
        "intelligence_upgrade_mode": "digest_first_low_compute_collect_only",
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
            "brain_refinery_v849_coordination_lineage_genome_mapper_bot",
            "brain_refinery_v879_adaptive_kernel_online_meta_learning_master_bot",
            "brain_refinery_v894_self_model_identity_cartographer_bot",
        ],
        "data_intake_collections": list(DATA_INTAKES),
        "storage_targets": list(STORAGE_TARGETS),
        "freshness_slo_seconds": 900,
        "retention_profile": "alpha_intelligence_hot_7d_warm_60d_cold_365d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "alpha_intelligence_collect_only_until_readiness_execution_source_and_novelty_gates_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "alpha_intelligence_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_alpha_intelligence_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "training_threshold_policy": {
            "minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_collection_days": MINIMUM_COLLECTION_DAYS,
            "requires_source_confidence": True,
            "requires_execution_reality_score": True,
            "requires_duplicate_alpha_clearance": True,
            "requires_runtime_pressure_clear": True,
        },
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "digest_first_delta_snapshots",
        "data_collection_sample_rate": 0.16,
        "data_collection_max_daily_storage_mb": 24,
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "sleeve_profile": SLEEVE_PROFILE,
        "sleeve_family": SLEEVE_FAMILY,
        "correlation_peer_sleeves": [
            "coordination_intelligence",
            "adaptive_intelligence_kernel",
            "advanced_intelligence_mesh",
            "system_self_awareness",
            "portfolio_risk",
            "execution_intelligence",
            "research_automation",
        ],
        "correlation_dependencies": [
            "system_self_model",
            "operator_cockpit",
            "training_quality_control",
            "execution_lab",
            "source_verification",
            "collector_contracts",
            "storage_backpressure_autopilot",
            "global_halt_status",
        ],
        "provider_capability_profile": "internal_control_plane_and_proxy_data_only",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "master_bot_registry",
            "core_bot_catalog",
            "operator_cockpit",
            "training_quality_control",
            "source_verification",
            "execution_lab",
            "global_halt",
            "system_self_model",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "alpha_intelligence_evolution_has_no_direct_execution_or_broker_dependency",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.76,
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
            f"alpha_system:{bot['system']}",
            "alpha_advancement",
            "self_awareness_upgrade",
            "intelligence_upgrade",
            "training_after_threshold",
            "global_halt_aware",
        ],
        "execution_policy_label": "collection_only_alpha_intelligence_no_execution",
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
            "alpha_novelty_awareness",
            "self_model_awareness",
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
        "alpha_intelligence_evolution_version": PACK_VERSION,
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
    pack_rows = [row for row in rows if str(row.get("alpha_intelligence_evolution_version") or "") == PACK_VERSION]
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
            "alpha_intelligence_evolution_bot_count": len(pack_rows),
            "latest_alpha_intelligence_evolution": PACK_VERSION,
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
        "objective": "Advance alpha quality, execution realism, data confidence, self-awareness, and intelligence routing without adding direct execution risk.",
        "system_count": len(SYSTEMS),
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "systems": list(SYSTEMS),
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "regression_guard_bot_id": contract["regression_guard_bot_id"],
        "alpha_admission_guard_bot_id": contract["alpha_admission_guard_bot_id"],
        "self_awareness_bridge_bot_id": contract["self_awareness_bridge_bot_id"],
        "intelligence_upgrades": list(contract["intelligence_upgrades"]),
        "self_awareness_upgrades": list(contract["self_awareness_upgrades"]),
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
        "alpha_intelligence_evolution_version": PACK_VERSION,
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
        "alpha_intelligence_evolution_version": PACK_VERSION,
        "system_count": plan["system_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "pack": plan["pack"],
        "recommended_apply_command": "./scripts/ops/opsctl.sh alpha-intelligence-evolution --apply --json",
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
        backup = backup_dir / f"master_bot_registry_before_alpha_intelligence_evolution_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
        project_root / "config" / "alpha_intelligence_evolution_v1.json",
        {
            "generated_at_utc": _utc_now(),
            "alpha_intelligence_evolution_version": PACK_VERSION,
            "pack": payload["pack"],
        },
    )
    _write_json(project_root / "governance" / "health" / "alpha_intelligence_evolution_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add alpha advancement, self-awareness, and intelligence evolution bots.")
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
            "alpha_intelligence_evolution "
            f"mode={payload['mode']} systems={payload['system_count']} "
            f"bots={payload['bot_count']} planned={payload['planned_bot_count']} "
            f"added={payload.get('added_bot_count', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
