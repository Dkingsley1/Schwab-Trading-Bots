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
BASE_VERSION = 849
COORDINATION_VERSION = "coordination_intelligence_pack_v1"
LABEL_CONTRACT_VERSION = "coordination_intelligence_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 15000
MINIMUM_COLLECTION_DAYS = 60
PAPER_RUNTIME_CAPACITY_FLOOR = 700

COORDINATION_SLUG = "coordination_intelligence"
COORDINATION_DISPLAY_NAME = "Coordination Intelligence Control Plane"
COORDINATION_FAMILY = "coordination_control_plane"
COORDINATION_PROFILE = "coordination_intelligence"

LAYERS: list[dict[str, Any]] = [
    {
        "slug": "bot_genome_lineage_map_v2",
        "display_name": "Bot Genome / Lineage Map v2",
        "objective": "Track parent ideas, founder DNA, sleeve ancestry, source lineage, and trait drift.",
        "outputs": ["bot_genome_map", "lineage_trait_drift_report", "pycharm_visible_family_map"],
    },
    {
        "slug": "strategy_conflict_resolver",
        "display_name": "Strategy Conflict Resolver",
        "objective": "Net conflicting cross-sleeve views before any paper or allocation surface can amplify them.",
        "outputs": ["cross_sleeve_conflict_matrix", "net_view_packet", "conflict_resolution_audit"],
    },
    {
        "slug": "capital_allocation_simulator",
        "display_name": "Capital Allocation Simulator",
        "objective": "Dry-run capital flow, margin pressure, drawdown sensitivity, and sleeve concentration.",
        "outputs": ["capital_flow_sandbox", "margin_sensitivity_packet", "allocation_risk_budget"],
    },
    {
        "slug": "market_regime_memory",
        "display_name": "Market Regime Memory",
        "objective": "Remember what worked, failed, decayed, or overloaded in each market regime.",
        "outputs": ["regime_memory_store", "regime_playbook_lookup", "regime_decay_flags"],
    },
    {
        "slug": "research_to_bot_pipeline",
        "display_name": "Research-to-Bot Pipeline",
        "objective": "Classify research notes and papers into bot, sleeve, model, report, or ignore decisions.",
        "outputs": ["research_classification_queue", "bot_spec_candidates", "research_admission_votes"],
    },
    {
        "slug": "feature_store_quality_layer",
        "display_name": "Feature Store Quality Layer",
        "objective": "Score features by freshness, missingness, leakage risk, predictive value, and overlap.",
        "outputs": ["feature_quality_scores", "feature_overlap_map", "feature_contracts"],
    },
    {
        "slug": "adversarial_paper_trading_lab",
        "display_name": "Adversarial Paper Trading Lab",
        "objective": "Stress paper trades under bad fills, stale quotes, data delays, broker faults, and liquidity gaps.",
        "outputs": ["adversarial_fill_replay", "stale_quote_delay_report", "broker_fault_lab_packet"],
    },
    {
        "slug": "sleeve_master_upgrade_pack",
        "display_name": "Sleeve Master Upgrade Pack",
        "objective": "Summarize sleeve conviction, data quality, halt pressure, and training readiness for grandmaster.",
        "outputs": ["sleeve_conviction_summary", "sleeve_health_route", "grandmaster_brief_packet"],
    },
    {
        "slug": "bot_admission_committee",
        "display_name": "Bot Admission Committee",
        "objective": "Screen every new bot for duplicate alpha, storage load, CPU load, data availability, and label quality.",
        "outputs": ["bot_admission_vote", "duplicate_idea_screen", "capacity_impact_report"],
    },
    {
        "slug": "system_explainability_dashboard",
        "display_name": "System Explainability Dashboard",
        "objective": "Explain what changed, which bots influenced a decision, and which surfaces deserve less trust.",
        "outputs": ["decision_provenance_story", "daily_system_change_narrative", "trust_surface_dashboard"],
    },
]

DATA_INTAKES = [
    "bot_genome_lineage_trace",
    "cross_sleeve_signal_conflict_trace",
    "capital_allocation_sandbox_trace",
    "margin_drawdown_sensitivity_trace",
    "market_regime_memory_trace",
    "research_to_bot_admission_trace",
    "feature_store_quality_trace",
    "adversarial_paper_trade_lab_trace",
    "sleeve_master_conviction_trace",
    "bot_admission_committee_trace",
    "decision_provenance_explainability_trace",
    "trust_surface_dashboard_trace",
]

STORAGE_TARGETS = [
    "governance/coordination_intelligence",
    "governance/coordination_intelligence/lineage",
    "governance/coordination_intelligence/conflicts",
    "governance/coordination_intelligence/capital_sim",
    "governance/coordination_intelligence/regime_memory",
    "governance/coordination_intelligence/research_admission",
    "governance/coordination_intelligence/feature_quality",
    "governance/coordination_intelligence/adversarial_paper_lab",
    "governance/coordination_intelligence/explainability",
    "governance/health/coordination_intelligence",
    "data/jsonl_link.sqlite3",
]

REQUIRED_LABELS = [
    "coordination_layer",
    "source_bot_id",
    "target_sleeve",
    "conflict_side",
    "net_view_direction",
    "regime_memory_key",
    "feature_quality_bucket",
    "adversarial_scenario_id",
    "admission_vote",
    "trust_surface_grade",
]

COORDINATION_CONTRACT = {
    "contract_version": COORDINATION_VERSION,
    "purpose": "coordinate_large_bot_fleet_views_admission_memory_allocation_and_explainability",
    "layers": [layer["slug"] for layer in LAYERS],
    "graduation_policy": "collection_only_until_conflict_capital_feature_adversarial_and_explainability_evidence_clears",
    "resource_contract": "low_sample_digest_first_control_plane_with_no_direct_execution",
    "global_halt_contract": "coordination_layers_must_switch_to_summary_only_before_hard_halt",
    "paper_lock_contract": "no_direct_paper_or_live_execution_all_outputs_are_advisory_until_graduated",
}

BOTS: list[dict[str, Any]] = [
    {
        "role_slug": "lineage_genome_mapper",
        "slug": "coordination_lineage_genome_mapper_bot",
        "label": "Coordination Lineage Genome Mapper",
        "bot_role": "infrastructure_sub_bot",
        "layer": "bot_genome_lineage_map_v2",
        "priority": "critical",
        "objective": "Map each bot to founder DNA, parent concept, sleeve ancestry, and source lineage.",
        "target_functions": ["bot_genome_map", "founder_dna_lineage", "pycharm_core_catalog"],
    },
    {
        "role_slug": "lineage_trait_drift_guard",
        "slug": "coordination_lineage_trait_drift_guard_bot",
        "label": "Coordination Lineage Trait Drift Guard",
        "bot_role": "infrastructure_sub_bot",
        "layer": "bot_genome_lineage_map_v2",
        "priority": "high",
        "objective": "Detect when a bot drifts away from its declared DNA, sleeve, or operating contract.",
        "target_functions": ["lineage_drift_detection", "registry_regression_guard", "label_taxonomy_guard"],
    },
    {
        "role_slug": "lineage_visual_map_narrator",
        "slug": "coordination_lineage_visual_map_narrator_bot",
        "label": "Coordination Lineage Visual Map Narrator",
        "bot_role": "infrastructure_sub_bot",
        "layer": "bot_genome_lineage_map_v2",
        "priority": "medium",
        "objective": "Produce report-ready lineage map packets for system overview and PyCharm navigation.",
        "target_functions": ["system_report_narrative", "bot_catalog", "framework_map"],
    },
    {
        "role_slug": "strategy_conflict_detector",
        "slug": "coordination_strategy_conflict_detector_bot",
        "label": "Coordination Strategy Conflict Detector",
        "bot_role": "signal_sub_bot",
        "layer": "strategy_conflict_resolver",
        "priority": "critical",
        "objective": "Detect contradictory sleeve views across direction, factor, volatility, liquidity, and hedge intent.",
        "target_functions": ["cross_sleeve_conflict_matrix", "view_disagreement_detection", "signal_governance"],
    },
    {
        "role_slug": "cross_sleeve_view_netting_master",
        "slug": "coordination_cross_sleeve_view_netting_master_bot",
        "label": "Coordination Cross-Sleeve View Netting Master",
        "bot_role": "infrastructure_sub_bot",
        "layer": "strategy_conflict_resolver",
        "priority": "critical",
        "objective": "Net sleeve-level views into a single advisory packet before paper allocation surfaces consume them.",
        "target_functions": ["net_view_packet", "portfolio_risk_layer", "grandmaster_control_plane"],
    },
    {
        "role_slug": "conflict_regression_guard",
        "slug": "coordination_conflict_regression_guard_bot",
        "label": "Coordination Conflict Regression Guard",
        "bot_role": "infrastructure_sub_bot",
        "layer": "strategy_conflict_resolver",
        "priority": "critical",
        "objective": "Prevent unresolved conflicts from becoming paper trades, reports, or training labels.",
        "target_functions": ["paper_trade_lock", "training_label_guard", "global_halt_preemption"],
    },
    {
        "role_slug": "capital_flow_simulator",
        "slug": "coordination_capital_flow_simulator_bot",
        "label": "Coordination Capital Flow Simulator",
        "bot_role": "signal_sub_bot",
        "layer": "capital_allocation_simulator",
        "priority": "critical",
        "objective": "Simulate advisory capital flow across sleeves without touching broker or execution paths.",
        "target_functions": ["capital_flow_sandbox", "portfolio_allocator", "sleeve_capacity_curves"],
    },
    {
        "role_slug": "margin_drawdown_sensitivity",
        "slug": "coordination_margin_drawdown_sensitivity_bot",
        "label": "Coordination Margin Drawdown Sensitivity",
        "bot_role": "infrastructure_sub_bot",
        "layer": "capital_allocation_simulator",
        "priority": "critical",
        "objective": "Estimate margin, drawdown, concentration, and liquidation sensitivity from proposed sleeve weights.",
        "target_functions": ["margin_guard", "drawdown_stress", "collateral_margin_liquidity"],
    },
    {
        "role_slug": "allocation_sandbox_guard",
        "slug": "coordination_allocation_sandbox_guard_bot",
        "label": "Coordination Allocation Sandbox Guard",
        "bot_role": "infrastructure_sub_bot",
        "layer": "capital_allocation_simulator",
        "priority": "critical",
        "objective": "Keep allocation simulations advisory-only and block accidental promotion into execution lanes.",
        "target_functions": ["execution_block", "allocation_sandbox", "policy_regression_guard"],
    },
    {
        "role_slug": "regime_memory_writer",
        "slug": "coordination_regime_memory_writer_bot",
        "label": "Coordination Regime Memory Writer",
        "bot_role": "infrastructure_sub_bot",
        "layer": "market_regime_memory",
        "priority": "high",
        "objective": "Write compact regime memories for what worked, failed, stalled, or overloaded by sleeve.",
        "target_functions": ["regime_memory_store", "model_lifecycle", "stress_lab"],
    },
    {
        "role_slug": "regime_playbook_retriever",
        "slug": "coordination_regime_playbook_retriever_bot",
        "label": "Coordination Regime Playbook Retriever",
        "bot_role": "signal_sub_bot",
        "layer": "market_regime_memory",
        "priority": "high",
        "objective": "Retrieve comparable prior regimes and suggest advisory playbooks for current conditions.",
        "target_functions": ["regime_playbook_lookup", "market_regime_router", "cognitive_control_plane"],
    },
    {
        "role_slug": "regime_memory_decay_guard",
        "slug": "coordination_regime_memory_decay_guard_bot",
        "label": "Coordination Regime Memory Decay Guard",
        "bot_role": "infrastructure_sub_bot",
        "layer": "market_regime_memory",
        "priority": "high",
        "objective": "Retire stale regime lessons when evidence decays or market structure changes.",
        "target_functions": ["memory_decay_guard", "model_decay_detector", "feature_freshness"],
    },
    {
        "role_slug": "research_ingest_classifier",
        "slug": "coordination_research_ingest_classifier_bot",
        "label": "Coordination Research Ingest Classifier",
        "bot_role": "infrastructure_sub_bot",
        "layer": "research_to_bot_pipeline",
        "priority": "high",
        "objective": "Classify papers, manual notes, and research ideas into sleeve, model, bot, report, or reject lanes.",
        "target_functions": ["research_classification_queue", "recursive_research_foundry", "research_automation"],
    },
    {
        "role_slug": "research_to_bot_spec_builder",
        "slug": "coordination_research_to_bot_spec_builder_bot",
        "label": "Coordination Research-to-Bot Spec Builder",
        "bot_role": "infrastructure_sub_bot",
        "layer": "research_to_bot_pipeline",
        "priority": "high",
        "objective": "Draft bounded bot specs with data, label, retention, training, and risk contracts.",
        "target_functions": ["bot_spec_candidates", "bot_admission_guard", "registry_schema"],
    },
    {
        "role_slug": "research_admission_queue_guard",
        "slug": "coordination_research_admission_queue_guard_bot",
        "label": "Coordination Research Admission Queue Guard",
        "bot_role": "infrastructure_sub_bot",
        "layer": "research_to_bot_pipeline",
        "priority": "critical",
        "objective": "Throttle research-to-bot admissions when data quality, storage, CPU, or duplicate-alpha checks are weak.",
        "target_functions": ["admission_queue_guard", "expansion_capacity_planner", "feature_quality_gate"],
    },
    {
        "role_slug": "feature_quality_scorer",
        "slug": "coordination_feature_quality_scorer_bot",
        "label": "Coordination Feature Quality Scorer",
        "bot_role": "signal_sub_bot",
        "layer": "feature_store_quality_layer",
        "priority": "critical",
        "objective": "Score features by freshness, missingness, predictive value, stability, and point-in-time safety.",
        "target_functions": ["feature_quality_scores", "feature_store", "label_quality"],
    },
    {
        "role_slug": "feature_overlap_leakage_guard",
        "slug": "coordination_feature_overlap_leakage_guard_bot",
        "label": "Coordination Feature Overlap Leakage Guard",
        "bot_role": "infrastructure_sub_bot",
        "layer": "feature_store_quality_layer",
        "priority": "critical",
        "objective": "Detect duplicate features, lookahead leakage, overfit joins, and redundant data ingestion.",
        "target_functions": ["feature_overlap_map", "leakage_guard", "data_ingestion_dedupe"],
    },
    {
        "role_slug": "feature_store_contract_master",
        "slug": "coordination_feature_store_contract_master_bot",
        "label": "Coordination Feature Store Contract Master",
        "bot_role": "infrastructure_sub_bot",
        "layer": "feature_store_quality_layer",
        "priority": "critical",
        "objective": "Define feature contracts so every sleeve knows what is fresh, stale, missing, or untrusted.",
        "target_functions": ["feature_contracts", "sleeve_data_quality", "grandmaster_feature_summary"],
    },
    {
        "role_slug": "adversarial_fill_stress",
        "slug": "coordination_adversarial_fill_stress_bot",
        "label": "Coordination Adversarial Fill Stress",
        "bot_role": "infrastructure_sub_bot",
        "layer": "adversarial_paper_trading_lab",
        "priority": "critical",
        "objective": "Replay paper trades under widened spreads, bad fills, slippage spikes, and partial fills.",
        "target_functions": ["adversarial_fill_replay", "execution_lab", "paper_live_parity"],
    },
    {
        "role_slug": "stale_quote_delay_lab",
        "slug": "coordination_stale_quote_delay_lab_bot",
        "label": "Coordination Stale Quote Delay Lab",
        "bot_role": "infrastructure_sub_bot",
        "layer": "adversarial_paper_trading_lab",
        "priority": "critical",
        "objective": "Inject stale quotes, delayed feeds, and data-source disagreement into paper-trade simulations.",
        "target_functions": ["stale_quote_delay_report", "provider_adapter_verification", "data_source_divergence"],
    },
    {
        "role_slug": "broker_fault_injection_guard",
        "slug": "coordination_broker_fault_injection_guard_bot",
        "label": "Coordination Broker Fault Injection Guard",
        "bot_role": "infrastructure_sub_bot",
        "layer": "adversarial_paper_trading_lab",
        "priority": "critical",
        "objective": "Simulate auth failures, entitlement gaps, route failures, and halt conditions without broker side effects.",
        "target_functions": ["broker_fault_lab_packet", "auth_lease_manager", "global_halt_clearance"],
    },
    {
        "role_slug": "sleeve_master_conviction_summarizer",
        "slug": "coordination_sleeve_master_conviction_summarizer_bot",
        "label": "Coordination Sleeve Master Conviction Summarizer",
        "bot_role": "infrastructure_sub_bot",
        "layer": "sleeve_master_upgrade_pack",
        "priority": "critical",
        "objective": "Summarize conviction, disagreement, and evidence quality for each major sleeve master.",
        "target_functions": ["sleeve_conviction_summary", "per_sleeve_master_bots", "signal_governance"],
    },
    {
        "role_slug": "sleeve_master_health_router",
        "slug": "coordination_sleeve_master_health_router_bot",
        "label": "Coordination Sleeve Master Health Router",
        "bot_role": "infrastructure_sub_bot",
        "layer": "sleeve_master_upgrade_pack",
        "priority": "critical",
        "objective": "Route sleeve health, halt pressure, data quality, and training readiness to grandmaster.",
        "target_functions": ["sleeve_health_route", "global_halt_status", "training_readiness"],
    },
    {
        "role_slug": "grandmaster_briefing_bridge",
        "slug": "coordination_grandmaster_briefing_bridge_bot",
        "label": "Coordination Grandmaster Briefing Bridge",
        "bot_role": "infrastructure_sub_bot",
        "layer": "sleeve_master_upgrade_pack",
        "priority": "critical",
        "objective": "Convert sleeve master outputs into a concise grandmaster-ready control packet.",
        "target_functions": ["grandmaster_brief_packet", "operator_cockpit", "professional_system_dashboard"],
    },
    {
        "role_slug": "bot_duplicate_impact_screen",
        "slug": "coordination_bot_duplicate_impact_screen_bot",
        "label": "Coordination Bot Duplicate Impact Screen",
        "bot_role": "infrastructure_sub_bot",
        "layer": "bot_admission_committee",
        "priority": "critical",
        "objective": "Reject duplicate or low-novelty bot proposals before they consume storage, CPU, or label space.",
        "target_functions": ["duplicate_idea_screen", "alpha_similarity", "bot_namespace_guard"],
    },
    {
        "role_slug": "bot_admission_capacity_vote",
        "slug": "coordination_bot_admission_capacity_vote_bot",
        "label": "Coordination Bot Admission Capacity Vote",
        "bot_role": "infrastructure_sub_bot",
        "layer": "bot_admission_committee",
        "priority": "critical",
        "objective": "Vote on new bots using storage, CPU, memory, data availability, and training maturity costs.",
        "target_functions": ["bot_admission_vote", "expansion_capacity_planner", "runtime_capacity_governance"],
    },
    {
        "role_slug": "admission_policy_regression_guard",
        "slug": "coordination_admission_policy_regression_guard_bot",
        "label": "Coordination Admission Policy Regression Guard",
        "bot_role": "infrastructure_sub_bot",
        "layer": "bot_admission_committee",
        "priority": "critical",
        "objective": "Keep the bot floor, paper lock, training thresholds, and storage guards intact during future expansions.",
        "target_functions": ["admission_policy_regression", "paper_trade_lock_recovery", "storage_guard"],
    },
    {
        "role_slug": "decision_provenance_explainer",
        "slug": "coordination_decision_provenance_explainer_bot",
        "label": "Coordination Decision Provenance Explainer",
        "bot_role": "infrastructure_sub_bot",
        "layer": "system_explainability_dashboard",
        "priority": "high",
        "objective": "Explain which bots, sleeves, features, and health gates influenced an advisory decision.",
        "target_functions": ["decision_provenance_story", "explainability_trace", "reporting_layer"],
    },
    {
        "role_slug": "what_changed_today_narrator",
        "slug": "coordination_what_changed_today_narrator_bot",
        "label": "Coordination What Changed Today Narrator",
        "bot_role": "infrastructure_sub_bot",
        "layer": "system_explainability_dashboard",
        "priority": "medium",
        "objective": "Narrate meaningful daily changes in bots, data quality, halts, pressure, and regime state.",
        "target_functions": ["daily_system_change_narrative", "system_overview_report", "operator_cockpit"],
    },
    {
        "role_slug": "trust_surface_dashboard_builder",
        "slug": "coordination_trust_surface_dashboard_builder_bot",
        "label": "Coordination Trust Surface Dashboard Builder",
        "bot_role": "infrastructure_sub_bot",
        "layer": "system_explainability_dashboard",
        "priority": "high",
        "objective": "Score which signals, sleeves, data sources, and models should be trusted less right now.",
        "target_functions": ["trust_surface_dashboard", "feature_quality_scores", "model_risk_validation"],
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


def _slot_kind(bot: dict[str, Any]) -> str:
    return f"{COORDINATION_SLUG}_{bot['role_slug']}"


def _next_available_version(used_versions: set[int], start: int) -> int:
    version = start
    while version in used_versions:
        version += 1
    used_versions.add(version)
    return version


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
        desired = BASE_VERSION + index
        version = desired if desired not in used_versions else _next_available_version(
            used_versions, max(max(used_versions, default=BASE_VERSION - 1) + 1, desired)
        )
        used_versions.add(version)
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


def _layer(bot: dict[str, Any]) -> dict[str, Any]:
    for layer in LAYERS:
        if layer["slug"] == bot["layer"]:
            return layer
    return {"slug": bot["layer"], "display_name": bot["layer"], "objective": bot["objective"], "outputs": []}


def _coordination_contract(assigned_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": COORDINATION_VERSION,
        "new_sleeve_or_subsleeve": {
            "sleeve_family": COORDINATION_FAMILY,
            "sleeve_profile": COORDINATION_PROFILE,
            "display_name": COORDINATION_DISPLAY_NAME,
        },
        "layer_count": len(LAYERS),
        "bot_count": len(BOTS),
        "bots_per_layer": 3,
        "bot_pack_size_rule": "10_layers_3_bots_each_30_total_max",
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "coordination_hot_7d_warm_45d_cold_365d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": 28,
            "capture_mode": "sampled_digest_first",
            "sample_rate": 0.18,
            "dedupe_required": True,
            "stale_deletion_policy": "summarize_to_control_digest_then_stage_low_value_raw_traces",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "sleeve_master_bot_id": assigned_ids.get(f"{COORDINATION_SLUG}_cross_sleeve_view_netting_master", ""),
        "regression_guard_bot_id": assigned_ids.get(f"{COORDINATION_SLUG}_conflict_regression_guard", ""),
        "admission_committee_bot_id": assigned_ids.get(f"{COORDINATION_SLUG}_bot_admission_capacity_vote", ""),
        "grandmaster_bridge_bot_id": assigned_ids.get(f"{COORDINATION_SLUG}_grandmaster_briefing_bridge", ""),
        "capacity_check": {
            "active_bot_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
            "runtime_stability_mode": "full_force_buffered",
            "runtime_control_refresh_seconds": 240,
            "queue_policy": "buffered_jsonl_batching",
            "compute_guard_mode": "sustain",
            "global_halt_mode": "summary_only_before_hard_halt",
            "heavy_reasoning_mode": "digest_only_until_host_pressure_clear",
        },
        "coordination_contract": COORDINATION_CONTRACT,
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    layer = _layer(bot)
    contract = _coordination_contract(assigned_ids)
    advanced_contract = {
        "contract_version": "coordination_intelligence_layers_v1",
        "coordination_version": COORDINATION_VERSION,
        "capability_pack": COORDINATION_SLUG,
        "coordination_layer": bot["layer"],
        "coordination_layer_display_name": layer["display_name"],
        "reports_to_sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "critic_guard_bot_id": contract["regression_guard_bot_id"],
        "admission_committee_bot_id": contract["admission_committee_bot_id"],
        "grandmaster_bridge_bot_id": contract["grandmaster_bridge_bot_id"],
        "layer_outputs": list(layer.get("outputs", [])),
        "conflict_resolution": "net_cross_sleeve_views_before_paper_or_allocation_surfaces",
        "capital_simulation": "dry_run_only_no_broker_or_order_path",
        "regime_memory": "point_in_time_regime_playbooks_with_decay",
        "feature_quality": "freshness_missingness_leakage_overlap_and_value_scoring",
        "adversarial_lab": "bad_fill_stale_quote_broker_fault_and_liquidity_gap_replays",
        "explainability": "decision_provenance_daily_change_and_trust_surface_outputs",
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
            "brain_refinery_v819_cognitive_planning_orchestrator_master_bot",
            "brain_refinery_v833_cognitive_grandmaster_cognition_bridge_bot",
            "brain_refinery_v848_recursive_foundry_grandmaster_foundry_bridge_bot",
        ],
        "data_intake_collections": list(DATA_INTAKES),
        "storage_targets": list(STORAGE_TARGETS),
        "freshness_slo_seconds": 1200,
        "retention_profile": "coordination_hot_7d_warm_45d_cold_365d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "coordination_control_plane_observer_until_conflict_capital_feature_adversarial_thresholds_clear",
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
        "training_exclusion_reason": "collecting_coordination_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "training_threshold_policy": "eligible_when_observations_days_conflict_feature_and_adversarial_checks_clear",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "promotion_blocked_until": "minimum_data_collection_threshold_met",
        "promotion_block_reason": "coordination_intelligence_collection_only_no_training_yet",
        "data_collection_storage_guarded": True,
        "data_collection_storage_guard_mode": "normal",
        "data_collection_capture_mode": "sampled_digest_first",
        "data_collection_sample_rate": 0.18,
        "data_collection_max_daily_storage_mb": 28,
        "data_collection_storage_guard_updated_utc": now,
        "storage_pressure_capture_reason": "coordination_distilled_control_plane_collection",
        "data_collection_compute_guard_mode": "sustain",
        "data_collection_resource_guard_reason": "coordination_low_sample_digest_first",
        "data_collection_max_daily_mb": 28,
        "collected_observation_count": 0,
        "data_collection_last_counted_utc": "",
        "data_collection_observation_rollup_source": "awaiting_first_incremental_rollup",
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "sleeve_profile": COORDINATION_PROFILE,
        "sleeve_family": COORDINATION_FAMILY,
        "coordination_layer": bot["layer"],
        "correlation_peer_sleeves": [
            "recursive_research_foundry",
            "cognitive_control_plane",
            "advanced_intelligence_mesh",
            "portfolio_risk_layer",
            "execution_intelligence",
            "event_intelligence",
            "model_lifecycle",
            "reporting_layer",
        ],
        "correlation_dependencies": [
            "bot_founder_dna",
            "cross_sleeve_correlation_matrix",
            "portfolio_allocator",
            "feature_store_quality",
            "global_halt_status",
            "ingestion_storage_control",
            "runtime_throttle",
        ],
        "provider_capability_profile": "internal_coordination_control_plane_only",
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": [
            "master_bot_registry",
            "bot_catalog",
            "platform_intelligence_expansion",
            "decision_provenance",
            "paper_execution_calibration",
            "execution_lab",
            "feature_quality_data_confidence",
            "regime_control_plane",
            "recursive_research_foundry",
        ],
        "schwab_direct_inputs": [],
        "proxy_only_reason": "coordination_intelligence_is_advisory_and_has_no_direct_broker_dependency",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.74,
            "freshness_slo_seconds": 1200,
            "regression_guard_bot_id": contract["regression_guard_bot_id"],
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": [
            "coordination_control_plane",
            "collection_only",
            "execution_blocked",
            "paper_only_floor",
            f"sleeve_family:{COORDINATION_FAMILY}",
            f"sleeve_profile:{COORDINATION_PROFILE}",
            f"capability_pack:{COORDINATION_SLUG}",
            f"coordination_layer:{bot['layer']}",
            f"data_floor:{MINIMUM_TRAINING_OBSERVATIONS}",
            "training_after_threshold",
            "global_halt_aware",
            "strategy_conflict_aware",
            "capital_simulation_only",
            "explainability_surface",
        ],
        "execution_policy_label": "collection_only_coordination_intelligence_no_execution",
        "eligible_for_master_vote": False,
        "data_collection_runtime_dependency_profile": "low_sample_digest_first_control_plane",
        "founder_bot_id": "brain_refinery_v1",
        "founder_dna_version": "founder_dna_v1",
        "founder_dna_scope": "brain_refinery_v1_to_full_fleet",
        "founder_dna_source": "registry_inferred_full_fleet_contract",
        "founder_dna_confidence": 0.82,
        "founder_dna_traits": [
            "market_data_observation",
            "paper_first_safety",
            "global_halt_awareness",
            "resource_throttle_awareness",
            "training_lineage",
            "decision_explanation_contract",
            "data_collection_before_training",
            "registry_auditable_identity",
            "cross_sleeve_coordination",
        ],
        "founder_dna_inheritance_mode": "explicit_contract_metadata",
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "lineage_regression_guard": "fail_if_founder_dna_missing_or_stale",
        "lineage_generation": 6,
        "paper_runtime_stability_mode": "full_force_buffered",
        "paper_execution_queue_policy": "buffered_jsonl_batching",
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "paper_trade_lock_required": True,
        "paper_runtime_control_refresh_seconds": 240,
        "capability_pack_version": COORDINATION_VERSION,
        "capability_pack_slug": COORDINATION_SLUG,
        "capability_pack_display_name": COORDINATION_DISPLAY_NAME,
        "coordination_intelligence_version": COORDINATION_VERSION,
        "capability_pack_contract": contract,
        "advanced_intelligence_layer_contract": advanced_contract,
    }


def _refresh_summary(registry: dict[str, Any]) -> None:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    capability_slugs = {str(row.get("capability_pack_slug") or "") for row in rows if str(row.get("capability_pack_slug") or "")}
    summary = dict(registry.get("summary") or {})
    summary.update(
        {
            "total_bots": len(rows),
            "active_bots": len(active),
            "inactive_bots": len(inactive),
            "active_signal_sub_bots": sum(1 for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"),
            "active_infrastructure_sub_bots": sum(1 for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"),
            "inactive_signal_sub_bots": sum(1 for row in inactive if str(row.get("bot_role") or "") == "signal_sub_bot"),
            "inactive_infrastructure_sub_bots": sum(1 for row in inactive if str(row.get("bot_role") or "") == "infrastructure_sub_bot"),
            "data_collection_active_bots": sum(1 for row in rows if bool(row.get("data_collection_active"))),
            "training_excluded_bots": sum(1 for row in rows if bool(row.get("training_excluded"))),
            "structured_capability_pack_count": len(capability_slugs),
            "structured_capability_pack_bot_count": sum(1 for row in rows if str(row.get("capability_pack_version") or "")),
            "advanced_intelligence_mesh_bot_count": sum(1 for row in rows if str(row.get("advanced_mesh_version") or "")),
            "cognitive_control_plane_bot_count": sum(1 for row in rows if str(row.get("cognitive_control_version") or "")),
            "recursive_research_foundry_bot_count": sum(1 for row in rows if str(row.get("recursive_foundry_version") or "")),
            "coordination_intelligence_bot_count": sum(
                1 for row in rows if str(row.get("coordination_intelligence_version") or "") == COORDINATION_VERSION
            ),
            "latest_coordination_intelligence": COORDINATION_VERSION,
        }
    )
    registry["summary"] = summary


def _coordination_summary(assigned_ids: dict[str, str]) -> dict[str, Any]:
    contract = _coordination_contract(assigned_ids)
    return {
        "slug": COORDINATION_SLUG,
        "display_name": COORDINATION_DISPLAY_NAME,
        "sleeve_family": COORDINATION_FAMILY,
        "sleeve_profile": COORDINATION_PROFILE,
        "objective": "Coordinate lineage, conflict resolution, allocation simulation, regime memory, research admission, feature quality, adversarial paper testing, sleeve masters, admission, and explainability.",
        "layer_count": len(LAYERS),
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "layers": list(LAYERS),
        "dedicated_data_intake": list(DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "regression_guard_bot_id": contract["regression_guard_bot_id"],
        "admission_committee_bot_id": contract["admission_committee_bot_id"],
        "grandmaster_bridge_bot_id": contract["grandmaster_bridge_bot_id"],
        "capacity_check": contract["capacity_check"],
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
        "coordination_version": COORDINATION_VERSION,
        "layer_count": len(LAYERS),
        "bot_count": len(BOTS),
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped_existing),
        "planned_rows": planned_rows,
        "skipped_existing_slot_kinds": skipped_existing,
        "coordination": _coordination_summary(assigned_ids),
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
        "coordination_version": COORDINATION_VERSION,
        "layer_count": plan["layer_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "coordination": plan["coordination"],
        "coordination_contract": COORDINATION_CONTRACT,
        "recommended_apply_command": "./scripts/ops/opsctl.sh coordination-intelligence --apply --json",
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
        backup = backup_dir / f"master_bot_registry_before_coordination_intelligence_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
        project_root / "config" / "coordination_intelligence_pack_v1.json",
        {
            "generated_at_utc": _utc_now(),
            "coordination_version": COORDINATION_VERSION,
            "coordination": payload["coordination"],
            "coordination_contract": payload["coordination_contract"],
        },
    )
    _write_json(project_root / "governance" / "health" / "coordination_intelligence_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add a 30-bot coordination intelligence control-plane expansion.")
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
            "coordination_intelligence "
            f"mode={payload['mode']} layers={payload['layer_count']} "
            f"bots={payload['bot_count']} planned={payload['planned_bot_count']} "
            f"added={payload.get('added_bot_count', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
