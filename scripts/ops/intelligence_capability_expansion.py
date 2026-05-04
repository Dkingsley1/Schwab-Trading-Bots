#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "intelligence_capability_packs_v1.json"
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "intelligence_capability_expansion_latest.json"
BACKUP_DIR = PROJECT_ROOT / "backups"

BASE_VERSION = 764
CAPABILITY_PACK_VERSION = "intelligence_capability_pack_v1"
LABEL_CONTRACT_VERSION = "structured_capability_label_v1"
MINIMUM_TRAINING_OBSERVATIONS = 3000
MINIMUM_COLLECTION_DAYS = 14
PAPER_RUNTIME_CAPACITY_FLOOR = 700

FOUNDER_DNA_TRAITS = [
    "market_data_observation",
    "paper_first_safety",
    "global_halt_awareness",
    "resource_throttle_awareness",
    "training_lineage",
    "decision_explanation_contract",
    "data_collection_before_training",
    "registry_auditable_identity",
    "structured_capability_pack_governance",
]

ADVANCED_INTELLIGENCE_LAYER_CONTRACT = {
    "contract_version": "advanced_intelligence_layers_v2",
    "layer_stack": [
        "perception_data_normalization",
        "reasoning_signal_and_hypothesis_scoring",
        "governance_sleeve_master_aggregation",
        "critic_regression_and_label_validation",
        "capacity_resource_budgeting",
        "memory_experience_accumulation",
    ],
    "cross_pack_memory": "experience_accumulation_memory_trace",
    "critic_loop": "observer_critic_loop_trace",
    "semantic_ontology": "semantic_feature_ontology_trace",
    "causal_evidence": "causal_intervention_ledger_trace",
    "market_state_graph": "cross_asset_correlation_context",
    "resource_budget": "compute_capital_allocation_trace",
    "global_halt_contract": "expansion_pressure_aware_soft_cap_before_hard_halt",
}


PACKS: list[dict[str, Any]] = [
    {
        "slug": "execution_intelligence",
        "display_name": "Execution Intelligence",
        "sleeve_family": "execution",
        "sleeve_profile": "execution_intelligence",
        "objective": "Measure fill quality, latency, slippage, and paper/live parity before any execution logic can graduate.",
        "preferred_regimes": ["all_weather", "fragile_transition", "high_volatility", "liquidity_thin"],
        "correlation_peer_sleeves": ["intraday_aggressive", "options", "futures", "crypto_futures"],
        "correlation_dependencies": ["execution_lab", "risk_service", "portfolio_allocator"],
        "data_intakes": [
            "fill_quality_trace",
            "paper_live_slippage_gap",
            "route_latency_histogram",
            "partial_fill_context",
            "spread_decay_context",
        ],
        "storage_targets": [
            "data/execution_intelligence/fill_quality",
            "data/execution_intelligence/latency",
            "governance/health/execution_intelligence",
            "data/jsonl_link.sqlite3",
        ],
        "required_labels": ["fill_quality_bucket", "slippage_bucket", "latency_bucket", "paper_live_parity_flag"],
        "proxy_data_sources": ["internal_paper_trades", "execution_lab", "broker_truth_snapshot", "market_micro_context"],
        "schwab_direct_inputs": ["quotes", "orders", "positions"],
        "retention_profile": "execution_hot_7d_warm_45d_cold_180d",
        "freshness_slo_seconds": 180,
        "bots": [
            {
                "role_slug": "fill_quality_collector",
                "slug": "execution_intelligence_fill_quality_collector_bot",
                "label": "Execution Intelligence Fill Quality Collector",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "perception",
                "priority": "high",
                "objective": "Collect fill, partial-fill, latency, and slippage traces as normalized execution evidence.",
                "target_functions": ["execution_lab", "paper_trade_lock", "broker_truth_snapshot", "decision_provenance"],
            },
            {
                "role_slug": "slippage_forecaster",
                "slug": "execution_intelligence_slippage_forecaster_bot",
                "label": "Execution Intelligence Slippage Forecaster",
                "bot_role": "signal_sub_bot",
                "intelligence_layer": "reasoning",
                "priority": "high",
                "objective": "Score slippage risk by spread, queue context, volatility, and time-of-day before routing.",
                "target_functions": ["execution_lab", "route_timing", "paper_execution_calibration"],
            },
            {
                "role_slug": "route_timing_master",
                "slug": "execution_intelligence_route_timing_master_bot",
                "label": "Execution Intelligence Route Timing Master",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "governance",
                "priority": "critical",
                "objective": "Own sleeve-level routing posture and summarize execution health for the grand master layer.",
                "target_functions": ["sleeve_master", "execution_lab", "risk_service", "operator_cockpit"],
            },
            {
                "role_slug": "paper_live_parity_regression_guard",
                "slug": "execution_intelligence_paper_live_parity_regression_guard_bot",
                "label": "Execution Intelligence Paper/Live Parity Regression Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "critic",
                "priority": "critical",
                "objective": "Block graduation when paper fills diverge from live-observable market conditions.",
                "target_functions": ["regression_guard", "paper_trade_lock", "global_halt_status", "execution_lab"],
            },
            {
                "role_slug": "latency_capacity_guard",
                "slug": "execution_intelligence_latency_capacity_guard_bot",
                "label": "Execution Intelligence Latency Capacity Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "capacity",
                "priority": "critical",
                "objective": "Throttle execution evidence collection when latency, storage, or queue backlog rises.",
                "target_functions": ["runtime_throttle", "backpressure_drainer_fleet", "ingestion_storage_control"],
            },
        ],
    },
    {
        "slug": "portfolio_risk_layer",
        "display_name": "Portfolio Risk Layer",
        "sleeve_family": "risk",
        "sleeve_profile": "portfolio_risk_layer",
        "objective": "Unify exposure, margin, convexity, correlation, and capital efficiency across every sleeve.",
        "preferred_regimes": ["all_weather", "risk_off_shock", "correlation_break", "margin_stress"],
        "correlation_peer_sleeves": ["options", "futures", "crypto", "dividend", "bond", "macro"],
        "correlation_dependencies": ["portfolio_allocator", "portfolio_capacity_curves", "risk_service"],
        "data_intakes": [
            "cross_sleeve_exposure_netting",
            "margin_guard_trace",
            "convexity_exposure_surface",
            "correlation_matrix",
            "capital_efficiency_surface",
        ],
        "storage_targets": [
            "data/portfolio_risk/exposures",
            "data/portfolio_risk/margin",
            "governance/health/portfolio_risk",
            "data/jsonl_link.sqlite3",
        ],
        "required_labels": ["net_exposure_bucket", "margin_pressure_bucket", "convexity_bucket", "correlation_cluster"],
        "proxy_data_sources": ["broker_truth_snapshot", "risk_service", "portfolio_capacity_curves", "paper_positions"],
        "schwab_direct_inputs": ["positions", "balances", "orders", "quotes"],
        "retention_profile": "risk_hot_14d_warm_90d_cold_365d",
        "freshness_slo_seconds": 240,
        "bots": [
            {
                "role_slug": "exposure_netting_collector",
                "slug": "portfolio_risk_exposure_netting_collector_bot",
                "label": "Portfolio Risk Exposure Netting Collector",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "perception",
                "priority": "critical",
                "objective": "Collect normalized net exposure and margin context across sleeves.",
                "target_functions": ["risk_service", "portfolio_allocator", "margin_guard_trace"],
            },
            {
                "role_slug": "correlation_cap_allocator",
                "slug": "portfolio_risk_correlation_cap_allocator_bot",
                "label": "Portfolio Risk Correlation Cap Allocator",
                "bot_role": "signal_sub_bot",
                "intelligence_layer": "reasoning",
                "priority": "high",
                "objective": "Estimate correlation concentration and recommend sleeve caps before margin pressure appears.",
                "target_functions": ["portfolio_allocator", "capacity_curves", "correlation_governor"],
            },
            {
                "role_slug": "sleeve_master",
                "slug": "portfolio_risk_sleeve_master_bot",
                "label": "Portfolio Risk Sleeve Master",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "governance",
                "priority": "critical",
                "objective": "Aggregate portfolio risk readiness and report capital posture upstream.",
                "target_functions": ["sleeve_master", "risk_service", "grand_master_reporting"],
            },
            {
                "role_slug": "margin_convexity_regression_guard",
                "slug": "portfolio_risk_margin_convexity_regression_guard_bot",
                "label": "Portfolio Risk Margin Convexity Regression Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "critic",
                "priority": "critical",
                "objective": "Detect margin guard slams, convexity mislabels, and exposure netting regressions.",
                "target_functions": ["regression_guard", "margin_guard", "risk_service"],
            },
            {
                "role_slug": "capacity_budget_guard",
                "slug": "portfolio_risk_capacity_budget_guard_bot",
                "label": "Portfolio Risk Capacity Budget Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "capacity",
                "priority": "critical",
                "objective": "Protect risk aggregation from oversized joins, redundant snapshots, and storage churn.",
                "target_functions": ["runtime_throttle", "ingestion_storage_control", "portfolio_capacity_curves"],
            },
        ],
    },
    {
        "slug": "event_intelligence",
        "display_name": "Event Intelligence",
        "sleeve_family": "macro_event",
        "sleeve_profile": "event_intelligence",
        "objective": "Fuse calendars, speakers, macro releases, transcripts, and earnings clusters into point-in-time event context.",
        "preferred_regimes": ["macro_release", "fed_day", "earnings_cluster", "auction_day"],
        "correlation_peer_sleeves": ["macro", "bond", "fx", "index", "dividend"],
        "correlation_dependencies": ["macro_context_sync", "macro_auto_watch", "event_store"],
        "data_intakes": [
            "fed_speaker_calendar",
            "earnings_cluster_events",
            "treasury_auction_schedule",
            "cpi_pce_nfp_surprise",
            "macro_media_transcript_trace",
        ],
        "storage_targets": [
            "data/event_intelligence/calendar",
            "data/event_intelligence/transcripts",
            "governance/health/event_intelligence",
            "data/event_store.sqlite3",
        ],
        "required_labels": ["event_type", "event_surprise_bucket", "speaker_importance", "event_decay_window"],
        "proxy_data_sources": ["federal_reserve_calendar", "treasury_calendar", "sec_edgar", "macro_media_ingest"],
        "schwab_direct_inputs": ["quotes"],
        "retention_profile": "event_hot_30d_warm_180d_cold_730d",
        "freshness_slo_seconds": 300,
        "bots": [
            {
                "role_slug": "macro_calendar_collector",
                "slug": "event_intelligence_macro_calendar_collector_bot",
                "label": "Event Intelligence Macro Calendar Collector",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "perception",
                "priority": "critical",
                "objective": "Collect point-in-time calendars for Fed, Treasury, earnings, and macro releases.",
                "target_functions": ["macro_context_sync", "event_store", "source_verification"],
            },
            {
                "role_slug": "surprise_decay_signal",
                "slug": "event_intelligence_surprise_decay_signal_bot",
                "label": "Event Intelligence Surprise Decay Signal",
                "bot_role": "signal_sub_bot",
                "intelligence_layer": "reasoning",
                "priority": "high",
                "objective": "Estimate how event surprises decay into sleeve-level signal usefulness.",
                "target_functions": ["market_regime_router", "macro_replay", "decision_provenance"],
            },
            {
                "role_slug": "cluster_sleeve_master",
                "slug": "event_intelligence_cluster_sleeve_master_bot",
                "label": "Event Intelligence Cluster Sleeve Master",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "governance",
                "priority": "critical",
                "objective": "Summarize event cluster posture and coordinate macro-aware sleeve traffic.",
                "target_functions": ["sleeve_master", "macro_auto_watch", "operator_cockpit"],
            },
            {
                "role_slug": "calendar_label_regression_guard",
                "slug": "event_intelligence_calendar_label_regression_guard_bot",
                "label": "Event Intelligence Calendar Label Regression Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "critic",
                "priority": "critical",
                "objective": "Catch stale, shifted, duplicated, or future-leaking event labels.",
                "target_functions": ["regression_guard", "artifact_freshness_slo", "source_verification"],
            },
            {
                "role_slug": "feed_freshness_capacity_guard",
                "slug": "event_intelligence_feed_freshness_capacity_guard_bot",
                "label": "Event Intelligence Feed Freshness Capacity Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "capacity",
                "priority": "high",
                "objective": "Budget transcript and calendar refreshes so event intake cannot overwhelm live loops.",
                "target_functions": ["runtime_throttle", "collector_contracts", "ingestion_storage_control"],
            },
        ],
    },
    {
        "slug": "market_microstructure",
        "display_name": "Market Microstructure",
        "sleeve_family": "microstructure",
        "sleeve_profile": "market_microstructure",
        "objective": "Track spread, queue, auction, imbalance, and liquidity toxicity signals without raw-feed overload.",
        "preferred_regimes": ["market_open", "market_close", "high_volatility", "thin_liquidity"],
        "correlation_peer_sleeves": ["intraday_aggressive", "execution", "options", "futures", "crypto"],
        "correlation_dependencies": ["market_micro_sync", "execution_lab", "vpIN_order_flow_toxicity"],
        "data_intakes": [
            "spread_decay_context",
            "queue_position_context",
            "opening_auction_imbalance",
            "closing_auction_imbalance",
            "order_book_imbalance",
        ],
        "storage_targets": [
            "data/market_microstructure/spreads",
            "data/market_microstructure/auction",
            "governance/health/market_microstructure",
            "data/jsonl_link.sqlite3",
        ],
        "required_labels": ["spread_regime", "imbalance_bucket", "auction_pressure", "liquidity_toxicity_flag"],
        "proxy_data_sources": ["market_micro_context", "quotes", "paper_fill_context", "crypto_l2_proxy"],
        "schwab_direct_inputs": ["quotes", "options_chains"],
        "retention_profile": "microstructure_hot_3d_warm_30d_cold_120d",
        "freshness_slo_seconds": 120,
        "bots": [
            {
                "role_slug": "spread_queue_collector",
                "slug": "microstructure_spread_queue_collector_bot",
                "label": "Microstructure Spread Queue Collector",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "perception",
                "priority": "critical",
                "objective": "Sample spread, quote, queue, auction, and imbalance context at controlled cadence.",
                "target_functions": ["market_micro_sync", "execution_lab", "collector_contracts"],
            },
            {
                "role_slug": "liquidity_reversion_signal",
                "slug": "microstructure_liquidity_reversion_signal_bot",
                "label": "Microstructure Liquidity Reversion Signal",
                "bot_role": "signal_sub_bot",
                "intelligence_layer": "reasoning",
                "priority": "high",
                "objective": "Score when extreme imbalance or spread decay is likely to revert instead of continue.",
                "target_functions": ["intraday_aggressive", "execution_lab", "market_regime_router"],
            },
            {
                "role_slug": "auction_orderflow_master",
                "slug": "microstructure_auction_orderflow_master_bot",
                "label": "Microstructure Auction Orderflow Master",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "governance",
                "priority": "critical",
                "objective": "Own auction and order-flow posture for intraday and execution sleeves.",
                "target_functions": ["sleeve_master", "market_micro_sync", "execution_lab"],
            },
            {
                "role_slug": "lob_label_regression_guard",
                "slug": "microstructure_lob_label_regression_guard_bot",
                "label": "Microstructure LOB Label Regression Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "critic",
                "priority": "critical",
                "objective": "Detect bad imbalance labels, stale quote joins, and accidental raw-book overcollection.",
                "target_functions": ["regression_guard", "collector_contracts", "stale_sweeper"],
            },
            {
                "role_slug": "storage_sampling_guard",
                "slug": "microstructure_storage_sampling_guard_bot",
                "label": "Microstructure Storage Sampling Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "capacity",
                "priority": "critical",
                "objective": "Keep microstructure collection sampled, deduplicated, and pressure-aware.",
                "target_functions": ["runtime_throttle", "data_retention", "storage_quota_guard"],
            },
        ],
    },
    {
        "slug": "research_automation",
        "display_name": "Research Automation",
        "sleeve_family": "research",
        "sleeve_profile": "research_automation",
        "objective": "Turn new quant papers and experiments into traceable hypotheses with duplicate-alpha and evidence guards.",
        "preferred_regimes": ["research_cycle", "model_refresh", "hypothesis_triage", "low_pressure"],
        "correlation_peer_sleeves": ["model_lifecycle", "stress_lab", "reporting_layer"],
        "correlation_dependencies": ["research_pipeline", "experiment_ledger", "immutable_experiment_ledger"],
        "data_intakes": [
            "arxiv_qfin_recent_research_intake",
            "ssrn_market_infrastructure_reference",
            "alpha_hypothesis_graph_trace",
            "bayesian_evidence_score_trace",
            "duplicate_alpha_retirement_queue",
        ],
        "storage_targets": [
            "data/research_automation/papers",
            "data/research_automation/hypotheses",
            "governance/health/research_automation",
            "governance/experiment_ledger",
        ],
        "required_labels": ["hypothesis_family", "evidence_grade", "duplicate_alpha_flag", "implementation_readiness"],
        "proxy_data_sources": ["arxiv_qfin_recent", "ssrn_references", "internal_experiment_ledger"],
        "schwab_direct_inputs": [],
        "retention_profile": "research_hot_30d_warm_365d_cold_evergreen",
        "freshness_slo_seconds": 3600,
        "bots": [
            {
                "role_slug": "paper_ingestion_collector",
                "slug": "research_automation_paper_ingestion_collector_bot",
                "label": "Research Automation Paper Ingestion Collector",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "perception",
                "priority": "medium",
                "objective": "Collect research paper metadata and connect it to local hypothesis records.",
                "target_functions": ["research_pipeline", "source_verification", "experiment_ledger"],
            },
            {
                "role_slug": "hypothesis_evidence_scorer",
                "slug": "research_automation_hypothesis_evidence_scorer_bot",
                "label": "Research Automation Hypothesis Evidence Scorer",
                "bot_role": "signal_sub_bot",
                "intelligence_layer": "reasoning",
                "priority": "high",
                "objective": "Score research ideas by novelty, feasibility, evidence, and overlap with existing sleeves.",
                "target_functions": ["research_pipeline", "multiple_testing_guard", "model_lifecycle"],
            },
            {
                "role_slug": "research_committee_master",
                "slug": "research_automation_research_committee_master_bot",
                "label": "Research Automation Research Committee Master",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "governance",
                "priority": "high",
                "objective": "Coordinate research priorities and package approved ideas for controlled buildout.",
                "target_functions": ["sleeve_master", "experiment_ledger", "operator_cockpit"],
            },
            {
                "role_slug": "duplicate_alpha_regression_guard",
                "slug": "research_automation_duplicate_alpha_regression_guard_bot",
                "label": "Research Automation Duplicate Alpha Regression Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "critic",
                "priority": "critical",
                "objective": "Block repeated, overfit, or already-retired alpha ideas from re-entering production paths.",
                "target_functions": ["regression_guard", "multiple_testing_guard", "model_lifecycle"],
            },
            {
                "role_slug": "experiment_budget_guard",
                "slug": "research_automation_experiment_budget_guard_bot",
                "label": "Research Automation Experiment Budget Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "capacity",
                "priority": "high",
                "objective": "Limit research automation to low-pressure refresh windows and bounded storage.",
                "target_functions": ["runtime_throttle", "storage_quota_guard", "data_retention"],
            },
        ],
    },
    {
        "slug": "stress_lab",
        "display_name": "Stress Lab",
        "sleeve_family": "stress",
        "sleeve_profile": "stress_lab",
        "objective": "Replay crisis regimes and supervisory scenarios as reusable regression evidence.",
        "preferred_regimes": ["crisis_replay", "risk_off_shock", "liquidity_freeze", "pandemic_replay"],
        "correlation_peer_sleeves": ["portfolio_risk_layer", "macro", "options", "futures", "credit"],
        "correlation_dependencies": ["golden_replay_guard", "replay_hash_registry", "fed_2026_stress_scenarios"],
        "data_intakes": [
            "covid_2020_pandemic_replay_trace",
            "gfc_2008_replay_trace",
            "fed_2026_supervisory_scenario_dataset_trace",
            "flash_crash_replay_trace",
            "liquidity_freeze_scenario_trace",
        ],
        "storage_targets": [
            "data/stress_lab/scenarios",
            "data/stress_lab/replays",
            "governance/health/stress_lab",
            "governance/replay_hash_registry",
        ],
        "required_labels": ["scenario_family", "drawdown_bucket", "liquidity_stress_bucket", "replay_integrity_hash"],
        "proxy_data_sources": ["fed_stress_scenario_public_data", "market_history_cache", "macro_context_archive"],
        "schwab_direct_inputs": ["quotes"],
        "retention_profile": "stress_hot_30d_warm_365d_cold_evergreen",
        "freshness_slo_seconds": 1800,
        "bots": [
            {
                "role_slug": "scenario_dataset_collector",
                "slug": "stress_lab_scenario_dataset_collector_bot",
                "label": "Stress Lab Scenario Dataset Collector",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "perception",
                "priority": "critical",
                "objective": "Collect crisis replay datasets and supervisory scenario traces with hashes.",
                "target_functions": ["golden_replay_guard", "replay_hash_registry", "source_verification"],
            },
            {
                "role_slug": "covid_2008_replay_signal",
                "slug": "stress_lab_covid_2008_replay_signal_bot",
                "label": "Stress Lab COVID/2008 Replay Signal",
                "bot_role": "signal_sub_bot",
                "intelligence_layer": "reasoning",
                "priority": "high",
                "objective": "Compare current sleeve behavior against pandemic and GFC replay signatures.",
                "target_functions": ["stress_scenario_lab", "risk_service", "model_lifecycle"],
            },
            {
                "role_slug": "crisis_sleeve_master",
                "slug": "stress_lab_crisis_sleeve_master_bot",
                "label": "Stress Lab Crisis Sleeve Master",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "governance",
                "priority": "critical",
                "objective": "Own crisis replay posture and gate scenario readiness for master/grand master review.",
                "target_functions": ["sleeve_master", "global_halt_status", "risk_service"],
            },
            {
                "role_slug": "replay_label_regression_guard",
                "slug": "stress_lab_replay_label_regression_guard_bot",
                "label": "Stress Lab Replay Label Regression Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "critic",
                "priority": "critical",
                "objective": "Catch replay drift, bad scenario hashes, and future-leaking crisis labels.",
                "target_functions": ["regression_guard", "golden_replay_guard", "replay_hash_registry"],
            },
            {
                "role_slug": "compute_storage_budget_guard",
                "slug": "stress_lab_compute_storage_budget_guard_bot",
                "label": "Stress Lab Compute Storage Budget Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "capacity",
                "priority": "critical",
                "objective": "Run heavy replay collection only within bounded compute and storage budgets.",
                "target_functions": ["runtime_throttle", "memory_efficiency", "storage_quota_guard"],
            },
        ],
    },
    {
        "slug": "model_lifecycle",
        "display_name": "Model Lifecycle",
        "sleeve_family": "model_governance",
        "sleeve_profile": "model_lifecycle",
        "objective": "Standardize drift, challenger, leakage, overfit, retrain, and promotion evidence across the expanded fleet.",
        "preferred_regimes": ["training_window", "drift_watch", "model_decay", "promotion_review"],
        "correlation_peer_sleeves": ["research_automation", "portfolio_risk_layer", "reporting_layer"],
        "correlation_dependencies": ["model_lifecycle_hygiene", "training_requalification", "promotion_autopilot"],
        "data_intakes": [
            "model_calibration_decay",
            "challenger_drift_validation",
            "overfit_leakage_replay",
            "walk_forward_requalification",
            "promotion_readiness",
        ],
        "storage_targets": [
            "governance/model_lifecycle",
            "governance/walk_forward",
            "governance/training_diagnostics",
            "data/jsonl_link.sqlite3",
        ],
        "required_labels": ["drift_bucket", "leakage_flag", "promotion_readiness", "challenger_status"],
        "proxy_data_sources": ["training_diagnostics", "walk_forward_requalification", "promotion_readiness_summary"],
        "schwab_direct_inputs": [],
        "retention_profile": "model_hot_30d_warm_180d_cold_730d",
        "freshness_slo_seconds": 900,
        "bots": [
            {
                "role_slug": "decay_drift_collector",
                "slug": "model_lifecycle_decay_drift_collector_bot",
                "label": "Model Lifecycle Decay Drift Collector",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "perception",
                "priority": "critical",
                "objective": "Collect decay, calibration, drift, and challenger validation evidence.",
                "target_functions": ["model_lifecycle", "training_requalification", "decay_monitor"],
            },
            {
                "role_slug": "challenger_selector",
                "slug": "model_lifecycle_challenger_selector_bot",
                "label": "Model Lifecycle Challenger Selector",
                "bot_role": "signal_sub_bot",
                "intelligence_layer": "reasoning",
                "priority": "high",
                "objective": "Rank challengers by walk-forward health, robustness, and sleeve diversity value.",
                "target_functions": ["promotion_autopilot", "training_quality", "multiple_testing_guard"],
            },
            {
                "role_slug": "promotion_master",
                "slug": "model_lifecycle_promotion_master_bot",
                "label": "Model Lifecycle Promotion Master",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "governance",
                "priority": "critical",
                "objective": "Own promotion readiness aggregation and hand clean recommendations to the grand master.",
                "target_functions": ["sleeve_master", "promotion_autopilot", "training_registry_audit"],
            },
            {
                "role_slug": "leakage_overfit_regression_guard",
                "slug": "model_lifecycle_leakage_overfit_regression_guard_bot",
                "label": "Model Lifecycle Leakage Overfit Regression Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "critic",
                "priority": "critical",
                "objective": "Block models that pass by leakage, overfit, duplicate alpha, or insufficient walk-forward evidence.",
                "target_functions": ["regression_guard", "multiple_testing_guard", "training_quality"],
            },
            {
                "role_slug": "retrain_capacity_guard",
                "slug": "model_lifecycle_retrain_capacity_guard_bot",
                "label": "Model Lifecycle Retrain Capacity Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "capacity",
                "priority": "critical",
                "objective": "Schedule retrain and validation workloads around live loop, swap, and storage pressure.",
                "target_functions": ["runtime_training_snapshot", "memory_efficiency", "runtime_throttle"],
            },
        ],
    },
    {
        "slug": "reporting_layer",
        "display_name": "Reporting Layer",
        "sleeve_family": "reporting",
        "sleeve_profile": "reporting_layer",
        "objective": "Make reports command-validated, readable, freshness-aware, and presentation-ready.",
        "preferred_regimes": ["daily_review", "external_review", "program_head_packet", "low_pressure"],
        "correlation_peer_sleeves": ["model_lifecycle", "research_automation", "portfolio_risk_layer"],
        "correlation_dependencies": ["report_quality_guard", "commands_hygiene", "system_summary"],
        "data_intakes": [
            "report_surface_freshness_contract",
            "paper_performance_graph_contract",
            "system_overview_document_contract",
            "strategy_inventory_pdf_contract",
            "commands_report_validation",
        ],
        "storage_targets": [
            "exports/reports",
            "exports/sql_reports",
            "governance/health/reporting_layer",
            "governance/report_quality",
        ],
        "required_labels": ["report_type", "openable_artifact_flag", "freshness_bucket", "presentation_ready_grade"],
        "proxy_data_sources": ["commands_md", "report_quality_guard", "artifact_freshness_slo", "export_manifest"],
        "schwab_direct_inputs": [],
        "retention_profile": "reports_hot_30d_warm_180d_cold_365d",
        "freshness_slo_seconds": 1800,
        "bots": [
            {
                "role_slug": "metric_contract_collector",
                "slug": "reporting_layer_metric_contract_collector_bot",
                "label": "Reporting Layer Metric Contract Collector",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "perception",
                "priority": "high",
                "objective": "Collect report freshness, file-open, metric-contract, and graph-readability checks.",
                "target_functions": ["report_quality_guard", "commands_hygiene", "artifact_freshness_slo"],
            },
            {
                "role_slug": "professional_narrative_builder",
                "slug": "reporting_layer_professional_narrative_builder_bot",
                "label": "Reporting Layer Professional Narrative Builder",
                "bot_role": "signal_sub_bot",
                "intelligence_layer": "reasoning",
                "priority": "high",
                "objective": "Turn platform evidence into clean, external-ready report narratives.",
                "target_functions": ["system_summary", "executive_summary", "reporter_quality"],
            },
            {
                "role_slug": "report_suite_master",
                "slug": "reporting_layer_report_suite_master_bot",
                "label": "Reporting Layer Report Suite Master",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "governance",
                "priority": "critical",
                "objective": "Own report suite readiness and keep command/report inventory aligned.",
                "target_functions": ["sleeve_master", "commands_hygiene", "report_quality_guard"],
            },
            {
                "role_slug": "pdf_open_regression_guard",
                "slug": "reporting_layer_pdf_open_regression_guard_bot",
                "label": "Reporting Layer PDF Open Regression Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "critic",
                "priority": "critical",
                "objective": "Catch broken, unreadable, stale, or non-opening PDF/report artifacts.",
                "target_functions": ["regression_guard", "chrome_pdf_guard", "report_quality_guard"],
            },
            {
                "role_slug": "freshness_capacity_guard",
                "slug": "reporting_layer_freshness_capacity_guard_bot",
                "label": "Reporting Layer Freshness Capacity Guard",
                "bot_role": "infrastructure_sub_bot",
                "intelligence_layer": "capacity",
                "priority": "high",
                "objective": "Refresh reports within budget and avoid headless-browser or PDF render pileups.",
                "target_functions": ["runtime_throttle", "chrome_headless_guard", "stale_sweeper"],
            },
        ],
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
    if not match:
        return None
    return int(match.group("version"))


def _next_available_version(used_versions: set[int], start: int) -> int:
    version = start
    while version in used_versions:
        version += 1
    used_versions.add(version)
    return version


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


def iter_bot_templates() -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for pack in PACKS:
        for bot in pack["bots"]:
            template = copy.deepcopy(bot)
            template["pack"] = copy.deepcopy(pack)
            template["slot_kind"] = f"intelligence_pack_{pack['slug']}_{bot['role_slug']}"
            templates.append(template)
    return templates


def _assign_bot_ids(rows: list[dict[str, Any]]) -> dict[str, str]:
    used_ids = {str(row.get("bot_id") or "") for row in rows if isinstance(row, dict)}
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
    for index, template in enumerate(iter_bot_templates()):
        slot_kind = str(template["slot_kind"])
        if slot_kind in existing_by_slot_kind:
            assigned[slot_kind] = existing_by_slot_kind[slot_kind]
            continue
        desired_version = BASE_VERSION + index
        desired_id = f"brain_refinery_v{desired_version}_{template['slug']}"
        if desired_id not in used_ids and desired_version not in used_versions:
            version = desired_version
            used_versions.add(version)
        else:
            version = _next_available_version(used_versions, max(max(used_versions, default=BASE_VERSION - 1) + 1, desired_version))
        bot_id = f"brain_refinery_v{version}_{template['slug']}"
        used_ids.add(bot_id)
        assigned[slot_kind] = bot_id
    return assigned


def _pack_contract(pack: dict[str, Any], assigned_ids: dict[str, str]) -> dict[str, Any]:
    master_id = assigned_ids.get(f"intelligence_pack_{pack['slug']}_{_master_role_slug(pack)}", "")
    guard_id = assigned_ids.get(f"intelligence_pack_{pack['slug']}_{_regression_guard_role_slug(pack)}", "")
    return {
        "contract_version": CAPABILITY_PACK_VERSION,
        "new_sleeve_or_subsleeve": {
            "sleeve_family": pack["sleeve_family"],
            "sleeve_profile": pack["sleeve_profile"],
            "display_name": pack["display_name"],
        },
        "bot_pack_size": len(pack["bots"]),
        "bot_pack_size_rule": "5_to_15_bots_max",
        "dedicated_data_intake": list(pack["data_intakes"]),
        "storage_retention_rule": {
            "retention_profile": pack["retention_profile"],
            "storage_targets": list(pack["storage_targets"]),
            "max_daily_mb_per_bot": 90,
            "capture_mode": "sampled",
            "sample_rate": 0.5,
            "dedupe_required": True,
            "stale_deletion_policy": "stage_low_value_after_hot_window_and_purge_under_quota_guard",
        },
        "paper_only_floor": {
            "paper_trade_lock_required": True,
            "live_trading_enabled": False,
            "execution_enabled": False,
            "allocation_enabled": False,
            "graduation_requires_minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "graduation_requires_collection_days": MINIMUM_COLLECTION_DAYS,
        },
        "sleeve_master_bot_id": master_id,
        "regression_guard_bot_id": guard_id,
        "capacity_check": {
            "active_bot_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
            "runtime_stability_mode": "full_force_buffered",
            "runtime_control_refresh_seconds": 240,
            "queue_policy": "buffered_jsonl_batching",
            "compute_guard_mode": "soft_cap",
            "global_halt_mode": "soft_cap_and_backpressure_before_hard_halt",
        },
    }


def _master_role_slug(pack: dict[str, Any]) -> str:
    for bot in pack["bots"]:
        if str(bot.get("intelligence_layer") or "") == "governance":
            return str(bot["role_slug"])
    return str(pack["bots"][0]["role_slug"])


def _regression_guard_role_slug(pack: dict[str, Any]) -> str:
    for bot in pack["bots"]:
        if str(bot.get("intelligence_layer") or "") == "critic":
            return str(bot["role_slug"])
    return str(pack["bots"][-1]["role_slug"])


def _row_for_template(template: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    pack = template["pack"]
    pack_contract = _pack_contract(pack, assigned_ids)
    label_tags = [
        "research_only",
        "collection_only",
        "execution_blocked",
        "paper_only_floor",
        f"sleeve_family:{pack['sleeve_family']}",
        f"sleeve_profile:{pack['sleeve_profile']}",
        f"capability_pack:{pack['slug']}",
        f"intelligence_layer:{template['intelligence_layer']}",
        f"data_floor:{MINIMUM_TRAINING_OBSERVATIONS}",
        "training_after_threshold",
        "global_halt_aware",
    ]
    advanced_contract = dict(ADVANCED_INTELLIGENCE_LAYER_CONTRACT)
    advanced_contract.update(
        {
            "capability_pack": pack["slug"],
            "bot_intelligence_layer": template["intelligence_layer"],
            "reports_to_sleeve_master_bot_id": pack_contract["sleeve_master_bot_id"],
            "critic_guard_bot_id": pack_contract["regression_guard_bot_id"],
        }
    )
    return {
        "bot_id": bot_id,
        "bot_role": template["bot_role"],
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
        "slot_label": template["label"],
        "slot_kind": template["slot_kind"],
        "slot_priority": template["priority"],
        "slot_objective": template["objective"],
        "target_functions": list(template["target_functions"]),
        "preferred_regimes": list(pack["preferred_regimes"]),
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v59_risk_sentinel",
            "brain_refinery_v86_risk_budget_allocator_v2",
        ],
        "data_intake_collections": list(pack["data_intakes"]),
        "storage_targets": list(pack["storage_targets"]),
        "freshness_slo_seconds": int(pack["freshness_slo_seconds"]),
        "retention_profile": pack["retention_profile"],
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "structured_capability_pack_observer_until_minimum_samples",
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
        "training_exclusion_reason": "collecting_observations_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "training_threshold_policy": "eligible_when_minimum_observations_and_days_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "promotion_blocked_until": "minimum_data_collection_threshold_met",
        "promotion_block_reason": "data_collection_only_no_training_yet",
        "data_collection_storage_guarded": True,
        "data_collection_storage_guard_mode": "normal",
        "data_collection_capture_mode": "sampled",
        "data_collection_sample_rate": 0.5,
        "data_collection_max_daily_storage_mb": 90,
        "data_collection_storage_guard_updated_utc": now,
        "storage_pressure_capture_reason": "structured_capability_pack_budgeted_collection",
        "data_collection_compute_guard_mode": "soft_cap",
        "data_collection_resource_guard_reason": "700_bot_full_force_stability_contract",
        "data_collection_max_daily_mb": 90,
        "collected_observation_count": 0,
        "data_collection_last_counted_utc": "",
        "data_collection_observation_rollup_source": "awaiting_first_incremental_rollup",
        "data_collection_threshold_progress": _training_threshold_progress(),
        "data_collection_training_ready": False,
        "sleeve_profile": pack["sleeve_profile"],
        "sleeve_family": pack["sleeve_family"],
        "correlation_peer_sleeves": list(pack["correlation_peer_sleeves"]),
        "correlation_dependencies": list(pack["correlation_dependencies"]),
        "provider_capability_profile": "internal_proxy_and_public_context_guarded",
        "direct_market_data_available": bool(pack["schwab_direct_inputs"]),
        "direct_execution_allowed": False,
        "proxy_data_sources": list(pack["proxy_data_sources"]),
        "schwab_direct_inputs": list(pack["schwab_direct_inputs"]),
        "proxy_only_reason": "direct_execution_blocked_until_collection_training_and_paper_gates_clear",
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "required_labels": list(pack["required_labels"]),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": 0.6,
            "freshness_slo_seconds": int(pack["freshness_slo_seconds"]),
            "regression_guard_bot_id": pack_contract["regression_guard_bot_id"],
        },
        "data_label_contract_version": LABEL_CONTRACT_VERSION,
        "labeling_tags": label_tags,
        "execution_policy_label": "collection_only_paper_locked_no_live_execution",
        "eligible_for_master_vote": False,
        "data_collection_runtime_dependency_profile": "low_pressure_sampled_async",
        "founder_bot_id": "brain_refinery_v1",
        "founder_dna_version": "founder_dna_v1",
        "founder_dna_scope": "brain_refinery_v1_to_full_fleet",
        "founder_dna_source": "registry_inferred_full_fleet_contract",
        "founder_dna_confidence": 0.74,
        "founder_dna_traits": list(FOUNDER_DNA_TRAITS),
        "founder_dna_inheritance_mode": "explicit_contract_metadata",
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "lineage_regression_guard": "fail_if_founder_dna_missing_or_stale",
        "lineage_generation": 2,
        "paper_runtime_stability_mode": "full_force_buffered",
        "paper_execution_queue_policy": "buffered_jsonl_batching",
        "paper_runtime_capacity_floor": PAPER_RUNTIME_CAPACITY_FLOOR,
        "paper_trade_lock_required": True,
        "paper_runtime_control_refresh_seconds": 240,
        "capability_pack_version": CAPABILITY_PACK_VERSION,
        "capability_pack_slug": pack["slug"],
        "capability_pack_display_name": pack["display_name"],
        "capability_pack_contract": pack_contract,
        "advanced_intelligence_layer_contract": advanced_contract,
    }


def plan_registry_expansion(registry: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in registry.get("sub_bots", []) if isinstance(row, dict)]
    existing_slot_kinds = {str(row.get("slot_kind") or "") for row in rows}
    assigned_ids = _assign_bot_ids(rows)
    now = _utc_now()
    planned_rows: list[dict[str, Any]] = []
    skipped_existing: list[str] = []
    for template in iter_bot_templates():
        slot_kind = str(template["slot_kind"])
        if slot_kind in existing_slot_kinds:
            skipped_existing.append(slot_kind)
            continue
        planned_rows.append(_row_for_template(template, assigned_ids[slot_kind], assigned_ids, now))
    return {
        "generated_at_utc": now,
        "capability_pack_version": CAPABILITY_PACK_VERSION,
        "pack_count": len(PACKS),
        "bot_count_per_pack": 5,
        "planned_bot_count": len(planned_rows),
        "skipped_existing_count": len(skipped_existing),
        "planned_rows": planned_rows,
        "skipped_existing_slot_kinds": skipped_existing,
        "advanced_intelligence_layer_contract": ADVANCED_INTELLIGENCE_LAYER_CONTRACT,
        "packs": [_pack_summary(pack, assigned_ids) for pack in PACKS],
    }


def _pack_summary(pack: dict[str, Any], assigned_ids: dict[str, str]) -> dict[str, Any]:
    contract = _pack_contract(pack, assigned_ids)
    return {
        "slug": pack["slug"],
        "display_name": pack["display_name"],
        "sleeve_family": pack["sleeve_family"],
        "sleeve_profile": pack["sleeve_profile"],
        "objective": pack["objective"],
        "bot_count": len(pack["bots"]),
        "bot_ids": [assigned_ids[f"intelligence_pack_{pack['slug']}_{bot['role_slug']}"] for bot in pack["bots"]],
        "dedicated_data_intake": list(pack["data_intakes"]),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "sleeve_master_bot_id": contract["sleeve_master_bot_id"],
        "regression_guard_bot_id": contract["regression_guard_bot_id"],
        "capacity_check": contract["capacity_check"],
        "intelligence_layers": [bot["intelligence_layer"] for bot in pack["bots"]],
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
            "structured_capability_pack_count": len(PACKS),
            "structured_capability_pack_bot_count": len(structured),
            "latest_structured_capability_expansion": CAPABILITY_PACK_VERSION,
        }
    )
    registry["summary"] = summary


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
        "capability_pack_version": CAPABILITY_PACK_VERSION,
        "pack_count": plan["pack_count"],
        "bot_count_per_pack": plan["bot_count_per_pack"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "packs": plan["packs"],
        "advanced_intelligence_layer_contract": plan["advanced_intelligence_layer_contract"],
        "recommended_apply_command": "./scripts/ops/opsctl.sh intelligence-capability-expansion --apply --json",
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
        backup = backup_dir / f"master_bot_registry_before_intelligence_capability_expansion_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
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
    config_payload = {
        "generated_at_utc": _utc_now(),
        "capability_pack_version": CAPABILITY_PACK_VERSION,
        "packs": payload["packs"],
        "advanced_intelligence_layer_contract": payload["advanced_intelligence_layer_contract"],
    }
    _write_json(project_root / "config" / "intelligence_capability_packs_v1.json", config_payload)
    _write_json(project_root / "governance" / "health" / "intelligence_capability_expansion_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Add structured 8-pack intelligence capability expansion bots.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--apply", action="store_true", help="Append missing capability-pack bots to master_bot_registry.json.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = apply_registry(project_root) if args.apply else build_payload(project_root)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "intelligence_capability_expansion "
            f"mode={payload['mode']} packs={payload['pack_count']} "
            f"planned={payload['planned_bot_count']} added={payload.get('added_bot_count', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
