#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_VERSION = 1614
TARGET_PLATFORM_TOTAL_BOTS = 1628
PACK_VERSION = "training_labeling_intelligence_v1"
PACK_SLUG = "training_labeling_intelligence"
PACK_DISPLAY_NAME = "Training And Labeling Intelligence Pack"
SLEEVE_FAMILY = "training_labeling_intelligence"
UNIVERSAL_LABEL_CONTRACT_VERSION = "universal_training_label_contract_v1"
LABEL_MATERIALIZATION_CONTRACT_VERSION = "training_label_materialization_contract_v2"
MINIMUM_TRAINING_OBSERVATIONS = 70000
MINIMUM_COLLECTION_DAYS = 180
SAMPLE_RATE = 0.01
MAX_DAILY_MB_PER_BOT = 1
DEFAULT_COLLECT_ONLY_DIAGNOSTIC_MIN_VERSION = 700
COLLECT_ONLY_USABLE_SAMPLE_GOAL = 200
COLLECT_ONLY_ELIGIBLE_SEQUENCE_GOAL = 4
COLLECT_ONLY_OBSERVATIONS_PER_SAMPLE_TARGET = 5
OPTIONS_CONTEXT_SOURCE_ID = "options_context_mesh"
OPTIONS_CONTEXT_LEGACY_SOURCE_ID = "polygon_unusual_whales_options_context"


INTELLIGENCE_SYSTEMS: list[dict[str, Any]] = [
    {
        "slug": "label_contract_normalizer",
        "layer": "labeling",
        "display_name": "Label Contract Normalizer",
        "objective": "Give every bot a point-in-time label contract without erasing specialized existing contracts.",
        "outputs": ["label_contract_diff", "missing_label_repair_packet", "label_family_map"],
    },
    {
        "slug": "point_in_time_label_guard",
        "layer": "labeling",
        "display_name": "Point-In-Time Label Guard",
        "objective": "Block future leakage, lookahead joins, unbounded raw-feed joins, and unlabeled promotion packets.",
        "outputs": ["join_contract_verdict", "leakage_risk_score", "label_context_gap_list"],
    },
    {
        "slug": "lane_balance_scheduler",
        "layer": "training",
        "display_name": "Lane Balance Scheduler",
        "objective": "Turn lane dominance, symbol concentration, and coverage shortfall into narrow retrain plans.",
        "outputs": ["lane_balanced_retrain_plan", "lookback_guidance", "dominance_cap_vote"],
    },
    {
        "slug": "coverage_repair_orchestrator",
        "layer": "training",
        "display_name": "Coverage Repair Orchestrator",
        "objective": "Prefer coverage repair candidates when the normal targeted retrain shortlist is empty.",
        "outputs": ["coverage_repair_queue", "runtime_input_repair_plan", "walk_forward_cycle_budget"],
    },
    {
        "slug": "schema_lineage_gatekeeper",
        "layer": "lineage",
        "display_name": "Schema Lineage Gatekeeper",
        "objective": "Keep schema, feature-store, replay, experiment, and promotion lineage gates synchronized before retrain.",
        "outputs": ["schema_lineage_gate_status", "missing_contract_repair_order", "promotion_packet_readiness"],
    },
    {
        "slug": "retrain_outcome_memory",
        "layer": "learning",
        "display_name": "Retrain Outcome Memory",
        "objective": "Record which targeted retrains reduced coverage gaps, label errors, gate blockers, and runtime failures.",
        "outputs": ["retrain_effect_delta", "retry_or_rotate_vote", "training_playbook_reward"],
    },
]


ROLE_TEMPLATES: list[dict[str, Any]] = [
    {"suffix": "telemetry_collector", "label": "Telemetry Collector", "bot_role": "infrastructure_sub_bot", "priority": "high"},
    {"suffix": "quality_scorer", "label": "Quality Scorer", "bot_role": "signal_sub_bot", "priority": "high"},
    {"suffix": "policy_guard", "label": "Policy Guard", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
    {"suffix": "training_bridge", "label": "Training Bridge", "bot_role": "infrastructure_sub_bot", "priority": "critical"},
]


BASE_DATA_INTAKES = [
    "training_quality_trace",
    "training_runtime_trace",
    "training_label_audit_trace",
    "runtime_training_snapshot_trace",
    "coverage_gap_closer_trace",
    "feature_store_lineage_trace",
    "promotion_gate_trace",
    "whole_system_governor_trace",
    "codex_handoff_trace",
]

FREE_LABEL_CONTEXT_SOURCE_MAP: dict[str, list[str]] = {
    "price_bars": ["free_equity_reference_context", "market_quote_profiles"],
    "daily_bars": ["free_equity_reference_context", "market_quote_profiles"],
    "total_return_bars": ["free_equity_reference_context", "market_quote_profiles"],
    "volume": ["free_equity_reference_context", "market_quote_profiles"],
    "relative_volume": ["free_equity_reference_context", "market_micro_context"],
    "market_context": ["free_equity_reference_context", "market_quote_profiles", "ticker_news_context"],
    "sector_context": ["free_equity_reference_context", "ticker_news_context", "sec_edgar_context"],
    "overnight_gap": ["free_equity_reference_context", "market_quote_profiles"],
    "one_minute_bars": ["market_micro_context", "market_quote_profiles"],
    "vwap": ["market_micro_context", "market_quote_profiles"],
    "spread_quality": ["market_micro_context", "extended_quant_context"],
    "market_micro_features": ["market_micro_context", "extended_quant_context"],
    "market_micro_context": ["market_micro_context"],
    "liquidity_state": ["market_micro_context", "extended_quant_context", "public_policy_context"],
    "execution_cost_context": ["market_micro_context", "extended_quant_context"],
    "execution_quality": ["market_micro_context", "extended_quant_context"],
    "fill_quality": ["market_micro_context", "extended_quant_context"],
    "fill_realism_context": ["market_micro_context", "extended_quant_context"],
    "latency_trace": ["market_micro_context"],
    "queue_position_context": ["market_micro_context"],
    "slippage_trace": ["market_micro_context", "extended_quant_context"],
    "spread_queue_response": ["market_micro_context", "extended_quant_context"],
    "transaction_cost_surface": ["market_micro_context", "extended_quant_context"],
    "vpin_order_flow_toxicity": ["market_micro_context", "extended_quant_context"],
    "dark_pool_off_exchange_volume": ["market_micro_context"],
    "options_chain": [OPTIONS_CONTEXT_SOURCE_ID],
    "iv_surface": [OPTIONS_CONTEXT_SOURCE_ID, "extended_quant_context"],
    "listed_option_surface": [OPTIONS_CONTEXT_SOURCE_ID, "extended_quant_context"],
    "open_interest": [OPTIONS_CONTEXT_SOURCE_ID, "extended_quant_context"],
    "bid_ask_spread": [OPTIONS_CONTEXT_SOURCE_ID, "market_micro_context"],
    "greeks": [OPTIONS_CONTEXT_SOURCE_ID],
    "skew": [OPTIONS_CONTEXT_SOURCE_ID, "extended_quant_context"],
    "realized_vol": ["free_equity_reference_context", "extended_quant_context"],
    "realized_volatility": ["free_equity_reference_context", "extended_quant_context"],
    "vix_term_structure": ["extended_quant_context"],
    "crypto_bars": ["crypto_market_context"],
    "order_book_proxy": ["crypto_market_context"],
    "funding_context": ["crypto_market_context", "fx_market_context"],
    "basis": ["crypto_market_context", "fx_market_context", "extended_quant_context"],
    "basis_context": ["crypto_market_context", "fx_market_context", "extended_quant_context"],
    "cross_asset_correlation": ["crypto_market_context", "fx_market_context", "extended_quant_context"],
    "rates_context": ["fx_market_context", "official_macro_context", "public_policy_context"],
    "rate_context": ["fx_market_context", "official_macro_context", "public_policy_context"],
    "rates_curve": ["fx_market_context", "official_macro_context", "extended_quant_context"],
    "duration_context": ["official_macro_context", "extended_quant_context"],
    "inflation_context": ["official_macro_context", "public_macro_feeds"],
    "macro_context": ["macro_crossstack", "official_macro_context", "public_macro_feeds", "public_policy_context"],
    "macro_calendar": ["official_macro_context", "public_macro_feeds"],
    "macro_event_window": ["official_macro_context", "public_macro_feeds", "ticker_news_context"],
    "market_breadth": ["free_equity_reference_context", "market_quote_profiles", "market_micro_context"],
    "credit_spread_context": ["extended_quant_context", "public_policy_context"],
    "credit_stress": ["extended_quant_context", "public_policy_context"],
    "news_source_consensus": ["ticker_news_context", "schwab_symbol_news"],
    "sec_filing_context": ["sec_edgar_context"],
    "earnings_calendar": ["ticker_news_context", "sec_edgar_context"],
    "ex_dividend_calendar": ["free_equity_reference_context", "sec_edgar_context"],
    "payout_metrics": ["sec_edgar_context", "free_equity_reference_context"],
    "balance_sheet_quality": ["sec_edgar_context"],
    "source_scores": ["source_verification", "collector_contracts"],
    "source_confidence": ["source_verification", "collector_contracts"],
    "source_quality": ["source_verification", "collector_contracts"],
    "correlation_matrix": ["free_equity_reference_context", "crypto_market_context", "fx_market_context", "extended_quant_context"],
    "session_calendar": ["market_quote_profiles", "official_macro_context", "public_macro_feeds"],
    "runtime_health": ["source_verification", "collector_contracts"],
    "incident_log": ["source_verification"],
    # Verified proxy routes for contexts whose raw authorities are live-tape,
    # broker, or research-specific. These are label evidence only.
    "feed_latency_schema_health": ["market_micro_context", "source_verification", "collector_contracts"],
    "futures_bars": ["market_quote_profiles", "extended_quant_context", "official_macro_context"],
    "mbo_mbp_depth_snapshot": ["market_micro_context", "extended_quant_context"],
    "model_price_sensitivity_grid": ["extended_quant_context", "source_verification"],
    "opra_nbbo_taq_sip_normalized_events": [OPTIONS_CONTEXT_SOURCE_ID, "market_micro_context", "extended_quant_context"],
    "quant_model_feature_surface": ["extended_quant_context", "source_verification"],
    "state_filter_diagnostics": ["source_verification", "collector_contracts", "market_micro_context"],
    "auction_imbalance_context": ["market_micro_context", "market_quote_profiles"],
    "cpi_pce_nfp_event_window": ["official_macro_context", "public_macro_feeds"],
    "earnings_cluster_context": ["ticker_news_context", "sec_edgar_context"],
    "fed_speaker_calendar_surprise": ["official_macro_context", "public_macro_feeds"],
    "halt_reopen_liquidity_context": ["market_micro_context", "market_quote_profiles"],
    "liquidity_cliff_context": ["market_micro_context", "extended_quant_context"],
    "macro_event_bulletins": ["official_macro_context", "public_macro_feeds", "ticker_news_context"],
    "market_microstructure_liquidity_proxy": ["market_micro_context", "extended_quant_context"],
    "order_flow_imbalance_context": ["market_micro_context", "extended_quant_context"],
    "quote_fade_context": ["market_micro_context", "extended_quant_context"],
    "rate_volatility_context": ["official_macro_context", "extended_quant_context"],
    "treasury_auction_context": ["official_macro_context", "public_macro_feeds"],
}

LABEL_CONTEXT_CLASSIFICATION_MAP: dict[str, dict[str, str]] = {
    "codex_handoff_trace": {"class": "internal_trace", "route": "codex_operator_bridge", "authority": "internal_governance"},
    "constraint_violation_trace": {"class": "internal_trace", "route": "optimization_search_trace", "authority": "internal_research"},
    "coverage_gap_closer_trace": {"class": "internal_trace", "route": "coverage_gap_closer", "authority": "internal_governance"},
    "coverage_gap_trace": {"class": "internal_trace", "route": "training_label_audit", "authority": "internal_governance"},
    "coverage_repair_orchestrator_effect_trace": {"class": "internal_trace", "route": "coverage_repair_orchestrator", "authority": "internal_governance"},
    "dark_pool_off_exchange_volume": {"class": "public_proxy_available", "route": "market_micro_context", "authority": "free_public_proxy"},
    "feature_store_lineage": {"class": "internal_trace", "route": "feature_store_lineage_trace", "authority": "internal_lineage"},
    "feature_store_lineage_trace": {"class": "internal_trace", "route": "feature_store_lineage", "authority": "internal_lineage"},
    "feed_latency_schema_health": {"class": "broker_or_live_tape_required", "route": "market_data_tape_normalization", "authority": "live_tape_or_adapter"},
    "futures_bars": {"class": "broker_or_live_tape_required", "route": "broker_market_data", "authority": "broker_or_exchange"},
    "kelly_fraction_trace": {"class": "internal_trace", "route": "optimization_search_trace", "authority": "internal_research"},
    "label_contract_normalizer_effect_trace": {"class": "internal_trace", "route": "label_contract_normalizer", "authority": "internal_governance"},
    "lane_balance_scheduler_effect_trace": {"class": "internal_trace", "route": "lane_balance_scheduler", "authority": "internal_governance"},
    "mbo_mbp_depth_snapshot": {"class": "broker_or_live_tape_required", "route": "depth_snapshot_collector", "authority": "live_tape_or_adapter"},
    "model_price_sensitivity_grid": {"class": "research_only", "route": "quant_model_feature_surface", "authority": "internal_research"},
    "objective_value_trace": {"class": "internal_trace", "route": "optimization_search_trace", "authority": "internal_research"},
    "operator_context": {"class": "internal_trace", "route": "operator_cockpit", "authority": "internal_governance"},
    "opra_nbbo_taq_sip_normalized_events": {"class": "broker_or_live_tape_required", "route": "market_data_tape_normalization", "authority": "opra_sip_tape"},
    "optimization_search_trace": {"class": "internal_trace", "route": "optimization_search", "authority": "internal_research"},
    "point_in_time_label_guard_effect_trace": {"class": "internal_trace", "route": "point_in_time_label_guard", "authority": "internal_governance"},
    "portfolio_exposure": {"class": "broker_truth_required", "route": "broker_truth_reconcile_v2", "authority": "broker_truth"},
    "promotion_gate_trace": {"class": "internal_trace", "route": "promotion_gate", "authority": "internal_governance"},
    "proxy_data_source_lineage": {"class": "internal_trace", "route": "source_verification", "authority": "internal_governance"},
    "quant_model_feature_surface": {"class": "research_only", "route": "quant_model_feature_surface", "authority": "internal_research"},
    "regime_transition_trace": {"class": "internal_trace", "route": "state_space_filter_diagnostics", "authority": "internal_research"},
    "retrain_outcome_memory_effect_trace": {"class": "internal_trace", "route": "retrain_outcome_memory", "authority": "internal_governance"},
    "risk_budget": {"class": "broker_truth_required", "route": "capital_rotation_control", "authority": "broker_truth"},
    "runtime_feature_history": {"class": "internal_trace", "route": "runtime_training_snapshot_trace", "authority": "internal_governance"},
    "runtime_snapshot_trace": {"class": "internal_trace", "route": "runtime_training_snapshot", "authority": "internal_governance"},
    "runtime_training_snapshot_trace": {"class": "internal_trace", "route": "runtime_training_snapshot", "authority": "internal_governance"},
    "schema_lineage_gatekeeper_effect_trace": {"class": "internal_trace", "route": "schema_lineage_gatekeeper", "authority": "internal_lineage"},
    "state_filter_diagnostics": {"class": "research_only", "route": "state_space_filter_diagnostics", "authority": "internal_research"},
    "training_label_audit_trace": {"class": "internal_trace", "route": "training_label_audit", "authority": "internal_governance"},
    "training_quality_trace": {"class": "internal_trace", "route": "training_quality_control", "authority": "internal_governance"},
    "training_runtime_trace": {"class": "internal_trace", "route": "training_runtime_control", "authority": "internal_governance"},
    "walk_forward_trace": {"class": "internal_trace", "route": "walk_forward_validate", "authority": "internal_training"},
    "whole_system_governor_trace": {"class": "internal_trace", "route": "whole_system_governor", "authority": "internal_governance"},
}


REQUIRED_LABELS = [
    "forward_return_bucket",
    "risk_adjusted_return_bucket",
    "action_effect_bucket",
    "label_quality_bucket",
    "lane_balance_bucket",
    "coverage_gap_status",
    "lineage_gate_status",
    "promotion_gate_status",
]

OPERATIONAL_REQUIRED_LABELS = [
    "action_effect_bucket",
    "incident_prevention_outcome",
    "false_positive_guard_outcome",
    "runtime_health_delta_bucket",
    "label_quality_bucket",
    "lineage_gate_status",
    "promotion_gate_status",
]

RESEARCH_REQUIRED_LABELS = [
    "walk_forward_effect_bucket",
    "out_of_sample_error_bucket",
    "stability_delta_bucket",
    "proxy_label_quality_bucket",
    "overfit_gap_status",
    "lineage_gate_status",
    "promotion_gate_status",
]

MARKET_OUTCOME_LABEL_FAMILIES = {
    "generic_directional",
    "intraday_fast",
    "multi_day",
    "same_session",
    "risk_adjusted_preservation",
    "income_total_return",
    "income_options_surface",
    "options_surface",
    "futures_event_session",
    "crypto_microstructure",
    "credit_spread",
    "fixed_income_rates",
    "correlation_risk_effect",
    "execution_cost_quality",
    "spread_convergence",
    "sector_rotation_master",
    "volatility_regime",
    "position_management",
}

OPERATIONAL_LABEL_FAMILIES = {
    "operational_guard_effect",
    "training_process_quality",
    "infrastructure_guard",
    "provider_adapter_verification_research",
    "institutional_data_plumbing_research",
    "low_latency_agent_orchestration",
    "privacy_zkp_controls",
    "adversarial_ml_security",
}

RESEARCH_VALIDATION_LABEL_FAMILIES = {
    "crowd_physics_games",
    "gpu_quant_acceleration",
    "limit_order_book_transformers",
    "neural_sde_kan_hedging",
    "quant_research_control",
    "signature_hawkes_generators",
}

_MARKET_SIGNAL_IDENTITY_TOKENS = (
    "alpha_",
    "bank",
    "bond",
    "breadth",
    "conservative",
    "credit",
    "crypto",
    "day_trading",
    "dividend",
    "earnings",
    "energy",
    "equity",
    "factor",
    "fed_",
    "fx_",
    "halt_reopen",
    "intraday",
    "liquidity",
    "market_neutral",
    "mega_cap",
    "momentum",
    "options",
    "rate",
    "reit",
    "russell",
    "small_cap",
    "spread",
    "swing",
    "volatility",
)

_CONTROL_PLANE_IDENTITY_TOKENS = (
    "autonomic_governance",
    "backlog",
    "bot_genome",
    "data_lineage",
    "frontier_",
    "institutional_operator",
    "memory_lymphatic",
    "operator_copilot",
    "platform_organ",
    "quant_operational",
    "storage_memory",
    "system_governor",
)

ADVANCED_QUANT_LABEL_FAMILIES = {
    "quant_pricing_research",
    "state_space_filter_research",
    "optimization_research",
    "transaction_cost_slippage_research",
    "cross_asset_basis_research",
    "market_data_tape_normalization_research",
    "order_flow_toxicity_research",
}

TARGETED_LABEL_CONTRACT_OVERRIDES: dict[str, dict[str, Any]] = {
    "brain_refinery_v31_defensive_rotation": {
        "label_family": "options_surface",
        "primary_horizon": "defensive_rotation_iv_realized_1d_5d",
        "aux_horizons": ["downside_capture_1d_5d", "skew_shift", "spread_quality", "event_vol_reset"],
        "required_context": ["options_chain", "iv_surface", "open_interest", "bid_ask_spread", "greeks", "skew"],
    },
    "brain_refinery_v99_defensive_dividend_concentration": {
        "label_family": "income_options_surface",
        "primary_horizon": "dividend_concentration_risk_adjusted_5d_20d",
        "aux_horizons": ["ex_dividend_window", "payout_safety", "skew_shift", "event_vol_reset"],
        "required_context": ["ex_dividend_calendar", "payout_metrics", "options_chain", "iv_surface", "open_interest", "bid_ask_spread"],
    },
    "brain_refinery_v96_credit_spread_rotation_bot": {
        "label_family": "credit_spread",
        "primary_horizon": "credit_spread_rotation_5d_20d",
        "aux_horizons": ["spread_widening_risk", "duration_adjusted_return", "credit_beta", "drawdown_avoidance_5d"],
        "required_context": ["credit_spread_context", "rates_curve", "sector_context", "market_breadth", "realized_volatility", "liquidity_state"],
    },
    "brain_refinery_v95_rates_regime_bond_bot": {
        "label_family": "fixed_income_rates",
        "primary_horizon": "rates_regime_total_return_5d_20d",
        "aux_horizons": ["yield_curve_shift", "duration_beta", "inflation_surprise", "credit_stress"],
        "required_context": ["rates_curve", "duration_context", "inflation_context", "macro_calendar", "credit_stress", "liquidity_state"],
    },
    "brain_refinery_v94_dividend_yield_trap_avoidance": {
        "label_family": "income_total_return",
        "primary_horizon": "dividend_yield_trap_avoidance_20d_total_return",
        "aux_horizons": ["payout_safety", "dividend_cut_risk", "earnings_quality", "drawdown_avoidance_5d"],
        "required_context": ["ex_dividend_calendar", "payout_metrics", "balance_sheet_quality", "earnings_calendar", "rate_context", "sector_context"],
    },
    "brain_refinery_v93_dividend_quality_compounder": {
        "label_family": "income_total_return",
        "primary_horizon": "dividend_quality_compounder_20d_total_return",
        "aux_horizons": ["payout_safety", "dividend_growth", "earnings_quality", "risk_adjusted_return"],
        "required_context": ["ex_dividend_calendar", "payout_metrics", "balance_sheet_quality", "earnings_calendar", "rate_context", "sector_context"],
    },
    "brain_refinery_v69_cost_aware_execution_filter": {
        "label_family": "execution_cost_quality",
        "primary_horizon": "execution_cost_avoidance_after_slippage",
        "aux_horizons": ["spread_cost_delta", "fill_quality", "latency_penalty", "market_impact_proxy"],
        "required_context": ["spread_quality", "execution_quality", "slippage_trace", "fill_quality", "latency_trace", "market_micro_context"],
    },
    "brain_refinery_v67_correlation_penalty_layer": {
        "label_family": "correlation_risk_effect",
        "primary_horizon": "correlation_penalty_risk_adjusted_5d",
        "aux_horizons": ["correlation_cluster_stability", "exposure_netting_delta", "drawdown_avoidance_5d", "diversification_score"],
        "required_context": ["correlation_matrix", "cross_asset_correlation", "sector_context", "portfolio_exposure", "risk_budget", "market_breadth"],
    },
}


STORAGE_TARGETS = [
    "governance/training_labeling_intelligence",
    *[f"governance/training_labeling_intelligence/{system['slug']}" for system in INTELLIGENCE_SYSTEMS],
    "governance/training_labeling_intelligence/all_bot_label_materialization_latest.json",
    "governance/health/training_labeling_intelligence_latest.json",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_first_json(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        payload = _load_json(path)
        if payload:
            return payload
    return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _ordered_unique(items: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _free_source_candidates_for_contexts(contexts: list[Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for raw in contexts:
        context = str(raw or "").strip()
        if not context:
            continue
        mapped = FREE_LABEL_CONTEXT_SOURCE_MAP.get(context)
        if not mapped:
            continue
        out[context] = list(mapped)
    return out


_RESEARCH_CONTEXT_TOKENS = (
    "backtest",
    "causal",
    "embedding",
    "geometric",
    "geometry",
    "greeks",
    "hawkes",
    "heston",
    "laplacian",
    "malliavin",
    "manifold",
    "mckean_vlasov",
    "mean_field",
    "neural_sde",
    "quantum",
    "rough_path",
    "rough_volatility",
    "simulation",
    "stochastic_differential",
    "tatonnement",
    "topological",
    "wasserstein",
    "zkp",
)

_INTERNAL_CONTEXT_SUFFIXES = (
    "_audit",
    "_contracts",
    "_coverage",
    "_detection",
    "_guard",
    "_health",
    "_memory",
    "_pressure",
    "_profile",
    "_queue",
    "_rank",
    "_requalification",
    "_score",
    "_state",
    "_trace",
    "_validation",
)

_INTERNAL_CONTEXT_NAMES = {
    "collector_contracts",
    "data_source_divergence",
    "golden_replay_regression",
    "model_drift_guard",
    "process_watchdog",
    "source_verification",
    "stale_feature_detection",
    "storage_backpressure",
    "walk_forward_requalification",
}


def _context_classification(context: str, candidate_ids: list[str]) -> dict[str, str]:
    context_id = str(context or "").strip()
    mapped = LABEL_CONTEXT_CLASSIFICATION_MAP.get(context_id)
    if mapped:
        return dict(mapped)
    if candidate_ids:
        return {"class": "free_public_or_verified_proxy", "route": "source_verification", "authority": "free_public_or_verified_proxy"}
    if any(token in context_id for token in _RESEARCH_CONTEXT_TOKENS):
        return {"class": "research_only", "route": context_id, "authority": "internal_research_snapshot"}
    if context_id in _INTERNAL_CONTEXT_NAMES or context_id.endswith(_INTERNAL_CONTEXT_SUFFIXES):
        return {"class": "internal_trace", "route": context_id, "authority": "internal_event_store"}
    return {"class": "unclassified", "route": "manual_context_triage", "authority": "unknown"}


def _context_weight_multiplier(coverage_status: str, context_class: str, confidence: float) -> float:
    if coverage_status == "verified":
        return round(max(0.65, min(1.0, 0.70 + confidence * 0.30)), 6)
    if context_class in {"internal_trace", "broker_truth_required"}:
        return 0.78
    if context_class == "public_proxy_available":
        return 0.70
    if context_class == "broker_or_live_tape_required":
        return 0.52
    if context_class == "research_only":
        return 0.48
    return 0.40


def _label_materialization_contract(context: str, classification: dict[str, str], coverage_status: str) -> dict[str, Any]:
    context_class = str(classification.get("class") or "unclassified")
    route = str(classification.get("route") or "")
    join_mode = "point_in_time_only"
    if context_class == "broker_or_live_tape_required":
        join_mode = "broker_or_tape_timestamp_only"
    elif context_class == "research_only":
        join_mode = "research_snapshot_id_only"
    elif context_class == "internal_trace":
        join_mode = "internal_event_timestamp_only"
    evidence_verified = bool(coverage_status == "verified")
    return {
        "context": str(context),
        "context_class": context_class,
        "materialization_route": route,
        "required_join_mode": join_mode,
        "required_join_keys": ["bot_id", "symbol", "timestamp_utc", "snapshot_id"],
        "required_outputs": [
            "sample_eligibility_reason",
            "side_specific_outcome",
            "abstention_outcome",
            "counterfactual_opportunity_trace",
            "label_source_confidence_norm",
        ],
        "evidence_verification_status": "verified" if evidence_verified else "pending_source_or_artifact_verification",
        "eligible_for_training": evidence_verified,
        "policy": "materialize_labels_before_training_use; never join future context into historical samples",
    }


def _source_status_aliases(source_id: str, row: dict[str, Any]) -> list[str]:
    aliases = [str(item or "").strip() for item in row.get("aliases") or [] if str(item or "").strip()]
    if source_id == OPTIONS_CONTEXT_SOURCE_ID:
        aliases.append(OPTIONS_CONTEXT_LEGACY_SOURCE_ID)
    elif source_id == OPTIONS_CONTEXT_LEGACY_SOURCE_ID:
        aliases.append(OPTIONS_CONTEXT_SOURCE_ID)
    return _ordered_unique(aliases)


def _source_verification_statuses(project_root: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(project_root / "governance" / "health" / "source_verification_latest.json")
    rows = payload.get("sources") if isinstance(payload.get("sources"), list) else []
    statuses: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id") or "").strip()
        if not source_id:
            continue
        verified = str(row.get("verification_status") or "").strip() != "single_source_unverified"
        status_row = {
            "source_id": source_id,
            "title": str(row.get("title") or source_id),
            "category": str(row.get("category") or ""),
            "verification_status": str(row.get("verification_status") or ""),
            "ok": bool(row.get("ok", False)),
            "fresh": bool(row.get("fresh", False)),
            "verified": bool(verified and row.get("ok", False) and row.get("fresh", False)),
            "source_confidence_score": _safe_float(row.get("source_confidence_score"), 1.0 if verified and row.get("ok", False) and row.get("fresh", False) else 0.0),
            "confidence_components": row.get("confidence_components") if isinstance(row.get("confidence_components"), dict) else {},
            "evidence": row.get("evidence") if isinstance(row.get("evidence"), dict) else {},
            "aliases": _source_status_aliases(source_id, row),
        }
        statuses[source_id] = status_row
        for alias in status_row["aliases"]:
            alias_row = dict(status_row)
            alias_row["source_id"] = alias
            alias_row["canonical_source_id"] = source_id
            statuses.setdefault(alias, alias_row)
    if bool(payload.get("ok", False)):
        statuses.setdefault(
            "source_verification",
            {
                "source_id": "source_verification",
                "title": "Source Verification",
                "category": "governance",
                "verification_status": str(payload.get("overall_status") or "ready"),
                "ok": True,
                "fresh": True,
                "verified": True,
            },
        )
    collector_contracts = _load_json(project_root / "governance" / "health" / "collector_contracts_latest.json")
    if collector_contracts:
        statuses.setdefault(
            "collector_contracts",
            {
                "source_id": "collector_contracts",
                "title": "Collector Contracts",
                "category": "governance",
                "verification_status": str(collector_contracts.get("overall_status") or collector_contracts.get("status") or ""),
                "ok": bool(collector_contracts.get("ok", True)),
                "fresh": True,
                "verified": bool(collector_contracts.get("ok", True)),
            },
        )
    return statuses


def _free_label_source_enrichment(project_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_statuses = _source_verification_statuses(project_root)
    required_context_counts: Counter[str] = Counter()
    label_family_contexts: dict[str, set[str]] = {}
    for row in rows:
        contract = _universal_contract(row)
        label_family = str(contract.get("label_family") or "unknown")
        contexts = [str(item or "").strip() for item in contract.get("required_context") or [] if str(item or "").strip()]
        for context in contexts:
            required_context_counts[context] += 1
            label_family_contexts.setdefault(label_family, set()).add(context)

    context_rows: list[dict[str, Any]] = []
    for context, required_count in sorted(required_context_counts.items()):
        candidate_ids = list(FREE_LABEL_CONTEXT_SOURCE_MAP.get(context) or [])
        verified_ids = [
            source_id
            for source_id in candidate_ids
            if bool(source_statuses.get(source_id, {}).get("verified", False))
        ]
        confidence_scores = [
            _safe_float(source_statuses.get(source_id, {}).get("source_confidence_score"), 0.0)
            for source_id in candidate_ids
        ]
        verified_confidence_scores = [
            _safe_float(source_statuses.get(source_id, {}).get("source_confidence_score"), 0.0)
            for source_id in verified_ids
        ]
        source_confidence = max(verified_confidence_scores or confidence_scores or [0.0])
        coverage_status = "verified" if verified_ids else "unmapped" if not candidate_ids else "unverified"
        classification = _context_classification(context, candidate_ids)
        context_class = str(classification.get("class") or "unclassified")
        context_rows.append(
            {
                "context": context,
                "required_by_bot_count": int(required_count),
                "candidate_source_ids": candidate_ids,
                "verified_source_ids": verified_ids,
                "coverage_status": coverage_status,
                "context_class": context_class,
                "authority": str(classification.get("authority") or ""),
                "materialization_route": str(classification.get("route") or ""),
                "source_confidence_norm": round(max(0.0, min(float(source_confidence), 1.0)), 6),
                "label_weight_multiplier": _context_weight_multiplier(coverage_status, context_class, float(source_confidence)),
                "materialization_contract": _label_materialization_contract(context, classification, coverage_status),
            }
        )

    verified_contexts = [row["context"] for row in context_rows if row["coverage_status"] == "verified"]
    unmapped_contexts = [row["context"] for row in context_rows if row["coverage_status"] == "unmapped"]
    unverified_contexts = [row["context"] for row in context_rows if row["coverage_status"] == "unverified"]
    classification_counts = dict(sorted(Counter(str(row.get("context_class") or "unclassified") for row in context_rows).items()))
    low_confidence_contexts = [
        str(row["context"])
        for row in context_rows
        if _safe_float(row.get("label_weight_multiplier"), 1.0) < 0.60
    ]
    materialization_ready_contexts = [
        str(row["context"])
        for row in context_rows
        if bool((row.get("materialization_contract") or {}).get("eligible_for_training", False))
    ]
    return {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "policy": "free_public_sources_are_label_context_evidence_only_not_execution_authority",
        "source_status_count": len(source_statuses),
        "free_context_mapping_count": len(FREE_LABEL_CONTEXT_SOURCE_MAP),
        "required_context_count": len(context_rows),
        "verified_context_count": len(verified_contexts),
        "unverified_context_count": len(unverified_contexts),
        "unmapped_context_count": len(unmapped_contexts),
        "classification_counts": classification_counts,
        "low_confidence_context_count": len(low_confidence_contexts),
        "low_confidence_contexts": low_confidence_contexts[:250],
        "materialization_ready_context_count": len(materialization_ready_contexts),
        "materialization_ready_contexts": materialization_ready_contexts[:250],
        "verified_contexts": verified_contexts[:250],
        "unverified_contexts": unverified_contexts[:250],
        "unmapped_contexts": unmapped_contexts[:250],
        "context_sources": context_rows,
        "label_family_context_sources": {
            family: _free_source_candidates_for_contexts(sorted(contexts))
            for family, contexts in sorted(label_family_contexts.items())
        },
        "source_statuses": source_statuses,
    }


def _label_materialization_plan(source_enrichment: dict[str, Any]) -> dict[str, Any]:
    context_rows = source_enrichment.get("context_sources") if isinstance(source_enrichment.get("context_sources"), list) else []
    contracts = [
        row.get("materialization_contract")
        for row in context_rows
        if isinstance(row, dict) and isinstance(row.get("materialization_contract"), dict)
    ]
    ready = [row for row in contracts if bool(row.get("eligible_for_training", False))]
    blocked = [row for row in contracts if not bool(row.get("eligible_for_training", False))]
    by_class = Counter(str(row.get("context_class") or "unclassified") for row in contracts)
    return {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "overall_status": "ready" if not blocked else "needs_materialization",
        "contract_count": len(contracts),
        "ready_contract_count": len(ready),
        "blocked_contract_count": len(blocked),
        "contract_counts_by_class": dict(sorted(by_class.items())),
        "required_outputs": [
            "sample_eligibility_reason",
            "side_specific_outcome",
            "abstention_outcome",
            "counterfactual_opportunity_trace",
            "label_source_confidence_norm",
        ],
        "ready_contexts": [str(row.get("context") or "") for row in ready[:250]],
        "blocked_contexts": [str(row.get("context") or "") for row in blocked[:250]],
        "materialization_queue": contracts[:500],
        "policy": "materialization_plan_is_required_before_any_collect_only_bot_graduates_to_training",
    }


def _registry_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else registry.get("bots")
    return [row for row in rows or [] if isinstance(row, dict)]


def _version_from_bot_id(bot_id: str) -> int | None:
    match = re.match(r"^brain_refinery_v(?P<version>\d+)", bot_id)
    return int(match.group("version")) if match else None


def _next_available_version(used_versions: set[int], start: int) -> int:
    version = start
    while version in used_versions:
        version += 1
    used_versions.add(version)
    return version


def _bot_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for system in INTELLIGENCE_SYSTEMS:
        for role in ROLE_TEMPLATES:
            role_slug = f"{system['slug']}_{role['suffix']}"
            specs.append(
                {
                    "role_slug": role_slug,
                    "slug": f"training_labeling_{role_slug}_bot",
                    "label": f"{system['display_name']} {role['label']}",
                    "system": system["slug"],
                    "layer": system["layer"],
                    "bot_role": role["bot_role"],
                    "priority": role["priority"],
                    "objective": f"{role['label']} for {system['objective']}",
                    "target_functions": list(system["outputs"]),
                }
            )
    return specs


BOTS = _bot_specs()


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
    return {"slug": bot["system"], "layer": bot.get("layer", ""), "display_name": bot["system"], "outputs": []}


def _existing_contract(row: dict[str, Any]) -> dict[str, Any]:
    contract = row.get("label_contract") or row.get("training_label_contract")
    return contract if isinstance(contract, dict) else {}


def _contract_complete(row: dict[str, Any]) -> bool:
    contract = _existing_contract(row)
    primary = str(contract.get("primary_horizon") or contract.get("primary_label_horizon") or "").strip()
    required_context = contract.get("required_context") or contract.get("required_label_context")
    required_labels = contract.get("required_labels")
    return bool(primary and (isinstance(required_context, list) or isinstance(required_labels, list) or row.get("data_label_contract_version")))


def _market_signal_label_override(row: dict[str, Any]) -> tuple[str, str, list[str], list[str]] | None:
    if str(row.get("bot_role") or "").strip().lower() != "signal_sub_bot":
        return None
    text = " ".join(
        str(row.get(key) or "").strip().lower()
        for key in ("bot_id", "slot_kind", "slot_label", "sleeve_profile", "sleeve_family", "strategy_family")
    )
    if any(token in text for token in _CONTROL_PLANE_IDENTITY_TOKENS):
        return None
    if not any(token in text for token in _MARKET_SIGNAL_IDENTITY_TOKENS):
        return None
    if any(token in text for token in ("option", "gamma", "iv_", "0dte")):
        return "options_surface", "iv_realized_1d_5d", ["gamma", "skew", "spread_quality"], ["options_chain", "iv_surface", "open_interest", "bid_ask_spread"]
    if any(token in text for token in ("future", "curve", "basis")):
        return "futures_event_session", "session_event_followthrough", ["basis", "curve", "macro_event_window"], ["futures_bars", "session_calendar", "basis_context", "macro_calendar"]
    if "crypto" in text:
        return "crypto_microstructure", "crypto_session_followthrough", ["liquidity_sweep", "basis", "funding_stress"], ["crypto_bars", "order_book_proxy", "funding_context"]
    if any(token in text for token in ("dividend", "income", "reit")):
        return "income_total_return", "20d_total_return_income", ["payout_safety", "dividend_cut_risk", "ex_dividend_window"], ["ex_dividend_calendar", "payout_metrics", "rate_context"]
    if any(token in text for token in ("intraday", "day_trading", "halt_reopen", "closing_auction", "low_float")):
        return "intraday_fast", "5m_30m_forward_return", ["1m", "5m", "15m", "60m"], ["one_minute_bars", "vwap", "spread_quality", "relative_volume"]
    if "swing" in text:
        return "multi_day", "2d_5d_forward_return", ["1d", "5d", "10d"], ["daily_bars", "sector_context", "macro_context", "overnight_gap"]
    if any(token in text for token in ("conservative", "credit", "rate", "bond", "liquidity_preservation")):
        return "risk_adjusted_preservation", "drawdown_avoidance_5d", ["vol_adjusted_return", "max_drawdown", "cash_parking"], ["volatility_budget", "credit_stress", "liquidity_state"]
    return "generic_directional", "1d_forward_return", ["5d_forward_return", "risk_adjusted_return"], ["price_bars", "volume", "market_context"]


def _infer_label_family(row: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    market_signal_override = _market_signal_label_override(row)
    if market_signal_override is not None:
        return market_signal_override
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in (
            "bot_id",
            "slot_kind",
            "slot_label",
            "sleeve_profile",
            "sleeve_family",
            "strategy_family",
            "bot_role",
            "intelligence_system",
            "governance_layer",
        )
    )
    rules: list[tuple[tuple[str, ...], str, str, list[str], list[str]]] = [
        (("label", "training", "retrain", "coverage"), "training_process_quality", "retrain_cycle_improves_gate_status", ["coverage_gap_delta", "label_quality_delta", "runtime_failure_delta"], ["training_quality_trace", "coverage_gap_trace", "runtime_snapshot_trace"]),
        (("governor", "backlog", "storage", "memory", "auth", "operator", "lineage", "guard"), "operational_guard_effect", "guard_prevents_bad_runtime_action", ["false_positive_guard", "incident_prevention", "pressure_delta"], ["runtime_health", "incident_log", "operator_context"]),
        (("quant_pricing", "merton", "heston", "monte_carlo", "quasi", "latin_hypercube", "finite_difference", "fft", "trinomial", "sabr", "svi", "dupire", "bates"), "quant_pricing_research", "pricing_model_dispersion_after_cost", ["surface_mispricing", "hedge_cost", "proxy_label_quality"], ["quant_model_feature_surface", "listed_option_surface", "realized_vol", "rates_context", "model_price_sensitivity_grid"]),
        (("state_space", "kalman", "particle_filter", "regime_filter", "hidden_markov", "changepoint"), "state_space_filter_research", "state_filter_regime_confidence", ["latent_state_stability", "regime_transition_quality", "sequence_depth"], ["runtime_feature_history", "market_micro_features", "state_filter_diagnostics", "regime_transition_trace"]),
        (("optimization", "kelly", "optimizer", "portfolio_fit", "convex", "genetic"), "optimization_research", "objective_improvement_after_constraints", ["constraint_violation", "kelly_sizing_quality", "portfolio_fit"], ["optimization_search_trace", "constraint_violation_trace", "objective_value_trace", "kelly_fraction_trace"]),
        (("transaction_cost", "slippage", "queue", "fill_realism", "market_impact"), "transaction_cost_slippage_research", "slippage_realism_after_queue_cost", ["queue_cost", "fill_quality", "market_impact_proxy"], ["transaction_cost_surface", "fill_realism_context", "queue_position_context", "latency_trace"]),
        (("basis", "cross_asset_basis", "funding_basis", "rates_fx"), "cross_asset_basis_research", "basis_convergence_after_funding_cost", ["basis_widening", "funding_stress", "cross_asset_confirmation"], ["basis", "basis_context", "cross_asset_correlation", "funding_context", "proxy_data_source_lineage"]),
        (("tape_normalization", "nbbo", "taq", "sip", "opra"), "market_data_tape_normalization_research", "tape_schema_quality_after_latency", ["feed_latency", "schema_quality", "off_exchange_volume"], ["opra_nbbo_taq_sip_normalized_events", "mbo_mbp_depth_snapshot", "dark_pool_off_exchange_volume", "feed_latency_schema_health"]),
        (("order_flow_toxicity", "vpin", "toxic_flow"), "order_flow_toxicity_research", "toxic_flow_spread_queue_response", ["vpin_toxicity", "toxic_flow", "spread_queue_response"], ["vpin_order_flow_toxicity", "market_micro_features", "spread_queue_response", "source_quality"]),
        (("option", "gamma", "iv", "0dte"), "options_surface", "iv_realized_1d_5d", ["gamma", "skew", "spread_quality", "event_vol_reset"], ["options_chain", "iv_surface", "open_interest", "bid_ask_spread"]),
        (("future", "basis", "curve"), "futures_event_session", "session_event_followthrough", ["basis", "curve", "macro_event_window"], ["futures_bars", "session_calendar", "basis_context", "macro_calendar"]),
        (("crypto",), "crypto_microstructure", "crypto_session_followthrough", ["liquidity_sweep", "basis", "funding_stress"], ["crypto_bars", "order_book_proxy", "funding_context"]),
        (("dividend", "income", "drip", "payout"), "income_total_return", "20d_total_return_income", ["payout_safety", "dividend_cut_risk", "ex_dividend_window"], ["ex_dividend_calendar", "payout_metrics", "rate_context"]),
        (("conservative", "capital_preservation", "cash_parking"), "risk_adjusted_preservation", "drawdown_avoidance_5d", ["vol_adjusted_return", "max_drawdown", "cash_parking"], ["volatility_budget", "credit_stress", "liquidity_state"]),
        (("intraday", "scalp", "vwap", "opening_range", "same_session"), "intraday_fast", "5m_30m_forward_return", ["1m", "5m", "15m", "60m"], ["one_minute_bars", "vwap", "spread_quality", "relative_volume"]),
        (("swing", "position", "multi_day"), "multi_day", "2d_5d_forward_return", ["1d", "5d", "10d"], ["daily_bars", "sector_context", "macro_context", "overnight_gap"]),
        (("quant", "alpha", "factor", "model", "research"), "alpha_research", "walk_forward_alpha_after_cost", ["regime_edge", "slippage_adjusted_edge", "overfit_gap"], ["feature_store_lineage", "walk_forward_trace", "execution_cost_context"]),
    ]
    for tokens, family, primary, aux, context in rules:
        if any(token in text for token in tokens):
            return family, primary, aux, context
    role = str(row.get("bot_role") or "").strip().lower()
    if role in {"infrastructure_bot", "infrastructure_sub_bot"}:
        intake = row.get("data_intake_collections") if isinstance(row.get("data_intake_collections"), list) else []
        context = _ordered_unique([*intake, "runtime_health", "incident_log", "operator_context"])
        return (
            "operational_guard_effect",
            "operational_action_improves_verified_runtime_state",
            ["incident_prevention", "false_positive_guard", "runtime_health_delta"],
            context[:16],
        )
    return "generic_directional", "1d_forward_return", ["5d_forward_return", "risk_adjusted_return"], ["price_bars", "volume", "market_context"]


def _training_lane_for_family(label_family: str) -> str:
    if label_family in {"intraday_fast", "options_surface", "income_options_surface", "futures_event_session", "crypto_microstructure", "execution_cost_quality"}:
        return "lane_specific_fast"
    if label_family in {"training_process_quality", "operational_guard_effect"}:
        return "governance_effect"
    if label_family in ADVANCED_QUANT_LABEL_FAMILIES:
        return "research_quant_proxy"
    if label_family in RESEARCH_VALIDATION_LABEL_FAMILIES:
        return "research_quant_proxy"
    if label_family in {"alpha_research"}:
        return "research_walk_forward"
    if label_family in {"income_total_return", "risk_adjusted_preservation", "multi_day", "fixed_income_rates", "credit_spread", "correlation_risk_effect"}:
        return "slow_lane_balanced"
    return "general_balanced"


def _with_free_source_context(contract: dict[str, Any]) -> dict[str, Any]:
    out = dict(contract)
    out["free_source_context_candidates"] = _free_source_candidates_for_contexts(list(out.get("required_context") or []))
    out["free_source_context_policy"] = "point_in_time_verified_free_public_sources_only"
    return out


def _universal_contract(row: dict[str, Any]) -> dict[str, Any]:
    existing = _existing_contract(row)
    bot_id = str(row.get("bot_id") or "").strip()
    override = TARGETED_LABEL_CONTRACT_OVERRIDES.get(bot_id)
    if override:
        label_family = str(override["label_family"])
        return _with_free_source_context({
            "version": UNIVERSAL_LABEL_CONTRACT_VERSION,
            "label_family": label_family,
            "primary_horizon": str(override["primary_horizon"]),
            "aux_horizons": list(override["aux_horizons"]),
            "required_context": list(override["required_context"]),
            "required_labels": list(REQUIRED_LABELS),
            "required_join_mode": "point_in_time_only",
            "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
            "quality_floor": _safe_float(existing.get("quality_floor"), 0.84),
            "training_lane": _training_lane_for_family(label_family),
            "source": "targeted_labeling_repair_override",
        })
    family, primary, aux, context = _infer_label_family(row)
    existing_family = str(existing.get("label_family") or existing.get("family") or "").strip()
    existing_primary = str(existing.get("primary_horizon") or existing.get("primary_label_horizon") or "").strip()
    existing_aux = existing.get("aux_horizons") or existing.get("aux_label_horizons")
    existing_context = existing.get("required_context") or existing.get("required_label_context")
    role = str(row.get("bot_role") or "").strip().lower()
    role_mismatch_repair = bool(
        role in {"infrastructure_bot", "infrastructure_sub_bot"}
        and existing_family == "generic_directional"
    )
    signal_market_mismatch_repair = bool(
        existing_family in OPERATIONAL_LABEL_FAMILIES
        and _market_signal_label_override(row) is not None
    )
    if role_mismatch_repair:
        existing_family = ""
        existing_primary = ""
        existing_aux = None
        existing_context = None
    elif signal_market_mismatch_repair:
        existing_family = ""
        existing_primary = ""
        existing_aux = None
        existing_context = None
    if role_mismatch_repair:
        required_labels = OPERATIONAL_REQUIRED_LABELS
    elif signal_market_mismatch_repair:
        required_labels = REQUIRED_LABELS
    else:
        required_labels = existing.get("required_labels") if isinstance(existing.get("required_labels"), list) else REQUIRED_LABELS
    label_family = existing_family or family
    return _with_free_source_context({
        "version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "label_family": label_family,
        "primary_horizon": existing_primary or primary,
        "aux_horizons": list(existing_aux) if isinstance(existing_aux, list) and existing_aux else aux,
        "required_context": list(existing_context) if isinstance(existing_context, list) and existing_context else context,
        "required_labels": list(required_labels),
        "required_join_mode": str(existing.get("required_join_mode") or "point_in_time_only"),
        "forbidden_join_modes": list(existing.get("forbidden_join_modes") or ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"]),
        "quality_floor": _safe_float(existing.get("quality_floor"), 0.84),
        "training_lane": _training_lane_for_family(label_family),
        "source": (
            "role_mismatch_repair_override"
            if role_mismatch_repair
            else "signal_market_label_repair_override"
            if signal_market_mismatch_repair
            else "preserved_existing_contract"
            if existing
            else "inferred_from_registry_identity"
        ),
    })


def _label_objective_class(row: dict[str, Any], contract: dict[str, Any]) -> str:
    family = str(contract.get("label_family") or "generic_directional").strip().lower()
    role = str(row.get("bot_role") or "").strip().lower()
    lane = str(contract.get("training_lane") or row.get("training_lane") or "").strip().lower()
    if family in OPERATIONAL_LABEL_FAMILIES:
        return "operational_effect"
    if role in {"infrastructure_bot", "infrastructure_sub_bot"}:
        return "operational_effect"
    if family in RESEARCH_VALIDATION_LABEL_FAMILIES:
        return "research_validation"
    if family.endswith("_research") or lane in {"research_walk_forward", "research_quant_proxy"}:
        return "research_validation"
    if family in MARKET_OUTCOME_LABEL_FAMILIES:
        return "market_outcome"
    return "market_outcome"


_HORIZON_UNIT_SECONDS = {
    "m": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
    "w": 7 * 24 * 60 * 60,
}


def _label_horizon_policy(primary_horizon: str, objective_class: str) -> dict[str, Any]:
    semantic_horizon = str(primary_horizon or "").strip().lower()
    if objective_class != "market_outcome":
        return {
            "semantic_horizon": semantic_horizon,
            "enforcement_mode": "authority_specific_outcome_boundary",
            "minimum_maturity_seconds": 0,
            "maximum_maturity_seconds": 0,
            "selection_mode": "authority_specific_join_only",
        }

    durations = sorted(
        int(value) * _HORIZON_UNIT_SECONDS[unit]
        for value, unit in re.findall(r"(?<![a-z0-9])(\d+)([mhdw])(?=_|$)", semantic_horizon)
    )
    if durations:
        minimum_seconds = int(durations[0])
        semantic_maximum_seconds = int(durations[-1])
        if semantic_maximum_seconds >= _HORIZON_UNIT_SECONDS["d"]:
            closure_slack_seconds = max(
                3 * _HORIZON_UNIT_SECONDS["d"],
                int(round(semantic_maximum_seconds * 0.35)),
            )
        elif semantic_maximum_seconds >= _HORIZON_UNIT_SECONDS["h"]:
            closure_slack_seconds = max(2 * _HORIZON_UNIT_SECONDS["h"], semantic_maximum_seconds)
        else:
            closure_slack_seconds = max(30 * _HORIZON_UNIT_SECONDS["m"], semantic_maximum_seconds * 3)
        return {
            "semantic_horizon": semantic_horizon,
            "enforcement_mode": "strict_wall_clock_range",
            "minimum_maturity_seconds": minimum_seconds,
            "semantic_maximum_maturity_seconds": semantic_maximum_seconds,
            "maximum_maturity_seconds": int(semantic_maximum_seconds + closure_slack_seconds),
            "closure_slack_seconds": int(closure_slack_seconds),
            "selection_mode": "first_same_symbol_mode_snapshot_at_or_after_minimum",
        }

    if "session" in semantic_horizon or "event" in semantic_horizon:
        return {
            "semantic_horizon": semantic_horizon,
            "enforcement_mode": "session_event_boundary",
            "minimum_maturity_seconds": 5 * 60,
            "maximum_maturity_seconds": 36 * 60 * 60,
            "selection_mode": "first_same_symbol_mode_snapshot_at_or_after_minimum",
        }
    return {
        "semantic_horizon": semantic_horizon,
        "enforcement_mode": "explicit_event_boundary_required",
        "minimum_maturity_seconds": 0,
        "maximum_maturity_seconds": 0,
        "selection_mode": "configured_row_horizon_with_evidence_audit",
    }


def _bot_label_materialization_contract(row: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    universal = dict(contract or _universal_contract(row))
    objective_class = _label_objective_class(row, universal)
    family = str(universal.get("label_family") or "generic_directional")
    primary_horizon = str(universal.get("primary_horizon") or "").strip()
    bot_id = str(row.get("bot_id") or "").strip()
    if objective_class == "operational_effect":
        authority = "verified_internal_control_outcome"
        join_mode = "internal_event_timestamp_and_action_id_only"
        join_keys = ["bot_id", "action_id", "event_timestamp_utc", "outcome_timestamp_utc", "artifact_sha256"]
        outputs = list(OPERATIONAL_REQUIRED_LABELS)
        maturity_rule = "outcome_timestamp_must_follow_action_timestamp_and_reference_a_verified_runtime_artifact"
        source_policy = "market_price_must_not_proxy_an_operational_or_governance_outcome"
    elif objective_class == "research_validation":
        authority = "walk_forward_out_of_sample_evidence"
        join_mode = "immutable_research_snapshot_and_fold_id_only"
        join_keys = ["bot_id", "experiment_id", "snapshot_id", "fold_id", "evaluated_at_utc", "artifact_sha256"]
        outputs = list(RESEARCH_REQUIRED_LABELS)
        maturity_rule = "label_exists_only_after_the_out_of_sample_fold_and_stability_checks_complete"
        source_policy = "in_sample_fit_and_unverified_proxy_outputs_are_never_promotion_labels"
    else:
        authority = "matured_market_and_paper_outcome_evidence"
        join_mode = "point_in_time_symbol_mode_snapshot_only"
        join_keys = [
            "bot_id",
            "symbol",
            "mode",
            "feature_timestamp_utc",
            "label_matured_at_utc",
            "feature_snapshot_id",
            "label_snapshot_id",
        ]
        outputs = [
            "forward_return_bucket",
            "risk_adjusted_return_bucket",
            "side_specific_outcome",
            "abstention_outcome",
            "counterfactual_opportunity_trace",
            "sample_eligibility_reason",
            "label_source_confidence_norm",
        ]
        maturity_rule = "label_matured_at_utc_must_be_strictly_after_feature_timestamp_utc_within_the_same_symbol_and_mode"
        source_policy = "paper_decisions_use_broker_or_verified_market_outcomes; missing_prices_never_become_class_zero"

    directional_fallback_allowed = bool(
        objective_class == "market_outcome" and family == "generic_directional"
    )
    lifecycle = str(row.get("lifecycle_state") or "").strip().lower()
    training_excluded = bool(row.get("training_excluded", row.get("exclude_from_training", False)))
    horizon_policy = _label_horizon_policy(primary_horizon, objective_class)
    evidence_state = (
        "lifecycle_ineligible"
        if lifecycle in {"deleted", "disabled", "inactive", "retired", "tombstoned"}
        else "collection_only_evidence_pending"
        if training_excluded or lifecycle == "data_collection_only"
        else "runtime_evidence_gate_required"
        if objective_class == "market_outcome"
        else "verified_outcome_join_required"
    )
    payload = {
        "version": LABEL_MATERIALIZATION_CONTRACT_VERSION,
        "bot_id": bot_id,
        "label_family": family,
        "primary_horizon": primary_horizon,
        "training_lane": str(universal.get("training_lane") or row.get("training_lane") or ""),
        "objective_class": objective_class,
        "outcome_authority": authority,
        "required_join_mode": join_mode,
        "required_join_keys": join_keys,
        "required_outputs": outputs,
        "maturity_rule": maturity_rule,
        "label_horizon_policy": horizon_policy,
        "minimum_label_maturity_seconds": int(horizon_policy.get("minimum_maturity_seconds", 0) or 0),
        "maximum_label_maturity_seconds": int(horizon_policy.get("maximum_maturity_seconds", 0) or 0),
        "source_policy": source_policy,
        "point_in_time_guard_required": True,
        "lineage_hash_required": True,
        "sample_eligibility_reason_required": True,
        "decision_trace_required_for_action_labels": True,
        "directional_fallback_allowed": directional_fallback_allowed,
        "sample_filter_bypass_allowed": False,
        "runtime_price_labeling_allowed": objective_class == "market_outcome",
        "evaluation_split_policy": "purged_chronological_only",
        "split_embargo_required": True,
        "feature_normalization_fit_scope": "train_partition_only",
        "evidence_state": evidence_state,
        "promotion_policy": "pending_or_unknown_outcomes_are_not_training_evidence_and_cannot_clear_promotion",
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _all_bot_label_materialization_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    routes: list[dict[str, Any]] = []
    objective_counts: Counter[str] = Counter()
    evidence_state_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    horizon_enforcement_counts: Counter[str] = Counter()
    misrouted_before: list[str] = []
    misrouted_after: list[str] = []
    market_signal_misroutes_before: list[str] = []
    market_signal_misroutes_after: list[str] = []
    for row in rows:
        bot_id = str(row.get("bot_id") or "").strip()
        role = str(row.get("bot_role") or "").strip().lower()
        existing = row.get("universal_label_contract") if isinstance(row.get("universal_label_contract"), dict) else {}
        if not existing:
            existing = _existing_contract(row)
        existing_family = str(existing.get("label_family") or "")
        repair_status = str(row.get("training_label_contract_status") or "")
        persisted_contract = _existing_contract(row)
        repair_source = str(persisted_contract.get("source") or existing.get("source") or "")
        if (
            repair_status == "role_mismatch_label_repair"
            or repair_source == "role_mismatch_repair_override"
            or role in {"infrastructure_bot", "infrastructure_sub_bot"} and existing_family == "generic_directional"
        ):
            misrouted_before.append(bot_id)
        if (
            repair_status == "signal_market_label_repair"
            or repair_source == "signal_market_label_repair_override"
            or existing_family in OPERATIONAL_LABEL_FAMILIES and _market_signal_label_override(row) is not None
        ):
            market_signal_misroutes_before.append(bot_id)
        universal = _universal_contract(row)
        materialization = _bot_label_materialization_contract(row, universal)
        objective = str(materialization["objective_class"])
        if role in {"infrastructure_bot", "infrastructure_sub_bot"} and objective == "market_outcome":
            misrouted_after.append(bot_id)
        if (
            existing_family in OPERATIONAL_LABEL_FAMILIES
            and _market_signal_label_override(row) is not None
            and objective != "market_outcome"
        ):
            market_signal_misroutes_after.append(bot_id)
        objective_counts[objective] += 1
        evidence_state_counts[str(materialization["evidence_state"])] += 1
        family_counts[str(materialization["label_family"])] += 1
        horizon_policy = materialization.get("label_horizon_policy") if isinstance(materialization.get("label_horizon_policy"), dict) else {}
        horizon_enforcement_counts[str(horizon_policy.get("enforcement_mode") or "missing")] += 1
        routes.append(
            {
                "bot_id": bot_id,
                "bot_role": str(row.get("bot_role") or ""),
                "lifecycle_state": str(row.get("lifecycle_state") or ""),
                "training_excluded": bool(row.get("training_excluded", row.get("exclude_from_training", False))),
                "label_family": str(materialization["label_family"]),
                "primary_horizon": str(materialization.get("primary_horizon") or ""),
                "training_lane": str(materialization["training_lane"]),
                "objective_class": objective,
                "outcome_authority": str(materialization["outcome_authority"]),
                "evidence_state": str(materialization["evidence_state"]),
                "directional_fallback_allowed": bool(materialization["directional_fallback_allowed"]),
                "horizon_enforcement_mode": str(horizon_policy.get("enforcement_mode") or ""),
                "minimum_label_maturity_seconds": int(materialization.get("minimum_label_maturity_seconds", 0) or 0),
                "maximum_label_maturity_seconds": int(materialization.get("maximum_label_maturity_seconds", 0) or 0),
                "contract_sha256": str(materialization["contract_sha256"]),
            }
        )
    return {
        "timestamp_utc": _utc_now(),
        "schema_version": 2,
        "contract_version": LABEL_MATERIALIZATION_CONTRACT_VERSION,
        "total_bot_count": len(rows),
        "routed_bot_count": len(routes),
        "route_coverage_ratio": round(len(routes) / max(len(rows), 1), 6),
        "unique_contract_hash_count": len({str(route["contract_sha256"]) for route in routes}),
        "objective_class_counts": dict(sorted(objective_counts.items())),
        "evidence_state_counts": dict(sorted(evidence_state_counts.items())),
        "label_family_counts": dict(sorted(family_counts.items())),
        "horizon_enforcement_counts": dict(sorted(horizon_enforcement_counts.items())),
        "directional_fallback_allowed_count": sum(1 for route in routes if route["directional_fallback_allowed"]),
        "directional_fallback_forbidden_count": sum(1 for route in routes if not route["directional_fallback_allowed"]),
        "misrouted_directional_infrastructure_before_count": len(misrouted_before),
        "misrouted_directional_infrastructure_before_bot_ids": misrouted_before[:500],
        "misrouted_directional_infrastructure_after_count": len(misrouted_after),
        "misrouted_directional_infrastructure_after_bot_ids": misrouted_after[:500],
        "misrouted_market_signal_guards_before_count": len(market_signal_misroutes_before),
        "misrouted_market_signal_guards_before_bot_ids": market_signal_misroutes_before[:500],
        "misrouted_market_signal_guards_after_count": len(market_signal_misroutes_after),
        "misrouted_market_signal_guards_after_bot_ids": market_signal_misroutes_after[:500],
        "all_bot_routes": routes,
        "policy": "contract coverage is not outcome evidence; each bot remains pending until its authority-specific labels mature",
    }


def _compact_materialization_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in coverage.items()
        if key != "all_bot_routes" and not key.endswith("_bot_ids")
    }
    compact["detailed_routes_artifact"] = (
        "governance/training_labeling_intelligence/all_bot_label_materialization_latest.json"
    )
    compact["detailed_route_rows_omitted_from_health"] = len(coverage.get("all_bot_routes") or [])
    return compact


def _compact_health_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    enrichment = dict(compact.get("free_label_source_enrichment") or {})
    context_sources = list(enrichment.pop("context_sources", []) or [])
    source_statuses = dict(enrichment.pop("source_statuses", {}) or {})
    enrichment["context_source_rows_omitted_from_health"] = len(context_sources)
    enrichment["source_status_rows_omitted_from_health"] = len(source_statuses)
    enrichment["detailed_artifact"] = (
        "governance/training_labeling_intelligence/free_label_source_enrichment_latest.json"
    )
    compact["free_label_source_enrichment"] = enrichment

    plan = dict(compact.get("label_materialization_plan") or {})
    materialization_queue = list(plan.pop("materialization_queue", []) or [])
    plan["materialization_queue_rows_omitted_from_health"] = len(materialization_queue)
    plan["detailed_artifact"] = (
        "governance/training_labeling_intelligence/label_materialization_plan_latest.json"
    )
    compact["label_materialization_plan"] = plan
    return compact


def _apply_bot_label_materialization_contracts(rows: list[dict[str, Any]], now: str) -> dict[str, Any]:
    changed = 0
    for row in rows:
        contract = _bot_label_materialization_contract(row, _universal_contract(row))
        if row.get("training_label_materialization_contract") != contract:
            changed += 1
        row["training_label_materialization_contract"] = contract
        row["training_label_materialization_contract_version"] = LABEL_MATERIALIZATION_CONTRACT_VERSION
        row["training_label_materialization_last_reviewed_utc"] = now
    coverage = _all_bot_label_materialization_coverage(rows)
    coverage["updated_bot_count"] = changed
    return coverage


def _apply_universal_label_contracts(rows: list[dict[str, Any]], now: str) -> dict[str, Any]:
    missing_before = 0
    normalized_missing = 0
    normalized_incomplete = 0
    preserved_explicit = 0
    family_counts: Counter[str] = Counter()
    lane_counts: Counter[str] = Counter()
    updated_bot_ids: list[str] = []

    for row in rows:
        had_any = bool(_existing_contract(row) or row.get("data_label_contract_version"))
        complete = _contract_complete(row)
        if not had_any:
            missing_before += 1
        contract = _universal_contract(row)
        source = str(contract.get("source") or "")
        targeted_override = source == "targeted_labeling_repair_override"
        role_mismatch_override = source == "role_mismatch_repair_override"
        signal_market_override = source == "signal_market_label_repair_override"
        family_counts[str(contract["label_family"])] += 1
        lane_counts[str(contract["training_lane"])] += 1
        status = (
            "targeted_labeling_repair"
            if targeted_override
            else "role_mismatch_label_repair"
            if role_mismatch_override
            else "signal_market_label_repair"
            if signal_market_override
            else "preserved_explicit"
            if complete
            else "normalized_incomplete"
            if had_any
            else "normalized_missing"
        )
        if status == "preserved_explicit":
            preserved_explicit += 1
        elif status == "normalized_incomplete":
            normalized_incomplete += 1
        elif status in {"targeted_labeling_repair", "role_mismatch_label_repair", "signal_market_label_repair"}:
            normalized_incomplete += 1
        else:
            normalized_missing += 1
        row["universal_label_contract"] = contract
        row["universal_label_contract_version"] = UNIVERSAL_LABEL_CONTRACT_VERSION
        row["training_labeling_intelligence_version"] = PACK_VERSION
        row["training_label_contract_status"] = status
        row["training_lane"] = contract["training_lane"]
        row["label_contract_last_reviewed_utc"] = now
        if targeted_override or role_mismatch_override or signal_market_override or not complete:
            row["label_contract"] = contract
            row["data_label_contract_version"] = UNIVERSAL_LABEL_CONTRACT_VERSION
            updated_bot_ids.append(str(row.get("bot_id") or ""))
        existing_tags = row.get("labeling_tags") if isinstance(row.get("labeling_tags"), list) else []
        row["labeling_tags"] = _ordered_unique(
            [
                *existing_tags,
                "universal_label_contract",
                "point_in_time_only",
                f"label_family:{contract['label_family']}",
                f"training_lane:{contract['training_lane']}",
                f"label_contract_version:{UNIVERSAL_LABEL_CONTRACT_VERSION}",
            ]
        )
    return {
        "total_rows": len(rows),
        "missing_contracts_before": missing_before,
        "normalized_missing_contracts": normalized_missing,
        "normalized_incomplete_contracts": normalized_incomplete,
        "preserved_explicit_contracts": preserved_explicit,
        "updated_label_contract_bot_count": len(updated_bot_ids),
        "updated_label_contract_bot_ids": updated_bot_ids[:250],
        "label_family_counts": dict(sorted(family_counts.items())),
        "training_lane_counts": dict(sorted(lane_counts.items())),
        "coverage_ratio_after": 1.0 if rows else 0.0,
    }


def _is_training_labeling_bot_identity(row: dict[str, Any]) -> bool:
    bot_id = str(row.get("bot_id") or "").strip()
    slot_kind = str(row.get("slot_kind") or "").strip()
    return bool(
        "_training_labeling_" in bot_id
        or str(row.get("sleeve_family") or "").strip() == SLEEVE_FAMILY
        or str(row.get("capability_pack_slug") or "").strip() == PACK_SLUG
        or slot_kind.startswith(f"{PACK_SLUG}_")
    )


def _is_training_labeling_structured_pack_row(row: dict[str, Any]) -> bool:
    slot_kind = str(row.get("slot_kind") or "").strip()
    return bool(
        str(row.get("capability_pack_slug") or "").strip() == PACK_SLUG
        or slot_kind.startswith(f"{PACK_SLUG}_")
    )


def _training_labeling_collection_guard_ready(row: dict[str, Any]) -> bool:
    return bool(
        row.get("active", False) is True
        and str(row.get("lifecycle_state") or "").strip() == "data_collection_only"
        and row.get("data_collection_active", False) is True
        and row.get("training_excluded", False) is True
        and row.get("exclude_from_training", False) is True
        and row.get("rotation_blocked", False) is True
        and _safe_float(row.get("weight"), 0.0) == 0.0
        and _safe_float(row.get("preference_score"), 0.0) == 0.0
        and row.get("trading_enabled", False) is False
        and row.get("paper_trading_enabled", False) is False
        and row.get("live_trading_enabled", False) is False
        and row.get("execution_enabled", False) is False
        and row.get("allocation_enabled", False) is False
    )


def _training_labeling_collection_guard_preview(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in rows if _is_training_labeling_bot_identity(row)]
    noncompliant = [row for row in matched if not _training_labeling_collection_guard_ready(row)]
    legacy = [row for row in matched if not _is_training_labeling_structured_pack_row(row)]
    return {
        "schema_version": 1,
        "mode": "preview",
        "matched_bot_count": len(matched),
        "structured_pack_bot_count": len(matched) - len(legacy),
        "legacy_training_labeling_bot_count": len(legacy),
        "noncompliant_before_count": len(noncompliant),
        "noncompliant_bot_ids": [str(row.get("bot_id") or "") for row in noncompliant[:250]],
        "policy": "training_labeling_bots_are_collection_only_zero_weight_excluded_until_threshold_clearance",
    }


def _apply_training_labeling_collection_guard(rows: list[dict[str, Any]], now: str) -> dict[str, Any]:
    matched_count = 0
    structured_pack_count = 0
    legacy_count = 0
    noncompliant_before: list[str] = []
    updated_bot_ids: list[str] = []
    legacy_repaired_bot_ids: list[str] = []

    def set_if_changed(row: dict[str, Any], key: str, value: Any) -> bool:
        if row.get(key) == value:
            return False
        row[key] = value
        return True

    for row in rows:
        if not _is_training_labeling_bot_identity(row):
            continue
        matched_count += 1
        structured_pack_row = _is_training_labeling_structured_pack_row(row)
        if structured_pack_row:
            structured_pack_count += 1
        else:
            legacy_count += 1
        guard_ready_before = _training_labeling_collection_guard_ready(row)
        if not guard_ready_before:
            noncompliant_before.append(str(row.get("bot_id") or ""))

        changed = False
        changed |= set_if_changed(row, "active", True)
        changed |= set_if_changed(row, "lifecycle_state", "data_collection_only")
        changed |= set_if_changed(row, "data_collection_active", True)
        changed |= set_if_changed(row, "data_collection_mode", "active_observer")
        if not str(row.get("data_collection_started_utc") or "").strip():
            row["data_collection_started_utc"] = now
            changed = True
        changed |= set_if_changed(
            row,
            "data_collection_reason",
            "training_labeling_intelligence_collect_only_until_label_and_training_effect_gates_clear",
        )
        changed |= set_if_changed(row, "weight", 0.0)
        changed |= set_if_changed(row, "preference_score", 0.0)
        changed |= set_if_changed(row, "promoted", False)
        changed |= set_if_changed(row, "trading_enabled", False)
        changed |= set_if_changed(row, "paper_trading_enabled", False)
        changed |= set_if_changed(row, "live_trading_enabled", False)
        changed |= set_if_changed(row, "allocation_enabled", False)
        changed |= set_if_changed(row, "execution_enabled", False)
        changed |= set_if_changed(row, "rotation_blocked", True)
        changed |= set_if_changed(row, "rotation_block_reason", "training_labeling_intelligence_collection_only_zero_weight")
        changed |= set_if_changed(row, "training_excluded", True)
        changed |= set_if_changed(row, "exclude_from_training", True)
        changed |= set_if_changed(row, "training_candidate_after_threshold", True)
        changed |= set_if_changed(row, "training_exclusion_reason", "collecting_training_labeling_effect_evidence_before_training")
        changed |= set_if_changed(row, "training_exclusion_until", "minimum_data_collection_threshold_met")
        changed |= set_if_changed(row, "data_collection_storage_guarded", True)
        changed |= set_if_changed(row, "data_collection_capture_mode", "thin_digest_with_heartbeat_fallback")
        changed |= set_if_changed(row, "data_collection_sample_rate", SAMPLE_RATE)
        changed |= set_if_changed(row, "data_collection_max_daily_storage_mb", MAX_DAILY_MB_PER_BOT)
        changed |= set_if_changed(row, "data_collection_max_daily_mb", float(MAX_DAILY_MB_PER_BOT))
        changed |= set_if_changed(row, "data_collection_compute_guard_mode", "pressure_self_accommodating")
        changed |= set_if_changed(row, "data_collection_training_ready", False)
        changed |= set_if_changed(row, "eligible_for_master_vote", False)
        changed |= set_if_changed(row, "direct_execution_allowed", False)
        changed |= set_if_changed(row, "paper_trade_lock_required", True)
        changed |= set_if_changed(row, "sleeve_family", SLEEVE_FAMILY)
        changed |= set_if_changed(row, "strategy_family", "training_and_labeling_governance")
        changed |= set_if_changed(row, "training_labeling_collection_guard_version", PACK_VERSION)
        if not str(row.get("training_labeling_collection_guarded_utc") or "").strip():
            row["training_labeling_collection_guarded_utc"] = now
            changed = True
        if not structured_pack_row:
            changed |= set_if_changed(row, "legacy_training_labeling_collection_guard_version", PACK_VERSION)
            if not guard_ready_before:
                legacy_repaired_bot_ids.append(str(row.get("bot_id") or ""))

        current_min_observations = _safe_int(row.get("minimum_training_observations"), 0)
        if current_min_observations < MINIMUM_TRAINING_OBSERVATIONS:
            row["minimum_training_observations"] = MINIMUM_TRAINING_OBSERVATIONS
            changed = True
        current_min_days = _safe_int(row.get("minimum_data_collection_days"), 0)
        if current_min_days < MINIMUM_COLLECTION_DAYS:
            row["minimum_data_collection_days"] = MINIMUM_COLLECTION_DAYS
            changed = True

        threshold_policy = row.get("training_threshold_policy") if isinstance(row.get("training_threshold_policy"), dict) else {}
        updated_policy = {
            **threshold_policy,
            "minimum_observations": max(_safe_int(threshold_policy.get("minimum_observations"), 0), MINIMUM_TRAINING_OBSERVATIONS),
            "minimum_collection_days": max(_safe_int(threshold_policy.get("minimum_collection_days"), 0), MINIMUM_COLLECTION_DAYS),
            "requires_label_contract_clearance": True,
            "requires_runtime_pressure_clearance": True,
            "requires_backpressure_clearance": True,
            "requires_schema_lineage_clearance": True,
            "requires_paper_live_separation_clearance": True,
            "requires_global_halt_clear": True,
        }
        changed |= set_if_changed(row, "training_threshold_policy", updated_policy)

        progress = row.get("data_collection_threshold_progress") if isinstance(row.get("data_collection_threshold_progress"), dict) else {}
        observations = max(
            _safe_int(progress.get("observations"), 0),
            _safe_int(row.get("data_collection_observations"), 0),
            _safe_int(row.get("observations"), 0),
        )
        updated_progress = {
            **_threshold_progress(),
            **progress,
            "observations": observations,
            "minimum_training_observations": max(
                _safe_int(progress.get("minimum_training_observations"), 0),
                MINIMUM_TRAINING_OBSERVATIONS,
            ),
            "observations_ready": False,
            "minimum_data_collection_days": max(
                _safe_int(progress.get("minimum_data_collection_days"), 0),
                MINIMUM_COLLECTION_DAYS,
            ),
            "days_ready": False,
            "training_ready": False,
        }
        changed |= set_if_changed(row, "data_collection_threshold_progress", updated_progress)

        existing_tags = row.get("labeling_tags") if isinstance(row.get("labeling_tags"), list) else []
        updated_tags = _ordered_unique(
            [
                *existing_tags,
                "research_only",
                "collection_only",
                "execution_blocked",
                f"sleeve_family:{SLEEVE_FAMILY}",
                f"collection_guard:{PACK_VERSION}",
            ]
        )
        changed |= set_if_changed(row, "labeling_tags", updated_tags)

        if changed:
            updated_bot_ids.append(str(row.get("bot_id") or ""))

    return {
        "schema_version": 1,
        "mode": "applied",
        "matched_bot_count": matched_count,
        "structured_pack_bot_count": structured_pack_count,
        "legacy_training_labeling_bot_count": legacy_count,
        "noncompliant_before_count": len(noncompliant_before),
        "updated_bot_count": len(updated_bot_ids),
        "legacy_repaired_bot_count": len(legacy_repaired_bot_ids),
        "noncompliant_before_bot_ids": noncompliant_before[:250],
        "updated_bot_ids": updated_bot_ids[:250],
        "legacy_repaired_bot_ids": legacy_repaired_bot_ids[:250],
        "policy": "training_labeling_bots_are_collection_only_zero_weight_excluded_until_threshold_clearance",
    }


def _pack_contract(assigned_ids: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": PACK_VERSION,
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "sleeve_family": SLEEVE_FAMILY,
        "display_name": PACK_DISPLAY_NAME,
        "system_count": len(INTELLIGENCE_SYSTEMS),
        "bot_count": len(BOTS),
        "bot_pack_size_rule": "6_training_labeling_systems_4_bots_each_24_bot_intelligence_layer",
        "dedicated_data_intake": list(BASE_DATA_INTAKES),
        "storage_retention_rule": {
            "retention_profile": "training_labeling_hot_3d_warm_120d_cold_540d",
            "storage_targets": list(STORAGE_TARGETS),
            "max_daily_mb_per_bot": MAX_DAILY_MB_PER_BOT,
            "capture_mode": "thin_digest_and_event_delta_only",
            "sample_rate": SAMPLE_RATE,
            "dedupe_required": True,
            "self_accommodation": "heartbeat_when_whole_system_governor_is_protective",
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
        "universal_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "authority_boundary": "advisory_labeling_and_training_process_intelligence_no_execution_no_allocation",
    }


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


def _minimum_observations_for_row(row: dict[str, Any]) -> int:
    standard = row.get("paper_promotion_standard") if isinstance(row.get("paper_promotion_standard"), dict) else {}
    threshold = row.get("data_collection_threshold") if isinstance(row.get("data_collection_threshold"), dict) else {}
    training_policy = row.get("training_threshold_policy") if isinstance(row.get("training_threshold_policy"), dict) else {}
    candidates = [
        standard.get("minimum_observations"),
        threshold.get("minimum_training_observations"),
        training_policy.get("minimum_observations"),
        row.get("minimum_training_observations"),
    ]
    for raw in candidates:
        value = _safe_int(raw, 0)
        if value > 0:
            return value
    return 0


def _minimum_collection_days_for_row(row: dict[str, Any]) -> int:
    standard = row.get("paper_promotion_standard") if isinstance(row.get("paper_promotion_standard"), dict) else {}
    training_policy = row.get("training_threshold_policy") if isinstance(row.get("training_threshold_policy"), dict) else {}
    candidates = [
        standard.get("minimum_collection_days"),
        training_policy.get("minimum_collection_days"),
        row.get("minimum_data_collection_days"),
    ]
    for raw in candidates:
        value = _safe_int(raw, 0)
        if value > 0:
            return value
    return 0


def _observation_count_for_row(row: dict[str, Any]) -> int:
    progress = row.get("data_collection_threshold_progress") if isinstance(row.get("data_collection_threshold_progress"), dict) else {}
    candidates = [
        row.get("data_collection_observations"),
        row.get("observations"),
        progress.get("observations"),
    ]
    return max(_safe_int(raw, 0) for raw in candidates)


def _collection_age_days_for_row(row: dict[str, Any], now: datetime) -> float:
    raw = str(row.get("data_collection_started_utc") or row.get("created_at_utc") or "").strip()
    if not raw:
        return 0.0
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return round(max((now - parsed.astimezone(timezone.utc)).total_seconds(), 0.0) / 86400.0, 3)
    except Exception:
        return 0.0


def _label_depth_contract(
    row: dict[str, Any],
    contract: dict[str, Any],
    *,
    observations: int,
    minimum_observations: int,
) -> dict[str, Any]:
    paper_standard = row.get("paper_promotion_standard") if isinstance(row.get("paper_promotion_standard"), dict) else {}
    training_policy = row.get("training_threshold_policy") if isinstance(row.get("training_threshold_policy"), dict) else {}
    usable_sample_goal = max(
        _safe_int(row.get("minimum_training_samples"), 0),
        _safe_int(paper_standard.get("minimum_samples"), 0),
        _safe_int(training_policy.get("minimum_samples"), 0),
        COLLECT_ONLY_USABLE_SAMPLE_GOAL,
    )
    eligible_sequence_goal = max(
        _safe_int(row.get("minimum_training_sequences"), 0),
        _safe_int(paper_standard.get("minimum_sequences"), 0),
        _safe_int(training_policy.get("minimum_sequences"), 0),
        COLLECT_ONLY_ELIGIBLE_SEQUENCE_GOAL,
    )
    estimated_capacity = min(
        usable_sample_goal,
        max(int(observations) // COLLECT_ONLY_OBSERVATIONS_PER_SAMPLE_TARGET, 0),
    )
    observation_gap = max(int(minimum_observations) - int(observations), 0)
    needs_label_materialization = bool(observations > 0 and estimated_capacity < usable_sample_goal)
    if observation_gap > 0 and needs_label_materialization:
        status = "collect_and_materialize_label_depth"
        next_action = "collect_more_raw_observations_while_materializing_point_in_time_label_depth"
    elif observation_gap > 0:
        status = "collect_more_observations"
        next_action = "collect_more_raw_observations_before_training"
    elif needs_label_materialization:
        status = "materialize_label_depth"
        next_action = "materialize_point_in_time_label_depth_from_existing_observations"
    else:
        status = "label_depth_ready_for_real_diagnostic_refresh"
        next_action = "refresh_with_real_samples"
    label_family = str(contract.get("label_family") or "generic_directional")
    primary_horizon = str(contract.get("primary_horizon") or "1d_forward_return")
    return {
        "version": "collect_only_label_depth_bridge_v1",
        "status": status,
        "next_action": next_action,
        "label_family": label_family,
        "primary_horizon": primary_horizon,
        "true_sample_count": 0,
        "usable_sample_goal": usable_sample_goal,
        "estimated_usable_sample_capacity": estimated_capacity,
        "usable_sample_gap": max(usable_sample_goal - estimated_capacity, 0),
        "eligible_sequence_goal": eligible_sequence_goal,
        "eligible_sequence_gap": eligible_sequence_goal,
        "observation_count": int(observations),
        "minimum_observations": int(minimum_observations),
        "observation_gap": observation_gap,
        "observations_per_usable_sample_target": COLLECT_ONLY_OBSERVATIONS_PER_SAMPLE_TARGET,
        "needs_more_raw_observations": observation_gap > 0,
        "needs_label_materialization": needs_label_materialization,
        "required_depth_events": [
            "accepted_candidate_trace",
            "rejected_candidate_trace",
            "abstained_candidate_trace",
            "counterfactual_opportunity_trace",
            "paper_live_outcome",
            "forward_return_bucket",
            "side_specific_outcome",
            "sample_eligibility_reason",
        ],
        "required_join_keys": ["bot_id", "symbol", "mode", "timestamp_utc", "snapshot_id", "decision_id"],
        "collection_actions": [
            "route every accepted, rejected, and abstained candidate through label_outcome_join",
            "write sample_eligibility_reason for filtered rows so conversion failures are visible",
            "keep neutral and counter-side examples instead of dropping them before training",
            "materialize side_specific_outcome and lane_balance_bucket before the next canary",
        ],
        "safe_training_policy": "diagnostic_only_until_real_sample_count_and_eligible_sequences_clear",
    }


def _collect_only_diagnostic_payload(row: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    bot_id = str(row.get("bot_id") or "").strip()
    contract = row.get("universal_label_contract") if isinstance(row.get("universal_label_contract"), dict) else {}
    if not contract:
        contract = _universal_contract(row)
    observations = _observation_count_for_row(row)
    minimum_observations = _minimum_observations_for_row(row)
    collection_age_days = _collection_age_days_for_row(row, now)
    minimum_collection_days = _minimum_collection_days_for_row(row)
    observations_ready = bool(minimum_observations > 0 and observations >= minimum_observations)
    days_ready = bool(minimum_collection_days > 0 and collection_age_days >= minimum_collection_days)
    required_context = list(contract.get("required_context") or [])
    required_labels = list(contract.get("required_labels") or [])
    label_family = str(contract.get("label_family") or "generic_directional")
    primary_horizon = str(contract.get("primary_horizon") or "1d_forward_return")
    label_depth = _label_depth_contract(
        row,
        contract,
        observations=observations,
        minimum_observations=minimum_observations,
    )
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "status": "collect_only_label_contract_ready",
        "label_depth_status": label_depth["status"],
        "bot_id": bot_id,
        "bot_role": str(row.get("bot_role") or ""),
        "lifecycle_state": str(row.get("lifecycle_state") or ""),
        "training_excluded": True,
        "data_collection_active": bool(row.get("data_collection_active", False)),
        "sample_count": 0,
        "observation_count": observations,
        "eligible_sequences": 0,
        "sequence_count": 0,
        "positive_rate": 0.0,
        "skipped_filtered": 0,
        "skipped_low_confidence": 0,
        "skipped_labels": 0,
        "metrics": {
            "acted_coverage": -1.0,
            "acted_accuracy": -1.0,
            "accuracy_lift_over_majority": 0.0,
            "long_precision": 0.0,
            "short_precision": 0.0,
            "label_balance_score": 0.0,
            "precision_balance_score": 0.0,
            "long_acted_count": 0,
            "short_acted_count": 0,
            "test_accuracy": 0.0,
            "quality_score": _safe_float(row.get("quality_score"), 0.0),
        },
        "runtime_meta": {
            "sample_count": 0,
            "observation_count": observations,
            "eligible_sequences": 0,
            "positive_rate": 0.0,
            "label_contract": contract,
            "training_label_contract": contract,
            "label_audit": {
                "label_family": label_family,
                "primary_horizon": primary_horizon,
                "required_context": required_context,
                "required_labels": required_labels,
                "required_join_mode": str(contract.get("required_join_mode") or "point_in_time_only"),
                "forbidden_join_modes": list(contract.get("forbidden_join_modes") or []),
                "label_contract_complete": bool(label_family and primary_horizon and required_context),
                "point_in_time_only": str(contract.get("required_join_mode") or "point_in_time_only") == "point_in_time_only",
            },
            "collection_threshold": {
                "minimum_observations": minimum_observations,
                "current_observations": observations,
                "observations_remaining": max(minimum_observations - observations, 0),
                "observations_ready": observations_ready,
                "minimum_collection_days": minimum_collection_days,
                "collection_age_days": collection_age_days,
                "days_remaining": max(round(minimum_collection_days - collection_age_days, 3), 0.0),
                "days_ready": days_ready,
                "training_ready": bool(observations_ready and days_ready),
            },
            "diagnostic_kind": "collect_only_label_contract_bootstrap",
            "label_depth_contract": label_depth,
            "usable_sample_bridge": {
                "true_sample_count": 0,
                "estimated_usable_sample_capacity": label_depth["estimated_usable_sample_capacity"],
                "usable_sample_goal": label_depth["usable_sample_goal"],
                "usable_sample_gap": label_depth["usable_sample_gap"],
                "policy": "do_not_count_estimated_capacity_as_real_training_samples",
            },
            "safe_next_step": label_depth["next_action"],
        },
        "diagnostic_contract": {
            "purpose": "Make collect-only high-numbered bots explain their label contract and collection threshold before canary training.",
            "no_training_started": True,
            "no_master_update": True,
            "no_execution": True,
            "protected_volumes": ["/Volumes/VIDEO"],
        },
    }


def _materialize_collect_only_diagnostics(
    project_root: Path,
    rows: list[dict[str, Any]],
    *,
    min_version: int,
    limit: int,
    overwrite: bool,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    diagnostics_dir = project_root / "governance" / "training_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    candidates = _collect_only_diagnostic_candidates(rows, min_version=min_version)
    selected = candidates[:limit] if limit and limit > 0 else candidates
    written: list[str] = []
    skipped_existing: list[str] = []
    for row in selected:
        bot_id = str(row.get("bot_id") or "").strip()
        out_path = diagnostics_dir / f"{bot_id}_latest.json"
        if out_path.exists():
            existing = _load_json(out_path)
            existing_runtime_meta = existing.get("runtime_meta") if isinstance(existing.get("runtime_meta"), dict) else {}
            existing_is_collect_only = bool(
                str(existing.get("status") or "").strip().lower() == "collect_only_label_contract_ready"
                or str(existing_runtime_meta.get("diagnostic_kind") or "").strip().lower()
                == "collect_only_label_contract_bootstrap"
            )
            lifecycle_state = str(row.get("lifecycle_state") or "").strip().lower()
            if not overwrite or (lifecycle_state != "data_collection_only" and not existing_is_collect_only):
                skipped_existing.append(bot_id)
                continue
        _write_json(out_path, _collect_only_diagnostic_payload(row, now=now))
        written.append(bot_id)
    rollup = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "min_bot_version": int(min_version),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "written_count": len(written),
        "skipped_existing_count": len(skipped_existing),
        "written_bot_ids": written[:250],
        "skipped_existing_bot_ids": skipped_existing[:250],
        "policy": "diagnostics_only_no_training_no_master_update",
    }
    _write_json(project_root / "governance" / "training_labeling_intelligence" / "collect_only_diagnostics_latest.json", rollup)
    return rollup


def _collect_only_diagnostic_candidates(rows: list[dict[str, Any]], *, min_version: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        bot_id = str(row.get("bot_id") or "").strip()
        version = _version_from_bot_id(bot_id)
        if version is None or version < int(min_version):
            continue
        if not bool(row.get("active", False)) or not bool(row.get("data_collection_active", False)):
            continue
        lifecycle_state = str(row.get("lifecycle_state") or "").strip()
        if lifecycle_state not in {"data_collection_only", "paper_live_data"}:
            continue
        if lifecycle_state != "data_collection_only" and not (
            bool(row.get("training_excluded", False)) or bool(row.get("exclude_from_training", False))
        ):
            continue
        candidates.append(row)
    candidates.sort(key=lambda item: _version_from_bot_id(str(item.get("bot_id") or "")) or 0)
    return candidates


def _collect_only_diagnostic_preview(
    project_root: Path,
    rows: list[dict[str, Any]],
    *,
    min_version: int,
    limit: int,
) -> dict[str, Any]:
    diagnostics_dir = project_root / "governance" / "training_diagnostics"
    candidates = _collect_only_diagnostic_candidates(rows, min_version=min_version)
    selected = candidates[:limit] if limit and limit > 0 else candidates
    existing: list[str] = []
    missing: list[str] = []
    for row in selected:
        bot_id = str(row.get("bot_id") or "").strip()
        if (diagnostics_dir / f"{bot_id}_latest.json").exists():
            existing.append(bot_id)
        else:
            missing.append(bot_id)
    return {
        "schema_version": 1,
        "mode": "preview",
        "min_bot_version": int(min_version),
        "limit": int(limit),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "existing_diagnostic_count": len(existing),
        "missing_diagnostic_count": len(missing),
        "missing_bot_ids": missing[:250],
        "existing_bot_ids": existing[:250],
        "recommended_apply_command": (
            "./scripts/ops/opsctl.sh training-labeling-intelligence --apply "
            "--materialize-collect-only-diagnostics "
            f"--collect-only-diagnostic-min-version {int(min_version)} "
            f"--collect-only-diagnostic-limit {int(limit)} --json"
        ),
        "policy": "diagnostics_only_no_training_no_master_update",
    }


def _row_for_bot(bot: dict[str, Any], bot_id: str, assigned_ids: dict[str, str], now: str) -> dict[str, Any]:
    system = _system(bot)
    contract = _pack_contract(assigned_ids)
    system_slug = str(system["slug"])
    layer = str(system["layer"])
    label_contract = _with_free_source_context({
        "version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "label_family": "training_process_quality",
        "primary_horizon": f"{system_slug}_improves_training_or_labeling_gate_status",
        "aux_horizons": ["coverage_gap_delta", "label_quality_delta", "runtime_failure_delta"],
        "required_context": [*BASE_DATA_INTAKES, f"{system_slug}_effect_trace"],
        "required_labels": list(REQUIRED_LABELS),
        "required_join_mode": "point_in_time_only",
        "forbidden_join_modes": ["future_leakage", "lookahead_join", "unbounded_raw_feed_join"],
        "quality_floor": 0.89,
        "training_lane": "governance_effect",
    })
    return {
        "bot_id": bot_id,
        "bot_role": bot["bot_role"],
        "active": True,
        "reason": "training_labeling_intelligence_expansion_slot",
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
        "promotion_reason": "training_labeling_intelligence_expansion_slot",
        "model_path": "",
        "log_file": "",
        "candidate_log_file": "",
        "lifecycle_state": "data_collection_only",
        "slot_label": bot["label"],
        "slot_kind": _slot_kind(bot),
        "slot_priority": bot["priority"],
        "slot_objective": bot["objective"],
        "target_functions": list(bot["target_functions"]),
        "preferred_regimes": ["protective_pressure", "coverage_repair", "label_audit", "schema_gate_repair", "off_hours_targeted_retrain"],
        "bootstrap_teacher_bot_ids": [
            "brain_refinery_v1",
            "brain_refinery_v1522_quant_operational_backlog_outcome_verifier_telemetry_collector_bot",
            "brain_refinery_v1554_quant_operational_operator_decision_packet_builder_telemetry_collector_bot",
            "brain_refinery_v1562_autonomic_governance_sleeve_budget_market_telemetry_collector_bot",
        ],
        "data_intake_collections": [*BASE_DATA_INTAKES, f"{system_slug}_effect_trace", f"{system_slug}_label_quality_trace"],
        "storage_targets": ["governance/training_labeling_intelligence", f"governance/training_labeling_intelligence/{system_slug}", "governance/health/training_labeling_intelligence_latest.json"],
        "freshness_slo_seconds": 1800,
        "retention_profile": "training_labeling_hot_3d_warm_120d_cold_540d",
        "data_collection_active": True,
        "data_collection_started_utc": now,
        "data_collection_observations": 0,
        "data_collection_mode": "active_observer",
        "data_collection_reason": "training_labeling_intelligence_collect_only_until_label_and_training_effect_gates_clear",
        "trading_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "allocation_enabled": False,
        "execution_enabled": False,
        "rotation_blocked": True,
        "rotation_block_reason": "training_labeling_intelligence_collection_only_zero_weight",
        "training_excluded": True,
        "exclude_from_training": True,
        "training_candidate_after_threshold": True,
        "training_exclusion_reason": "collecting_training_labeling_effect_evidence_before_training",
        "training_exclusion_until": "minimum_data_collection_threshold_met",
        "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
        "minimum_data_collection_days": MINIMUM_COLLECTION_DAYS,
        "training_threshold_policy": {
            "minimum_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_collection_days": MINIMUM_COLLECTION_DAYS,
            "requires_label_contract_clearance": True,
            "requires_runtime_pressure_clearance": True,
            "requires_backpressure_clearance": True,
            "requires_schema_lineage_clearance": True,
            "requires_paper_live_separation_clearance": True,
            "requires_global_halt_clear": True,
        },
        "data_collection_storage_guarded": True,
        "data_collection_capture_mode": "thin_digest_with_heartbeat_fallback",
        "data_collection_sample_rate": SAMPLE_RATE,
        "data_collection_max_daily_storage_mb": MAX_DAILY_MB_PER_BOT,
        "data_collection_max_daily_mb": float(MAX_DAILY_MB_PER_BOT),
        "data_collection_compute_guard_mode": "pressure_self_accommodating",
        "self_accommodating_policy": {
            "steady": "thin_digest",
            "protective": "heartbeat",
            "critical": "parked_until_operator_review",
            "raw_trace_allowed": False,
        },
        "data_collection_threshold_progress": _threshold_progress(),
        "data_collection_training_ready": False,
        "sleeve_profile": system_slug,
        "sleeve_family": SLEEVE_FAMILY,
        "training_labeling_layer": layer,
        "intelligence_system": system_slug,
        "strategy_family": "training_and_labeling_governance",
        "correlation_peer_sleeves": ["whole_system_governor", "autonomic_governance_mesh", "quant_operational_intelligence", "system_self_model", "codex_handoff"],
        "correlation_dependencies": ["training_quality_control", "training_runtime_control", "coverage_gap_closer", "schema_migration_guard", "feature_store_manifest", "promotion_quality_gate"],
        "direct_market_data_available": False,
        "direct_execution_allowed": False,
        "proxy_data_sources": ["master_bot_registry", "governance_health", "training_quality_control", "coverage_gap_closer", "feature_store_manifest", "codex_handoff"],
        "label_contract": label_contract,
        "universal_label_contract": label_contract,
        "data_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "universal_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "training_labeling_intelligence_version": PACK_VERSION,
        "training_lane": "governance_effect",
        "training_label_contract_status": "pack_native",
        "labeling_tags": [
            "research_only",
            "collection_only",
            "execution_blocked",
            "universal_label_contract",
            "point_in_time_only",
            f"sleeve_family:{SLEEVE_FAMILY}",
            f"training_labeling_layer:{layer}",
            "label_family:training_process_quality",
            "training_lane:governance_effect",
        ],
        "execution_policy_label": "collection_only_training_labeling_intelligence_no_execution",
        "eligible_for_master_vote": False,
        "founder_bot_id": "brain_refinery_v1",
        "founder_dna_version": "founder_dna_v1",
        "founder_dna_traits": ["paper_first_safety", "global_halt_awareness", "resource_throttle_awareness", "decision_explanation_contract", "registry_auditable_identity", "point_in_time_labeling"],
        "founder_dna_applied_utc": now,
        "lineage_root_bot_id": "brain_refinery_v1",
        "lineage_guard_enabled": True,
        "lineage_revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "paper_trade_lock_required": True,
        "capability_pack_version": PACK_VERSION,
        "capability_pack_slug": PACK_SLUG,
        "capability_pack_display_name": PACK_DISPLAY_NAME,
        "capability_pack_contract": contract,
        "training_labeling_intelligence_contract": {
            "contract_version": "training_labeling_intelligence_layers_v1",
            "capability_pack": PACK_SLUG,
            "training_labeling_layer": layer,
            "intelligence_system": system_slug,
            "system_display_name": system["display_name"],
            "system_outputs": list(system["outputs"]),
            "authority_boundary": "collection_only_advisory_no_execution_no_allocation_no_halt_clearance",
        },
    }


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


def _pack_summary(assigned_ids: dict[str, str]) -> dict[str, Any]:
    contract = _pack_contract(assigned_ids)
    return {
        "slug": PACK_SLUG,
        "display_name": PACK_DISPLAY_NAME,
        "sleeve_family": SLEEVE_FAMILY,
        "objective": "Add a collect-only intelligence layer for label contracts, point-in-time labeling, lane-balanced retrain planning, coverage repair, schema lineage, and retrain outcome memory.",
        "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
        "system_count": len(INTELLIGENCE_SYSTEMS),
        "bot_count": len(BOTS),
        "bot_ids": [assigned_ids[_slot_kind(bot)] for bot in BOTS],
        "intelligence_systems": list(INTELLIGENCE_SYSTEMS),
        "dedicated_data_intake": list(BASE_DATA_INTAKES),
        "storage_retention_rule": contract["storage_retention_rule"],
        "paper_only_floor": contract["paper_only_floor"],
        "anchor_bot_ids": contract["anchor_bot_ids"],
        "universal_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
    }


def _training_process_intelligence(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    walk_forward_root = project_root / "governance" / "walk_forward"
    training_quality = _load_json(health_root / "training_quality_control_latest.json")
    runtime_control = _load_json(health_root / "training_runtime_control_latest.json")
    coverage_seed = _load_first_json(
        [
            walk_forward_root / "coverage_seed_latest.json",
            health_root / "walk_forward_coverage_seed_latest.json",
        ]
    )
    coverage_gap = _load_first_json(
        [
            walk_forward_root / "coverage_gap_closer_latest.json",
            health_root / "coverage_gap_closer_latest.json",
        ]
    )
    schema_compat = _load_json(health_root / "retrain_schema_compatibility_latest.json")
    schema_migration = _load_json(health_root / "schema_migration_guard_latest.json")
    feature_store = _load_json(project_root / "governance" / "feature_store" / "latest.json")
    lineage = _load_json(health_root / "training_lineage_manifest_latest.json")
    targeted_actions = training_quality.get("targeted_actions") if isinstance(training_quality.get("targeted_actions"), dict) else {}
    normal_targets = _ordered_unique(targeted_actions.get("targeted_retrain_bot_ids") if isinstance(targeted_actions.get("targeted_retrain_bot_ids"), list) else [])
    stage_candidates = coverage_gap.get("active_stage_candidates") if isinstance(coverage_gap.get("active_stage_candidates"), list) else []
    seed_rows = coverage_seed.get("seed_queue") if isinstance(coverage_seed.get("seed_queue"), list) else []
    coverage_targets = _ordered_unique(
        [str(row.get("bot_id") or "") for row in stage_candidates if isinstance(row, dict)]
        or [
            str(row.get("bot_id") or "")
            for row in seed_rows
            if isinstance(row, dict)
            and (
                bool(row.get("needs_runtime_input_repair", False))
                or bool(row.get("needs_diagnostic_refresh", False))
                or "targeted_retrain" in [str(action or "") for action in row.get("actions") or []]
            )
        ]
    )
    precompute_targets = runtime_control.get("precompute_targets") if isinstance(runtime_control.get("precompute_targets"), list) else []
    precompute_ids = _ordered_unique([str(row.get("bot_id") or "") for row in precompute_targets if isinstance(row, dict)])
    launch_contract = (coverage_gap.get("autopilot_contract") or {}).get("launch_contract") if isinstance(coverage_gap.get("autopilot_contract"), dict) else {}
    autopilot_blocking_reasons = []
    if isinstance(coverage_gap.get("autopilot_contract"), dict):
        raw_blocking_reasons = (coverage_gap.get("autopilot_contract") or {}).get("blocking_reasons")
        autopilot_blocking_reasons = raw_blocking_reasons if isinstance(raw_blocking_reasons, list) else []
    blocked_reasons = _ordered_unique(
        [
            *_ordered_unique(autopilot_blocking_reasons),
            "training_quality_blocked" if str(training_quality.get("overall_status") or "") == "blocked" else "",
            "schema_migration_guard_blocked" if schema_migration and not bool(schema_migration.get("ok", False)) else "",
            "lineage_manifest_not_ready" if lineage and str(lineage.get("overall_status") or "") in {"blocked", "needs_attention"} else "",
        ]
    )
    selected_targets = coverage_targets or precompute_ids or normal_targets
    return {
        "process_version": "training_process_intelligence_v1",
        "normal_targeted_retrain_bot_ids": normal_targets,
        "coverage_repair_bot_ids": coverage_targets,
        "precompute_target_bot_ids": precompute_ids[:12],
        "selected_targeted_retrain_bot_ids": selected_targets[:12],
        "selected_target_source": "coverage_repair" if coverage_targets else "runtime_precompute" if precompute_ids else "normal_targeted_shortlist",
        "recommended_retrain_profile": "coverage_canary" if coverage_targets else "lane_specific",
        "blocked_reasons": blocked_reasons,
        "launch_contract": launch_contract if isinstance(launch_contract, dict) else {},
        "quality_snapshot": {
            "overall_status": training_quality.get("overall_status"),
            "training_quality_score": training_quality.get("training_quality_score") or training_quality.get("training_quality_index"),
            "top_priorities": training_quality.get("top_priorities") if isinstance(training_quality.get("top_priorities"), list) else [],
        },
        "runtime_snapshot": {
            "overall_status": runtime_control.get("overall_status"),
            "snapshot_ready": runtime_control.get("snapshot_ready"),
            "resource_guard": runtime_control.get("resource_guard") if isinstance(runtime_control.get("resource_guard"), dict) else {},
        },
        "schema_and_lineage": {
            "schema_compatibility_status": schema_compat.get("overall_status") or schema_compat.get("status"),
            "schema_migration_status": schema_migration.get("overall_status") or schema_migration.get("status"),
            "feature_store_ok": feature_store.get("ok"),
            "lineage_status": lineage.get("overall_status"),
        },
        "safe_preflight_order": [
            "./scripts/ops/opsctl.sh schema-migration --json",
            "./scripts/ops/opsctl.sh feature-store --json",
            "./scripts/ops/opsctl.sh training-label-audit --json",
            "./scripts/ops/opsctl.sh runtime-training-snapshot --json",
            "./scripts/ops/opsctl.sh coverage-gap-closer --apply-stage --json",
            "./scripts/ops/opsctl.sh training-runtime-control --json",
        ],
        "safe_targeted_retrain_template": [
            "./scripts/ops/opsctl.sh",
            "retrain-force-targeted",
            "--include-bot-ids",
            ",".join(selected_targets[:4]),
            "--retrain-profile",
            "coverage_canary",
            "--skip-master-update",
            "--runtime-train-use-snapshot",
            "--thread-cap",
            "1",
            "--memory-guard",
        ],
    }


def plan_registry_expansion(registry: dict[str, Any]) -> dict[str, Any]:
    rows = _registry_rows(registry)
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
        "training_labeling_intelligence_version": PACK_VERSION,
        "system_count": len(INTELLIGENCE_SYSTEMS),
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


def _refresh_summary(
    registry: dict[str, Any],
    label_summary: dict[str, Any],
    materialization_coverage: dict[str, Any] | None = None,
) -> None:
    rows = _registry_rows(registry)
    active = [row for row in rows if bool(row.get("active"))]
    inactive = [row for row in rows if not bool(row.get("active"))]
    signal_active = [row for row in active if str(row.get("bot_role") or "") == "signal_sub_bot"]
    infra_active = [row for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    data_collection_only = [row for row in rows if str(row.get("lifecycle_state") or "") == "data_collection_only"]
    zero_weight_research = [
        row
        for row in rows
        if _safe_float(row.get("weight"), 0.0) == 0.0
        and (
            str(row.get("lifecycle_state") or "") == "data_collection_only"
            or "research_only" in [str(tag or "") for tag in row.get("labeling_tags") or []]
        )
    ]
    structured = [row for row in rows if str(row.get("capability_pack_version") or "")]
    pack_rows = [row for row in rows if str(row.get("training_labeling_intelligence_version") or "") == PACK_VERSION and str(row.get("capability_pack_slug") or "") == PACK_SLUG]
    universal_rows = [row for row in rows if str(row.get("universal_label_contract_version") or "") == UNIVERSAL_LABEL_CONTRACT_VERSION]
    contract_rows = [row for row in rows if isinstance(row.get("label_contract"), dict) or str(row.get("data_label_contract_version") or "")]
    materialization_rows = [
        row
        for row in rows
        if str(row.get("training_label_materialization_contract_version") or "")
        == LABEL_MATERIALIZATION_CONTRACT_VERSION
    ]
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
            "data_collection_only_bots": len(data_collection_only),
            "zero_weight_research_bots": len(zero_weight_research),
            "training_excluded_bots": sum(1 for row in rows if bool(row.get("training_excluded")) or bool(row.get("exclude_from_training"))),
            "structured_capability_pack_bot_count": len(structured),
            "training_labeling_intelligence_bot_count": len(pack_rows),
            "latest_training_labeling_intelligence": PACK_VERSION,
            "training_label_contract_bot_count": len(contract_rows),
            "universal_label_contract_bot_count": len(universal_rows),
            "universal_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
            "training_label_contract_coverage_ratio": round(len(contract_rows) / max(len(rows), 1), 4),
            "training_label_contracts_normalized_latest": label_summary.get("updated_label_contract_bot_count", 0),
            "training_label_materialization_contract_version": LABEL_MATERIALIZATION_CONTRACT_VERSION,
            "training_label_materialization_contract_bot_count": len(materialization_rows),
            "training_label_materialization_contract_coverage_ratio": round(
                len(materialization_rows) / max(len(rows), 1), 4
            ),
            "training_label_objective_class_counts": dict(
                (materialization_coverage or {}).get("objective_class_counts") or {}
            ),
            "max_bot_version": max(versions) if versions else None,
            "target_platform_total_bots": TARGET_PLATFORM_TOTAL_BOTS,
            "target_platform_total_bots_met": len(rows) >= TARGET_PLATFORM_TOTAL_BOTS,
        }
    )
    registry["summary"] = summary


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    collect_only_diagnostic_min_version: int = DEFAULT_COLLECT_ONLY_DIAGNOSTIC_MIN_VERSION,
    collect_only_diagnostic_limit: int = 0,
) -> dict[str, Any]:
    registry = _load_json(project_root / "master_bot_registry.json")
    rows = _registry_rows(registry)
    plan = plan_registry_expansion(registry)
    missing_contracts = sum(1 for row in rows if not (_existing_contract(row) or row.get("data_label_contract_version")))
    incomplete_contracts = sum(1 for row in rows if (_existing_contract(row) or row.get("data_label_contract_version")) and not _contract_complete(row))
    source_enrichment = _free_label_source_enrichment(project_root, rows)
    materialization_plan = _label_materialization_plan(source_enrichment)
    all_bot_materialization = _compact_materialization_coverage(
        _all_bot_label_materialization_coverage(rows)
    )
    collection_guard = _training_labeling_collection_guard_preview(rows)
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
        "training_labeling_intelligence_version": PACK_VERSION,
        "universal_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "label_materialization_contract_version": LABEL_MATERIALIZATION_CONTRACT_VERSION,
        "system_count": plan["system_count"],
        "bot_count": plan["bot_count"],
        "planned_bot_count": plan["planned_bot_count"],
        "skipped_existing_count": plan["skipped_existing_count"],
        "missing_label_contract_count": missing_contracts,
        "incomplete_label_contract_count": incomplete_contracts,
        "pack": plan["pack"],
        "free_label_source_enrichment": source_enrichment,
        "label_materialization_plan": materialization_plan,
        "all_bot_label_materialization": all_bot_materialization,
        "training_labeling_collection_guard": collection_guard,
        "training_process_intelligence": _training_process_intelligence(project_root),
        "collect_only_diagnostics": _collect_only_diagnostic_preview(
            project_root,
            rows,
            min_version=collect_only_diagnostic_min_version,
            limit=collect_only_diagnostic_limit,
        ),
        "recommended_apply_command": "./scripts/ops/opsctl.sh training-labeling-intelligence --apply --json",
    }


def apply_registry(
    project_root: Path = PROJECT_ROOT,
    *,
    materialize_collect_only_diagnostics: bool = False,
    collect_only_diagnostic_min_version: int = DEFAULT_COLLECT_ONLY_DIAGNOSTIC_MIN_VERSION,
    collect_only_diagnostic_limit: int = 0,
    overwrite_collect_only_diagnostics: bool = False,
) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    registry = _load_json(registry_path)
    rows = _registry_rows(registry)
    plan = plan_registry_expansion(registry)
    added_rows = list(plan["planned_rows"])
    storage_targets_ready = _ensure_storage_targets(project_root)
    backup_dir = project_root / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"master_bot_registry_before_training_labeling_intelligence_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(registry_path, backup)
    now = _utc_now()
    if added_rows:
        rows.extend(added_rows)
    collection_guard = _apply_training_labeling_collection_guard(rows, now)
    materialization_before = _all_bot_label_materialization_coverage(rows)
    label_summary = _apply_universal_label_contracts(rows, now)
    materialization_coverage = _apply_bot_label_materialization_contracts(rows, now)
    materialization_coverage["misrouted_directional_infrastructure_before_count"] = int(
        materialization_before.get("misrouted_directional_infrastructure_before_count", 0) or 0
    )
    materialization_coverage["misrouted_directional_infrastructure_before_bot_ids"] = list(
        materialization_before.get("misrouted_directional_infrastructure_before_bot_ids") or []
    )
    materialization_coverage["misrouted_market_signal_guards_before_count"] = int(
        materialization_before.get("misrouted_market_signal_guards_before_count", 0) or 0
    )
    materialization_coverage["misrouted_market_signal_guards_before_bot_ids"] = list(
        materialization_before.get("misrouted_market_signal_guards_before_bot_ids") or []
    )
    registry["sub_bots"] = rows
    registry["updated_at_utc"] = now
    _refresh_summary(registry, label_summary, materialization_coverage)
    _write_json(registry_path, registry)

    process = _training_process_intelligence(project_root)
    source_enrichment = _free_label_source_enrichment(project_root, rows)
    materialization_plan = _label_materialization_plan(source_enrichment)
    collect_only_diagnostics = (
        _materialize_collect_only_diagnostics(
            project_root,
            rows,
            min_version=collect_only_diagnostic_min_version,
            limit=collect_only_diagnostic_limit,
            overwrite=overwrite_collect_only_diagnostics,
        )
        if materialize_collect_only_diagnostics
        else _collect_only_diagnostic_preview(
            project_root,
            rows,
            min_version=collect_only_diagnostic_min_version,
            limit=collect_only_diagnostic_limit,
        )
    )
    payload = build_payload(
        project_root,
        collect_only_diagnostic_min_version=collect_only_diagnostic_min_version,
        collect_only_diagnostic_limit=collect_only_diagnostic_limit,
    )
    payload.update(
        {
            "mode": "applied",
            "added_bot_count": len(added_rows),
            "added_bot_ids": [str(row.get("bot_id") or "") for row in added_rows],
            "backup_path": str(backup),
            "new_total_bots": len(rows),
            "new_active_bots": sum(1 for row in rows if bool(row.get("active"))),
            "target_platform_total_bots_met": len(rows) >= TARGET_PLATFORM_TOTAL_BOTS,
            "storage_targets_ready": storage_targets_ready,
            "label_contract_summary": label_summary,
            "all_bot_label_materialization": _compact_materialization_coverage(materialization_coverage),
            "free_label_source_enrichment": source_enrichment,
            "label_materialization_plan": materialization_plan,
            "training_labeling_collection_guard": collection_guard,
            "training_process_intelligence": process,
            "collect_only_diagnostics": collect_only_diagnostics,
        }
    )
    config_payload = {
        "generated_at_utc": _utc_now(),
        "training_labeling_intelligence_version": PACK_VERSION,
        "universal_label_contract_version": UNIVERSAL_LABEL_CONTRACT_VERSION,
        "label_materialization_contract_version": LABEL_MATERIALIZATION_CONTRACT_VERSION,
        "pack": payload["pack"],
    }
    _write_json(project_root / "config" / "training_labeling_intelligence_v1.json", config_payload)
    _write_json(
        project_root / "governance" / "health" / "training_labeling_intelligence_latest.json",
        _compact_health_payload(payload),
    )
    _write_json(project_root / "governance" / "training_labeling_intelligence" / "label_coverage_latest.json", label_summary)
    _write_json(project_root / "governance" / "training_labeling_intelligence" / "free_label_source_enrichment_latest.json", source_enrichment)
    _write_json(project_root / "governance" / "training_labeling_intelligence" / "label_materialization_plan_latest.json", materialization_plan)
    _write_json(
        project_root / "governance" / "training_labeling_intelligence" / "all_bot_label_materialization_latest.json",
        materialization_coverage,
    )
    _write_json(project_root / "governance" / "training_labeling_intelligence" / "training_process_intelligence_latest.json", process)
    return payload


def refresh_artifacts(
    project_root: Path = PROJECT_ROOT,
    *,
    collect_only_diagnostic_min_version: int = DEFAULT_COLLECT_ONLY_DIAGNOSTIC_MIN_VERSION,
    collect_only_diagnostic_limit: int = 0,
) -> dict[str, Any]:
    payload = build_payload(
        project_root,
        collect_only_diagnostic_min_version=collect_only_diagnostic_min_version,
        collect_only_diagnostic_limit=collect_only_diagnostic_limit,
    )
    payload["mode"] = "refreshed_artifacts"
    registry = _load_json(project_root / "master_bot_registry.json")
    detailed_materialization = _all_bot_label_materialization_coverage(_registry_rows(registry))
    _write_json(
        project_root / "governance" / "health" / "training_labeling_intelligence_latest.json",
        _compact_health_payload(payload),
    )
    _write_json(
        project_root / "governance" / "training_labeling_intelligence" / "free_label_source_enrichment_latest.json",
        dict(payload.get("free_label_source_enrichment") or {}),
    )
    _write_json(
        project_root / "governance" / "training_labeling_intelligence" / "label_materialization_plan_latest.json",
        dict(payload.get("label_materialization_plan") or {}),
    )
    _write_json(
        project_root / "governance" / "training_labeling_intelligence" / "all_bot_label_materialization_latest.json",
        detailed_materialization,
    )
    _write_json(
        project_root / "governance" / "training_labeling_intelligence" / "training_process_intelligence_latest.json",
        dict(payload.get("training_process_intelligence") or {}),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Improve training process intelligence and normalize universal label contracts.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--refresh-artifacts", action="store_true")
    parser.add_argument("--materialize-collect-only-diagnostics", action="store_true")
    parser.add_argument("--collect-only-diagnostic-min-version", type=int, default=DEFAULT_COLLECT_ONLY_DIAGNOSTIC_MIN_VERSION)
    parser.add_argument("--collect-only-diagnostic-limit", type=int, default=0)
    parser.add_argument("--overwrite-collect-only-diagnostics", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.apply and args.refresh_artifacts:
        parser.error("--apply and --refresh-artifacts are mutually exclusive")

    project_root = Path(args.project_root).resolve()
    payload = (
        apply_registry(
            project_root,
            materialize_collect_only_diagnostics=bool(args.materialize_collect_only_diagnostics),
            collect_only_diagnostic_min_version=int(args.collect_only_diagnostic_min_version),
            collect_only_diagnostic_limit=int(args.collect_only_diagnostic_limit),
            overwrite_collect_only_diagnostics=bool(args.overwrite_collect_only_diagnostics),
        )
        if args.apply
        else refresh_artifacts(
            project_root,
            collect_only_diagnostic_min_version=int(args.collect_only_diagnostic_min_version),
            collect_only_diagnostic_limit=int(args.collect_only_diagnostic_limit),
        )
        if args.refresh_artifacts
        else build_payload(
            project_root,
            collect_only_diagnostic_min_version=int(args.collect_only_diagnostic_min_version),
            collect_only_diagnostic_limit=int(args.collect_only_diagnostic_limit),
        )
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        diagnostics = payload.get("collect_only_diagnostics") if isinstance(payload.get("collect_only_diagnostics"), dict) else {}
        print(
            "training_labeling_intelligence "
            f"mode={payload['mode']} systems={payload['system_count']} bots={payload['bot_count']} "
            f"planned={payload['planned_bot_count']} added={payload.get('added_bot_count', 0)} "
            f"missing_labels={payload.get('missing_label_contract_count', 0)} "
            f"selected_targets={len(payload['training_process_intelligence']['selected_targeted_retrain_bot_ids'])} "
            f"collect_only_diagnostics={diagnostics.get('written_count', diagnostics.get('missing_diagnostic_count', 0))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
