#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "training_data_intake_expansion_latest.json"
DEFAULT_FOCUS_PATH = PROJECT_ROOT / "governance" / "training_labeling_intelligence" / "data_intake_focus_latest.json"
DEFAULT_BOT_NEEDS_PATH = PROJECT_ROOT / "governance" / "health" / "bot_needs_intelligence_latest.json"
DEFAULT_TRAINING_QUALITY_PATH = PROJECT_ROOT / "governance" / "health" / "training_quality_control_latest.json"
DEFAULT_PAPER_PROFITABILITY_CONTROL_PATH = PROJECT_ROOT / "governance" / "health" / "paper_profitability_control_latest.json"

SAMPLE_FLOOR = 200
USABLE_SAMPLE_GOAL = 240
ELIGIBLE_SEQUENCE_FLOOR = 4
ELIGIBLE_SEQUENCE_GOAL = 8
OBSERVATION_FLOOR_DEFAULT = SAMPLE_FLOOR

BASE_CONTEXT_BY_LABEL_FAMILY: dict[str, list[str]] = {
    "generic_directional": [
        "price_bars",
        "volume",
        "market_context",
        "sector_context",
        "market_breadth",
        "realized_volatility",
        "source_quality",
        "execution_quality",
    ],
    "intraday_fast": [
        "one_minute_bars",
        "vwap",
        "spread_quality",
        "relative_volume",
        "order_flow",
        "market_micro_context",
        "auction_imbalance_context",
        "execution_latency",
    ],
    "multi_day": [
        "daily_bars",
        "sector_context",
        "macro_context",
        "overnight_gap",
        "market_breadth",
        "realized_volatility",
        "earnings_calendar",
        "source_quality",
    ],
    "options_surface": [
        "options_chain",
        "iv_surface",
        "open_interest",
        "bid_ask_spread",
        "greeks",
        "skew",
        "earnings_calendar",
        "realized_volatility",
    ],
    "income_options_surface": [
        "ex_dividend_calendar",
        "payout_metrics",
        "options_chain",
        "iv_surface",
        "open_interest",
        "bid_ask_spread",
        "earnings_calendar",
        "realized_volatility",
    ],
    "income_total_return": [
        "ex_dividend_calendar",
        "payout_metrics",
        "balance_sheet_quality",
        "earnings_calendar",
        "rate_context",
        "sector_context",
        "realized_volatility",
        "source_quality",
    ],
    "fixed_income_rates": [
        "rates_curve",
        "duration_context",
        "inflation_context",
        "macro_calendar",
        "credit_stress",
        "liquidity_state",
        "realized_volatility",
        "source_quality",
    ],
    "credit_spread": [
        "credit_spread_context",
        "rates_curve",
        "sector_context",
        "market_breadth",
        "realized_volatility",
        "liquidity_state",
        "source_quality",
        "execution_quality",
    ],
    "execution_cost_quality": [
        "spread_quality",
        "execution_quality",
        "slippage_trace",
        "fill_quality",
        "latency_trace",
        "market_micro_context",
        "source_quality",
        "paper_live_outcome",
    ],
    "correlation_risk_effect": [
        "correlation_matrix",
        "cross_asset_correlation",
        "sector_context",
        "portfolio_exposure",
        "risk_budget",
        "market_breadth",
        "realized_volatility",
        "paper_live_outcome",
    ],
    "crypto_microstructure": [
        "crypto_order_book",
        "crypto_funding",
        "basis",
        "exchange_liquidity",
        "cross_asset_correlation",
        "market_micro_context",
        "source_quality",
        "execution_quality",
    ],
    "futures_event_session": [
        "futures_curve",
        "session_context",
        "macro_calendar",
        "basis",
        "volume_profile",
        "market_micro_context",
        "source_quality",
    ],
    "operational_guard_effect": [
        "runtime_health",
        "incident_log",
        "operator_context",
        "backlog_pressure",
        "memory_pressure",
        "storage_latency",
        "process_state",
        "guard_action_outcome",
    ],
}

DEFAULT_RESEARCH_CONTEXT = [
    "research_evidence",
    "feature_lineage",
    "replay_context",
    "experiment_ledger",
    "source_quality",
    "paper_live_outcome",
]

ADVANCED_QUANT_COLLECTION_SECTIONS: dict[str, list[str]] = {
    "registry_identity": [
        "registry_slot_metadata",
        "provider_capability_profile",
        "data_intake_collections_manifest",
        "collection_threshold_progress",
    ],
    "proxy_context": [
        "source_quality",
        "provider_freshness",
        "cross_provider_agreement",
        "proxy_data_source_lineage",
    ],
    "feature_surface": [
        "quant_model_feature_surface",
        "feature_confidence_matrix",
        "model_price_sensitivity_grid",
        "runtime_feature_history",
    ],
    "research_labels": [
        "label_contract_quality",
        "point_in_time_label_quality",
        "proxy_label_quality",
        "label_outcome_join",
    ],
    "training_gate": [
        "walk_forward_trace",
        "promotion_gate_trace",
        "training_runtime_snapshot",
        "non_regression_marker",
    ],
}

ADVANCED_QUANT_CONTEXT_BY_LABEL_FAMILY: dict[str, list[str]] = {
    "quant_research_control": [
        "quant_model_control_trace",
        "resource_profile",
        "feature_store_lineage_trace",
        "label_contract_quality",
        "model_resource_pressure",
    ],
    "quant_pricing_research": [
        "listed_option_surface",
        "realized_vol",
        "rates_context",
        "quant_model_feature_surface",
        "model_price_sensitivity_grid",
        "pricing_model_dispersion",
        "quantlib_pricing_benchmark",
    ],
    "state_space_filter_research": [
        "runtime_feature_history",
        "market_micro_features",
        "state_filter_diagnostics",
        "regime_transition_trace",
        "latent_state_confidence",
    ],
    "optimization_research": [
        "optimization_search_trace",
        "constraint_violation_trace",
        "objective_value_trace",
        "portfolio_fit_trace",
        "kelly_fraction_trace",
    ],
    "transaction_cost_slippage_research": [
        "transaction_cost_surface",
        "fill_realism_context",
        "queue_position_context",
        "market_impact_proxy",
        "latency_trace",
    ],
    "transport_topology_research": [
        "optimal_transport_bridge_trace",
        "topological_regime_shape",
        "cross_asset_distance_matrix",
        "regime_shape_persistence",
    ],
    "signature_hawkes_generators": [
        "signature_path_trace",
        "hawkes_self_exciting_trace",
        "event_cluster_trace",
        "path_lead_lag_trace",
    ],
    "qemc_path_volatility_research": [
        "quantum_enhanced_mc_trace",
        "functional_ito_path_trace",
        "rough_volatility_fbm_trace",
        "path_dependent_volatility",
    ],
    "xva_counterparty_margin_research": [
        "xva_exposure_ladder",
        "collateral_margin_waterfall",
        "isda_simm_proxy_grid",
        "counterparty_credit_context",
    ],
    "credit_derivatives_research": [
        "credit_spread_surface",
        "hazard_rate_proxy",
        "recovery_rate_proxy",
        "credit_equity_linkage_trace",
    ],
    "securitized_products_research": [
        "mbs_abs_clo_proxy_universe",
        "prepayment_oas_proxy",
        "loan_pool_credit_proxy",
        "duration_convexity_trace",
    ],
    "repo_securities_lending_research": [
        "repo_funding_curve_proxy",
        "securities_lending_borrow_fee_proxy",
        "short_interest_locate_pressure",
        "funding_liquidity_context",
    ],
    "market_data_tape_normalization_research": [
        "opra_nbbo_taq_sip_normalized_events",
        "mbo_mbp_depth_snapshot",
        "dark_pool_off_exchange_volume",
        "feed_latency_schema_health",
    ],
    "limit_order_book_transformers": [
        "mbo_mbp_depth_snapshot",
        "lob_sequence_tensor",
        "queue_position_context",
        "market_micro_features",
    ],
    "lobdif_crisis_microstructure_research": [
        "lobdif_crisis_microstructure_trace",
        "flash_freeze_slippage_trace",
        "toxic_liquidity_injection_trace",
        "market_micro_features",
    ],
    "rlbf_dms_equivariant_research": [
        "rlbf_feedback_trace",
        "differentiable_market_simulator_trace",
        "equivariant_network_trace",
        "policy_backtracking_trace",
    ],
    "proof_quantum_formal_backend_research": [
        "formal_verification_safety_trace",
        "proof_backend_readiness",
        "quantum_backend_readiness",
        "regression_guard_verdict",
    ],
    "model_risk_validation_research": [
        "model_risk_validation_trace",
        "benchmark_model_comparison",
        "overfit_gap_trace",
        "non_regression_marker",
    ],
    "neural_sde_kan_hedging": [
        "neural_sde_stability_trace",
        "kan_hedging_confidence_trace",
        "hedge_cost_trace",
        "pathwise_greek_trace",
    ],
    "gpu_quant_acceleration": [
        "gpu_runtime_profile",
        "mlx_metal_profile",
        "cuda_rocm_profile",
        "quant_model_resource_pressure",
    ],
    "order_flow_toxicity_research": [
        "vpin_order_flow_toxicity",
        "toxic_flow_bucket",
        "spread_queue_response",
        "market_micro_features",
    ],
    "cross_asset_basis_research": [
        "basis",
        "basis_context",
        "cross_asset_correlation",
        "funding_context",
        "proxy_data_source_lineage",
    ],
}

ADVANCED_QUANT_LABEL_FAMILIES = set(ADVANCED_QUANT_CONTEXT_BY_LABEL_FAMILY)

ADVANCED_QUANT_SAMPLE_TARGETS: dict[str, dict[str, int]] = {
    "quant_research_control": {"sample": 320, "sequence": 10, "observations": 1200, "lookback": 90},
    "quant_pricing_research": {"sample": 360, "sequence": 12, "observations": 1500, "lookback": 120},
    "state_space_filter_research": {"sample": 360, "sequence": 16, "observations": 1800, "lookback": 120},
    "optimization_research": {"sample": 320, "sequence": 12, "observations": 1200, "lookback": 90},
    "transaction_cost_slippage_research": {"sample": 300, "sequence": 12, "observations": 1000, "lookback": 60},
    "market_data_tape_normalization_research": {"sample": 300, "sequence": 12, "observations": 1000, "lookback": 45},
    "limit_order_book_transformers": {"sample": 320, "sequence": 16, "observations": 1500, "lookback": 45},
    "order_flow_toxicity_research": {"sample": 320, "sequence": 16, "observations": 1500, "lookback": 45},
    "gpu_quant_acceleration": {"sample": 260, "sequence": 8, "observations": 900, "lookback": 45},
    "cross_asset_basis_research": {"sample": 360, "sequence": 12, "observations": 1500, "lookback": 120},
}

ROLE_CONTEXT_EXTRAS: dict[str, list[str]] = {
    "signal_sub_bot": ["signal_trace", "regime_context", "paper_live_outcome"],
    "options_sub_bot": ["options_chain", "iv_surface", "greeks", "event_vol_context"],
    "infrastructure_sub_bot": ["runtime_health", "queue_pressure", "resource_pressure", "guard_action_outcome"],
    "sleeve_master_bot": ["sleeve_state", "allocation_context", "risk_budget", "paper_live_outcome"],
    "master_bot": ["cross_sleeve_context", "allocation_context", "risk_budget", "paper_live_outcome"],
    "grand_master_bot": ["whole_system_state", "cross_sleeve_context", "risk_budget", "operator_context"],
}

WEAKNESS_CONTEXT: dict[str, list[str]] = {
    "sample_starved": ["decision_explanations", "paper_live_outcome", "feature_lineage"],
    "label_depth_gap": ["label_outcome_join", "sample_eligibility_reason", "rejected_candidate_trace", "abstained_candidate_trace"],
    "sequence_starved": ["runtime_training_snapshot", "sequence_history", "feature_lineage"],
    "label_imbalanced": ["lane_balance_bucket", "action_effect_bucket", "side_specific_outcome"],
    "overacting": ["confidence_trace", "abstention_threshold_trace", "side_specific_outcome"],
    "quality_weak": ["counterfactual_replay", "paper_live_outcome", "feature_importance_trace"],
    "runtime_depth_debt": ["runtime_training_snapshot", "sequence_history", "runtime_health"],
    "paper_loss_drag": ["paper_loss_hard_negative", "paper_profile_strategy_pair", "unrealized_drag", "post_entry_path"],
    "confirmation_bias": [
        "independent_evidence_trace",
        "cross_asset_confirmation",
        "source_quality",
        "fill_quality",
        "spread_quality",
        "portfolio_conflict",
        "event_catalyst_confirmation",
    ],
    "advanced_quant_depth_debt": [
        "quant_model_feature_surface",
        "runtime_feature_history",
        "feature_confidence_matrix",
        "proxy_label_quality",
        "walk_forward_trace",
    ],
    "advanced_quant_proxy_gap": [
        "provider_freshness",
        "cross_provider_agreement",
        "proxy_data_source_lineage",
        "source_quality",
    ],
    "advanced_quant_label_gap": [
        "label_contract_quality",
        "point_in_time_label_quality",
        "proxy_label_quality",
        "label_outcome_join",
    ],
}

ENRICHMENT_CONTEXT_BY_WEAKNESS: dict[str, list[str]] = {
    "sample_starved": [
        "accepted_candidate_trace",
        "rejected_candidate_trace",
        "label_outcome_join",
        "confidence_trace",
        "symbol_session_bucket",
        "decision_reason_taxonomy",
    ],
    "label_depth_gap": [
        "label_outcome_join",
        "sample_eligibility_reason",
        "abstained_candidate_trace",
        "rejected_candidate_trace",
        "neutral_examples",
        "counterfactual_opportunity_trace",
        "point_in_time_label_quality",
    ],
    "sequence_starved": [
        "longer_lookback_sequences",
        "cross_symbol_sequence_pool",
        "session_bucket_continuity",
        "snapshot_join_keys",
        "sequence_gap_reason",
    ],
    "label_imbalanced": [
        "counter_side_examples",
        "neutral_examples",
        "lane_balance_bucket",
        "side_specific_outcome",
    ],
    "overacting": [
        "abstention_threshold_trace",
        "confidence_decay_trace",
        "false_positive_candidate_trace",
        "side_specific_precision_trace",
    ],
    "quality_weak": [
        "counterfactual_replay",
        "feature_importance_trace",
        "paper_live_outcome",
        "non_regression_marker",
    ],
    "runtime_depth_debt": [
        "runtime_training_snapshot",
        "runtime_health",
        "resource_pressure",
        "writer_pressure",
    ],
    "paper_loss_drag": [
        "paper_loss_cause",
        "paper_profile",
        "paper_strategy",
        "paper_loss_hard_negative",
        "exit_quality_trace",
        "post_entry_mfe_mae",
        "unrealized_drag_bucket",
    ],
    "confirmation_bias": [
        "independent_evidence_channel_count",
        "confirmation_bias_bucket",
        "source_fill_spread_quality_bucket",
        "cross_asset_disagreement_trace",
        "portfolio_conflict_clearance",
        "event_confirmation_strength",
    ],
    "advanced_quant_depth_debt": [
        "feature_surface_snapshot_id",
        "model_parameter_trace",
        "runtime_feature_history",
        "cross_symbol_sequence_pool",
        "walk_forward_fold_trace",
        "non_regression_marker",
    ],
    "advanced_quant_proxy_gap": [
        "provider_freshness",
        "cross_provider_agreement",
        "proxy_data_source_lineage",
        "source_confidence_bucket",
    ],
    "advanced_quant_label_gap": [
        "point_in_time_label_quality",
        "proxy_label_quality_bucket",
        "label_outcome_join",
        "sample_eligibility_reason",
    ],
}

LOOKBACK_DAYS_BY_LABEL_FAMILY: dict[str, int] = {
    "generic_directional": 45,
    "intraday_fast": 45,
    "multi_day": 90,
    "options_surface": 60,
    "income_options_surface": 90,
    "income_total_return": 120,
    "fixed_income_rates": 120,
    "credit_spread": 90,
    "execution_cost_quality": 45,
    "correlation_risk_effect": 90,
    "crypto_microstructure": 45,
    "futures_event_session": 60,
    "operational_guard_effect": 30,
}
LOOKBACK_DAYS_BY_LABEL_FAMILY.update(
    {
        family: targets["lookback"]
        for family, targets in ADVANCED_QUANT_SAMPLE_TARGETS.items()
        if "lookback" in targets
    }
)

COMMON_LABEL_REPAIR_OUTPUTS = [
    "label_outcome_join",
    "accepted_candidate_trace",
    "rejected_candidate_trace",
    "forward_return_bucket",
    "risk_adjusted_return_bucket",
    "label_quality_bucket",
    "lane_balance_bucket",
    "side_specific_outcome",
    "decision_reason_taxonomy",
    "sample_eligibility_reason",
    "paper_loss_cause",
    "paper_profile",
    "paper_strategy",
    "confirmation_bias_bucket",
    "independent_evidence_channel_count",
    "source_fill_spread_quality_bucket",
]

LABEL_REPAIR_OUTPUTS_BY_FAMILY: dict[str, list[str]] = {
    "options_surface": ["iv_realized_bucket", "skew_shift_bucket", "spread_quality_bucket", "event_vol_reset_bucket"],
    "income_options_surface": ["ex_dividend_window_bucket", "payout_safety_bucket", "iv_realized_bucket", "skew_shift_bucket"],
    "income_total_return": ["payout_safety_bucket", "dividend_cut_risk_bucket", "earnings_quality_bucket", "income_total_return_bucket"],
    "fixed_income_rates": ["yield_curve_shift_bucket", "duration_beta_bucket", "inflation_surprise_bucket", "credit_stress_bucket"],
    "credit_spread": ["spread_widening_risk_bucket", "credit_beta_bucket", "duration_adjusted_return_bucket", "liquidity_state_bucket"],
    "execution_cost_quality": ["spread_cost_delta_bucket", "fill_quality_bucket", "latency_penalty_bucket", "market_impact_proxy_bucket"],
    "correlation_risk_effect": ["correlation_cluster_bucket", "exposure_netting_delta_bucket", "diversification_score_bucket"],
    "quant_research_control": ["quant_control_readiness_bucket", "resource_pressure_bucket", "feature_lineage_quality_bucket"],
    "quant_pricing_research": ["pricing_model_dispersion_bucket", "surface_mispricing_bucket", "hedge_cost_bucket", "proxy_label_quality_bucket"],
    "state_space_filter_research": ["regime_filter_confidence_bucket", "state_transition_quality_bucket", "latent_state_stability_bucket"],
    "optimization_research": ["objective_improvement_bucket", "constraint_violation_bucket", "kelly_sizing_quality_bucket"],
    "transaction_cost_slippage_research": ["slippage_realism_bucket", "queue_cost_bucket", "market_impact_proxy_bucket"],
    "transport_topology_research": ["transport_distance_bucket", "topology_regime_persistence_bucket", "cross_asset_shape_shift_bucket"],
    "signature_hawkes_generators": ["path_signature_bucket", "hawkes_event_cluster_bucket", "lead_lag_quality_bucket"],
    "qemc_path_volatility_research": ["path_volatility_bucket", "rough_vol_quality_bucket", "qemc_variance_reduction_bucket"],
    "xva_counterparty_margin_research": ["xva_exposure_bucket", "margin_waterfall_bucket", "counterparty_stress_bucket"],
    "credit_derivatives_research": ["credit_spread_surface_bucket", "hazard_rate_bucket", "recovery_rate_bucket"],
    "securitized_products_research": ["prepayment_oas_bucket", "loan_pool_credit_bucket", "duration_convexity_bucket"],
    "repo_securities_lending_research": ["repo_funding_bucket", "borrow_fee_pressure_bucket", "locate_pressure_bucket"],
    "market_data_tape_normalization_research": ["tape_schema_quality_bucket", "feed_latency_bucket", "off_exchange_volume_bucket"],
    "limit_order_book_transformers": ["lob_sequence_quality_bucket", "queue_position_bucket", "depth_imbalance_bucket"],
    "lobdif_crisis_microstructure_research": ["crisis_microstructure_bucket", "flash_freeze_bucket", "toxic_liquidity_bucket"],
    "rlbf_dms_equivariant_research": ["policy_backtracking_quality_bucket", "dms_realism_bucket", "equivariant_stability_bucket"],
    "proof_quantum_formal_backend_research": ["formal_safety_bucket", "backend_readiness_bucket", "regression_guard_bucket"],
    "model_risk_validation_research": ["model_risk_bucket", "benchmark_gap_bucket", "overfit_gap_bucket"],
    "neural_sde_kan_hedging": ["neural_sde_stability_bucket", "kan_hedge_quality_bucket", "pathwise_greek_bucket"],
    "gpu_quant_acceleration": ["gpu_runtime_bucket", "mlx_metal_profile_bucket", "resource_pressure_bucket"],
    "order_flow_toxicity_research": ["vpin_toxicity_bucket", "toxic_flow_bucket", "spread_queue_response_bucket"],
    "cross_asset_basis_research": ["basis_widening_bucket", "basis_convergence_bucket", "funding_stress_bucket"],
}

LABEL_REPAIR_ACTIONS_BY_FAMILY: dict[str, list[str]] = {
    "options_surface": ["join option-chain snapshots to realized volatility and spread outcomes"],
    "income_options_surface": ["join dividend calendar events with option surface and payout safety outcomes"],
    "income_total_return": ["join dividend quality, payout safety, and ex-dividend windows to total-return outcomes"],
    "fixed_income_rates": ["join rate-curve and duration-context buckets to bond/rates follow-through outcomes"],
    "credit_spread": ["join credit-spread and liquidity-state buckets to risk-adjusted follow-through outcomes"],
    "execution_cost_quality": ["join slippage, fill quality, spread cost, and latency traces to execution outcomes"],
    "correlation_risk_effect": ["join correlation clusters and exposure-netting deltas to risk-adjusted portfolio outcomes"],
    "quant_research_control": ["join quant model control traces to feature-lineage and resource-pressure outcomes"],
    "quant_pricing_research": ["join pricing surfaces, benchmark prices, realized volatility, and hedge-cost outcomes point-in-time"],
    "state_space_filter_research": ["join state filter diagnostics to next-regime and latent-state stability outcomes"],
    "optimization_research": ["join optimizer search traces to objective improvement, constraint, and Kelly-sizing outcomes"],
    "transaction_cost_slippage_research": ["join transaction-cost surfaces to fill realism, queue cost, and slippage outcomes"],
    "transport_topology_research": ["join transport/topology traces to cross-asset shape-shift outcomes"],
    "signature_hawkes_generators": ["join path signatures and Hawkes event clusters to lead-lag and event outcome buckets"],
    "qemc_path_volatility_research": ["join QEMC and rough-volatility traces to path-volatility realized outcomes"],
    "xva_counterparty_margin_research": ["join XVA exposure ladders and margin waterfalls to counterparty-stress outcomes"],
    "credit_derivatives_research": ["join credit-spread surfaces, hazard-rate proxies, and recovery proxies to credit outcomes"],
    "securitized_products_research": ["join securitized-product proxy universes to OAS, prepayment, and credit outcomes"],
    "repo_securities_lending_research": ["join repo curves, borrow fees, locate pressure, and funding liquidity outcomes"],
    "market_data_tape_normalization_research": ["join normalized tape events to schema quality, latency, and off-exchange volume labels"],
    "limit_order_book_transformers": ["join LOB tensors to queue position, depth imbalance, and next-move outcomes"],
    "lobdif_crisis_microstructure_research": ["join crisis microstructure traces to flash-freeze and toxic-liquidity labels"],
    "rlbf_dms_equivariant_research": ["join simulator and RLBF traces to policy-backtracking and stability labels"],
    "proof_quantum_formal_backend_research": ["join formal backend checks to safety and regression-guard labels"],
    "model_risk_validation_research": ["join benchmark comparisons, overfit gaps, and non-regression markers to model-risk labels"],
    "neural_sde_kan_hedging": ["join neural SDE and KAN hedging traces to hedge-cost and pathwise Greek labels"],
    "gpu_quant_acceleration": ["join GPU/MLX runtime profiles to resource-pressure and throughput labels"],
    "order_flow_toxicity_research": ["join VPIN/toxic-flow traces to spread and queue-response outcomes"],
    "cross_asset_basis_research": ["join basis, funding, and cross-asset correlation traces to convergence outcomes"],
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _bot_id(row: dict[str, Any]) -> str:
    return str(row.get("bot_id") or row.get("id") or row.get("name") or "").strip()


def _registry_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _is_active_collector(row: dict[str, Any]) -> bool:
    state = str(row.get("lifecycle_state") or "").strip().lower()
    return bool(
        _bot_id(row)
        and bool(row.get("active", False))
        and bool(row.get("data_collection_active", False))
        and state not in {"retired", "deleted", "deactivated"}
    )


def _is_explicit_intake_target(row: dict[str, Any]) -> bool:
    state = str(row.get("lifecycle_state") or "").strip().lower()
    return bool(
        _bot_id(row)
        and bool(row.get("active", False))
        and state not in {"retired", "deleted", "deactivated"}
    )


def _csv_set(raw: str) -> set[str]:
    return {item.strip().lower() for item in str(raw or "").split(",") if item.strip()}


def _label_contract(row: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    contract = _as_dict(row.get("label_contract"))
    if contract:
        return contract
    observed = _as_dict(diagnostic.get("observed_label_contract"))
    if observed:
        return observed
    return _as_dict(diagnostic.get("label_contract"))


def _label_family(contract: dict[str, Any], row: dict[str, Any]) -> str:
    return str(contract.get("label_family") or row.get("label_family") or "generic_directional").strip().lower()


def _training_lane(contract: dict[str, Any], row: dict[str, Any]) -> str:
    return str(contract.get("training_lane") or row.get("training_lane") or "").strip().lower()


def _is_advanced_quant_family(label_family: str) -> bool:
    normalized = str(label_family or "").strip().lower()
    return normalized in ADVANCED_QUANT_LABEL_FAMILIES or (
        ("quant" in normalized or "research" in normalized)
        and normalized not in {"generic_directional", "alpha_research"}
    )


def _context_defaults(label_family: str, training_lane: str, bot_role: str) -> list[str]:
    contexts: list[str] = []
    contexts.extend(BASE_CONTEXT_BY_LABEL_FAMILY.get(label_family, []))
    if "research" in label_family or "research" in training_lane:
        contexts.extend(DEFAULT_RESEARCH_CONTEXT)
    if _is_advanced_quant_family(label_family):
        for section_contexts in ADVANCED_QUANT_COLLECTION_SECTIONS.values():
            contexts.extend(section_contexts)
        contexts.extend(ADVANCED_QUANT_CONTEXT_BY_LABEL_FAMILY.get(label_family, []))
    contexts.extend(ROLE_CONTEXT_EXTRAS.get(bot_role, []))
    if not contexts:
        contexts.extend(BASE_CONTEXT_BY_LABEL_FAMILY["generic_directional"])
    return ordered_unique(contexts)


def _minimum_observations(row: dict[str, Any]) -> int:
    standard = _as_dict(row.get("paper_promotion_standard"))
    if _safe_int(standard.get("minimum_observations"), 0) > 0:
        return _safe_int(standard.get("minimum_observations"), 0)
    threshold = _as_dict(row.get("data_collection_threshold"))
    if _safe_int(threshold.get("minimum_training_observations"), 0) > 0:
        return _safe_int(threshold.get("minimum_training_observations"), 0)
    if _safe_int(row.get("minimum_training_observations"), 0) > 0:
        return _safe_int(row.get("minimum_training_observations"), 0)
    return OBSERVATION_FLOOR_DEFAULT


def _diagnostic_path(project_root: Path, bot_id: str) -> Path:
    return project_root / "governance" / "training_diagnostics" / f"{bot_id}_latest.json"


def _diagnostic_for(project_root: Path, bot_id: str) -> dict[str, Any]:
    return load_json(_diagnostic_path(project_root, bot_id))


def _index_bot_needs(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _as_list(payload.get("bot_needs")):
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if bot_id:
            indexed[bot_id] = row
    return indexed


def _runtime_depth_debt_ids(training_quality: dict[str, Any]) -> set[str]:
    targeted = _as_dict(training_quality.get("targeted_actions"))
    return {
        str(bot_id).strip()
        for bot_id in _as_list(targeted.get("runtime_input_depth_debt_bot_ids"))
        if str(bot_id or "").strip()
    }


def _paper_loss_index(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = _as_list(payload.get("strategy_controls"))
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        if bot_id:
            indexed[bot_id].append(row)
    return indexed


def _paper_scout_collection_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contract = _as_dict(payload.get("scout_collection_contract"))
    bot_ids = [
        str(item).strip()
        for item in _as_list(contract.get("target_bot_ids"))
        if str(item or "").strip()
    ]
    if not bot_ids:
        hardening = _as_dict(payload.get("paper_profitability_hardening_contract"))
        contract = _as_dict(hardening.get("scout_collection_contract"))
        bot_ids = [
            str(item).strip()
            for item in _as_list(contract.get("target_bot_ids"))
            if str(item or "").strip()
        ]
    return {bot_id: contract for bot_id in bot_ids if contract}


def _paper_loss_control_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in rows:
        confirmation = _as_dict(row.get("confirmation_bias_control"))
        if not confirmation:
            confirmation = _as_dict(_as_dict(row.get("upgrade_contracts")).get("confirmation_bias_control"))
        summaries.append(
            {
                "profile": str(row.get("profile") or ""),
                "strategy": str(row.get("strategy") or ""),
                "mode": str(row.get("mode") or ""),
                "ending_net_pnl_total": _safe_float(row.get("ending_net_pnl_total"), 0.0),
                "score_penalty_norm": _safe_float(row.get("score_penalty_norm"), 0.0),
                "confirmation_bias_score_norm": _safe_float(
                    row.get("confirmation_bias_score_norm"),
                    _safe_float(confirmation.get("confirmation_bias_score_norm"), 0.0),
                ),
                "loss_causes": [str(item) for item in _as_list(row.get("loss_causes")) if str(item or "").strip()],
                "required_context": _as_list(_as_dict(row.get("data_intake_enrichment")).get("required_context")),
                "required_label_outputs": _as_list(_as_dict(row.get("data_intake_enrichment")).get("required_label_outputs")),
            }
        )
    return summaries


def _advanced_quant_section_contract(
    *,
    bot_id: str,
    label_family: str,
    weaknesses: list[str],
    sample_plan: dict[str, Any],
    label_repair_plan: dict[str, Any],
) -> dict[str, Any]:
    if not _is_advanced_quant_family(label_family):
        return {"active": False}
    section_records: list[dict[str, Any]] = []
    for section, contexts in ADVANCED_QUANT_COLLECTION_SECTIONS.items():
        section_records.append(
            {
                "section": section,
                "required_context": list(contexts),
                "status": "needs_repair" if weaknesses else "ready",
                "weakness_repair_focus": [
                    weakness
                    for weakness in weaknesses
                    if weakness.startswith("advanced_quant_") or weakness in {"sample_starved", "label_depth_gap", "sequence_starved", "label_imbalanced"}
                ],
            }
        )
    return {
        "active": True,
        "contract_version": "advanced_quant_collection_sections_v1",
        "bot_id": bot_id,
        "label_family": label_family,
        "section_count": len(section_records),
        "sections": section_records,
        "sample_targets": {
            "usable_sample_goal": sample_plan.get("usable_sample_goal"),
            "eligible_sequence_goal": sample_plan.get("eligible_sequence_goal"),
            "observation_goal": sample_plan.get("observation_goal"),
            "lookback_days": sample_plan.get("recommended_lookback_days"),
        },
        "required_label_outputs": label_repair_plan.get("required_label_outputs"),
        "stop_when": (
            "all five collection sections have point-in-time joined rows, "
            f"sample_count >= {sample_plan.get('usable_sample_goal')}, "
            f"eligible_sequences >= {sample_plan.get('eligible_sequence_goal')}, "
            "and proxy_label_quality_bucket is present"
        ),
    }


def _weaknesses(
    *,
    row: dict[str, Any],
    label_family: str,
    contract: dict[str, Any],
    diagnostic: dict[str, Any],
    need: dict[str, Any],
    runtime_depth_debt_ids: set[str],
    paper_loss_controls: list[dict[str, Any]],
) -> list[str]:
    bot_id = _bot_id(row)
    evidence = _as_dict(need.get("evidence"))
    prescription = _as_dict(need.get("effectiveness_prescription"))
    source = evidence or diagnostic
    sample_count = max(_safe_int(source.get("sample_count"), 0), _safe_int(row.get("sample_count"), 0))
    observation_count = max(
        _safe_int(source.get("observation_count"), 0),
        _safe_int(row.get("data_collection_observations"), 0),
        _safe_int(row.get("collected_observation_count"), 0),
    )
    eligible_sequences = _safe_int(source.get("eligible_sequences"), 0)
    positive_rate = _safe_float(source.get("positive_rate"), -1.0)
    acted_coverage = _safe_float(source.get("acted_coverage"), -1.0)
    quality_score = _safe_float(source.get("quality_score"), 0.0)
    test_accuracy = _safe_float(source.get("test_accuracy"), 0.0)
    minimum_observations = _minimum_observations(row)
    primary_need = str(need.get("primary_need") or "").strip()
    can_train_now = bool(prescription.get("can_train_now", False))
    training_topoff_need = primary_need in {"targeted_quality_retrain", "top_off_walk_forward_runs"}
    enough_usable_samples = sample_count >= SAMPLE_FLOOR

    out: list[str] = []
    if not ((can_train_now or training_topoff_need) and enough_usable_samples) and (
        sample_count < SAMPLE_FLOOR or observation_count < minimum_observations
    ):
        out.append("sample_starved")
    if sample_count < SAMPLE_FLOOR and observation_count > 0:
        out.append("label_depth_gap")
    if eligible_sequences < ELIGIBLE_SEQUENCE_FLOOR:
        out.append("sequence_starved")
    if sample_count > 0 and positive_rate >= 0.0 and (positive_rate < 0.25 or positive_rate > 0.75):
        out.append("label_imbalanced")
    if acted_coverage > 0.32:
        out.append("overacting")
    if quality_score and quality_score < 0.5:
        out.append("quality_weak")
    elif test_accuracy and test_accuracy < 0.52:
        out.append("quality_weak")
    if bot_id in runtime_depth_debt_ids:
        out.append("runtime_depth_debt")
    if paper_loss_controls:
        out.append("paper_loss_drag")
        if any(
            _safe_float(control.get("confirmation_bias_score_norm"), 0.0) >= 0.20
            or bool(_as_dict(control.get("confirmation_bias_control")).get("active", False))
            or bool(_as_dict(_as_dict(control.get("upgrade_contracts")).get("confirmation_bias_control")).get("active", False))
            for control in paper_loss_controls
        ):
            out.append("confirmation_bias")
    if _is_advanced_quant_family(label_family):
        if "sample_starved" in out or "sequence_starved" in out or eligible_sequences < ADVANCED_QUANT_SAMPLE_TARGETS.get(label_family, {}).get("sequence", ELIGIBLE_SEQUENCE_GOAL):
            out.append("advanced_quant_depth_debt")
        known_contexts = {
            str(item).strip()
            for item in (
                _as_list(row.get("data_intake_collections"))
                + _as_list(contract.get("required_context"))
                + _as_list(row.get("proxy_data_sources"))
            )
            if str(item or "").strip()
        }
        if not known_contexts.intersection({"provider_freshness", "cross_provider_agreement", "proxy_data_source_lineage", "source_quality"}):
            out.append("advanced_quant_proxy_gap")
        if not known_contexts.intersection({"label_contract_quality", "point_in_time_label_quality", "proxy_label_quality", "label_outcome_join"}):
            out.append("advanced_quant_label_gap")
    return ordered_unique(out)


def _focus_context(required: list[str], expanded: list[str], weaknesses: list[str], *, label_family: str = "") -> list[str]:
    focus: list[str] = []
    focus.extend(required)
    if _is_advanced_quant_family(label_family):
        for section_contexts in ADVANCED_QUANT_COLLECTION_SECTIONS.values():
            focus.extend(section_contexts)
        focus.extend(ADVANCED_QUANT_CONTEXT_BY_LABEL_FAMILY.get(label_family, []))
    for weakness in weaknesses:
        focus.extend(WEAKNESS_CONTEXT.get(weakness, []))
    focus.extend(expanded)
    limit = 24 if _is_advanced_quant_family(label_family) else 16
    return ordered_unique(focus)[:limit]


def _enrichment_context(required: list[str], focus: list[str], weaknesses: list[str], *, label_family: str = "") -> list[str]:
    contexts: list[str] = []
    contexts.extend(required)
    contexts.extend(focus)
    if _is_advanced_quant_family(label_family):
        for section_contexts in ADVANCED_QUANT_COLLECTION_SECTIONS.values():
            contexts.extend(section_contexts)
        contexts.extend(ADVANCED_QUANT_CONTEXT_BY_LABEL_FAMILY.get(label_family, []))
    for weakness in weaknesses:
        contexts.extend(ENRICHMENT_CONTEXT_BY_WEAKNESS.get(weakness, []))
    limit = 48 if _is_advanced_quant_family(label_family) else 32
    return ordered_unique(contexts)[:limit]


def _recommended_lookback_days(label_family: str, weaknesses: list[str]) -> int:
    base = int(LOOKBACK_DAYS_BY_LABEL_FAMILY.get(label_family, 45))
    if _is_advanced_quant_family(label_family):
        base = max(base, ADVANCED_QUANT_SAMPLE_TARGETS.get(label_family, {}).get("lookback", 90))
    if "sample_starved" in weaknesses and label_family in {"multi_day", "options_surface"}:
        base = max(base, 90)
    elif "sample_starved" in weaknesses:
        base = max(base, 60)
    if "sequence_starved" in weaknesses:
        base = max(base, 45)
    return base


def _sample_enrichment_plan(
    *,
    bot_id: str,
    label_family: str,
    primary_need: str,
    weaknesses: list[str],
    sample_count: int,
    observation_count: int,
    eligible_sequences: int,
    minimum_observations: int,
    required_context: list[str],
    focus_context: list[str],
) -> dict[str, Any]:
    advanced_targets = ADVANCED_QUANT_SAMPLE_TARGETS.get(label_family, {}) if _is_advanced_quant_family(label_family) else {}
    base_sample_goal = max(USABLE_SAMPLE_GOAL, _safe_int(advanced_targets.get("sample"), USABLE_SAMPLE_GOAL))
    base_sequence_goal = max(ELIGIBLE_SEQUENCE_GOAL, _safe_int(advanced_targets.get("sequence"), ELIGIBLE_SEQUENCE_GOAL))
    sample_goal = base_sample_goal if ("sample_starved" in weaknesses or "label_depth_gap" in weaknesses or "advanced_quant_depth_debt" in weaknesses) else SAMPLE_FLOOR
    eligible_goal = base_sequence_goal if ("sequence_starved" in weaknesses or "advanced_quant_depth_debt" in weaknesses) else ELIGIBLE_SEQUENCE_FLOOR
    observation_goal = max(minimum_observations, observation_count)
    if advanced_targets:
        observation_goal = max(observation_goal, _safe_int(advanced_targets.get("observations"), 0))
    if "sample_starved" in weaknesses:
        observation_goal = max(observation_goal, observation_count + max(sample_goal - sample_count, 0) * 3)
    if "runtime_depth_debt" in weaknesses:
        observation_goal = max(observation_goal, 1000)
    sample_gap = max(sample_goal - sample_count, 0)
    sequence_gap = max(eligible_goal - eligible_sequences, 0)
    observation_gap = max(observation_goal - observation_count, 0)
    enrichment_context = _enrichment_context(required_context, focus_context, weaknesses, label_family=label_family)
    actions: list[str] = []
    if sample_gap:
        actions.append("materialize accepted and rejected candidate rows into point-in-time decision explanations")
        actions.append("persist label_outcome_join fields so raw observations become usable samples")
    if "label_depth_gap" in weaknesses:
        actions.append("run the label-depth bridge over existing raw observations before requiring another blind collection pass")
        actions.append("keep abstained and neutral candidates as eligible calibration examples instead of discarding them")
    if sequence_gap:
        actions.append("increase sequence history coverage across session buckets and related symbols")
    if "label_imbalanced" in weaknesses:
        actions.append("rebalance label builder with counter-side and neutral examples")
    if "overacting" in weaknesses:
        actions.append("keep stricter abstention and record false-positive candidate traces")
    if "runtime_depth_debt" in weaknesses:
        actions.append("prioritize runtime snapshot and resource-pressure rows for this bot until the 1000-observation floor clears")
    if "advanced_quant_depth_debt" in weaknesses:
        actions.append("persist compact quant feature-surface snapshots with snapshot_id and model_parameter_trace")
        actions.append("pool related symbols into cross_symbol_sequence_pool while keeping point-in-time joins isolated")
    if "advanced_quant_proxy_gap" in weaknesses:
        actions.append("attach provider_freshness, cross_provider_agreement, and proxy_data_source_lineage to every proxy-derived row")
    if "advanced_quant_label_gap" in weaknesses:
        actions.append("materialize point_in_time_label_quality and proxy_label_quality_bucket before the next canary")
    if not actions:
        actions.append("keep current collection route warm and recheck bot-needs before the next canary")
    intensity = "normal"
    if sample_gap >= 200 or observation_gap >= 500 or "runtime_depth_debt" in weaknesses or "advanced_quant_depth_debt" in weaknesses:
        intensity = "high"
    if sample_gap >= 500 or observation_gap >= 1000:
        intensity = "critical"
    return {
        "plan_version": "sample_enrichment_v2",
        "bot_id": bot_id,
        "primary_need": primary_need,
        "intensity": intensity,
        "usable_sample_goal": sample_goal,
        "eligible_sequence_goal": eligible_goal,
        "observation_goal": observation_goal,
        "usable_sample_gap": sample_gap,
        "eligible_sequence_gap": sequence_gap,
        "observation_gap": observation_gap,
        "recommended_lookback_days": _recommended_lookback_days(label_family, weaknesses),
        "enrichment_context": enrichment_context,
        "collection_actions": actions,
        "validation_command": [
            "./scripts/ops/opsctl.sh",
            "bot-needs",
            "--include-bot-ids",
            bot_id,
            "--json",
        ],
        "stop_when": (
            f"sample_count >= {sample_goal}, eligible_sequences >= {eligible_goal}, "
            f"and observation_count >= {observation_goal}"
        ),
    }


def _label_repair_plan(
    *,
    bot_id: str,
    label_family: str,
    contract: dict[str, Any],
    sample_count: int,
    observation_count: int,
    eligible_sequences: int,
    sample_plan: dict[str, Any],
    extra_required_outputs: list[str] | None = None,
    extra_collection_actions: list[str] | None = None,
) -> dict[str, Any]:
    primary_horizon = str(contract.get("primary_horizon") or contract.get("primary_label_horizon") or "1d_forward_return")
    aux_horizons = [str(item) for item in _as_list(contract.get("aux_horizons") or contract.get("aux_label_horizons")) if str(item or "").strip()]
    required_outputs = ordered_unique(
        COMMON_LABEL_REPAIR_OUTPUTS
        + LABEL_REPAIR_OUTPUTS_BY_FAMILY.get(label_family, [])
        + [str(item) for item in (extra_required_outputs or []) if str(item or "").strip()]
    )
    actions = [
        "backfill point-in-time label_outcome_join rows for accepted and rejected decisions",
        "emit sample_eligibility_reason for every rejected training sample",
        "write side_specific_outcome and lane_balance_bucket before the next canary",
        "dedupe labels by bot_id, symbol, mode, timestamp_utc, snapshot_id, and decision_id",
    ]
    actions.extend(LABEL_REPAIR_ACTIONS_BY_FAMILY.get(label_family, []))
    actions.extend([str(item) for item in (extra_collection_actions or []) if str(item or "").strip()])
    actions = ordered_unique(actions)
    sample_goal = _safe_int(sample_plan.get("usable_sample_goal"), USABLE_SAMPLE_GOAL)
    sequence_goal = _safe_int(sample_plan.get("eligible_sequence_goal"), ELIGIBLE_SEQUENCE_GOAL)
    observation_goal = _safe_int(sample_plan.get("observation_goal"), max(observation_count, OBSERVATION_FLOOR_DEFAULT))
    blockers: list[str] = []
    if sample_count < sample_goal:
        blockers.append("usable_sample_gap")
    if eligible_sequences < sequence_goal:
        blockers.append("eligible_sequence_gap")
    if observation_count < observation_goal:
        blockers.append("observation_gap")
    return {
        "plan_version": "label_repair_v1",
        "bot_id": bot_id,
        "label_family": label_family,
        "primary_horizon": primary_horizon,
        "aux_horizons": aux_horizons,
        "required_join_mode": "point_in_time_only",
        "required_join_keys": ["bot_id", "symbol", "mode", "timestamp_utc", "snapshot_id", "decision_id"],
        "required_label_outputs": required_outputs,
        "balance_targets": {
            "positive_rate_min": 0.35,
            "positive_rate_max": 0.65,
            "min_long_precision": 0.52,
            "min_short_precision": 0.52,
            "max_acted_coverage_until_quality_passes": 0.32,
        },
        "lookback_days": sample_plan.get("recommended_lookback_days"),
        "sample_targets": {
            "usable_sample_goal": sample_goal,
            "eligible_sequence_goal": sequence_goal,
            "observation_goal": observation_goal,
            "current_sample_count": sample_count,
            "current_eligible_sequences": eligible_sequences,
            "current_observation_count": observation_count,
        },
        "blockers": blockers,
        "collection_actions": actions,
        "validation_command": [
            "./scripts/ops/opsctl.sh",
            "training-data-intake",
            "--include-bot-ids",
            bot_id,
            "--json",
        ],
        "stop_when": (
            f"label_outcome_join is present, sample_count >= {sample_goal}, "
            f"eligible_sequences >= {sequence_goal}, observation_count >= {observation_goal}, "
            "and positive_rate is between 0.35 and 0.65"
        ),
    }


def _label_depth_bridge(
    *,
    bot_id: str,
    label_family: str,
    sample_count: int,
    observation_count: int,
    eligible_sequences: int,
    sample_plan: dict[str, Any],
    label_repair_plan: dict[str, Any],
) -> dict[str, Any]:
    sample_goal = _safe_int(sample_plan.get("usable_sample_goal"), USABLE_SAMPLE_GOAL)
    sequence_goal = _safe_int(sample_plan.get("eligible_sequence_goal"), ELIGIBLE_SEQUENCE_GOAL)
    observation_goal = _safe_int(sample_plan.get("observation_goal"), max(observation_count, OBSERVATION_FLOOR_DEFAULT))
    status = "ready"
    if sample_count < sample_goal and observation_count > 0:
        status = "materialize_from_existing_observations"
    if observation_count < observation_goal:
        status = "collect_and_materialize"
    if sample_count >= sample_goal and eligible_sequences >= sequence_goal:
        status = "depth_ready"
    return {
        "version": "label_depth_bridge_v1",
        "bot_id": bot_id,
        "label_family": label_family,
        "status": status,
        "current_sample_count": sample_count,
        "current_observation_count": observation_count,
        "current_eligible_sequences": eligible_sequences,
        "usable_sample_goal": sample_goal,
        "eligible_sequence_goal": sequence_goal,
        "observation_goal": observation_goal,
        "usable_sample_gap": max(sample_goal - sample_count, 0),
        "eligible_sequence_gap": max(sequence_goal - eligible_sequences, 0),
        "observation_gap": max(observation_goal - observation_count, 0),
        "conversion_target_min": 0.12,
        "required_join_mode": "point_in_time_only",
        "required_join_keys": list(label_repair_plan.get("required_join_keys") or []),
        "required_label_outputs": list(label_repair_plan.get("required_label_outputs") or []),
        "required_event_mix": [
            "accepted_candidate_trace",
            "rejected_candidate_trace",
            "abstained_candidate_trace",
            "neutral_examples",
            "counter_side_examples",
            "paper_live_outcome",
        ],
        "next_action": (
            "materialize label_outcome_join and sample_eligibility_reason from the existing observation pool"
            if observation_count > 0
            else "collect point-in-time observations before materializing label depth"
        ),
        "stop_when": (
            f"real sample_count >= {sample_goal}, eligible_sequences >= {sequence_goal}, "
            f"and observation_count >= {observation_goal}"
        ),
    }


def _priority(
    *,
    need: dict[str, Any],
    weaknesses: list[str],
    sample_count: int,
    observation_count: int,
    minimum_observations: int,
) -> float:
    priority = _safe_float(need.get("priority"), 0.0)
    if priority <= 0:
        priority = 10.0
    if "runtime_depth_debt" in weaknesses:
        priority += 20.0
    if "paper_loss_drag" in weaknesses:
        priority += 18.0
    if "confirmation_bias" in weaknesses:
        priority += 16.0
    if "label_depth_gap" in weaknesses:
        priority += 14.0
    if "sample_starved" in weaknesses:
        priority += min(max(minimum_observations - observation_count, SAMPLE_FLOOR - sample_count, 0) / 25.0, 30.0)
    if "quality_weak" in weaknesses:
        priority += 12.0
    if "overacting" in weaknesses:
        priority += 8.0
    return round(priority, 3)


def _row_record(
    *,
    project_root: Path,
    row: dict[str, Any],
    need: dict[str, Any],
    runtime_depth_debt_ids: set[str],
    paper_loss_index: dict[str, list[dict[str, Any]]],
    paper_scout_collection_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    bot_id = _bot_id(row)
    diagnostic = _diagnostic_for(project_root, bot_id)
    paper_loss_controls = paper_loss_index.get(bot_id, [])
    scout_collection_contract = _as_dict(paper_scout_collection_index.get(bot_id))
    contract = _label_contract(row, diagnostic)
    label_family = _label_family(contract, row)
    training_lane = _training_lane(contract, row)
    bot_role = str(row.get("bot_role") or "").strip()
    scout_required_context = [
        str(item)
        for item in _as_list(scout_collection_contract.get("required_context"))
        if str(item or "").strip()
    ]
    scout_required_outputs = [
        str(item)
        for item in _as_list(scout_collection_contract.get("required_label_outputs"))
        if str(item or "").strip()
    ]
    scout_collection_rules = [
        str(item)
        for item in _as_list(scout_collection_contract.get("collection_rules"))
        if str(item or "").strip()
    ]
    required = ordered_unique(
        [str(item) for item in _as_list(contract.get("required_context")) if str(item or "").strip()]
        + scout_required_context
    )
    expanded = ordered_unique(required + _context_defaults(label_family, training_lane, bot_role))
    weaknesses = _weaknesses(
        row=row,
        label_family=label_family,
        contract=contract,
        diagnostic=diagnostic,
        need=need,
        runtime_depth_debt_ids=runtime_depth_debt_ids,
        paper_loss_controls=paper_loss_controls,
    )
    focus = _focus_context(required, expanded, weaknesses, label_family=label_family)
    paper_control_summaries = _paper_loss_control_summary(paper_loss_controls)
    for summary in paper_control_summaries:
        focus_limit = 24 if _is_advanced_quant_family(label_family) else 16
        focus = ordered_unique(focus + [str(item) for item in _as_list(summary.get("required_context")) if str(item or "").strip()])[:focus_limit]
    evidence = _as_dict(need.get("evidence")) or diagnostic
    sample_count = _safe_int(evidence.get("sample_count"), 0)
    observation_count = max(
        _safe_int(evidence.get("observation_count"), 0),
        _safe_int(row.get("data_collection_observations"), 0),
        _safe_int(row.get("collected_observation_count"), 0),
    )
    eligible_sequences = _safe_int(evidence.get("eligible_sequences"), 0)
    minimum_observations = _minimum_observations(row)
    priority = _priority(
        need=need,
        weaknesses=weaknesses,
        sample_count=sample_count,
        observation_count=observation_count,
        minimum_observations=minimum_observations,
    )
    primary_need = str(need.get("primary_need") or "").strip() or ("collect_more_data" if weaknesses else "monitor")
    enrichment_plan = _sample_enrichment_plan(
        bot_id=bot_id,
        label_family=label_family,
        primary_need=primary_need,
        weaknesses=weaknesses,
        sample_count=sample_count,
        observation_count=observation_count,
        eligible_sequences=eligible_sequences,
        minimum_observations=minimum_observations,
        required_context=required,
        focus_context=focus,
    )
    label_repair_plan = _label_repair_plan(
        bot_id=bot_id,
        label_family=label_family,
        contract=contract,
        sample_count=sample_count,
        observation_count=observation_count,
        eligible_sequences=eligible_sequences,
        sample_plan=enrichment_plan,
        extra_required_outputs=scout_required_outputs,
        extra_collection_actions=scout_collection_rules,
    )
    label_depth_bridge = _label_depth_bridge(
        bot_id=bot_id,
        label_family=label_family,
        sample_count=sample_count,
        observation_count=observation_count,
        eligible_sequences=eligible_sequences,
        sample_plan=enrichment_plan,
        label_repair_plan=label_repair_plan,
    )
    advanced_quant_section_contract = _advanced_quant_section_contract(
        bot_id=bot_id,
        label_family=label_family,
        weaknesses=weaknesses,
        sample_plan=enrichment_plan,
        label_repair_plan=label_repair_plan,
    )
    stop_when = enrichment_plan["stop_when"] if "sample_starved" in weaknesses else _stop_when(primary_need, minimum_observations)
    if advanced_quant_section_contract.get("active") and _as_list(weaknesses):
        stop_when = str(advanced_quant_section_contract.get("stop_when") or stop_when)
    return {
        "bot_id": bot_id,
        "bot_role": bot_role,
        "lifecycle_state": str(row.get("lifecycle_state") or ""),
        "label_family": label_family,
        "training_lane": training_lane,
        "primary_need": primary_need,
        "priority": priority,
        "sample_count": sample_count,
        "observation_count": observation_count,
        "minimum_observations": minimum_observations,
        "observations_needed": max(minimum_observations - observation_count, 0),
        "eligible_sequences": eligible_sequences,
        "weaknesses": weaknesses,
        "paper_loss_controls": paper_control_summaries,
        "profitability_scout_collection": {
            "active": bool(scout_collection_contract),
            "mode": str(scout_collection_contract.get("mode") or ""),
            "required_context": scout_required_context,
            "required_label_outputs": scout_required_outputs,
            "collection_rules": scout_collection_rules,
        },
        "required_context": required,
        "expanded_context": expanded,
        "focus_context": focus,
        "enrichment_context": enrichment_plan["enrichment_context"],
        "sample_enrichment_plan": enrichment_plan,
        "label_repair_plan": label_repair_plan,
        "label_depth_bridge": label_depth_bridge,
        "advanced_quant_collection_contract": advanced_quant_section_contract,
        "diagnostic_path": str(_diagnostic_path(project_root, bot_id)),
        "next_action": _next_action(primary_need, weaknesses),
        "stop_when": stop_when,
    }


def _next_action(primary_need: str, weaknesses: list[str]) -> str:
    if primary_need == "materialize_label_depth" or "label_depth_gap" in weaknesses:
        return "materialize point-in-time label joins and sample eligibility reasons from existing observations"
    if primary_need == "collect_more_data" or "sample_starved" in weaknesses:
        return "route more point-in-time observations through the focus_context before another canary"
    if "overacting" in weaknesses:
        return "keep abstention calibration tight and validate with a micro-canary only"
    if "quality_weak" in weaknesses:
        return "run a bounded quality canary and keep only non-regressing artifacts"
    return "monitor and route to promotion review when promotion gates clear"


def _stop_when(primary_need: str, minimum_observations: int) -> str:
    if primary_need == "collect_more_data":
        return f"sample_count >= {SAMPLE_FLOOR}, eligible_sequences >= {ELIGIBLE_SEQUENCE_FLOOR}, and observation_count >= {minimum_observations}"
    if primary_need in {"targeted_quality_retrain", "top_off_walk_forward_runs"}:
        return "quality guard passes without increasing acted coverage or side imbalance"
    return "primary_need changes to monitor_passing_candidate or promotion review"


def _summaries(records: list[dict[str, Any]]) -> dict[str, Any]:
    context_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    weakness_counts: Counter[str] = Counter()
    enrichment_counts: Counter[str] = Counter()
    intensity_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    need_counts: Counter[str] = Counter()
    advanced_quant_sections: Counter[str] = Counter()
    advanced_quant_contract_count = 0
    for row in records:
        family_counts[str(row.get("label_family") or "")] += 1
        role_counts[str(row.get("bot_role") or "")] += 1
        need_counts[str(row.get("primary_need") or "")] += 1
        advanced_contract = _as_dict(row.get("advanced_quant_collection_contract"))
        if bool(advanced_contract.get("active", False)):
            advanced_quant_contract_count += 1
            for section in _as_list(advanced_contract.get("sections")):
                if isinstance(section, dict):
                    advanced_quant_sections[str(section.get("section") or "")] += 1
        for context in _as_list(row.get("focus_context")):
            context_counts[str(context)] += 1
        for context in _as_list(row.get("enrichment_context")):
            enrichment_counts[str(context)] += 1
        for weakness in _as_list(row.get("weaknesses")):
            weakness_counts[str(weakness)] += 1
        plan = _as_dict(row.get("sample_enrichment_plan"))
        intensity_counts[str(plan.get("intensity") or "normal")] += 1
    return {
        "context_counts": dict(context_counts.most_common()),
        "enrichment_context_counts": dict(enrichment_counts.most_common()),
        "label_family_counts": dict(family_counts.most_common()),
        "bot_role_counts": dict(role_counts.most_common()),
        "need_counts": dict(need_counts.most_common()),
        "weakness_counts": dict(weakness_counts.most_common()),
        "enrichment_intensity_counts": dict(intensity_counts.most_common()),
        "advanced_quant_contract_count": advanced_quant_contract_count,
        "advanced_quant_section_counts": dict(advanced_quant_sections.most_common()),
    }


def _select_records(records: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ranked = sorted(records, key=lambda row: (_safe_float(row.get("priority"), 0.0), str(row.get("bot_id") or "")), reverse=True)
    if limit > 0:
        return ranked[:limit]
    return ranked


def _apply_focus_to_registry(
    *,
    registry_path: Path,
    registry: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    timestamp = iso_now()
    indexed = {str(row.get("bot_id") or ""): row for row in records}
    changed_bot_ids: list[str] = []
    for row in _registry_rows(registry):
        bot_id = _bot_id(row)
        record = indexed.get(bot_id)
        if not record:
            continue
        focus_payload = {
            "version": "training_data_intake_expansion_v1",
            "updated_at_utc": timestamp,
            "primary_need": record.get("primary_need"),
            "priority": record.get("priority"),
            "label_family": record.get("label_family"),
            "weaknesses": record.get("weaknesses"),
            "paper_loss_controls": record.get("paper_loss_controls"),
            "profitability_scout_collection": record.get("profitability_scout_collection"),
            "required_context": record.get("required_context"),
            "expanded_context": record.get("expanded_context"),
            "focus_context": record.get("focus_context"),
            "enrichment_context": record.get("enrichment_context"),
            "sample_enrichment_plan": record.get("sample_enrichment_plan"),
            "label_repair_plan": record.get("label_repair_plan"),
            "label_depth_bridge": record.get("label_depth_bridge"),
            "advanced_quant_collection_contract": record.get("advanced_quant_collection_contract"),
            "stop_when": record.get("stop_when"),
            "source_artifact": str(DEFAULT_OUT_PATH.relative_to(PROJECT_ROOT)),
        }
        if row.get("data_intake_expansion") != focus_payload:
            row["data_intake_expansion"] = focus_payload
            row["data_collection_focus_context"] = list(record.get("focus_context") or [])
            row["data_collection_context_demand"] = list(record.get("expanded_context") or [])
            row["data_collection_enrichment_context"] = list(record.get("enrichment_context") or [])
            row["data_collection_sample_enrichment_plan"] = dict(record.get("sample_enrichment_plan") or {})
            row["data_collection_label_repair_plan"] = dict(record.get("label_repair_plan") or {})
            row["data_collection_label_depth_bridge"] = dict(record.get("label_depth_bridge") or {})
            row["data_collection_advanced_quant_contract"] = dict(record.get("advanced_quant_collection_contract") or {})
            row["data_collection_paper_loss_controls"] = list(record.get("paper_loss_controls") or [])
            row["data_collection_profitability_scout_contract"] = dict(record.get("profitability_scout_collection") or {})
            row["data_collection_focus_priority"] = record.get("priority")
            changed_bot_ids.append(bot_id)
    if changed_bot_ids:
        backup_path = _backup_registry(registry_path)
        _refresh_registry_summary(registry)
        write_payload(registry_path, registry)
    else:
        backup_path = ""
    return {
        "registry_updated": bool(changed_bot_ids),
        "updated_bot_count": len(changed_bot_ids),
        "updated_bot_ids": changed_bot_ids[:80],
        "registry_backup_path": backup_path,
    }


def _backup_registry(registry_path: Path) -> str:
    if not registry_path.exists():
        return ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = registry_path.parent / "governance" / "lifecycle" / f"master_bot_registry.data_intake_expansion_backup_{stamp}.json"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
    return str(backup_path)


def _refresh_registry_summary(registry: dict[str, Any]) -> None:
    rows = _registry_rows(registry)
    summary = _as_dict(registry.get("summary"))
    summary["total_bots"] = len(rows)
    summary["active_bots"] = sum(1 for row in rows if bool(row.get("active", False)))
    summary["data_collection_active_bots"] = sum(
        1 for row in rows if bool(row.get("active", False)) and bool(row.get("data_collection_active", False))
    )
    summary["data_intake_expansion_bots"] = sum(1 for row in rows if isinstance(row.get("data_intake_expansion"), dict))
    registry["summary"] = summary
    registry["updated_at_utc"] = iso_now()


def build_payload(
    *,
    project_root: Path = PROJECT_ROOT,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    bot_needs_path: Path = DEFAULT_BOT_NEEDS_PATH,
    training_quality_path: Path = DEFAULT_TRAINING_QUALITY_PATH,
    paper_profitability_path: Path = DEFAULT_PAPER_PROFITABILITY_CONTROL_PATH,
    focus_limit: int = 80,
    include_bot_ids: set[str] | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    if paper_profitability_path == DEFAULT_PAPER_PROFITABILITY_CONTROL_PATH and project_root != PROJECT_ROOT:
        paper_profitability_path = project_root / "governance" / "health" / "paper_profitability_control_latest.json"
    registry = load_json(registry_path)
    bot_needs = load_json(bot_needs_path)
    training_quality = load_json(training_quality_path)
    paper_profitability = load_json(paper_profitability_path)
    needs_index = _index_bot_needs(bot_needs)
    runtime_depth_ids = _runtime_depth_debt_ids(training_quality)
    paper_loss_controls_by_bot = _paper_loss_index(paper_profitability)
    paper_scout_collection_by_bot = _paper_scout_collection_index(paper_profitability)
    records: list[dict[str, Any]] = []
    explicit_include = bool(include_bot_ids)
    for row in _registry_rows(registry):
        bot_id = _bot_id(row).lower()
        if include_bot_ids and bot_id not in include_bot_ids:
            continue
        if explicit_include:
            if not _is_explicit_intake_target(row):
                continue
        elif not _is_active_collector(row):
            continue
        records.append(
            _row_record(
                project_root=project_root,
                row=row,
                need=needs_index.get(bot_id, {}),
                runtime_depth_debt_ids=runtime_depth_ids,
                paper_loss_index=paper_loss_controls_by_bot,
                paper_scout_collection_index=paper_scout_collection_by_bot,
            )
        )

    focus_records = _select_records(records, limit=max(int(focus_limit), 0))
    summaries = _summaries(records)
    apply_result = (
        _apply_focus_to_registry(registry_path=registry_path, registry=registry, records=records)
        if apply
        else {
            "registry_updated": False,
            "updated_bot_count": 0,
            "updated_bot_ids": [],
            "registry_backup_path": "",
        }
    )
    weak_records = [row for row in records if _as_list(row.get("weaknesses"))]
    trainable_candidates = [
        row
        for row in records
        if str(row.get("primary_need") or "") in {"targeted_quality_retrain", "top_off_walk_forward_runs"}
        and "sample_starved" not in _as_list(row.get("weaknesses"))
    ]
    collect_first = [row for row in records if str(row.get("primary_need") or "") == "collect_more_data"]
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready" if records else "missing",
        "mode": "applied" if apply else "dry_run",
        "collector_count": len(records),
        "weak_record_count": len(weak_records),
        "focus_record_count": len(focus_records),
        "trainable_candidate_count": len(trainable_candidates),
        "collect_first_count": len(collect_first),
        "summaries": summaries,
        "focus_records": focus_records,
        "trainable_candidates": _select_records(trainable_candidates, limit=20),
        "collect_first_top": _select_records(collect_first, limit=20),
        "apply_result": apply_result,
        "recommended_actions": [
            "use focus_context to route richer point-in-time observations before retraining sample-starved bots",
            "for advanced quant bots, fill registry identity, proxy context, feature surface, research label, and training-gate sections before promoting",
            "run only micro-canaries while training-runtime-control caps batch size at 1",
            "keep quality-weak or overacting bots in calibration until the next canary passes strict guards",
            "promote monitor_passing_candidate bots through promotion review instead of blind retrains",
        ],
        "artifacts": {
            "registry_path": str(registry_path),
            "bot_needs": str(bot_needs_path),
            "training_quality": str(training_quality_path),
            "paper_profitability": str(paper_profitability_path),
            "focus_path": str(DEFAULT_FOCUS_PATH),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand bot data-intake focus from label contracts, diagnostics, and bot-needs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--bot-needs-path", default=str(DEFAULT_BOT_NEEDS_PATH))
    parser.add_argument("--training-quality-path", default=str(DEFAULT_TRAINING_QUALITY_PATH))
    parser.add_argument("--paper-profitability-path", default=str(DEFAULT_PAPER_PROFITABILITY_CONTROL_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--focus-file", default=str(DEFAULT_FOCUS_PATH))
    parser.add_argument("--focus-limit", type=int, default=80)
    parser.add_argument("--include-bot-ids", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root=project_root,
        registry_path=Path(args.registry_path).expanduser(),
        bot_needs_path=Path(args.bot_needs_path).expanduser(),
        training_quality_path=Path(args.training_quality_path).expanduser(),
        paper_profitability_path=Path(args.paper_profitability_path).expanduser(),
        focus_limit=int(args.focus_limit),
        include_bot_ids=_csv_set(args.include_bot_ids) or None,
        apply=bool(args.apply),
    )
    out_path = Path(args.out_file).expanduser()
    focus_path = Path(args.focus_file).expanduser()
    write_payload(out_path, payload)
    write_payload(
        focus_path,
        {
            "timestamp_utc": payload["timestamp_utc"],
            "schema_version": payload["schema_version"],
            "mode": payload["mode"],
            "focus_records": payload["focus_records"],
            "trainable_candidates": payload["trainable_candidates"],
            "collect_first_top": payload["collect_first_top"],
            "summaries": payload["summaries"],
            "sample_starvation_recovery": {
                "sample_floor": SAMPLE_FLOOR,
                "usable_sample_goal": USABLE_SAMPLE_GOAL,
                "eligible_sequence_floor": ELIGIBLE_SEQUENCE_FLOOR,
                "eligible_sequence_goal": ELIGIBLE_SEQUENCE_GOAL,
                "label_depth_bridge_version": "label_depth_bridge_v1",
                "conversion_target_min": 0.12,
            },
            "label_repair_policy": {
                "version": "label_repair_v1",
                "required_join_mode": "point_in_time_only",
                "required_join_keys": ["bot_id", "symbol", "mode", "timestamp_utc", "snapshot_id", "decision_id"],
                "positive_rate_target": [0.35, 0.65],
                "max_acted_coverage_until_quality_passes": 0.32,
            },
            "advanced_quant_collection_sections": ADVANCED_QUANT_COLLECTION_SECTIONS,
            "advanced_quant_sample_targets": ADVANCED_QUANT_SAMPLE_TARGETS,
        },
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "training_data_intake_expansion "
            f"status={payload['overall_status']} "
            f"collectors={payload['collector_count']} "
            f"weak={payload['weak_record_count']} "
            f"trainable={payload['trainable_candidate_count']} "
            f"mode={payload['mode']}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
