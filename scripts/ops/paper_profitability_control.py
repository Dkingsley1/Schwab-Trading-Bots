#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "paper_profitability_control_latest.json"
DEFAULT_CONTROL_PATH = PROJECT_ROOT / "governance" / "health" / "paper_runtime_profitability_controls_latest.json"

LOSS_CAUSE_FAMILY = {
    "source_quality:low": "source_quality",
    "tradeability:low": "tradeability",
    "fill_quality:unknown": "fill_quality",
    "fill_quality:poor": "fill_quality",
    "fill_quality:fair": "fill_quality",
    "spread_regime:unknown": "spread_quality",
    "spread_regime:wide": "spread_quality",
    "event_proximity:low": "catalyst_confirmation",
    "conflict:low": "portfolio_conflict",
}

CATALYST_PROFILES = {
    "aggressive",
    "intraday_aggressive",
    "swing_aggressive",
    "schwab_futures",
    "crypto_futures",
}

UPGRADE_LANE_IDS = [
    "outcome_weighted_training",
    "per_sleeve_profit_score",
    "dynamic_sizing",
    "regime_specific_promotion",
    "loser_quarantine",
    "exit_intelligence",
    "execution_aware_alpha",
    "portfolio_conflict_control",
    "confirmation_bias_control",
    "profit_harvest_intelligence",
]

CONFIRMATION_BIAS_CAUSES = {
    "conflict:low",
    "event_proximity:low",
    "fill_quality:unknown",
    "fill_quality:poor",
    "fill_quality:fair",
    "source_quality:low",
    "spread_regime:unknown",
    "spread_regime:wide",
    "tradeability:low",
}

CONFIRMATION_EVIDENCE_CHANNELS = [
    "source_quality",
    "execution_quality",
    "spread_quality",
    "cross_asset_confirmation",
    "event_catalyst_confirmation",
    "portfolio_conflict_clearance",
]

PROFITABILITY_HARDENING_ACTIONS = [
    "stop_new_entries_in_worst_sleeves",
    "accelerate_unrealized_drag_reduction",
    "require_independent_evidence_before_action",
    "deweight_losing_profile_strategy_pairs",
    "expand_scout_labels_for_profitability_feedback",
]

PROFITABILITY_REALIZATION_LEVERS = [
    "stop_weak_sleeve_drag",
    "scale_winning_sleeves",
    "harvest_regret_control_lift",
    "laddered_partial_exit_policy",
    "strategy_level_promotion",
    "punitive_loss_attribution",
    "unrealized_loser_training_debt",
    "harvest_force_guard",
]

PROFITABILITY_COMPOUNDING_AUTOPILOT_ACTIONS = [
    "freeze_weak_sleeve_fresh_adds",
    "reconcile_reduce_only_harvest_intents",
    "run_harvest_regret_replay",
    "promote_winning_strategy_pairs",
    "assign_unrealized_loser_training_debt",
    "tighten_punitive_loss_attribution",
    "scale_clean_winning_sleeves",
    "hold_profit_as_paper_cash_when_force_guard_blocks",
]

HARVEST_REPLAY_OUTCOME_LABELS = [
    "trim_too_early_bucket",
    "trim_too_late_bucket",
    "partial_harvest_helped_bucket",
    "runner_deserved_room_bucket",
    "held_too_long_giveback_bucket",
    "reduce_only_fill_quality_bucket",
    "post_trim_followthrough_bucket",
    "no_trim_counterfactual_outcome",
]

QUANT_STRATEGY_EXPANSION_FAMILIES = [
    {
        "family_id": "volatility_risk_premium_harvesting",
        "preferred_sleeves": ["volatility", "volatility_arbitrage", "variance_volatility_swaps"],
        "purpose": "harvest implied-versus-realized volatility carry only when event, hedge-cost, and fill evidence agree",
        "required_labels": [
            "implied_realized_vol_gap",
            "tenor_carry_bucket",
            "event_volatility_flag",
            "hedge_cost_bucket",
            "harvest_regret_bucket",
        ],
    },
    {
        "family_id": "options_convexity_muscle",
        "preferred_sleeves": ["volatility", "single_name_options_event", "gamma_scalping", "options_on_futures_aggressive"],
        "purpose": "stage convexity candidates using gamma, vanna, charm, skew, and event-window evidence",
        "required_labels": [
            "gamma_convexity_bucket",
            "vanna_charm_bucket",
            "skew_reset_flag",
            "event_window_risk",
            "fill_quality_bucket",
        ],
    },
    {
        "family_id": "options_income_muscle",
        "preferred_sleeves": ["dividend", "dividend_income", "options_flow", "structured_products"],
        "purpose": "separate premium-quality income ideas from assignment, dividend, borrow, and tail-loss risk",
        "required_labels": [
            "premium_quality_bucket",
            "assignment_risk_bucket",
            "dividend_event_flag",
            "borrow_pressure_bucket",
            "tail_loss_bucket",
        ],
    },
    {
        "family_id": "volatility_arbitrage_muscle",
        "preferred_sleeves": ["volatility", "volatility_arbitrage", "dispersion_trading", "variance_volatility_swaps"],
        "purpose": "convert surface, dispersion, and event-vol dislocations into collection-only arbitrage candidates",
        "required_labels": [
            "surface_dislocation_bucket",
            "iv_rv_spread_bucket",
            "dispersion_gap_bucket",
            "event_vol_reset_flag",
            "hedge_slippage_bucket",
        ],
    },
    {
        "family_id": "options_risk_intelligence_v2",
        "preferred_sleeves": ["volatility", "options_flow", "single_name_options_event", "options_on_futures_aggressive"],
        "purpose": "validate options Greek, margin, assignment, liquidity, source, and replay risk before option sleeves widen",
        "required_labels": [
            "greek_margin_bucket",
            "assignment_risk_bucket",
            "spread_quality_bucket",
            "option_chain_source_quality",
            "replay_determinism_status",
        ],
    },
    {
        "family_id": "cliquet_ratchet_options",
        "preferred_sleeves": ["default", "structured_products", "barrier_lookback_options", "quant_pricing_models"],
        "purpose": "track reset, local-cap, global-floor, and ratchet payoff behavior as path-dependent collection evidence",
        "required_labels": [
            "reset_window_bucket",
            "local_cap_global_floor_state",
            "path_dependency_bucket",
            "gap_risk_bucket",
            "pricing_model_dispersion",
        ],
    },
    {
        "family_id": "quanto_compo_options",
        "preferred_sleeves": ["default", "rainbow_options", "international_macro", "cross_asset_basis_training"],
        "purpose": "stage cross-currency and cross-asset option payoffs where FX beta and correlation alter the edge",
        "required_labels": [
            "fx_beta_bucket",
            "cross_currency_basis_bucket",
            "correlation_skew_bucket",
            "quanto_drift_adjustment",
            "hedge_cost_bucket",
        ],
    },
    {
        "family_id": "vix_options_on_volatility",
        "preferred_sleeves": ["volatility", "variance_volatility_swaps", "black_swan_hedging"],
        "purpose": "collect volatility-option evidence around VIX term structure, vol-of-vol, event shocks, and roll decay",
        "required_labels": [
            "vix_term_structure_bucket",
            "vvix_proxy_bucket",
            "contango_backwardation_state",
            "event_shock_flag",
            "roll_decay_bucket",
        ],
    },
    {
        "family_id": "dividend_borrow_early_exercise_intelligence",
        "preferred_sleeves": ["dividend", "dividend_income", "options_flow", "repo_securities_lending"],
        "purpose": "prevent income and convexity sleeves from ignoring dividend, borrow, assignment, and early-exercise risk",
        "required_labels": [
            "ex_dividend_window",
            "borrow_fee_bucket",
            "hard_to_borrow_flag",
            "early_exercise_moneyness",
            "assignment_outcome_bucket",
        ],
    },
    {
        "family_id": "skew_surface_arbitrage",
        "preferred_sleeves": ["volatility", "volatility_arbitrage", "second_third_order_greeks", "vanna_volga_hedging"],
        "purpose": "find skew and surface dislocations only when arbitrage-free, source, and fill checks are clean",
        "required_labels": [
            "skew_slope_bucket",
            "smile_curvature_bucket",
            "term_structure_bucket",
            "butterfly_arbitrage_flag",
            "surface_source_quality",
        ],
    },
    {
        "family_id": "calendar_diagonal_spread_intelligence",
        "preferred_sleeves": ["default", "compound_options", "single_name_options_event", "options_flow"],
        "purpose": "collect term-structure and roll evidence before any calendar or diagonal spread candidate widens",
        "required_labels": [
            "front_back_vol_gap",
            "theta_decay_bucket",
            "roll_timing_bucket",
            "event_calendar_flag",
            "spread_liquidity_bucket",
        ],
    },
    {
        "family_id": "gamma_theta_scalping_optimizer",
        "preferred_sleeves": ["volatility", "gamma_scalping", "market_making_liquidity", "order_flow_market_microstructure"],
        "purpose": "balance gamma capture against theta bleed, rebalance cost, spread quality, and realized-vol follow-through",
        "required_labels": [
            "gamma_capture_bucket",
            "theta_bleed_bucket",
            "rebalance_cost_bucket",
            "realized_vol_followthrough",
            "spread_crossing_cost",
        ],
    },
    {
        "family_id": "dispersion_basket_optimizer",
        "preferred_sleeves": ["default", "dispersion_trading", "rainbow_options", "portfolio_construction"],
        "purpose": "rank index-versus-single-name dispersion baskets by correlation, liquidity, crowding, and hedge cost",
        "required_labels": [
            "index_single_name_vol_gap",
            "correlation_realization_bucket",
            "basket_liquidity_bucket",
            "sector_concentration_bucket",
            "hedge_slippage_bucket",
        ],
    },
    {
        "family_id": "callable_autocallable_payoff_monitor",
        "preferred_sleeves": ["default", "structured_products", "barrier_lookback_options", "xva_counterparty_margin"],
        "purpose": "monitor callable and autocallable payoff states before structured-product proxies influence sizing",
        "required_labels": [
            "autocall_trigger_distance",
            "coupon_memory_state",
            "barrier_touch_risk",
            "issuer_credit_bucket",
            "secondary_liquidity_bucket",
        ],
    },
    {
        "family_id": "bermudan_exercise_monte_carlo_policy",
        "preferred_sleeves": ["bond", "quant_pricing_models", "martingale_flow_pricing", "swaptions"],
        "purpose": "use Monte Carlo exercise-policy evidence for Bermudan-style optionality without execution authority",
        "required_labels": [
            "exercise_boundary_bucket",
            "continuation_value_gap",
            "path_count_quality",
            "model_dispersion_bucket",
            "exercise_regret_bucket",
        ],
    },
    {
        "family_id": "market_neutral_pairs",
        "preferred_sleeves": ["default", "stat_arb_market_neutral", "pairs_correlation"],
        "purpose": "add hedged relative-value alpha without depending on broad market direction",
        "required_labels": [
            "spread_zscore",
            "cointegration_stability",
            "borrow_or_proxy_cost",
            "pair_break_risk",
            "exit_regret_bucket",
        ],
    },
    {
        "family_id": "intraday_mean_reversion",
        "preferred_sleeves": ["intraday_aggressive", "market_making_liquidity", "high_frequency_market_making"],
        "purpose": "capture short-horizon overextension only when spread, fill, and liquidity evidence agree",
        "required_labels": [
            "micro_trend_exhaustion",
            "spread_regime",
            "fill_quality_bucket",
            "order_flow_toxicity",
            "no_trade_counterfactual_outcome",
        ],
    },
    {
        "family_id": "volatility_risk_premium",
        "preferred_sleeves": ["volatility", "volatility_arbitrage", "options_on_futures"],
        "purpose": "separate harvestable vol premium from event-vol danger and poor fill regimes",
        "required_labels": [
            "implied_realized_vol_gap",
            "event_volatility_flag",
            "gamma_exposure_bucket",
            "spread_regime",
            "harvest_regret_bucket",
        ],
    },
    {
        "family_id": "carry_term_structure",
        "preferred_sleeves": ["crypto_futures", "crypto_futures_basis", "futures_rates_curve"],
        "purpose": "use carry, funding, and curve structure as confirmation rather than blind trend chasing",
        "required_labels": [
            "basis_or_carry_norm",
            "funding_regime",
            "roll_yield_bucket",
            "liquidity_depth_bucket",
            "crowding_risk_bucket",
        ],
    },
    {
        "family_id": "cross_asset_confirmation",
        "preferred_sleeves": ["default", "international_macro", "cross_asset_basis_training"],
        "purpose": "raise conviction only when cross-asset evidence agrees with the sleeve signal",
        "required_labels": [
            "correlation_break_flag",
            "macro_confirmation_norm",
            "sector_confirmation_norm",
            "conflict_score",
            "independent_evidence_channel_count",
        ],
    },
    {
        "family_id": "liquidity_microstructure",
        "preferred_sleeves": ["market_making_liquidity", "order_flow_market_microstructure", "order_flow_toxicity"],
        "purpose": "teach the system when liquidity is tradable and when visible flow is toxic",
        "required_labels": [
            "quoted_spread_bucket",
            "effective_spread_bucket",
            "depth_imbalance",
            "toxicity_score",
            "fill_slippage_bucket",
        ],
    },
    {
        "family_id": "event_reaction",
        "preferred_sleeves": ["earnings_event", "event_intelligence", "futures_event_reaction"],
        "purpose": "canary event reaction without letting headlines override source and fill quality",
        "required_labels": [
            "event_surprise_norm",
            "pre_event_drift",
            "post_event_followthrough",
            "headline_source_quality",
            "event_fade_risk",
        ],
    },
    {
        "family_id": "defensive_rotation",
        "preferred_sleeves": ["conservative", "bond", "cash_rotation_tactical"],
        "purpose": "give the portfolio a calm rotation lane when aggressive sleeves are under containment",
        "required_labels": [
            "risk_off_regime",
            "rates_confirmation",
            "credit_spread_pressure",
            "defensive_relative_strength",
            "opportunity_cost_bucket",
        ],
    },
]

PAPER_HARVEST_INFRABOTS = [
    {
        "bot_id": "paper_realized_share_goalkeeper_infrabot",
        "bot_role": "paper_profit_harvest_infrastructure_bot",
        "mission": "track realized-share progress against each sleeve daily goal",
    },
    {
        "bot_id": "paper_reduce_only_fill_reconciler_infrabot",
        "bot_role": "paper_profit_harvest_infrastructure_bot",
        "mission": "verify reduce-only SELL intents produce paper fills and never authorize live execution",
    },
    {
        "bot_id": "paper_runner_protection_sentinel_infrabot",
        "bot_role": "paper_profit_harvest_infrastructure_bot",
        "mission": "stop overharvesting when continuation and regret risk say the runner deserves room",
    },
    {
        "bot_id": "paper_harvest_intent_staleness_infrabot",
        "bot_role": "paper_profit_harvest_infrastructure_bot",
        "mission": "retire stale harvest intents and force a fresh control refresh before reuse",
    },
    {
        "bot_id": "paper_sleeve_profit_explainer_infrabot",
        "bot_role": "paper_profit_harvest_infrastructure_bot",
        "mission": "explain why each sleeve harvested, held, blocked adds, or protected a runner",
    },
]

FINANCIAL_APLUS_MIN_NET_PNL = 50_000.0
FINANCIAL_APLUS_MIN_REALIZED_PNL = 1_000.0
FINANCIAL_APLUS_MIN_CHANGE_PNL = 10_000.0
FINANCIAL_APLUS_MIN_EXECUTIONS = 100
RAW_OP_PROFILE_MATERIALITY_FLOOR = 250.0
RAW_OP_PROFILE_MATERIALITY_CAP = 750.0
RAW_OP_PROFILE_MATERIALITY_SHARE = 0.015
RAW_OP_PROFILE_MIN_GRADEABLE_EXECUTIONS = 25
RAW_OP_STRATEGY_MATERIALITY_FLOOR = 100.0
RAW_OP_STRATEGY_MATERIALITY_CAP = 525.0
RAW_OP_STRATEGY_MATERIALITY_SHARE = 0.010
PROFIT_HARVEST_MIN_UNREALIZED_PNL = 1_000.0
PROFIT_HARVEST_MIN_NET_PNL = 1_000.0
PROFIT_HARVEST_SMALL_MIN_UNREALIZED_PNL = 50.0
PROFIT_HARVEST_SMALL_MIN_NET_PNL = 75.0
PROFIT_HARVEST_SMALL_MIN_EXECUTIONS = 20
PROFIT_HARVEST_SMALL_MIN_UNREALIZED_SHARE = 0.80
PROFIT_HARVEST_SMALL_TARGET_REALIZED_SHARE = 0.25
PROFIT_HARVEST_SMALL_MAX_UNREALIZED_SHARE = 0.82
PROFIT_HARVEST_SMALL_MAX_TRIM_FRACTION = 0.24
PROFIT_HARVEST_TARGET_REALIZED_SHARE = 0.35
PROFIT_HARVEST_MAX_UNREALIZED_SHARE = 0.70
PROFIT_HARVEST_REPLAY_LOOKAHEAD_MINUTES = [30, 90, 240]
DAILY_SLEEVE_HARVEST_MIN_TARGET_PNL = 25.0
DAILY_SLEEVE_HARVEST_MAX_TARGET_SHARE_OF_UNREALIZED = 0.38
DAILY_SLEEVE_HARVEST_INTENT_LIMIT = 32
DAILY_SLEEVE_TARGET_RAISE_MIN_MULTIPLIER = 1.08
DAILY_SLEEVE_TARGET_RAISE_MAX_MULTIPLIER = 1.35
PROFIT_HARVEST_APLUSPLUS_MIN_SCORE = 0.98
PROFIT_HARVEST_APLUSPLUS_MIN_REALIZED_PROGRESS = 0.98
PROFIT_HARVEST_APLUSPLUS_MIN_UNREALIZED_CONTROL = 0.96
PROFIT_HARVEST_APLUSPLUS_MIN_REGRET_CONTROL = 0.80
PROFIT_HARVEST_APLUS_MIN_SCORE = 0.92
PROFIT_HARVEST_APLUS_MIN_REALIZED_PROGRESS = 0.90
PROFIT_HARVEST_APLUS_MIN_UNREALIZED_CONTROL = 0.88
PROFIT_HARVEST_C_MIN_SCORE = 0.58
PROFIT_HARVEST_RAW_C_RESCUE_MAX_CREDIT = 0.025
PROFIT_HARVEST_RAW_C_RESCUE_MIN_LEDGER_POSITIONS = 100
PROFIT_HARVEST_RAW_B_RESCUE_MAX_CREDIT = 0.12
PROFIT_HARVEST_RAW_B_RESCUE_MIN_LEDGER_POSITIONS = 100
PROFIT_HARVEST_CARRY_FORWARD_STATUSES = {
    "current",
    "current_live_no_fills",
}
PROFIT_HARVEST_CARRY_FORWARD_LIVE_STATES = {
    "running",
    "forex_weekend_closed",
}
PROFIT_TIER_RULES = [
    {
        "tier": "tier_1_lock_seed_profit",
        "min_strategy_net_pnl": 250.0,
        "trim_fraction_norm": 0.15,
        "mode": "partial_trim",
    },
    {
        "tier": "tier_2_pay_the_system",
        "min_strategy_net_pnl": 1_000.0,
        "trim_fraction_norm": 0.25,
        "mode": "partial_trim",
    },
    {
        "tier": "tier_3_protect_runner",
        "min_strategy_net_pnl": 3_000.0,
        "trim_fraction_norm": 0.35,
        "mode": "trim_and_trail",
    },
    {
        "tier": "tier_4_banked_winner",
        "min_strategy_net_pnl": 5_000.0,
        "trim_fraction_norm": 0.45,
        "mode": "bank_then_trail",
    },
]
PROFIT_HARVEST_PROFILE_PARAMS = {
    "crypto_futures": {
        "target_realized_share": 0.32,
        "max_unrealized_share": 0.76,
        "trend_prior": 0.66,
        "min_trim_fraction": 0.10,
        "max_trim_fraction": 0.44,
    },
    "fx": {
        "target_realized_share": 0.38,
        "max_unrealized_share": 0.66,
        "trend_prior": 0.52,
        "min_trim_fraction": 0.14,
        "max_trim_fraction": 0.50,
    },
    "default": {
        "target_realized_share": 0.35,
        "max_unrealized_share": 0.70,
        "trend_prior": 0.56,
        "min_trim_fraction": 0.12,
        "max_trim_fraction": 0.48,
    },
    "aggressive": {
        "target_realized_share": 0.42,
        "max_unrealized_share": 0.62,
        "trend_prior": 0.58,
        "min_trim_fraction": 0.16,
        "max_trim_fraction": 0.54,
    },
    "intraday_aggressive": {
        "target_realized_share": 0.45,
        "max_unrealized_share": 0.60,
        "trend_prior": 0.50,
        "min_trim_fraction": 0.18,
        "max_trim_fraction": 0.56,
    },
    "swing_aggressive": {
        "target_realized_share": 0.40,
        "max_unrealized_share": 0.64,
        "trend_prior": 0.62,
        "min_trim_fraction": 0.14,
        "max_trim_fraction": 0.52,
    },
}

SCOUT_PROFITABILITY_CONTEXT = [
    "paper_profile_strategy_pair",
    "paper_position_state",
    "paper_trade_outcome",
    "paper_loss_cause",
    "source_quality",
    "fill_quality",
    "spread_quality",
    "cross_asset_confirmation",
    "event_catalyst_confirmation",
    "portfolio_conflict",
    "exit_drag_trace",
    "no_trade_counterfactual",
]

SCOUT_PROFITABILITY_LABEL_OUTPUTS = [
    "paper_loss_cause",
    "paper_profile",
    "paper_strategy",
    "paper_unrealized_drag_bucket",
    "paper_exit_quality_bucket",
    "paper_profit_harvest_bucket",
    "paper_realized_conversion_bucket",
    "paper_harvest_regret_bucket",
    "paper_trend_continuation_bucket",
    "paper_realized_conversion_skill_bucket",
    "post_trim_followthrough_bucket",
    "entry_evidence_gate_result",
    "confirmation_bias_bucket",
    "independent_evidence_channel_count",
    "source_fill_spread_quality_bucket",
    "no_trade_counterfactual_outcome",
    "false_confirmation_source",
]

UPPER_LAYER_TRAINING_TARGETS = [
    "master_trend_bot",
    "master_mean_revert_bot",
    "master_shock_bot",
    "grand_master_bot",
]

SUB_BOT_ACCURACY_TARGET_CONTRACT = {
    "active": True,
    "desired_out_of_sample_accuracy_band": {
        "min": 0.80,
        "max": 0.90,
    },
    "target_is_not_forced": True,
    "min_walk_forward_runs": 12,
    "min_regime_count": 3,
    "min_oos_samples": 300,
    "max_train_test_accuracy_gap": 0.08,
    "max_single_side_action_share": 0.70,
    "min_side_precision": 0.50,
    "min_calibration_score": 0.68,
    "max_duplicate_alpha_overlap_norm": 0.82,
    "accept_only_if": [
        "walk_forward_out_of_sample_accuracy_in_80_90_band",
        "train_test_gap_at_or_below_0_08",
        "no_single_side_or_overacted_collapse",
        "label_balance_and_side_precision_pass",
        "cross_regime_validation_passes",
        "duplicate_alpha_overlap_below_cap",
        "paper_profitability_drag_controls_are_clean_or_deweighted",
    ],
    "reject_if": [
        "accuracy_above_0_90_without_large_cross_regime_sample",
        "accuracy_above_0_90_with_train_test_gap_breach",
        "positive_or_negative_label_collapse",
        "overacted_or_one_sided_decision_surface",
        "future_leakage_or_same_bar_outcome_feature_detected",
        "duplicate_alpha_overlap_cluster_is_high",
    ],
    "operator_note": "80-90% is treated as a clean out-of-sample target band, not a memorization target.",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def _latest_history_row(paper: dict[str, Any]) -> dict[str, Any]:
    rows = paper.get("history_daily_series") if isinstance(paper.get("history_daily_series"), list) else []
    for row in reversed(rows):
        if isinstance(row, dict):
            return row
    return {}


def _normal_profile(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _strategy_bot_id(strategy: str) -> str:
    text = str(strategy or "").strip()
    if "::" in text:
        text = text.split("::", 1)[1]
    return text.strip().lower()


def _loss_causes(row: dict[str, Any]) -> list[dict[str, Any]]:
    causes = row.get("top_loss_causes") if isinstance(row.get("top_loss_causes"), list) else []
    return [cause for cause in causes if isinstance(cause, dict)]


def _cause_names(row: dict[str, Any]) -> list[str]:
    return [str(cause.get("cause") or "").strip().lower() for cause in _loss_causes(row) if str(cause.get("cause") or "").strip()]


def _quality_families(cause_names: list[str]) -> list[str]:
    families = []
    for cause in cause_names:
        family = LOSS_CAUSE_FAMILY.get(cause)
        if family:
            families.append(family)
    return ordered_unique(families)


def _confirmation_bias_score(cause_names: list[str], *, drag: float, net: float, win_rate: float | None) -> float:
    if not cause_names:
        return 0.0
    bias_hits = sum(1 for cause in cause_names if cause in CONFIRMATION_BIAS_CAUSES)
    cause_component = _clamp(bias_hits / max(len(CONFIRMATION_BIAS_CAUSES), 1))
    breadth_component = _clamp(bias_hits / max(len(cause_names), 1))
    net_component = _clamp(abs(min(net, 0.0)) / 1500.0)
    win_component = _clamp(max(0.42 - float(win_rate), 0.0) / 0.42) if win_rate is not None else 0.0
    return _clamp(
        0.34 * cause_component
        + 0.26 * breadth_component
        + 0.22 * _clamp(drag)
        + 0.10 * net_component
        + 0.08 * win_component
    )


def _profit_score(row: dict[str, Any], drag: float) -> float:
    net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
    unrealized = _safe_float(row.get("ending_unrealized_pnl_total"), 0.0)
    win_rate_raw = row.get("win_rate")
    win_rate = _safe_float(win_rate_raw, 0.5) if win_rate_raw is not None else 0.5
    executions = _safe_int(row.get("executions"), 0)
    net_component = _clamp(0.50 + (net / 1800.0), 0.0, 1.0)
    unrealized_component = _clamp(0.50 + (unrealized / 1800.0), 0.0, 1.0)
    confidence = _clamp(executions / 80.0)
    return _clamp(
        (0.34 * net_component)
        + (0.20 * unrealized_component)
        + (0.24 * _clamp(win_rate))
        + (0.14 * (1.0 - drag))
        + (0.08 * confidence)
    )


def _profit_grade(score: float) -> str:
    if score >= 0.92:
        return "A+"
    if score >= 0.78:
        return "A"
    if score >= 0.62:
        return "B"
    if score >= 0.46:
        return "C"
    if score >= 0.30:
        return "D"
    return "F"


LOW_GRADE_VALUES = {"D", "F"}


def _financial_grade(
    *,
    net_sum: float,
    realized_sum: float,
    unrealized_sum: float,
    change_vs_previous_day: float,
    executions: int,
) -> str:
    if (
        net_sum >= FINANCIAL_APLUS_MIN_NET_PNL
        and realized_sum >= FINANCIAL_APLUS_MIN_REALIZED_PNL
        and unrealized_sum >= 0.0
        and change_vs_previous_day >= FINANCIAL_APLUS_MIN_CHANGE_PNL
        and executions >= FINANCIAL_APLUS_MIN_EXECUTIONS
    ):
        return "A+"
    if net_sum >= 0.0:
        return "A"
    if net_sum >= -1000.0:
        return "B"
    if net_sum >= -5000.0:
        return "C"
    return "D"


def _operational_outcome_grade(*, weak_count: int, strategy_count: int) -> str:
    if weak_count == 0 and strategy_count == 0:
        return "A+"
    if weak_count <= 2 and strategy_count <= 5:
        return "A"
    if weak_count <= 5 and strategy_count <= 12:
        return "B"
    if weak_count <= 9 and strategy_count <= 24:
        return "C"
    return "D"


def _profile_materiality_threshold(net_sum: float) -> float:
    return round(
        min(
            RAW_OP_PROFILE_MATERIALITY_CAP,
            max(RAW_OP_PROFILE_MATERIALITY_FLOOR, abs(float(net_sum)) * RAW_OP_PROFILE_MATERIALITY_SHARE),
        ),
        6,
    )


def _strategy_materiality_threshold(net_sum: float) -> float:
    return round(
        min(
            RAW_OP_STRATEGY_MATERIALITY_CAP,
            max(RAW_OP_STRATEGY_MATERIALITY_FLOOR, abs(float(net_sum)) * RAW_OP_STRATEGY_MATERIALITY_SHARE),
        ),
        6,
    )


def _raw_operational_materiality_filter(
    *,
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
    net_sum: float,
) -> dict[str, Any]:
    profile_threshold = _profile_materiality_threshold(net_sum)
    strategy_threshold = _strategy_materiality_threshold(net_sum)
    gradeable_profiles: dict[str, dict[str, Any]] = {}
    probation_profiles: list[dict[str, Any]] = []
    for profile, control in active_profile_controls.items():
        if not isinstance(control, dict):
            continue
        net = _safe_float(control.get("ending_net_pnl_total"), 0.0)
        executions = _safe_int(control.get("executions"), 0)
        gradeable = bool(
            net <= -profile_threshold
            or executions >= RAW_OP_PROFILE_MIN_GRADEABLE_EXECUTIONS
        )
        if gradeable:
            gradeable_profiles[profile] = control
            continue
        probation_profiles.append(
            {
                "profile": str(profile),
                "ending_net_pnl_total": round(net, 6),
                "executions": executions,
                "drag_score_norm": _safe_float(control.get("drag_score"), 0.0),
                "profit_grade": str(control.get("profit_grade") or ""),
                "reason": "below raw operational materiality and sample floor",
            }
        )

    gradeable_strategies: list[dict[str, Any]] = []
    probation_strategies: list[dict[str, Any]] = []
    for row in strategy_controls:
        if not isinstance(row, dict):
            continue
        net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        if net <= -strategy_threshold:
            gradeable_strategies.append(row)
            continue
        probation_strategies.append(
            {
                "profile": str(row.get("profile") or ""),
                "strategy": str(row.get("strategy") or ""),
                "bot_id": str(row.get("bot_id") or ""),
                "ending_net_pnl_total": round(net, 6),
                "score_penalty_norm": _safe_float(row.get("score_penalty_norm"), 0.0),
                "reason": "below raw operational strategy materiality",
            }
        )

    return {
        "active": True,
        "mode": "raw_operational_materiality_filter",
        "net_pnl_reference": round(float(net_sum), 6),
        "profile_loss_materiality_threshold": profile_threshold,
        "profile_min_gradeable_executions": RAW_OP_PROFILE_MIN_GRADEABLE_EXECUTIONS,
        "strategy_loss_materiality_threshold": strategy_threshold,
        "gross_weak_profile_count": len(active_profile_controls),
        "gradeable_weak_profile_count": len(gradeable_profiles),
        "probationary_weak_profile_count": len(probation_profiles),
        "gross_strategy_control_count": len(strategy_controls),
        "gradeable_strategy_control_count": len(gradeable_strategies),
        "probationary_strategy_control_count": len(probation_strategies),
        "probationary_profiles": probation_profiles,
        "probationary_strategy_pairs": probation_strategies,
        "grade_basis": "materiality_adjusted_raw_counts",
        "safety_rule": "probationary losses stay blocked/deweighted but do not hold the raw operational letter hostage until material or sufficiently sampled",
        "_gradeable_profile_controls": gradeable_profiles,
        "_gradeable_strategy_controls": gradeable_strategies,
    }


def _profile_loss_contained(control: dict[str, Any]) -> bool:
    if _profile_loss_protected(control):
        return True
    action = str(control.get("action") or "").strip().lower()
    runtime_policy = control.get("runtime_policy") if isinstance(control.get("runtime_policy"), dict) else {}
    loser = control.get("loser_quarantine") if isinstance(control.get("loser_quarantine"), dict) else {}
    return bool(
        action == "quarantine_new_entries"
        and _safe_int(control.get("new_entry_cap"), 1) == 0
        and (
            bool(control.get("a_plus_recovery_mode", False))
            or bool(runtime_policy.get("a_plus_lock_in", False))
            or bool(loser.get("block_new_entries", False))
        )
    )


def _profile_loss_protected(control: dict[str, Any]) -> bool:
    action = str(control.get("action") or "").strip().lower()
    runtime_policy = control.get("runtime_policy") if isinstance(control.get("runtime_policy"), dict) else {}
    loser = control.get("loser_quarantine") if isinstance(control.get("loser_quarantine"), dict) else {}
    return bool(
        action == "quarantine_new_entries"
        and _safe_int(control.get("new_entry_cap"), 1) == 0
        and (
            bool(control.get("a_plus_recovery_mode", False))
            or bool(control.get("protective_tightening_mode", False))
            or bool(runtime_policy.get("a_plus_lock_in", False))
            or bool(runtime_policy.get("protective_tightening_lock", False))
            or bool(loser.get("block_new_entries", False))
        )
    )


def _strategy_loss_contained(row: dict[str, Any]) -> bool:
    if _strategy_loss_protected(row):
        return True
    loser = (row.get("upgrade_contracts") or {}).get("loser_quarantine") if isinstance(row.get("upgrade_contracts"), dict) else {}
    return bool(
        str(row.get("mode") or "").strip().lower() == "paper_quarantine"
        and _safe_int(row.get("new_entry_cap"), 1) == 0
        and (
            bool(row.get("block_new_entries", False))
            or bool(loser.get("block_new_entries", False))
            or bool(row.get("a_plus_recovery_mode", False))
        )
    )


def _strategy_loss_protected(row: dict[str, Any]) -> bool:
    loser = (row.get("upgrade_contracts") or {}).get("loser_quarantine") if isinstance(row.get("upgrade_contracts"), dict) else {}
    return bool(
        str(row.get("mode") or "").strip().lower() == "paper_quarantine"
        and _safe_int(row.get("new_entry_cap"), 1) == 0
        and (
            bool(row.get("block_new_entries", False))
            or bool(loser.get("block_new_entries", False))
            or bool(row.get("a_plus_recovery_mode", False))
            or bool(row.get("protective_tightening_mode", False))
        )
    )


def _profile_requires_full_protection(control: dict[str, Any]) -> bool:
    grade = str(control.get("profit_grade") or "").strip().upper()
    drag = _safe_float(control.get("drag_score"), 0.0)
    net = _safe_float(control.get("ending_net_pnl_total"), 0.0)
    return bool(grade in LOW_GRADE_VALUES or drag >= 0.75 or net <= -RAW_OP_PROFILE_MATERIALITY_FLOOR)


def _raw_operational_containment_filter(
    *,
    gradeable_profile_controls: dict[str, dict[str, Any]],
    gradeable_strategy_controls: list[dict[str, Any]],
    base_grade: str,
) -> dict[str, Any]:
    contained_profiles: list[dict[str, Any]] = []
    active_profiles: dict[str, dict[str, Any]] = {}
    for profile, control in gradeable_profile_controls.items():
        if not isinstance(control, dict):
            continue
        row = {
            "profile": str(profile),
            "ending_net_pnl_total": _safe_float(control.get("ending_net_pnl_total"), 0.0),
            "drag_score_norm": _safe_float(control.get("drag_score"), 0.0),
            "profit_grade": str(control.get("profit_grade") or ""),
            "action": str(control.get("action") or ""),
            "new_entry_cap": _safe_int(control.get("new_entry_cap"), 1),
        }
        if _profile_loss_contained(control):
            contained_profiles.append(row)
        else:
            active_profiles[profile] = control

    contained_strategies: list[dict[str, Any]] = []
    active_strategies: list[dict[str, Any]] = []
    for row in gradeable_strategy_controls:
        if not isinstance(row, dict):
            continue
        view = {
            "profile": str(row.get("profile") or ""),
            "strategy": str(row.get("strategy") or ""),
            "bot_id": str(row.get("bot_id") or ""),
            "ending_net_pnl_total": _safe_float(row.get("ending_net_pnl_total"), 0.0),
            "mode": str(row.get("mode") or ""),
            "new_entry_cap": _safe_int(row.get("new_entry_cap"), 1),
        }
        if _strategy_loss_contained(row):
            contained_strategies.append(view)
        else:
            active_strategies.append(row)

    active_grade = _operational_outcome_grade(
        weak_count=len(active_profiles),
        strategy_count=len(active_strategies),
    )
    return {
        "active": True,
        "mode": "raw_operational_containment_filter",
        "base_grade_before_containment": str(base_grade or ""),
        "contained_grade": active_grade,
        "grade_basis": "materiality_adjusted_and_containment_qualified_raw_counts",
        "gradeable_weak_profile_count_before_containment": len(gradeable_profile_controls),
        "active_weak_profile_count_after_containment": len(active_profiles),
        "contained_weak_profile_count": len(contained_profiles),
        "gradeable_strategy_control_count_before_containment": len(gradeable_strategy_controls),
        "active_strategy_control_count_after_containment": len(active_strategies),
        "contained_strategy_control_count": len(contained_strategies),
        "contained_profiles": contained_profiles,
        "contained_strategy_pairs": contained_strategies,
        "active_profiles": [
            {
                "profile": str(profile),
                "ending_net_pnl_total": _safe_float(control.get("ending_net_pnl_total"), 0.0),
                "action": str(control.get("action") or ""),
            }
            for profile, control in active_profiles.items()
        ],
        "active_strategy_pairs": [
            {
                "profile": str(row.get("profile") or ""),
                "strategy": str(row.get("strategy") or ""),
                "ending_net_pnl_total": _safe_float(row.get("ending_net_pnl_total"), 0.0),
                "mode": str(row.get("mode") or ""),
            }
            for row in active_strategies
        ],
        "safety_rule": "contained raw losses remain visible, blocked, and training-weighted, but no-new-entry containment removes them from active raw operational blockers",
        "_active_profile_controls": active_profiles,
        "_active_strategy_controls": active_strategies,
    }


def _operational_next_grade_target(grade: str) -> dict[str, Any]:
    current = str(grade or "D").strip().upper()
    if current == "D":
        return {"next_grade": "C", "max_weak_profiles": 9, "max_strategy_controls": 24}
    if current == "C":
        return {"next_grade": "B", "max_weak_profiles": 5, "max_strategy_controls": 12}
    if current == "B":
        return {"next_grade": "A", "max_weak_profiles": 2, "max_strategy_controls": 5}
    if current == "A":
        return {"next_grade": "A+", "max_weak_profiles": 0, "max_strategy_controls": 0}
    return {"next_grade": "A+", "max_weak_profiles": 0, "max_strategy_controls": 0}


def _raw_operational_grade_lift_contract(
    *,
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
    raw_operational_grade: str,
) -> dict[str, Any]:
    weak_count = len(active_profile_controls)
    strategy_count = len(strategy_controls)
    target = _operational_next_grade_target(raw_operational_grade)
    target_weak = _safe_int(target.get("max_weak_profiles"), 0)
    target_strategies = _safe_int(target.get("max_strategy_controls"), 0)
    profile_clear_count = max(weak_count - target_weak, 0)
    strategy_clear_count = max(strategy_count - target_strategies, 0)

    profile_rows = sorted(
        [
            {
                "profile": str(profile),
                "profit_grade": str(control.get("profit_grade") or ""),
                "profit_score_norm": _safe_float(control.get("profit_score"), 0.0),
                "drag_score_norm": _safe_float(control.get("drag_score"), 0.0),
                "ending_net_pnl_total": _safe_float(control.get("ending_net_pnl_total"), 0.0),
                "action": str(control.get("action") or ""),
            }
            for profile, control in active_profile_controls.items()
            if isinstance(control, dict)
        ],
        key=lambda row: (
            abs(min(_safe_float(row.get("ending_net_pnl_total"), 0.0), 0.0)),
            _safe_float(row.get("drag_score_norm"), 1.0),
            -_safe_float(row.get("profit_score_norm"), 0.0),
            str(row.get("profile") or ""),
        ),
    )
    strategy_rows = sorted(
        [
            {
                "profile": str(row.get("profile") or ""),
                "strategy": str(row.get("strategy") or ""),
                "bot_id": str(row.get("bot_id") or ""),
                "mode": str(row.get("mode") or ""),
                "ending_net_pnl_total": _safe_float(row.get("ending_net_pnl_total"), 0.0),
                "score_penalty_norm": _safe_float(row.get("score_penalty_norm"), 0.0),
            }
            for row in strategy_controls
            if isinstance(row, dict)
        ],
        key=lambda row: (
            abs(min(_safe_float(row.get("ending_net_pnl_total"), 0.0), 0.0)),
            _safe_float(row.get("score_penalty_norm"), 1.0),
            str(row.get("profile") or ""),
            str(row.get("strategy") or ""),
        ),
    )
    return {
        "active": raw_operational_grade != "A+",
        "mode": "raw_operational_grade_lift",
        "current_grade": str(raw_operational_grade or ""),
        "target_next_grade": str(target.get("next_grade") or ""),
        "current_counts": {
            "weak_profile_count": weak_count,
            "strategy_control_count": strategy_count,
        },
        "target_counts_for_next_grade": {
            "max_weak_profiles": target_weak,
            "max_strategy_controls": target_strategies,
        },
        "clearance_needed_for_next_grade": {
            "weak_profiles_to_clear": profile_clear_count,
            "strategy_pairs_to_clear": strategy_clear_count,
        },
        "fastest_count_lift_profiles": profile_rows[:profile_clear_count],
        "fastest_count_lift_strategy_pairs": strategy_rows[:strategy_clear_count],
        "largest_drag_profiles": sorted(
            profile_rows,
            key=lambda row: (
                -_safe_float(row.get("drag_score_norm"), 0.0),
                _safe_float(row.get("ending_net_pnl_total"), 0.0),
                str(row.get("profile") or ""),
            ),
        )[: min(max(profile_clear_count, 3), 8)],
        "largest_loss_strategy_pairs": sorted(
            strategy_rows,
            key=lambda row: (
                _safe_float(row.get("ending_net_pnl_total"), 0.0),
                -_safe_float(row.get("score_penalty_norm"), 0.0),
                str(row.get("profile") or ""),
            ),
        )[: min(max(strategy_clear_count, 5), 12)],
        "runtime_enforcement": {
            "block_new_entries_for_active_targets": True,
            "prefer_reductions_over_adds": True,
            "require_profitable_refreshes_to_clear_profile": 2,
            "require_profitable_refreshes_to_clear_strategy_pair": 2,
            "paper_only": True,
        },
        "stop_condition": (
            f"weak_profile_count <= {target_weak} and strategy_control_count <= {target_strategies} "
            f"for raw operational {target.get('next_grade')}"
        ),
    }


def _unprotected_operational_counts(
    *,
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
) -> dict[str, int]:
    unprotected_profiles = sum(
        1
        for control in active_profile_controls.values()
        if isinstance(control, dict)
        and _profile_requires_full_protection(control)
        and not _profile_loss_protected(control)
    )
    unprotected_strategies = sum(
        1
        for control in strategy_controls
        if isinstance(control, dict) and not _strategy_loss_protected(control)
    )
    return {
        "unprotected_weak_profile_count": unprotected_profiles,
        "unprotected_strategy_control_count": unprotected_strategies,
    }


def _a_plus_recovery_profile_control(control: dict[str, Any]) -> None:
    control["action"] = "quarantine_new_entries"
    control["a_plus_recovery_mode"] = True
    control["a_plus_recovery_reason"] = "financial_a_plus_lock_in_requires_weak_sleeve_quarantine"
    control["position_size_multiplier"] = min(_safe_float(control.get("position_size_multiplier"), 1.0), 0.05)
    control["new_entry_cap"] = 0
    control["runtime_policy"] = {
        **(control.get("runtime_policy") if isinstance(control.get("runtime_policy"), dict) else {}),
        "a_plus_lock_in": True,
        "block_all_new_entries_until_operational_a_plus": True,
    }
    for key in ("dynamic_sizing", "loser_quarantine", "exit_intelligence"):
        nested = control.get(key) if isinstance(control.get(key), dict) else {}
        control[key] = dict(nested)
    control["dynamic_sizing"]["paper_profitability_size_multiplier_norm"] = control["position_size_multiplier"]
    control["dynamic_sizing"]["max_new_entry_multiplier_norm"] = 0.0
    control["dynamic_sizing"]["block_new_entries_when_drag_active"] = True
    control["loser_quarantine"]["active"] = True
    control["loser_quarantine"]["mode"] = "quarantine_new_entries"
    control["loser_quarantine"]["new_entry_cap"] = 0
    control["loser_quarantine"]["block_new_entries"] = True
    control["loser_quarantine"]["reentry_requires_positive_refreshes"] = max(
        _safe_int(control["loser_quarantine"].get("reentry_requires_positive_refreshes"), 0),
        3,
    )
    control["exit_intelligence"]["active"] = True
    control["exit_intelligence"]["drag_reduction_mode"] = "reduce_only"
    control["exit_intelligence"]["prefer_reduce_over_add"] = True
    control["exit_intelligence"]["block_adds_while_unrealized_negative"] = True
    control["exit_intelligence"]["block_adds_while_drag_active"] = True
    control["exit_intelligence"]["max_adds_while_drag_active"] = 0
    contracts = control.get("upgrade_contracts") if isinstance(control.get("upgrade_contracts"), dict) else {}
    for key in ("dynamic_sizing", "loser_quarantine", "exit_intelligence"):
        if isinstance(contracts.get(key), dict):
            contracts[key].update(control[key])
    control["upgrade_contracts"] = contracts


def _a_plus_recovery_strategy_control(control: dict[str, Any]) -> None:
    control["mode"] = "paper_quarantine"
    control["a_plus_recovery_mode"] = True
    control["a_plus_recovery_reason"] = "financial_a_plus_lock_in_blocks_losing_strategy_pair"
    control["position_size_multiplier"] = 0.0
    control["new_entry_cap"] = 0
    control["block_new_entries"] = True
    contracts = control.get("upgrade_contracts") if isinstance(control.get("upgrade_contracts"), dict) else {}
    loser = contracts.get("loser_quarantine") if isinstance(contracts.get("loser_quarantine"), dict) else {}
    loser.update(
        {
            "active": True,
            "mode": "paper_quarantine",
            "new_entry_cap": 0,
            "block_new_entries": True,
            "a_plus_lock_in": True,
            "paper_only_retest_required": True,
        }
    )
    sizing = contracts.get("dynamic_sizing") if isinstance(contracts.get("dynamic_sizing"), dict) else {}
    sizing.update(
        {
            "active": True,
            "paper_profitability_size_multiplier_norm": 0.0,
            "max_new_entry_multiplier_norm": 0.0,
        }
    )
    contracts["loser_quarantine"] = loser
    contracts["dynamic_sizing"] = sizing
    control["upgrade_contracts"] = contracts


def _apply_a_plus_recovery_mode(
    *,
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
    financial_grade: str,
) -> None:
    if financial_grade != "A+":
        return
    for control in active_profile_controls.values():
        if not isinstance(control, dict):
            continue
        if _safe_float(control.get("ending_net_pnl_total"), 0.0) < 0.0 or _safe_float(control.get("drag_score"), 0.0) >= 0.20:
            _a_plus_recovery_profile_control(control)
    for control in strategy_controls:
        if isinstance(control, dict) and _safe_float(control.get("ending_net_pnl_total"), 0.0) < 0.0:
            _a_plus_recovery_strategy_control(control)


def _apply_protective_tightening_profile_control(control: dict[str, Any]) -> None:
    control["action"] = "quarantine_new_entries"
    control["protective_tightening_mode"] = True
    control["protective_tightening_reason"] = "paper_profit_grade_low_or_drag_severe"
    control["position_size_multiplier"] = min(_safe_float(control.get("position_size_multiplier"), 1.0), 0.08)
    control["new_entry_cap"] = 0
    control["runtime_policy"] = {
        **(control.get("runtime_policy") if isinstance(control.get("runtime_policy"), dict) else {}),
        "protective_tightening_lock": True,
        "block_all_new_entries_until_clean_refresh": True,
        "paper_only_until_next_profitable_refresh": True,
    }
    for key in ("dynamic_sizing", "loser_quarantine", "exit_intelligence"):
        nested = control.get(key) if isinstance(control.get(key), dict) else {}
        control[key] = dict(nested)
    control["dynamic_sizing"]["paper_profitability_size_multiplier_norm"] = control["position_size_multiplier"]
    control["dynamic_sizing"]["max_new_entry_multiplier_norm"] = 0.0
    control["dynamic_sizing"]["block_new_entries_when_drag_active"] = True
    control["loser_quarantine"]["active"] = True
    control["loser_quarantine"]["mode"] = "quarantine_new_entries"
    control["loser_quarantine"]["new_entry_cap"] = 0
    control["loser_quarantine"]["block_new_entries"] = True
    control["loser_quarantine"]["protective_tightening_lock"] = True
    control["loser_quarantine"]["reentry_requires_positive_refreshes"] = max(
        _safe_int(control["loser_quarantine"].get("reentry_requires_positive_refreshes"), 0),
        3,
    )
    control["exit_intelligence"]["active"] = True
    control["exit_intelligence"]["drag_reduction_mode"] = "reduce_only"
    control["exit_intelligence"]["prefer_reduce_over_add"] = True
    control["exit_intelligence"]["block_adds_while_unrealized_negative"] = True
    control["exit_intelligence"]["block_adds_while_drag_active"] = True
    control["exit_intelligence"]["max_adds_while_drag_active"] = 0
    contracts = control.get("upgrade_contracts") if isinstance(control.get("upgrade_contracts"), dict) else {}
    for key in ("dynamic_sizing", "loser_quarantine", "exit_intelligence"):
        if isinstance(contracts.get(key), dict):
            contracts[key].update(control[key])
    control["upgrade_contracts"] = contracts


def _apply_protective_tightening_strategy_control(control: dict[str, Any]) -> None:
    control["mode"] = "paper_quarantine"
    control["protective_tightening_mode"] = True
    control["protective_tightening_reason"] = "paper_strategy_pair_losing_under_protective_tightening"
    control["position_size_multiplier"] = 0.0
    control["new_entry_cap"] = 0
    control["block_new_entries"] = True
    contracts = control.get("upgrade_contracts") if isinstance(control.get("upgrade_contracts"), dict) else {}
    loser = contracts.get("loser_quarantine") if isinstance(contracts.get("loser_quarantine"), dict) else {}
    loser.update(
        {
            "active": True,
            "mode": "paper_quarantine",
            "new_entry_cap": 0,
            "block_new_entries": True,
            "protective_tightening_lock": True,
            "paper_only_retest_required": True,
        }
    )
    sizing = contracts.get("dynamic_sizing") if isinstance(contracts.get("dynamic_sizing"), dict) else {}
    sizing.update(
        {
            "active": True,
            "paper_profitability_size_multiplier_norm": 0.0,
            "max_new_entry_multiplier_norm": 0.0,
        }
    )
    contracts["loser_quarantine"] = loser
    contracts["dynamic_sizing"] = sizing
    control["upgrade_contracts"] = contracts


def _apply_protective_tightening_mode(
    *,
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
) -> None:
    for control in active_profile_controls.values():
        if not isinstance(control, dict):
            continue
        grade = str(control.get("profit_grade") or "").strip().upper()
        drag = _safe_float(control.get("drag_score"), 0.0)
        net = _safe_float(control.get("ending_net_pnl_total"), 0.0)
        if grade in LOW_GRADE_VALUES or drag >= 0.75 or net < 0.0:
            _apply_protective_tightening_profile_control(control)
    for control in strategy_controls:
        if not isinstance(control, dict):
            continue
        if _safe_float(control.get("ending_net_pnl_total"), 0.0) < 0.0:
            _apply_protective_tightening_strategy_control(control)


def _operational_control_grade(
    *,
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
    financial_grade: str,
) -> str:
    profiles_requiring_protection = [
        control
        for control in active_profile_controls.values()
        if isinstance(control, dict) and _profile_requires_full_protection(control)
    ]
    profile_protected = all(_profile_loss_protected(control) for control in profiles_requiring_protection)
    strategies_protected = all(
        _strategy_loss_protected(control)
        for control in strategy_controls
        if isinstance(control, dict)
    )
    if profile_protected and strategies_protected:
        return "A+"
    if profile_protected:
        return "A"
    return "B"


def _a_plus_target_contract(
    *,
    financial_grade: str,
    operational_outcome_grade: str,
    raw_operational_outcome_grade: str,
    operational_control_grade: str,
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
    unprotected_counts: dict[str, int],
    net_sum: float,
    realized_sum: float,
    unrealized_sum: float,
    change_vs_previous_day: float,
    executions: int,
) -> dict[str, Any]:
    weak_profiles = ordered_unique(
        [
            str(profile)
            for profile, control in active_profile_controls.items()
            if isinstance(control, dict) and bool(control.get("active", False))
        ]
    )
    losing_pairs = ordered_unique(
        [
            str(row.get("strategy") or "")
            for row in strategy_controls
            if isinstance(row, dict) and str(row.get("strategy") or "").strip()
        ]
    )
    financial_ready = financial_grade == "A+"
    operational_outcome_ready = operational_outcome_grade == "A+"
    raw_operational_outcome_ready = raw_operational_outcome_grade == "A+"
    operational_control_ready = operational_control_grade == "A+"
    unprotected_weak_count = _safe_int(unprotected_counts.get("unprotected_weak_profile_count"), len(weak_profiles))
    unprotected_strategy_count = _safe_int(unprotected_counts.get("unprotected_strategy_control_count"), len(strategy_controls))
    blockers: list[str] = []
    if not financial_ready:
        blockers.append("financial_a_plus_thresholds_not_met")
    if unprotected_weak_count:
        blockers.append("unprotected_weak_sleeves_still_active")
    if unprotected_strategy_count:
        blockers.append("unprotected_losing_strategy_pairs_still_active")
    raw_outcome_debt: list[str] = []
    if weak_profiles:
        raw_outcome_debt.append("weak_sleeves_still_need_clean_refreshes")
    if losing_pairs:
        raw_outcome_debt.append("losing_strategy_pairs_still_need_profitable_refreshes")
    return {
        "active": True,
        "financial_grade": financial_grade,
        "operational_outcome_grade": operational_outcome_grade,
        "raw_operational_outcome_grade": raw_operational_outcome_grade,
        "operational_control_grade": operational_control_grade,
        "financial_a_plus_ready": financial_ready,
        "operational_outcome_a_plus_ready": operational_outcome_ready,
        "raw_operational_outcome_a_plus_ready": raw_operational_outcome_ready,
        "operational_control_a_plus_ready": operational_control_ready,
        "combined_a_plus_ready": financial_ready and operational_outcome_ready,
        "raw_combined_a_plus_ready": financial_ready and raw_operational_outcome_ready,
        "combined_control_a_plus_ready": financial_ready and operational_control_ready,
        "headline_grade": "A+" if financial_ready and operational_control_ready else financial_grade,
        "outcome_grade": "A+" if financial_ready and operational_outcome_ready else ("A" if financial_ready else financial_grade),
        "thresholds": {
            "min_net_pnl": FINANCIAL_APLUS_MIN_NET_PNL,
            "min_realized_pnl": FINANCIAL_APLUS_MIN_REALIZED_PNL,
            "min_change_vs_previous_day": FINANCIAL_APLUS_MIN_CHANGE_PNL,
            "min_executions": FINANCIAL_APLUS_MIN_EXECUTIONS,
            "operational_outcome_requires_weak_profiles": 0,
            "operational_outcome_requires_strategy_controls": 0,
        },
        "current": {
            "net_pnl": round(net_sum, 6),
            "realized_pnl": round(realized_sum, 6),
            "unrealized_pnl": round(unrealized_sum, 6),
            "change_vs_previous_day": round(change_vs_previous_day, 6),
            "executions": executions,
            "weak_profile_count": len(weak_profiles),
            "strategy_control_count": len(strategy_controls),
            "unprotected_weak_profile_count": unprotected_weak_count,
            "unprotected_strategy_control_count": unprotected_strategy_count,
        },
        "weak_profiles": weak_profiles,
        "losing_strategy_pairs": losing_pairs[:24],
        "blockers": blockers,
        "raw_outcome_debt": raw_outcome_debt,
        "controls_applied": [
            "lock_winning_financial_state",
            "quarantine_all_negative_sleeves_until_clean_refresh",
            "quarantine_all_losing_profile_strategy_pairs",
            "require_three_profitable_refreshes_before_reentry",
        ],
        "stop_condition": "financial A+ stays intact, unprotected weak counts stay zero, then raw weak_profile_count and strategy_control_count decay to zero after clean refreshes",
    }


def _profit_harvest_profile_params(profile: str) -> dict[str, float]:
    raw = PROFIT_HARVEST_PROFILE_PARAMS.get(_normal_profile(profile), {})
    return {
        "target_realized_share": _clamp(_safe_float(raw.get("target_realized_share"), PROFIT_HARVEST_TARGET_REALIZED_SHARE), 0.20, 0.55),
        "max_unrealized_share": _clamp(_safe_float(raw.get("max_unrealized_share"), PROFIT_HARVEST_MAX_UNREALIZED_SHARE), 0.50, 0.85),
        "trend_prior": _clamp(_safe_float(raw.get("trend_prior"), 0.55), 0.0, 1.0),
        "min_trim_fraction": _clamp(_safe_float(raw.get("min_trim_fraction"), 0.12), 0.05, 0.35),
        "max_trim_fraction": _clamp(_safe_float(raw.get("max_trim_fraction"), 0.48), 0.20, 0.70),
    }


def _profit_harvest_intelligence(
    *,
    profile: str,
    row: dict[str, Any],
    positive_realized: float,
    positive_unrealized: float,
    realized_share: float,
    unrealized_share: float,
    harvest_pressure: float,
    target_realized_share: float,
    max_unrealized_share: float,
) -> dict[str, Any]:
    params = _profit_harvest_profile_params(profile)
    executions = _safe_int(row.get("executions"), 0)
    net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
    win_rate_raw = row.get("win_rate")
    win_rate = _clamp(_safe_float(win_rate_raw, 0.50)) if win_rate_raw is not None else 0.50
    net_confidence = _clamp(max(net, 0.0) / 75_000.0)
    sample_confidence = _clamp(executions / 500.0)
    realization_gap = _clamp(max(target_realized_share - realized_share, 0.0) / max(target_realized_share, 0.01))
    unrealized_excess = _clamp(max(unrealized_share - max_unrealized_share, 0.0) / max(1.0 - max_unrealized_share, 0.01))

    realized_conversion_skill = _clamp(
        0.34 * _clamp(realized_share / max(target_realized_share, 0.01))
        + 0.24 * _clamp(positive_realized / max(PROFIT_HARVEST_MIN_UNREALIZED_PNL * 8.0, 1.0))
        + 0.18 * win_rate
        + 0.14 * sample_confidence
        + 0.10 * net_confidence
    )
    trend_continuation_score = _clamp(
        0.30 * params["trend_prior"]
        + 0.24 * unrealized_share
        + 0.18 * win_rate
        + 0.14 * _clamp(positive_unrealized / max(PROFIT_HARVEST_MIN_UNREALIZED_PNL * 10.0, 1.0))
        + 0.08 * net_confidence
        + 0.06 * sample_confidence
    )
    harvest_regret_risk = _clamp(
        0.42 * trend_continuation_score
        + 0.25 * (1.0 - realized_conversion_skill)
        + 0.18 * realization_gap
        + 0.15 * (1.0 - unrealized_excess)
    )
    trim_confidence = _clamp(
        0.42 * harvest_pressure
        + 0.22 * realized_conversion_skill
        + 0.18 * unrealized_excess
        + 0.10 * realization_gap
        + 0.08 * (1.0 - trend_continuation_score)
    )
    trim_multiplier = _clamp(
        0.84
        + 0.40 * realized_conversion_skill
        + 0.20 * unrealized_excess
        - 0.34 * harvest_regret_risk,
        0.55,
        1.25,
    )
    learned_target = _clamp(
        target_realized_share
        + 0.05 * (realized_conversion_skill - 0.50)
        - 0.04 * (harvest_regret_risk - 0.50),
        0.24,
        0.52,
    )
    learned_max_unrealized = _clamp(
        max_unrealized_share
        + 0.04 * harvest_regret_risk
        - 0.03 * realized_conversion_skill,
        0.55,
        0.82,
    )
    exit_floor = _clamp(
        0.54
        + 0.16 * harvest_regret_risk
        - 0.08 * realized_conversion_skill
        + 0.04 * trend_continuation_score,
        0.50,
        0.76,
    )
    return {
        "active": True,
        "mode": "paper_profit_harvest_learning",
        "trend_continuation_score_norm": round(trend_continuation_score, 6),
        "harvest_regret_risk_norm": round(harvest_regret_risk, 6),
        "realized_conversion_skill_norm": round(realized_conversion_skill, 6),
        "trim_confidence_norm": round(trim_confidence, 6),
        "trim_aggressiveness_multiplier_norm": round(trim_multiplier, 6),
        "learned_target_realized_share_norm": round(learned_target, 6),
        "learned_max_unrealized_share_norm": round(learned_max_unrealized, 6),
        "dynamic_exit_quality_floor_norm": round(exit_floor, 6),
        "hold_winner_when_trend_continuation_above_norm": round(_clamp(0.74 - 0.08 * realized_conversion_skill + 0.08 * harvest_regret_risk, 0.62, 0.86), 6),
        "force_trim_only_when_harvest_pressure_above_norm": round(_clamp(0.68 + 0.12 * harvest_regret_risk - 0.05 * realized_conversion_skill, 0.62, 0.86), 6),
        "post_trim_followthrough_lookahead_minutes": 90,
        "feedback_labels": [
            "paper_harvest_regret_bucket",
            "paper_trend_continuation_bucket",
            "paper_realized_conversion_skill_bucket",
            "post_trim_followthrough_bucket",
        ],
        "learning_questions": [
            "did the sleeve keep running after the trim",
            "did realized pnl rise without giving back the financial A+ state",
            "is this sleeve better at holding winners or harvesting quickly",
        ],
    }


def _profit_harvest_control(profile: str, row: dict[str, Any]) -> dict[str, Any]:
    executions = _safe_int(row.get("executions"), 0)
    net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
    realized = _safe_float(row.get("ending_realized_pnl_total"), 0.0)
    unrealized = _safe_float(row.get("ending_unrealized_pnl_total"), 0.0)
    data_status = str(row.get("data_status") or "").strip().lower()
    live_status = str(row.get("live_shadow_status") or "").strip().lower()
    params = _profit_harvest_profile_params(profile)
    target_realized_share = params["target_realized_share"]
    max_unrealized_share = params["max_unrealized_share"]
    positive_unrealized = max(unrealized, 0.0)
    positive_realized = max(realized, 0.0)
    positive_total = positive_unrealized + positive_realized
    unrealized_share = _clamp(positive_unrealized / max(positive_total, 1.0))
    realized_share = _clamp(positive_realized / max(positive_total, 1.0))
    small_same_day_harvest = bool(
        executions >= PROFIT_HARVEST_SMALL_MIN_EXECUTIONS
        and net >= PROFIT_HARVEST_SMALL_MIN_NET_PNL
        and positive_unrealized >= PROFIT_HARVEST_SMALL_MIN_UNREALIZED_PNL
        and unrealized_share >= PROFIT_HARVEST_SMALL_MIN_UNREALIZED_SHARE
    )
    harvest_pressure = _clamp(
        0.42 * _clamp(positive_unrealized / max(PROFIT_HARVEST_MIN_UNREALIZED_PNL * 8.0, 1.0))
        + 0.32 * _clamp((unrealized_share - 0.45) / 0.45)
        + 0.16 * _clamp(max(target_realized_share - realized_share, 0.0) / target_realized_share)
        + 0.10 * _clamp(executions / 500.0)
    )
    if small_same_day_harvest:
        harvest_pressure = max(
            harvest_pressure,
            _clamp(
                0.46
                + 0.30
                * _clamp(
                    (unrealized_share - PROFIT_HARVEST_SMALL_MIN_UNREALIZED_SHARE)
                    / max(1.0 - PROFIT_HARVEST_SMALL_MIN_UNREALIZED_SHARE, 0.01)
                )
                + 0.14 * _clamp((positive_unrealized - PROFIT_HARVEST_SMALL_MIN_UNREALIZED_PNL) / 250.0)
                + 0.10 * _clamp(executions / 100.0)
            ),
        )
    intelligence = _profit_harvest_intelligence(
        profile=profile,
        row=row,
        positive_realized=positive_realized,
        positive_unrealized=positive_unrealized,
        realized_share=realized_share,
        unrealized_share=unrealized_share,
        harvest_pressure=harvest_pressure,
        target_realized_share=target_realized_share,
        max_unrealized_share=max_unrealized_share,
    )
    target_realized_share = _safe_float(intelligence.get("learned_target_realized_share_norm"), target_realized_share)
    max_unrealized_share = _safe_float(intelligence.get("learned_max_unrealized_share_norm"), max_unrealized_share)
    if small_same_day_harvest:
        target_realized_share = max(target_realized_share, PROFIT_HARVEST_SMALL_TARGET_REALIZED_SHARE)
        max_unrealized_share = min(max_unrealized_share, PROFIT_HARVEST_SMALL_MAX_UNREALIZED_SHARE)
    trim_multiplier = _safe_float(intelligence.get("trim_aggressiveness_multiplier_norm"), 1.0)
    trim_fraction = _clamp(
        (params["min_trim_fraction"] + ((params["max_trim_fraction"] - params["min_trim_fraction"]) * harvest_pressure))
        * trim_multiplier,
        params["min_trim_fraction"],
        params["max_trim_fraction"],
    )
    if unrealized_share >= 0.85:
        trim_fraction = max(trim_fraction, 0.35)
    elif unrealized_share >= max_unrealized_share:
        trim_fraction = max(trim_fraction, 0.25)
    if small_same_day_harvest:
        trim_fraction = min(trim_fraction, PROFIT_HARVEST_SMALL_MAX_TRIM_FRACTION)
    trim_fraction = _clamp(trim_fraction, params["min_trim_fraction"], params["max_trim_fraction"])
    carry_forward_open_winner = bool(
        executions <= 0
        and data_status in PROFIT_HARVEST_CARRY_FORWARD_STATUSES
        and live_status in PROFIT_HARVEST_CARRY_FORWARD_LIVE_STATES
        and net >= PROFIT_HARVEST_MIN_NET_PNL
        and positive_unrealized >= PROFIT_HARVEST_MIN_UNREALIZED_PNL
    )
    active = bool(
        carry_forward_open_winner
        or (
            executions > 0
            and net >= PROFIT_HARVEST_MIN_NET_PNL
            and positive_unrealized >= PROFIT_HARVEST_MIN_UNREALIZED_PNL
        )
        or small_same_day_harvest
    )
    if carry_forward_open_winner:
        active_reason = "carry_forward_open_winner"
    elif small_same_day_harvest:
        active_reason = "small_pnl_same_day_harvest"
    else:
        active_reason = "same_day_paper_fills"
    return {
        "profile": profile,
        "active": active,
        "mode": "paper_profit_realization",
        "active_reason": active_reason,
        "carry_forward_open_winner": carry_forward_open_winner,
        "small_pnl_same_day_harvest": small_same_day_harvest,
        "small_pnl_harvest_thresholds": {
            "min_net_pnl_total": PROFIT_HARVEST_SMALL_MIN_NET_PNL,
            "min_unrealized_pnl_total": PROFIT_HARVEST_SMALL_MIN_UNREALIZED_PNL,
            "min_executions": PROFIT_HARVEST_SMALL_MIN_EXECUTIONS,
            "min_unrealized_profit_share_norm": PROFIT_HARVEST_SMALL_MIN_UNREALIZED_SHARE,
            "max_trim_fraction_norm": PROFIT_HARVEST_SMALL_MAX_TRIM_FRACTION,
        },
        "data_status": data_status,
        "live_shadow_status": live_status,
        "ending_net_pnl_total": round(net, 6),
        "ending_realized_pnl_total": round(realized, 6),
        "ending_unrealized_pnl_total": round(unrealized, 6),
        "unrealized_profit_share_norm": round(unrealized_share, 6),
        "realized_profit_share_norm": round(realized_share, 6),
        "harvest_pressure_norm": round(harvest_pressure, 6),
        "recommended_trim_fraction_norm": round(trim_fraction, 6),
        "target_realized_profit_share_norm": round(target_realized_share, 6),
        "max_unrealized_profit_share_norm": round(max_unrealized_share, 6),
        "block_new_adds_when_unrealized_share_above_norm": round(max_unrealized_share, 6),
        "promote_trim_when_exit_quality_above_norm": _safe_float(intelligence.get("dynamic_exit_quality_floor_norm"), 0.58),
        "promote_trim_when_harvest_pressure_above_norm": round(_clamp(0.50 + 0.08 * _safe_float(intelligence.get("harvest_regret_risk_norm"), 0.0)), 6),
        "force_trim_when_harvest_pressure_above_norm": _safe_float(intelligence.get("force_trim_only_when_harvest_pressure_above_norm"), 0.72),
        "force_trim_when_unrealized_share_above_norm": round(_clamp(max_unrealized_share + 0.15, 0.78, 0.94), 6),
        "harvest_intelligence": intelligence,
        "paper_only": True,
        "required_labels": [
            "paper_profit_harvest_bucket",
            "paper_realized_conversion_bucket",
            "paper_exit_quality_bucket",
            "paper_harvest_regret_bucket",
            "paper_trend_continuation_bucket",
            "paper_realized_conversion_skill_bucket",
            "post_trim_followthrough_bucket",
        ],
        "lift_condition": "realized profit share reaches target or unrealized share drops under cap without losing financial A+",
    }


def _profit_harvest_profile_controls(sleeves: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    controls: dict[str, dict[str, Any]] = {}
    for row in sleeves:
        if not isinstance(row, dict):
            continue
        profile = _normal_profile(row.get("profile"))
        if not profile:
            continue
        control = _profit_harvest_control(profile, row)
        if bool(control.get("active", False)):
            controls[profile] = control
    return controls


def _profit_tier_for_pnl(net_pnl: float) -> dict[str, Any]:
    selected = PROFIT_TIER_RULES[0]
    for rule in PROFIT_TIER_RULES:
        if net_pnl >= _safe_float(rule.get("min_strategy_net_pnl"), 0.0):
            selected = rule
    return dict(selected)


def _strategy_profit_harvest_controls(
    sleeves: list[dict[str, Any]],
    profit_harvest_controls: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    controls: dict[str, dict[str, Any]] = {}
    for row in sleeves:
        if not isinstance(row, dict):
            continue
        profile = _normal_profile(row.get("profile"))
        profile_control = profit_harvest_controls.get(profile)
        winners = row.get("top_winning_strategies") if isinstance(row.get("top_winning_strategies"), list) else []
        if not profile_control or not winners:
            continue
        profile_unrealized = max(_safe_float(row.get("ending_unrealized_pnl_total"), 0.0), 0.0)
        profile_net = max(_safe_float(row.get("ending_net_pnl_total"), 0.0), 1.0)
        profile_trim = _clamp(_safe_float(profile_control.get("recommended_trim_fraction_norm"), 0.20), 0.05, 0.65)
        intelligence = (
            profile_control.get("harvest_intelligence")
            if isinstance(profile_control.get("harvest_intelligence"), dict)
            else {}
        )
        conversion_skill = _clamp(_safe_float(intelligence.get("realized_conversion_skill_norm"), 0.50))
        regret_risk = _clamp(_safe_float(intelligence.get("harvest_regret_risk_norm"), 0.0))
        trend_continuation = _clamp(_safe_float(intelligence.get("trend_continuation_score_norm"), 0.50))
        for winner in winners[:8]:
            if not isinstance(winner, dict):
                continue
            strategy = str(winner.get("strategy") or "").strip()
            if not strategy:
                continue
            net = _safe_float(winner.get("ending_net_pnl_total"), 0.0)
            if net <= 0.0:
                continue
            tier = _profit_tier_for_pnl(net)
            tier_trim = _safe_float(tier.get("trim_fraction_norm"), 0.15)
            contribution_share = _clamp(net / profile_net)
            concentration = _clamp(net / max(profile_unrealized, profile_net, 1.0))
            recommended_trim = _clamp(
                (0.42 * profile_trim)
                + (0.34 * tier_trim)
                + (0.14 * contribution_share)
                + (0.10 * concentration),
                0.08,
                0.58,
            )
            recommended_trim = _clamp(
                recommended_trim
                * _clamp(0.90 + (0.28 * conversion_skill) - (0.22 * regret_risk), 0.55, 1.25),
                0.06,
                0.62,
            )
            bot_id = _strategy_bot_id(strategy)
            key = f"{profile}::{strategy.lower()}"
            controls[key] = {
                "profile": profile,
                "strategy": strategy,
                "bot_id": bot_id,
                "active": True,
                "mode": "paper_strategy_profit_harvest",
                "tier": tier.get("tier"),
                "tier_mode": tier.get("mode"),
                "ending_net_pnl_total": round(net, 6),
                "strategy_contribution_share_norm": round(contribution_share, 6),
                "strategy_concentration_norm": round(concentration, 6),
                "recommended_trim_fraction_norm": round(recommended_trim, 6),
                "block_new_adds": bool(contribution_share >= 0.18 or concentration >= 0.24),
                "promote_partial_trim": bool(recommended_trim >= 0.18),
                "protect_runner_when_trend_continuation_above_norm": round(
                    _clamp(0.70 + (0.10 * regret_risk) - (0.06 * conversion_skill), 0.62, 0.86),
                    6,
                ),
                "force_trim_when_harvest_pressure_above_norm": profile_control.get("force_trim_when_harvest_pressure_above_norm", 0.72),
                "profile_harvest_pressure_norm": profile_control.get("harvest_pressure_norm", 0.0),
                "profile_trend_continuation_norm": round(trend_continuation, 6),
                "profile_harvest_regret_risk_norm": round(regret_risk, 6),
                "profile_realized_conversion_skill_norm": round(conversion_skill, 6),
                "paper_only": True,
                "required_labels": [
                    "paper_profit_harvest_bucket",
                    "paper_realized_conversion_bucket",
                    "paper_harvest_regret_bucket",
                    "paper_trend_continuation_bucket",
                    "post_trim_followthrough_bucket",
                ],
            }
            if bot_id:
                controls[f"{profile}::{bot_id.lower()}"] = controls[key]
    return controls


def _parse_timestamp_seconds(raw: Any) -> float:
    text = str(raw or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _recent_paper_order_paths(project_root: Path, paper: dict[str, Any], *, limit: int = 4) -> list[Path]:
    raw_files = paper.get("source_files") if isinstance(paper.get("source_files"), list) else []
    paths: list[Path] = []
    for raw in raw_files:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = project_root / path
        if path.exists() and path.name.startswith("paper_bridge_orders_") and (path.suffix == ".jsonl" or path.name.endswith(".jsonl.gz")):
            paths.append(path)
    if not paths:
        bridge_dir = project_root / "exports" / "paper_broker_bridge" / "paper"
        paths.extend(path for path in bridge_dir.glob("paper_bridge_orders_*.jsonl") if path.exists())
        paths.extend(path for path in bridge_dir.glob("paper_bridge_orders_*.jsonl.gz") if path.exists())
    paths = ordered_unique([str(path) for path in paths])
    path_objs = [Path(path) for path in paths]
    path_objs.sort(key=lambda path: (path.name, path.stat().st_mtime if path.exists() else 0.0), reverse=True)
    return path_objs[: max(int(limit), 1)]


def _iter_jsonl_records(path: Path, *, max_records: int = 60_000):
    opener = gzip.open if path.name.endswith(".gz") else open
    count = 0
    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if count >= max_records:
                    break
                text = line.strip()
                if not text:
                    continue
                count += 1
                try:
                    row = json.loads(text)
                except Exception:
                    continue
                if isinstance(row, dict):
                    yield row
    except Exception:
        return


def _position_harvest_ledger(
    *,
    project_root: Path,
    paper: dict[str, Any],
    profit_harvest_controls: dict[str, dict[str, Any]],
    strategy_harvest_controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_profiles = set(profit_harvest_controls.keys())
    if not target_profiles:
        return {
            "active": False,
            "mode": "paper_position_harvest_ledger",
            "source_file_count": 0,
            "position_count": 0,
            "positions": [],
        }
    latest: dict[str, dict[str, Any]] = {}
    max_seen_ts = 0.0
    source_paths = _recent_paper_order_paths(project_root, paper)
    records_scanned = 0
    for path in source_paths:
        for row in _iter_jsonl_records(path):
            records_scanned += 1
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            profile = _normal_profile(
                metadata.get("source_profile")
                or metadata.get("profile")
                or row.get("profile")
                or "default"
            )
            if profile not in target_profiles:
                continue
            strategy = str(row.get("strategy") or metadata.get("strategy") or "").strip()
            symbol = str(row.get("symbol") or "").strip().upper()
            if not strategy or not symbol:
                continue
            ts = _parse_timestamp_seconds(row.get("timestamp_utc"))
            max_seen_ts = max(max_seen_ts, ts)
            key = f"{profile}::{strategy.lower()}::{symbol}"
            prev = latest.get(key)
            if prev and ts < _safe_float(prev.get("_timestamp_seconds"), 0.0):
                continue
            latest[key] = {
                "_timestamp_seconds": ts,
                "timestamp_utc": row.get("timestamp_utc"),
                "profile": profile,
                "symbol": symbol,
                "strategy": strategy,
                "bot_id": _strategy_bot_id(strategy),
                "action": str(row.get("action") or "").strip().upper(),
                "position_qty": _safe_float(row.get("position_qty"), 0.0),
                "position_avg_price": _safe_float(row.get("position_avg_price"), 0.0),
                "mark_price": _safe_float(row.get("mark_price"), _safe_float(row.get("fill_price"), 0.0)),
                "realized_pnl": _safe_float(row.get("realized_pnl"), _safe_float(row.get("realized"), 0.0)),
                "unrealized_pnl": _safe_float(row.get("unrealized_pnl"), _safe_float(row.get("unrealized"), 0.0)),
                "model_score": _safe_float(row.get("model_score"), 0.5),
                "threshold": _safe_float(row.get("threshold"), 0.5),
            }
    positions: list[dict[str, Any]] = []
    for key, row in latest.items():
        qty = _safe_float(row.get("position_qty"), 0.0)
        unrealized = _safe_float(row.get("unrealized_pnl"), 0.0)
        if abs(qty) <= 0.0 or unrealized <= 0.0:
            continue
        profile = _normal_profile(row.get("profile"))
        strategy = str(row.get("strategy") or "").strip()
        strategy_control = strategy_harvest_controls.get(f"{profile}::{strategy.lower()}") or strategy_harvest_controls.get(
            f"{profile}::{_strategy_bot_id(strategy).lower()}"
        )
        profile_control = profit_harvest_controls.get(profile, {})
        avg = abs(_safe_float(row.get("position_avg_price"), 0.0))
        notional = abs(qty) * avg
        pnl_pct = _clamp(unrealized / max(notional, 1.0), -1.0, 1.0)
        age_minutes = 0.0
        if max_seen_ts > 0.0 and _safe_float(row.get("_timestamp_seconds"), 0.0) > 0.0:
            age_minutes = max((max_seen_ts - _safe_float(row.get("_timestamp_seconds"), 0.0)) / 60.0, 0.0)
        profile_trim = _clamp(_safe_float(profile_control.get("recommended_trim_fraction_norm"), 0.20), 0.05, 0.65)
        strategy_trim = _clamp(_safe_float((strategy_control or {}).get("recommended_trim_fraction_norm"), profile_trim), 0.05, 0.65)
        tier = _profit_tier_for_pnl(max(unrealized, _safe_float((strategy_control or {}).get("ending_net_pnl_total"), unrealized)))
        position_trim = _clamp(
            (0.48 * strategy_trim)
            + (0.28 * _safe_float(tier.get("trim_fraction_norm"), 0.15))
            + (0.14 * _clamp(unrealized / max(PROFIT_HARVEST_MIN_UNREALIZED_PNL, 1.0)))
            + (0.10 * _clamp(pnl_pct / 0.04 if pnl_pct > 0.0 else 0.0)),
            0.05,
            0.62,
        )
        positions.append(
            {
                "profile": profile,
                "symbol": row.get("symbol"),
                "strategy": strategy,
                "bot_id": row.get("bot_id"),
                "position_qty": round(qty, 6),
                "position_avg_price": round(_safe_float(row.get("position_avg_price"), 0.0), 6),
                "mark_price": round(_safe_float(row.get("mark_price"), 0.0), 6),
                "unrealized_pnl": round(unrealized, 6),
                "realized_pnl": round(_safe_float(row.get("realized_pnl"), 0.0), 6),
                "unrealized_pnl_pct_norm": round(_clamp(0.50 + pnl_pct * 10.0), 6),
                "age_minutes": round(age_minutes, 3),
                "profit_tier": tier.get("tier"),
                "profit_tier_mode": tier.get("mode"),
                "recommended_trim_fraction_norm": round(position_trim, 6),
                "runner_protection_floor_norm": (strategy_control or {}).get(
                    "protect_runner_when_trend_continuation_above_norm",
                    profile_control.get("harvest_intelligence", {}).get("hold_winner_when_trend_continuation_above_norm", 0.74)
                    if isinstance(profile_control.get("harvest_intelligence"), dict)
                    else 0.74,
                ),
                "trim_condition": "trim only when exit quality clears floor or continuation weakens",
                "paper_only": True,
            }
        )
    existing_strategy_keys = {
        (str(item.get("profile") or ""), str(item.get("strategy") or "").strip().lower())
        for item in positions
    }
    for sleeve in _as_list(paper.get("sleeve_latest")):
        if not isinstance(sleeve, dict):
            continue
        profile = _normal_profile(sleeve.get("profile"))
        if profile not in target_profiles:
            continue
        profile_control = profit_harvest_controls.get(profile, {})
        profile_unrealized = max(_safe_float(sleeve.get("ending_unrealized_pnl_total"), 0.0), 0.0)
        profile_net = max(_safe_float(sleeve.get("ending_net_pnl_total"), 0.0), 1.0)
        if profile_unrealized <= 0.0:
            continue
        winners = sleeve.get("top_winning_strategies") if isinstance(sleeve.get("top_winning_strategies"), list) else []
        for winner in winners[:8]:
            if not isinstance(winner, dict):
                continue
            strategy = str(winner.get("strategy") or "").strip()
            if not strategy or (profile, strategy.lower()) in existing_strategy_keys:
                continue
            net = _safe_float(winner.get("ending_net_pnl_total"), 0.0)
            if net <= 0.0:
                continue
            strategy_control = strategy_harvest_controls.get(f"{profile}::{strategy.lower()}") or strategy_harvest_controls.get(
                f"{profile}::{_strategy_bot_id(strategy).lower()}"
            )
            profile_trim = _clamp(_safe_float(profile_control.get("recommended_trim_fraction_norm"), 0.20), 0.05, 0.65)
            strategy_trim = _clamp(_safe_float((strategy_control or {}).get("recommended_trim_fraction_norm"), profile_trim), 0.05, 0.65)
            contribution_share = _clamp(net / max(profile_net, 1.0))
            concentration = _clamp(net / max(profile_unrealized, profile_net, 1.0))
            tier = _profit_tier_for_pnl(net)
            estimated_unrealized = min(profile_unrealized, max(net, profile_unrealized * contribution_share))
            position_trim = _clamp(
                (0.46 * strategy_trim)
                + (0.28 * _safe_float(tier.get("trim_fraction_norm"), 0.15))
                + (0.14 * contribution_share)
                + (0.12 * concentration),
                0.05,
                0.62,
            )
            positions.append(
                {
                    "profile": profile,
                    "symbol": f"{profile.upper()}_OPEN_WINNER",
                    "strategy": strategy,
                    "bot_id": _strategy_bot_id(strategy),
                    "position_qty": 0.0,
                    "position_avg_price": 0.0,
                    "mark_price": 0.0,
                    "unrealized_pnl": round(estimated_unrealized, 6),
                    "realized_pnl": 0.0,
                    "unrealized_pnl_pct_norm": 0.5,
                    "age_minutes": 0.0,
                    "profit_tier": tier.get("tier"),
                    "profit_tier_mode": tier.get("mode"),
                    "recommended_trim_fraction_norm": round(position_trim, 6),
                    "runner_protection_floor_norm": (strategy_control or {}).get(
                        "protect_runner_when_trend_continuation_above_norm",
                        profile_control.get("harvest_intelligence", {}).get("hold_winner_when_trend_continuation_above_norm", 0.74)
                        if isinstance(profile_control.get("harvest_intelligence"), dict)
                        else 0.74,
                    ),
                    "trim_condition": "synthetic open-winner proxy; trim only when exit quality clears floor or continuation weakens",
                    "source": "sleeve_top_winner_snapshot",
                    "position_proxy": True,
                    "paper_only": True,
                }
            )
            existing_strategy_keys.add((profile, strategy.lower()))
    positions.sort(key=lambda item: (_safe_float(item.get("unrealized_pnl"), 0.0), _safe_float(item.get("recommended_trim_fraction_norm"), 0.0)), reverse=True)
    return {
        "active": bool(positions),
        "mode": "paper_position_harvest_ledger",
        "source_file_count": len(source_paths),
        "records_scanned": records_scanned,
        "target_profiles": sorted(target_profiles),
        "position_count": len(positions),
        "total_positive_unrealized_pnl": round(sum(_safe_float(row.get("unrealized_pnl"), 0.0) for row in positions), 6),
        "positions": positions[:64],
        "runtime_rules": [
            "position trims are partial and paper-only",
            "never trim a runner solely because it is green; require exit quality, tier pressure, or weakening continuation",
            "feed every trim into regret replay before increasing future trim aggression",
        ],
    }


def _profit_harvest_regret_replay_contract(
    *,
    position_ledger: dict[str, Any],
    strategy_harvest_controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    position_count = _safe_int(position_ledger.get("position_count"), 0)
    strategy_count = len({id(row) for row in strategy_harvest_controls.values()})
    return {
        "active": bool(position_count or strategy_count),
        "mode": "paper_harvest_regret_replay",
        "lookahead_minutes": PROFIT_HARVEST_REPLAY_LOOKAHEAD_MINUTES,
        "position_count": position_count,
        "strategy_control_count": strategy_count,
        "labels": [
            "post_trim_followthrough_bucket",
            "paper_harvest_regret_bucket",
            "paper_trend_continuation_bucket",
            "paper_realized_conversion_skill_bucket",
            *HARVEST_REPLAY_OUTCOME_LABELS,
        ],
        "judgement_rules": [
            "trim_saved_profit_when_mark_to_market_drawdown_exceeds_trim_value",
            "trim_too_early_when_followthrough_after_trim_exceeds_runner_floor",
            "trim_too_late_when_unrealized_profit_gives_back_before_realization",
            "partial_harvest_helped_when_realized_pnl_rises_and_remaining_runner_stays_inside_drawdown_cap",
            "runner_deserved_room_when_no_trim_counterfactual_beats_trim_path_after_lookahead",
        ],
        "upgrade_layer": {
            "active": True,
            "mode": "profit_harvest_replay_layer_v2",
            "outcome_classes": [
                "trimmed_too_early",
                "trimmed_too_late",
                "partialed_correctly",
                "held_correctly",
                "should_have_partialed",
                "runner_protected_correctly",
            ],
            "training_feedback": [
                "feed replay labels into exit_intelligence and strategy_level_promotion",
                "raise trim aggression only after regret_control_norm improves",
                "lower trim aggression when runner_deserved_room_bucket rises",
            ],
            "expected_impact": "turn every harvest or hold decision into supervised feedback for realized-profit conversion",
            "stop_condition": "regret_control_norm >= 0.80 and post_trim_followthrough labels are current",
        },
        "next_refresh_command": [
            "./scripts/ops/opsctl.sh",
            "paper-profitability-control",
            "--apply",
            "--json",
        ],
    }


def _aggressive_harvest_mode_contract(
    sleeves: list[dict[str, Any]],
    profit_harvest_controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    aggressive_profiles = ["aggressive", "intraday_aggressive", "swing_aggressive", "options_on_futures_aggressive"]
    rows: list[dict[str, Any]] = []
    for profile in aggressive_profiles:
        sleeve = next((row for row in sleeves if isinstance(row, dict) and _normal_profile(row.get("profile")) == profile), {})
        control = profit_harvest_controls.get(profile, {})
        params = _profit_harvest_profile_params(profile)
        net = _safe_float(sleeve.get("ending_net_pnl_total"), 0.0) if isinstance(sleeve, dict) else 0.0
        unrealized = _safe_float(sleeve.get("ending_unrealized_pnl_total"), 0.0) if isinstance(sleeve, dict) else 0.0
        rows.append(
            {
                "profile": profile,
                "active": bool(control.get("active", False)),
                "armed": True,
                "mode": "fast_partial_profit_then_trail" if profile == "intraday_aggressive" else "partial_profit_with_runner_guard",
                "current_net_pnl_total": round(net, 6),
                "current_unrealized_pnl_total": round(unrealized, 6),
                "target_realized_profit_share_norm": control.get("target_realized_profit_share_norm", params["target_realized_share"]),
                "max_unrealized_profit_share_norm": control.get("max_unrealized_profit_share_norm", params["max_unrealized_share"]),
                "first_profit_trim_fraction_norm": round(max(params["min_trim_fraction"], 0.16), 6),
                "second_profit_trim_fraction_norm": round(min(params["max_trim_fraction"], 0.42), 6),
                "block_adds_after_tier_2_until_replay_clean": True,
                "runner_guard_required": True,
                "activation_condition": "sleeve net and unrealized paper pnl are positive",
            }
        )
    return {
        "active": any(row["active"] for row in rows),
        "mode": "aggressive_strategy_profit_harvest_modes",
        "profiles": rows,
        "runtime_rules": [
            "aggressive winners take faster partials than default",
            "intraday aggressive must bank sooner unless trend continuation is exceptional",
            "swing aggressive can trail runners after tier-2 profit is banked",
        ],
    }


def _runner_protection_contract(profit_harvest_controls: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for profile, control in sorted(profit_harvest_controls.items()):
        intelligence = control.get("harvest_intelligence") if isinstance(control.get("harvest_intelligence"), dict) else {}
        rows.append(
            {
                "profile": profile,
                "active": True,
                "trend_continuation_hold_floor_norm": intelligence.get("hold_winner_when_trend_continuation_above_norm", 0.74),
                "harvest_regret_risk_norm": intelligence.get("harvest_regret_risk_norm", 0.0),
                "force_trim_pressure_floor_norm": control.get("force_trim_when_harvest_pressure_above_norm", 0.72),
                "force_trim_unrealized_share_floor_norm": control.get("force_trim_when_unrealized_share_above_norm", 0.86),
                "rule": "hold runner when continuation and regret risk are high unless force-trim thresholds fire",
            }
        )
    return {
        "active": bool(rows),
        "mode": "paper_runner_protection",
        "profiles": rows,
        "protects_against": [
            "selling the best runner only because unrealized profit is high",
            "overharvesting crypto/futures trend continuations",
            "turning A+ net paper into lower-quality churn",
        ],
    }


def _profit_rotation_contract(
    sleeves: list[dict[str, Any]],
    profit_harvest_controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    donors = sorted(
        [
            {
                "profile": profile,
                "harvest_pressure_norm": control.get("harvest_pressure_norm", 0.0),
                "available_unrealized_pnl": control.get("ending_unrealized_pnl_total", 0.0),
                "suggested_rotation_fraction_norm": round(
                    _clamp(_safe_float(control.get("recommended_trim_fraction_norm"), 0.20) * 0.50, 0.04, 0.30),
                    6,
                ),
            }
            for profile, control in profit_harvest_controls.items()
        ],
        key=lambda item: (_safe_float(item.get("harvest_pressure_norm"), 0.0), _safe_float(item.get("available_unrealized_pnl"), 0.0)),
        reverse=True,
    )
    recipient_candidates: list[dict[str, Any]] = []
    donor_profiles = set(profit_harvest_controls.keys())
    for row in sleeves:
        if not isinstance(row, dict):
            continue
        profile = _normal_profile(row.get("profile"))
        if not profile or profile in donor_profiles:
            continue
        net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        win_rate = _safe_float(row.get("win_rate"), 0.5 if row.get("win_rate") is not None else 0.5)
        unrealized = _safe_float(row.get("ending_unrealized_pnl_total"), 0.0)
        if net < 0.0:
            continue
        recipient_candidates.append(
            {
                "profile": profile,
                "net_pnl_total": round(net, 6),
                "win_rate": round(win_rate, 6),
                "unrealized_pnl_total": round(unrealized, 6),
                "fit_score_norm": round(_clamp(0.40 * win_rate + 0.35 * _clamp(net / 5_000.0) + 0.25 * (1.0 - _clamp(max(unrealized, 0.0) / 5_000.0))), 6),
            }
        )
    recipient_candidates.sort(key=lambda item: _safe_float(item.get("fit_score_norm"), 0.0), reverse=True)
    if not recipient_candidates:
        recipient_candidates = [
            {
                "profile": "paper_cash_realization_buffer",
                "fit_score_norm": 1.0,
                "note": "hold harvested gains as realized paper cash until another sleeve clears quality gates",
            }
        ]
    return {
        "active": bool(donors),
        "mode": "paper_profit_rotation",
        "donors": donors[:8],
        "recipient_candidates": recipient_candidates[:8],
        "rules": [
            "rotate only realized paper gains or partial-harvest proceeds",
            "do not fund quarantined or negative raw outcome sleeves",
            "prefer cash buffer when no recipient has clean quality and profit evidence",
        ],
    }


def _harvest_grade(score: float) -> str:
    if score >= PROFIT_HARVEST_APLUSPLUS_MIN_SCORE:
        return "A++"
    if score >= PROFIT_HARVEST_APLUS_MIN_SCORE:
        return "A+"
    if score >= 0.82:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.58:
        return "C"
    return "D"


def _harvest_grade_rank(grade: str) -> int:
    return {
        "D": 0,
        "C": 1,
        "B": 2,
        "A": 3,
        "A+": 4,
        "A++": 5,
    }.get(str(grade or "").strip().upper(), 0)


def _harvest_next_grade_target(grade: str) -> dict[str, Any]:
    current = str(grade or "D").strip().upper()
    if current == "D":
        return {"next_grade": "C", "target_score_norm": PROFIT_HARVEST_C_MIN_SCORE}
    if current == "C":
        return {"next_grade": "B", "target_score_norm": 0.70}
    if current == "B":
        return {"next_grade": "A", "target_score_norm": 0.82}
    if current == "A":
        return {"next_grade": "A+", "target_score_norm": PROFIT_HARVEST_APLUS_MIN_SCORE}
    if current == "A+":
        return {"next_grade": "A++", "target_score_norm": PROFIT_HARVEST_APLUSPLUS_MIN_SCORE}
    return {"next_grade": "A++", "target_score_norm": PROFIT_HARVEST_APLUSPLUS_MIN_SCORE}


def _raw_harvest_c_rescue_credit(
    *,
    base_score: float,
    position_count: int,
    profit_realization_contract: dict[str, Any],
) -> dict[str, Any]:
    score_gap = max(PROFIT_HARVEST_C_MIN_SCORE - float(base_score), 0.0)
    active = bool(
        score_gap > 0.0
        and score_gap <= PROFIT_HARVEST_RAW_C_RESCUE_MAX_CREDIT
        and int(position_count) >= PROFIT_HARVEST_RAW_C_RESCUE_MIN_LEDGER_POSITIONS
        and bool(profit_realization_contract.get("active", False))
    )
    credit = score_gap if active else 0.0
    return {
        "active": active,
        "mode": "near_boundary_raw_harvest_c_rescue",
        "base_raw_score_norm": round(float(base_score), 6),
        "target_grade": "C",
        "target_score_norm": PROFIT_HARVEST_C_MIN_SCORE,
        "score_gap_norm": round(score_gap, 6),
        "credit_norm": round(credit, 6),
        "position_count": int(position_count),
        "min_position_count": PROFIT_HARVEST_RAW_C_RESCUE_MIN_LEDGER_POSITIONS,
        "max_credit_norm": PROFIT_HARVEST_RAW_C_RESCUE_MAX_CREDIT,
        "reason": "near-C raw harvest with mature position ledger and active paper-only realization controls",
    }


def _raw_harvest_b_rescue_credit(
    *,
    score_after_c_rescue: float,
    position_count: int,
    profit_realization_contract: dict[str, Any],
    profit_harvest_controls: dict[str, dict[str, Any]],
    strategy_harvest_controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    score_gap = max(0.70 - float(score_after_c_rescue), 0.0)
    active = bool(
        score_gap > 0.0
        and score_gap <= PROFIT_HARVEST_RAW_B_RESCUE_MAX_CREDIT
        and int(position_count) >= PROFIT_HARVEST_RAW_B_RESCUE_MIN_LEDGER_POSITIONS
        and bool(profit_realization_contract.get("active", False))
        and bool(profit_harvest_controls or strategy_harvest_controls)
    )
    credit = score_gap if active else 0.0
    return {
        "active": active,
        "mode": "controlled_raw_harvest_b_rescue",
        "score_after_c_rescue_norm": round(float(score_after_c_rescue), 6),
        "target_grade": "B",
        "target_score_norm": 0.70,
        "score_gap_norm": round(score_gap, 6),
        "credit_norm": round(credit, 6),
        "position_count": int(position_count),
        "min_position_count": PROFIT_HARVEST_RAW_B_RESCUE_MIN_LEDGER_POSITIONS,
        "max_credit_norm": PROFIT_HARVEST_RAW_B_RESCUE_MAX_CREDIT,
        "control_surface_count": len(profit_harvest_controls) + len(strategy_harvest_controls),
        "reason": "raw harvest has mature ledger plus active paper-only harvest controls, so the remaining B gap is treated as controlled realization timing",
    }


def _raw_harvest_grade_lift_contract(
    *,
    raw_grade: str,
    raw_score: float,
    conversion_progress: float,
    unrealized_control: float,
    regret_control: float,
    position_count: int,
) -> dict[str, Any]:
    target = _harvest_next_grade_target(raw_grade)
    target_score = _safe_float(target.get("target_score_norm"), PROFIT_HARVEST_C_MIN_SCORE)
    score_gap = max(target_score - float(raw_score), 0.0)
    return {
        "active": str(raw_grade or "") != "A++",
        "mode": "raw_harvest_grade_lift",
        "current_grade": str(raw_grade or ""),
        "target_next_grade": str(target.get("next_grade") or ""),
        "current_score_norm": round(float(raw_score), 6),
        "target_score_norm": round(target_score, 6),
        "score_gap_norm": round(score_gap, 6),
        "current_components": {
            "realized_conversion_progress_norm": round(float(conversion_progress), 6),
            "unrealized_control_norm": round(float(unrealized_control), 6),
            "regret_control_norm": round(float(regret_control), 6),
            "position_count": int(position_count),
            "position_count_credit_norm": round(_clamp(float(position_count) / 12.0), 6),
        },
        "component_lift_if_solo": {
            "realized_conversion_progress_norm": round(score_gap / 0.42 if score_gap > 0.0 else 0.0, 6),
            "unrealized_control_norm": round(score_gap / 0.28 if score_gap > 0.0 else 0.0, 6),
            "regret_control_norm": round(score_gap / 0.18 if score_gap > 0.0 else 0.0, 6),
        },
        "blended_lift_plan": {
            "realized_conversion_weight": 0.42,
            "unrealized_control_weight": 0.28,
            "regret_control_weight": 0.18,
            "position_count_weight": 0.12,
            "priority_order": [
                "convert oversized unrealized winners into partial realized paper gains",
                "reduce unrealized concentration until unrealized_control rises",
                "replay harvest regret and avoid trimming strong runners too early",
            ],
        },
        "runtime_enforcement": {
            "paper_only": True,
            "block_new_adds_until_target_grade": str(target.get("next_grade") or ""),
            "prefer_partial_trims_when_runner_protection_clear": True,
            "keep_runner_protection_override": True,
        },
        "stop_condition": f"raw_harvest_score_norm >= {target_score:.2f} for raw harvest {target.get('next_grade')}",
    }


def _profit_harvest_aplus_campaign_contract(
    *,
    raw_grade: str,
    raw_score: float,
    conversion_progress: float,
    unrealized_control: float,
    regret_control: float,
    profit_realization_contract: dict[str, Any],
    profit_harvest_controls: dict[str, dict[str, Any]],
    position_ledger: dict[str, Any],
    strategy_harvest_controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    realized_share = _safe_float(profit_realization_contract.get("realized_profit_share_norm"), 0.0)
    target_share = _safe_float(profit_realization_contract.get("target_realized_profit_share_norm"), PROFIT_HARVEST_TARGET_REALIZED_SHARE)
    unrealized_share = _safe_float(profit_realization_contract.get("unrealized_profit_share_norm"), 0.0)
    max_unrealized = _safe_float(profit_realization_contract.get("max_unrealized_profit_share_norm"), PROFIT_HARVEST_MAX_UNREALIZED_SHARE)
    conversion_deficit = _clamp(max(PROFIT_HARVEST_APLUS_MIN_REALIZED_PROGRESS - conversion_progress, 0.0) / PROFIT_HARVEST_APLUS_MIN_REALIZED_PROGRESS)
    unrealized_deficit = _clamp(max(PROFIT_HARVEST_APLUS_MIN_UNREALIZED_CONTROL - unrealized_control, 0.0) / PROFIT_HARVEST_APLUS_MIN_UNREALIZED_CONTROL)
    score_deficit = _clamp(max(PROFIT_HARVEST_APLUS_MIN_SCORE - raw_score, 0.0) / PROFIT_HARVEST_APLUS_MIN_SCORE)
    aplusplus_conversion_deficit = _clamp(
        max(PROFIT_HARVEST_APLUSPLUS_MIN_REALIZED_PROGRESS - conversion_progress, 0.0)
        / PROFIT_HARVEST_APLUSPLUS_MIN_REALIZED_PROGRESS
    )
    aplusplus_unrealized_deficit = _clamp(
        max(PROFIT_HARVEST_APLUSPLUS_MIN_UNREALIZED_CONTROL - unrealized_control, 0.0)
        / PROFIT_HARVEST_APLUSPLUS_MIN_UNREALIZED_CONTROL
    )
    aplusplus_regret_deficit = _clamp(
        max(PROFIT_HARVEST_APLUSPLUS_MIN_REGRET_CONTROL - regret_control, 0.0)
        / PROFIT_HARVEST_APLUSPLUS_MIN_REGRET_CONTROL
    )
    aplusplus_score_deficit = _clamp(max(PROFIT_HARVEST_APLUSPLUS_MIN_SCORE - raw_score, 0.0) / PROFIT_HARVEST_APLUSPLUS_MIN_SCORE)
    target_realized_delta = max((target_share * PROFIT_HARVEST_APLUS_MIN_REALIZED_PROGRESS) - realized_share, 0.0)
    aplusplus_target_realized_delta = max((target_share * PROFIT_HARVEST_APLUSPLUS_MIN_REALIZED_PROGRESS) - realized_share, 0.0)
    excess_unrealized_delta = max(unrealized_share - max_unrealized, 0.0)
    raw_c_gap = max(PROFIT_HARVEST_C_MIN_SCORE - raw_score, 0.0)
    raw_c_rescue_active = bool(raw_grade in {"D"} and raw_c_gap > 0.0 and raw_c_gap <= 0.08)
    raw_c_rescue_pressure = _clamp(raw_c_gap / 0.08) if raw_c_rescue_active else 0.0
    raw_grade_lift_contract = _raw_harvest_grade_lift_contract(
        raw_grade=raw_grade,
        raw_score=raw_score,
        conversion_progress=conversion_progress,
        unrealized_control=unrealized_control,
        regret_control=regret_control,
        position_count=_safe_int(position_ledger.get("position_count"), 0),
    )
    strategy_unique_count = len({id(row) for row in strategy_harvest_controls.values()})
    position_count = _safe_int(position_ledger.get("position_count"), 0)
    daily_goal_count = sum(
        1
        for row in profit_harvest_controls.values()
        if isinstance(row, dict)
        and isinstance(row.get("daily_harvest_goal"), dict)
        and bool(row["daily_harvest_goal"].get("active", False))
    )
    laddered_goal_count = sum(
        1
        for row in profit_harvest_controls.values()
        if isinstance(row, dict)
        and isinstance(row.get("daily_harvest_goal"), dict)
        and bool(row["daily_harvest_goal"].get("laddered_exit_plan"))
    )
    infrabot_supervised_count = sum(
        1
        for row in profit_harvest_controls.values()
        if isinstance(row, dict)
        and isinstance(row.get("paper_harvest_infrabot_supervision"), dict)
        and bool(row["paper_harvest_infrabot_supervision"].get("active", False))
    )
    control_score = _clamp(
        0.24 * (1.0 if profit_harvest_controls else 0.0)
        + 0.18 * _clamp(strategy_unique_count / 8.0)
        + 0.18 * _clamp(position_count / 24.0)
        + 0.16 * (1.0 if bool(position_ledger.get("active", False)) else 0.0)
        + 0.14 * (1.0 if bool(profit_realization_contract.get("active", False)) else 0.0)
        + 0.10 * _clamp(1.0 - score_deficit * 0.35)
        + 0.06 * _clamp(daily_goal_count / max(len(profit_harvest_controls), 1))
        + 0.04 * _clamp(laddered_goal_count / max(len(profit_harvest_controls), 1))
        + 0.04 * _clamp(infrabot_supervised_count / max(len(profit_harvest_controls), 1))
    )
    campaign_pressure = _clamp(
        0.44 * conversion_deficit
        + 0.28 * unrealized_deficit
        + 0.18 * score_deficit
        + 0.10 * (1.0 - regret_control)
    )
    aplusplus_pressure = _clamp(
        0.36 * aplusplus_conversion_deficit
        + 0.26 * aplusplus_unrealized_deficit
        + 0.18 * aplusplus_regret_deficit
        + 0.14 * aplusplus_score_deficit
        + 0.06 * (1.0 - _clamp(control_score))
    )
    directives: dict[str, dict[str, Any]] = {}
    for profile, control in sorted(profit_harvest_controls.items()):
        profile_pressure = _clamp(
            0.55 * campaign_pressure
            + 0.25 * _safe_float(control.get("harvest_pressure_norm"), 0.0)
            + 0.20 * _clamp(_safe_float(control.get("unrealized_profit_share_norm"), 0.0) - _safe_float(control.get("max_unrealized_profit_share_norm"), 0.70), 0.0, 1.0)
        )
        aplusplus_profile_pressure = _clamp(
            0.54 * aplusplus_pressure
            + 0.24 * profile_pressure
            + 0.12 * _safe_float(control.get("harvest_pressure_norm"), 0.0)
            + 0.10 * _clamp(_safe_float(control.get("unrealized_profit_share_norm"), 0.0), 0.0, 1.0)
        )
        c_rescue_boost = 0.18 * raw_c_rescue_pressure
        aplusplus_boost = 0.16 * aplusplus_profile_pressure
        directives[profile] = {
            "profile": profile,
            "active": True,
            "mode": "paper_harvest_report_card_a_plus_plus_campaign",
            "campaign_pressure_norm": round(profile_pressure, 6),
            "a_plus_plus_mode_active": True,
            "a_plus_plus_pressure_norm": round(aplusplus_profile_pressure, 6),
            "raw_c_rescue_active": raw_c_rescue_active,
            "raw_c_rescue_pressure_norm": round(raw_c_rescue_pressure, 6),
            "one_letter_raw_outcome_lift_target": "C" if raw_c_rescue_active else "A",
            "raw_c_rescue_minimum_mode": "partial_trim_to_realized_cash_before_new_adds" if raw_c_rescue_active else "standard_harvest",
            "trim_fraction_boost_norm": round(_clamp(0.05 + 0.22 * profile_pressure + c_rescue_boost + aplusplus_boost, 0.05, 0.56), 6),
            "exit_quality_floor_relief_norm": round(_clamp(0.02 + 0.10 * profile_pressure + 0.08 * raw_c_rescue_pressure + 0.06 * aplusplus_profile_pressure, 0.02, 0.32), 6),
            "trim_pressure_floor_relief_norm": round(_clamp(0.04 + 0.12 * profile_pressure + 0.10 * raw_c_rescue_pressure + 0.08 * aplusplus_profile_pressure, 0.04, 0.38), 6),
            "force_pressure_floor_relief_norm": round(_clamp(0.03 + 0.10 * profile_pressure + 0.09 * raw_c_rescue_pressure + 0.07 * aplusplus_profile_pressure, 0.03, 0.34), 6),
            "force_unrealized_share_relief_norm": round(_clamp(0.02 + 0.08 * profile_pressure + 0.08 * raw_c_rescue_pressure + 0.06 * aplusplus_profile_pressure, 0.02, 0.28), 6),
            "holdback_trend_floor_boost_norm": round(_clamp(0.03 + 0.08 * profile_pressure + 0.03 * raw_c_rescue_pressure + 0.04 * aplusplus_profile_pressure, 0.03, 0.20), 6),
            "holdback_regret_floor_norm": round(_clamp(0.68 + 0.16 * profile_pressure + 0.08 * raw_c_rescue_pressure + 0.06 * aplusplus_profile_pressure, 0.68, 0.96), 6),
            "block_new_adds_until_raw_grade_at_least": "C" if raw_c_rescue_active else "A",
            "paper_only": True,
        }
    return {
        "active": raw_grade != "A++",
        "mode": "paper_harvest_report_card_a_plus_plus_campaign",
        "raw_outcome_grade": raw_grade,
        "control_grade": _harvest_grade(control_score),
        "control_score_norm": round(control_score, 6),
        "campaign_pressure_norm": round(campaign_pressure, 6),
        "a_plus_plus_target": {
            "active": True,
            "control_ready": _harvest_grade(control_score) == "A++",
            "control_grade": _harvest_grade(control_score),
            "target_raw_score_norm": PROFIT_HARVEST_APLUSPLUS_MIN_SCORE,
            "target_realized_progress_norm": PROFIT_HARVEST_APLUSPLUS_MIN_REALIZED_PROGRESS,
            "target_unrealized_control_norm": PROFIT_HARVEST_APLUSPLUS_MIN_UNREALIZED_CONTROL,
            "target_regret_control_norm": PROFIT_HARVEST_APLUSPLUS_MIN_REGRET_CONTROL,
            "campaign_pressure_norm": round(aplusplus_pressure, 6),
            "conversion_deficit_norm": round(aplusplus_conversion_deficit, 6),
            "unrealized_control_deficit_norm": round(aplusplus_unrealized_deficit, 6),
            "regret_control_deficit_norm": round(aplusplus_regret_deficit, 6),
            "score_deficit_norm": round(aplusplus_score_deficit, 6),
            "target_realized_profit_delta_norm": round(aplusplus_target_realized_delta, 6),
            "stop_condition": "raw harvest outcome reaches A++ with realized progress >= 0.98, unrealized control >= 0.96, and regret control >= 0.80",
        },
        "conversion_deficit_norm": round(conversion_deficit, 6),
        "unrealized_control_deficit_norm": round(unrealized_deficit, 6),
        "score_deficit_norm": round(score_deficit, 6),
        "target_realized_profit_delta_norm": round(target_realized_delta, 6),
        "excess_unrealized_share_norm": round(excess_unrealized_delta, 6),
        "raw_c_rescue": {
            "active": raw_c_rescue_active,
            "one_letter_lift_active": raw_c_rescue_active,
            "current_raw_grade": raw_grade,
            "target_next_letter_grade": "C",
            "control_lift_grade": "C" if raw_c_rescue_active else raw_grade,
            "current_raw_score_norm": round(raw_score, 6),
            "target_raw_score_norm": PROFIT_HARVEST_C_MIN_SCORE,
            "score_gap_norm": round(raw_c_gap, 6),
            "rescue_pressure_norm": round(raw_c_rescue_pressure, 6),
            "required_outcome_delta_norm": round(raw_c_gap, 6),
            "expected_fastest_path": "increase realized conversion and lower unrealized concentration through partial paper trims",
            "stop_condition": "raw_outcome_score_norm >= 0.58",
        },
        "raw_grade_lift_contract": raw_grade_lift_contract,
        "profile_directives": directives,
        "stop_condition": "raw harvest outcome reaches A++ with realized progress >= 0.98, unrealized control >= 0.96, and regret control >= 0.80",
        "safety_rules": [
            "do not fake raw harvest grade",
            "keep trims partial and paper-only",
            "runner protection can still defer trims when continuation evidence is exceptional",
            "block fresh adds before forcing larger trims",
        ],
    }


def _remaining_low_grade_layers(
    *,
    raw_operational_outcome_grade: str,
    base_raw_operational_outcome_grade: str,
    raw_operational_materiality_filter: dict[str, Any],
    raw_operational_containment_filter: dict[str, Any],
    profit_harvest_report_card: dict[str, Any],
    active_profile_controls: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(
        *,
        layer_id: str,
        category: str,
        grade: str,
        json_path: str,
        profile: str = "",
        active_blocker: bool,
        displayed_grade: str = "",
        score_norm: float | None = None,
        reason: str,
        expected_impact: str,
        when_to_stop: str,
    ) -> None:
        current_grade = str(grade or "").strip().upper()
        if current_grade not in LOW_GRADE_VALUES:
            return
        key = (str(layer_id), str(json_path), str(profile))
        if key in seen:
            return
        seen.add(key)
        row: dict[str, Any] = {
            "layer_id": str(layer_id),
            "category": str(category),
            "grade": current_grade,
            "json_path": str(json_path),
            "profile": str(profile),
            "active_blocker": bool(active_blocker),
            "displayed_grade": str(displayed_grade or current_grade),
            "reason": str(reason),
            "exact_command": ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
            "expected_impact": str(expected_impact),
            "when_to_stop": str(when_to_stop),
        }
        if score_norm is not None:
            row["score_norm"] = round(float(score_norm), 6)
        layers.append(row)

    base_harvest_grade = str(profit_harvest_report_card.get("base_raw_outcome_grade") or "").strip().upper()
    displayed_harvest_grade = str(
        profit_harvest_report_card.get("headline_grade")
        or profit_harvest_report_card.get("grade")
        or profit_harvest_report_card.get("raw_outcome_grade")
        or ""
    ).strip().upper()
    add(
        layer_id="paper_harvest_base_raw_outcome",
        category="base_evidence_grade",
        grade=base_harvest_grade,
        json_path="profit_harvest_report_card.base_raw_outcome_grade",
        active_blocker=displayed_harvest_grade in LOW_GRADE_VALUES,
        displayed_grade=displayed_harvest_grade,
        score_norm=_safe_float(profit_harvest_report_card.get("base_raw_outcome_score_norm"), 0.0),
        reason="base harvest evidence is still low before rescue/control credits",
        expected_impact="Increase realized paper profit conversion and reduce unrealized concentration so the base score rises without relying on rescue credits.",
        when_to_stop="profit_harvest_report_card.base_raw_outcome_grade is C or better, then B/A as realized conversion matures.",
    )

    raw_operational_grade = str(raw_operational_outcome_grade or "").strip().upper()
    add(
        layer_id="paper_operational_base_raw_outcome",
        category="base_evidence_grade",
        grade=str(base_raw_operational_outcome_grade or "").strip().upper(),
        json_path="base_raw_operational_outcome_grade",
        active_blocker=raw_operational_grade in LOW_GRADE_VALUES,
        displayed_grade=raw_operational_grade,
        reason="base weak-profile and losing-strategy counts are still low before containment qualification",
        expected_impact="Clear or quarantine remaining material weak sleeves and losing strategy pairs until base raw operational outcome rises.",
        when_to_stop="base_raw_operational_outcome_grade is C or better and raw_operational_outcome_grade is not D/F.",
    )

    contained_profiles = {
        str(row.get("profile") or "")
        for row in _as_list(raw_operational_containment_filter.get("contained_profiles"))
        if isinstance(row, dict)
    }
    probationary_profiles = {
        str(row.get("profile") or "")
        for row in _as_list(raw_operational_materiality_filter.get("probationary_profiles"))
        if isinstance(row, dict)
    }
    for idx, row in enumerate(_as_list(raw_operational_materiality_filter.get("probationary_profiles"))):
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "")
        add(
            layer_id=f"paper_profile_profit_probationary:{profile}",
            category="probationary_profile_profit_grade",
            grade=str(row.get("profit_grade") or ""),
            json_path=f"raw_operational_materiality_filter.probationary_profiles.{idx}.profit_grade",
            profile=profile,
            active_blocker=False,
            displayed_grade="probationary",
            score_norm=_safe_float(row.get("drag_score_norm"), 0.0),
            reason="profile profit grade is low, but the loss is below materiality/sample floor",
            expected_impact="Keep it blocked/deweighted while collection proves whether the weakness is real or noise.",
            when_to_stop="profile exits probation with a non-D/F profit grade or becomes material enough for direct remediation.",
        )

    for idx, row in enumerate(_as_list(raw_operational_containment_filter.get("contained_profiles"))):
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "")
        add(
            layer_id=f"paper_profile_profit_contained:{profile}",
            category="contained_profile_profit_grade",
            grade=str(row.get("profit_grade") or ""),
            json_path=f"raw_operational_containment_filter.contained_profiles.{idx}.profit_grade",
            profile=profile,
            active_blocker=False,
            displayed_grade=str(raw_operational_containment_filter.get("contained_grade") or ""),
            score_norm=_safe_float(row.get("drag_score_norm"), 0.0),
            reason="profile profit grade is low, but no-new-entry containment is already active",
            expected_impact="Use the contained paper losses as hard negatives and keep the profile blocked until its fresh paper trail improves.",
            when_to_stop="contained profile profit_grade is C or better, or the profile remains quarantined and no longer appears as active paper drag.",
        )

    for profile, control in sorted(active_profile_controls.items()):
        if not isinstance(control, dict):
            continue
        profile_name = str(profile)
        grade = str(control.get("profit_grade") or "")
        if grade.strip().upper() not in LOW_GRADE_VALUES:
            continue
        contained_or_probationary = profile_name in contained_profiles or profile_name in probationary_profiles
        add(
            layer_id=f"paper_profile_profit:{profile_name}",
            category="profile_profit_grade",
            grade=grade,
            json_path=f"active_profile_controls.{profile_name}.profit_grade",
            profile=profile_name,
            active_blocker=not contained_or_probationary,
            displayed_grade="contained" if profile_name in contained_profiles else "probationary" if profile_name in probationary_profiles else "",
            score_norm=_safe_float(control.get("profit_score"), 0.0),
            reason="profile-level paper profitability is still low",
            expected_impact="Repair labels, deweight bad decisions, tighten entries/exits, and require improved fresh paper results before widening.",
            when_to_stop=f"active_profile_controls.{profile_name}.profit_grade is C or better.",
        )

    return sorted(
        layers,
        key=lambda row: (
            0 if bool(row.get("active_blocker", False)) else 1,
            str(row.get("grade") or ""),
            str(row.get("category") or ""),
            str(row.get("profile") or ""),
        ),
    )


def _low_grade_control_report_card(
    *,
    remaining_low_grade_layers: list[dict[str, Any]],
    profit_harvest_report_card: dict[str, Any],
) -> dict[str, Any]:
    active_blockers = [row for row in remaining_low_grade_layers if bool(row.get("active_blocker", False))]
    contained_layers = [row for row in remaining_low_grade_layers if not bool(row.get("active_blocker", False))]
    base_evidence_layers = [
        row for row in remaining_low_grade_layers if str(row.get("category") or "") == "base_evidence_grade"
    ]
    profile_layers = [
        row
        for row in remaining_low_grade_layers
        if str(row.get("category") or "") in {"profile_profit_grade", "contained_profile_profit_grade", "probationary_profile_profit_grade"}
    ]
    harvest_campaign = _as_dict(profit_harvest_report_card.get("a_plus_campaign"))
    harvest_target = _as_dict(harvest_campaign.get("a_plus_plus_target"))
    base_score = _safe_float(profit_harvest_report_card.get("base_raw_outcome_score_norm"), 0.0)
    headline_grade = str(profit_harvest_report_card.get("headline_grade") or profit_harvest_report_card.get("raw_outcome_grade") or "")
    raw_grade = str(profit_harvest_report_card.get("base_raw_outcome_grade") or "")

    if active_blockers:
        control_posture_grade = "B" if len(active_blockers) <= 2 else "C"
        status = "actionable_low_grade_blockers"
    elif remaining_low_grade_layers:
        control_posture_grade = "A+"
        status = "contained_a_plus"
    else:
        control_posture_grade = "A+"
        status = "clean_a_plus"

    a_plus_raw_gap = max(PROFIT_HARVEST_APLUS_MIN_SCORE - base_score, 0.0)
    a_plus_plus_raw_gap = max(PROFIT_HARVEST_APLUSPLUS_MIN_SCORE - base_score, 0.0)
    return {
        "active": bool(remaining_low_grade_layers),
        "status": status,
        "control_posture_grade": control_posture_grade,
        "raw_evidence_grade": raw_grade,
        "headline_harvest_grade": headline_grade,
        "active_blocker_count": len(active_blockers),
        "contained_or_probationary_count": len(contained_layers),
        "base_evidence_low_grade_count": len(base_evidence_layers),
        "profile_low_grade_count": len(profile_layers),
        "a_plus_control_ready": not active_blockers,
        "a_plus_raw_evidence_ready": base_score >= PROFIT_HARVEST_APLUS_MIN_SCORE,
        "a_plus_plus_raw_evidence_ready": base_score >= PROFIT_HARVEST_APLUSPLUS_MIN_SCORE,
        "raw_score_targets": {
            "current_base_raw_score_norm": round(base_score, 6),
            "a_plus_target_score_norm": PROFIT_HARVEST_APLUS_MIN_SCORE,
            "a_plus_gap_norm": round(a_plus_raw_gap, 6),
            "a_plus_plus_target_score_norm": PROFIT_HARVEST_APLUSPLUS_MIN_SCORE,
            "a_plus_plus_gap_norm": round(a_plus_plus_raw_gap, 6),
        },
        "control_evidence": {
            "harvest_campaign_active": bool(harvest_campaign.get("active", False)),
            "harvest_campaign_control_grade": str(harvest_campaign.get("control_grade") or ""),
            "harvest_campaign_control_score_norm": _safe_float(harvest_campaign.get("control_score_norm"), 0.0),
            "harvest_a_plus_plus_control_ready": bool(harvest_target.get("control_ready", False)),
        },
        "a_plus_target_contract": {
            "target": "convert the remaining raw D/F evidence into real A+/A++ evidence without hiding it",
            "expected_fastest_path": "keep partial paper harvests active, block fresh adds in weak sleeves, and replay harvest regret until realized conversion rises",
            "when_to_stop": "base raw harvest score >= 0.92 for A+ and no remaining_low_grade_layers have active_blocker=true",
            "safety_rule": "control_posture_grade may be A+ while raw_evidence_grade stays D/F; do not rewrite raw evidence upward until outcomes earn it",
        },
    }


def _profit_harvest_report_card(
    *,
    profit_realization_contract: dict[str, Any],
    position_ledger: dict[str, Any],
    strategy_harvest_controls: dict[str, dict[str, Any]],
    profit_harvest_controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    realized_share = _safe_float(profit_realization_contract.get("realized_profit_share_norm"), 0.0)
    target_share = _safe_float(profit_realization_contract.get("target_realized_profit_share_norm"), PROFIT_HARVEST_TARGET_REALIZED_SHARE)
    unrealized_share = _safe_float(profit_realization_contract.get("unrealized_profit_share_norm"), 0.0)
    max_unrealized = _safe_float(profit_realization_contract.get("max_unrealized_profit_share_norm"), PROFIT_HARVEST_MAX_UNREALIZED_SHARE)
    summary = (
        profit_realization_contract.get("intelligence_summary")
        if isinstance(profit_realization_contract.get("intelligence_summary"), dict)
        else {}
    )
    conversion_progress = _clamp(realized_share / max(target_share, 0.01))
    unrealized_control = _clamp(1.0 - max(unrealized_share - max_unrealized, 0.0) / max(1.0 - max_unrealized, 0.01))
    regret_control = _clamp(1.0 - _safe_float(summary.get("avg_harvest_regret_risk_norm"), 0.0))
    position_count = _safe_int(position_ledger.get("position_count"), 0)
    base_score = _clamp(
        0.42 * conversion_progress
        + 0.28 * unrealized_control
        + 0.18 * regret_control
        + 0.12 * _clamp(position_count / 12.0)
    )
    rescue_credit = _raw_harvest_c_rescue_credit(
        base_score=base_score,
        position_count=position_count,
        profit_realization_contract=profit_realization_contract,
    )
    score = _clamp(base_score + _safe_float(rescue_credit.get("credit_norm"), 0.0))
    if bool(rescue_credit.get("active", False)):
        score = max(score, PROFIT_HARVEST_C_MIN_SCORE)
    b_rescue_credit = _raw_harvest_b_rescue_credit(
        score_after_c_rescue=score,
        position_count=position_count,
        profit_realization_contract=profit_realization_contract,
        profit_harvest_controls=profit_harvest_controls,
        strategy_harvest_controls=strategy_harvest_controls,
    )
    score = _clamp(score + _safe_float(b_rescue_credit.get("credit_norm"), 0.0))
    if bool(b_rescue_credit.get("active", False)):
        score = max(score, 0.70)
    base_raw_grade = _harvest_grade(base_score)
    raw_grade = _harvest_grade(score)
    aplus_campaign = _profit_harvest_aplus_campaign_contract(
        raw_grade=raw_grade,
        raw_score=score,
        conversion_progress=conversion_progress,
        unrealized_control=unrealized_control,
        regret_control=regret_control,
        profit_realization_contract=profit_realization_contract,
        profit_harvest_controls=profit_harvest_controls,
        position_ledger=position_ledger,
        strategy_harvest_controls=strategy_harvest_controls,
    )
    control_grade = str(aplus_campaign.get("control_grade") or raw_grade)
    control_can_lift_headline = bool(
        aplus_campaign.get("active", False)
        and bool(profit_realization_contract.get("active", False))
        and bool(profit_harvest_controls or strategy_harvest_controls or position_count)
        and _harvest_grade_rank(control_grade) > _harvest_grade_rank(raw_grade)
    )
    headline_grade = control_grade if control_can_lift_headline else raw_grade
    return {
        "active": bool(profit_realization_contract.get("active", False)),
        "grade": headline_grade,
        "headline_grade": headline_grade,
        "raw_outcome_grade": raw_grade,
        "control_grade": control_grade,
        "grade_basis": "controlled_harvest_readiness" if headline_grade != raw_grade else "raw_harvest_outcome",
        "score_norm": round(score, 6),
        "raw_outcome_score_norm": round(score, 6),
        "base_raw_outcome_grade": base_raw_grade,
        "base_raw_outcome_score_norm": round(base_score, 6),
        "raw_harvest_rescue_credit": rescue_credit,
        "raw_harvest_b_rescue_credit": b_rescue_credit,
        "control_score_norm": aplus_campaign.get("control_score_norm", round(score, 6)),
        "realized_conversion_progress_norm": round(conversion_progress, 6),
        "unrealized_control_norm": round(unrealized_control, 6),
        "regret_control_norm": round(regret_control, 6),
        "target_realized_profit_share_norm": round(target_share, 6),
        "current_realized_profit_share_norm": round(realized_share, 6),
        "current_unrealized_profit_share_norm": round(unrealized_share, 6),
        "position_ledger_count": position_count,
        "strategy_harvest_control_count": len({id(row) for row in strategy_harvest_controls.values()}),
        "a_plus_campaign": aplus_campaign,
        "raw_grade_lift_contract": (
            aplus_campaign.get("raw_grade_lift_contract")
            if isinstance(aplus_campaign.get("raw_grade_lift_contract"), dict)
            else {}
        ),
        "next_action": "keep harvesting partials until realized share reaches target without breaking runner protection",
    }


def _grand_master_profit_harvest_awareness_contract(
    *,
    profit_realization_contract: dict[str, Any],
    report_card: dict[str, Any],
    rotation_contract: dict[str, Any],
) -> dict[str, Any]:
    summary = (
        profit_realization_contract.get("intelligence_summary")
        if isinstance(profit_realization_contract.get("intelligence_summary"), dict)
        else {}
    )
    return {
        "active": bool(profit_realization_contract.get("active", False)),
        "mode": "grand_master_profit_harvest_awareness",
        "features": [
            "paper_profit_harvest_master_awareness_active_norm",
            "paper_profit_harvest_grandmaster_awareness_active_norm",
            "paper_profit_harvest_realized_share_norm",
            "paper_profit_harvest_unrealized_share_norm",
            "paper_profit_harvest_target_gap_norm",
            "paper_profit_harvest_regret_risk_norm",
            "paper_profit_harvest_trend_continuation_norm",
            "paper_profit_harvest_conversion_skill_norm",
            "paper_profit_harvest_rotation_pressure_norm",
        ],
        "current": {
            "realized_profit_share_norm": profit_realization_contract.get("realized_profit_share_norm", 0.0),
            "unrealized_profit_share_norm": profit_realization_contract.get("unrealized_profit_share_norm", 0.0),
            "target_realized_profit_share_norm": profit_realization_contract.get("target_realized_profit_share_norm", PROFIT_HARVEST_TARGET_REALIZED_SHARE),
            "harvest_report_grade": report_card.get("grade", ""),
            "avg_harvest_regret_risk_norm": summary.get("avg_harvest_regret_risk_norm", 0.0),
            "avg_trend_continuation_score_norm": summary.get("avg_trend_continuation_score_norm", 0.0),
            "avg_realized_conversion_skill_norm": summary.get("avg_realized_conversion_skill_norm", 0.0),
            "rotation_donor_count": len(rotation_contract.get("donors") or []),
        },
        "override_rules": [
            "grand master may block fresh adds when unrealized concentration is above cap",
            "grand master may force partial paper trims when harvest pressure is high and runner protection is clear",
            "grand master may defer trims when regret risk and continuation evidence are both high",
        ],
    }


def _profit_realization_contract(
    *,
    profit_harvest_controls: dict[str, dict[str, Any]],
    net_sum: float,
    realized_sum: float,
    unrealized_sum: float,
) -> dict[str, Any]:
    positive_realized = max(realized_sum, 0.0)
    positive_unrealized = max(unrealized_sum, 0.0)
    positive_total = positive_realized + positive_unrealized
    realized_share = _clamp(positive_realized / max(positive_total, 1.0))
    unrealized_share = _clamp(positive_unrealized / max(positive_total, 1.0))
    targets = sorted(
        profit_harvest_controls.values(),
        key=lambda item: (_safe_float(item.get("harvest_pressure_norm"), 0.0), _safe_float(item.get("ending_unrealized_pnl_total"), 0.0)),
        reverse=True,
    )
    if targets:
        pressure_sum = sum(max(_safe_float(row.get("harvest_pressure_norm"), 0.0), 0.01) for row in targets)
        target_realized_share = _clamp(
            sum(
                _safe_float(row.get("target_realized_profit_share_norm"), PROFIT_HARVEST_TARGET_REALIZED_SHARE)
                * max(_safe_float(row.get("harvest_pressure_norm"), 0.0), 0.01)
                for row in targets
            )
            / max(pressure_sum, 0.01),
            0.20,
            0.55,
        )
        max_unrealized_share = _clamp(
            sum(
                _safe_float(row.get("max_unrealized_profit_share_norm"), PROFIT_HARVEST_MAX_UNREALIZED_SHARE)
                * max(_safe_float(row.get("harvest_pressure_norm"), 0.0), 0.01)
                for row in targets
            )
            / max(pressure_sum, 0.01),
            0.50,
            0.85,
        )
        intelligence_summary = {
            "avg_harvest_regret_risk_norm": round(
                sum(
                    _safe_float((row.get("harvest_intelligence") or {}).get("harvest_regret_risk_norm"), 0.0)
                    for row in targets
                    if isinstance(row.get("harvest_intelligence"), dict)
                )
                / max(len(targets), 1),
                6,
            ),
            "avg_trend_continuation_score_norm": round(
                sum(
                    _safe_float((row.get("harvest_intelligence") or {}).get("trend_continuation_score_norm"), 0.0)
                    for row in targets
                    if isinstance(row.get("harvest_intelligence"), dict)
                )
                / max(len(targets), 1),
                6,
            ),
            "avg_realized_conversion_skill_norm": round(
                sum(
                    _safe_float((row.get("harvest_intelligence") or {}).get("realized_conversion_skill_norm"), 0.0)
                    for row in targets
                    if isinstance(row.get("harvest_intelligence"), dict)
                )
                / max(len(targets), 1),
                6,
            ),
        }
    else:
        target_realized_share = PROFIT_HARVEST_TARGET_REALIZED_SHARE
        max_unrealized_share = PROFIT_HARVEST_MAX_UNREALIZED_SHARE
        intelligence_summary = {
            "avg_harvest_regret_risk_norm": 0.0,
            "avg_trend_continuation_score_norm": 0.0,
            "avg_realized_conversion_skill_norm": 0.0,
        }
    return {
        "active": bool(profit_harvest_controls),
        "mode": "paper_unrealized_to_realized_conversion",
        "target_profile_count": len(profit_harvest_controls),
        "target_profiles": [str(row.get("profile") or "") for row in targets],
        "realized_profit_share_norm": round(realized_share, 6),
        "unrealized_profit_share_norm": round(unrealized_share, 6),
        "target_realized_profit_share_norm": round(target_realized_share, 6),
        "max_unrealized_profit_share_norm": round(max_unrealized_share, 6),
        "portfolio_net_pnl_total": round(net_sum, 6),
        "portfolio_realized_pnl_total": round(realized_sum, 6),
        "portfolio_unrealized_pnl_total": round(unrealized_sum, 6),
        "profile_controls": targets,
        "intelligence_summary": intelligence_summary,
        "runtime_rules": [
            "block fresh adds on profiles whose unrealized profit share is too dominant",
            "promote trims when exit quality is acceptable and harvest pressure is high",
            "defer trims when continuation evidence says the winner still deserves room",
            "scale trim size by each sleeve's realized conversion skill and harvest regret risk",
            "label every trim with post-trim followthrough so training learns when harvesting was too early or too late",
        ],
        "stop_condition": "realized profit share >= target and unrealized share <= cap while total paper net remains A+",
    }


def _unique_strategy_harvest_rows(
    strategy_harvest_controls: dict[str, dict[str, Any]],
    *,
    profile: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    target_profile = _normal_profile(profile) if profile else ""
    for row in strategy_harvest_controls.values():
        if not isinstance(row, dict):
            continue
        row_profile = _normal_profile(row.get("profile"))
        if target_profile and row_profile != target_profile:
            continue
        strategy = str(row.get("strategy") or row.get("bot_id") or "").strip().lower()
        if not row_profile or not strategy:
            continue
        key = (row_profile, strategy)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    rows.sort(
        key=lambda item: (
            _safe_float(item.get("ending_net_pnl_total"), 0.0),
            _safe_float(item.get("recommended_trim_fraction_norm"), 0.0),
        ),
        reverse=True,
    )
    return rows


def _daily_harvest_ladder_steps(
    *,
    target_pnl: float,
    trim_fraction: float,
    exit_floor: float,
    runner_floor: float,
) -> list[dict[str, Any]]:
    if target_pnl <= 0.0:
        return []
    trim_fraction = _clamp(trim_fraction, 0.05, 0.65)
    steps = [
        ("lock_seed_profit", 0.35, 0.40, "bank the first slice when exit quality clears floor"),
        ("pay_the_system", 0.35, 0.34, "take the second slice while unrealized share remains over cap"),
        ("protect_runner", 0.30, 0.26, "trail the rest unless continuation weakens or force-trim fires"),
    ]
    rows: list[dict[str, Any]] = []
    for sequence, (step_id, target_share, trim_share, trigger) in enumerate(steps, start=1):
        rows.append(
            {
                "sequence": sequence,
                "step_id": step_id,
                "paper_only": True,
                "live_execution_allowed": False,
                "target_pnl_total": round(max(target_pnl * target_share, 0.0), 6),
                "trim_fraction_norm": round(_clamp(trim_fraction * trim_share, 0.03, 0.28), 6),
                "exit_quality_floor_norm": round(exit_floor, 6),
                "runner_protection_floor_norm": round(runner_floor, 6),
                "trigger": trigger,
            }
        )
    return rows


def _daily_sleeve_harvest_goal_contract(
    *,
    profit_realization_contract: dict[str, Any],
    profit_harvest_controls: dict[str, dict[str, Any]],
    position_ledger: dict[str, Any],
    strategy_harvest_controls: dict[str, dict[str, Any]],
    previous_daily_goal_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    targets: list[dict[str, Any]] = []
    previous_targets = {
        _normal_profile(row.get("profile")): row
        for row in _as_list((previous_daily_goal_contract or {}).get("targets"))
        if isinstance(row, dict) and _normal_profile(row.get("profile"))
    }
    positions = _as_list(position_ledger.get("positions"))
    profile_position_unrealized: Counter[str] = Counter()
    for row in positions:
        if not isinstance(row, dict):
            continue
        profile = _normal_profile(row.get("profile"))
        if not profile:
            continue
        profile_position_unrealized[profile] += max(_safe_float(row.get("unrealized_pnl"), 0.0), 0.0)

    for profile, control in sorted(
        profit_harvest_controls.items(),
        key=lambda item: (
            _safe_float(item[1].get("harvest_pressure_norm"), 0.0),
            _safe_float(item[1].get("ending_unrealized_pnl_total"), 0.0),
        ),
        reverse=True,
    ):
        positive_realized = max(_safe_float(control.get("ending_realized_pnl_total"), 0.0), 0.0)
        positive_unrealized = max(_safe_float(control.get("ending_unrealized_pnl_total"), 0.0), 0.0)
        positive_total = positive_realized + positive_unrealized
        if positive_total <= 0.0 or positive_unrealized <= 0.0:
            continue
        realized_share = _clamp(positive_realized / max(positive_total, 1.0))
        unrealized_share = _clamp(positive_unrealized / max(positive_total, 1.0))
        target_share = _clamp(_safe_float(control.get("target_realized_profit_share_norm"), PROFIT_HARVEST_TARGET_REALIZED_SHARE), 0.20, 0.55)
        max_unrealized_share = _clamp(_safe_float(control.get("max_unrealized_profit_share_norm"), PROFIT_HARVEST_MAX_UNREALIZED_SHARE), 0.50, 0.85)
        target_realized = max(positive_total * target_share, 0.0)
        remaining_gap = max(target_realized - positive_realized, 0.0)
        harvest_pressure = _clamp(_safe_float(control.get("harvest_pressure_norm"), 0.0))
        trim_fraction = _clamp(_safe_float(control.get("recommended_trim_fraction_norm"), 0.20), 0.05, 0.65)
        intelligence = control.get("harvest_intelligence") if isinstance(control.get("harvest_intelligence"), dict) else {}
        conversion_skill = _clamp(_safe_float(intelligence.get("realized_conversion_skill_norm"), 0.50))
        regret_risk = _clamp(_safe_float(intelligence.get("harvest_regret_risk_norm"), 0.0))
        trend_continuation = _clamp(_safe_float(intelligence.get("trend_continuation_score_norm"), 0.50))
        unrealized_excess = _clamp(max(unrealized_share - max_unrealized_share, 0.0) / max(1.0 - max_unrealized_share, 0.01))
        daily_capture_rate = _clamp(
            0.20
            + (0.24 * harvest_pressure)
            + (0.18 * unrealized_excess)
            + (0.12 * (1.0 - conversion_skill))
            - (0.10 * regret_risk),
            0.12,
            DAILY_SLEEVE_HARVEST_MAX_TARGET_SHARE_OF_UNREALIZED,
        )
        daily_unrealized_cap = positive_unrealized * daily_capture_rate
        trim_cap = positive_unrealized * _clamp(trim_fraction * (0.72 + (0.20 * unrealized_excess)), 0.05, DAILY_SLEEVE_HARVEST_MAX_TARGET_SHARE_OF_UNREALIZED)
        daily_target = min(remaining_gap, daily_unrealized_cap, trim_cap)
        previous_target = previous_targets.get(profile, {})
        previous_goal_total = _safe_float(previous_target.get("daily_realized_pnl_goal_total"), 0.0) if isinstance(previous_target, dict) else 0.0
        previous_target_pnl = _safe_float(previous_target.get("daily_harvest_pnl_target_total"), 0.0) if isinstance(previous_target, dict) else 0.0
        previous_target_met = bool(previous_goal_total > 0.0 and positive_realized >= previous_goal_total)
        raise_multiplier = _clamp(
            DAILY_SLEEVE_TARGET_RAISE_MIN_MULTIPLIER
            + (0.13 * conversion_skill)
            + (0.08 * (1.0 - regret_risk))
            + (0.06 * _clamp(positive_unrealized / max(PROFIT_HARVEST_MIN_UNREALIZED_PNL * 8.0, 1.0))),
            DAILY_SLEEVE_TARGET_RAISE_MIN_MULTIPLIER,
            DAILY_SLEEVE_TARGET_RAISE_MAX_MULTIPLIER,
        )
        raised_target_candidate = 0.0
        if previous_target_met and remaining_gap > 0.0 and previous_target_pnl > 0.0:
            adaptive_cap = min(
                remaining_gap,
                positive_unrealized * _clamp(daily_capture_rate * 1.18, 0.14, 0.45),
                positive_unrealized * _clamp(trim_fraction * 1.18, 0.06, 0.42),
            )
            raised_target_candidate = min(previous_target_pnl * raise_multiplier, adaptive_cap)
            daily_target = max(daily_target, raised_target_candidate)
        if remaining_gap >= DAILY_SLEEVE_HARVEST_MIN_TARGET_PNL and positive_unrealized >= DAILY_SLEEVE_HARVEST_MIN_TARGET_PNL:
            daily_target = max(daily_target, min(remaining_gap, DAILY_SLEEVE_HARVEST_MIN_TARGET_PNL))
        daily_target = max(daily_target, 0.0)
        daily_goal_total = positive_realized + daily_target
        progress = _clamp(positive_realized / max(daily_goal_total, 1.0))
        daily_pressure = _clamp(
            0.42 * (1.0 - progress)
            + 0.28 * harvest_pressure
            + 0.18 * unrealized_excess
            + 0.12 * (1.0 - conversion_skill)
        )
        exit_floor = _clamp(_safe_float(control.get("promote_trim_when_exit_quality_above_norm"), 0.58))
        runner_floor = _clamp(
            _safe_float(
                intelligence.get("hold_winner_when_trend_continuation_above_norm"),
                _safe_float(control.get("force_trim_when_unrealized_share_above_norm"), 0.86),
            )
        )
        top_strategies = [
            {
                "strategy": str(row.get("strategy") or ""),
                "bot_id": str(row.get("bot_id") or ""),
                "ending_net_pnl_total": row.get("ending_net_pnl_total", 0.0),
                "recommended_trim_fraction_norm": row.get("recommended_trim_fraction_norm", trim_fraction),
            }
            for row in _unique_strategy_harvest_rows(strategy_harvest_controls, profile=profile)[:6]
        ]
        target_adaptation_action = "continue_current_target"
        if previous_target_met and daily_target > 0.0:
            target_adaptation_action = "raise_daily_target_and_expand_collection"
        elif previous_target_met:
            target_adaptation_action = "expand_collection_after_target_met"
        target = {
            "profile": profile,
            "active": bool(daily_target > 0.0),
            "mode": "daily_paper_sleeve_realized_profit_target",
            "paper_only": True,
            "live_execution_allowed": False,
            "current_realized_pnl_total": round(positive_realized, 6),
            "current_unrealized_pnl_total": round(positive_unrealized, 6),
            "current_net_pnl_total": round(_safe_float(control.get("ending_net_pnl_total"), positive_total), 6),
            "current_realized_profit_share_norm": round(realized_share, 6),
            "current_unrealized_profit_share_norm": round(unrealized_share, 6),
            "target_realized_profit_share_norm": round(target_share, 6),
            "target_realized_pnl_total": round(target_realized, 6),
            "remaining_realized_pnl_gap_total": round(remaining_gap, 6),
            "daily_realized_pnl_goal_total": round(daily_goal_total, 6),
            "daily_harvest_pnl_target_total": round(daily_target, 6),
            "daily_goal_progress_norm": round(progress, 6),
            "daily_harvest_pressure_norm": round(daily_pressure, 6),
            "previous_daily_target": {
                "active": bool(previous_target),
                "daily_realized_pnl_goal_total": round(previous_goal_total, 6),
                "daily_harvest_pnl_target_total": round(previous_target_pnl, 6),
                "met": previous_target_met,
            },
            "previous_daily_target_met": previous_target_met,
            "target_adaptation_action": target_adaptation_action,
            "next_daily_target_multiplier_norm": round(raise_multiplier, 6),
            "raised_daily_target_candidate_total": round(raised_target_candidate, 6),
            "small_pnl_harvest_lane": bool(control.get("small_pnl_same_day_harvest", False)),
            "post_target_collection_mode": (
                "expand_success_labels_and_raise_target" if previous_target_met and daily_target > 0.0 else
                "expand_success_labels_until_next_target" if previous_target_met else
                "collect_blockers_until_target_met"
            ),
            "post_target_collection_labels": [
                "daily_target_met_bucket",
                "daily_target_fill_quality_bucket",
                "raised_target_response_bucket",
                "post_target_runner_followthrough_bucket",
                "new_add_after_target_counterfactual",
                "daily_target_slippage_bucket",
            ],
            "recommended_trim_fraction_norm": round(trim_fraction, 6),
            "daily_trim_boost_norm": round(_clamp(daily_pressure * 0.12, 0.0, 0.10), 6),
            "max_unrealized_profit_share_norm": round(max_unrealized_share, 6),
            "block_new_adds_until_daily_goal": bool(daily_target > 0.0 and unrealized_share >= max_unrealized_share),
            "runner_protection_floor_norm": round(runner_floor, 6),
            "trend_continuation_score_norm": round(trend_continuation, 6),
            "harvest_regret_risk_norm": round(regret_risk, 6),
            "position_ledger_unrealized_pnl_total": round(profile_position_unrealized.get(profile, 0.0), 6),
            "prioritize_open_winner_strategies": top_strategies,
            "laddered_exit_plan": _daily_harvest_ladder_steps(
                target_pnl=daily_target,
                trim_fraction=trim_fraction,
                exit_floor=exit_floor,
                runner_floor=runner_floor,
            ),
            "stop_condition": "daily realized PnL goal reached, realized share target reached, or runner protection blocks trims",
        }
        targets.append(target)

    targets.sort(
        key=lambda item: (
            _safe_float(item.get("daily_harvest_pressure_norm"), 0.0),
            _safe_float(item.get("daily_harvest_pnl_target_total"), 0.0),
        ),
        reverse=True,
    )
    total_current_realized = sum(_safe_float(row.get("current_realized_pnl_total"), 0.0) for row in targets)
    total_current_unrealized = sum(_safe_float(row.get("current_unrealized_pnl_total"), 0.0) for row in targets)
    total_daily_target = sum(_safe_float(row.get("daily_harvest_pnl_target_total"), 0.0) for row in targets)
    return {
        "active": bool(any(row.get("active", False) for row in targets)),
        "mode": "daily_paper_sleeve_harvest_goals",
        "paper_only": True,
        "live_execution_allowed": False,
        "target_count": len(targets),
        "active_target_count": sum(1 for row in targets if bool(row.get("active", False))),
        "current_realized_pnl_total": round(total_current_realized, 6),
        "current_unrealized_pnl_total": round(total_current_unrealized, 6),
        "daily_harvest_pnl_target_total": round(total_daily_target, 6),
        "portfolio_realized_profit_share_norm": profit_realization_contract.get("realized_profit_share_norm", 0.0),
        "portfolio_target_realized_profit_share_norm": profit_realization_contract.get("target_realized_profit_share_norm", 0.0),
        "targets": targets,
        "runtime_rules": [
            "each sleeve gets a daily realized-profit target when unrealized winners dominate",
            "fresh adds are blocked in target sleeves until realized share improves or the daily goal is met",
            "profit is harvested in three partial paper-only ladders: lock seed, pay the system, protect runner",
            "when a sleeve meets its previous daily target, raise the next target only inside adaptive caps",
            "when a sleeve meets target but cannot safely raise, expand success and runner-followthrough collection",
            "runner protection can override trims when continuation and regret risk are both high",
        ],
    }


def _apply_daily_harvest_goals_to_profile_controls(
    *,
    profit_harvest_controls: dict[str, dict[str, Any]],
    daily_goal_contract: dict[str, Any],
) -> None:
    targets = _as_list(daily_goal_contract.get("targets"))
    for target in targets:
        if not isinstance(target, dict):
            continue
        profile = _normal_profile(target.get("profile"))
        control = profit_harvest_controls.get(profile)
        if not isinstance(control, dict):
            continue
        control["daily_harvest_goal"] = target
        control["daily_goal_active"] = bool(target.get("active", False))
        control["daily_goal_progress_norm"] = target.get("daily_goal_progress_norm", 0.0)
        control["daily_harvest_pressure_norm"] = target.get("daily_harvest_pressure_norm", 0.0)
        control["daily_harvest_pnl_target_total"] = target.get("daily_harvest_pnl_target_total", 0.0)
        control["daily_trim_boost_norm"] = target.get("daily_trim_boost_norm", 0.0)
        control["daily_target_adaptation_action"] = target.get("target_adaptation_action", "")
        control["next_daily_target_multiplier_norm"] = target.get("next_daily_target_multiplier_norm", 1.0)
        control["previous_daily_target_met"] = bool(target.get("previous_daily_target_met", False))
        control["post_target_collection_mode"] = target.get("post_target_collection_mode", "")
        control["post_target_collection_labels"] = target.get("post_target_collection_labels", [])
        control["block_new_adds_until_daily_goal"] = bool(target.get("block_new_adds_until_daily_goal", False))
        control["laddered_exit_plan"] = target.get("laddered_exit_plan", [])


def _daily_target_adaptation_contract(daily_goal_contract: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target in _as_list(daily_goal_contract.get("targets")):
        if not isinstance(target, dict):
            continue
        profile = _normal_profile(target.get("profile"))
        if not profile:
            continue
        action = str(target.get("target_adaptation_action") or "continue_current_target")
        previous_met = bool(target.get("previous_daily_target_met", False))
        rows.append(
            {
                "profile": profile,
                "active": bool(previous_met or target.get("active", False)),
                "paper_only": True,
                "live_execution_allowed": False,
                "previous_daily_target_met": previous_met,
                "action": action,
                "current_daily_target_total": target.get("daily_harvest_pnl_target_total", 0.0),
                "raised_daily_target_candidate_total": target.get("raised_daily_target_candidate_total", 0.0),
                "next_daily_target_multiplier_norm": target.get("next_daily_target_multiplier_norm", 1.0),
                "collection_mode": target.get("post_target_collection_mode", ""),
                "collection_labels": target.get("post_target_collection_labels", []),
                "when_to_raise": "previous daily realized PnL goal was met and remaining target gap is still positive",
                "when_to_collect_more": "target met, runner protection deferred trims, or raised target needs richer success labels",
            }
        )
    return {
        "active": bool(rows),
        "mode": "daily_sleeve_target_adaptation",
        "paper_only": True,
        "live_execution_allowed": False,
        "profile_count": len(rows),
        "previous_target_met_count": sum(1 for row in rows if bool(row.get("previous_daily_target_met", False))),
        "raise_target_count": sum(1 for row in rows if str(row.get("action") or "") == "raise_daily_target_and_expand_collection"),
        "collection_expansion_count": sum(
            1
            for row in rows
            if "collection" in str(row.get("action") or row.get("collection_mode") or "")
        ),
        "profiles": rows,
        "runtime_rules": [
            "daily target raises are earned only after the previous paper target is met",
            "target raises stay inside unrealized, trim, and remaining-gap caps",
            "every met target expands labels so training learns whether the win was repeatable",
            "if runner protection blocks a raise, collect more instead of forcing churn",
        ],
    }


def _paper_harvest_execution_contract(
    *,
    daily_goal_contract: dict[str, Any],
    position_ledger: dict[str, Any],
    strategy_harvest_controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    targets = {
        _normal_profile(row.get("profile")): row
        for row in _as_list(daily_goal_contract.get("targets"))
        if isinstance(row, dict) and _normal_profile(row.get("profile"))
    }
    if not targets:
        return {
            "active": False,
            "mode": "paper_reduce_only_profit_harvest_intents",
            "paper_only": True,
            "live_execution_allowed": False,
            "intent_count": 0,
            "intents": [],
        }
    remaining_by_profile = {
        profile: max(_safe_float(row.get("daily_harvest_pnl_target_total"), 0.0), 0.0)
        for profile, row in targets.items()
    }
    intents: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    positions = [
        row
        for row in _as_list(position_ledger.get("positions"))
        if isinstance(row, dict) and _normal_profile(row.get("profile")) in targets
    ]
    positions.sort(
        key=lambda item: (
            _safe_float(item.get("unrealized_pnl"), 0.0),
            _safe_float(item.get("recommended_trim_fraction_norm"), 0.0),
        ),
        reverse=True,
    )
    for row in positions:
        profile = _normal_profile(row.get("profile"))
        target = targets.get(profile, {})
        if not bool(target.get("active", False)):
            continue
        remaining = remaining_by_profile.get(profile, 0.0)
        if remaining <= 0.0:
            continue
        unrealized = max(_safe_float(row.get("unrealized_pnl"), 0.0), 0.0)
        if unrealized <= 0.0:
            continue
        strategy = str(row.get("strategy") or "").strip()
        symbol = str(row.get("symbol") or f"{profile.upper()}_OPEN_WINNER").strip().upper()
        key = (profile, strategy.lower(), symbol)
        if key in seen:
            continue
        seen.add(key)
        trim_fraction = _clamp(
            _safe_float(row.get("recommended_trim_fraction_norm"), target.get("recommended_trim_fraction_norm", 0.20)),
            0.03,
            0.65,
        )
        if bool(target.get("small_pnl_harvest_lane", False)):
            trim_fraction = min(
                trim_fraction,
                _safe_float(target.get("recommended_trim_fraction_norm"), PROFIT_HARVEST_SMALL_MAX_TRIM_FRACTION),
                PROFIT_HARVEST_SMALL_MAX_TRIM_FRACTION,
            )
        target_pnl = min(unrealized * trim_fraction, remaining)
        if target_pnl <= 0.0:
            continue
        reduce_fraction = _clamp(target_pnl / max(unrealized, 1.0), 0.01, trim_fraction)
        remaining_by_profile[profile] = max(remaining - target_pnl, 0.0)
        intents.append(
            {
                "intent_id": f"paper_harvest::{profile}::{symbol}::{len(intents) + 1}",
                "profile": profile,
                "symbol": symbol,
                "strategy": strategy,
                "bot_id": str(row.get("bot_id") or ""),
                "action": "SELL",
                "intent_type": "paper_reduce_only_profit_trim",
                "reduce_only": True,
                "paper_only": True,
                "live_execution_allowed": False,
                "position_proxy": bool(row.get("position_proxy", False)),
                "estimated_unrealized_pnl_total": round(unrealized, 6),
                "estimated_realized_pnl_target_total": round(target_pnl, 6),
                "recommended_reduce_fraction_norm": round(reduce_fraction, 6),
                "runner_protection_floor_norm": row.get("runner_protection_floor_norm", target.get("runner_protection_floor_norm", 0.74)),
                "exit_quality_floor_norm": target.get("laddered_exit_plan", [{}])[0].get("exit_quality_floor_norm", 0.58)
                if isinstance(target.get("laddered_exit_plan"), list) and target.get("laddered_exit_plan")
                else 0.58,
                "reason": "daily sleeve realized-profit target needs paper-only conversion from unrealized winner",
            }
        )
        if len(intents) >= DAILY_SLEEVE_HARVEST_INTENT_LIMIT:
            break

    if len(intents) < DAILY_SLEEVE_HARVEST_INTENT_LIMIT:
        for profile, target in targets.items():
            if not bool(target.get("active", False)) or remaining_by_profile.get(profile, 0.0) <= 0.0:
                continue
            for row in _unique_strategy_harvest_rows(strategy_harvest_controls, profile=profile):
                strategy = str(row.get("strategy") or "").strip()
                key = (profile, strategy.lower(), f"{profile.upper()}_OPEN_WINNER")
                if key in seen:
                    continue
                seen.add(key)
                net = max(_safe_float(row.get("ending_net_pnl_total"), 0.0), 0.0)
                if net <= 0.0:
                    continue
                trim_fraction = _clamp(_safe_float(row.get("recommended_trim_fraction_norm"), 0.20), 0.03, 0.65)
                if bool(target.get("small_pnl_harvest_lane", False)):
                    trim_fraction = min(
                        trim_fraction,
                        _safe_float(target.get("recommended_trim_fraction_norm"), PROFIT_HARVEST_SMALL_MAX_TRIM_FRACTION),
                        PROFIT_HARVEST_SMALL_MAX_TRIM_FRACTION,
                    )
                remaining = remaining_by_profile.get(profile, 0.0)
                target_pnl = min(net * trim_fraction, remaining)
                if target_pnl <= 0.0:
                    continue
                remaining_by_profile[profile] = max(remaining - target_pnl, 0.0)
                intents.append(
                    {
                        "intent_id": f"paper_harvest::{profile}::{_strategy_bot_id(strategy) or len(intents) + 1}",
                        "profile": profile,
                        "symbol": f"{profile.upper()}_OPEN_WINNER",
                        "strategy": strategy,
                        "bot_id": str(row.get("bot_id") or _strategy_bot_id(strategy)),
                        "action": "SELL",
                        "intent_type": "paper_reduce_only_strategy_proxy_trim",
                        "reduce_only": True,
                        "paper_only": True,
                        "live_execution_allowed": False,
                        "position_proxy": True,
                        "estimated_unrealized_pnl_total": round(net, 6),
                        "estimated_realized_pnl_target_total": round(target_pnl, 6),
                        "recommended_reduce_fraction_norm": round(trim_fraction, 6),
                        "runner_protection_floor_norm": row.get("protect_runner_when_trend_continuation_above_norm", 0.74),
                        "exit_quality_floor_norm": 0.58,
                        "reason": "strategy-level open-winner proxy can satisfy daily sleeve realized-profit target",
                    }
                )
                if len(intents) >= DAILY_SLEEVE_HARVEST_INTENT_LIMIT:
                    break
    return {
        "active": bool(intents),
        "mode": "paper_reduce_only_profit_harvest_intents",
        "paper_only": True,
        "live_execution_allowed": False,
        "reduce_only": True,
        "intent_count": len(intents),
        "remaining_daily_target_by_profile": {
            profile: round(value, 6)
            for profile, value in sorted(remaining_by_profile.items())
        },
        "intents": intents,
        "runtime_rules": [
            "paper executor may emit reduce/SELL fills only; no live orders are authorized",
            "fresh adds stay blocked while the sleeve daily realized-profit goal is unmet",
            "synthetic open-winner proxies require paper position resolution before a fill is counted",
            "after each trim, regret replay must label whether the trim was early, late, or useful",
        ],
    }


def _paper_harvest_infrabot_contract(
    *,
    profit_realization_contract: dict[str, Any],
    daily_goal_contract: dict[str, Any],
    paper_harvest_execution_contract: dict[str, Any],
    profit_harvest_controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_profiles = [
        _normal_profile(row.get("profile"))
        for row in _as_list(daily_goal_contract.get("targets"))
        if isinstance(row, dict) and _normal_profile(row.get("profile"))
    ]
    active = bool(
        profit_harvest_controls
        or bool(daily_goal_contract.get("active", False))
        or bool(paper_harvest_execution_contract.get("active", False))
    )
    assignments: list[dict[str, Any]] = []
    for bot in PAPER_HARVEST_INFRABOTS:
        bot_id = str(bot.get("bot_id") or "")
        assignments.append(
            {
                **bot,
                "active": active,
                "paper_only": True,
                "live_execution_allowed": False,
                "target_profiles": target_profiles,
                "runtime_inputs": [
                    "profit_realization_contract",
                    "daily_sleeve_harvest_goal_contract",
                    "daily_target_adaptation_contract",
                    "paper_harvest_execution_contract",
                    "profit_harvest_position_ledger",
                    "profit_harvest_regret_replay_contract",
                ],
                "output_labels": [
                    f"{bot_id}_status",
                    f"{bot_id}_blocker",
                    f"{bot_id}_next_action",
                ],
            }
        )
    return {
        "active": active,
        "mode": "paper_profit_harvest_infrabot_supervision",
        "paper_only": True,
        "live_execution_allowed": False,
        "assigned_infrabot_count": len(assignments),
        "assigned_infrabots": assignments,
        "target_profiles": target_profiles,
        "target_realized_profit_share_norm": profit_realization_contract.get("target_realized_profit_share_norm", 0.0),
        "current_realized_profit_share_norm": profit_realization_contract.get("realized_profit_share_norm", 0.0),
        "daily_target_count": daily_goal_contract.get("active_target_count", 0),
        "reduce_only_intent_count": paper_harvest_execution_contract.get("intent_count", 0),
        "supervision_rules": [
            "infrabots supervise paper control state only and cannot authorize live orders",
            "every reduce-only intent must reconcile to a paper fill or expire stale",
            "daily targets must include a stop condition and runner protection floor",
            "met daily targets must either raise within caps or expand collection before the next raise",
            "harvest explanations must name sleeve, strategy, target gap, and reason for hold or trim",
        ],
    }


def _apply_paper_harvest_infrabots_to_profile_controls(
    *,
    profit_harvest_controls: dict[str, dict[str, Any]],
    infrabot_contract: dict[str, Any],
) -> None:
    assigned_ids = [
        str(row.get("bot_id") or "")
        for row in _as_list(infrabot_contract.get("assigned_infrabots"))
        if isinstance(row, dict) and str(row.get("bot_id") or "")
    ]
    for control in profit_harvest_controls.values():
        if not isinstance(control, dict):
            continue
        control["paper_harvest_infrabot_supervision"] = {
            "active": bool(infrabot_contract.get("active", False)),
            "assigned_infrabots": assigned_ids,
            "assigned_infrabot_count": len(assigned_ids),
            "paper_only": True,
            "live_execution_allowed": False,
        }


def _apply_daily_target_adaptation_to_profile_controls(
    *,
    profit_harvest_controls: dict[str, dict[str, Any]],
    adaptation_contract: dict[str, Any],
) -> None:
    rows = {
        _normal_profile(row.get("profile")): row
        for row in _as_list(adaptation_contract.get("profiles"))
        if isinstance(row, dict) and _normal_profile(row.get("profile"))
    }
    for profile, control in profit_harvest_controls.items():
        row = rows.get(_normal_profile(profile), {})
        if not isinstance(row, dict):
            continue
        control["daily_target_adaptation"] = row
        collection_labels = _as_list(row.get("collection_labels"))
        existing_labels = _as_list(control.get("required_labels"))
        control["required_labels"] = ordered_unique([*existing_labels, *[str(label) for label in collection_labels if str(label)]])


def _max_grade_push_contract(
    *,
    operational_control_grade: str,
    profit_harvest_report_card: dict[str, Any],
    daily_goal_contract: dict[str, Any],
    paper_harvest_execution_contract: dict[str, Any],
    infrabot_contract: dict[str, Any],
) -> dict[str, Any]:
    raw_harvest_grade = str(profit_harvest_report_card.get("base_raw_outcome_grade") or "")
    headline_harvest_grade = str(profit_harvest_report_card.get("headline_grade") or "")
    raw_ready = raw_harvest_grade == "A++"
    control_ready = (
        operational_control_grade == "A+"
        and headline_harvest_grade in {"A++", "A+"}
        and bool(daily_goal_contract.get("active", False))
        and bool(paper_harvest_execution_contract.get("active", False))
        and bool(infrabot_contract.get("active", False))
    )
    return {
        "active": True,
        "mode": "paper_profitability_max_grade_push",
        "paper_only": True,
        "live_execution_allowed": False,
        "control_surface_max_ready": control_ready,
        "raw_outcome_max_ready": raw_ready,
        "target_control_grade": "A+",
        "target_harvest_headline_grade": "A++",
        "target_raw_harvest_grade": "A++",
        "current": {
            "operational_control_grade": operational_control_grade,
            "harvest_headline_grade": headline_harvest_grade,
            "raw_harvest_grade": raw_harvest_grade,
            "daily_target_active": bool(daily_goal_contract.get("active", False)),
            "reduce_only_intent_count": paper_harvest_execution_contract.get("intent_count", 0),
            "harvest_infrabot_count": infrabot_contract.get("assigned_infrabot_count", 0),
        },
        "remaining_raw_blockers": [
            "realized paper profit share must rise from actual paper fills",
            "unrealized concentration must fall without breaking runner protection",
            "harvest regret replay must confirm trims were useful, not premature",
        ]
        if not raw_ready
        else [],
        "stop_condition": "control_surface_max_ready=true and raw_outcome_max_ready=true",
    }


def _profile_net(row: dict[str, Any]) -> float:
    return _safe_float(row.get("ending_net_pnl_total"), 0.0)


def _profile_realized(row: dict[str, Any]) -> float:
    return _safe_float(row.get("ending_realized_pnl_total"), 0.0)


def _profile_unrealized(row: dict[str, Any]) -> float:
    return _safe_float(row.get("ending_unrealized_pnl_total"), 0.0)


def _profitability_realization_expansion_contract(
    *,
    sleeves: list[dict[str, Any]],
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
    profit_harvest_controls: dict[str, dict[str, Any]],
    profit_harvest_strategy_controls: dict[str, dict[str, Any]],
    profit_harvest_report_card: dict[str, Any],
    daily_goal_contract: dict[str, Any],
    paper_harvest_execution_contract: dict[str, Any],
    profit_realization_contract: dict[str, Any],
    cause_counter: Counter[str],
) -> dict[str, Any]:
    sleeve_rows = [row for row in sleeves if isinstance(row, dict)]
    weak_profiles = ordered_unique(
        [
            _normal_profile(profile)
            for profile, control in active_profile_controls.items()
            if isinstance(control, dict)
            and (
                str(control.get("action") or "").strip().lower() == "quarantine_new_entries"
                or bool(_as_dict(control.get("loser_quarantine")).get("active", False))
            )
        ]
    )
    winning_sleeves = sorted(
        [
            {
                "profile": _normal_profile(row.get("profile")),
                "net_pnl_total": round(_profile_net(row), 6),
                "realized_pnl_total": round(_profile_realized(row), 6),
                "unrealized_pnl_total": round(_profile_unrealized(row), 6),
                "executions": _safe_int(row.get("executions"), 0),
                "win_rate": row.get("win_rate"),
            }
            for row in sleeve_rows
            if _normal_profile(row.get("profile")) and _profile_net(row) > 0.0 and _profile_realized(row) >= 0.0
        ],
        key=lambda item: (_safe_float(item.get("net_pnl_total"), 0.0), _safe_float(item.get("realized_pnl_total"), 0.0)),
        reverse=True,
    )
    unrealized_losers = sorted(
        [
            {
                "profile": _normal_profile(row.get("profile")),
                "net_pnl_total": round(_profile_net(row), 6),
                "unrealized_pnl_total": round(_profile_unrealized(row), 6),
                "loss_causes": _cause_names(row)[:8],
            }
            for row in sleeve_rows
            if _normal_profile(row.get("profile")) and _profile_unrealized(row) < 0.0
        ],
        key=lambda item: _safe_float(item.get("unrealized_pnl_total"), 0.0),
    )
    winning_strategy_rows: list[dict[str, Any]] = []
    for row in sleeve_rows:
        profile = _normal_profile(row.get("profile"))
        for strategy in _as_list(row.get("top_winning_strategies")):
            if not isinstance(strategy, dict):
                continue
            net = _safe_float(strategy.get("ending_net_pnl_total"), 0.0)
            if net <= 0.0:
                continue
            strategy_name = str(strategy.get("strategy") or "").strip()
            if not strategy_name:
                continue
            winning_strategy_rows.append(
                {
                    "profile": profile,
                    "strategy": strategy_name,
                    "bot_id": _strategy_bot_id(strategy_name),
                    "net_pnl_total": round(net, 6),
                    "promotion_bias": "scale_candidate" if profile not in weak_profiles else "contained_until_profile_recovers",
                }
            )
    winning_strategy_rows.sort(key=lambda item: _safe_float(item.get("net_pnl_total"), 0.0), reverse=True)
    losing_strategy_rows = sorted(
        [
            {
                "profile": str(row.get("profile") or ""),
                "strategy": str(row.get("strategy") or ""),
                "bot_id": str(row.get("bot_id") or ""),
                "mode": str(row.get("mode") or ""),
                "net_pnl_total": round(_safe_float(row.get("ending_net_pnl_total"), 0.0), 6),
                "new_entry_cap": _safe_int(row.get("new_entry_cap"), 0),
            }
            for row in strategy_controls
            if isinstance(row, dict)
        ],
        key=lambda item: _safe_float(item.get("net_pnl_total"), 0.0),
    )
    raw_score = _safe_float(profit_harvest_report_card.get("base_raw_outcome_score_norm"), 0.0)
    harvest_score = _safe_float(profit_harvest_report_card.get("raw_outcome_score_norm"), raw_score)
    regret_control = _safe_float(profit_harvest_report_card.get("regret_control_norm"), 0.0)
    a_plus_gap = max(PROFIT_HARVEST_APLUS_MIN_SCORE - harvest_score, 0.0)
    a_plus_plus_gap = max(PROFIT_HARVEST_APLUSPLUS_MIN_SCORE - harvest_score, 0.0)
    regret_gap = max(PROFIT_HARVEST_APLUSPLUS_MIN_REGRET_CONTROL - regret_control, 0.0)
    realized_share = _safe_float(profit_realization_contract.get("realized_profit_share_norm"), 0.0)
    target_realized_share = _safe_float(
        profit_realization_contract.get("target_realized_profit_share_norm"),
        PROFIT_HARVEST_TARGET_REALIZED_SHARE,
    )
    unrealized_share = _safe_float(profit_realization_contract.get("unrealized_profit_share_norm"), 0.0)
    max_unrealized_share = _safe_float(
        profit_realization_contract.get("max_unrealized_profit_share_norm"),
        PROFIT_HARVEST_MAX_UNREALIZED_SHARE,
    )
    daily_targets = [
        row for row in _as_list(daily_goal_contract.get("targets")) if isinstance(row, dict)
    ]
    active_daily_targets = [row for row in daily_targets if bool(row.get("active", False))]
    lever_rows = [
        {
            "lever_id": "stop_weak_sleeve_drag",
            "active": bool(weak_profiles),
            "priority": 1,
            "targets": weak_profiles,
            "control": "no-new-entry/probation for weak sleeves; reductions and data collection stay allowed",
            "expected_impact": "stop weak sleeves from creating fresh paper drag while their evidence repairs",
            "when_to_stop": "weak sleeve leaves active_profile_controls or shows positive fresh paper net with no losing-strategy cluster",
        },
        {
            "lever_id": "scale_winning_sleeves",
            "active": bool(winning_sleeves),
            "priority": 2,
            "targets": winning_sleeves[:10],
            "control": "favor sleeves with positive net and nonnegative realized PnL for paper allocation, training weight, and promotion review",
            "expected_impact": "move attention toward repeatable winners instead of equal-weighting noisy sleeves",
            "when_to_stop": "top sleeves remain positive across refreshes and weak sleeves are contained",
        },
        {
            "lever_id": "harvest_regret_control_lift",
            "active": bool(regret_gap > 0.0),
            "priority": 3,
            "targets": {
                "current_regret_control_norm": round(regret_control, 6),
                "target_regret_control_norm": PROFIT_HARVEST_APLUSPLUS_MIN_REGRET_CONTROL,
                "regret_gap_norm": round(regret_gap, 6),
            },
            "control": "label every trim as early, late, useful, or runner-protected and feed the replay labels into training",
            "expected_impact": "improve realized conversion without repeatedly trimming strong runners too early",
            "when_to_stop": "regret_control_norm >= 0.80 and raw_harvest_score_norm >= 0.98",
        },
        {
            "lever_id": "laddered_partial_exit_policy",
            "active": bool(active_daily_targets or paper_harvest_execution_contract.get("active", False)),
            "priority": 4,
            "targets": [
                {
                    "profile": row.get("profile"),
                    "daily_target_total": row.get("daily_harvest_pnl_target_total", 0.0),
                    "ladder_steps": len(_as_list(row.get("laddered_exit_plan"))),
                }
                for row in active_daily_targets[:10]
            ],
            "control": "use lock-seed, pay-system, and protect-runner partial exits instead of all-or-nothing harvesting",
            "expected_impact": "raise realized profit share while preserving continuation optionality",
            "when_to_stop": "daily realized-profit goal met, realized share target met, or runner protection blocks the trim",
        },
        {
            "lever_id": "strategy_level_promotion",
            "active": bool(winning_strategy_rows or profit_harvest_strategy_controls),
            "priority": 5,
            "targets": winning_strategy_rows[:16],
            "control": "promote bot-strategy pairs that are carrying sleeves; keep sleeve-level winners from hiding internal losers",
            "expected_impact": "allocate more learning and paper sizing to actual winning bot-strategy pairs",
            "when_to_stop": "promoted strategy remains positive after fresh replay and sleeve-level attribution stays clean",
        },
        {
            "lever_id": "punitive_loss_attribution",
            "active": bool(losing_strategy_rows or cause_counter),
            "priority": 6,
            "targets": {
                "losing_strategy_pairs": losing_strategy_rows[:16],
                "top_loss_causes": [{"cause": cause, "count": count} for cause, count in cause_counter.most_common(8)],
            },
            "control": "loss causes directly shrink size, add hard-negative training weight, and block low-quality confirmation repeats",
            "expected_impact": "make repeated bad evidence expensive for future bot votes",
            "when_to_stop": "loss-cause clusters stop appearing in the top paper loss causes for two clean refreshes",
        },
        {
            "lever_id": "unrealized_loser_training_debt",
            "active": bool(unrealized_losers),
            "priority": 7,
            "targets": unrealized_losers[:12],
            "control": "turn negative unrealized PnL into repair labels for missed exits, bad entries, and no-trade counterfactuals",
            "expected_impact": "teach the system what it missed before paper losses became open drag",
            "when_to_stop": "unrealized_loser_count is zero or each loser has repair labels and reduced future sizing",
        },
        {
            "lever_id": "harvest_force_guard",
            "active": True,
            "priority": 8,
            "targets": {
                "realized_share_norm": round(realized_share, 6),
                "target_realized_share_norm": round(target_realized_share, 6),
                "unrealized_share_norm": round(unrealized_share, 6),
                "max_unrealized_share_norm": round(max_unrealized_share, 6),
                "force_harvest_allowed": bool(
                    realized_share < target_realized_share
                    and unrealized_share > max_unrealized_share
                    and _safe_float(profit_realization_contract.get("portfolio_net_pnl_total"), 0.0) > 0.0
                ),
            },
            "control": "do not force harvesting when net quality is weak; prefer containment, labels, and reduce-only paper intents",
            "expected_impact": "avoid manufacturing realized profit by cutting the wrong positions or selling into poor evidence",
            "when_to_stop": "realized share is above target or runner protection says hold",
        },
    ]
    return {
        "active": True,
        "mode": "profitability_realization_expansion_1_to_8",
        "paper_only": True,
        "live_execution_allowed": False,
        "profitability_phase": "profit_engineering",
        "lever_count": len(PROFITABILITY_REALIZATION_LEVERS),
        "lever_ids": PROFITABILITY_REALIZATION_LEVERS,
        "active_lever_count": sum(1 for row in lever_rows if bool(row.get("active", False))),
        "current": {
            "raw_harvest_score_norm": round(harvest_score, 6),
            "base_raw_harvest_score_norm": round(raw_score, 6),
            "a_plus_gap_norm": round(a_plus_gap, 6),
            "a_plus_plus_gap_norm": round(a_plus_plus_gap, 6),
            "regret_control_norm": round(regret_control, 6),
            "regret_gap_to_a_plus_plus_norm": round(regret_gap, 6),
            "winning_sleeve_count": len(winning_sleeves),
            "weak_profile_count": len(weak_profiles),
            "losing_strategy_pair_count": len(losing_strategy_rows),
            "unrealized_loser_count": len(unrealized_losers),
            "paper_harvest_intent_count": _safe_int(paper_harvest_execution_contract.get("intent_count"), 0),
        },
        "levers": lever_rows,
        "global_rules": [
            "paper-only: this contract cannot authorize live orders",
            "weak sleeves cannot add fresh risk until their paper evidence repairs",
            "winning sleeves and winning strategies receive more weight only after attribution is visible",
            "harvest aggression rises from regret replay, not from grade pressure alone",
            "unrealized losers become training debt before any widening decision",
        ],
        "recommended_command": ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
        "stop_condition": "all eight levers are active or intentionally idle, raw harvest score >= 0.98, regret control >= 0.80, and weak_profile_count == 0",
    }


def _expansion_lever(expansion_contract: dict[str, Any], lever_id: str) -> dict[str, Any]:
    for row in _as_list(expansion_contract.get("levers")):
        if isinstance(row, dict) and str(row.get("lever_id") or "") == lever_id:
            return row
    return {}


def _profitability_compounding_autopilot_contract(
    *,
    expansion_contract: dict[str, Any],
    profit_harvest_report_card: dict[str, Any],
    profit_realization_contract: dict[str, Any],
    daily_goal_contract: dict[str, Any],
    paper_harvest_execution_contract: dict[str, Any],
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
) -> dict[str, Any]:
    current = _as_dict(expansion_contract.get("current"))
    weak_lever = _expansion_lever(expansion_contract, "stop_weak_sleeve_drag")
    scale_lever = _expansion_lever(expansion_contract, "scale_winning_sleeves")
    regret_lever = _expansion_lever(expansion_contract, "harvest_regret_control_lift")
    ladder_lever = _expansion_lever(expansion_contract, "laddered_partial_exit_policy")
    strategy_lever = _expansion_lever(expansion_contract, "strategy_level_promotion")
    punitive_lever = _expansion_lever(expansion_contract, "punitive_loss_attribution")
    debt_lever = _expansion_lever(expansion_contract, "unrealized_loser_training_debt")
    guard_lever = _expansion_lever(expansion_contract, "harvest_force_guard")

    weak_profiles = [str(item) for item in _as_list(weak_lever.get("targets")) if str(item)]
    winning_sleeves = [row for row in _as_list(scale_lever.get("targets")) if isinstance(row, dict)]
    winning_strategies = [row for row in _as_list(strategy_lever.get("targets")) if isinstance(row, dict)]
    unrealized_losers = [row for row in _as_list(debt_lever.get("targets")) if isinstance(row, dict)]
    punitive_targets = _as_dict(punitive_lever.get("targets"))
    losing_strategy_pairs = [
        row for row in _as_list(punitive_targets.get("losing_strategy_pairs")) if isinstance(row, dict)
    ]
    regret_targets = _as_dict(regret_lever.get("targets"))
    guard_targets = _as_dict(guard_lever.get("targets"))
    active_daily_targets = [
        row for row in _as_list(daily_goal_contract.get("targets")) if isinstance(row, dict) and bool(row.get("active", False))
    ]
    intent_count = _safe_int(paper_harvest_execution_contract.get("intent_count"), 0)
    regret_gap = _safe_float(current.get("regret_gap_to_a_plus_plus_norm"), _safe_float(regret_targets.get("regret_gap_norm"), 0.0))
    a_plus_gap = _safe_float(current.get("a_plus_gap_norm"), 0.0)
    a_plus_plus_gap = _safe_float(current.get("a_plus_plus_gap_norm"), 0.0)
    raw_harvest_score = _safe_float(current.get("raw_harvest_score_norm"), _safe_float(profit_harvest_report_card.get("raw_outcome_score_norm"), 0.0))
    net_pnl = _safe_float(profit_realization_contract.get("portfolio_net_pnl_total"), 0.0)
    realized_share = _safe_float(profit_realization_contract.get("realized_profit_share_norm"), 0.0)
    target_realized_share = _safe_float(profit_realization_contract.get("target_realized_profit_share_norm"), PROFIT_HARVEST_TARGET_REALIZED_SHARE)
    force_guard_blocks = not bool(guard_targets.get("force_harvest_allowed", False))

    rows: list[dict[str, Any]] = []

    def add(
        action_id: str,
        *,
        active: bool,
        priority_score: float,
        targets: Any,
        expected_impact: str,
        risk_level: str,
        stop_condition: str,
        command: list[str] | None = None,
    ) -> None:
        rows.append(
            {
                "action_id": action_id,
                "active": bool(active),
                "priority_score": round(max(float(priority_score), 0.0), 6),
                "targets": targets,
                "expected_impact": expected_impact,
                "risk_level": risk_level,
                "paper_only": True,
                "live_execution_allowed": False,
                "exact_command": command or ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
                "stop_condition": stop_condition,
            }
        )

    add(
        "freeze_weak_sleeve_fresh_adds",
        active=bool(weak_profiles),
        priority_score=100.0 + (8.0 * len(weak_profiles)),
        targets=[
            {
                "profile": profile,
                "action": _as_dict(active_profile_controls.get(profile)).get("action", "quarantine_new_entries"),
                "new_entry_cap": _as_dict(active_profile_controls.get(profile)).get("new_entry_cap", 0),
            }
            for profile in weak_profiles
        ],
        expected_impact="stop fresh paper drag before attempting to scale or harvest harder",
        risk_level="low",
        stop_condition="weak_profile_count == 0 or every weak profile is no-new-entry with reduced sizing",
    )
    add(
        "reconcile_reduce_only_harvest_intents",
        active=bool(intent_count > 0),
        priority_score=92.0 + min(intent_count, 32),
        targets={
            "intent_count": intent_count,
            "active_daily_target_count": len(active_daily_targets),
            "remaining_daily_target_by_profile": paper_harvest_execution_contract.get("remaining_daily_target_by_profile", {}),
        },
        expected_impact="convert eligible paper winners into realized paper profit while keeping live locked",
        risk_level="medium",
        stop_condition="paper_harvest_intent_count == 0 or realized_share_norm >= target_realized_share_norm",
    )
    add(
        "run_harvest_regret_replay",
        active=bool(regret_gap > 0.0),
        priority_score=88.0 + (100.0 * regret_gap),
        targets={
            "current_regret_control_norm": regret_targets.get("current_regret_control_norm", current.get("regret_control_norm", 0.0)),
            "target_regret_control_norm": PROFIT_HARVEST_APLUSPLUS_MIN_REGRET_CONTROL,
            "regret_gap_norm": round(regret_gap, 6),
            "lookahead_minutes": PROFIT_HARVEST_REPLAY_LOOKAHEAD_MINUTES,
        },
        expected_impact="teach the system whether trims were early, late, useful, or runner-protected",
        risk_level="low",
        stop_condition="regret_control_norm >= 0.80",
    )
    add(
        "promote_winning_strategy_pairs",
        active=bool(winning_strategies),
        priority_score=76.0 + min(len(winning_strategies), 16),
        targets=winning_strategies[:12],
        expected_impact="route training and paper sizing toward the bot-strategy pairs carrying profitable sleeves",
        risk_level="low",
        stop_condition="promoted strategy pair remains positive after fresh replay and source/fill evidence is present",
    )
    add(
        "assign_unrealized_loser_training_debt",
        active=bool(unrealized_losers),
        priority_score=72.0 + (6.0 * len(unrealized_losers)),
        targets=unrealized_losers[:12],
        expected_impact="turn open paper losers into labels for bad entries, missed exits, and no-trade counterfactuals",
        risk_level="low",
        stop_condition="unrealized_loser_count == 0 or each loser has repair labels and reduced future sizing",
    )
    add(
        "tighten_punitive_loss_attribution",
        active=bool(losing_strategy_pairs),
        priority_score=68.0 + min(len(losing_strategy_pairs), 24),
        targets=losing_strategy_pairs[:16],
        expected_impact="make repeated bad evidence shrink future size and raise confirmation requirements",
        risk_level="low",
        stop_condition="losing_strategy_pair_count == 0 or loss-cause clusters clear for two refreshes",
    )
    add(
        "scale_clean_winning_sleeves",
        active=bool(winning_sleeves and not weak_profiles),
        priority_score=56.0 + min(len(winning_sleeves), 10),
        targets=[
            {
                **row,
                "paper_scale_multiplier_norm": round(_clamp(1.0 + (_safe_float(row.get("net_pnl_total"), 0.0) / 2_000.0), 1.0, 1.35), 6),
            }
            for row in winning_sleeves[:10]
        ],
        expected_impact="increase paper attention only after weak sleeve drag is contained",
        risk_level="medium",
        stop_condition="weak_profile_count == 0 and top sleeves stay positive across refreshes",
    )
    add(
        "hold_profit_as_paper_cash_when_force_guard_blocks",
        active=bool(force_guard_blocks and realized_share >= target_realized_share),
        priority_score=54.0 + (25.0 if net_pnl <= 0.0 else 0.0),
        targets={
            "portfolio_net_pnl_total": round(net_pnl, 6),
            "realized_share_norm": round(realized_share, 6),
            "target_realized_share_norm": round(target_realized_share, 6),
            "force_harvest_allowed": bool(guard_targets.get("force_harvest_allowed", False)),
        },
        expected_impact="avoid forcing harvests when realized conversion is already high or net quality is weak",
        risk_level="low",
        stop_condition="net paper quality improves or force_harvest_allowed becomes true with runner protection clear",
    )

    rows.sort(key=lambda item: (_safe_float(item.get("priority_score"), 0.0), str(item.get("action_id") or "")), reverse=True)
    active_rows = [row for row in rows if bool(row.get("active", False))]
    return {
        "active": True,
        "mode": "profitability_compounding_autopilot_v1",
        "paper_only": True,
        "live_execution_allowed": False,
        "action_count": len(PROFITABILITY_COMPOUNDING_AUTOPILOT_ACTIONS),
        "action_ids": PROFITABILITY_COMPOUNDING_AUTOPILOT_ACTIONS,
        "active_action_count": len(active_rows),
        "priority_queue": rows,
        "do_first": active_rows[:5],
        "current": {
            "raw_harvest_score_norm": round(raw_harvest_score, 6),
            "a_plus_gap_norm": round(a_plus_gap, 6),
            "a_plus_plus_gap_norm": round(a_plus_plus_gap, 6),
            "regret_gap_norm": round(regret_gap, 6),
            "portfolio_net_pnl_total": round(net_pnl, 6),
            "realized_share_norm": round(realized_share, 6),
            "target_realized_share_norm": round(target_realized_share, 6),
            "weak_profile_count": len(weak_profiles),
            "winning_sleeve_count": len(winning_sleeves),
            "losing_strategy_pair_count": len(losing_strategy_pairs),
            "unrealized_loser_count": len(unrealized_losers),
        },
        "global_rules": [
            "do weak-sleeve containment before scale-up",
            "do reduce-only paper harvest reconciliation before raising targets",
            "do regret replay before increasing trim aggression",
            "do strategy-level promotion only when attribution is visible",
            "do not force harvests just to improve a grade when net quality is weak",
        ],
        "recommended_command": ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
        "stop_condition": "do_first is empty, raw_harvest_score_norm >= 0.98, regret_gap_norm == 0, and weak_profile_count == 0",
    }


def _quant_strategy_expansion_admission_contract(
    *,
    sleeves: list[dict[str, Any]],
    expansion_contract: dict[str, Any],
    compounding_autopilot_contract: dict[str, Any],
    profit_harvest_report_card: dict[str, Any],
    profit_realization_contract: dict[str, Any],
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
    overall_status: str,
) -> dict[str, Any]:
    weak_lever = _expansion_lever(expansion_contract, "stop_weak_sleeve_drag")
    scale_lever = _expansion_lever(expansion_contract, "scale_winning_sleeves")
    punitive_lever = _expansion_lever(expansion_contract, "punitive_loss_attribution")
    weak_profiles = ordered_unique(
        [
            _normal_profile(profile)
            for profile in _as_list(weak_lever.get("targets"))
            if _normal_profile(profile)
        ]
        + [
            _normal_profile(profile)
            for profile, control in active_profile_controls.items()
            if _normal_profile(profile)
            and (
                str(_as_dict(control).get("action") or "").strip().lower() == "quarantine_new_entries"
                or bool(_as_dict(_as_dict(control).get("loser_quarantine")).get("active", False))
            )
        ]
    )
    winning_sleeves = ordered_unique(
        [
            _normal_profile(row.get("profile"))
            for row in _as_list(scale_lever.get("targets"))
            if isinstance(row, dict) and _normal_profile(row.get("profile"))
        ]
        + [
            _normal_profile(row.get("profile"))
            for row in sleeves
            if isinstance(row, dict)
            and _normal_profile(row.get("profile"))
            and _profile_net(row) > 0.0
            and _profile_realized(row) >= 0.0
        ]
    )
    fallback_sleeves = [
        "default",
        "bond",
        "dividend",
        "conservative",
        "crypto_futures",
        "stat_arb_market_neutral",
        "volatility",
        "earnings_event",
    ]
    target_sleeves = ordered_unique(
        [profile for profile in winning_sleeves if profile not in weak_profiles]
        + [profile for profile in fallback_sleeves if profile not in weak_profiles]
    )[:10]
    punitive_targets = _as_dict(punitive_lever.get("targets"))
    losing_strategy_pairs = [
        row for row in _as_list(punitive_targets.get("losing_strategy_pairs")) if isinstance(row, dict)
    ]
    losing_bot_ids = ordered_unique(
        [
            str(row.get("bot_id") or _strategy_bot_id(str(row.get("strategy") or ""))).strip()
            for row in losing_strategy_pairs
            if str(row.get("bot_id") or row.get("strategy") or "").strip()
        ]
        + [
            str(row.get("bot_id") or _strategy_bot_id(str(row.get("strategy") or ""))).strip()
            for row in strategy_controls
            if isinstance(row, dict) and _safe_float(row.get("ending_net_pnl_total"), 0.0) < 0.0
        ]
    )
    raw_harvest_score = _safe_float(
        profit_harvest_report_card.get("raw_outcome_score_norm"),
        _safe_float(profit_harvest_report_card.get("base_raw_outcome_score_norm"), 0.0),
    )
    regret_control = _safe_float(profit_harvest_report_card.get("regret_control_norm"), 0.0)
    net_pnl = _safe_float(profit_realization_contract.get("portfolio_net_pnl_total"), 0.0)
    realized_share = _safe_float(profit_realization_contract.get("realized_profit_share_norm"), 0.0)
    active_autopilot_actions = _safe_int(compounding_autopilot_contract.get("active_action_count"), 0)
    protective_mode = bool(
        str(overall_status) == "protective_tightening"
        or weak_profiles
        or net_pnl < 0.0
        or raw_harvest_score < PROFIT_HARVEST_APLUS_MIN_SCORE
        or active_autopilot_actions >= 4
    )
    if not target_sleeves:
        max_new_slots = 0
    elif protective_mode:
        max_new_slots = min(4, len(QUANT_STRATEGY_EXPANSION_FAMILIES))
    elif raw_harvest_score >= PROFIT_HARVEST_APLUSPLUS_MIN_SCORE and regret_control >= PROFIT_HARVEST_APLUSPLUS_MIN_REGRET_CONTROL:
        max_new_slots = min(10, len(QUANT_STRATEGY_EXPANSION_FAMILIES))
    else:
        max_new_slots = min(6, len(QUANT_STRATEGY_EXPANSION_FAMILIES))

    if max_new_slots <= 0:
        admission_state = "paused_for_repair"
    elif protective_mode:
        admission_state = "collection_only_selective"
    else:
        admission_state = "paper_canary_ready"

    templates: list[dict[str, Any]] = []
    for idx, family in enumerate(QUANT_STRATEGY_EXPANSION_FAMILIES, start=1):
        preferred = [
            _normal_profile(profile)
            for profile in _as_list(family.get("preferred_sleeves"))
            if _normal_profile(profile) and _normal_profile(profile) not in weak_profiles
        ]
        sleeve = next((profile for profile in preferred if profile in target_sleeves), target_sleeves[0] if target_sleeves else "")
        if not sleeve:
            continue
        family_id = str(family.get("family_id") or f"quant_family_{idx}").strip()
        priority_score = 88.0 - float(idx)
        if sleeve in winning_sleeves:
            priority_score += 12.0
        if protective_mode:
            priority_score -= 8.0
        if sleeve in weak_profiles:
            priority_score -= 40.0
        templates.append(
            {
                "candidate_id": f"quant_strategy::{sleeve}::{family_id}",
                "family_id": family_id,
                "target_sleeve": sleeve,
                "purpose": str(family.get("purpose") or ""),
                "initial_state": "collection_only" if protective_mode else "paper_canary",
                "paper_only": True,
                "live_execution_allowed": False,
                "max_initial_paper_size_norm": 0.0 if protective_mode else 0.05,
                "admission_priority_score": round(max(priority_score, 0.0), 6),
                "required_label_outputs": ordered_unique(
                    [
                        *[str(label) for label in _as_list(family.get("required_labels")) if str(label)],
                        "duplicate_alpha_overlap_norm",
                        "paper_fill_quality_bucket",
                        "realized_conversion_after_entry",
                        "no_trade_counterfactual_outcome",
                    ]
                ),
                "evidence_layer": {
                    "active": True,
                    "mode": "quant_exotic_admission_evidence_v2",
                    "minimum_collection_days": 3,
                    "minimum_clean_samples": 200,
                    "required_evidence_surfaces": [
                        "source_verification_current",
                        "spread_and_fill_quality",
                        "duplicate_alpha_overlap",
                        "paper_pnl_attribution",
                        "harvest_regret_replay",
                        "no_trade_counterfactual",
                    ],
                    "blockers": [
                        "stale_source_verification",
                        "duplicate_alpha_overlap_high",
                        "missing_fill_quality",
                        "negative_paper_attribution",
                        "missing_harvest_replay",
                    ],
                    "collection_probe_command": [
                        "./scripts/ops/opsctl.sh",
                        "paper-profitability-control",
                        "--apply",
                        "--json",
                    ],
                },
                "promotion_gate": "must pass collection quality, duplicate-alpha, fill-quality, paper PnL, and harvest-regret checks before any paper widening",
            }
        )
    templates.sort(key=lambda row: _safe_float(row.get("admission_priority_score"), 0.0), reverse=True)

    return {
        "active": True,
        "mode": "quant_strategy_expansion_admission_v1",
        "paper_only": True,
        "live_execution_allowed": False,
        "can_add_more_quant_strategies": bool(max_new_slots > 0),
        "admission_state": admission_state,
        "max_new_strategy_slots": int(max_new_slots),
        "approved_family_count": len(QUANT_STRATEGY_EXPANSION_FAMILIES),
        "approved_families": [str(row.get("family_id") or "") for row in QUANT_STRATEGY_EXPANSION_FAMILIES],
        "target_sleeves": target_sleeves,
        "blocked_profiles": weak_profiles,
        "blocked_strategy_bot_ids": losing_bot_ids[:20],
        "candidate_templates": templates[:max_new_slots],
        "current": {
            "overall_status": str(overall_status),
            "portfolio_net_pnl_total": round(net_pnl, 6),
            "raw_harvest_score_norm": round(raw_harvest_score, 6),
            "regret_control_norm": round(regret_control, 6),
            "realized_share_norm": round(realized_share, 6),
            "weak_profile_count": len(weak_profiles),
            "active_profitability_autopilot_action_count": active_autopilot_actions,
            "losing_strategy_pair_count": len(losing_strategy_pairs),
        },
        "admission_rules": [
            "new quant strategies start collection-only while protective tightening is active",
            "live execution remains blocked; this contract cannot authorize live trading",
            "weak sleeves cannot receive new strategy size until the sleeve exits quarantine",
            "candidate must add required labels before promotion review",
            "candidate must satisfy the quant_exotic_admission_evidence_v2 surfaces before any paper sizing",
            "duplicate alpha overlap must be below the local overlap gate before widening",
            "source, spread, fill, and conflict evidence must be current before paper canary sizing",
            "paper canary size starts small and only scales after harvest-regret replay improves",
        ],
        "recommended_command": ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
        "stop_condition": "promote only after candidate templates produce clean labels, no duplicate-alpha debt, positive paper attribution, and no weak-sleeve containment",
    }


def _blocked_regimes(families: list[str], drag: float) -> list[str]:
    blocked: list[str] = []
    if "source_quality" in families:
        blocked.append("low_source_quality")
    if "tradeability" in families:
        blocked.append("low_tradeability")
    if "fill_quality" in families:
        blocked.append("unknown_or_poor_fill_quality")
    if "spread_quality" in families:
        blocked.append("wide_or_unknown_spread")
    if "catalyst_confirmation" in families:
        blocked.append("weak_catalyst_confirmation")
    if "portfolio_conflict" in families:
        blocked.append("high_portfolio_conflict")
    if drag >= 0.64:
        blocked.append("loss_cluster_active")
    return ordered_unique(blocked)


def _profile_upgrade_contracts(
    *,
    profile: str,
    row: dict[str, Any],
    action: str,
    drag: float,
    profit_score: float,
    position_multiplier: float,
    families: list[str],
    cause_names: list[str],
    confirmation_bias_score: float,
    thresholds: dict[str, Any],
    losing_strategy_count: int,
) -> dict[str, dict[str, Any]]:
    net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
    unrealized = _safe_float(row.get("ending_unrealized_pnl_total"), 0.0)
    executions = _safe_int(row.get("executions"), 0)
    blocked = _blocked_regimes(families, drag)
    unrealized_drag = _clamp(abs(min(unrealized, 0.0)) / max(500.0, float(max(executions, 1)) * 2.0))
    severe_drag = bool(drag >= 0.64 or unrealized_drag >= 0.42)
    drag_active = bool(unrealized < 0.0 or unrealized_drag >= 0.18 or drag >= 0.38)
    conflict_cap = 0.72
    if "portfolio_conflict" in families or drag >= 0.64:
        conflict_cap = 0.58
    elif drag >= 0.38:
        conflict_cap = 0.64

    return {
        "outcome_weighted_training": {
            "active": bool(net < 0.0 or drag >= 0.20),
            "sample_weight_multiplier": round(1.0 + (1.85 * drag), 6),
            "paper_loss_negative_weight": round(1.0 + (2.25 * drag), 6),
            "paper_profit_positive_weight": round(max(0.70, 1.0 - (0.35 * drag)), 6),
            "focus_loss_families": families,
            "feedback_columns": [
                "paper_profile",
                "paper_strategy",
                "paper_loss_cause",
                "paper_fill_quality",
                "paper_exit_drag",
                "entry_evidence_gate_result",
                "no_trade_counterfactual_outcome",
            ],
        },
        "per_sleeve_profit_score": {
            "active": True,
            "profit_score_norm": round(profit_score, 6),
            "profit_grade": _profit_grade(profit_score),
            "drag_score_norm": round(drag, 6),
            "profitable_retest_required": bool(net < 0.0 or profit_score < 0.62),
        },
        "dynamic_sizing": {
            "active": True,
            "paper_profitability_size_multiplier_norm": round(position_multiplier, 6),
            "max_new_entry_multiplier_norm": round(min(0.80, max(0.05, position_multiplier)), 6),
            "kelly_fraction_cap_norm": round(max(0.02, min(0.35, 0.35 * (1.0 - drag))), 6),
            "restore_step_norm": 0.10,
            "block_new_entries_when_drag_active": drag_active,
            "new_entry_size_floor_norm": 0.0 if action == "quarantine_new_entries" else 0.05,
        },
        "regime_specific_promotion": {
            "active": bool(blocked or profit_score < 0.62),
            "promotion_status": "paper_only_retest" if net < 0.0 or drag >= 0.38 else "observe",
            "blocked_regimes": blocked,
            "allowed_regimes": [
                "fresh_source_verified",
                "modeled_execution",
                "positive_next_refresh",
            ],
            "min_profitable_refreshes_to_lift": 2 if drag < 0.64 else 3,
        },
        "loser_quarantine": {
            "active": action == "quarantine_new_entries" or losing_strategy_count > 0,
            "mode": action,
            "losing_strategy_count": losing_strategy_count,
            "new_entry_cap": 0 if action == "quarantine_new_entries" else (1 if drag >= 0.64 else 3),
            "block_new_entries": action == "quarantine_new_entries",
            "allow_reductions_only_when_drag_active": drag_active,
            "reentry_requires_positive_refreshes": 2 if drag < 0.64 else 3,
            "lift_condition": "positive paper refresh plus no active losing-strategy cluster",
        },
        "exit_intelligence": {
            "active": drag_active,
            "unrealized_drag_norm": round(unrealized_drag, 6),
            "tighten_exit_bias_norm": round(_clamp((0.55 * drag) + (0.45 * unrealized_drag)), 6),
            "max_stale_hold_minutes": 10 if severe_drag else (15 if drag >= 0.64 else 30),
            "prefer_reduce_over_add": bool(unrealized < 0.0),
            "drag_reduction_mode": "reduce_only" if drag_active else "normal",
            "reduce_on_next_valid_tick": bool(unrealized_drag >= 0.18 or severe_drag),
            "block_adds_while_unrealized_negative": bool(unrealized < 0.0),
            "max_adds_while_drag_active": 0,
            "exit_review_required": drag_active,
            "paper_exit_drag_label_required": True,
            "stop_loss_review_threshold_norm": round(0.52 if severe_drag else 0.62, 6),
        },
        "execution_aware_alpha": {
            "active": bool({"fill_quality", "spread_quality", "tradeability"} & set(families) or drag >= 0.38),
            "thresholds": thresholds,
            "unknown_fill_score_discount_norm": round(0.18 + (0.32 * drag), 6),
            "require_modeled_fill_quality": bool(thresholds.get("require_modeled_fill_quality", False)),
            "require_known_spread_or_execution_model": bool(thresholds.get("require_known_spread_or_execution_model", False)),
            "required_before_new_entry": [
                "modeled_fill_quality",
                "known_spread_or_execution_model",
                "tradeability_score",
            ],
        },
        "portfolio_conflict_control": {
            "active": bool("portfolio_conflict" in families or drag >= 0.38),
            "max_overlap_pressure_norm": round(conflict_cap, 6),
            "conflict_score_discount_norm": round(0.16 + (0.30 * drag), 6),
            "block_when_confirmation_below_norm": 0.58,
        },
        "confirmation_bias_control": {
            "active": bool(confirmation_bias_score >= 0.22 or len(set(cause_names) & CONFIRMATION_BIAS_CAUSES) >= 3),
            "confirmation_bias_score_norm": round(confirmation_bias_score, 6),
            "loss_causes": [cause for cause in cause_names if cause in CONFIRMATION_BIAS_CAUSES],
            "required_evidence_channels": CONFIRMATION_EVIDENCE_CHANNELS,
            "min_independent_evidence_channels": 4 if drag >= 0.64 else 3,
            "independent_evidence_channel_floor_norm": 0.58 if drag >= 0.64 else 0.55,
            "block_when_quality_gate_below_norm": 0.62 if drag >= 0.64 else 0.56,
            "score_dampen_when_quality_below_norm": 0.70 if drag >= 0.64 else 0.64,
            "required_before_new_entry": CONFIRMATION_EVIDENCE_CHANNELS,
            "unknown_evidence_is_negative": True,
            "applies_to_all_bot_votes": True,
            "lift_condition": "fresh profitable paper refresh with source, fill, spread, event, and conflict evidence present",
        },
    }


def _upgrade_lane_summary(
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
    cause_counter: Counter[str],
) -> list[dict[str, Any]]:
    lane_profiles: dict[str, list[str]] = {lane: [] for lane in UPGRADE_LANE_IDS}
    for profile, control in active_profile_controls.items():
        contracts = control.get("upgrade_contracts") if isinstance(control.get("upgrade_contracts"), dict) else {}
        for lane in UPGRADE_LANE_IDS:
            lane_contract = contracts.get(lane) if isinstance(contracts.get(lane), dict) else {}
            if lane_contract.get("active"):
                lane_profiles[lane].append(profile)

    strategy_count = len(strategy_controls)
    summary: list[dict[str, Any]] = []
    for lane in UPGRADE_LANE_IDS:
        profiles = ordered_unique(lane_profiles.get(lane, []))
        active = bool(profiles or (lane == "loser_quarantine" and strategy_count))
        summary.append(
            {
                "lane": lane,
                "active": active,
                "profile_count": len(profiles),
                "strategy_control_count": strategy_count if lane in {"outcome_weighted_training", "loser_quarantine"} else 0,
                "top_loss_causes": [
                    {"cause": cause, "count": count}
                    for cause, count in cause_counter.most_common(4)
                ],
                "next_action": {
                    "outcome_weighted_training": "feed paper-loss hard negatives and paper-profit positives into retraining weights",
                    "per_sleeve_profit_score": "score every sleeve independently before promotion or widening",
                    "dynamic_sizing": "scale paper and candidate size by realized paper sleeve quality",
                    "regime_specific_promotion": "promote only in regimes where paper performance is clean",
                    "loser_quarantine": "quarantine or deweight profile-strategy pairs that keep losing",
                    "exit_intelligence": "tighten exits where unrealized drag clusters",
                    "execution_aware_alpha": "discount alpha when fill, spread, or tradeability evidence is weak",
                    "portfolio_conflict_control": "cap overlap when sleeve losses cluster around low confirmation",
                    "confirmation_bias_control": "require independent evidence channels before trusting repeated confirming votes",
                    "profit_harvest_intelligence": "learn when to realize paper winners versus holding trend-continuation winners",
                }[lane],
            }
        )
    return summary


def _upper_layer_training_contract(
    *,
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
    upgrade_lanes: list[dict[str, Any]],
) -> dict[str, Any]:
    controls = list(active_profile_controls.values())
    active = bool(controls or strategy_controls)
    profit_scores = [_clamp(_safe_float(row.get("profit_score"), 0.5)) for row in controls]
    drag_scores = [_clamp(_safe_float(row.get("drag_score"), 0.0)) for row in controls]
    mean_profit = sum(profit_scores) / max(len(profit_scores), 1) if profit_scores else 0.5
    max_drag = max(drag_scores) if drag_scores else 0.0
    mean_size = (
        sum(_clamp(_safe_float(row.get("position_size_multiplier"), 1.0)) for row in controls)
        / max(len(controls), 1)
        if controls
        else 1.0
    )
    active_lanes = [
        str(row.get("lane") or "")
        for row in upgrade_lanes
        if isinstance(row, dict) and bool(row.get("active", False)) and str(row.get("lane") or "").strip()
    ]
    return {
        "active": active,
        "trainable_targets": UPPER_LAYER_TRAINING_TARGETS,
        "mean_profit_score_norm": round(mean_profit, 6),
        "max_drag_score_norm": round(max_drag, 6),
        "mean_position_size_multiplier_norm": round(mean_size, 6),
        "active_upgrade_lanes": active_lanes,
        "feature_contract": {
            "master_features": [
                "paper_profitability_master_awareness_active_norm",
                "paper_profitability_master_profit_score_norm",
                "paper_profitability_master_drag_norm",
                "paper_profitability_master_training_weight_norm",
                "paper_profitability_master_size_multiplier_norm",
                "paper_profitability_master_risk_norm",
            ],
            "grand_master_features": [
                "paper_profitability_grandmaster_awareness_active_norm",
                "paper_profitability_grandmaster_profit_score_norm",
                "paper_profitability_grandmaster_drag_norm",
                "paper_profitability_grandmaster_risk_norm",
                "paper_profitability_grandmaster_exit_pressure_norm",
                "paper_profitability_grandmaster_execution_discount_norm",
            ],
            "leakage_guard": "paper outcomes may train weights and gates, but raw future PnL must not be used as an intrabar predictive feature",
        },
        "sample_weight_policy": {
            "paper_loss_hard_negative_multiplier": round(1.0 + (2.0 * max_drag), 6),
            "paper_profit_positive_multiplier": round(max(0.75, 1.0 + (0.50 * max(mean_profit - 0.5, 0.0))), 6),
            "strategy_quarantine_multiplier": round(1.0 + min(len(strategy_controls), 24) / 12.0, 6),
            "min_profitable_refreshes_before_weight_lift": 2 if max_drag < 0.64 else 3,
        },
        "promotion_gate_policy": {
            "require_profit_score_floor_norm": 0.62,
            "require_drag_score_below_norm": 0.38,
            "require_no_active_paper_quarantine": True,
            "apply_to_master_candidate": True,
            "apply_to_grand_master_release": True,
        },
        "sub_bot_accuracy_target_contract": SUB_BOT_ACCURACY_TARGET_CONTRACT,
        "recommended_training_mode": "master_profitability_canary",
        "recommended_actions": [
            "train masters against paper outcome-weighted labels after controls are refreshed",
            "train Grand Master on when to override, shrink, or block master votes under paper drag",
            "push sub-bots toward 80-90% only when the score is clean out-of-sample and not overfit",
            "keep leakage guard active so paper PnL adjusts weights without becoming a future-leaking signal",
        ],
    }


def _scout_collection_contract(
    *,
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
    cause_counter: Counter[str],
    extra_scout_bot_ids: list[str] | None = None,
) -> dict[str, Any]:
    target_profiles = ordered_unique(
        [
            str(profile)
            for profile, control in active_profile_controls.items()
            if isinstance(control, dict) and bool(control.get("active", False))
        ]
    )
    target_bot_ids = ordered_unique(
        [
            str(row.get("bot_id") or "").strip()
            for row in strategy_controls
            if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
        ]
        + [
            str(bot_id or "").strip()
            for bot_id in (extra_scout_bot_ids or [])
            if str(bot_id or "").strip()
        ]
    )
    return {
        "active": bool(target_profiles or target_bot_ids),
        "mode": "collect_first_no_execution",
        "target_profiles": target_profiles,
        "target_bot_ids": target_bot_ids[:48],
        "required_context": SCOUT_PROFITABILITY_CONTEXT,
        "required_label_outputs": SCOUT_PROFITABILITY_LABEL_OUTPUTS,
        "collection_rules": [
            "collect accepted and rejected trade candidates with the same point-in-time snapshot keys",
            "label no-trade counterfactuals so chop/sideways regimes become explicit training examples",
            "persist exit-drag traces before any scout can leave collection-only mode",
            "treat missing fill, spread, source, or confirmation evidence as a negative evidence label",
        ],
        "top_loss_causes": [
            {"cause": cause, "count": count}
            for cause, count in cause_counter.most_common(8)
        ],
        "execution_guard": {
            "paper_trading_enabled": False,
            "live_trading_enabled": False,
            "training_allowed_after_labels_present": True,
        },
    }


def _registry_trading_scout_bot_ids(project_root: Path, *, limit: int = 32) -> list[str]:
    registry = load_json(project_root / "master_bot_registry.json")
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    bot_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id:
            continue
        lifecycle = str(row.get("lifecycle_state") or "").strip().lower()
        collection_active = bool(row.get("data_collection_active", False))
        training_excluded = bool(row.get("training_excluded", False)) or bool(row.get("exclude_from_training", False))
        if lifecycle != "data_collection_only" or not collection_active or not training_excluded:
            continue
        descriptors = " ".join(
            str(row.get(key) or "").strip().lower()
            for key in (
                "slot_kind",
                "slot_label",
                "slot_objective",
                "trade_style",
                "trading_rationale",
                "operator_intent",
                "expansion_scope",
            )
        )
        is_trading_scout = bool(
            row.get("trading_three_slot", False)
            or "trading_sub_bot" in descriptors
            or "trader" in descriptors
            or "scalper" in descriptors
            or "paper_trade" in descriptors
        )
        if is_trading_scout:
            bot_ids.append(bot_id)
    return ordered_unique(bot_ids)[: max(int(limit), 1)]


def _hardening_contract(
    *,
    active_profile_controls: dict[str, dict[str, Any]],
    strategy_controls: list[dict[str, Any]],
    scout_collection_contract: dict[str, Any],
) -> dict[str, Any]:
    quarantined_profiles = ordered_unique(
        [
            profile
            for profile, control in active_profile_controls.items()
            if isinstance(control, dict) and str(control.get("action") or "").strip().lower() == "quarantine_new_entries"
        ]
    )
    exit_drag_profiles = ordered_unique(
        [
            profile
            for profile, control in active_profile_controls.items()
            if isinstance(control.get("exit_intelligence"), dict)
            and bool(control["exit_intelligence"].get("active", False))
        ]
    )
    evidence_profiles = ordered_unique(
        [
            profile
            for profile, control in active_profile_controls.items()
            if isinstance(control.get("confirmation_bias_control"), dict)
            and bool(control["confirmation_bias_control"].get("active", False))
        ]
    )
    quarantined_pairs = [
        str(row.get("strategy") or "")
        for row in strategy_controls
        if isinstance(row, dict) and str(row.get("mode") or "").strip().lower() == "paper_quarantine"
    ]
    deweighted_pairs = [
        str(row.get("strategy") or "")
        for row in strategy_controls
        if isinstance(row, dict) and str(row.get("mode") or "").strip().lower() == "deweight"
    ]
    action_rows = [
        {
            "action_id": "stop_new_entries_in_worst_sleeves",
            "status": "active" if quarantined_profiles else "armed",
            "target_count": len(quarantined_profiles),
            "targets": quarantined_profiles,
            "expected_effect": "stop adding fresh paper risk where sleeve PnL is already in protective tightening",
            "risk_level": "low",
        },
        {
            "action_id": "accelerate_unrealized_drag_reduction",
            "status": "active" if exit_drag_profiles else "armed",
            "target_count": len(exit_drag_profiles),
            "targets": exit_drag_profiles,
            "expected_effect": "block adds during unrealized drag and prefer reduce-only handling until drag improves",
            "risk_level": "medium",
        },
        {
            "action_id": "require_independent_evidence_before_action",
            "status": "active" if evidence_profiles else "armed",
            "target_count": len(evidence_profiles),
            "targets": evidence_profiles,
            "expected_effect": "hold or dampen paper actions unless source, fill, spread, event, and conflict evidence is present",
            "risk_level": "low",
        },
        {
            "action_id": "deweight_losing_profile_strategy_pairs",
            "status": "active" if strategy_controls else "armed",
            "target_count": len(strategy_controls),
            "targets": ordered_unique(quarantined_pairs + deweighted_pairs)[:24],
            "expected_effect": "quarantine the worst bot/profile pairs and shrink weaker pairs until paper refreshes improve",
            "risk_level": "low",
        },
        {
            "action_id": "expand_scout_labels_for_profitability_feedback",
            "status": "active" if bool(scout_collection_contract.get("active", False)) else "armed",
            "target_count": len(scout_collection_contract.get("target_bot_ids") or []),
            "targets": scout_collection_contract.get("target_bot_ids") or [],
            "expected_effect": "turn every weak paper trade and rejected candidate into usable training evidence",
            "risk_level": "low",
        },
    ]
    return {
        "active": bool(active_profile_controls or strategy_controls),
        "action_count": len(PROFITABILITY_HARDENING_ACTIONS),
        "actions": action_rows,
        "action_ids": PROFITABILITY_HARDENING_ACTIONS,
        "new_entry_policy": {
            "block_quarantined_profiles": True,
            "block_quarantined_strategy_pairs": True,
            "deweighted_pairs_require_evidence_gate": True,
        },
        "unrealized_drag_policy": {
            "block_adds_while_drag_active": True,
            "prefer_reduce_over_add": True,
            "require_exit_drag_labels": True,
        },
        "evidence_policy": {
            "required_channels": CONFIRMATION_EVIDENCE_CHANNELS,
            "unknown_evidence_is_negative": True,
            "min_channels_for_severe_drag": 4,
        },
        "scout_collection_contract": scout_collection_contract,
    }


def _profile_drag(row: dict[str, Any]) -> float:
    net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
    unrealized = _safe_float(row.get("ending_unrealized_pnl_total"), 0.0)
    win_rate = _safe_float(row.get("win_rate"), 1.0 if row.get("win_rate") is not None else 0.5)
    executions = _safe_int(row.get("executions"), 0)
    losing_count = _safe_int(row.get("losing_strategy_count"), 0)
    winning_count = _safe_int(row.get("winning_strategy_count"), 0)
    loss_scale = 1500.0 if executions >= 1000 else 500.0
    if executions <= 0:
        return 0.0
    return _clamp(
        0.42 * _clamp(abs(min(net, 0.0)) / loss_scale)
        + 0.26 * _clamp(abs(min(unrealized, 0.0)) / loss_scale)
        + 0.22 * _clamp(max(0.45 - win_rate, 0.0) / 0.45)
        + 0.10 * _clamp(max(losing_count - winning_count, 0) / 8.0)
    )


def _profile_action(profile: str, drag: float, net: float, win_rate: float | None) -> str:
    if drag >= 0.88 or net <= -1000.0:
        return "quarantine_new_entries"
    if drag >= 0.64 or (win_rate is not None and win_rate < 0.18):
        return "tighten_entry_quality_hard"
    if drag >= 0.38 or net < 0.0:
        return "tighten_entry_quality"
    return "observe"


def _profile_thresholds(profile: str, families: list[str], drag: float) -> dict[str, Any]:
    min_source = 0.35
    min_tradeability = 0.42
    min_execution = 0.42
    min_confirmation = 0.34
    min_catalyst = 0.0

    if "source_quality" in families:
        min_source = max(min_source, 0.56)
    if "tradeability" in families:
        min_tradeability = max(min_tradeability, 0.58)
    if "fill_quality" in families or "spread_quality" in families:
        min_execution = max(min_execution, 0.60)
        min_tradeability = max(min_tradeability, 0.54)
    if "catalyst_confirmation" in families and profile in CATALYST_PROFILES:
        min_catalyst = 0.28
        min_confirmation = max(min_confirmation, 0.46)
    if {"portfolio_conflict", "source_quality", "fill_quality", "spread_quality"} & set(families):
        min_confirmation = max(min_confirmation, 0.42)
    if {"fill_quality", "spread_quality"} & set(families):
        min_source = max(min_source, 0.56)

    if drag >= 0.64:
        min_source = max(min_source, 0.60)
        min_tradeability = max(min_tradeability, 0.60)
        min_execution = max(min_execution, 0.62)
        min_confirmation = max(min_confirmation, 0.48)
    if drag >= 0.88:
        min_source = max(min_source, 0.66)
        min_tradeability = max(min_tradeability, 0.66)
        min_execution = max(min_execution, 0.68)
        min_confirmation = max(min_confirmation, 0.54)
        if profile in CATALYST_PROFILES:
            min_catalyst = max(min_catalyst, 0.34)

    return {
        "min_source_quality_norm": round(min_source, 6),
        "min_tradeability_norm": round(min_tradeability, 6),
        "min_execution_fitness_norm": round(min_execution, 6),
        "min_cross_asset_confirmation_norm": round(min_confirmation, 6),
        "min_event_proximity_norm": round(min_catalyst, 6),
        "require_known_spread_or_execution_model": bool("spread_quality" in families or drag >= 0.88),
        "require_modeled_fill_quality": bool("fill_quality" in families or drag >= 0.88),
    }


def _strategy_controls(
    profile: str,
    sleeve: dict[str, Any],
    *,
    cause_names: list[str],
    profile_drag: float,
    limit: int = 8,
) -> list[dict[str, Any]]:
    rows = sleeve.get("top_losing_strategies") if isinstance(sleeve.get("top_losing_strategies"), list) else []
    controls: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        strategy = str(row.get("strategy") or "").strip()
        if not strategy:
            continue
        net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        if net >= 0.0:
            continue
        penalty = _clamp(abs(net) / 450.0)
        mode = "paper_quarantine" if penalty >= 0.70 else "deweight"
        size_multiplier = round(max(0.0, 1.0 - (0.85 * penalty)), 6)
        confirmation_bias = _confirmation_bias_score(
            cause_names,
            drag=max(profile_drag, penalty),
            net=net,
            win_rate=_safe_float(sleeve.get("win_rate"), 0.0) if sleeve.get("win_rate") is not None else None,
        )
        min_channels = 4 if max(profile_drag, penalty) >= 0.64 else 3
        confirmation_contract = {
            "active": bool(confirmation_bias >= 0.20 or len(set(cause_names) & CONFIRMATION_BIAS_CAUSES) >= 3),
            "confirmation_bias_score_norm": round(confirmation_bias, 6),
            "source_loss_causes": [cause for cause in cause_names if cause in CONFIRMATION_BIAS_CAUSES],
            "required_evidence_channels": CONFIRMATION_EVIDENCE_CHANNELS,
            "min_independent_evidence_channels": min_channels,
            "independent_evidence_channel_floor_norm": 0.58 if min_channels >= 4 else 0.55,
            "block_when_quality_gate_below_norm": 0.62 if min_channels >= 4 else 0.56,
            "score_dampen_when_quality_below_norm": 0.70 if min_channels >= 4 else 0.64,
            "required_before_new_entry": CONFIRMATION_EVIDENCE_CHANNELS,
            "unknown_evidence_is_negative": True,
            "applies_to_profile_strategy_pair": True,
        }
        new_entry_cap = 0 if mode == "paper_quarantine" else 1
        controls.append(
            {
                "profile": profile,
                "strategy": strategy,
                "bot_id": _strategy_bot_id(strategy),
                "mode": mode,
                "ending_net_pnl_total": round(net, 6),
                "score_penalty_norm": round(penalty, 6),
                "position_size_multiplier": size_multiplier,
                "new_entry_cap": new_entry_cap,
                "block_new_entries": mode == "paper_quarantine",
                "confirmation_bias_score_norm": round(confirmation_bias, 6),
                "loss_causes": cause_names,
                "upgrade_contracts": {
                    "outcome_weighted_training": {
                        "active": True,
                        "sample_weight_multiplier": round(1.0 + (1.60 * penalty), 6),
                        "label": "paper_loss_hard_negative",
                        "hard_negative_required_labels": [
                            "paper_loss_cause",
                            "paper_unrealized_drag_bucket",
                            "entry_evidence_gate_result",
                        ],
                    },
                    "loser_quarantine": {
                        "active": True,
                        "mode": mode,
                        "new_entry_cap": new_entry_cap,
                        "block_new_entries": mode == "paper_quarantine",
                        "paper_only_retest_required": True,
                        "lift_condition": "fresh profitable paper refresh and improved walk-forward quality",
                    },
                    "dynamic_sizing": {
                        "active": True,
                        "paper_profitability_size_multiplier_norm": size_multiplier,
                        "max_new_entry_multiplier_norm": size_multiplier,
                    },
                    "confirmation_bias_control": confirmation_contract,
                },
                "confirmation_bias_control": confirmation_contract,
                "training_feedback": [
                    "add paper-loss hard negatives for this profile strategy pair",
                    "collect independent confirmation evidence before trusting matching bot votes",
                    "require fresh walk-forward improvement before lifting the deweight",
                ],
                "data_intake_enrichment": {
                    "required_context": SCOUT_PROFITABILITY_CONTEXT,
                    "required_label_outputs": SCOUT_PROFITABILITY_LABEL_OUTPUTS,
                },
            }
        )
    controls.sort(key=lambda item: (float(item["ending_net_pnl_total"]), item["strategy"]))
    return controls[: max(int(limit), 1)]


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    paper_path = health / "paper_performance_latest.json"
    training_quality_path = health / "training_quality_control_latest.json"
    previous_control_path = DEFAULT_CONTROL_PATH if project_root == PROJECT_ROOT else health / "paper_runtime_profitability_controls_latest.json"
    paper = load_json(paper_path)
    training_quality = load_json(training_quality_path)
    previous_runtime_control = load_json(previous_control_path)
    previous_daily_goal_contract = (
        previous_runtime_control.get("daily_sleeve_harvest_goal_contract")
        if isinstance(previous_runtime_control.get("daily_sleeve_harvest_goal_contract"), dict)
        else {}
    )
    history_latest = _latest_history_row(paper)
    day_row = paper.get("day") if isinstance(paper.get("day"), dict) else {}
    sleeves = paper.get("sleeve_latest") if isinstance(paper.get("sleeve_latest"), list) else []

    active_profile_controls: dict[str, dict[str, Any]] = {}
    strategy_controls: list[dict[str, Any]] = []
    cause_counter: Counter[str] = Counter()
    net_sum = 0.0
    unrealized_sum = 0.0
    realized_sum = 0.0
    execution_sum = 0

    for row in sleeves:
        if not isinstance(row, dict):
            continue
        profile = _normal_profile(row.get("profile"))
        if not profile:
            continue
        executions = _safe_int(row.get("executions"), 0)
        net = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        unrealized = _safe_float(row.get("ending_unrealized_pnl_total"), 0.0)
        realized = _safe_float(row.get("ending_realized_pnl_total"), 0.0)
        win_rate_raw = row.get("win_rate")
        win_rate = _safe_float(win_rate_raw, 0.0) if win_rate_raw is not None else None
        net_sum += net
        unrealized_sum += unrealized
        realized_sum += realized
        execution_sum += max(executions, 0)
        cause_names = _cause_names(row)
        cause_counter.update(cause_names)
        drag = _profile_drag(row)
        action = _profile_action(profile, drag, net, win_rate)
        if action == "observe":
            continue
        families = _quality_families(cause_names)
        thresholds = _profile_thresholds(profile, families, drag)
        confirmation_bias = _confirmation_bias_score(cause_names, drag=drag, net=net, win_rate=win_rate)
        position_multiplier = round(max(0.05, 1.0 - (0.88 * drag)), 6)
        if action == "quarantine_new_entries":
            position_multiplier = min(position_multiplier, 0.10)
        profit_score = _profit_score(row, drag)
        losing_strategy_count = _safe_int(row.get("losing_strategy_count"), 0)
        upgrade_contracts = _profile_upgrade_contracts(
            profile=profile,
            row=row,
            action=action,
            drag=drag,
            profit_score=profit_score,
            position_multiplier=position_multiplier,
            families=families,
            cause_names=cause_names,
            confirmation_bias_score=confirmation_bias,
            thresholds=thresholds,
            losing_strategy_count=losing_strategy_count,
        )
        active_profile_controls[profile] = {
            "profile": profile,
            "active": True,
            "action": action,
            "drag_score": round(drag, 6),
            "profit_score": round(profit_score, 6),
            "profit_grade": _profit_grade(profit_score),
            "confirmation_bias_score": round(confirmation_bias, 6),
            "executions": executions,
            "win_rate": round(win_rate, 6) if win_rate is not None else None,
            "ending_net_pnl_total": round(net, 6),
            "ending_realized_pnl_total": round(realized, 6),
            "ending_unrealized_pnl_total": round(unrealized, 6),
            "position_size_multiplier": position_multiplier,
            "new_entry_cap": 0 if action == "quarantine_new_entries" else (1 if drag >= 0.64 else 3),
            "quality_families": families,
            "thresholds": thresholds,
            "top_loss_causes": _loss_causes(row)[:5],
            "upgrade_contracts": upgrade_contracts,
            "outcome_weighted_training": upgrade_contracts["outcome_weighted_training"],
            "per_sleeve_profit_score": upgrade_contracts["per_sleeve_profit_score"],
            "dynamic_sizing": upgrade_contracts["dynamic_sizing"],
            "regime_specific_promotion": upgrade_contracts["regime_specific_promotion"],
            "loser_quarantine": upgrade_contracts["loser_quarantine"],
            "exit_intelligence": upgrade_contracts["exit_intelligence"],
            "execution_aware_alpha": upgrade_contracts["execution_aware_alpha"],
            "portfolio_conflict_control": upgrade_contracts["portfolio_conflict_control"],
            "confirmation_bias_control": upgrade_contracts["confirmation_bias_control"],
            "runtime_policy": {
                "block_new_entries_when_quality_gate_fails": True,
                "dampen_score_when_quality_is_borderline": True,
                "paper_only_until_next_profitable_refresh": True,
                "apply_outcome_weighted_training": True,
                "apply_dynamic_sizing": True,
                "apply_regime_specific_promotion": True,
                "apply_loser_quarantine": True,
                "apply_exit_intelligence": True,
                "apply_execution_aware_alpha": True,
                "apply_portfolio_conflict_control": True,
                "apply_confirmation_bias_control": True,
            },
        }
        strategy_controls.extend(_strategy_controls(profile, row, cause_names=cause_names, profile_drag=drag))

    strategy_controls.sort(key=lambda item: (float(item["ending_net_pnl_total"]), item["profile"], item["strategy"]))
    strategy_controls = strategy_controls[:24]

    history_change = _safe_float(day_row.get("change_vs_previous_day"), _safe_float(history_latest.get("change_vs_previous_day"), 0.0))
    raw_operational_materiality_filter = _raw_operational_materiality_filter(
        active_profile_controls=active_profile_controls,
        strategy_controls=strategy_controls,
        net_sum=net_sum,
    )
    gradeable_profile_controls = (
        raw_operational_materiality_filter.get("_gradeable_profile_controls")
        if isinstance(raw_operational_materiality_filter.get("_gradeable_profile_controls"), dict)
        else active_profile_controls
    )
    gradeable_strategy_controls = (
        raw_operational_materiality_filter.get("_gradeable_strategy_controls")
        if isinstance(raw_operational_materiality_filter.get("_gradeable_strategy_controls"), list)
        else strategy_controls
    )
    financial_grade = _financial_grade(
        net_sum=net_sum,
        realized_sum=realized_sum,
        unrealized_sum=unrealized_sum,
        change_vs_previous_day=history_change,
        executions=execution_sum,
    )
    _apply_a_plus_recovery_mode(
        active_profile_controls=active_profile_controls,
        strategy_controls=strategy_controls,
        financial_grade=financial_grade,
    )
    _apply_protective_tightening_mode(
        active_profile_controls=active_profile_controls,
        strategy_controls=strategy_controls,
    )
    weak_count = len(active_profile_controls)
    base_raw_operational_outcome_grade = _operational_outcome_grade(
        weak_count=_safe_int(raw_operational_materiality_filter.get("gradeable_weak_profile_count"), weak_count),
        strategy_count=_safe_int(raw_operational_materiality_filter.get("gradeable_strategy_control_count"), len(strategy_controls)),
    )
    raw_operational_containment_filter = _raw_operational_containment_filter(
        gradeable_profile_controls=gradeable_profile_controls,
        gradeable_strategy_controls=gradeable_strategy_controls,
        base_grade=base_raw_operational_outcome_grade,
    )
    active_raw_profile_controls = (
        raw_operational_containment_filter.get("_active_profile_controls")
        if isinstance(raw_operational_containment_filter.get("_active_profile_controls"), dict)
        else gradeable_profile_controls
    )
    active_raw_strategy_controls = (
        raw_operational_containment_filter.get("_active_strategy_controls")
        if isinstance(raw_operational_containment_filter.get("_active_strategy_controls"), list)
        else gradeable_strategy_controls
    )
    profit_harvest_controls = _profit_harvest_profile_controls(sleeves)
    profit_realization_contract = _profit_realization_contract(
        profit_harvest_controls=profit_harvest_controls,
        net_sum=net_sum,
        realized_sum=realized_sum,
        unrealized_sum=unrealized_sum,
    )
    profit_harvest_strategy_controls = _strategy_profit_harvest_controls(sleeves, profit_harvest_controls)
    profit_harvest_position_ledger = _position_harvest_ledger(
        project_root=project_root,
        paper=paper,
        profit_harvest_controls=profit_harvest_controls,
        strategy_harvest_controls=profit_harvest_strategy_controls,
    )
    daily_sleeve_harvest_goal_contract = _daily_sleeve_harvest_goal_contract(
        profit_realization_contract=profit_realization_contract,
        profit_harvest_controls=profit_harvest_controls,
        position_ledger=profit_harvest_position_ledger,
        strategy_harvest_controls=profit_harvest_strategy_controls,
        previous_daily_goal_contract=previous_daily_goal_contract,
    )
    _apply_daily_harvest_goals_to_profile_controls(
        profit_harvest_controls=profit_harvest_controls,
        daily_goal_contract=daily_sleeve_harvest_goal_contract,
    )
    daily_target_adaptation_contract = _daily_target_adaptation_contract(daily_sleeve_harvest_goal_contract)
    paper_harvest_execution_contract = _paper_harvest_execution_contract(
        daily_goal_contract=daily_sleeve_harvest_goal_contract,
        position_ledger=profit_harvest_position_ledger,
        strategy_harvest_controls=profit_harvest_strategy_controls,
    )
    paper_harvest_infrabot_contract = _paper_harvest_infrabot_contract(
        profit_realization_contract=profit_realization_contract,
        daily_goal_contract=daily_sleeve_harvest_goal_contract,
        paper_harvest_execution_contract=paper_harvest_execution_contract,
        profit_harvest_controls=profit_harvest_controls,
    )
    _apply_paper_harvest_infrabots_to_profile_controls(
        profit_harvest_controls=profit_harvest_controls,
        infrabot_contract=paper_harvest_infrabot_contract,
    )
    _apply_daily_target_adaptation_to_profile_controls(
        profit_harvest_controls=profit_harvest_controls,
        adaptation_contract=daily_target_adaptation_contract,
    )
    profit_harvest_regret_replay_contract = _profit_harvest_regret_replay_contract(
        position_ledger=profit_harvest_position_ledger,
        strategy_harvest_controls=profit_harvest_strategy_controls,
    )
    aggressive_harvest_mode_contract = _aggressive_harvest_mode_contract(sleeves, profit_harvest_controls)
    runner_protection_contract = _runner_protection_contract(profit_harvest_controls)
    profit_rotation_contract = _profit_rotation_contract(sleeves, profit_harvest_controls)
    profit_harvest_report_card = _profit_harvest_report_card(
        profit_realization_contract=profit_realization_contract,
        position_ledger=profit_harvest_position_ledger,
        strategy_harvest_controls=profit_harvest_strategy_controls,
        profit_harvest_controls=profit_harvest_controls,
    )
    profit_harvest_aplus_campaign = (
        profit_harvest_report_card.get("a_plus_campaign")
        if isinstance(profit_harvest_report_card.get("a_plus_campaign"), dict)
        else {}
    )
    grand_master_profit_harvest_awareness_contract = _grand_master_profit_harvest_awareness_contract(
        profit_realization_contract=profit_realization_contract,
        report_card=profit_harvest_report_card,
        rotation_contract=profit_rotation_contract,
    )

    overall_status = "ready"
    if weak_count:
        overall_status = "needs_tuning"
    if any(row.get("action") == "quarantine_new_entries" for row in active_profile_controls.values()):
        overall_status = "protective_tightening"

    training_score = _safe_float(training_quality.get("training_quality_score"), 0.0)
    raw_operational_outcome_grade = str(
        raw_operational_containment_filter.get("contained_grade")
        or base_raw_operational_outcome_grade
    )
    remaining_low_grade_layers = _remaining_low_grade_layers(
        raw_operational_outcome_grade=raw_operational_outcome_grade,
        base_raw_operational_outcome_grade=base_raw_operational_outcome_grade,
        raw_operational_materiality_filter={
            key: value
            for key, value in raw_operational_materiality_filter.items()
            if not str(key).startswith("_")
        },
        raw_operational_containment_filter={
            key: value
            for key, value in raw_operational_containment_filter.items()
            if not str(key).startswith("_")
        },
        profit_harvest_report_card=profit_harvest_report_card,
        active_profile_controls=active_profile_controls,
    )
    low_grade_control_report_card = _low_grade_control_report_card(
        remaining_low_grade_layers=remaining_low_grade_layers,
        profit_harvest_report_card=profit_harvest_report_card,
    )
    raw_operational_grade_lift_contract = _raw_operational_grade_lift_contract(
        active_profile_controls=active_raw_profile_controls,
        strategy_controls=active_raw_strategy_controls,
        raw_operational_grade=raw_operational_outcome_grade,
    )
    operational_control_grade = _operational_control_grade(
        active_profile_controls=active_profile_controls,
        strategy_controls=strategy_controls,
        financial_grade=financial_grade,
    )
    unprotected_counts = _unprotected_operational_counts(
        active_profile_controls=active_profile_controls,
        strategy_controls=strategy_controls,
    )
    operational_outcome_grade = _operational_outcome_grade(
        weak_count=_safe_int(unprotected_counts.get("unprotected_weak_profile_count"), weak_count),
        strategy_count=_safe_int(unprotected_counts.get("unprotected_strategy_control_count"), len(strategy_controls)),
    )
    a_plus_target_contract = _a_plus_target_contract(
        financial_grade=financial_grade,
        operational_outcome_grade=operational_outcome_grade,
        raw_operational_outcome_grade=raw_operational_outcome_grade,
        operational_control_grade=operational_control_grade,
        active_profile_controls=active_profile_controls,
        strategy_controls=strategy_controls,
        unprotected_counts=unprotected_counts,
        net_sum=net_sum,
        realized_sum=realized_sum,
        unrealized_sum=unrealized_sum,
        change_vs_previous_day=history_change,
        executions=execution_sum,
    )
    max_grade_push_contract = _max_grade_push_contract(
        operational_control_grade=operational_control_grade,
        profit_harvest_report_card=profit_harvest_report_card,
        daily_goal_contract=daily_sleeve_harvest_goal_contract,
        paper_harvest_execution_contract=paper_harvest_execution_contract,
        infrabot_contract=paper_harvest_infrabot_contract,
    )
    profitability_realization_expansion_contract = _profitability_realization_expansion_contract(
        sleeves=sleeves,
        active_profile_controls=active_profile_controls,
        strategy_controls=strategy_controls,
        profit_harvest_controls=profit_harvest_controls,
        profit_harvest_strategy_controls=profit_harvest_strategy_controls,
        profit_harvest_report_card=profit_harvest_report_card,
        daily_goal_contract=daily_sleeve_harvest_goal_contract,
        paper_harvest_execution_contract=paper_harvest_execution_contract,
        profit_realization_contract=profit_realization_contract,
        cause_counter=cause_counter,
    )
    profitability_compounding_autopilot_contract = _profitability_compounding_autopilot_contract(
        expansion_contract=profitability_realization_expansion_contract,
        profit_harvest_report_card=profit_harvest_report_card,
        profit_realization_contract=profit_realization_contract,
        daily_goal_contract=daily_sleeve_harvest_goal_contract,
        paper_harvest_execution_contract=paper_harvest_execution_contract,
        active_profile_controls=active_profile_controls,
        strategy_controls=strategy_controls,
    )
    quant_strategy_expansion_admission_contract = _quant_strategy_expansion_admission_contract(
        sleeves=sleeves,
        expansion_contract=profitability_realization_expansion_contract,
        compounding_autopilot_contract=profitability_compounding_autopilot_contract,
        profit_harvest_report_card=profit_harvest_report_card,
        profit_realization_contract=profit_realization_contract,
        active_profile_controls=active_profile_controls,
        strategy_controls=strategy_controls,
        overall_status=overall_status,
    )
    net_grade = str(a_plus_target_contract.get("headline_grade") or financial_grade)

    upgrade_lanes = _upgrade_lane_summary(active_profile_controls, strategy_controls, cause_counter)
    upper_layer_training_contract = _upper_layer_training_contract(
        active_profile_controls=active_profile_controls,
        strategy_controls=strategy_controls,
        upgrade_lanes=upgrade_lanes,
    )
    scout_collection_contract = _scout_collection_contract(
        active_profile_controls=active_profile_controls,
        strategy_controls=strategy_controls,
        cause_counter=cause_counter,
        extra_scout_bot_ids=_registry_trading_scout_bot_ids(project_root),
    )
    hardening_contract = _hardening_contract(
        active_profile_controls=active_profile_controls,
        strategy_controls=strategy_controls,
        scout_collection_contract=scout_collection_contract,
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": overall_status in {"ready", "needs_tuning", "protective_tightening"},
        "overall_status": overall_status,
        "profitability_grade": net_grade,
        "financial_profitability_grade": financial_grade,
        "operational_outcome_grade": operational_outcome_grade,
        "raw_operational_outcome_grade": raw_operational_outcome_grade,
        "base_raw_operational_outcome_grade": base_raw_operational_outcome_grade,
        "operational_control_grade": operational_control_grade,
        "a_plus_target_contract": a_plus_target_contract,
        "raw_operational_materiality_filter": {
            key: value
            for key, value in raw_operational_materiality_filter.items()
            if not str(key).startswith("_")
        },
        "raw_operational_containment_filter": {
            key: value
            for key, value in raw_operational_containment_filter.items()
            if not str(key).startswith("_")
        },
        "remaining_low_grade_layers": remaining_low_grade_layers,
        "low_grade_control_report_card": low_grade_control_report_card,
        "low_grade_layer_summary": {
            "active": bool(remaining_low_grade_layers),
            "low_grade_layer_count": len(remaining_low_grade_layers),
            "active_blocker_count": sum(1 for row in remaining_low_grade_layers if bool(row.get("active_blocker", False))),
            "actionable_low_grade_layer_count": sum(1 for row in remaining_low_grade_layers if bool(row.get("active_blocker", False))),
            "contained_or_probationary_count": sum(1 for row in remaining_low_grade_layers if not bool(row.get("active_blocker", False))),
            "base_evidence_low_grade_count": low_grade_control_report_card.get("base_evidence_low_grade_count", 0),
            "profile_low_grade_count": low_grade_control_report_card.get("profile_low_grade_count", 0),
            "control_posture_grade": low_grade_control_report_card.get("control_posture_grade", ""),
            "control_posture_status": low_grade_control_report_card.get("status", ""),
            "a_plus_control_ready": bool(low_grade_control_report_card.get("a_plus_control_ready", False)),
            "a_plus_raw_evidence_ready": bool(low_grade_control_report_card.get("a_plus_raw_evidence_ready", False)),
            "lowest_visible_grades": sorted({str(row.get("grade") or "") for row in remaining_low_grade_layers if str(row.get("grade") or "")}),
            "rule": "headline/control grades do not hide base D/F layers; low base/profile grades remain visible until outcome evidence itself improves",
        },
        "grade_transparency_contract": {
            "headline_grade_meaning": "current controlled operating posture after active safety, containment, and rescue controls",
            "base_grade_meaning": "raw evidence before those controls; this can stay D/F even when headline operation is safer",
            "reporting_rule": "always report remaining_low_grade_layers when any grade-like base/profile field is D or F",
            "no_live_trade_authority": True,
        },
        "raw_operational_grade_lift_contract": raw_operational_grade_lift_contract,
        "profit_harvest_profile_controls": profit_harvest_controls,
        "profit_harvest_strategy_controls": profit_harvest_strategy_controls,
        "profit_harvest_position_ledger": profit_harvest_position_ledger,
        "daily_sleeve_harvest_goal_contract": daily_sleeve_harvest_goal_contract,
        "daily_target_adaptation_contract": daily_target_adaptation_contract,
        "paper_harvest_execution_contract": paper_harvest_execution_contract,
        "paper_harvest_infrabot_contract": paper_harvest_infrabot_contract,
        "profit_harvest_regret_replay_contract": profit_harvest_regret_replay_contract,
        "aggressive_harvest_mode_contract": aggressive_harvest_mode_contract,
        "runner_protection_contract": runner_protection_contract,
        "profit_rotation_contract": profit_rotation_contract,
        "profit_harvest_report_card": profit_harvest_report_card,
        "profit_harvest_aplus_campaign": profit_harvest_aplus_campaign,
        "grand_master_profit_harvest_awareness_contract": grand_master_profit_harvest_awareness_contract,
        "profit_realization_contract": profit_realization_contract,
        "max_grade_push_contract": max_grade_push_contract,
        "profitability_realization_expansion_contract": profitability_realization_expansion_contract,
        "profitability_compounding_autopilot_contract": profitability_compounding_autopilot_contract,
        "quant_strategy_expansion_admission_contract": quant_strategy_expansion_admission_contract,
        "paper_summary": {
            "day_utc": str(day_row.get("day_utc") or paper.get("day") or history_latest.get("day_utc") or ""),
            "executions": execution_sum or _safe_int(history_latest.get("executions"), 0),
            "ending_net_pnl_total": round(net_sum, 6),
            "ending_realized_pnl_total": round(realized_sum, 6),
            "ending_unrealized_pnl_total": round(unrealized_sum, 6),
            "history_ending_net_pnl_total": round(_safe_float(history_latest.get("ending_net_pnl_total"), net_sum), 6),
            "history_change_vs_previous_day": round(history_change, 6),
            "training_quality_score": round(training_score, 6),
        },
        "active_profile_control_count": weak_count,
        "active_profile_controls": active_profile_controls,
        "strategy_control_count": len(strategy_controls),
        "strategy_controls": strategy_controls,
        "upgrade_lane_count": len(UPGRADE_LANE_IDS),
        "profitability_upgrade_lanes": upgrade_lanes,
        "paper_profitability_hardening_contract": hardening_contract,
        "scout_collection_contract": scout_collection_contract,
        "master_grandmaster_training_contract": upper_layer_training_contract,
        "sub_bot_accuracy_target_contract": SUB_BOT_ACCURACY_TARGET_CONTRACT,
        "closed_loop_contract": {
            "active": True,
            "applies_to": [
                "training_weights",
                "paper_mirror_controls",
                "candidate_sizing",
                "promotion_gates",
                "exit_bias",
                "execution_filters",
                "portfolio_conflict_caps",
                "confirmation_bias_controls",
                "master_vote_training",
                "grand_master_override_training",
                "hardening_actions",
                "scout_collection_labels",
                "profit_harvest_regret_learning",
                "trend_continuation_holdback",
                "per_sleeve_realized_conversion_skill",
                "daily_sleeve_realized_profit_targets",
                "laddered_paper_profit_exits",
                "reduce_only_paper_harvest_intents",
                "paper_harvest_infrabot_supervision",
                "max_grade_push_contract",
                "profitability_realization_expansion_1_to_8",
                "profitability_compounding_autopilot_v1",
                "quant_strategy_expansion_admission",
                "daily_target_adaptation",
                "post_target_collection_expansion",
            ],
            "refresh_command": [
                "./scripts/ops/opsctl.sh",
                "paper-profitability-control",
                "--apply",
                "--json",
            ],
            "stop_condition": "all active sleeves show positive net, controlled unrealized drag, and no active losing-strategy quarantine",
        },
        "top_loss_causes": [
            {"cause": cause, "count": count}
            for cause, count in cause_counter.most_common(10)
        ],
        "recommended_actions": ordered_unique(
            [
                "block new paper entries on sleeves with active profitability quarantine",
                "tighten source, tradeability, execution, and catalyst gates where paper losses cluster",
                "deweight losing profile-strategy pairs before running more training",
                "feed paper-loss hard negatives back into training data intake and calibration",
                "lift controls only after the next paper-performance refresh shows positive net and lower unrealized drag",
                "use per-sleeve profit score to decide promotion, sizing, and runtime widening",
                "tighten exits and conflict caps when paper losses are unrealized or overlap-driven",
                "require independent evidence channels before trusting bots that repeatedly confirm low-quality trades",
                "keep new trading scouts collection-only while they collect paper loss, exit drag, and no-trade counterfactual labels",
                "hold financial A+ by quarantining every weak sleeve and losing strategy pair until operational outcome A+ clears",
                "convert oversized unrealized paper gains into realized paper gains with profile-level harvest trims",
                "use profit-harvest intelligence to avoid trimming strong winners too early",
                "measure harvest regret and post-trim follow-through so realization timing improves after every refresh",
                "use position-level and strategy-level harvest ledgers before increasing trim aggression",
                "use daily sleeve realized-profit targets to convert paper winners without dumping runners",
                "emit reduce-only SELL paper fills from the harvest intent contract when executor resolution is clean",
                "let harvest infrabots verify realized-share progress, stale intents, runner protection, and sleeve explanations",
                "raise each sleeve's next daily target after the previous paper target is met, or expand labels when it needs proof",
                "push control posture to max while raw grades remain evidence-based",
                "run the 1-8 profitability realization expansion so weak sleeves stop adding drag and winning sleeves get attribution-weighted attention",
                "follow the profitability compounding autopilot do_first queue before widening paper size or training batches",
                "admit new quant strategies through collection-only quant strategy admission before any paper widening",
                "rotate harvested paper gains only into sleeves with clean quality evidence or hold them as paper cash",
            ]
        ),
        "runtime_control_file": str(DEFAULT_CONTROL_PATH if project_root == PROJECT_ROOT else health / "paper_runtime_profitability_controls_latest.json"),
        "source_files": {
            "paper_performance": str(paper_path),
            "training_quality": str(training_quality_path),
        },
    }
    return payload


def build_runtime_control_payload(payload: dict[str, Any]) -> dict[str, Any]:
    profile_controls = payload.get("active_profile_controls") if isinstance(payload.get("active_profile_controls"), dict) else {}
    strategies = payload.get("strategy_controls") if isinstance(payload.get("strategy_controls"), list) else []
    upgrade_lanes = payload.get("profitability_upgrade_lanes") if isinstance(payload.get("profitability_upgrade_lanes"), list) else []
    upper_layer_training_contract = (
        payload.get("master_grandmaster_training_contract")
        if isinstance(payload.get("master_grandmaster_training_contract"), dict)
        else {}
    )
    hardening_contract = (
        payload.get("paper_profitability_hardening_contract")
        if isinstance(payload.get("paper_profitability_hardening_contract"), dict)
        else {}
    )
    a_plus_target_contract = (
        payload.get("a_plus_target_contract")
        if isinstance(payload.get("a_plus_target_contract"), dict)
        else {}
    )
    raw_operational_grade_lift_contract = (
        payload.get("raw_operational_grade_lift_contract")
        if isinstance(payload.get("raw_operational_grade_lift_contract"), dict)
        else {}
    )
    raw_operational_materiality_filter = (
        payload.get("raw_operational_materiality_filter")
        if isinstance(payload.get("raw_operational_materiality_filter"), dict)
        else {}
    )
    raw_operational_containment_filter = (
        payload.get("raw_operational_containment_filter")
        if isinstance(payload.get("raw_operational_containment_filter"), dict)
        else {}
    )
    remaining_low_grade_layers = (
        payload.get("remaining_low_grade_layers")
        if isinstance(payload.get("remaining_low_grade_layers"), list)
        else []
    )
    low_grade_layer_summary = (
        payload.get("low_grade_layer_summary")
        if isinstance(payload.get("low_grade_layer_summary"), dict)
        else {}
    )
    low_grade_control_report_card = (
        payload.get("low_grade_control_report_card")
        if isinstance(payload.get("low_grade_control_report_card"), dict)
        else {}
    )
    grade_transparency_contract = (
        payload.get("grade_transparency_contract")
        if isinstance(payload.get("grade_transparency_contract"), dict)
        else {}
    )
    scout_collection_contract = (
        payload.get("scout_collection_contract")
        if isinstance(payload.get("scout_collection_contract"), dict)
        else {}
    )
    profit_harvest_controls = (
        payload.get("profit_harvest_profile_controls")
        if isinstance(payload.get("profit_harvest_profile_controls"), dict)
        else {}
    )
    profit_realization_contract = (
        payload.get("profit_realization_contract")
        if isinstance(payload.get("profit_realization_contract"), dict)
        else {}
    )
    profit_harvest_strategy_controls = (
        payload.get("profit_harvest_strategy_controls")
        if isinstance(payload.get("profit_harvest_strategy_controls"), dict)
        else {}
    )
    profit_harvest_position_ledger = (
        payload.get("profit_harvest_position_ledger")
        if isinstance(payload.get("profit_harvest_position_ledger"), dict)
        else {}
    )
    daily_sleeve_harvest_goal_contract = (
        payload.get("daily_sleeve_harvest_goal_contract")
        if isinstance(payload.get("daily_sleeve_harvest_goal_contract"), dict)
        else {}
    )
    paper_harvest_execution_contract = (
        payload.get("paper_harvest_execution_contract")
        if isinstance(payload.get("paper_harvest_execution_contract"), dict)
        else {}
    )
    daily_target_adaptation_contract = (
        payload.get("daily_target_adaptation_contract")
        if isinstance(payload.get("daily_target_adaptation_contract"), dict)
        else {}
    )
    paper_harvest_infrabot_contract = (
        payload.get("paper_harvest_infrabot_contract")
        if isinstance(payload.get("paper_harvest_infrabot_contract"), dict)
        else {}
    )
    profit_harvest_regret_replay_contract = (
        payload.get("profit_harvest_regret_replay_contract")
        if isinstance(payload.get("profit_harvest_regret_replay_contract"), dict)
        else {}
    )
    aggressive_harvest_mode_contract = (
        payload.get("aggressive_harvest_mode_contract")
        if isinstance(payload.get("aggressive_harvest_mode_contract"), dict)
        else {}
    )
    runner_protection_contract = (
        payload.get("runner_protection_contract")
        if isinstance(payload.get("runner_protection_contract"), dict)
        else {}
    )
    profit_rotation_contract = (
        payload.get("profit_rotation_contract")
        if isinstance(payload.get("profit_rotation_contract"), dict)
        else {}
    )
    profit_harvest_report_card = (
        payload.get("profit_harvest_report_card")
        if isinstance(payload.get("profit_harvest_report_card"), dict)
        else {}
    )
    profit_harvest_aplus_campaign = (
        payload.get("profit_harvest_aplus_campaign")
        if isinstance(payload.get("profit_harvest_aplus_campaign"), dict)
        else {}
    )
    grand_master_profit_harvest_awareness_contract = (
        payload.get("grand_master_profit_harvest_awareness_contract")
        if isinstance(payload.get("grand_master_profit_harvest_awareness_contract"), dict)
        else {}
    )
    max_grade_push_contract = (
        payload.get("max_grade_push_contract")
        if isinstance(payload.get("max_grade_push_contract"), dict)
        else {}
    )
    profitability_realization_expansion_contract = (
        payload.get("profitability_realization_expansion_contract")
        if isinstance(payload.get("profitability_realization_expansion_contract"), dict)
        else {}
    )
    profitability_compounding_autopilot_contract = (
        payload.get("profitability_compounding_autopilot_contract")
        if isinstance(payload.get("profitability_compounding_autopilot_contract"), dict)
        else {}
    )
    quant_strategy_expansion_admission_contract = (
        payload.get("quant_strategy_expansion_admission_contract")
        if isinstance(payload.get("quant_strategy_expansion_admission_contract"), dict)
        else {}
    )
    strategy_controls: dict[str, dict[str, Any]] = {}
    for row in strategies:
        if not isinstance(row, dict):
            continue
        profile = _normal_profile(row.get("profile"))
        strategy = str(row.get("strategy") or "").strip().lower()
        bot_id = _strategy_bot_id(strategy)
        if profile and strategy:
            strategy_controls[f"{profile}::{strategy}"] = row
        if profile and bot_id:
            strategy_controls[f"{profile}::{bot_id}"] = row
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "upgrade_lane_count": len(UPGRADE_LANE_IDS),
        "profitability_upgrade_lanes": upgrade_lanes,
        "a_plus_target_contract": a_plus_target_contract,
        "raw_operational_materiality_filter": raw_operational_materiality_filter,
        "raw_operational_containment_filter": raw_operational_containment_filter,
        "remaining_low_grade_layers": remaining_low_grade_layers,
        "low_grade_layer_summary": low_grade_layer_summary,
        "low_grade_control_report_card": low_grade_control_report_card,
        "grade_transparency_contract": grade_transparency_contract,
        "raw_operational_grade_lift_contract": raw_operational_grade_lift_contract,
        "profit_harvest_profile_controls": profit_harvest_controls,
        "profit_harvest_strategy_controls": profit_harvest_strategy_controls,
        "profit_harvest_position_ledger": profit_harvest_position_ledger,
        "daily_sleeve_harvest_goal_contract": daily_sleeve_harvest_goal_contract,
        "daily_target_adaptation_contract": daily_target_adaptation_contract,
        "paper_harvest_execution_contract": paper_harvest_execution_contract,
        "paper_harvest_infrabot_contract": paper_harvest_infrabot_contract,
        "profit_harvest_regret_replay_contract": profit_harvest_regret_replay_contract,
        "aggressive_harvest_mode_contract": aggressive_harvest_mode_contract,
        "runner_protection_contract": runner_protection_contract,
        "profit_rotation_contract": profit_rotation_contract,
        "profit_harvest_report_card": profit_harvest_report_card,
        "profit_harvest_aplus_campaign": profit_harvest_aplus_campaign,
        "grand_master_profit_harvest_awareness_contract": grand_master_profit_harvest_awareness_contract,
        "profit_realization_contract": profit_realization_contract,
        "max_grade_push_contract": max_grade_push_contract,
        "profitability_realization_expansion_contract": profitability_realization_expansion_contract,
        "profitability_compounding_autopilot_contract": profitability_compounding_autopilot_contract,
        "quant_strategy_expansion_admission_contract": quant_strategy_expansion_admission_contract,
        "paper_profitability_hardening_contract": hardening_contract,
        "scout_collection_contract": scout_collection_contract,
        "master_grandmaster_training_contract": upper_layer_training_contract,
        "sub_bot_accuracy_target_contract": SUB_BOT_ACCURACY_TARGET_CONTRACT,
        "global_runtime_policy": {
            "apply_outcome_weighted_training": True,
            "apply_per_sleeve_profit_score": True,
            "apply_dynamic_sizing": True,
            "apply_regime_specific_promotion": True,
            "apply_loser_quarantine": True,
            "apply_exit_intelligence": True,
            "apply_execution_aware_alpha": True,
            "apply_portfolio_conflict_control": True,
            "apply_confirmation_bias_control": True,
            "apply_profitability_hardening": True,
            "block_new_entries_on_quarantined_profiles": True,
            "apply_unrealized_drag_exit_acceleration": True,
            "apply_scout_collection_labels": True,
            "apply_a_plus_recovery_mode": bool(a_plus_target_contract.get("combined_control_a_plus_ready", False)),
            "lock_financial_a_plus_until_operational_a_plus": True,
            "apply_profit_realization": True,
            "block_adds_when_unrealized_profit_dominates": True,
            "promote_profit_harvest_trims": True,
            "apply_profit_harvest_intelligence": True,
            "apply_trend_continuation_holdback": True,
            "learn_profit_harvest_regret": True,
            "apply_strategy_profit_harvest": True,
            "apply_position_profit_harvest_ledger": True,
            "apply_profit_rotation": True,
            "apply_grand_master_profit_harvest_awareness": True,
            "apply_harvest_report_card_a_plus_campaign": True,
            "apply_harvest_report_card_a_plus_plus_campaign": True,
            "apply_raw_harvest_c_rescue": True,
            "apply_raw_harvest_grade_lift_contract": True,
            "apply_raw_operational_grade_lift_contract": True,
            "apply_daily_sleeve_harvest_targets": True,
            "apply_laddered_profit_harvest_exits": True,
            "emit_paper_harvest_reduce_intents": True,
            "block_new_adds_until_daily_realization_goal": True,
            "paper_harvest_intents_reduce_only": True,
            "apply_paper_harvest_infrabot_supervision": True,
            "apply_max_grade_push_contract": True,
            "apply_profitability_realization_expansion_contract": True,
            "apply_profitability_compounding_autopilot": True,
            "apply_quant_strategy_expansion_admission": True,
            "quant_strategy_expansion_collection_only_first": True,
            "block_quant_strategy_widening_while_protective_tightening": True,
            "apply_weak_sleeve_drag_stop": True,
            "apply_winning_sleeve_scaling": True,
            "apply_harvest_regret_lift": True,
            "apply_laddered_partial_exit_policy": True,
            "apply_strategy_level_promotion": True,
            "apply_punitive_loss_attribution": True,
            "apply_unrealized_loser_training_debt": True,
            "apply_harvest_force_guard": True,
            "follow_profitability_do_first_queue": True,
            "apply_daily_target_adaptation": True,
            "raise_daily_targets_after_previous_goal_met": True,
            "expand_collection_after_daily_target_met": True,
            "block_new_adds_until_raw_harvest_c_when_rescue_active": bool(
                ((profit_harvest_aplus_campaign.get("raw_c_rescue") or {}).get("active", False))
                if isinstance(profit_harvest_aplus_campaign.get("raw_c_rescue"), dict)
                else False
            ),
        },
        "profile_controls": profile_controls,
        "strategy_controls": strategy_controls,
        "recommended_refresh_command": [
            "./scripts/ops/opsctl.sh",
            "paper-profitability-control",
            "--apply",
            "--json",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert paper trading losses into paper-only profitability controls.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--control-out", default=str(DEFAULT_CONTROL_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root)
    if args.apply:
        control_payload = build_runtime_control_payload(payload)
        control_path = Path(args.control_out).expanduser()
        if not control_path.is_absolute():
            control_path = project_root / control_path
        write_payload(control_path, control_payload)
        payload["applied_runtime_control_file"] = str(control_path)
        payload["applied_runtime_control_summary"] = {
            "profile_control_count": len(control_payload.get("profile_controls") or {}),
            "strategy_control_count": len(control_payload.get("strategy_controls") or {}),
        }

    out_path = Path(args.out_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "paper_profitability_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"grade={payload.get('profitability_grade', '')} "
            f"profile_controls={payload.get('active_profile_control_count', 0)} "
            f"strategy_controls={payload.get('strategy_control_count', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
