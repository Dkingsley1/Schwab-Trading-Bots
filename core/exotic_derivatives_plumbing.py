from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping


EXOTIC_DERIVATIVE_SLEEVES = {
    "compound_options",
    "swaptions",
    "structured_products",
    "synthetic_cdo",
    "cdo_squared",
    "cdo_cubed",
    "variance_volatility_swaps",
    "barrier_lookback_options",
    "second_third_order_greeks",
    "high_frequency_market_making",
    "tail_risk_parity",
    "black_swan_hedging",
    "sovereign_debt_macro",
    "gamma_scalping",
    "statistical_arbitrage",
    "vanna_volga_hedging",
    "order_flow_market_microstructure",
    "dispersion_trading",
    "cross_asset_basis_training",
    "volatility_arbitrage",
    "advanced_derivatives_infrastructure",
    "rainbow_options",
}

SCHWAB_LISTED_INPUTS = (
    "listed_equity_etf_quotes",
    "listed_option_chains",
    "listed_futures_quotes_when_entitled",
)

PUBLIC_MACRO_INPUTS = ("fred_macro_series", "treasury_yield_curve", "sofr_rates")
PUBLIC_CREDIT_INPUTS = ("finra_trace_aggregates", "credit_etf_quotes", "rates_credit_etf_quotes")
PUBLIC_FILING_INPUTS = ("sec_edgar_filings", "structured_note_prospectus_text")
PUBLIC_OPTIONS_INPUTS = ("occ_options_volume_open_interest",)
PUBLIC_VOL_INPUTS = ("cboe_volatility_indices", "occ_option_chain_aggregates", "listed_volatility_etps")
PUBLIC_MICROSTRUCTURE_INPUTS = ("finra_ats_blocks", "exchange_breadth_quotes", "public_sip_quote_proxies")
PUBLIC_SOVEREIGN_INPUTS = ("treasury_auction_calendar", "sovereign_curve_etfs", "central_bank_calendar")

EXOTIC_PROVIDER_CAPABILITY_MATRIX: dict[str, dict[str, Any]] = {
    "compound_options": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_option_chains", "listed_equity_etf_quotes"),
        "public_inputs": PUBLIC_OPTIONS_INPUTS + PUBLIC_MACRO_INPUTS,
        "proxy_mode": "listed_options_vol_surface_proxy",
        "notes": "Model option-on-option convexity from listed chain surfaces; do not route compound-option orders.",
    },
    "swaptions": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_futures_quotes_when_entitled", "rates_etf_quotes"),
        "public_inputs": PUBLIC_MACRO_INPUTS + PUBLIC_CREDIT_INPUTS,
        "proxy_mode": "rates_curve_vol_proxy",
        "notes": "Model swaption exposure from Treasury futures, rates ETFs, SOFR, and curve movement.",
    },
    "structured_products": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_equity_etf_quotes", "listed_option_chains"),
        "public_inputs": PUBLIC_FILING_INPUTS + PUBLIC_CREDIT_INPUTS + PUBLIC_OPTIONS_INPUTS,
        "proxy_mode": "payoff_barrier_credit_proxy",
        "notes": "Use issuer filings, credit proxies, listed options, and buffered/covered-call ETP analogs.",
    },
    "synthetic_cdo": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("credit_etf_quotes", "financial_sector_quotes"),
        "public_inputs": PUBLIC_CREDIT_INPUTS + PUBLIC_MACRO_INPUTS,
        "proxy_mode": "credit_tranche_stress_proxy",
        "notes": "Use credit ETF spreads, TRACE aggregates, rates stress, and default-correlation proxies.",
    },
    "cdo_squared": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("credit_etf_quotes", "financial_sector_quotes", "listed_equity_etf_quotes"),
        "public_inputs": PUBLIC_CREDIT_INPUTS + PUBLIC_MACRO_INPUTS + PUBLIC_FILING_INPUTS,
        "proxy_mode": "second_order_credit_tranche_proxy",
        "notes": "Model CDO-squared stress from credit-tranche, issuer, and macro proxies only; do not route OTC structured-credit orders.",
    },
    "cdo_cubed": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("credit_etf_quotes", "financial_sector_quotes", "listed_equity_etf_quotes"),
        "public_inputs": PUBLIC_CREDIT_INPUTS + PUBLIC_MACRO_INPUTS + PUBLIC_FILING_INPUTS,
        "proxy_mode": "third_order_credit_tranche_tail_proxy",
        "notes": "Treat CDO-cubed behavior as a tail-risk research sleeve fed by credit contagion and macro stress proxies.",
    },
    "variance_volatility_swaps": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_option_chains", "listed_volatility_etps", "listed_futures_quotes_when_entitled"),
        "public_inputs": PUBLIC_OPTIONS_INPUTS + PUBLIC_VOL_INPUTS + PUBLIC_MACRO_INPUTS,
        "proxy_mode": "listed_options_variance_vol_swap_proxy",
        "notes": "Estimate variance and volatility swap behavior from listed option surfaces, realized-vol proxies, and CBOE-style vol inputs.",
    },
    "barrier_lookback_options": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_option_chains", "listed_equity_etf_quotes", "listed_futures_quotes_when_entitled"),
        "public_inputs": PUBLIC_OPTIONS_INPUTS + PUBLIC_VOL_INPUTS + PUBLIC_MACRO_INPUTS,
        "proxy_mode": "path_dependent_barrier_lookback_proxy",
        "notes": "Use listed options, price-path, barrier-distance, and realized-vol proxies; no direct barrier or lookback routing.",
    },
    "second_third_order_greeks": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_option_chains", "listed_equity_etf_quotes"),
        "public_inputs": PUBLIC_OPTIONS_INPUTS + PUBLIC_VOL_INPUTS,
        "proxy_mode": "higher_order_greeks_surface_proxy",
        "notes": "Track vanna, charm, vomma, speed, color, zomma, and ultima as research features around listed-option surfaces.",
    },
    "high_frequency_market_making": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_equity_etf_quotes", "listed_option_chains"),
        "public_inputs": PUBLIC_MICROSTRUCTURE_INPUTS + PUBLIC_OPTIONS_INPUTS,
        "proxy_mode": "hft_microstructure_research_proxy",
        "notes": "Collect spread, quote-fade, imbalance, and fill-quality proxies only; do not enable HFT routing.",
    },
    "tail_risk_parity": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_equity_etf_quotes", "listed_option_chains", "rates_credit_etf_quotes"),
        "public_inputs": PUBLIC_VOL_INPUTS + PUBLIC_CREDIT_INPUTS + PUBLIC_MACRO_INPUTS,
        "proxy_mode": "tail_risk_parity_overlay_proxy",
        "notes": "Balance sleeve stress, volatility, credit, and rates proxies before any risk-parity-style hedging research is trusted.",
    },
    "black_swan_hedging": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_equity_etf_quotes", "listed_option_chains", "listed_volatility_etps"),
        "public_inputs": PUBLIC_VOL_INPUTS + PUBLIC_CREDIT_INPUTS + PUBLIC_MACRO_INPUTS + PUBLIC_OPTIONS_INPUTS,
        "proxy_mode": "crash_convexity_hedging_proxy",
        "notes": "Model crash-convexity and hedge-cost conditions from listed, public, and macro proxies only.",
    },
    "sovereign_debt_macro": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("rates_credit_etf_quotes", "listed_futures_quotes_when_entitled", "listed_equity_etf_quotes"),
        "public_inputs": PUBLIC_SOVEREIGN_INPUTS + PUBLIC_MACRO_INPUTS + PUBLIC_CREDIT_INPUTS,
        "proxy_mode": "sovereign_curve_macro_proxy",
        "notes": "Use Treasury, sovereign ETF, central-bank, FX, and credit proxies rather than direct sovereign OTC instruments.",
    },
    "gamma_scalping": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_option_chains", "listed_equity_etf_quotes", "listed_futures_quotes_when_entitled"),
        "public_inputs": PUBLIC_OPTIONS_INPUTS + PUBLIC_VOL_INPUTS + PUBLIC_MICROSTRUCTURE_INPUTS,
        "proxy_mode": "listed_options_gamma_scalping_research_proxy",
        "notes": "Collect gamma-scalping pressure and hedge-cost labels from listed options; do not route automated gamma hedges.",
    },
    "statistical_arbitrage": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_equity_etf_quotes", "listed_option_chains"),
        "public_inputs": PUBLIC_MICROSTRUCTURE_INPUTS + PUBLIC_MACRO_INPUTS,
        "proxy_mode": "stat_arb_residual_reversion_research_proxy",
        "notes": "Use residual, factor, pair-spread, and microstructure proxies in collection-only mode until walk-forward gates approve.",
    },
    "vanna_volga_hedging": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_option_chains", "listed_equity_etf_quotes"),
        "public_inputs": PUBLIC_OPTIONS_INPUTS + PUBLIC_VOL_INPUTS,
        "proxy_mode": "vanna_volga_surface_hedging_proxy",
        "notes": "Use higher-order Greek proxies from listed option surfaces; do not route automated vanna-volga hedges.",
    },
    "order_flow_market_microstructure": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_equity_etf_quotes", "listed_option_chains", "listed_futures_quotes_when_entitled"),
        "public_inputs": PUBLIC_MICROSTRUCTURE_INPUTS + PUBLIC_OPTIONS_INPUTS,
        "proxy_mode": "order_flow_microstructure_research_proxy",
        "notes": "Collect order-flow, imbalance, sweep, quote-fade, and spread proxies only; no automated microstructure execution.",
    },
    "dispersion_trading": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_option_chains", "listed_equity_etf_quotes"),
        "public_inputs": PUBLIC_OPTIONS_INPUTS + PUBLIC_VOL_INPUTS + PUBLIC_MACRO_INPUTS,
        "proxy_mode": "index_single_name_dispersion_proxy",
        "notes": "Model dispersion from index/single-name option surfaces and correlation proxies without routing dispersion packages.",
    },
    "cross_asset_basis_training": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_equity_etf_quotes", "listed_futures_quotes_when_entitled", "rates_credit_etf_quotes"),
        "public_inputs": PUBLIC_MACRO_INPUTS + PUBLIC_CREDIT_INPUTS + PUBLIC_MICROSTRUCTURE_INPUTS,
        "proxy_mode": "cross_asset_basis_training_proxy",
        "notes": "Train on ETF/futures/FX/rates/crypto basis relationships through proxy inputs before any allocation exposure.",
    },
    "volatility_arbitrage": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_option_chains", "listed_volatility_etps", "listed_futures_quotes_when_entitled"),
        "public_inputs": PUBLIC_OPTIONS_INPUTS + PUBLIC_VOL_INPUTS + PUBLIC_MACRO_INPUTS,
        "proxy_mode": "listed_options_volatility_arbitrage_proxy",
        "notes": "Collect IV/RV, skew, term-structure, and hedge-cost signals only until vol-arbitrage training gates pass.",
    },
    "advanced_derivatives_infrastructure": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_equity_etf_quotes", "listed_option_chains", "listed_futures_quotes_when_entitled"),
        "public_inputs": PUBLIC_OPTIONS_INPUTS + PUBLIC_VOL_INPUTS + PUBLIC_MACRO_INPUTS + PUBLIC_CREDIT_INPUTS + PUBLIC_MICROSTRUCTURE_INPUTS,
        "proxy_mode": "advanced_derivatives_regression_guard_proxy",
        "notes": "Infrastructure lane for feature/schema/storage/report regression checks across advanced derivative sleeves.",
    },
    "rainbow_options": {
        "direct_product_market_data": False,
        "direct_execution_allowed": False,
        "schwab_direct_inputs": ("listed_equity_etf_quotes", "listed_option_chains", "listed_futures_quotes_when_entitled"),
        "public_inputs": PUBLIC_OPTIONS_INPUTS + PUBLIC_MACRO_INPUTS + PUBLIC_VOL_INPUTS,
        "proxy_mode": "cross_asset_correlation_dispersion_proxy",
        "notes": "Use multi-underlier listed proxies, correlations, dispersion, and vol-surface context.",
    },
}

EXOTIC_DERIVATIVE_FEATURE_KEYS = [
    "exotic_sleeve_active_norm",
    "exotic_direct_market_data_available_norm",
    "exotic_proxy_mode_norm",
    "exotic_direct_execution_allowed_norm",
    "exotic_direct_execution_blocked_norm",
    "exotic_schwab_listed_input_norm",
    "exotic_public_macro_input_norm",
    "exotic_public_credit_input_norm",
    "exotic_public_filing_input_norm",
    "exotic_public_options_input_norm",
    "exotic_public_source_count_norm",
    "exotic_proxy_rates_curve_norm",
    "exotic_proxy_credit_stress_norm",
    "exotic_proxy_vol_surface_norm",
    "exotic_proxy_correlation_dispersion_norm",
    "exotic_proxy_structured_payoff_norm",
    "exotic_proxy_variance_vol_swap_norm",
    "exotic_proxy_barrier_path_norm",
    "exotic_proxy_higher_order_greeks_norm",
    "exotic_proxy_tail_risk_norm",
    "exotic_proxy_market_microstructure_norm",
    "exotic_proxy_sovereign_macro_norm",
    "exotic_proxy_tranche_correlation_norm",
    "exotic_proxy_gamma_scalping_norm",
    "exotic_proxy_order_flow_norm",
    "exotic_proxy_stat_arb_norm",
    "exotic_proxy_cross_asset_basis_norm",
    "exotic_proxy_vol_arbitrage_norm",
    "exotic_proxy_rough_vvix_norm",
    "exotic_proxy_quantum_barrier_path_norm",
    "exotic_proxy_correlation_heat_swap_norm",
    "exotic_proxy_cliquet_floor_cap_norm",
    "exotic_proxy_signature_trend_follower_norm",
    "exotic_proxy_esg_ccds_norm",
    "exotic_data_confidence_norm",
    "exotic_proxy_only_guard_norm",
]


def _clamp01(value: Any) -> float:
    try:
        f = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return max(0.0, min(f, 1.0))


def _feature(features: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in features:
            return _clamp01(features.get(key))
    return _clamp01(default)


def normalize_exotic_sleeve(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def is_exotic_derivative_sleeve(value: Any) -> bool:
    return normalize_exotic_sleeve(value) in EXOTIC_DERIVATIVE_SLEEVES


def provider_capabilities_for_sleeve(sleeve: Any) -> dict[str, Any]:
    normalized = normalize_exotic_sleeve(sleeve)
    row = EXOTIC_PROVIDER_CAPABILITY_MATRIX.get(normalized)
    return deepcopy(row) if isinstance(row, dict) else {}


def exotic_direct_execution_allowed(sleeve: Any, *, broker: str = "schwab") -> bool:
    if not is_exotic_derivative_sleeve(sleeve):
        return True
    caps = provider_capabilities_for_sleeve(sleeve)
    if str(broker or "").strip().lower() == "schwab":
        return False
    return bool(caps.get("direct_execution_allowed", False))


def default_exotic_derivative_features() -> dict[str, float]:
    return {key: 0.0 for key in EXOTIC_DERIVATIVE_FEATURE_KEYS}


def summarize_exotic_derivative_proxy_features(
    *,
    sleeve: Any,
    broker: str,
    features: Mapping[str, Any],
    external_snapshots: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Build research-only exotic derivative features from listed/proxy inputs."""
    out = default_exotic_derivative_features()
    normalized = normalize_exotic_sleeve(sleeve)
    if normalized not in EXOTIC_DERIVATIVE_SLEEVES:
        return out

    caps = provider_capabilities_for_sleeve(normalized)
    public_inputs = tuple(caps.get("public_inputs") or ())
    schwab_inputs = tuple(caps.get("schwab_direct_inputs") or ())
    direct_market_data = bool(caps.get("direct_product_market_data", False))
    direct_execution = exotic_direct_execution_allowed(normalized, broker=broker)

    options_surface = max(
        _feature(features, "options_chain_available"),
        _feature(features, "options_iv_atm_norm"),
        _feature(features, "options_iv_skew_norm"),
        _feature(features, "options_iv_term_structure_norm"),
        _feature(features, "tasty_iv_rank_norm"),
        _feature(features, "tasty_implied_volatility_index_norm"),
    )
    rates_curve = max(
        _feature(features, "sofr_term_pressure_norm"),
        _feature(features, "sofr_funding_stress_norm"),
        _feature(features, "futures_term_structure_norm"),
        _feature(features, "futures_calendar_spread_curve_norm"),
        _feature(features, "bond_duration_pressure_norm"),
        _feature(features, "calendar_fomc_event_norm"),
    )
    credit_stress = max(
        _feature(features, "credit_spread_stress_norm"),
        _feature(features, "credit_context_stress_norm"),
        _feature(features, "short_borrow_fee_norm"),
        _feature(features, "sec_financing_stress_7d_norm"),
        _feature(features, "cboe_put_call_stress_norm"),
        _feature(features, "live_macro_risk_off_pressure_norm"),
    )
    correlation_dispersion = max(
        _feature(features, "market_crypto_sleeve_dispersion_norm"),
        _feature(features, "fx_macro_dispersion_norm"),
        _feature(features, "pairs_correlation_break_norm"),
        _feature(features, "lead_lag_break_norm"),
        _feature(features, "regime_dislocation_norm"),
    )
    structured_payoff = max(
        _feature(features, "sec_filing_count_7d_norm"),
        _feature(features, "sec_high_impact_7d_norm"),
        _feature(features, "etf_nav_premium_discount_norm"),
        _feature(features, "etf_creation_redemption_stress_norm"),
        _feature(features, "options_assignment_risk_norm"),
    )
    variance_vol_swap = max(
        _feature(features, "options_variance_swap_proxy_norm"),
        _feature(features, "options_volatility_swap_proxy_norm"),
        _feature(features, "options_vol_of_vol_change_norm"),
        _feature(features, "options_iv_realized_spread_norm"),
        _feature(features, "cboe_vix_spot_norm"),
        _feature(features, "tasty_iv_rank_norm"),
    )
    barrier_path = max(
        _feature(features, "options_barrier_touch_risk_norm"),
        _feature(features, "options_lookback_path_dependency_norm"),
        _feature(features, "options_gamma_flip_distance_norm"),
        _feature(features, "options_strike_expiry_concentration_change_norm"),
        structured_payoff,
    )
    higher_order_greeks = max(
        _feature(features, "options_higher_order_greek_pressure_norm"),
        _feature(features, "options_vanna_volga_hedge_pressure_norm"),
        _feature(features, "options_vanna_mean_norm"),
        _feature(features, "options_charm_abs_mean_norm"),
        _feature(features, "options_vomma_mean_norm"),
        _feature(features, "options_speed_abs_mean_norm"),
        _feature(features, "options_color_abs_mean_norm"),
        _feature(features, "options_zomma_abs_mean_norm"),
        _feature(features, "options_ultima_abs_mean_norm"),
    )
    tail_risk = max(
        credit_stress,
        variance_vol_swap,
        _feature(features, "cboe_put_call_stress_norm"),
        _feature(features, "options_negative_bias_norm"),
        _feature(features, "short_borrow_fee_norm"),
        _feature(features, "live_macro_risk_off_pressure_norm"),
        _feature(features, "regime_dislocation_norm"),
    )
    market_microstructure = max(
        _feature(features, "market_micro_range_expansion_norm"),
        _feature(features, "market_micro_spread_quality_norm"),
        _feature(features, "market_micro_quote_fade_norm"),
        _feature(features, "bid_ask_spread_quality_norm"),
        _feature(features, "futures_order_book_imbalance_norm"),
        _feature(features, "options_spread_execution_risk_norm"),
    )
    sovereign_macro = max(
        rates_curve,
        _feature(features, "calendar_treasury_auction_norm"),
        _feature(features, "calendar_macro_surprise_norm"),
        _feature(features, "fx_macro_dispersion_norm"),
        _feature(features, "usd_funding_stress_norm"),
        _feature(features, "international_macro_sovereign_stress_norm"),
    )
    tranche_correlation = max(
        credit_stress,
        correlation_dispersion,
        _feature(features, "credit_correlation_stress_norm"),
        _feature(features, "default_correlation_surface_norm"),
        _feature(features, "correlation_cluster_risk_norm"),
    )
    gamma_scalping = max(
        _feature(features, "options_gamma_scalping_pressure_norm"),
        _feature(features, "options_gamma_front_share_norm"),
        _feature(features, "options_gamma_expiry_skew_norm"),
        _feature(features, "options_zero_dte_regime_norm"),
        market_microstructure,
    )
    order_flow = max(
        market_microstructure,
        _feature(features, "options_sweep_flow_norm"),
        _feature(features, "options_block_flow_norm"),
        _feature(features, "tasty_dealer_gamma_pressure_norm"),
        _feature(features, "futures_taker_imbalance_norm"),
        _feature(features, "market_micro_order_flow_imbalance_norm"),
    )
    stat_arb = max(
        correlation_dispersion,
        _feature(features, "pairs_correlation_break_norm"),
        _feature(features, "stat_arb_residual_zscore_norm"),
        _feature(features, "factor_residual_dispersion_norm"),
        _feature(features, "market_neutral_drawdown_guard_norm"),
    )
    cross_asset_basis = max(
        correlation_dispersion,
        rates_curve,
        _feature(features, "futures_basis_bps_norm"),
        _feature(features, "futures_basis_divergence_norm"),
        _feature(features, "crypto_deribit_basis_norm"),
        _feature(features, "etf_nav_premium_discount_norm"),
    )
    vol_arbitrage = max(
        variance_vol_swap,
        _feature(features, "options_volatility_arbitrage_proxy_norm"),
        _feature(features, "options_dispersion_trade_proxy_norm"),
        _feature(features, "options_vanna_volga_hedge_pressure_norm"),
        _feature(features, "options_iv_realized_spread_norm"),
    )
    rough_vvix = max(
        variance_vol_swap,
        _feature(features, "vvix_stress_norm"),
        _feature(features, "vix_on_vix_exotic_proxy_norm"),
        _feature(features, "vol_of_vol_stress_norm"),
        _feature(features, "rough_volatility_swap_proxy_norm"),
    )
    quantum_barrier_path = max(
        barrier_path,
        _feature(features, "quantum_barrier_path_amplitude_norm"),
        _feature(features, "path_amplitude_barrier_proxy_norm"),
        _feature(features, "quantum_enhanced_mc_trace_norm"),
    )
    correlation_heat_swap = max(
        correlation_dispersion,
        tranche_correlation,
        _feature(features, "cross_asset_correlation_heat_norm"),
        _feature(features, "correlation_heat_swap_proxy_norm"),
    )
    cliquet_floor_cap = max(
        structured_payoff,
        barrier_path,
        _feature(features, "cliquet_global_floor_local_cap_norm"),
        _feature(features, "coupon_barrier_stress_norm"),
        _feature(features, "floor_cap_distance_norm"),
    )
    signature_trend_follower = max(
        _feature(features, "signature_trend_follower_options_norm"),
        _feature(features, "signature_lead_lag_detector_norm"),
        _feature(features, "trend_persistence_norm"),
        correlation_dispersion,
    )
    esg_ccds = max(
        credit_stress,
        _feature(features, "esg_contingent_cds_norm"),
        _feature(features, "issuer_esg_event_norm"),
        _feature(features, "esg_controversy_stress_norm"),
    )

    public_source_score = 0.0
    snapshots = external_snapshots if isinstance(external_snapshots, Mapping) else {}
    for name in (
        "official_macro",
        "tradingeconomics",
        "sec_edgar",
        "bond_reference",
        "options_flow",
        "market_breadth",
        "treasury",
        "fred",
        "cboe",
        "finra",
        "occ",
    ):
        payload = snapshots.get(name)
        if isinstance(payload, Mapping) and payload:
            public_source_score += 1.0

    out.update(
        {
            "exotic_sleeve_active_norm": 1.0,
            "exotic_direct_market_data_available_norm": 1.0 if direct_market_data else 0.0,
            "exotic_proxy_mode_norm": 0.0 if direct_market_data else 1.0,
            "exotic_direct_execution_allowed_norm": 1.0 if direct_execution else 0.0,
            "exotic_direct_execution_blocked_norm": 0.0 if direct_execution else 1.0,
            "exotic_schwab_listed_input_norm": 1.0 if schwab_inputs else 0.0,
            "exotic_public_macro_input_norm": 1.0 if any(item in public_inputs for item in PUBLIC_MACRO_INPUTS) else 0.0,
            "exotic_public_credit_input_norm": 1.0 if any(item in public_inputs for item in PUBLIC_CREDIT_INPUTS) else 0.0,
            "exotic_public_filing_input_norm": 1.0 if any(item in public_inputs for item in PUBLIC_FILING_INPUTS) else 0.0,
            "exotic_public_options_input_norm": 1.0 if any(item in public_inputs for item in PUBLIC_OPTIONS_INPUTS) else 0.0,
            "exotic_public_source_count_norm": _clamp01(public_source_score / 11.0),
            "exotic_proxy_rates_curve_norm": rates_curve,
            "exotic_proxy_credit_stress_norm": credit_stress,
            "exotic_proxy_vol_surface_norm": options_surface,
            "exotic_proxy_correlation_dispersion_norm": correlation_dispersion,
            "exotic_proxy_structured_payoff_norm": structured_payoff,
            "exotic_proxy_variance_vol_swap_norm": variance_vol_swap,
            "exotic_proxy_barrier_path_norm": barrier_path,
            "exotic_proxy_higher_order_greeks_norm": higher_order_greeks,
            "exotic_proxy_tail_risk_norm": tail_risk,
            "exotic_proxy_market_microstructure_norm": market_microstructure,
            "exotic_proxy_sovereign_macro_norm": sovereign_macro,
            "exotic_proxy_tranche_correlation_norm": tranche_correlation,
            "exotic_proxy_gamma_scalping_norm": gamma_scalping,
            "exotic_proxy_order_flow_norm": order_flow,
            "exotic_proxy_stat_arb_norm": stat_arb,
            "exotic_proxy_cross_asset_basis_norm": cross_asset_basis,
            "exotic_proxy_vol_arbitrage_norm": vol_arbitrage,
            "exotic_proxy_rough_vvix_norm": rough_vvix,
            "exotic_proxy_quantum_barrier_path_norm": quantum_barrier_path,
            "exotic_proxy_correlation_heat_swap_norm": correlation_heat_swap,
            "exotic_proxy_cliquet_floor_cap_norm": cliquet_floor_cap,
            "exotic_proxy_signature_trend_follower_norm": signature_trend_follower,
            "exotic_proxy_esg_ccds_norm": esg_ccds,
            "exotic_proxy_only_guard_norm": 1.0,
        }
    )

    relevant_proxy_values = [
        out["exotic_schwab_listed_input_norm"],
        out["exotic_public_macro_input_norm"],
        out["exotic_public_credit_input_norm"],
        out["exotic_public_filing_input_norm"],
        out["exotic_public_options_input_norm"],
        rates_curve,
        credit_stress,
        options_surface,
        correlation_dispersion,
        structured_payoff,
        variance_vol_swap,
        barrier_path,
        higher_order_greeks,
        tail_risk,
        market_microstructure,
        sovereign_macro,
        tranche_correlation,
        gamma_scalping,
        order_flow,
        stat_arb,
        cross_asset_basis,
        vol_arbitrage,
        rough_vvix,
        quantum_barrier_path,
        correlation_heat_swap,
        cliquet_floor_cap,
        signature_trend_follower,
        esg_ccds,
    ]
    out["exotic_data_confidence_norm"] = _clamp01(sum(relevant_proxy_values) / max(len(relevant_proxy_values), 1))
    return out
