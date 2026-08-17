from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STRATEGY_BATCH: dict[str, set[str]] = {
    "bond_rates": {
        "real_yield_breakout",
        "breakeven_inflation_rotation",
        "belly_curve_butterfly_watch",
        "credit_quality_migration_proxy",
        "treasury_auction_tail_risk",
    },
    "fx_macro": {
        "cross_currency_basis_stress",
        "dollar_smile_regime_switch",
        "carry_unwind_alert",
        "reserve_currency_rotation",
        "commodity_terms_of_trade_fx",
    },
    "crypto_spot": {
        "spot_etf_flow_impulse",
        "stablecoin_liquidity_expansion",
        "exchange_reserve_drawdown",
        "bitcoin_dominance_rotation",
        "onchain_risk_appetite_proxy",
    },
    "crypto_futures": {
        "funding_term_structure_kink",
        "open_interest_liquidation_ladder",
        "perp_spot_basis_compression",
        "weekend_gap_risk_premium",
        "options_expiry_max_pain_crypto",
    },
    "schwab_futures": {
        "futures_opening_drive_context",
        "futures_vwap_acceptance_rejection",
        "futures_basis_spread_watch",
        "futures_globex_range_breakout",
        "futures_macro_liquidity_sweep",
    },
    "commodity_inflation": {
        "refinery_crack_margin_proxy",
        "grains_weather_shock_rotation",
        "copper_growth_inflation_divergence",
        "uranium_supply_squeeze_watch",
        "shipping_freight_inflation_proxy",
    },
    "international_macro": {
        "japan_yen_equity_feedback",
        "china_credit_impulse_proxy",
        "europe_energy_margin_rotation",
        "em_fx_reserve_stress",
        "global_real_rate_dispersion",
    },
    "market_making_liquidity": {
        "queue_position_decay_watch",
        "spread_regime_shift_detector",
        "dark_pool_print_absorption",
        "lit_venue_depth_imbalance",
        "microprice_reversion_quality",
    },
    "short_bias_hedge": {
        "crowded_long_unwind_detector",
        "negative_revision_breakdown",
        "weak_balance_sheet_momentum_short",
        "beta_hedge_rebalance_trigger",
        "bear_flag_failure_continuation",
    },
    "rates_credit_macro": {
        "ig_hy_dispersion_stress",
        "swap_spread_dislocation_watch",
        "sofr_futures_policy_path_shift",
        "bank_cds_equity_divergence",
        "credit_curve_inversion_alert",
    },
    "sector_master": {
        "sector_volatility_risk_budget",
        "sector_earnings_revision_breadth",
        "sector_factor_crowding_unwind",
        "sector_defensive_cyclical_barbell",
        "sector_liquidity_lead_lag",
    },
}


def test_strategy_manifest_has_exact_55_strategy_expansion_batch() -> None:
    config = json.loads((PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json").read_text(encoding="utf-8"))
    sleeves = {row["name"]: set(row["strategies"]) for row in config["sleeves"]}

    assert sum(len(strategies) for strategies in STRATEGY_BATCH.values()) == 55
    assert sum(len(row["strategies"]) for row in config["sleeves"]) >= 710

    for sleeve, expected_strategies in STRATEGY_BATCH.items():
        assert expected_strategies <= sleeves[sleeve]
