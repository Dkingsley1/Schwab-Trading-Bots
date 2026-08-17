from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STRATEGY_WAVE: dict[str, set[str]] = {
    "options_flow": {
        "zero_dte_dealer_inventory_flip",
        "put_call_skew_surge",
        "unusual_sweep_confirmation",
        "iv_rank_event_filter",
        "gamma_expiry_pin_break",
    },
    "single_name_options_event": {
        "analyst_day_volatility_setup",
        "fda_adcom_event_convexity",
        "product_launch_gap_continuation",
        "activist_stake_disclosure_followthrough",
        "litigation_ruling_vol_reset",
    },
    "cash_rotation_tactical": {
        "ultra_short_duration_yield_switch",
        "cash_to_quality_reentry_ladder",
        "treasury_bill_auction_yield_capture",
        "liquidity_stress_cash_buffer",
        "defensive_cash_drag_optimizer",
    },
    "futures_index_intraday": {
        "index_futures_liquidity_vacuum",
        "opening_imbalance_futures_confirmation",
        "cash_session_breakout_alignment",
        "overnight_inventory_unwind",
        "futures_close_auction_positioning",
    },
    "futures_rates_curve": {
        "two_ten_curve_repricing_scout",
        "sofr_roll_yield_watch",
        "duration_convexity_squeeze",
        "fed_speaker_curve_impulse",
        "treasury_refunding_steepener_watch",
    },
    "futures_commodity_macro": {
        "crude_inventory_surprise_reaction",
        "natural_gas_weather_delta",
        "gold_real_yield_divergence",
        "copper_dollar_growth_proxy",
        "commodity_curve_backwardation_watch",
    },
    "crypto_futures_basis": {
        "crypto_calendar_basis_ladder",
        "perp_funding_crowding_score",
        "etf_premium_basis_bridge",
        "liquidation_heatmap_reversal",
        "cross_exchange_basis_dislocation",
    },
    "futures_event_reaction": {
        "cpi_release_futures_impulse",
        "fomc_press_conference_second_move",
        "nfp_whipsaw_reversion",
        "treasury_auction_tail_followthrough",
        "geopolitical_gap_fade_or_follow",
    },
    "position_lifecycle": {
        "partial_profit_ladder_optimizer",
        "trailing_stop_volatility_band",
        "position_age_decay_exit",
        "add_on_pullback_quality",
        "rebalance_tax_lot_priority",
    },
    "execution_quality": {
        "venue_fill_quality_ranker",
        "order_type_slippage_attribution",
        "partial_fill_patience_model",
        "spread_crossing_cost_guard",
        "queue_delay_execution_penalty",
    },
}


def test_strategy_manifest_has_first_1000_path_wave() -> None:
    config = json.loads((PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json").read_text(encoding="utf-8"))
    sleeves = {row["name"]: set(row["strategies"]) for row in config["sleeves"]}

    assert sum(len(strategies) for strategies in STRATEGY_WAVE.values()) == 50
    assert sum(len(row["strategies"]) for row in config["sleeves"]) >= 760

    for sleeve, expected_strategies in STRATEGY_WAVE.items():
        assert expected_strategies <= sleeves[sleeve]
