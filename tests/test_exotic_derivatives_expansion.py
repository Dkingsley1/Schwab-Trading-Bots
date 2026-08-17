import json
from pathlib import Path

import scripts.run_all_sleeves as run_all_sleeves
import scripts.run_specialized_sleeve_shadow as specialized
from scripts.ops import roster_expansion_slots


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MORE_EXOTIC_BOT_INTAKES = {
    "brain_refinery_v1466_exotic_forward_start_skew_reset_bot": "forward_start_option_surface",
    "brain_refinery_v1467_exotic_chooser_option_event_optionality_bot": "chooser_option_switch_value",
    "brain_refinery_v1468_exotic_asian_path_average_option_bot": "asian_path_average_surface",
    "brain_refinery_v1469_exotic_digital_binary_event_risk_bot": "digital_binary_event_risk",
    "brain_refinery_v1470_exotic_corridor_variance_realized_range_bot": "corridor_variance_realized_range",
    "brain_refinery_v1471_exotic_quanto_fx_equity_vol_beta_bot": "quanto_fx_equity_beta",
    "brain_refinery_v1472_exotic_worst_of_airbag_autocall_bot": "airbag_autocall_barrier_ladder",
    "brain_refinery_v1473_exotic_recovery_lock_credit_note_guard_bot": "recovery_lock_credit_note",
    "brain_refinery_v1474_exotic_participation_ratchet_protection_bot": "capital_protection_participation_ratchet",
    "brain_refinery_v1475_exotic_base_correlation_convexity_bot": "base_correlation_convexity",
    "brain_refinery_v1476_exotic_nth_to_default_contagion_ladder_bot": "nth_to_default_contagion_ladder",
    "brain_refinery_v1477_exotic_gap_option_jump_risk_bot": "overnight_gap_option_jump",
}

EXOTIC_SLEEVES = {
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
    "rainbow_options",
}


def test_exotic_derivative_sleeves_are_registered_for_collection() -> None:
    for sleeve in EXOTIC_SLEEVES:
        assert sleeve in specialized.SLEEVE_DEFAULTS
        assert sleeve in run_all_sleeves.SPECIALIZED_SLEEVE_PROFILES
        assert specialized.SLEEVE_DEFAULTS[sleeve]["domain"] == "exotic_derivatives"
        assert specialized.SLEEVE_DEFAULTS[sleeve]["correlation_peers"]


def test_exotic_derivative_bots_are_collection_first_and_correlated() -> None:
    specs = [
        row
        for row in roster_expansion_slots.DEFAULT_SLOT_SPECS
        if str(row.get("sleeve_profile")) in EXOTIC_SLEEVES
        or str(row.get("sleeve_profile")) == "advanced_derivatives_infrastructure"
    ]

    sleeve_counts: dict[str, int] = {}
    for spec in specs:
        sleeve = str(spec.get("sleeve_profile"))
        sleeve_counts[sleeve] = sleeve_counts.get(sleeve, 0) + 1
        row = roster_expansion_slots._slot_registry_row(spec)
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["allocation_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["minimum_training_observations"] >= 2500
        assert row["minimum_data_collection_days"] >= 14
        assert row["sleeve_family"] == "exotic_derivatives"
        assert row["direct_market_data_available"] is False
        assert row["direct_execution_allowed"] is False
        assert row["provider_capability_profile"]
        assert row["proxy_data_sources"]
        assert row["schwab_direct_inputs"]
        assert "cross_sleeve_correlation_matrix" in row["data_intake_collections"]
        assert "correlation_risk_surface" in row["data_intake_collections"]
        assert "governance/exotic_derivatives" in row["storage_targets"]
        assert row["correlation_peer_sleeves"]
        assert row["correlation_dependencies"]

    for sleeve in EXOTIC_SLEEVES:
        assert sleeve_counts[sleeve] >= 5
    assert sleeve_counts["advanced_derivatives_infrastructure"] == 1
    guard = roster_expansion_slots._slot_registry_row(
        next(row for row in specs if str(row.get("bot_id")) == "brain_refinery_v438_advanced_derivatives_data_regression_guard_bot")
    )
    assert guard["bot_role"] == "infrastructure_bot"
    assert "advanced_derivatives_feature_schema_guard" in guard["data_intake_collections"]
    assert "strategy_inventory_report_guard" in guard["data_intake_collections"]


def test_more_exotic_bots_are_proxy_only_collection_slots() -> None:
    specs = {str(row.get("bot_id")): row for row in roster_expansion_slots.DEFAULT_SLOT_SPECS}

    for bot_id, intake in MORE_EXOTIC_BOT_INTAKES.items():
        spec = specs[bot_id]
        row = roster_expansion_slots._slot_registry_row(spec)
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["paper_trading_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["direct_market_data_available"] is False
        assert row["direct_execution_allowed"] is False
        assert row["minimum_training_observations"] >= 2500
        assert row["sleeve_profile"] in EXOTIC_SLEEVES
        assert intake in row["data_intake_collections"]
        assert "cross_sleeve_correlation_matrix" in row["data_intake_collections"]
        assert "governance/exotic_derivatives" in row["storage_targets"]


def test_exotic_derivative_launchers_are_visible() -> None:
    for script_name in [
        "run_compound_options_shadow.py",
        "run_swaptions_shadow.py",
        "run_structured_products_shadow.py",
        "run_synthetic_cdo_shadow.py",
        "run_rainbow_options_shadow.py",
        "run_cdo_squared_shadow.py",
        "run_cdo_cubed_shadow.py",
        "run_variance_volatility_swaps_shadow.py",
        "run_barrier_lookback_options_shadow.py",
        "run_second_third_order_greeks_shadow.py",
        "run_high_frequency_market_making_shadow.py",
        "run_tail_risk_parity_shadow.py",
        "run_black_swan_hedging_shadow.py",
        "run_sovereign_debt_macro_shadow.py",
        "run_gamma_scalping_shadow.py",
        "run_statistical_arbitrage_shadow.py",
        "run_vanna_volga_hedging_shadow.py",
        "run_order_flow_market_microstructure_shadow.py",
        "run_dispersion_trading_shadow.py",
        "run_cross_asset_basis_training_shadow.py",
        "run_volatility_arbitrage_shadow.py",
    ]:
        assert (PROJECT_ROOT / "scripts" / script_name).exists()


def test_exotic_derivative_sleeves_are_in_strategy_coverage_config() -> None:
    config = json.loads((PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json").read_text(encoding="utf-8"))
    sleeves = {str(row.get("name")): row for row in config["sleeves"]}

    for sleeve in EXOTIC_SLEEVES:
        assert sleeve in config["ticker_universes"]
        assert len(config["ticker_universes"][sleeve]) >= 10
        assert sleeve in sleeves
        assert sleeves[sleeve]["runtime_status"] == "active_data_collection"
        assert len(sleeves[sleeve]["strategies"]) >= 5

    assert "rough_volatility_vvix_exotics" in sleeves["variance_volatility_swaps"]["strategies"]
    assert "corridor_variance_realized_range" in sleeves["variance_volatility_swaps"]["strategies"]
    assert "quantum_barrier_path_amplitude_options" in sleeves["barrier_lookback_options"]["strategies"]
    assert "asian_path_average_option_proxy" in sleeves["barrier_lookback_options"]["strategies"]
    assert "digital_binary_event_risk" in sleeves["barrier_lookback_options"]["strategies"]
    assert "overnight_gap_option_jump_risk" in sleeves["barrier_lookback_options"]["strategies"]
    assert "cross_asset_correlation_heat_swaps" in sleeves["dispersion_trading"]["strategies"]
    assert "cliquet_global_floor_local_cap" in sleeves["structured_products"]["strategies"]
    assert "worst_of_airbag_autocall" in sleeves["structured_products"]["strategies"]
    assert "recovery_lock_credit_note_guard" in sleeves["structured_products"]["strategies"]
    assert "capital_protection_participation_ratchet" in sleeves["structured_products"]["strategies"]
    assert "signature_trend_follower_options" in sleeves["rainbow_options"]["strategies"]
    assert "quanto_fx_equity_vol_beta" in sleeves["rainbow_options"]["strategies"]
    assert "forward_start_skew_reset" in sleeves["compound_options"]["strategies"]
    assert "chooser_event_optionality" in sleeves["compound_options"]["strategies"]
    assert "base_correlation_convexity" in sleeves["cdo_squared"]["strategies"]
    assert "nth_to_default_contagion_ladder" in sleeves["cdo_cubed"]["strategies"]
    assert "esg_linked_contingent_credit_default_swaps" in sleeves["synthetic_cdo"]["strategies"]
