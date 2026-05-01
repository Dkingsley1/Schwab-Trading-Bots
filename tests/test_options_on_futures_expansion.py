import scripts.run_all_sleeves as run_all_sleeves
import scripts.run_specialized_sleeve_shadow as specialized
from scripts.ops import roster_expansion_slots


def test_options_on_futures_sleeves_are_registered_for_collection() -> None:
    assert "options_on_futures" in specialized.SLEEVE_DEFAULTS
    assert "options_on_futures_aggressive" in specialized.SLEEVE_DEFAULTS
    assert "options_on_futures" in run_all_sleeves.SPECIALIZED_SLEEVE_PROFILES
    assert "options_on_futures_aggressive" in run_all_sleeves.SPECIALIZED_SLEEVE_PROFILES


def test_conservative_and_aggressive_options_bots_are_collection_first() -> None:
    wanted = {
        "brain_refinery_v318_conservative_protective_put_hedge_bot",
        "brain_refinery_v319_conservative_put_spread_hedge_bot",
        "brain_refinery_v320_conservative_collar_protection_bot",
        "brain_refinery_v321_conservative_covered_call_income_guard",
        "brain_refinery_v322_conservative_cash_secured_put_entry_bot",
        "brain_refinery_v323_conservative_defined_risk_credit_spread_bot",
        "brain_refinery_v324_conservative_debit_spread_directional_bot",
        "brain_refinery_v325_conservative_iron_condor_range_guard",
        "brain_refinery_v326_conservative_calendar_diagonal_income_guard",
        "brain_refinery_v327_options_on_futures_defined_risk_hedge_bot",
        "brain_refinery_v328_aggressive_options_on_futures_open_drive_debit_spread_bot",
        "brain_refinery_v329_aggressive_options_on_futures_macro_event_gamma_bot",
        "brain_refinery_v330_aggressive_options_on_futures_vol_expansion_breakout_bot",
        "brain_refinery_v331_aggressive_options_on_futures_curve_oil_event_spread_bot",
        "brain_refinery_v332_aggressive_options_on_futures_gamma_momentum_scalper",
    }
    specs = {
        str(row.get("bot_id")): row
        for row in roster_expansion_slots.DEFAULT_SLOT_SPECS
        if str(row.get("bot_id")) in wanted
    }

    assert set(specs) == wanted
    for spec in specs.values():
        row = roster_expansion_slots._slot_registry_row(spec)
        assert row["active"] is True
        assert row["lifecycle_state"] == "data_collection_only"
        assert row["training_excluded"] is True
        assert row["allocation_enabled"] is False
        assert row["live_trading_enabled"] is False
        assert row["minimum_training_observations"] >= 1500
        assert "options_chain_surface" in row["data_intake_collections"]

    fop_row = roster_expansion_slots._slot_registry_row(specs["brain_refinery_v327_options_on_futures_defined_risk_hedge_bot"])
    assert "futures_options_chain_surface" in fop_row["data_intake_collections"]
    assert "governance/futures_options" in fop_row["storage_targets"]
