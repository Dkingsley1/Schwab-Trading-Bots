from core.base_trader import BaseTrader
from core.exotic_derivatives_plumbing import (
    EXOTIC_DERIVATIVE_FEATURE_KEYS,
    EXOTIC_DERIVATIVE_SLEEVES,
    default_exotic_derivative_features,
    exotic_direct_execution_allowed,
    provider_capabilities_for_sleeve,
    summarize_exotic_derivative_proxy_features,
)
from scripts.ops import roster_expansion_slots


def test_exotic_provider_matrix_keeps_schwab_proxy_only() -> None:
    for sleeve in EXOTIC_DERIVATIVE_SLEEVES:
        caps = provider_capabilities_for_sleeve(sleeve)
        assert caps["direct_product_market_data"] is False
        assert caps["direct_execution_allowed"] is False
        assert exotic_direct_execution_allowed(sleeve, broker="schwab") is False
        assert caps["schwab_direct_inputs"]
        assert caps["public_inputs"]
        assert caps["proxy_mode"]


def test_exotic_proxy_features_use_public_and_listed_inputs() -> None:
    features = {
        "options_chain_available": 1.0,
        "options_iv_skew_norm": 0.72,
        "sofr_term_pressure_norm": 0.66,
        "credit_spread_stress_norm": 0.61,
        "sec_high_impact_7d_norm": 0.58,
        "market_crypto_sleeve_dispersion_norm": 0.53,
        "options_vanna_volga_hedge_pressure_norm": 0.67,
        "options_gamma_scalping_pressure_norm": 0.62,
        "options_variance_swap_proxy_norm": 0.71,
        "market_micro_order_flow_imbalance_norm": 0.59,
        "vvix_stress_norm": 0.64,
        "quantum_barrier_path_amplitude_norm": 0.57,
        "cross_asset_correlation_heat_norm": 0.62,
        "cliquet_global_floor_local_cap_norm": 0.55,
        "signature_trend_follower_options_norm": 0.60,
        "esg_contingent_cds_norm": 0.52,
    }
    out = summarize_exotic_derivative_proxy_features(
        sleeve="structured_products",
        broker="schwab",
        features=features,
        external_snapshots={
            "official_macro": {"timestamp_utc": "2026-05-01T13:00:00Z"},
            "sec_edgar": {"timestamp_utc": "2026-05-01T13:00:00Z"},
            "bond_reference": {"timestamp_utc": "2026-05-01T13:00:00Z"},
            "options_flow": {"timestamp_utc": "2026-05-01T13:00:00Z"},
        },
    )

    assert set(default_exotic_derivative_features()) == set(EXOTIC_DERIVATIVE_FEATURE_KEYS)
    assert out["exotic_sleeve_active_norm"] == 1.0
    assert out["exotic_proxy_mode_norm"] == 1.0
    assert out["exotic_direct_execution_blocked_norm"] == 1.0
    assert out["exotic_schwab_listed_input_norm"] == 1.0
    assert out["exotic_public_filing_input_norm"] == 1.0
    assert out["exotic_public_source_count_norm"] > 0.0
    assert out["exotic_proxy_higher_order_greeks_norm"] > 0.0
    assert out["exotic_proxy_gamma_scalping_norm"] > 0.0
    assert out["exotic_proxy_vol_arbitrage_norm"] > 0.0
    assert out["exotic_proxy_rough_vvix_norm"] > 0.0
    assert out["exotic_proxy_quantum_barrier_path_norm"] > 0.0
    assert out["exotic_proxy_correlation_heat_swap_norm"] > 0.0
    assert out["exotic_proxy_cliquet_floor_cap_norm"] > 0.0
    assert out["exotic_proxy_signature_trend_follower_norm"] > 0.0
    assert out["exotic_proxy_esg_ccds_norm"] > 0.0
    assert out["exotic_data_confidence_norm"] > 0.0


def test_exotic_roster_rows_include_capability_metadata() -> None:
    row = roster_expansion_slots._slot_registry_row(
        {
            "bot_id": "brain_refinery_v999_test_structured_products_proxy_bot",
            "bot_role": "options_sub_bot",
            "slot_kind": "structured_products_test",
            "sleeve_profile": "structured_products",
            "sleeve_family": "exotic_derivatives",
        }
    )

    assert row["direct_market_data_available"] is False
    assert row["direct_execution_allowed"] is False
    assert row["provider_capability_profile"] == "payoff_barrier_credit_proxy"
    assert "sec_edgar_filings" in row["proxy_data_sources"]
    assert "listed_option_chains" in row["schwab_direct_inputs"]


def test_base_trader_blocks_exotic_execution_even_when_execution_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SHADOW_PROFILE", "compound_options")
    monkeypatch.setenv("SHADOW_DOMAIN", "exotic_derivatives")
    monkeypatch.setenv("EXOTIC_DERIVATIVE_RESEARCH_ONLY", "1")
    monkeypatch.setenv("EXOTIC_DIRECT_EXECUTION_ALLOWED", "0")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("PAPER_BROKER_BRIDGE_ENABLED", "0")

    trader = BaseTrader("", "", "", mode="paper", broker="schwab")
    result = trader.execute_decision(
        symbol="SPY",
        action="BUY",
        quantity=1,
        model_score=0.9,
        threshold=0.5,
        features={"last_price": 500.0},
        gates={"market_data_ok": True},
        reasons=["test"],
        strategy="test_exotic_guard",
        metadata={"source_profile": "compound_options", "shadow_domain": "exotic_derivatives"},
    )

    assert result["status"] == "EXOTIC_DERIVATIVE_EXECUTION_BLOCKED"
    assert result["safety"]["exotic_derivative_execution_blocked"] is True
