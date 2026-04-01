import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "core"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from core import brain_refinery_v97_futures_event_order_book as v97
from core import brain_refinery_v98_crypto_execution_throttle_reentry as v98
from core import brain_refinery_v99_defensive_dividend_concentration as v99
from core import brain_refinery_v100_stock_crypto_overlap_context as v100


def _obs(symbol="SPY", last_price=100.0, **overrides):
    base_features = {
        "pct_from_close": 0.0024,
        "mom_5m": 0.0015,
        "vol_30m": 0.008,
        "range_pos": 0.69,
        "spread_bps": 8.0,
        "queue_depth": 3.0,
        "market_data_latency_ms": 35.0,
        "data_quality_market_data_latency_norm": 0.12,
        "lag_expected_fill_delta_bps": 0.08,
        "lag_latency_ms": 18.0,
        "lag_slippage_bps": 0.07,
        "market_micro_relative_volume_norm": 0.66,
        "data_quality_quote_agreement_norm": 0.96,
        "data_quality_quote_deviation_norm": 0.05,
        "behavior_prior": 0.26,
        "futures_specialist_vote": 0.72,
        "futures_order_book_imbalance_norm": 0.78,
        "futures_taker_imbalance_norm": 0.74,
        "futures_basis_bps_norm": 0.61,
        "futures_basis_divergence_norm": 0.38,
        "futures_term_structure_norm": 0.58,
        "futures_session_volume_profile_norm": 0.76,
        "futures_negative_bias_norm": 0.24,
        "futures_liquidation_risk_norm": 0.16,
        "calendar_event_proximity_norm": 0.74,
        "calendar_high_impact_24h_norm": 0.78,
        "calendar_macro_event_norm": 0.62,
        "calendar_fomc_event_norm": 0.36,
        "calendar_cpi_event_norm": 0.30,
        "calendar_labor_event_norm": 0.22,
        "infra_risk_throttle_norm": 0.26,
        "crypto_cross_provider_price_agreement_norm": 0.88,
        "crypto_deribit_mark_iv_norm": 0.64,
        "crypto_deribit_basis_norm": 0.61,
        "crypto_hyperliquid_open_interest_norm": 0.73,
        "crypto_hyperliquid_funding_norm": 0.58,
        "crypto_hyperliquid_basis_norm": 0.63,
        "crypto_coingecko_momentum_norm": 0.67,
        "crypto_defillama_dex_volume_growth_norm": 0.55,
        "crypto_defillama_stablecoin_growth_norm": 0.48,
        "crypto_etherscan_gas_norm": 0.52,
        "ctx_VIX_X_pct_from_close": 0.011,
        "bond_duration_regime_norm": 0.58,
        "bond_credit_risk_off_norm": 0.66,
        "dividend_yield_norm": 0.64,
        "dividend_payout_ratio_norm": 0.54,
        "dividend_quality_score_norm": 0.82,
        "dividend_safety_composite_norm": 0.84,
        "dividend_growth_momentum_norm": 0.70,
        "dividend_capture_entry_signal_norm": 0.58,
        "dividend_drip_active_norm": 0.66,
        "long_term_quality_dividend_norm": 0.76,
        "capital_flow_outflow_norm": 0.14,
        "flow_risk_on_norm": 0.22,
        "breadth_risk_off_norm": 0.64,
        "options_negative_bias_norm": 0.40,
        "market_crypto_risk_corr_norm": 0.60,
        "market_crypto_spy_corr_norm": 0.62,
        "market_crypto_qqq_corr_norm": 0.61,
        "market_crypto_tlt_corr_norm": 0.44,
        "market_crypto_uup_inverse_corr_norm": 0.38,
        "market_crypto_gold_corr_norm": 0.32,
        "market_crypto_corr_confidence_norm": 0.42,
        "fx_crypto_alignment_norm": 0.58,
        "fx_corr_confidence_norm": 0.44,
    }
    base_features.update(overrides)
    return {"symbol": symbol, "price": float(last_price), "features": base_features}


def test_runtime_feature_vector_shapes_batch3() -> None:
    sequence = [_obs()]
    assert v97._runtime_feature_vector(sequence, 0).shape == (30,)
    assert v98._runtime_feature_vector(sequence, 0).shape == (29,)
    assert v99._runtime_feature_vector(sequence, 0).shape == (30,)
    assert v100._runtime_feature_vector(sequence, 0).shape == (25,)


def test_v97_runtime_label_returns_positive_for_supported_futures_breakout() -> None:
    sequence = [
        _obs(symbol="SPY", last_price=100.0),
        _obs(symbol="SPY", last_price=100.14),
        _obs(symbol="SPY", last_price=100.30),
        _obs(symbol="SPY", last_price=100.48),
        _obs(symbol="SPY", last_price=100.66),
    ]
    assert v97._runtime_futures_event_label(sequence, 0, 4) == 1.0


def test_v98_runtime_label_returns_positive_for_crypto_reentry() -> None:
    sequence = [
        _obs(symbol="BTC-USD", last_price=100.0),
        _obs(symbol="BTC-USD", last_price=100.18),
        _obs(symbol="BTC-USD", last_price=100.36),
        _obs(symbol="BTC-USD", last_price=100.55),
    ]
    assert v98._runtime_crypto_reentry_label(sequence, 0, 3) == 1.0


def test_v99_runtime_label_returns_positive_for_defensive_dividend_repeat() -> None:
    sequence = [
        _obs(symbol="PG", last_price=100.0),
        _obs(symbol="PG", last_price=100.12),
        _obs(symbol="PG", last_price=100.26),
        _obs(symbol="PG", last_price=100.40),
        _obs(symbol="PG", last_price=100.56),
        _obs(symbol="PG", last_price=100.73),
        _obs(symbol="PG", last_price=100.90),
        _obs(symbol="PG", last_price=101.08),
        _obs(symbol="PG", last_price=101.25),
        _obs(symbol="PG", last_price=101.42),
        _obs(symbol="PG", last_price=101.60),
        _obs(symbol="PG", last_price=101.78),
        _obs(symbol="PG", last_price=101.96),
    ]
    assert v99._runtime_defensive_dividend_label(sequence, 0, 12) == 1.0


def test_v100_runtime_label_returns_positive_for_mild_overlap_context() -> None:
    sequence = [
        _obs(symbol="SPY", last_price=100.0),
        _obs(symbol="SPY", last_price=100.14),
        _obs(symbol="SPY", last_price=100.29),
        _obs(symbol="SPY", last_price=100.45),
        _obs(symbol="SPY", last_price=100.61),
    ]
    assert v100._runtime_overlap_label(sequence, 0, 4) == 1.0


def test_train_brain_uses_runtime_path_without_silent_fallback_batch3(monkeypatch) -> None:
    modules = [
        (v97, "brain_refinery_v97_futures_event_order_book", "SPY"),
        (v98, "brain_refinery_v98_crypto_execution_throttle_reentry", "BTC-USD"),
        (v99, "brain_refinery_v99_defensive_dividend_concentration", "PG"),
        (v100, "brain_refinery_v100_stock_crypto_overlap_context", "SPY"),
    ]
    for module, run_tag, expected_symbol in modules:
        captured = {}

        def _fake_train_runtime_indicator_bot(**kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(module, "train_runtime_indicator_bot", _fake_train_runtime_indicator_bot)

        result = module.train_brain()

        assert result == "ok"
        assert captured["run_tag"] == run_tag
        assert captured["allow_fallback_on_insufficient_data"] is False
        assert callable(captured["runtime_label_builder"])
        assert callable(captured["sample_filter"])
        assert captured["mode_allowlist"]
        assert expected_symbol in captured["symbol_allowlist"]
        assert captured["require_both_sides_precision"] is True
        assert captured["min_accuracy_lift_over_majority"] == 0.02
